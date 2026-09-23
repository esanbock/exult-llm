"""driver.py - Let an Ollama-hosted LLM play Ultima VII through Exult.

Runs an observe -> think -> act loop:
  1. observe(): pull the current game state (JSON) from the Exult bridge.
  2. think(): send the state to Ollama and ask for reasoning + the next action.
  3. act(): forward the chosen action to Exult.

Optionally opens a live "LLM thinking" window (--show-thoughts) that displays
the observation, dialog/characters/objects, the model's reasoning, and the
chosen action each turn.

Prerequisites:
  * Build Exult with USE_LLM_AGENT and run it with:  Exult.exe --bg --nomenu --llmagent
    (--nomenu skips the intro/menu and drops straight into the game world.)
  * Have Ollama running with a model pulled, e.g.:   ollama run llama3.1
  * Game data installed (see tools/llm_agent/README.md).

Usage:
  python driver.py --model llama3.1 --steps 50 --show-thoughts
  python driver.py --dry-run --show-thoughts        # no Ollama; scripted moves
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

from exult_client import ExultClient
from ollama_client import OllamaClient
from thoughts_window import ThoughtsWindow
from knowledge import KnowledgeBase

SYSTEM_PROMPT = """\
You are an autonomous agent playing Ultima VII: The Black Gate as the Avatar.
You interact ONLY through the JSON tool interface described below. You cannot
click, use a mouse, or do anything not listed as a TOOL. Think of this as an
API: the only way to affect the game is to call one TOOL per turn.

# MISSION (your purpose - let this drive every decision)
You are the Avatar, the hero of Ultima VII: The Black Gate - an open-world
role-playing adventure full of towns, people, mysteries, quests, dungeons, and
a larger unfolding plot. There is no single scripted objective from your side:
you must discover goals by playing. Your enduring purpose is to explore the
world, understand what is happening, help people, follow leads, and advance the
main story as it reveals itself.

General principles (apply to ANY situation, not one specific puzzle):
  * INVESTIGATE by talking: NPCs are your main source of information and quests.
    Ask them their name, job, and about any topic they or others mention. New
    dialog topics often appear as answer choices - explore the useful ones.
  * GREET NEW PEOPLE: whenever you encounter someone you have NOT talked to yet
    (status "new" in nearby), talk to them before moving on - even while you are
    pursuing another goal. Every new person may give you a quest, a clue, an
    item, or offer to JOIN YOUR PARTY (companions are extremely valuable). Do
    not walk past unmet people to chase a single objective; a hint to find a
    specific person is NOT a reason to ignore everyone else you pass.
  * FOLLOW LEADS: when someone mentions a person, place, item, or event, treat
    it as a lead worth pursuing. Use your journal to remember what you learned.
  * EXAMINE THE WORLD: investigate notable objects, bodies, and containers you
    come across. OPEN containers (chests, barrels, bags, crates) and search
    bodies to see what is inside. Try interesting-looking devices - levers,
    switches, buttons, mechanisms - as they often reveal secrets or open the way.
  * GATHER USEFUL THINGS: pick up items that could help later - gold and gems
    (money), food (you must eat), weapons and armour, keys, potions, scrolls,
    tools, and anything quest-related or unusual. When unsure, taking a small
    item is usually worth it; your pack holds a lot.
  * DO NOT STEAL: only take items that are unowned, abandoned, or given to you.
    Items inside shops, homes, and other people's property are OWNED - taking
    them is theft that angers people and can make guards attack you. If a
    shopkeeper or owner is present, ask/buy rather than grab. Loot from enemies
    you defeat and from the dead/abandoned is fair game.
  * MAKE PROGRESS: prefer purposeful action over aimless wandering or repeating
    yourself. If you have exhausted a person or place, move on to somewhere new.
  * DON'T LINGER: do not re-interview people you've already learned from
    (especially your own party companions - they have nothing new). When you
    have gathered the local leads, LEAVE the area: travel to the town gate/edge
    and out to new regions to advance the story. The world is far bigger than
    one town - staying put stalls the whole adventure.
  * SURVIVE: keep fed and stay alive; avoid needless danger.
You are not told the solution to anything - reason from what you observe and are
told, as a curious, capable adventurer would.

# PROTOCOL
Each turn you receive a STATE object (schema below) and must reply with EXACTLY
one JSON object, nothing else. Put "action" FIRST so it is never lost, with the
tool's parameters at the TOP LEVEL of "action" (do NOT nest them under "params").
Keep "reason" LAST and to ONE short clause (~12 words) - a long reason can get
cut off and waste the whole turn:
  {"action": {"type": "move", "dir": "n"}, "reason": "head north"}
  {"action": {"type": "talk", "name": "Iolo"}, "reason": "greet him"}
  {"action": {"type": "answer", "index": 0}, "reason": "ask his name"}

You MAY also add an optional "new_quest" field in the SAME reply to record a
goal without spending your action - it does not use up your turn:
  {"action": {"type": "answer", "index": 3},
   "new_quest": {"title": "Ask Mayor Finnigan about the murder",
                 "priority": 2, "notes": "Iolo suggested talking to Finnigan",
                 "depends_on": ["investigate_the_trinsic_murder"]},
   "reason": "Iolo says Finnigan may know more"}
"new_quest" can be one quest object or a list. Break a big goal into smaller
sub-quests (use depends_on with the parent quest's id, which is its title
lower-cased with underscores). To mark a goal finished, add "resolve_quest":
"<quest id or title>". Capture a quest whenever your reasoning names something
you intend to do - keep your quest log current and prioritized.

You may ALSO add an optional "topic" field in the same reply to build your own
understanding of recurring subjects (people, groups, places, mysteries) as you
reason - this is YOUR notebook of insights, and it does not use your turn:
  {"action": {"type": "answer", "index": 2},
   "topic": {"name": "The Fellowship",
             "note": "A popular group; several murder victims had joined it - suspicious"},
   "reason": "ask about the Fellowship"}
"topic" can be one object or a list, each {name, note}. Add or update a topic
whenever you form a theory or learn something meaningful about a subject; add a
NEW note to the same topic name as your understanding evolves, so your thinking
accumulates over time. Use "recall" with a topic name to review all your notes.

# STATE SCHEMA (what you receive each turn)
  alert (string, optional)    - urgent guidance for THIS turn. If it contains a
                                "HINT from your operator", follow that hint as
                                your top priority this turn. Also warns you if
                                your last move was blocked.
  world_loaded (bool)         - is a game world loaded
  gump_open (bool)            - a container/body window is OPEN and blocks your
                                movement. The items INSIDE the open window are
                                listed in "gump_contents" (NOT objects[].contents,
                                which are other nearby containers). "take" the
                                ones you want by name, then "close" it. If
                                gump_contents is empty, just "close".
  gump_contents (list)        - item names inside the currently OPEN container/
                                body window (empty if it holds nothing).
  player: {tx,ty (your tile), hp, dead, food}
  time_of_day (string)        - morning/afternoon/evening/night, plus hour (0-23)
                                and is_night. At night most townsfolk are asleep
                                (see condition:"sleeping"); use "wait_until" to
                                pass time to morning if you need someone awake.
  in_combat (bool)            - are you in combat mode
  conversation_in_progress (bool) - true while a conversation is open (faces shown)
  conversation_active (bool)  - true when NPC answer choices are on screen NOW
  number_prompt (bool)        - true when the game is asking you to pick a NUMBER
                                on a slider (e.g. "how many?"). When true you also
                                get number_min, number_max, number_current. Use
                                the "set_number" tool to answer it.
  npc_text (string|null)      - the last thing an NPC said to you
  ambient_speech (list)       - things characters/creatures say OUT LOUD near you
                                without a formal conversation ({who, said}), e.g.
                                a cat's "Meeow" or a townsperson's remark. Notice
                                these - they can be hints or reactions.
  answers (list[string])      - the reply choices you may pick (only when
                                conversation_active is true)
  already_asked_this_npc (list) - topics you have ALREADY asked this character in
                                the past (don't waste turns re-asking these).
  not_yet_asked_this_npc (list) - topics this character can discuss that you have
                                NOT asked yet. Prefer these - ask the important
                                unasked topics (e.g. "key", "password") before
                                leaving the conversation.
  nearby (list)               - NPCs you can TALK to: {name, dx, dy, status,
                                condition?}. "condition" (if present) is a
                                physical state: "sleeping" (CANNOT be talked to -
                                wait for day or leave them), "paralyzed",
                                "poisoned", "charmed", "cursed", or "hostile".
                                (your own party is listed separately in "party")
                                dx>0 = east, dx<0 = west, dy>0 = south, dy<0 = north
                                status = new (never talked) | talked (spoken to)
                                | exhausted (talked several times without new
                                progress) - this is ADVISORY, not a ban. Before
                                re-talking someone, READ their entry in npc_notes
                                (their recorded lines + talk count) and decide
                                yourself: if you have a NEW reason (a new topic,
                                lead, item, or quest progress) it is worth talking
                                again; if their notes show you already covered
                                everything, move on. Trust your own judgement.
  party (list)                - names of your companions travelling WITH you.
                                They follow you and have no new information - do
                                NOT "talk" to them to investigate; just travel
                                and act, and they come along.
  objects (list)              - items on the ground: {name, dx, dy}. May include
                                "owned": true - that item is someone's property;
                                taking it is STEALING (avoid it). Items without
                                "owned" are free to take. "body":true means a
                                searchable corpse.
  grid (string)               - top-down ASCII map centered on you (@):
                                  @ you   C companion   & other NPC   x body
                                  T tree   W wall/building   = fence/gate
                                  n container   H furniture   s sign   ~ water
                                  + closed door (PASSAGE-openable)  / open door
                                  o obstacle   * item   . open ground   # blocked
                                north=up, south=down, east=right, west=left.
                                Walk only on '.', items '*', or open door '/';
                                everything else (T W = n H ~ o #) blocks you.
                                COORDINATES: you are at player.tx/ty (grid center).
                                A grid cell at row r, col c is tile
                                (grid_origin_tx + c, grid_origin_ty + r). "look",
                                nearby, objects, and doors also give coordinates,
                                and you can "goto" any tx,ty.
  doors (list)                - nearby doors: {name, dx, dy, closed}

# YOUR JOURNAL (you maintain this - it persists across turns)
  quests: {open[] (priority-sorted), recently_resolved[], resolved, unresolved}
      - Your quest log is YOUR plan: you choose which quest to work on and its
        priority. Add new ones as you discover goals; RESOLVE quests the moment
        you complete them so the log stays accurate.
      - open: your unresolved quests, PRIORITY-SORTED (priority 1=highest). Pick
        whichever you judge best to work on now.
      - A quest may list "depends_on" (prerequisite quest ids) and
        "prereqs_unmet" (those not yet done). This is INFORMATION for you to
        reason about ordering - it does NOT stop you acting; you decide whether
        a prerequisite really must come first.
      - recently_resolved: quests you already finished - do NOT redo these.
      - Each quest: {id, title, priority, status, npc (who it involves),
        depends_on[], prereqs_unmet[]}
      - A quest with "npc" set can be pursued by going to/talking to that NPC.
  npc_notes: what you have recorded about nearby/known NPCs (their leads, wants),
      including "last_seen":{tx,ty} - the tile where you most recently saw each
      person. To return to someone (e.g. a companion to recruit after progress),
      "goto" their name or their last_seen coordinates.
  known_places: your MENTAL MAP of discovered locations (landmarks, buildings,
      gates, shops, etc.), nearest first, each {name, kind, dx, dy}. You can
      "goto" any of these by name to travel back to them - useful for returning
      to a town, building, or quest location you found earlier.
  known_topics: YOUR OWN notebook of subjects you have been thinking about
      (people, groups, places, mysteries), each {topic, notes (how many notes
      you've written)}. These are authored by you via the "topic" field. Use
      "recall" on a topic to re-read all your notes on it, and keep adding notes
      as your understanding grows.
  recent_dialogue: a running transcript of the last conversation exchanges
      ({"npc":name,"said":...} for NPC lines, {"me":...} for your replies).
      Use this to remember what you have learned and what was said earlier.
  recent_actions: your last few meaningful actions (talk/open/pickup/etc).
      Use this to avoid repeating something you just did.
  story_so_far: a compact running summary of OLDER events/clues that have
      scrolled out of recent_dialogue. Older detail is compressed here (not
      lost) so you can still recall earlier story and leads on a long journey.
  operator_hints: guidance your human operator has given you over time (newest
      last). Treat these as important standing instructions, not just for one
      turn - honor earlier hints even if they are no longer repeated.
  observed: notable things you have SEEN or OVERHEARD (deduped), each like
      "seen: chest at (x,y)" or 'heard: cat: "Meeow"'. Your durable record of
      what you encountered; use it to recall and return to things of interest.

# TOOLS (the complete list of things you can do - nothing else is possible)
  move    - Walk one step. params: {"dir": one of n,s,e,w,ne,nw,se,sw}
            Use the grid: step onto '.' tiles, never into '#'. To reach an
            NPC/object, move toward its (dx,dy).
  goto    - PATHFIND to a destination and walk there automatically, routing
            around walls and THROUGH doorways. params: {"name": "<NPC/object/
            place>"} to go to the nearest thing with that name OR a remembered
            place/NPC from known_places/npc memory (even if off-screen), OR
            {"tx":<int>,"ty":<int>} for an absolute tile. PREFER "goto" over many
            "move" steps to reach an NPC, item, building, or known place.
  stop    - Stop walking. params: none.
  talk    - START a conversation with a nearby NPC. params: {"name": "<NPC name>"}
            This is the ONLY way to begin dialog. Walking next to an NPC does
            NOT start dialog. You do NOT need to be adjacent - it finds the
            named NPC in your view and opens the conversation. Works for both
            townspeople and party companions (in_party:true).
  open    - Open (or close) the nearest door within a few tiles. params: none.
            Doors show as '+' (closed) or '/' (open) on the grid and in "doors".
            A closed door '+' is NOT a wall - it is a passage you can use. To go
            through: either use "goto" a tile/room beyond it (goto opens doors on
            the way automatically), OR move adjacent to the '+' door, "open" it,
            then "move" through the '/' opening. Never treat '+' as impassable.
  search  - Open the nearest body or container to see what is inside. params:
            none. Its "contents" also appear in the "objects" list. After
            searching, use "take" to grab items, then "close" it.
  close   - Close an open container/body gump (like pressing the checkmark).
            params: none. Do this when done looting so you can move again.
  take    - Take an item OUT of a nearby body/container (searches inside bags
            too) into your pack. params: {"name":"<item>"} for a specific item,
            or omit to take the first. Use this to loot bodies/chests.
  pickup  - Take an item off the GROUND (a loose world tile) into your pack.
            params: {"name":"<item name>"} (optional). For items inside a
            body/container use "take" instead.
  inventory - Report what you are WEARING (per slot) and CARRYING. params: none.
  annotate - Mark the current location (or a given tile) on your map with a
            label so you can return later. params: {"label":"<name>"} (uses your
            current position) or add {"tx","ty"} for a specific tile, and an
            optional {"note"}. The label then appears in known_places and you can
            "goto" it by name. Mark important spots (e.g. a crime scene, a shop,
            a quest location) so you never lose them.
  equip   - Wear/wield an item you have (or one in a nearby container): it goes
            into its correct slot (weapon, head, torso, legs, feet, shield,
            belt, amulet, cloak, gloves, ring). params: {"name":"<item>"}.
  look    - Get a DETAILED description of your surroundings (setting, every
            nearby person with what you know about them, items on the ground,
            doors/exits, terrain features). params: none. Use it when you enter
            a new area or want to understand a scene before acting.
  recall  - Retrieve your FULL saved knowledge about a character OR a TOPIC.
            params: {"name":"<character or topic>"} (fuzzy). For a person you get
            their transcript, topics you asked, and topics NOT yet asked. For a
            world topic (e.g. "Fellowship", "Batlin", "gargoyles") you get every
            line any NPC told you about it and who mentioned it. Result appears
            next turn as "recalled". Use it to remember instructions (e.g. the
            Mayor telling you to find someone) and to understand recurring themes
            before deciding what to do.
  answer  - Choose a reply during a conversation. params: {"index": <int>} (0-based
            into the "answers" list) OR {"text": "<answer text>"}.
            Only valid when conversation_active is true.
  set_number - Answer a numeric slider prompt. params: {"value": <int>}. Only
            valid when number_prompt is true; value is clamped to
            [number_min, number_max]. Use this to pick a quantity/amount.
  key     - Press a key. params: {"key": "space"|"escape"|"a".."z"|"0".."9"}.
            Use "space" to advance NPC text when there is npc_text but no answers.
  combat  - Toggle combat/attack mode on or off. params: none.
  feed    - Eat food to refill your food level (prevents starving). params: none.
  save    - Save the game so progress is not lost. params: none. (The driver
            also auto-saves periodically; you rarely need this.)
  wait    - Do nothing this turn. params: none.
  wait_until - Pass time until a target hour (0-23), e.g. wait for morning so
            sleeping NPCs wake. params: {"hour": 7} (default 7 = morning). Use
            this when the people you need are "sleeping" and it is night.

# JOURNAL TOOLS (manage your own quest log & notes - do NOT affect the game)
  add_quest    - Record a goal you discovered. params: {"title": "...",
                 "priority": 1-9 (1=highest), "notes": "...",
                 "depends_on": ["<quest id>", ...] (optional prerequisites)}.
                 Use whenever you form an intention or infer a goal from what
                 you observe or are told - e.g. "investigate the docks", "find
                 the man who fled", "ask the Mayor about the murder". If your
                 REASONING this turn identifies something you want to do next,
                 capture it as a quest so you remember and can prioritise it. If
                 quest B requires finishing quest A first, set B.depends_on=["<A id>"].
  update_quest - Change a quest. params: {"id": "<quest id>", "status":
                 "active|blocked|done", "priority": n, "notes": "...",
                 "depends_on": [...]}. Mark a quest "done" when you complete it,
                 and add notes as you learn more about it.
  note_npc     - Save a note about an NPC. params: {"name": "...", "note": "..."}.
                 Record leads, what they want, or what they told you.
  (These journal tools do not advance the game. Prefer capturing a quest the
   moment you decide on a goal - a well-kept quest log is how you stay strategic
   across many turns. After using one, take a game action the same or next turn.)

# HOW TO DECIDE (policy)
  1. If conversation_active is true -> use "answer" (pick the index of the reply
     you want). PRIORITISE topics in "not_yet_asked_this_npc" - especially
     important ones like a key, password, name, or a person/place mentioned -
     and avoid re-picking anything in "already_asked_this_npc". Once you have
     asked the useful unasked topics (or you see the same choices again), END
     the conversation by choosing the "bye"/"leave" reply. Do NOT loop.
  2. Else if conversation_in_progress is true (a conversation is open but no
     choices yet) -> use "key" with "space" to advance the NPC's text until the
     answer choices appear. Do NOT "talk" again or "move" during a conversation.
  3. YOUR QUEST LOG IS YOUR PLAN - own it. You decide which quests matter and
     their priority (1=highest). Each turn, CONSULT "quests": pick whichever
     open quest you judge most important right now and act on
     it (talk to its npc if nearby, "goto" them if not, else act on its notes).
     Consider each quest's prereqs_unmet when ordering, but you decide.
     You are free to reprioritise as you learn more (update_quest priority).
     - ADD a quest (or inline "new_quest") whenever you decide on a goal.
     - RESOLVE quests you complete: the moment a goal is achieved (you got the
       password, spoke to the person, found the item, solved the puzzle), add
       "resolve_quest":"<quest id or title>" (or update_quest status:"done").
       An unresolved log you never close becomes useless - keep it accurate so
       "focus" always shows what truly remains.
     - REFER BACK: before acting, check whether your intended action matches an
       open quest; if a quest is already done, resolve it instead of repeating.
     - Your "topic" notebook is your long-term memory: record insights there so
       that even after a quest is closed, what you learned about people, groups,
       and mysteries persists.
  3a. Talking to NEW people is how you discover quests. Prefer nearby NPCs with
     status "new", then "talked". For an "exhausted" NPC, check their npc_notes
     first: re-talk them ONLY if you now have a new reason (new topic/lead/item
     or quest progress). If their notes show you already learned what they know,
     move on rather than repeating the same conversation.
  4. Only if you have NO actionable quest and no new NPC to meet -> explore to a
     NEW area to find fresh people/places. To travel anywhere more than a step
     or two (an NPC, item, building, or new part of town) ALWAYS use "goto" - it
     pathfinds around walls and through doors. If you have wandered far from
     where your quests are (e.g. out in the wilderness with no one around),
     "goto" a relevant known_place - especially "start area"/your town - to get
     back to where the story and NPCs are. Use single "move" steps only for
     tiny local adjustments, and NEVER move onto a '#' wall: on the
     grid you (@) can only step onto '.', items '*', or an open door '/'. If you
     keep bumping the same spot, you are against a wall - use "goto" to route
     around it.
  4b. PAY ATTENTION TO OBJECTS: the "objects" list names what is on the ground
     around you; bodies also show as 'x' on the grid. When something looks
     relevant to your goals or curiosity, interact with it rather than pacing:
     "goto"/move adjacent, "search" bodies and containers to see their contents,
     and "pickup" useful items. Read object names to understand your surroundings.
  5. If your food is low, use "feed". If threatened, "combat".

OUTPUT RULES (critical - follow exactly):
- Output ONLY one JSON object. Start your reply with '{' as the very first
  character. No preamble, no thinking, no explanation, no markdown, no code
  fences before or after.
- Keep it short: the object is just {"action":{...},"reason":"..."} (plus an
  optional "new_quest"). Do NOT write anything outside the JSON.
"""


def summarize_state(state: dict, kb: "KnowledgeBase | None" = None, last_look: str = "", alert: str = "", squeeze: int = 0) -> str:
    """Compact the observation to keep the prompt small and focused.

    `squeeze` is a context-pressure level (0 = plenty of room .. 3 = very
    tight). Higher levels shrink the raw rolling windows (dialogue/actions/
    objects) so the prompt stays within num_ctx as the playthrough grows. The
    durable structured memory (quests, places, hints, episodic summary) is
    always kept - only the verbose recent-context tiers are trimmed."""
    # Window sizes per squeeze level (dialogue, actions, objects, nearby, places).
    _TIERS = [
        (30, 10, 14, 8, 12),   # 0: roomy
        (18, 8, 10, 8, 10),    # 1: trim
        (10, 6, 8, 6, 8),      # 2: tight
        (6, 4, 6, 5, 6),       # 3: very tight
    ]
    lvl = max(0, min(int(squeeze), len(_TIERS) - 1))
    dlg_n, act_n, obj_n, near_n, place_n = _TIERS[lvl]
    p = state.get("player") or {}
    nearby = state.get("nearby") or []
    objects = state.get("objects") or []
    # Only surface dialog fields when a conversation is actually open, so the
    # model isn't misled by stale npc_text into pressing space forever.
    in_convo = bool(state.get("conversation_in_progress"))
    view = {
        "player": {
            "tx": p.get("tx"), "ty": p.get("ty"),
            "hp": p.get("hp"), "dead": p.get("dead"),
        },
        "time_of_day": state.get("time_of_day"),
        "hour": state.get("hour"),
        "is_night": state.get("is_night"),
        "in_combat": state.get("in_combat"),
        "conversation_in_progress": in_convo,
        "conversation_active": state.get("conversation_active"),
        "npc_text": state.get("npc_text") if in_convo else None,
        "answers": state.get("answers") if in_convo else [],
        "ambient_speech": state.get("ambient_speech") or [],
        "nearby": [
            {"name": n.get("name"), "dx": n.get("dx"), "dy": n.get("dy"),
             "status": (kb.talk_status(n.get("name")) if kb and n.get("name") else "new"),
             **({"condition": n["condition"]} if n.get("condition") else {})}
            for n in nearby[:near_n] if not n.get("in_party")
        ],
        # Party companions are shown separately - they follow you and have no
        # new information, so do NOT "talk" to them to investigate.
        "party": [n.get("name") for n in nearby if n.get("in_party") and n.get("name")],
        "objects": [
            {"name": o.get("name"), "dx": o.get("dx"), "dy": o.get("dy"),
             **({"body": True} if o.get("body") else {})}
            for o in objects[:obj_n]
        ],
        "grid_legend": state.get("grid_legend"),
        "grid": state.get("grid"),
    }
    doors = state.get("doors") or []
    if doors:
        view["doors"] = [
            {"name": d.get("name"), "dx": d.get("dx"), "dy": d.get("dy"),
             "closed": d.get("closed")}
            for d in doors[:6]
        ]
    if kb is not None:
        view["quests"] = kb.quest_view()
        # Names of NPCs already met (helps avoid re-greeting). Cap it under
        # context pressure - a long playthrough meets many NPCs and this list
        # can bloat the prompt.
        _met = sorted(kb.npcs.keys())
        view["already_talked_to"] = _met if lvl < 2 else _met[:12]
        # Durable memory that is always kept regardless of context pressure:
        # a rolling episodic summary of older events (clues/story compressed),
        # and the operator's hint history so earlier steering isn't forgotten.
        if kb.episodic_summary:
            view["story_so_far"] = kb.episodic_summary
        rh = kb.recent_hints(5)
        if rh:
            view["operator_hints"] = rh
        # Notable things seen / overheard, deduped (persistent observation mem).
        obs = kb.observations_view(8 if lvl >= 2 else 12)
        if obs:
            view["observed"] = obs
        # Known places (mental map), nearest first, with direction from here.
        places = kb.places_view(p.get("tx", 0), p.get("ty", 0), limit=place_n)
        if places:
            view["known_places"] = places
        # Known TOPICS (cross-cutting world subjects like the Fellowship). A
        # compact menu so the agent can pursue themes and "recall" them for
        # detail. Fewer under context pressure.
        topics = kb.topics_view(8 if lvl >= 2 else 16)
        if topics:
            view["known_topics"] = topics
        # If we're in a conversation, proactively show what we've already asked
        # THIS person and what we have NOT asked yet, so the agent doesn't
        # re-ask covered topics or forget an important one (e.g. "key").
        _cnpc = state.get("_convo_npc")
        if in_convo and _cnpc:
            _rec = kb.recall_npc(_cnpc)
            if _rec.get("known"):
                if _rec.get("topics_asked"):
                    view["already_asked_this_npc"] = _rec["topics_asked"]
                if _rec.get("topics_unasked"):
                    view["not_yet_asked_this_npc"] = _rec["topics_unasked"]
        # NPC notes: focus on those currently nearby. Fewer notes each under
        # context pressure to keep the prompt within budget.
        nearby_names = [n.get("name") for n in nearby[:near_n] if n.get("name")]
        notes = kb.npc_view(nearby_names, note_limit=(4 if lvl >= 2 else 12))
        if notes:
            view["npc_notes"] = notes
        # Growing window of recent CONVERSATION (story/clues live here) and a
        # short window of recent ACTIONS (to avoid repeating yourself). These
        # shrink under context pressure; what scrolls off is folded into
        # story_so_far by the driver's budget loop, so clues are not lost.
        dh = kb.dialogue_view(dlg_n)
        if dh:
            view["recent_dialogue"] = dh
        ah = kb.action_view(act_n)
        if ah:
            view["recent_actions"] = ah
    if last_look:
        view["look_description"] = last_look
    if alert:
        view["alert"] = alert
    if state.get("recalled") is not None:
        # The full saved dialogue tree the agent asked to recall this turn.
        view["recalled"] = state["recalled"]
    return json.dumps(view)

def format_dialog(state: dict) -> str:
    """Human-readable dialog/characters/objects panel for the thoughts window."""
    lines = []
    amb = state.get("ambient_speech") or []
    if amb:
        lines.append("Overheard:")
        for a in amb[:6]:
            who = a.get("who") or "someone"
            lines.append(f'  {who}: "{a.get("said")}"')
        lines.append("")
    if state.get("conversation_active"):
        lines.append("== CONVERSATION ACTIVE ==")
    npc_text = state.get("npc_text")
    if npc_text:
        lines.append(f"NPC says: {npc_text}")
    answers = state.get("answers") or []
    if answers:
        lines.append("Answers:")
        for i, a in enumerate(answers):
            lines.append(f"  [{i}] {a}")
    nearby = state.get("nearby") or []
    if nearby:
        lines.append("")
        lines.append("Characters nearby:")
        for n in nearby[:12]:
            lines.append(f"  {n.get('name')}  (dx={n.get('dx')}, dy={n.get('dy')})")
    objects = state.get("objects") or []
    if objects:
        lines.append("")
        lines.append("Objects nearby:")
        for o in objects[:12]:
            lines.append(f"  {o.get('name')}  (dx={o.get('dx')}, dy={o.get('dy')})")
    return "\n".join(lines) if lines else "(nothing notable on screen)"


def parse_reply(text: str) -> tuple[str, dict]:
    """Extract (reason, action) from the model's reply."""
    text = text.strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                obj = None
        # Repair a TRUNCATED reply (model hit the response token limit mid-JSON,
        # e.g. an over-long "reason" that never closes). Try to salvage the
        # action by extracting a complete "action":{...} object even if the
        # outer object is unterminated.
        if obj is None and start != -1:
            m = re.search(r'"action"\s*:\s*(\{[^{}]*\})', text)
            if m:
                try:
                    act = json.loads(m.group(1))
                    rm = re.search(r'"reason"\s*:\s*"([^"]*)', text)
                    return (rm.group(1) if rm else "(recovered)", act)
                except json.JSONDecodeError:
                    pass
    if not isinstance(obj, dict):
        return ("(could not parse reply)", {"type": "wait"})
    reason = str(obj.get("reason", ""))
    action = obj.get("action")
    if not isinstance(action, dict):
        # Maybe the model returned a bare action.
        action = obj if "type" in obj else {"type": "wait"}
    # Some models nest parameters under "params"/"parameters"/"arguments".
    # Flatten them up to the action level so {"type":"move","params":{"dir":"n"}}
    # behaves the same as {"type":"move","dir":"n"}.
    for key in ("params", "parameters", "arguments", "args"):
        nested = action.get(key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                action.setdefault(k, v)
            action.pop(key, None)
    return (reason, action)


def _where(px: int, py: int, dx: int, dy: int) -> str:
    """Describe a position both as a compass direction+distance AND its absolute
    tile coordinate, so it matches the map grid and can be used with goto."""
    ns = "north" if dy < 0 else ("south" if dy > 0 else "")
    ew = "east" if dx > 0 else ("west" if dx < 0 else "")
    d = (ns + ew) or "here"
    dist = abs(dx) + abs(dy)
    near = "adjacent" if dist <= 1 else ("close" if dist <= 4 else ("nearby" if dist <= 10 else "far"))
    tx, ty = px + dx, py + dy
    where = "right here" if d == "here" else f"{near} to the {d}"
    return f"{where} at ({tx},{ty})"


def describe_scene(state: dict, kb=None) -> str:
    """A detailed natural-language description of what the Avatar sees now,
    built from the same observation data. General - narrates whatever is present
    (setting, characters, items, doors/exits, map features)."""
    from collections import Counter
    p = state.get("player") or {}
    px, py = p.get("tx", 0), p.get("ty", 0)
    lines = [f"You are at ({px},{py})."]
    if state.get("in_dungeon"):
        lines.append("You are inside a dungeon/enclosed space.")
    if state.get("in_combat"):
        lines.append("You are in COMBAT.")

    nearby = state.get("nearby") or []
    if nearby:
        lines.append("\nPeople and creatures you can see:")
        for n in nearby[:12]:
            who = n.get("name", "someone")
            tag = " (your companion)" if n.get("in_party") else ""
            dead = " - dead" if n.get("dead") else ""
            note = ""
            if kb is not None:
                note = f" [{kb.talk_status(who)}]"
                last = (kb.npcs.get(who, {}).get("notes") or [""])[-1]
                if last:
                    note += f' - last said: "{last[:60]}"'
            lines.append(f"  - {who}{tag}{dead}, {_where(px, py, n.get('dx',0), n.get('dy',0))}{note}")

    objects = state.get("objects") or []
    if objects:
        lines.append("\nObjects and items on the ground:")
        names = Counter(o.get("name") for o in objects if o.get("name"))
        for o in objects[:14]:
            nm = o.get("name")
            cnt = names.get(nm, 1)
            multi = f" (you see {cnt} of these nearby)" if cnt > 1 else ""
            body = " - a body you can search" if o.get("body") else ""
            lines.append(f"  - {nm}, {_where(px, py, o.get('dx',0), o.get('dy',0))}{body}{multi}")

    doors = state.get("doors") or []
    if doors:
        lines.append("\nDoors/exits nearby:")
        for d in doors[:8]:
            stt = "closed" if d.get("closed") else "open"
            lines.append(f"  - a {stt} door {_where(px, py, d.get('dx',0), d.get('dy',0))}")

    grid = state.get("grid") or ""
    counts = {}
    for ch in grid:
        if ch not in "\n.@":
            counts[ch] = counts.get(ch, 0) + 1
    feat = {"T": "trees", "W": "walls/buildings", "=": "fences/gates",
            "~": "water", "n": "containers", "H": "furniture", "#": "blocked areas"}
    present = [feat[c] for c in counts if c in feat and counts[c] >= 2]
    if present:
        lines.append("\nThe surroundings include: " + ", ".join(present) + ".")

    return "\n".join(lines)


def scripted_reply(step: int, state: dict) -> tuple[str, dict]:
    """Deterministic policy used by --dry-run (no Ollama needed)."""
    if state.get("conversation_active") and state.get("answers"):
        return ("Conversation active; picking first answer.", {"type": "answer", "index": 0})
    if state.get("npc_text") and not state.get("answers"):
        return ("NPC talking; advancing text.", {"type": "key", "key": "space"})
    dirs = ["n", "e", "s", "w"]
    d = dirs[step % len(dirs)]
    return (f"Exploring; moving {d}.", {"type": "move", "dir": d})


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _find_exult_exe(explicit: str | None) -> str | None:
    """Locate Exult.exe: explicit path, else repo root above tools/llm_agent."""
    if explicit and os.path.isfile(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))       # tools/llm_agent
    repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    for name in ("Exult.exe", "exult.exe", "exult"):
        cand = os.path.join(repo, name)
        if os.path.isfile(cand):
            return cand
    return None


def ensure_exult_running(args) -> subprocess.Popen | None:
    """If the bridge port isn't open, launch Exult and wait for it. Returns the
    process we started (so we can shut it down), or None if it was already up."""
    if _port_open(args.host, args.port):
        print(f"[+] Exult already listening on {args.host}:{args.port}")
        return None
    if args.no_launch:
        raise ConnectionError(
            f"Exult not running on {args.host}:{args.port} and --no-launch set")

    exe = _find_exult_exe(args.exult_exe)
    if not exe:
        raise FileNotFoundError(
            "Could not find Exult.exe. Pass --exult-exe <path> or start Exult "
            "manually with: Exult.exe --bg --nomenu --llmagent")

    cwd = os.path.dirname(exe)
    cmd = [exe, "--bg", "--nomenu", "--llmagent"]
    if args.port != 45999:
        cmd += ["--llmagent-port", str(args.port)]
    print(f"[+] Launching Exult: {' '.join(cmd)}")
    out = open(os.path.join(cwd, "run_out.log"), "w")
    err = open(os.path.join(cwd, "run_err.log"), "w")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=err)

    # Wait for the bridge to come up (Exult loads data, then opens the port).
    for _ in range(60):
        if _port_open(args.host, args.port):
            print(f"[+] Exult bridge is up on {args.host}:{args.port}")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"Exult exited during startup (code {proc.returncode}); "
                f"see {cwd}\\run_err.log")
        time.sleep(1.0)
    raise TimeoutError("Exult did not open the agent port within 60s")


META_TOOLS = {"add_quest", "update_quest", "note_npc"}


def _apply_meta(action: dict, kb: "KnowledgeBase") -> str:
    """Apply a journal meta-tool to the knowledge base. Returns a short note."""
    t = action.get("type")
    if t == "add_quest":
        qid = kb.add_quest(
            title=action.get("title", "quest"),
            priority=action.get("priority", 5),
            notes=action.get("notes", ""),
            depends_on=action.get("depends_on"),
            status=action.get("status", "active"))
        return f"added quest '{qid}'"
    if t == "update_quest":
        qid = action.get("id") or action.get("title", "")
        status = action.get("status")
        kb.update_quest(
            qid,
            status=status,
            priority=action.get("priority"),
            notes=action.get("notes"),
            depends_on=action.get("depends_on"),
            title=action.get("title"))
        # Only COMPLETING a quest counts as progress worth revisiting NPCs for.
        if status == "done":
            kb.reset_talk_gate()
        return f"updated quest '{qid}'"
    if t == "note_npc":
        kb.note_npc(action.get("name", ""), action.get("note", ""))
        return f"noted NPC '{action.get('name','')}'"
    return "no-op"


def _pursue_focus_quest(state: dict, kb: "KnowledgeBase"):
    """Turn the highest-priority actionable quest into a concrete action:
    if its NPC is nearby -> talk to them; else -> goto their last-known
    position. Returns (action, reason) or (None, None) if not applicable."""
    party = {n.get("name") for n in (state.get("nearby") or []) if n.get("in_party")}
    qv = kb.quest_view()
    # Consider the focus quest, then other actionable quests, skipping any whose
    # NPC is a party companion (they follow you, so "go to them" is pointless).
    for focus in qv.get("open", []):
        if not focus:
            continue
        npc = focus.get("npc")
        if not npc or npc in party:
            continue
        # Is the quest's NPC visible right now?
        seen = False
        for n in (state.get("nearby") or []):
            if n.get("name") == npc and not n.get("dead"):
                seen = True
                if kb.talk_status(npc) != "exhausted":
                    return ({"type": "talk", "name": npc},
                            f"(quest) pursuing '{focus.get('title')}': talk {npc}")
                break
        if not seen:
            pos = kb.npc_last_pos(npc)
            if pos and pos[0] is not None:
                p = state.get("player") or {}
                # If we're already essentially at their last-known spot but they
                # aren't here, that info is stale - don't goto our own tile in a
                # loop; skip to the next quest / exploration instead.
                if abs(pos[0] - p.get("tx", 0)) + abs(pos[1] - p.get("ty", 0)) <= 2:
                    continue
                return ({"type": "goto", "tx": pos[0], "ty": pos[1]},
                        f"(quest) pursuing '{focus.get('title')}': goto {npc}")
    return None, None


def _explore_far(state: dict, session: dict, wedged: bool) -> dict:
    """Pick a distant goto target in a direction that is actually OPEN on the
    grid (avoid heading into ocean/walls). When wedged, rotate to a brand-new
    direction each turn until we break free."""
    p = state.get("player") or {}
    tx, ty = p.get("tx", 0), p.get("ty", 0)
    rows = (state.get("grid") or "").split("\n")
    cx = cy = 12

    def openness(dx, dy):
        score = 0
        for r in range(1, 10):
            x, y = cx + dx * r, cy + dy * r
            if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                ch = rows[y][x]
                if ch in ".&C*x/":
                    score += 1
                elif ch == "#":
                    break
        return score

    dirs = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0),
            "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1)}
    tried = session.setdefault("failed_dirs", set())
    ranked = sorted(dirs.items(), key=lambda kv: -openness(*kv[1]))
    choice = None
    for name, (dx, dy) in ranked:
        if wedged and name in tried:
            continue
        if openness(dx, dy) >= 3:
            choice = (name, dx, dy)
            break
    if not choice and ranked:
        name, (dx, dy) = ranked[0]
        choice = (name, dx, dy)
    name, dx, dy = choice
    if wedged:
        tried.add(name)
        if len(tried) >= 6:
            tried.clear()
    else:
        tried.clear()
    # Shorter hop (6 tiles): the engine A* has a bounded search budget, so far
    # targets often fail; a nearer target in the most-open direction routes
    # reliably, and the engine now single-steps toward it if A* still gives up.
    return {"type": "goto", "tx": tx + dx * 6, "ty": ty + dy * 6}


def _do_turn(args, window, ollama, exult, step, recent_positions, kb, session) -> None:
    state = exult.observe()
    if window.available:
        window.update_turn(step)
        window.set_map(state.get("grid") or "(no map)")
        window.set_dialog(format_dialog(state))
        # Inspector panels: quests, NPC knowledge, and stats.
        try:
            nearby_names = [n.get("name") for n in (state.get("nearby") or []) if n.get("name")]
            window.set_quests(kb.quests_pretty())
            window.set_npc_tree(kb.npcs_tree_data())
            window.set_topics_tree(kb.topics_tree_data())
            p = state.get("player") or {}
            stats = [
                f"pos: ({p.get('tx')},{p.get('ty')})  hp:{p.get('hp')}  food:{p.get('food')}",
                f"in_combat:{state.get('in_combat')}  conv:{state.get('conversation_in_progress')}",
                f"NPCs met: {len(kb.npcs)}   places mapped: {len(kb.places)}",
                f"quests: {len(kb.quests)}   journal: {len(kb.journal)}",
                f"dialogue mem: {len(kb.dialogue_history)}   actions mem: {len(kb.action_history)}",
                f"hints: {len(kb.hints)}   observations: {len(kb.observations)}",
                f"story_so_far: {len(kb.episodic_summary)} chars",
            ]
            if session.get("last_ctx"):
                stats.append(session["last_ctx"])
            # Tool-call stats: turns, parse failures, and per-tool ok/err.
            stats.append("--- tool calls ---")
            stats.append(kb.tool_stats_pretty(top=10))
            window.set_stats("\n".join(stats))
        except Exception:
            pass

    # Safety net: keep the party fed so a long run can't starve.
    if args.auto_feed and step % args.auto_feed_every == 0:
        exult.act({"type": "feed", "level": 30})

    if not state.get("world_loaded"):
        if window.available:
            window.set_thinking("World not loaded yet; waiting...")
        time.sleep(args.delay)
        return

    # Remember where nearby NPCs are, so quests involving them can be navigated
    # to later even after we walk away.
    _p = state.get("player") or {}
    _ptx, _pty = _p.get("tx", 0), _p.get("ty", 0)
    # Record a persistent "home base" (starting area) the very first loaded turn
    # so the agent can always navigate back even after wandering far.
    if not kb.place_pos("start area"):
        kb.record_place("start area", _ptx, _pty, kind="home",
                        note="where you began; a safe town to return to")
    for _n in (state.get("nearby") or []):
        if _n.get("name") and not _n.get("dead"):
            kb.see_npc(_n["name"], _ptx + _n.get("dx", 0), _pty + _n.get("dy", 0))
    # Build the mental map: record only genuine NAVIGATION landmarks a human
    # would note (buildings/purpose, gates, signs, stairs, wells, bridges) - not
    # furniture/clutter (bed, chest, door, inkwell) which added noise without
    # helping navigation. Purpose-y words get a "building" kind so the agent can
    # reason about where things are (e.g. an inn/smithy/temple).
    _BUILDING_WORDS = ("inn", "tavern", "shop", "temple", "smithy", "forge",
                       "stables", "church", "shrine", "guild", "bank", "market")
    _LANDMARK_WORDS = ("sign", "gate", "gateway", "stairs", "ladder", "bridge",
                       "well", "fountain", "statue", "fortress")
    for _o in (state.get("objects") or []):
        nm = (_o.get("name") or "").lower()
        if any(w in nm for w in _BUILDING_WORDS):
            kb.record_place(_o["name"], _ptx + _o.get("dx", 0),
                            _pty + _o.get("dy", 0), kind="building")
        elif any(w in nm for w in _LANDMARK_WORDS):
            kb.record_place(_o["name"], _ptx + _o.get("dx", 0),
                            _pty + _o.get("dy", 0), kind="landmark")

    # Persist notable OBSERVATIONS (deduped): ambient speech overheard, plus
    # notable objects seen (bodies, chests, keys, etc). This gives the agent a
    # durable record of "things I saw / heard" beyond the current frame.
    for _a in (state.get("ambient_speech") or []):
        _said = (_a.get("said") or "").strip()
        if _said:
            _who = _a.get("who") or "someone"
            if kb.note_observation(f'{_who}: "{_said}"', kind="heard", step=step):
                print(f"[{step:03d}] overheard {_who}: {_said}")
    for _o in (state.get("objects") or []):
        _onm = _o.get("name") or ""
        if _o.get("body") or kb.is_notable_object(_onm):
            _ox, _oy = _ptx + _o.get("dx", 0), _pty + _o.get("dy", 0)
            kb.note_observation(f"{_onm} at ({_ox},{_oy})", kind="seen", step=step)

    # Record NPC dialog into the journal + dialogue history (with speaker).
    npc_text = state.get("npc_text")
    if state.get("conversation_in_progress") and npc_text:
        kb.add_journal(npc_text)
        cur_npc = session.get("current_npc", "?")
        # If we don't have a reliable partner (conversation started via goto or
        # some path that didn't set current_npc), infer it as the NEAREST
        # non-party NPC. Without this, dialogue trees for such NPCs (e.g.
        # Finnigan reached via goto) stay empty. Only infer when unset.
        if cur_npc in ("?", "", None):
            _cands = [n for n in (state.get("nearby") or [])
                      if n.get("name") and not n.get("in_party") and not n.get("dead")]
            if _cands:
                _near = min(_cands, key=lambda n: abs(n.get("dx", 99)) + abs(n.get("dy", 99)))
                cur_npc = _near["name"]
                session["current_npc"] = cur_npc
        kb.record_npc_line(cur_npc, npc_text)
        # Capture the answer TOPICS offered now (the dialogue-tree branches) so
        # the agent has a persistent record of what it can still ask this NPC.
        if state.get("answers"):
            kb.record_npc_choices(cur_npc, state.get("answers"))
        # Passively capture durable knowledge (NPC notes + task-like quests)
        # from each NEW line, so the structured memory builds up even though
        # the model rarely calls the journal tools itself.
        if npc_text != session.get("last_captured_line"):
            session["last_captured_line"] = npc_text
            # Party companions follow you, so a "follow up with them" quest is a
            # useless navigation target - just note what they said, no quest.
            _party = {n.get("name") for n in (state.get("nearby") or []) if n.get("in_party")}
            if cur_npc in _party:
                kb.note_npc(cur_npc, npc_text[:200])
            else:
                kb.auto_note_from_dialogue(cur_npc, npc_text)
            # Cross-cutting TOPIC capture: file this line under the subject the
            # agent last asked about (the chosen dialogue topic), so knowledge
            # about themes like "the Fellowship" aggregates across every NPC.
            # NOTE: topics are now authored solely by the LLM (via the add_topic
            # tool) so the topic list reflects its own evolving understanding -
            # we no longer auto-file verbatim dialogue into topics here.
    if state.get("conversation_in_progress"):
        # Remember who we're conversing with, to auto-close trivial 'talk to X'
        # quests when the conversation ends.
        _cn = session.get("current_npc")
        if _cn and _cn != "?":
            session["was_conversing_with"] = _cn
    else:
        session.pop("current_topic", None)
        _wc = session.pop("was_conversing_with", None)
        if _wc:
            _done = kb.auto_resolve_talk_quests(_wc)
            for _t in _done:
                print(f"[{step:03d}] auto-resolved quest (talked to {_wc}): {_t}")
            if _done:
                kb.reset_talk_gate()

    if args.dry_run:
        reason, action = scripted_reply(step, state)
    else:
        # Pull any user hint typed into the GUI; keep it active for a few turns
        # AND record it permanently so the agent can recall it later.
        h = None
        if window.available:
            h = window.get_hint()
        # Also accept a hint dropped into a file (hint.txt next to the driver),
        # so hints can be sent without the GUI (scriptable). The file is
        # consumed (emptied) once read.
        if not h:
            try:
                hf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hint.txt")
                if os.path.isfile(hf):
                    txt = open(hf, encoding="utf-8").read().strip()
                    if txt:
                        h = txt
                        open(hf, "w", encoding="utf-8").close()  # consume it
            except OSError:
                pass
        if h:
            session["hint"] = h
            session["hint_ttl"] = 3
            kb.record_hint(h, step)
            print(f"[{step:03d}] USER HINT: {h}")
        alert_parts = []
        if session.get("hint") and session.get("hint_ttl", 0) > 0:
            alert_parts.append("HINT from your operator (follow it): " + session["hint"])
            session["hint_ttl"] -= 1
            if session["hint_ttl"] <= 0:
                session.pop("hint", None)
        if session.get("last_bump"):
            alert_parts.append(session["last_bump"])
        alert = "  ".join(alert_parts)
        # --- Context budget feedback loop -------------------------------
        # Decide how hard to squeeze the raw context tiers based on LAST turn's
        # measured prompt size relative to num_ctx. As the playthrough grows and
        # the prompt creeps toward the limit, shrink the verbose windows and
        # fold the dialogue that scrolls off into the durable episodic summary,
        # so clues survive compression instead of being dropped.
        ctx_max = session.get("ctx_max") or ollama.num_ctx or 8192
        last_pt = session.get("last_prompt_tokens", 0)
        # Reserve headroom for the RESPONSE: if prompt+response would exceed
        # num_ctx, Ollama returns an EMPTY reply (the observed 90-96% -> ''
        # parse failures). Budget the prompt against a usable ceiling that
        # leaves room to generate, and squeeze hard well before the real limit.
        reserve = 500  # tokens kept free for the model's answer
        usable = max(1, ctx_max - reserve)
        frac = last_pt / usable
        if frac >= 0.80:
            squeeze = 3
        elif frac >= 0.60:
            squeeze = 2
        elif frac >= 0.45:
            squeeze = 1
        else:
            squeeze = 0
        session["squeeze"] = squeeze
        # When under pressure, compress old dialogue into story_so_far and drop
        # it from the raw window (keep the most recent exchanges intact).
        if squeeze >= 2:
            keep = 8 if squeeze == 2 else 4
            folded = kb.fold_dialogue_into_summary(keep_recent=keep)
            if folded:
                print(f"[{step:03d}] context {int(100*last_pt/ctx_max)}%: folded {folded} old "
                      f"dialogue lines into story_so_far (squeeze={squeeze})")
        # If the agent just used "recall", surface that character's saved
        # dialogue tree in THIS turn's state (one-shot, then cleared).
        _recalled = session.pop("recalled", None)
        if _recalled is not None:
            state["recalled"] = _recalled
        # Tell summarize_state who we're talking to, so it can proactively show
        # which topics we've already asked this NPC and which we have NOT.
        if state.get("conversation_in_progress"):
            state["_convo_npc"] = session.get("current_npc")
        _user = summarize_state(state, kb, session.pop("last_look", ""), alert, squeeze)
        res = ollama.chat_ex(SYSTEM_PROMPT, _user)
        reply = res["content"]
        reason, action = parse_reply(reply)
        # Tool-call stats: count this turn and whether the model's reply parsed
        # (a parse failure means we could not read an action and fell back to
        # wait - visible now as parse-fail in the stats panel).
        parsed_ok = reason != "(could not parse reply)"
        kb.record_turn(parsed_ok)
        # Raw comms log: append the exact prompt+reply for this turn so parse
        # errors can be diagnosed from ground truth. Always logs parse failures;
        # logs everything when --raw-log is set. Written next to the driver.
        if (not parsed_ok) or getattr(args, "raw_log", False):
            try:
                _rl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_comms.log")
                with open(_rl, "a", encoding="utf-8") as _f:
                    _f.write(f"\n===== step {step} | parsed_ok={parsed_ok} | "
                             f"ptok={res.get('prompt_tokens')} rtok={res.get('response_tokens')} "
                             f"retried={res.get('retried')} dropped_json={res.get('dropped_json')} =====\n")
                    _f.write("----- USER PROMPT -----\n" + _user + "\n")
                    _f.write("----- RAW REPLY -----\n" + repr(reply) + "\n")
            except OSError:
                pass
        if not parsed_ok:
            print(f"[{step:03d}] parse-fail: {reply[:120]!r}")
        # Inline quest capture: the model may include a "new_quest" (or "quests")
        # field ALONGSIDE its action, so it can record a goal WITHOUT spending
        # its one game action on a journal tool. Accept a dict or list of dicts
        # with at least a title; also accept "resolve_quest" to mark one done.
        try:
            _obj = json.loads(reply)
        except Exception:
            _obj = None
        if isinstance(_obj, dict):
            _nq = _obj.get("new_quest") or _obj.get("quests")
            _items = _nq if isinstance(_nq, list) else ([_nq] if isinstance(_nq, dict) else [])
            for q in _items:
                if isinstance(q, dict) and q.get("title"):
                    qid = kb.add_quest(title=str(q["title"]),
                                       priority=q.get("priority", 5),
                                       notes=str(q.get("notes", "")),
                                       depends_on=q.get("depends_on"),
                                       status=q.get("status", "active"))
                    print(f"[{step:03d}] inline add_quest: {q['title']!r} -> {qid}")
            _rq = _obj.get("resolve_quest")
            if isinstance(_rq, str) and _rq:
                kb.resolve_quest(_rq)
                kb.reset_talk_gate()
                print(f"[{step:03d}] inline resolve_quest: {_rq}")
            # Inline LLM-authored TOPICS: {"topic":"Fellowship","note":"..."} or a
            # list. Topics now come ONLY from the LLM, so the topic list shows
            # its own thinking accumulating over time.
            _tp = _obj.get("topic") or _obj.get("topics") or _obj.get("add_topic")
            _tp_items = _tp if isinstance(_tp, list) else ([_tp] if _tp else [])
            for t in _tp_items:
                if isinstance(t, dict) and (t.get("name") or t.get("topic")):
                    tname = str(t.get("name") or t.get("topic"))
                    tkey = kb.add_topic(tname, str(t.get("note", "")), step)
                    print(f"[{step:03d}] inline add_topic: {tname!r} -> {tkey}")
                elif isinstance(t, str) and t.strip():
                    kb.add_topic(t.strip(), "", step)
        # Track context usage so we can see if the prompt is bloating/truncating.
        pt = res.get("prompt_tokens", 0)
        session["last_prompt_tokens"] = pt
        pct = int(100 * pt / ctx_max) if ctx_max else 0
        session["last_ctx"] = (f"context: {pt} prompt + {res.get('response_tokens',0)} resp "
                               f"tok / {ctx_max} ({pct}%)  squeeze={squeeze}")
        if pt > 0.9 * ctx_max:
            print(f"[{step:03d}] WARNING: prompt {pt} tok near context limit {ctx_max}")
        if window.available:
            window.set_context(pct, f"{pt} prompt + {res.get('response_tokens',0)} resp / {ctx_max} tok ({pct}%)")
            window.set_thinking(reason if reason else reply)

    if window.available and args.dry_run:
        window.set_thinking(reason)

    # --- "recall": retrieve the FULL saved dialogue tree for a character -
    #     everything they said (their transcript), topics already asked, and
    #     topics still unasked. Does not advance the game; the result is shown
    #     in next turn's observation as "recalled" so the agent can remember,
    #     e.g., the Mayor's instructions from a past run.
    if isinstance(action, dict) and action.get("type") == "recall":
        who = str(action.get("name") or action.get("npc") or action.get("topic") or "").strip()
        rec = kb.recall_npc(who) if who else {"known": False, "name": who}
        # If it's not a known person, try the shared topic KB (e.g. "Fellowship").
        if who and not rec.get("known"):
            trec = kb.recall_topic(who)
            if trec.get("known"):
                rec = {"topic": trec["name"], "known": True,
                       "notes": trec.get("notes", [])}
        session["recalled"] = rec
        kb.record_action(f"recalled {who}")
        if window.available:
            window.set_action(f"[recall] {who}: {rec}")
        print(f"[{step:03d}] recall {who}: known={rec.get('known')}")
        action = {"type": "wait"}
        reason = f"(recalled {who})"

    # --- "annotate": mark a location on the mental map so the agent can find
    #     its way back later (via goto <label> or known_places). Records the
    #     given tile, or the avatar's current position if none given.
    if isinstance(action, dict) and action.get("type") == "annotate":
        label = str(action.get("label") or action.get("name") or "").strip()
        pp = state.get("player") or {}
        tx = action.get("tx", pp.get("tx", 0))
        ty = action.get("ty", pp.get("ty", 0))
        if label:
            kb.record_place(label, tx, ty, kind="marked", note=str(action.get("note", "")))
            kb.record_action(f"annotated '{label}' @({tx},{ty})")
            if window.available:
                window.set_action(f"[annotate] {label} @ ({tx},{ty})")
            print(f"[{step:03d}] annotate: {label} @ ({tx},{ty})")
        action = {"type": "wait"}
        reason = f"(marked '{label}' on the map)"

    # --- "look": produce a detailed description of the surroundings. It does
    #     not advance the game; we record it so the model sees it next turn and
    #     avoids looking repeatedly.
    if isinstance(action, dict) and action.get("type") == "look":
        # If we just looked and haven't moved, looking again is wasted - the
        # scene is unchanged. Redirect a repeat look into real progress.
        _recent = kb.action_view(3)
        _pp = state.get("player") or {}
        _here = (_pp.get("tx"), _pp.get("ty"))
        if _recent and _recent[-1] == "looked around" and session.get("last_look_pos") == _here:
            action = _explore_far(state, session, session.get("stuck_count", 0) >= 3)
            reason = "(guard) already looked here; exploring instead of looking again"
        else:
            desc = describe_scene(state, kb)
            kb.record_action("looked around")
            session["last_look"] = desc
            session["last_look_pos"] = _here
            if window.available:
                window.set_action("[look]\n" + desc[:1500])
            print(f"[{step:03d}] look:\n{desc}")
            # Fall through to a light action so the turn still progresses.
            action = {"type": "wait"}
            reason = "(looked around; see description)"

    # --- Journal meta-tools: update the KB, then take a game action too. ---
    if isinstance(action, dict) and action.get("type") in META_TOOLS:
        note = _apply_meta(action, kb)
        if window.available:
            window.set_action(f"[journal] {note}\n{json.dumps(action)}")
        print(f"[{step:03d}] journal: {note} :: {reason!r}")
        # Meta-tools don't advance the game; fall through with a light game
        # action so the turn still does something useful.
        action = {"type": "wait"}
        reason = f"(after {note})"

    # --- Driver-side guards to keep behavior sane ------------------------
    MAX_TALKS = 3              # re-talk limit *since last progress* (resets on
                               # meaningful progress so NPCs can be revisited)
    MAX_CONVO_TURNS = 12       # force-end a conversation that drags on
    def talked(nm):
        return kb.talked_recently(nm)

    def exhausted(nm):
        # True if this NPC is tapped out either since last progress OR by total
        # talk count (the hard cap). Used by the anti-re-talk guards so a heavily
        # talked NPC (e.g. Johnson x44) is not re-approached after progress
        # resets the per-epoch counter.
        return bool(nm) and kb.talk_status(nm) == "exhausted"

    in_convo = bool(state.get("conversation_in_progress"))
    answers = state.get("answers") or []

    # Global stuck detection: if the avatar's tile hasn't changed for several
    # turns while not in a conversation, we're wedged (e.g. against the ocean
    # or a wall). Track this so exploration can pick a genuinely new heading.
    p_now = state.get("player") or {}
    pos_now = (p_now.get("tx"), p_now.get("ty"))
    if in_convo:
        session["stuck_count"] = 0
    elif pos_now == session.get("last_pos"):
        session["stuck_count"] = session.get("stuck_count", 0) + 1
    else:
        session["stuck_count"] = 0
    session["last_pos"] = pos_now
    wedged = session.get("stuck_count", 0) >= 3
    # Also detect the specific "same action, no progress" trap (e.g. repeating a
    # goto that silently fails): if position is unchanged across recent turns
    # regardless of conversation, treat as wedged too.
    hist = session.setdefault("pos_hist", [])
    hist.append(pos_now)
    if len(hist) > 6:
        hist.pop(0)
    if len(hist) >= 5 and len(set(hist)) == 1 and pos_now[0] is not None and not in_convo:
        wedged = True
    if not wedged:
        session["wedge_try"] = 0

    # Track how long we've been in the current conversation, and which answer
    # choices we've already picked, so we can detect a loop and bail out.
    if in_convo:
        session["convo_turns"] = session.get("convo_turns", 0) + 1
    else:
        session["convo_turns"] = 0
        session["picked_answers"] = set()

    def _bye_action():
        """Choose the answer that ends the conversation ('bye'/'leave'/last)."""
        low = [a.lower() for a in answers]
        for kw in ("bye", "leave", "farewell", "goodbye", "nothing", "done"):
            for i, a in enumerate(low):
                if kw in a:
                    return {"type": "answer", "index": i}
        # Fall back to the last option (usually the exit), else press escape.
        if answers:
            return {"type": "answer", "index": len(answers) - 1}
        return {"type": "key", "key": "escape"}

    # 0a) A numeric slider prompt is up: answer it. Use the model's value if it
    #     chose set_number, else default to a sensible amount (the max, i.e.
    #     "all"). This prevents getting stuck on the slider/checkbox GUI.
    if state.get("number_prompt"):
        lo = state.get("number_min", 0)
        hi = state.get("number_max", lo)
        if isinstance(action, dict) and action.get("type") == "set_number" \
                and isinstance(action.get("value"), int):
            val = max(lo, min(hi, action["value"]))
        else:
            val = hi  # default: take the full/maximum amount
        exult.act({"type": "set_number", "value": val})
        if window.available:
            window.set_action(f'{{"type":"set_number","value":{val}}}')
        print(f"[{step:03d}] number_prompt -> set_number {val} (range {lo}-{hi})")
        time.sleep(args.delay)
        return

    # 0b) A container/body gump is open (blocks movement). Let the model take
    #     or close it, but if it dithers (or does anything else), auto-loot the
    #     remaining contents then close so it never gets stuck at an open bag.
    if state.get("gump_open") and not state.get("conversation_in_progress"):
        act_type = action.get("type") if isinstance(action, dict) else None
        if act_type in ("take", "close", "equip"):
            pass  # let the model's own take/close/equip proceed
        else:
            session["gump_wait"] = session.get("gump_wait", 0) + 1
            # Loot ONLY the container whose gump is actually open (reported as
            # gump_contents by the engine), not every nearby container - that
            # bug made us 'take torch/scroll' from a different bag when the open
            # body was already empty.
            loot = list(state.get("gump_contents") or [])
            already = session.setdefault("gump_taken", set())
            todo = [c for c in loot if c not in already]
            if todo and session["gump_wait"] <= 6:
                item = todo[0]
                already.add(item)
                action = {"type": "take", "name": item}
                reason = f"(guard) container open; taking {item}"
            else:
                action = {"type": "close"}
                reason = "(guard) done looting; closing container"
                session["gump_wait"] = 0
                session["gump_taken"] = set()
                _pp = state.get("player") or {}
                looted = session.setdefault("looted_spots", set())
                looted.add((_pp.get("tx"), _pp.get("ty")))
        # CRITICAL: a gump BLOCKS movement - the avatar cannot walk until it is
        # closed. Execute the take/close/equip NOW and end the turn so the later
        # wedge/goto/move guards can't clobber it with a movement action that
        # would silently no-op (this was the 'tried to move before closing the
        # body' + stuck-in-place bug).
        _atype = action.get("type") if isinstance(action, dict) else "?"
        result = exult.act(action)
        kb.record_tool(_atype, result.get("ok") if isinstance(result, dict) else None)
        session["last_action_type"] = _atype
        if window.available:
            window.set_action(json.dumps(action) + "\n\n-> " + json.dumps(result))
            window.set_thinking(reason)
        return
    else:
        session["gump_wait"] = 0
        session["gump_taken"] = set()

    # 0-pre) LEAVING LATCH: once we decide to end a conversation, keep driving
    #   toward closing it every turn until it is actually closed. Conversations
    #   are unreliable to end (picking 'bye' shows a farewell line, needs a
    #   space, and sometimes re-opens the choice list). Without a latch the
    #   model re-engages a topic and we get stuck. While leaving: if choices are
    #   up, pick the bye/exit answer; if a text page is up, space past it; when
    #   in_progress goes false, clear the latch.
    if session.get("leaving"):
        if not state.get("conversation_in_progress"):
            session["leaving"] = False
        else:
            if state.get("conversation_active") and answers:
                action = _bye_action()
                reason = "(guard) leaving: choosing exit reply"
            else:
                action = {"type": "key", "key": "space"}
                reason = "(guard) leaving: clearing dialog text"
            # Safety: if we've been "leaving" too many turns, force escape.
            session["leaving_n"] = session.get("leaving_n", 0) + 1
            if session["leaving_n"] > 8:
                action = {"type": "key", "key": "escape"}
                reason = "(guard) leaving: forcing escape"
                session["leaving_n"] = 0
            # Skip the rest of the conversation guards this turn.
            _atype = action.get("type") if isinstance(action, dict) else "?"
            if _atype == "talk":
                tname = action.get("name", "")
                if tname:
                    kb.mark_talked(tname); session["current_npc"] = tname
                result = exult.talk(tname)
            else:
                result = exult.act(action)
            kb.record_tool(_atype, result.get("ok") if isinstance(result, dict) else None)
            if window.available:
                window.set_action(json.dumps(action) + "\n\n-> " + json.dumps(result))
                window.set_thinking(reason)
            return

    # 0) Auto-end a conversation that has gone on too long or is repeating the
    #    same answer choices (the model won't pick 'bye' on its own).
    if state.get("conversation_active") and answers:
        picked = session.setdefault("picked_answers", set())
        chosen_idx = action.get("index") if isinstance(action, dict) and action.get("type") == "answer" else None
        too_long = session.get("convo_turns", 0) >= MAX_CONVO_TURNS
        repeating = chosen_idx is not None and chosen_idx in picked and len(picked) >= max(1, len(answers) - 1)
        if too_long or repeating:
            # Engage the leaving latch so we drive to a clean close, not just
            # a single bye that may re-open choices.
            session["leaving"] = True
            session["leaving_n"] = 0
            action = _bye_action()
            reason = f"(guard) ending conversation ({'too long' if too_long else 'looping'})"
        elif chosen_idx is not None:
            picked.add(chosen_idx)

    # 1) If a conversation is open but no choices are shown yet, advance text.
    #    But if it has stayed 'in progress' with no choices for many turns
    #    (a hung conversation), Escape out of it instead of pressing space
    #    forever - space isn't advancing it.
    if state.get("conversation_in_progress") and not state.get("conversation_active"):
        cur_text = state.get("npc_text")
        if cur_text == session.get("stuck_convo_text"):
            session["stuck_convo_n"] = session.get("stuck_convo_n", 0) + 1
        else:
            session["stuck_convo_n"] = 0
            session["stuck_convo_text"] = cur_text
        if session.get("stuck_convo_n", 0) >= 4:
            action = {"type": "key", "key": "escape"}
            reason = "(guard) conversation hung with no choices; escaping"
            session["stuck_convo_n"] = 0
        else:
            action = {"type": "key", "key": "space"}
            reason = "(guard) advancing NPC dialog"
    # 0.5) Wedged: try hard to escape. Likely sealed in a building (closed
    #      door) or against terrain. Try in order: open a door, then goto a
    #      ring of far tiles (pathfinder opens doors it walks through), then raw
    #      moves toward open grid tiles.
    elif wedged:
        n = session.get("wedge_try", 0)
        session["wedge_try"] = n + 1
        p = state.get("player") or {}
        px, py = p.get("tx", 0), p.get("ty", 0)
        rows = (state.get("grid") or "").split("\n")
        cx = cy = 12
        def _cell(dx, dy):
            x, y = cx + dx, cy + dy
            if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                return rows[y][x]
            return "#"
        WALK = ".*&C/xno~+"  # walkable-ish glyphs (incl. open/closed door, items)
        # Flood-fill from the avatar over walkable cells to find reachable open
        # tiles, then pick the FARTHEST reachable one as a concrete goto target.
        # This escapes tight enclosures (e.g. a fenced murder scene with one
        # gate) far better than blind directional steps or fixed-offset gotos,
        # because it only targets tiles that are actually connected to us.
        from collections import deque as _deque
        seen = {(0, 0)}
        q = _deque([(0, 0)])
        best = None
        best_d = -1
        R = 12
        while q:
            dx, dy = q.popleft()
            d = abs(dx) + abs(dy)
            if d > best_d and (dx, dy) != (0, 0):
                best_d, best = d, (dx, dy)
            for ndx, ndy in ((0,-1),(0,1),(1,0),(-1,0)):
                nx, ny = dx+ndx, dy+ndy
                if abs(nx) > R or abs(ny) > R or (nx, ny) in seen:
                    continue
                if _cell(nx, ny) in WALK:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        # If the flood found nothing reachable except through a closed door,
        # open it first; otherwise goto the farthest reachable open tile.
        has_door = any(_cell(dx, dy) == "+"
                       for dx in range(-3, 4) for dy in range(-3, 4))
        if best and best_d >= 2:
            action = {"type": "goto", "tx": px + best[0], "ty": py + best[1]}
            reason = f"(guard) wedged; flood-fill escape to open tile ({px+best[0]},{py+best[1]})"
        elif has_door or (state.get("doors") or []):
            action = {"type": "open"}
            reason = "(guard) wedged; opening a nearby door to escape"
        else:
            # Truly boxed in on the visible grid - step toward any adjacent
            # walkable cell, rotating by attempt to avoid oscillating.
            deltas = {"n": (0,-1),"s": (0,1),"e": (1,0),"w": (-1,0),
                      "ne": (1,-1),"nw": (-1,-1),"se": (1,1),"sw": (-1,1)}
            pref = list(deltas.keys())
            off = n % len(pref)
            pref = pref[off:] + pref[:off]
            picked = next((d for d in pref if _cell(*deltas[d]) in WALK), pref[0])
            action = {"type": "move", "dir": picked, "speed": 120}
            reason = f"(guard) wedged; stepping {picked}"
    else:
        # 1b) A "space"/"key" press outside a conversation does nothing.
        if (isinstance(action, dict) and action.get("type") == "key"
                and not state.get("conversation_in_progress")):
            fresh = [n for n in (state.get("nearby") or [])
                     if not n.get("dead") and not exhausted(n.get("name"))]
            if fresh:
                target = min(fresh, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
                action = {"type": "talk", "name": target["name"]}
                reason = f"(guard) no conversation open; talking to {target['name']}"
            else:
                qa, qr = _pursue_focus_quest(state, kb)
                if qa:
                    action, reason = qa, qr
                else:
                    action = _explore_far(state, session, wedged)
                    reason = "(guard) no active quest lead; exploring a new area"
        # 2) Let the model talk to whoever it chose - it can see each NPC's
        #    recorded notes + talk count and decides for itself whether there is
        #    more to learn. We only break a genuine TIGHT LOOP: the same NPC
        #    chosen many turns in a row with nothing else happening (a stuck
        #    model), not merely "talked before".
        if isinstance(action, dict) and action.get("type") == "talk":
            nm = action.get("name", "")
            # Don't try to talk to a SLEEPING NPC - the conversation won't open.
            _sleeping = {n.get("name") for n in (state.get("nearby") or [])
                         if n.get("condition") == "sleeping"}
            if nm and nm in _sleeping:
                session["last_bump"] = (
                    f"{nm} is SLEEPING and cannot be talked to right now. Come "
                    f"back in the daytime, or pursue another goal meanwhile.")
                qa, qr = _pursue_focus_quest(state, kb)
                if qa and not (qa.get("type") == "talk" and qa.get("name") == nm):
                    action, reason = qa, qr
                else:
                    action = _explore_far(state, session, wedged)
                    reason = f"(guard) {nm} is sleeping; doing something else"
                print(f"[{step:03d}] talk-guard: {nm} is sleeping; skipping")
            # Party COMPANIONS (in_party) have mostly static dialogue - once
            # you've spoken to them a couple times there is rarely anything new,
            # yet the model loves to re-interview them (Iolo was talked to 140+
            # times). If the target is a companion we've already talked to,
            # redirect to pursuing a quest lead or exploring somewhere new.
            _party = {n.get("name") for n in (state.get("nearby") or []) if n.get("in_party")}
            if nm in _party and kb.times_talked(nm) >= 2:
                qa, qr = _pursue_focus_quest(state, kb)
                if qa and not (qa.get("type") == "talk" and qa.get("name") in _party):
                    action, reason = qa, qr
                else:
                    # Commit to a STICKY exploration destination for several
                    # turns instead of recomputing a new heading each turn (which
                    # made the avatar oscillate between two tiles while the
                    # companion kept following and re-triggering this guard).
                    _pp = state.get("player") or {}
                    _here = (_pp.get("tx"), _pp.get("ty"))
                    goal = session.get("explore_goal")
                    ttl = session.get("explore_goal_ttl", 0)
                    reached = goal and abs(goal[0]-_here[0]) + abs(goal[1]-_here[1]) <= 3
                    if not goal or ttl <= 0 or reached:
                        far = _explore_far(state, session, wedged)
                        goal = (far.get("tx"), far.get("ty")) if far.get("type") == "goto" else None
                        session["explore_goal"] = goal
                        session["explore_goal_ttl"] = 8
                        action = far
                    else:
                        session["explore_goal_ttl"] = ttl - 1
                        action = {"type": "goto", "tx": goal[0], "ty": goal[1]}
                    reason = f"(guard) {nm} is a companion; travelling to explore (goal {goal})"
                session["last_talk_target"] = nm
            else:
                last_nm = session.get("last_talk_target")
                if nm and nm == last_nm:
                    session["same_talk_streak"] = session.get("same_talk_streak", 0) + 1
                else:
                    session["same_talk_streak"] = 0
                session["last_talk_target"] = nm
                # Only intervene after re-picking the SAME npc 4+ turns straight.
                if nm and session.get("same_talk_streak", 0) >= 4:
                    session["same_talk_streak"] = 0
                    fresh = [n for n in (state.get("nearby") or [])
                             if not n.get("dead") and n.get("name") != nm
                             and not n.get("in_party")
                             and kb.talk_status(n.get("name")) != "exhausted"]
                    if fresh:
                        target = min(fresh, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
                        action = {"type": "talk", "name": target["name"]}
                        reason = f"(guard) stuck re-picking {nm}; trying {target['name']} instead"
                    else:
                        qa, qr = _pursue_focus_quest(state, kb)
                        if qa and not (qa.get("type") == "talk" and qa.get("name") == nm):
                            action, reason = qa, qr
                        else:
                            action = _explore_far(state, session, wedged)
                            reason = f"(guard) stuck re-picking {nm}; exploring instead"
        # 3) Anti-chase / auto-loot when stuck.
        elif (isinstance(action, dict) and action.get("type") == "move"
              and not state.get("conversation_in_progress")):
            pos = (state.get("player") or {}).get("tx"), (state.get("player") or {}).get("ty")
            recent_positions.append(pos)
            if len(recent_positions) > 6:
                recent_positions.pop(0)
            stuck = len(recent_positions) >= 4 and len(set(recent_positions)) <= 2
            fresh = [n for n in (state.get("nearby") or [])
                     if not n.get("dead") and abs(n.get("dx", 99)) <= 4
                     and abs(n.get("dy", 99)) <= 4 and not exhausted(n.get("name"))]
            if stuck and fresh:
                target = min(fresh, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
                action = {"type": "talk", "name": target["name"]}
                reason = f"(guard) stuck near {target['name']}; talking instead of moving"
                recent_positions.clear()
            elif stuck:
                objs = [o for o in (state.get("objects") or [])
                        if abs(o.get("dx", 99)) <= 3 and abs(o.get("dy", 99)) <= 3
                        and not o.get("owned")  # never auto-grab owned property
                        and o.get("name") not in session["picked"]]
                dead_bodies = [n for n in (state.get("nearby") or [])
                               if n.get("dead") and abs(n.get("dx", 99)) <= 2
                               and abs(n.get("dy", 99)) <= 2]
                if dead_bodies and not session.get("searched_body"):
                    action = {"type": "search"}
                    reason = "(guard) stuck near a body; searching it"
                    session["searched_body"] = True
                    recent_positions.clear()
                elif objs:
                    it = min(objs, key=lambda o: abs(o["dx"]) + abs(o["dy"]))
                    action = {"type": "pickup", "name": it["name"]}
                    reason = f"(guard) stuck near {it['name']}; picking it up"
                    session["picked"].add(it["name"])
                    recent_positions.clear()
                else:
                    d = ["n", "e", "s", "w", "ne", "sw"][step % 6]
                    action = {"type": "move", "dir": d}
                    reason = f"(guard) stuck; exploring {d} for new areas"
                    recent_positions.clear()

    # --- Greet-new-people guard: don't walk/goto/explore PAST someone we have
    #     never talked to. If the model's action is movement (move/goto) or a
    #     wait while a NEW (never-talked) NPC is within a few tiles, redirect to
    #     talking to them first. New people can give quests, clues, items, or
    #     JOIN THE PARTY - too valuable to skip while tunnel-visioning a hinted
    #     goal. General behavior, not tied to any specific puzzle. -------------
    if (isinstance(action, dict)
            and action.get("type") in ("move", "goto", "wait", "look")
            and not state.get("conversation_in_progress")
            and not state.get("gump_open")):
        new_npcs = [n for n in (state.get("nearby") or [])
                    if n.get("name") and not n.get("dead")
                    and not n.get("in_party")
                    and abs(n.get("dx", 99)) <= 5 and abs(n.get("dy", 99)) <= 5
                    and kb.talk_status(n.get("name")) == "new"]
        # Don't re-trigger forever on someone we just tried to greet.
        greeted = session.setdefault("greeted", set())
        new_npcs = [n for n in new_npcs if n["name"] not in greeted]
        if new_npcs:
            target = min(new_npcs, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
            greeted.add(target["name"])
            action = {"type": "talk", "name": target["name"]}
            reason = f"(guard) greeting new person '{target['name']}' before moving on"
            print(f"[{step:03d}] greet-guard: talk to new NPC {target['name']}")

    # --- Search reachability guard: if the model wants to SEARCH but no body/
    #     container is within reach (search scans ~4 tiles engine-side), yet a
    #     dead body or a container is VISIBLE further away, walk to it first so
    #     the search will actually hit something (fixes endless "no body nearby"
    #     when the corpse is a few tiles off). ---------------------------------
    if (isinstance(action, dict) and action.get("type") == "search"
            and not state.get("conversation_in_progress")):
        _pp = state.get("player") or {}
        _here = (_pp.get("tx"), _pp.get("ty"))
        # Already looted from this exact spot? Don't re-search - the body is
        # empty. Move on to the next objective instead of looping.
        if _here in session.get("looted_spots", set()):
            session["last_bump"] = ("You already searched and emptied the body here - "
                                    "there is nothing left to take. Move on: pursue your "
                                    "other goals (e.g. find and talk to the person you need).")
            qa, qr = _pursue_focus_quest(state, kb)
            if qa and qa.get("type") != "search":
                action, reason = qa, qr
            else:
                action = _explore_far(state, session, wedged)
                reason = "(guard) body already looted; moving on to explore"
            print(f"[{step:03d}] search-guard: body at {_here} already looted; moving on")
        else:
            _here_close = lambda dx, dy: abs(dx) <= 1 and abs(dy) <= 1
            # Anything searchable right next to us? then let the search run.
            adjacent = any(_here_close(n.get("dx", 9), n.get("dy", 9))
                           for n in (state.get("nearby") or []) if n.get("dead"))
            adjacent = adjacent or any(_here_close(o.get("dx", 9), o.get("dy", 9))
                                       for o in (state.get("objects") or [])
                                       if o.get("body"))
            if not adjacent:
                # Nearest dead body (from NPC list) or body-flagged object.
                cands = [(abs(n.get("dx", 99)) + abs(n.get("dy", 99)),
                          _pp.get("tx", 0) + n.get("dx", 0),
                          _pp.get("ty", 0) + n.get("dy", 0), n.get("name", "body"))
                         for n in (state.get("nearby") or []) if n.get("dead")]
                cands += [(abs(o.get("dx", 99)) + abs(o.get("dy", 99)),
                           _pp.get("tx", 0) + o.get("dx", 0),
                           _pp.get("ty", 0) + o.get("dy", 0), o.get("name", "body"))
                          for o in (state.get("objects") or []) if o.get("body")]
                if cands:
                    cands.sort(key=lambda c: c[0])
                    _, btx, bty, bnm = cands[0]
                    action = {"type": "goto", "tx": btx, "ty": bty}
                    reason = f"(guard) walking to '{bnm}' @({btx},{bty}) before searching"
                    print(f"[{step:03d}] search-guard: goto body '{bnm}' @({btx},{bty})")
                else:
                    # Nothing searchable anywhere (the agent tried to 'search the
                    # garbage' etc). search only opens BODIES/CONTAINERS, so this
                    # would waste the turn. Redirect to look (examine) so the
                    # turn is useful, and tell the agent what search is for.
                    action = {"type": "look"}
                    reason = "(guard) nothing to search here; examining instead"
                    session["last_bump"] = (
                        "'search' only opens a nearby BODY or CONTAINER (chest, "
                        "barrel, bag). There is none within reach, so it does "
                        "nothing on scenery like garbage/tables. To inspect the "
                        "area use 'look'; to grab a loose item use 'pickup'.")
                    print(f"[{step:03d}] search-guard: no searchable target; look instead")

    # --- Repeat-failed-pickup guard: if the agent keeps trying to pick up an
    #     item that has already failed 2+ times (out of reach / owned / not
    #     takeable), stop retrying and move on / walk toward it instead. -------
    if (isinstance(action, dict) and action.get("type") == "pickup"
            and not state.get("conversation_in_progress")):
        _pn = (action.get("name") or "").lower()
        if _pn and session.get("pickup_fails", {}).get(_pn, 0) >= 2:
            # If the item is visible but far, walk to it once; else give up on it.
            _pp = state.get("player") or {}
            match = next((o for o in (state.get("objects") or [])
                          if (o.get("name") or "").lower() == _pn), None)
            if match and (abs(match.get("dx", 9)) > 1 or abs(match.get("dy", 9)) > 1):
                action = {"type": "goto",
                          "tx": _pp.get("tx", 0) + match.get("dx", 0),
                          "ty": _pp.get("ty", 0) + match.get("dy", 0)}
                reason = f"(guard) '{_pn}' pickup kept failing; walking to it first"
            else:
                action = _explore_far(state, session, wedged)
                reason = f"(guard) '{_pn}' cannot be taken; moving on"
            print(f"[{step:03d}] pickup-guard: stop retrying '{_pn}'")

    # --- Wall-aware move guard: never walk into a '#'. ------------------
    # The grid is centered on the avatar (radius 12 -> center [12][12]).
    # A '.' or open door '/' or an item '*' or NPC is walkable; '#' and a
    # closed door '+' are not directly walkable (a closed door needs goto,
    # which opens it). If the model's move heads into a wall, pick the closest
    # walkable direction toward the same heading, else pathfind/turn.
    if isinstance(action, dict) and action.get("type") == "move":
        grid = state.get("grid") or ""
        rows = grid.split("\n")
        if len(rows) >= 25 and len(rows[0]) >= 25:
            cx = cy = 12
            deltas = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0),
                      "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1)}
            def cell(dx, dy):
                x, y = cx + dx, cy + dy
                if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                    return rows[y][x]
                return "#"
            def walkable(ch):
                return ch in ".*&Cx/@"   # open, item, npc, body, open door
            d = action.get("dir", "")
            dxy = deltas.get(d)
            if dxy and not walkable(cell(*dxy)):
                # Requested direction is blocked. Try nearby directions in
                # order of similarity to the intended heading.
                order = {
                    "n": ["n","ne","nw","e","w"], "s": ["s","se","sw","e","w"],
                    "e": ["e","ne","se","n","s"], "w": ["w","nw","sw","n","s"],
                    "ne": ["ne","n","e","nw","se"], "nw": ["nw","n","w","ne","sw"],
                    "se": ["se","s","e","sw","ne"], "sw": ["sw","s","w","se","nw"],
                }.get(d, ["n","e","s","w","ne","nw","se","sw"])
                picked = None
                for cand in order:
                    if walkable(cell(*deltas[cand])):
                        picked = cand
                        break
                if picked:
                    action = {"type": "move", "dir": picked}
                    reason = f"(guard) '{d}' hits a wall; moving {picked} instead"
                else:
                    # Fully boxed in locally -> pathfind toward the heading.
                    p = state.get("player") or {}
                    hx, hy = deltas.get(d, (0, -1))
                    action = {"type": "goto", "tx": p.get("tx", 0) + hx * 12,
                              "ty": p.get("ty", 0) + hy * 12}
                    reason = f"(guard) '{d}' blocked; pathfinding around walls"

    # Record the agent's own reply into the dialogue history (so the model
    # remembers what IT said, not just what NPCs said).
    if isinstance(action, dict) and action.get("type") == "answer":
        idx = action.get("index")
        if isinstance(idx, int) and 0 <= idx < len(answers):
            kb.record_my_reply(answers[idx])
        elif action.get("text"):
            kb.record_my_reply(str(action.get("text")))

    # If the model asks to "goto" a named target that isn't visible but IS a
    # remembered place or NPC, resolve it to coordinates from the mental map.
    if (isinstance(action, dict) and action.get("type") == "goto"
            and action.get("name") and "tx" not in action):
        nm = action["name"]
        nearby_names = {n.get("name") for n in (state.get("nearby") or [])}
        if nm not in nearby_names:
            pos = kb.place_pos(nm) or kb.npc_last_pos(nm)
            if pos and pos[0] is not None:
                action = {"type": "goto", "tx": pos[0], "ty": pos[1]}
                reason = f"{reason} [mapped '{nm}' -> ({pos[0]},{pos[1]})]"
            else:
                # Unresolvable target: it's not visible and not in our mental map
                # or NPC memory. Don't hand the engine a goto it can't route
                # (that makes the agent flail). Tell the model it's unknown and
                # list what IS known so it can pick a real destination or explore.
                _pp = state.get("player") or {}
                known = kb.places_view(_pp.get("tx", 0), _pp.get("ty", 0), limit=8)
                names = ", ".join(k["name"] for k in known) or "(none yet)"
                session["last_bump"] = (
                    f"You tried to goto '{nm}', but that place is not on your map "
                    f"and you can't see it. Known places you CAN goto: {names}. "
                    f"To find a PERSON (like a mayor), enter buildings: go through "
                    f"doors ('+' closed / '/' open) and look inside. Explore to find "
                    f"'{nm}', or annotate it once you reach it.")
                print(f"[{step:03d}] goto unresolved: '{nm}' (known: {names})")
                # If we're hunting a person, prefer entering a nearby building
                # via an unvisited door rather than wandering outdoors.
                door = None
                for d in (state.get("doors") or []):
                    key = (_pp.get("tx", 0) + d.get("dx", 0), _pp.get("ty", 0) + d.get("dy", 0))
                    if key not in session.setdefault("entered_doors", set()):
                        door = (key, d)
                        break
                if door is not None:
                    (dtx, dty), _d = door
                    session["entered_doors"].add((dtx, dty))
                    action = {"type": "goto", "tx": dtx, "ty": dty}
                    reason = f"(guard) '{nm}' unknown; entering a building via door @({dtx},{dty})"
                else:
                    action = _explore_far(state, session, wedged)
                    reason = f"(guard) '{nm}' unknown; exploring to find it"

    # A goto that isn't actually moving the avatar loops forever (the engine
    # sometimes reports ok/stepped for a goto but the avatar never advances,
    # while a plain "move" reliably walks several tiles). So: if we didn't move
    # since last turn, convert this goto into a direct MOVE toward the target -
    # move is the dependable primitive. Also handle a goto to ~our own tile.
    if isinstance(action, dict) and action.get("type") == "goto" and "tx" in action:
        _pp = state.get("player") or {}
        here = (_pp.get("tx"), _pp.get("ty"))
        tgt = (action["tx"], action["ty"])
        # Did we move at all since the previous turn?
        moved = here != session.get("prev_pos_for_goto")
        session["prev_pos_for_goto"] = here
        near_here = abs(tgt[0]-here[0]) + abs(tgt[1]-here[1]) <= 1
        last_was_goto = session.get("last_action_type") == "goto"
        if near_here:
            # Target is basically where we stand - explore instead of no-op.
            action = _explore_far(state, session, wedged)
            reason = "(guard) goto target is here; exploring instead"
        elif last_was_goto and not moved:
            # The previous goto didn't move us. Drive a reliable MOVE toward the
            # target's compass direction, preferring an open grid cell.
            dx = (1 if tgt[0] > here[0] else -1 if tgt[0] < here[0] else 0)
            dy = (1 if tgt[1] > here[1] else -1 if tgt[1] < here[1] else 0)
            rows = (state.get("grid") or "").split("\n")
            cx = cy = 12
            def _cell(ddx, ddy):
                x, y = cx + ddx, cy + ddy
                if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                    return rows[y][x]
                return "#"
            WALK = ".*&C/xno~"
            name_of = {(0,-1):"n",(0,1):"s",(1,0):"e",(-1,0):"w",
                       (1,-1):"ne",(1,1):"se",(-1,1):"sw",(-1,-1):"nw"}
            # Try the intended diagonal/cardinal toward target, then fall back
            # to any open neighbor rotating by a stall counter.
            n = session.get("goto_stall", 0) + 1
            session["goto_stall"] = n
            cand = [(dx,dy),(dx,0),(0,dy)]
            pick = next((c for c in cand if c != (0,0) and _cell(*c) in WALK), None)
            if pick is None:
                alld = [(0,-1),(1,0),(0,1),(-1,0),(1,-1),(1,1),(-1,1),(-1,-1)]
                opens = [c for c in alld if _cell(*c) in WALK]
                pick = opens[n % len(opens)] if opens else None
            if pick:
                action = {"type": "move", "dir": name_of[pick], "speed": 180}
                reason = f"(guard) goto not moving; stepping {name_of[pick]} toward target"
            else:
                action = _explore_far(state, session, True)
                reason = "(guard) goto stuck; no open step, exploring"
        else:
            session["goto_stall"] = 0

    # "talk" is a top-level command, not an act() action.
    if isinstance(action, dict) and action.get("type") == "talk":
        tname = action.get("name", "")
        if tname:
            kb.mark_talked(tname)
            session["current_npc"] = tname
        result = exult.talk(tname)
    else:
        # If answering a dialogue choice, record which TOPIC branch we took for
        # the current NPC (builds the per-NPC dialogue tree of asked topics).
        if (isinstance(action, dict) and action.get("type") == "answer"
                and isinstance(action.get("index"), int)):
            _ans = state.get("answers") or []
            _i = action["index"]
            if 0 <= _i < len(_ans):
                kb.record_npc_choice_taken(session.get("current_npc", "?"), _ans[_i])
                # Remember the subject we just asked about, so the NPC's reply
                # next turn gets filed under this topic in the shared topic KB.
                session["current_topic"] = _ans[_i]
        result = exult.act(action)
    # Tool-call stats: count the action type and its outcome (ok/err).
    _atype = action.get("type") if isinstance(action, dict) else "?"
    _ok = result.get("ok") if isinstance(result, dict) else None
    kb.record_tool(_atype, _ok)
    session["last_action_type"] = _atype
    # Track FAILED pickups so we don't retry the same un-takeable item over and
    # over (tongs x4 etc). Key by item name; after a couple of failures, tell
    # the agent to stop trying that item.
    if _atype == "pickup" and isinstance(result, dict):
        _pname = (action.get("name") or "").lower()
        fails = session.setdefault("pickup_fails", {})
        if result.get("ok") is False and _pname:
            fails[_pname] = fails.get(_pname, 0) + 1
            session["last_bump"] = (
                f"You could not pick up '{action.get('name')}' ({result.get('error','no reason')}). "
                f"It may be OUT OF REACH (walk adjacent first), OWNED (do not steal), "
                f"or not takeable. Do NOT keep retrying it - move on or approach it first.")
        elif result.get("ok") and _pname:
            fails.pop(_pname, None)
    if window.available:
        window.set_action(json.dumps(action) + "\n\n-> " + json.dumps(result))

    # Explicit "you bumped into something" feedback: if a move reported it was
    # blocked, tell the model next turn and note it (so it tries another way).
    if (isinstance(action, dict) and action.get("type") == "move"
            and isinstance(result, dict) and result.get("blocked")):
        d = action.get("dir", "")
        session["last_bump"] = f"Your last move {d} was BLOCKED - something (a wall/obstacle) is that way. Try a different direction or use goto to route around it."
        kb.record_action(f"bumped a wall moving {d}")
    else:
        session.pop("last_bump", None)

    # Log meaningful actions (not routine moves/waits) to the short action
    # history so the agent can avoid repeating itself.
    atype = action.get("type") if isinstance(action, dict) else None
    if atype in ("talk", "open", "pickup", "search", "goto", "combat", "feed"):
        p0 = state.get("player") or {}
        detail = action.get("name") or action.get("dir") or ""
        kb.record_action(f"{atype} {detail}".strip()
                         + f" @({p0.get('tx')},{p0.get('ty')})")
    # A successful pickup/search changes the world -> NPCs may now have new
    # dialogue, so allow revisiting them.
    if atype in ("pickup", "search") and isinstance(result, dict) and result.get("ok"):
        kb.reset_talk_gate()
    # A searched body/container OR a spot where we picked something up is a
    # notable location -> auto-mark it so the agent can find its way back even
    # if it never annotates on its own (e.g. returning to a crime scene).
    if atype in ("search", "pickup") and isinstance(result, dict) and result.get("ok"):
        pp = state.get("player") or {}
        tgt = result.get("target") or result.get("item") or action.get("name") or "spot"
        verb = "searched" if atype == "search" else "found items at"
        kb.record_place(f"where I {verb} {tgt}", pp.get("tx", 0), pp.get("ty", 0),
                        kind="marked")
        # Auto-resolve 'investigate/search/find <subject>' quests now that we've
        # actually examined/collected that subject.
        for _t in kb.auto_resolve_examine_quests(str(tgt)):
            print(f"[{step:03d}] auto-resolved quest (examined {tgt}): {_t}")

    p = state.get("player") or {}
    print(f"[{step:03d}] pos=({p.get('tx')},{p.get('ty')}) "
          f"conv={state.get('conversation_active')} "
          f"reason={reason!r} action={action} -> {result}"
          + (f"  [{session.get('last_ctx')}]" if session.get('last_ctx') else ""))
    time.sleep(args.delay)


def run_loop(args, window: ThoughtsWindow, ollama, exult_proc=None) -> None:
    recent_positions: list = []
    session = {"picked": set(), "searched_body": False}
    kb = KnowledgeBase.load(args.memory_file) if args.memory_file else KnowledgeBase()
    if args.memory_file:
        print(f"[+] Loaded journal: {len(kb.quests)} quests, "
              f"{len(kb.npcs)} NPCs, {len(kb.journal)} notes")
    exult = ExultClient(args.host, args.port)
    saved_this_run = False
    try:
        exult.connect()
        print(f"[+] Connected to Exult bridge: {exult.ping()}")
        for step in range(args.steps):
            try:
                _do_turn(args, window, ollama, exult, step, recent_positions, kb, session)
                if args.memory_file and step % 5 == 0:
                    kb.save(args.memory_file)
                # Periodically save the GAME so progress survives a crash/close.
                if step > 0 and step % args.save_every == 0:
                    try:
                        r = exult.act({"type": "save"})
                        saved_this_run = True
                        print(f"[{step:03d}] game saved -> {r}")
                    except Exception as e:
                        print(f"[{step:03d}] game save failed: {e}")
            except (ConnectionError, OSError) as e:
                print(f"[{step:03d}] connection issue: {e}; reconnecting...")
                try:
                    exult.close()
                    exult.connect()
                except Exception as e2:
                    print(f"[{step:03d}] reconnect failed: {e2}")
                    time.sleep(args.delay)
            except Exception as e:  # any other per-turn error: log and continue
                print(f"[{step:03d}] turn error: {type(e).__name__}: {e}")
                time.sleep(args.delay)
    finally:
        if args.memory_file:
            kb.save(args.memory_file)
        # Save the GAME before disconnecting. This is essential when we (the
        # driver) launched Exult and will terminate it on exit, so in-game
        # progress is not lost.
        try:
            r = exult.act({"type": "save"})
            print(f"[+] Final game save -> {r}")
        except Exception as e:
            print(f"[!] Final game save failed: {e}")
        exult.close()
        if window.available:
            window.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Drive Exult with an Ollama LLM.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=45999)
    ap.add_argument("--model", default="gemma4:latest")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between turns")
    ap.add_argument("--dry-run", action="store_true", help="skip Ollama; scripted moves")
    ap.add_argument("--show-thoughts", action="store_true", help="open the LLM thinking window")
    ap.add_argument("--raw-log", action="store_true",
                    help="append the exact prompt+reply for EVERY turn to raw_comms.log "
                         "(parse failures are always logged regardless)")
    ap.add_argument("--auto-feed", action="store_true",
                    help="periodically restore food so the party can't starve")
    ap.add_argument("--auto-feed-every", type=int, default=20,
                    help="feed every N turns when --auto-feed is set")
    ap.add_argument("--exult-exe", default=None,
                    help="path to Exult.exe (auto-detected at repo root if omitted)")
    ap.add_argument("--no-launch", action="store_true",
                    help="do not auto-launch Exult; require it to be running")
    ap.add_argument("--memory-file", default="agent_memory.json",
                    help="persist the agent's journal/known-NPCs here across "
                         "driver restarts (set to '' to disable)")
    ap.add_argument("--save-every", type=int, default=40,
                    help="save the in-game progress every N turns")
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="Ollama context window (tokens). Must exceed the prompt "
                         "size or the prompt is silently truncated (Ollama default "
                         "is only 2048).")
    args = ap.parse_args()

    ollama = None
    if not args.dry_run:
        ollama = OllamaClient(model=args.model, host=args.ollama_host,
                              num_ctx=args.num_ctx)
        if not ollama.is_up():
            print(
                f"[!] Ollama not reachable at {args.ollama_host}. "
                f"Start it (ollama serve) or use --dry-run.",
                file=sys.stderr,
            )
            return 2
        maxctx = ollama.context_size()
        print(f"[+] Model '{args.model}': max context {maxctx or '?'} tokens; "
              f"using num_ctx={args.num_ctx} (full prompt must fit here or it is "
              f"silently truncated).")

    # Make sure the game is running (launch it if needed).
    try:
        exult_proc = ensure_exult_running(args)
    except Exception as e:
        print(f"[!] {e}", file=sys.stderr)
        return 3

    window = ThoughtsWindow()
    if args.show_thoughts:
        window.start()

    try:
        if window.available and args.show_thoughts:
            # Tkinter must own the main thread; run the game loop in a worker.
            worker = threading.Thread(target=run_loop, args=(args, window, ollama), daemon=True)
            worker.start()
            window.mainloop()
        else:
            run_loop(args, window, ollama)
    finally:
        # Only shut down Exult if WE launched it. run_loop already issued a
        # final in-game save before disconnecting; give it a moment to flush.
        if exult_proc is not None:
            print("[+] Shutting down the Exult instance we launched (progress saved)...")
            time.sleep(2.0)
            try:
                exult_proc.terminate()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
