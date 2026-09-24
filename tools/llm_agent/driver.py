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
    Ask them their name, job, and EVERY topic offered. Asking one topic often
    reveals new topics, so keep asking until there are no new ones left - a
    single overlooked topic can hold the key clue.
  * GREET NEW PEOPLE: whenever you encounter someone you have NOT talked to yet
    (status "new" in nearby), talk to them before moving on - even while you are
    pursuing another goal. Every new person may give you a quest, a clue, an
    item, or offer to JOIN YOUR PARTY (companions are extremely valuable). Do
    not walk past unmet people to chase a single objective; a hint to find a
    specific person is NOT a reason to ignore everyone else you pass.
  * FOLLOW LEADS: when someone mentions a person, place, item, or event, treat
    it as a lead worth pursuing. Use your journal to remember what you learned.

RPG PLAYER WISDOM (genre habits a seasoned player relies on):
  * EXHAUST DIALOGUE: work through the WHOLE conversation tree with each person -
    ask every available topic (especially names, jobs, and any proper noun).
    Clues are often buried in a topic you might skip.
  * EXPLORE EVERYWHERE: check every DOOR and enter every BUILDING. Open doors
    ('+' closed / '/' open) are passages, not walls. Rooms hold people, loot,
    and clues you cannot see from outside.
  * OPEN AND SEARCH: open every container (chest, barrel, bag, crate) and search
    bodies. Try devices - levers, switches, buttons - they reveal secrets/paths.
  * GATHER USEFUL THINGS: pick up gold and gems (money), food (you must eat),
    weapons, armour, keys, potions, scrolls, reagents, and tools. Your pack
    holds a lot - when unsure, take it.
  * KEEP ODD ITEMS: an item that seems useless to you may be needed later for a
    QUEST or puzzle (a specific key, a token, a letter, a trinket). Hold onto
    unusual items rather than ignoring them.
  * STEALING HAS CONSEQUENCES: items with an "owned" flag are someone's
    property. Taking them is theft - if witnessed it angers people and can
    summon guards who may attack and kill you. Unowned/abandoned/given items and
    loot from defeated enemies or the dead are free. Weigh the risk: steal only
    if it is worth the danger and you can avoid being caught.
  * FIGHT TO GROW: defeating monsters/enemies earns experience that raises your
    stats and LEVEL over time, making you stronger. Engage winnable fights (use
    "combat"); flee ones that would kill you. (XP comes from solving quests AND
    slaying monsters; leveling raises your attributes and Hits. Strength = carry
    capacity + melee damage + Hits; Dexterity = combat hit-chance + speed +
    lockpicking; Intelligence = magic skill + max mana.)
  * MAKE PROGRESS: prefer purposeful action over aimless wandering or repeating
    yourself. If you have exhausted a person or place, move on to somewhere new.
  * DON'T LINGER: do not re-interview people you've already learned from
    (especially your own party companions - they have nothing new). When you
    have gathered the local leads, LEAVE the area: travel to the town gate/edge
    and out to new regions to advance the story. The world is far bigger than
    one town - staying put stalls the whole adventure.
  * SURVIVE: keep fed and stay alive; avoid needless danger. Hunger, poison, and
    damage all reduce your Hits; at 0 Hits you fall unconscious. "feed" when
    food is low; rest/heal when hurt.
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

DECLARE YOUR FOCUS: add "set_current_quest": "<quest title>" whenever you START
or SWITCH the quest you are actively working on. This is shown back to you as
"current_quest" and tagged onto your action_log so you can SEE, over time,
whether the quest you're on is actually making progress or looping. If your
action_log shows many turns on the same current_quest with no result, SWITCH to
a different quest.

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

You should also MAINTAIN A RUNNING PLOT SUMMARY: include an optional
"plot_summary" field (<=1000 chars) in your reply to record the overall story so
far - who and what matter, key clues, and your current objective. It replaces
the previous summary and is ALWAYS kept in your context as "story_so_far", so
update it whenever something important happens (you'll also be reminded
periodically). For finer detail you have the recall/quests tools.
  {"action": {"type": "wait"}, "plot_summary": "Trinsic: blacksmith Christopher
   murdered; a man+wingless gargoyle fled to the dock. I have his chest key.
   Need: report to Mayor Finnigan for the gate password to leave.",
   "reason": "record progress"}

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
  turn (int)                  - the current turn number (increments each action).
                                Time is passing - use it to notice when you have
                                spent many turns on one thing without progress.
  turns_since_progress (int)  - turns since you last made REAL progress (resolved
                                a quest or met a new person). If this grows large,
                                what you're doing ISN'T working - change strategy.
  player: {tx,ty (your absolute tile on the world map), elevation, hp, max_hp, hp_pct, food}
                                ELEVATION: 0 = GROUND level. >0 = UP (on a wall
                                walkway / upper floor / rooftop). <0 = UNDERGROUND
                                (cave/cellar/dungeon). The "level" field states
                                this in words. The world is 3D: people & items on
                                a DIFFERENT level than you are NOT reachable until
                                you change levels (climb stairs up, or descend).
                                If you are at elevation 0 you are ALREADY at
                                ground - do not try to "descend to ground".
                                HEALTH: hp is CURRENT, max_hp is your MAXIMUM
                                (== your strength). hp == max_hp means FULL
                                health - you do NOT need healing. Only seek a
                                healer/rest when hp is well below max_hp (hp_pct
                                low). food is hunger (eat when it gets low).
                                COORDINATES: everything uses ONE consistent frame.
                                Your tile is (tx,ty). Every object/person gives a
                                RELATIVE offset {dx,dy} from you, so its absolute
                                tile is (tx+dx, ty+dy). The ASCII map is centered
                                on you (@ = your tile); moving right/east is +dx,
                                down/south is +dy. To walk somewhere, "goto" an
                                absolute (tx,ty) - e.g. an object at dx=5,dy=-3
                                means goto (tx+5, ty-3). You do NOT need the map's
                                coordinate range; @ is always your anchor.
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
                                LOOTABLE body/container - "search" next to it
                                empties its contents into your pack. BUT
                                "corpse_not_lootable":true means a corpse with
                                NOTHING to take - do NOT search it, it wastes
                                turns; examine ("look") it instead if curious.
                                "town_exit":true marks a town
                                gate/portcullis - the way OUT of town to the
                                wider world (goto it to leave, once any gate
                                password/lock is dealt with).
  grid (string)               - top-down ASCII map centered on you (@). Its job
                                is WALKABILITY and ROUTES; item identity is in
                                the objects list. Glyphs:
                                  @ you   C companion   & person
                                  b lootable body/container-corpse (search it)
                                  x corpse (nothing to take)   n container
                                  * loose item (pickup)   E exit/route (gate,
                                    stairs, ladder - the way through/out).
                                    STAIRS ARE DIRECTIONAL: you climb them only
                                    from the BOTTOM STEP, not the side. If a step
                                    onto stairs is "blocked", walk AROUND to line
                                    up with the bottom of the stairs, then MOVE
                                    onto them (goto won't land on a stairs tile).
                                  ~ water   = fence/barrier (find a gap or gate)
                                  + closed door (goto opens it)   / open door
                                  . walkable ground   # blocked
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
  action_log: YOUR TEMPORAL MEMORY - a turn-by-turn log of your recent actions
      with OUTCOMES, one per line as "[T<turn>] <action> -> <result>", and
      ">>> now working: <quest>" markers where you switched quests. READ THIS to
      see what you've been doing OVER TIME: if you see the same action/goal
      repeated across many turns with no useful result, you are in a FRUITLESS
      LOOP - stop and do something different.
  current_quest: the quest you told me you are working on (via set_current_quest).
      Shown so you stay focused; if it's stalling across many log lines, switch.
  already_searched_empty (list) - bodies/containers you ALREADY searched and
      found empty. Do NOT return to search these again - move on.
  story_so_far: YOUR running plot summary (you maintain it via "plot_summary").
      Always in context - the big picture of the story, key clues, and current
      objective. Keep it updated; use recall/quests tools for finer detail.
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
            {"tx":<int>,"ty":<int>} for an absolute tile. You may also add
            {"tz":<int>} to target a specific ELEVATION (0=ground, >0=up a
            wall/floor, <0=underground); omit tz to target your current level.
            To reach a person/item on a DIFFERENT level, goto with their tz.
            PREFER "goto" over many "move" steps to reach an NPC, item, building,
            or known place. If the exact spot can't be reached (walls), goto
            still walks you as far toward it as it can (result "partial":true) -
            so repeating goto to a far target makes steady progress; you do NOT
            need a clear line.
  stop    - Stop walking. params: none.
  talk    - START a conversation with a nearby NPC. params: {"name": "<NPC name>"}
            This is the ONLY way to begin dialog. Walking next to an NPC does
            NOT start dialog. You do NOT need to be adjacent - it finds the
            named NPC in your view and opens the conversation. But PREFER to be
            CLOSE to the NPC first (goto them, within a few tiles) - it's more
            reliable and natural, though not strictly required. The NPC must be
            AWAKE (a "sleeping" condition NPC won't respond - wait_until morning).
            Works for townspeople; your own party is in "party" (nothing new).
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
  read    - Read a nearby SIGN or readable object (a human double-clicks it).
            Returns its "text". params: omit to read the nearest sign, or
            {"name":"<obj>"} to read a specific object. Signs give shop names,
            directions, and place names - useful when you goto a sign.
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
  unequip - Take a worn/wielded item OFF and put it back in your pack.
            params: {"name":"<item>"}.
  drop    - Drop a carried or worn item on the ground at your feet (e.g. to get
            rid of junk or free up space). params: {"name":"<item>"}.
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
            before deciding what to do. You can "recall" a character AT ANY TIME
            to see their past transcript and which topics you've already asked
            vs not - handy before (re)talking to them so you don't waste turns
            re-asking covered topics. (This is a guide, not a rule: NPCs may
            offer NEW topics as quests progress, so re-visiting someone can still
            be worthwhile - use your judgement.)
  quests  - Review your FULL quest log with notes and prerequisites. params:
            none. The always-on "quests" field is a COMPACT list (titles only);
            use this tool when planning to see each quest's notes/details.
            Result appears next turn as "quest_detail".
  answer  - Choose a reply during a conversation. params: {"index": <int>} (0-based
            into the "answers" list) OR {"text": "<answer text>"}.
            Only valid when conversation_active is true.
  set_number - Answer a numeric slider prompt. params: {"value": <int>}. Only
            valid when number_prompt is true; value is clamped to
            [number_min, number_max]. Use this to pick a quantity/amount.
  continue - Advance an NPC's speech to the next page when there is npc_text
            showing but no answer choices yet (conversation_in_progress but not
            conversation_active). params: none. This is how you read through a
            character's multi-page dialogue until choices appear.
  dismiss - Close/cancel a menu, sign, or popup. params: none.
  combat  - Toggle combat/attack mode on or off. params: none.
  set_combat_mode - Set how you and your party fight in combat. params:
            {"mode": one of "nearest"|"weakest"|"strongest"|"berserk"|"defend"|
            "flank"|"flee"|"protect"|"random"|"manual"}. Guide: "attack weakest"
            to finish off wounded foes, "defend" (dodge more) when hurt, "flee"
            to retreat from a losing fight, "berserk" to never retreat.
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
  1. If conversation_active is true -> use "answer". Be THOROUGH like a good
     detective: ask EVERY topic in "not_yet_asked_this_npc" before leaving -
     each one may reveal a lead, a name, a clue, or a new topic. Always ask a
     person's "name" and "job", and especially any proper noun (a person, place,
     group, or event - e.g. "Inamo", "stables", "Fellowship"). Asking a topic
     often UNLOCKS new topics, so keep going until "not_yet_asked_this_npc" is
     empty. Do NOT leave a conversation early with useful topics unasked. Avoid
     re-picking anything in "already_asked_this_npc". Only choose "bye"/"leave"
     once there are no useful unasked topics left.
  2. Else if conversation_in_progress is true (a conversation is open but no
     choices yet) -> use "continue" to advance the NPC's text until the answer
     choices appear. Do NOT "talk" again or "move" during a conversation.
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

    `squeeze` is a context-pressure LEVEL that dynamically sizes the short-term
    (rolling) memory windows to USE the available context:
      -2 = huge room  .. 0 = default .. +3 = very tight.
    Negative levels EXPAND the windows (more recent dialogue/actions/objects
    kept in view) when the prompt is well under num_ctx; positive levels shrink
    them as the prompt approaches the limit. The durable structured memory
    (quests, places, hints, episodic summary) is always kept; only these verbose
    recent-context tiers grow/shrink. This makes short-term memory as large as
    the context budget comfortably allows."""
    # Window sizes per level (dialogue, actions, objects, nearby, places).
    # Index 0 in this list is the MOST expanded; the default (level 0) maps to
    # _EXPAND_BASE below so negative levels index earlier (bigger) rows.
    _TIERS = [
        (80, 24, 30, 14, 24),  # -2: huge room (big short-term memory)
        (50, 16, 20, 12, 18),  # -1: roomy+
        (30, 10, 14, 8, 12),   #  0: default
        (18, 8, 10, 8, 10),    # +1: trim
        (10, 6, 8, 6, 8),      # +2: tight
        (6, 4, 6, 5, 6),       # +3: very tight
    ]
    _EXPAND_BASE = 2  # list index that corresponds to squeeze level 0
    idx = max(0, min(_EXPAND_BASE + int(squeeze), len(_TIERS) - 1))
    dlg_n, act_n, obj_n, near_n, place_n = _TIERS[idx]
    lvl = max(0, int(squeeze))  # only positive levels trim durable-view caps
    p = state.get("player") or {}
    # Transient-zero guard: the engine occasionally reports hp:0 for a frame
    # (e.g. right after a save loads or during a schedule transition) even when
    # the avatar is alive and full. A lone hp:0 with dead:false made the model
    # panic and hunt a healer. If hp reads 0 but we're NOT dead, treat it as the
    # last-known-good hp (or max_hp) so the model isn't misled by a glitch.
    _hpz = p.get("hp")
    if _hpz == 0 and not p.get("dead"):
        _lastgood = summarize_state._last_good_hp if hasattr(summarize_state, "_last_good_hp") else None
        p = dict(p)
        p["hp"] = _lastgood if _lastgood else (p.get("max_hp") or 1)
    elif isinstance(_hpz, int) and _hpz > 0:
        summarize_state._last_good_hp = _hpz
    nearby = state.get("nearby") or []
    objects = state.get("objects") or []
    # Only surface dialog fields when a conversation is actually open, so the
    # model isn't misled by stale npc_text into pressing space forever.
    in_convo = bool(state.get("conversation_in_progress"))
    view = {
        "turn": state.get("turn"),
        "turns_since_progress": state.get("turns_since_progress"),
        "player": {
            "tx": p.get("tx"), "ty": p.get("ty"),
            "elevation": p.get("tz", 0),
            "level": (
                "GROUND LEVEL (elevation 0 - you are NOT up high and NOT "
                "underground; do NOT try to climb down or up to 'reach ground', "
                "you are already at ground level)"
                if (p.get("tz", 0) or 0) == 0
                else (f"UP HIGH at elevation {p.get('tz')} (on a wall / upper "
                      "floor / rooftop). To get DOWN, 'goto' a ground tile with "
                      "tz:0 (e.g. a place you know) - the engine walks you down "
                      "the ramp. Ground-level people & items are only reachable "
                      "once you are back down."
                      if (p.get("tz", 0) or 0) > 0
                      else f"UNDERGROUND at elevation {p.get('tz')} (in a "
                           "cave/cellar/dungeon below ground - go UP to return "
                           "to the surface)")),
            "hp": p.get("hp"), "max_hp": p.get("max_hp"),
            "hp_pct": (round(100 * p.get("hp", 0) / p["max_hp"])
                       if p.get("max_hp") else None),
            "dead": p.get("dead"),
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
             **({"condition": n["condition"]} if n.get("condition") else {}),
             **({"different_level": True} if n.get("same_level") is False else {})}
            for n in nearby[:near_n] if not n.get("in_party")
        ],
        # Party companions are shown separately - they follow you and have no
        # new information, so do NOT "talk" to them to investigate.
        "party": [n.get("name") for n in nearby if n.get("in_party") and n.get("name")],
        "objects": [
            {"name": o.get("name"), "dx": o.get("dx"), "dy": o.get("dy"),
             **({"body": True} if o.get("body") else {}),
             **({"corpse_not_lootable": True} if o.get("corpse") else {}),
             **({"owned": True} if o.get("owned") else {}),
             **({"town_exit": True} if o.get("town_exit") else {}),
             **({"different_level": True} if o.get("same_level") is False else {})}
            for o in objects[:obj_n]
        ],
        "grid_legend": state.get("grid_legend"),
        "grid": state.get("grid"),
    }
    # Advisory: how many recent turns pursued the SAME goal. Surfacing this lets
    # the model NOTICE a cycle and change tack on its own (no steering).
    _streak = state.get("same_goal_streak")
    if _streak and _streak >= 3:
        view["same_goal_streak"] = _streak
        view["progress_note"] = (
            f"You have pursued the same goal ~{_streak} turns running. If it is "
            "not producing progress, STOP repeating it: review your quests/notes "
            "(quests/recall tools), question your assumptions (is this goal even "
            "real?), and try a different lead.")
    if state.get("last_move_failed"):
        view["MOVE_FAILED"] = (
            "Your LAST move did NOT change your position - it was BLOCKED (a wall, "
            "water, a barrier, or stairs from the wrong side). Do not just repeat "
            "the same move. Try a DIFFERENT direction, go around, or pick another "
            "route/target.")
    _tsp = state.get("turns_since_progress") or 0
    if _tsp >= 15:
        view["NO_PROGRESS_WARNING"] = (
            f"You have made NO real progress (no quest resolved, no new person "
            f"met) for {_tsp} turns. Whatever you are doing is NOT working. STOP "
            "and change strategy completely: go somewhere you have NOT been, talk "
            "to someone NEW, or pick a different quest. Do not keep repeating the "
            "same attempt.")
    if state.get("stuck_in_place"):
        view["STUCK_WARNING"] = (
            "You have NOT MOVED for ~10 turns - you are re-trying variations of "
            "the same thing in one spot. This is a dead end. Do something "
            "DIFFERENT now: walk AWAY to a new area (goto a distant known place "
            "or explore), or talk to a NEW person. Whatever you keep trying here "
            "is NOT working - abandon it.")
    if (p.get("tz", 0) or 0) >= 1:
        view["ELEVATION_NOTE"] = (
            f"You are UP on an elevated surface (a wall walkway / upper floor, "
            f"elevation {p.get('tz')}). Most people and items are at GROUND level "
            "and are NOT reachable from up here. If you're looking for someone, "
            "come back DOWN (walk back to the stairs/ramp and descend) unless you "
            "specifically need something up here.")
    elif (p.get("tz", 0) or 0) <= -1:
        view["ELEVATION_NOTE"] = (
            f"You are UNDERGROUND (elevation {p.get('tz')}, in a cave/cellar/"
            "dungeon). To return to the surface, find stairs/a ladder and go UP.")
    doors = state.get("doors") or []
    if doors:
        view["doors"] = [
            {"name": d.get("name"), "dx": d.get("dx"), "dy": d.get("dy"),
             "closed": d.get("closed")}
            for d in doors[:6]
        ]
    if kb is not None:
        # FULL quest log always in context (with notes + prereqs), not just
        # titles: it's the stateless agent's to-do list / plan and drives every
        # decision - it shouldn't have to remember to call the quests tool to
        # see its own plan. Bounded (top open quests + notes) and we have ample
        # context headroom. NPC/topic transcripts stay on-demand via recall.
        view["quests"] = kb.quest_view(max_open=(10 if lvl >= 2 else 18))
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
        # Spots already searched and empty - stated IMPERATIVELY so the agent
        # stops returning to the same looted body/container/bag.
        se = kb.searched_empty_view(10)
        if se:
            view["already_searched_empty_DO_NOT_RETURN"] = se
            view["already_searched_note"] = (
                f"You have already searched {len(se)} spot(s) and found them EMPTY "
                "(listed above). Searching or gotoing them again wastes turns - the "
                "item is NOT there. Pursue a DIFFERENT lead.")
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
        notes = kb.npc_view(nearby_names, note_limit=(2 if lvl >= 2 else 4))
        if notes:
            view["npc_notes"] = notes
        # Growing window of recent CONVERSATION (story/clues live here) and a
        # short window of recent ACTIONS (to avoid repeating yourself). These
        # shrink under context pressure; what scrolls off is folded into
        # story_so_far by the driver's budget loop, so clues are not lost.
        dh = kb.dialogue_view(dlg_n)
        if dh:
            view["recent_dialogue"] = dh
        ah = kb.action_view(300)   # long temporal memory: we have context to spare
        if ah:
            view["action_log"] = ah
        if kb.current_quest:
            view["current_quest"] = kb.current_quest
    if last_look:
        view["look_description"] = last_look
    if alert:
        view["alert"] = alert
    if state.get("recalled") is not None:
        # The full saved dialogue tree the agent asked to recall this turn.
        view["recalled"] = state["recalled"]
    if state.get("quest_detail") is not None:
        # Full quest log the agent asked to review this turn (notes/prereqs).
        view["quest_detail"] = state["quest_detail"]
    # Pretty-print (indent=2) so lists like action_log render ONE ITEM PER LINE
    # and the whole state is human/LLM-readable, not a run-on blob. We have
    # context headroom (typically ~30-40%), so the extra whitespace is worth the
    # clarity - especially for the temporal action log.
    return json.dumps(view, indent=2)

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
    # Normalize the clearer 'press_key' alias to the internal canonical 'key'
    # so all existing type=="key" logic keeps working. (The prompt now presents
    # it as press_key to avoid confusion with a game key-ITEM.)
    if action.get("type") == "press_key":
        action["type"] = "key"
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
        return ("NPC talking; advancing text.", {"type": "continue"})
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
    known_npcs = list(kb.npcs.keys())
    for focus in qv.get("open", []):
        if not focus:
            continue
        npc = focus.get("npc")
        # If the quest has no explicit npc, try to infer one from its title by
        # matching a known NPC name (e.g. "Speak to Mayor Finnigan" -> Finnigan).
        # gemma often names the NPC in the title but not the npc field.
        if not npc:
            title = (focus.get("title") or "")
            for kn in known_npcs:
                if kn and kn.lower() in title.lower() and kn.lower() not in (
                        "dog", "cat", "horse", "sheep", "fox"):
                    npc = kn
                    break
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


def _grid_center(rows):
    """Center (cx, cy) of an @-centered grid, derived from its actual size so it
    works regardless of the rx/ry the engine chose for the visible window."""
    cy = len(rows) // 2 if rows else 12
    cx = (len(rows[0]) // 2) if rows and rows[0] else 12
    return cx, cy


def _explore_far(state: dict, session: dict, wedged: bool) -> dict:
    """Pick a distant goto target in a direction that is actually OPEN on the
    grid (avoid heading into ocean/walls). When wedged, rotate to a brand-new
    direction each turn until we break free."""
    p = state.get("player") or {}
    tx, ty = p.get("tx", 0), p.get("ty", 0)
    rows = (state.get("grid") or "").split("\n")
    cx, cy = _grid_center(rows)

    def openness(dx, dy):
        score = 0
        for r in range(1, 10):
            x, y = cx + dx * r, cy + dy * r
            if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                ch = rows[y][x]
                if ch in ".&C*xb/nE+":
                    score += 1
                elif ch in "#~=":
                    break
        return score

    dirs = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0),
            "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1)}
    tried = session.setdefault("failed_dirs", set())
    # Anti-oscillation: remember the last few explore targets and avoid picking
    # a direction that lands ~back where we just were (this caused a two-tile
    # A<->B ping-pong when the guard re-explored every turn).
    recent_targets = session.setdefault("explore_recent", [])
    ranked = sorted(dirs.items(), key=lambda kv: -openness(*kv[1]))

    def _lands_on_recent(dx, dy):
        t = (tx + dx * 6, ty + dy * 6)
        return any(abs(t[0] - rx) + abs(t[1] - ry) <= 3 for (rx, ry) in recent_targets)

    choice = None
    for name, (dx, dy) in ranked:
        if wedged and name in tried:
            continue
        if _lands_on_recent(dx, dy):
            continue
        if openness(dx, dy) >= 3:
            choice = (name, dx, dy)
            break
    # Relax the recent-target filter if everything was filtered out.
    if not choice:
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
    target = {"type": "goto", "tx": tx + dx * 6, "ty": ty + dy * 6}
    recent_targets.append((target["tx"], target["ty"]))
    if len(recent_targets) > 5:
        del recent_targets[0]
    # Shorter hop (6 tiles): the engine A* has a bounded search budget, so far
    # targets often fail; a nearer target in the most-open direction routes
    # reliably, and the engine now single-steps toward it if A* still gives up.
    return target


def _do_turn(args, window, ollama, exult, step, recent_positions, kb, session) -> None:
    state = exult.observe()
    kb.current_turn = step   # so record_action tags each entry with the game turn
    if window.available:
        window.update_turn(step)
        window.set_map(state.get("grid") or "(no map)")
        window.set_dialog(format_dialog(state))
        # Inspector panels: quests, NPC knowledge, and stats.
        try:
            nearby_names = [n.get("name") for n in (state.get("nearby") or []) if n.get("name")]
            window.set_quests(kb.quests_pretty())
            window.set_resolved_quests(kb.resolved_quests_list())
            window.set_npc_tree(kb.npcs_tree_data())
            window.set_topics_tree(kb.topics_tree_data())
            window.set_plot(kb.episodic_summary or "(no plot summary yet - the LLM builds this)")
            p = state.get("player") or {}
            # Always-visible game status line.
            window.set_gstatus(
                f"{state.get('time_of_day','?')} (h{state.get('hour','?')})  "
                f"pos({p.get('tx')},{p.get('ty')})  hp {p.get('hp')}/{p.get('max_hp')}  food {p.get('food')}  "
                f"str {p.get('str')} dex {p.get('dex')} int {p.get('int')}  "
                f"{'IN COMBAT' if state.get('in_combat') else ''}")
            # Structured stats: labelled key/value pairs for distinct boxes.
            _resolved = sum(1 for q in kb.quests.values() if q.get("status") == "done")
            _ts = kb.tool_stats or {}
            stats_kv = {
                "Position (abs x,y,z)": f"({p.get('tx')},{p.get('ty')},{p.get('tz',0)})",
                "HP": f"{p.get('hp')}/{p.get('max_hp')}",
                "Str/Dex/Int": f"{p.get('str')}/{p.get('dex')}/{p.get('int')}",
                "Food": p.get("food"),
                "NPCs met": len(kb.npcs),
                "Places": len(kb.places),
                "Topics": len(kb.topics),
                "Quests": len(kb.quests),
                "Quests resolved": _resolved,
                "Dialogue mem": len(kb.dialogue_history),
                "Actions mem": len(kb.action_history),
                "Hints": len(kb.hints),
                "Observations": len(kb.observations),
                "Searched empty": len(kb.searched_empty),
                "Plot summary (chars)": len(kb.episodic_summary),
                "Turns": _ts.get("turns", 0),
                "Parse fails": _ts.get("parse_fail", 0),
            }
            window.set_stats_kv(stats_kv)
            window.set_tool_stats(kb.tool_stats_data(top=20))
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
        # NAME REVEAL: NPCs are shown by a generic role ("shopkeeper", "peasant",
        # "guard") until they tell you their name. When a line reveals "My name
        # is X", merge the generic-role record into the real name so we don't
        # split one NPC across two records (e.g. shopkeeper -> Apollonia).
        import re as _re
        _mn = _re.search(r"[Mm]y name is ([A-Z][A-Za-z'\-]+)", npc_text or "")
        if _mn:
            _real = _mn.group(1)
            _generic_roles = ("shopkeeper", "peasant", "guard", "man", "woman",
                              "noble", "fighter", "sage", "merchant", "beggar",
                              "child", "sailor", "monk", "healer", "innkeeper")
            if cur_npc.lower() in _generic_roles and _real != cur_npc:
                kb.merge_npc(cur_npc, _real)
                cur_npc = _real
                session["current_npc"] = _real
                print(f"[{step:03d}] name reveal: merged '{_mn.group(0)}' -> {_real}")
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
        _rn = session.pop("review_nudge", None)
        if _rn:
            alert_parts.append(_rn)
        if session.get("hint") and session.get("hint_ttl", 0) > 0:
            alert_parts.append("HINT from your operator (follow it): " + session["hint"])
            session["hint_ttl"] -= 1
            if session["hint_ttl"] <= 0:
                session.pop("hint", None)
        if session.get("last_bump"):
            alert_parts.append(session["last_bump"])
        # Periodically remind the agent to refresh its running plot summary so
        # 'story_so_far' stays current (it's the always-in-context memory; detail
        # is in recall/quests tools). Every ~15 turns.
        if step > 0 and step % 15 == 0:
            alert_parts.append(
                "Update your running plot summary now: include a \"plot_summary\" "
                "field (<=1000 chars) capturing the overall story so far - who/what "
                "matters, key clues, and your current objective.")
        # Contradiction check (any turn, cheap): if the plot summary still frames
        # a body/search as a goal but we've already searched bodies empty, flag
        # it so the LLM rewrites its OWN narrative (we never edit its prose).
        _summ_low = (kb.episodic_summary or "").lower()
        if (kb.searched_empty and "body" in _summ_low
                and ("search" in _summ_low or "unsearched" in _summ_low
                     or "evidence" in _summ_low)):
            alert_parts.append(
                "NOTE: your plot summary still treats searching a body as a goal, "
                "but already_searched_empty shows you already emptied it (nothing "
                "there). Rewrite \"plot_summary\" to drop that dead lead and set a "
                "real next objective (a person to ask, a place to explore).")
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
        # Dynamic sizing: pick the level that keeps the prompt near a target
        # band. Below the band we EXPAND short-term memory (negative levels) to
        # use the spare context; above it we shrink. This fills the available
        # window instead of leaving it idle at ~55%.
        if frac >= 0.85:
            squeeze = 3
        elif frac >= 0.72:
            squeeze = 2
        elif frac >= 0.60:
            squeeze = 1
        elif frac >= 0.45:
            squeeze = 0
        elif frac >= 0.30:
            squeeze = -1    # expand: plenty of room
        else:
            squeeze = -2    # expand a lot: lots of idle context
        # Damp the controller: move at most ONE level per turn toward the
        # target so a big expansion doesn't overshoot and cause oscillation.
        _prev = session.get("squeeze", 0)
        if squeeze > _prev + 1:
            squeeze = _prev + 1
        elif squeeze < _prev - 1:
            squeeze = _prev - 1
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
        _qd = session.pop("quest_detail", None)
        if _qd is not None:
            state["quest_detail"] = _qd
        # Time/progress awareness: give the model the turn number and how many
        # recent turns it has pursued the SAME goal, so it can notice it is
        # stuck in a cycle and change tack (self-sufficiency, not steering).
        state["turn"] = step
        # Did-not-move signal: if the LAST action was a move/goto but our
        # position is unchanged, the move was blocked/failed. The LLM often
        # doesn't notice this on its own, so flag it EXPLICITLY.
        _cur_xy = ((state.get("player") or {}).get("tx"),
                   (state.get("player") or {}).get("ty"))
        _last_at = session.get("last_action_type")
        if (_last_at in ("move", "goto")
                and _cur_xy == session.get("xy_before_last_action")
                and _cur_xy[0] is not None):
            state["last_move_failed"] = True
        session["xy_before_last_action"] = _cur_xy
        _hist = session.get("reason_hist", [])
        if _hist:
            _head = " ".join(_hist[-1].split()[:3])
            state["same_goal_streak"] = sum(
                1 for r in _hist if " ".join(r.split()[:3]) == _head)
        # Position-based stuck signal (robust to varied reasoning text and
        # interleaved guard turns): if the last ~10 turns barely moved, tell the
        # model plainly. This catches "standing in one spot re-trying variations"
        # that the goal-head streak misses.
        _ph = session.setdefault("stuck_pos_x", [])
        _ph.append((state.get("player") or {}).get("tx"))
        _ph2 = session.setdefault("stuck_pos_y", [])
        _ph2.append((state.get("player") or {}).get("ty"))
        del _ph[:-10]
        del _ph2[:-10]
        _recent_tiles = set(zip(_ph, _ph2))
        if len(_ph) >= 8 and len(_recent_tiles) <= 2:
            state["stuck_in_place"] = True
        # Turns-since-real-progress: a stronger "you're wasting time" signal than
        # a raw turn count. Progress = a quest resolved or a NEW npc met. If many
        # turns pass with NO progress, surface it so the agent gives up on what
        # isn't working (it tends to loop without a sense of futility).
        _prog_key = (sum(1 for q in kb.quests.values() if q.get("status") == "done"),
                     len(kb.npcs))
        if _prog_key != session.get("last_progress_key"):
            session["last_progress_key"] = _prog_key
            session["last_progress_turn"] = step
            session["last_deprio_turn"] = step   # reset the decay clock on progress
        state["turns_since_progress"] = step - session.get("last_progress_turn", step)
        # Quest-priority DECAY: if the agent keeps pursuing the same goal with NO
        # progress, the quest it's effectively 'on' (matched by its recent goal
        # text) gets its priority lowered so it naturally moves to other quests.
        # It's stateless and owns its quest log, so we don't delete - we let
        # futility decay priority. Trigger on EITHER a long no-progress stall OR
        # a tight physical loop (confined to a small area many turns), since
        # trivial quest auto-resolves can keep resetting the no-progress timer
        # while the avatar is really stuck (e.g. the fortress descend churn).
        _confined = (state.get("stuck_in_place")
                     or (session.get("wedge_recent") and len(session["wedge_recent"]) >= 6
                         and (max(t[0] for t in session["wedge_recent"])
                              - min(t[0] for t in session["wedge_recent"])) <= 6
                         and (max(t[1] for t in session["wedge_recent"])
                              - min(t[1] for t in session["wedge_recent"])) <= 6))
        _stalled = state["turns_since_progress"] >= 40
        if ((_stalled or _confined)
                and step - session.get("last_deprio_turn", -999) >= 25):
            session["last_deprio_turn"] = step
            _recent_goal = " ".join((session.get("reason_hist") or [])[-3:])
            demoted = kb.deprioritize_matching_quest(_recent_goal)
            if demoted:
                print(f"[{step:03d}] quest-decay: lowered priority of '{demoted}' "
                      f"(stalled/confined loop)")
                session["last_bump"] = (
                    f"You've spent many turns on '{demoted}' with no progress, so "
                    "its priority was lowered. Work a DIFFERENT quest now - check "
                    "your quest log and pick another high-priority one, and LEAVE "
                    "this area to pursue it.")
        # Tell summarize_state who we're talking to, so it can proactively show
        # which topics we've already asked this NPC and which we have NOT.
        if state.get("conversation_in_progress"):
            state["_convo_npc"] = session.get("current_npc")
        _user = summarize_state(state, kb, session.pop("last_look", ""), alert, squeeze)
        res = ollama.chat_ex(SYSTEM_PROMPT, _user)
        reply = res["content"]
        reason, action = parse_reply(reply)
        # Honor the blocked-breaker: if the last few moves were blocked (e.g.
        # stairs approached from the wrong side - they only climb from the
        # bottom step), and the model is AGAIN trying to move/goto the same way,
        # override with a committed explore AWAY so it can re-approach from a
        # different side or pursue the goal elsewhere.
        if session.pop("force_explore_next", False):
            if isinstance(action, dict) and action.get("type") in ("move", "goto"):
                action = _explore_far(state, session, True)
                reason = "(guard) previous route impassable; exploring away to re-approach"
                print(f"[{step:03d}] blocked-breaker: exploring away from impassable route")
        # Repeated-goal detector: if the model keeps stating the SAME short goal
        # turn after turn, it is likely re-deriving a goal instead of consulting
        # what it already knows. Set an advisory review-nudge for NEXT turn
        # (self-sufficiency: point it at its own quests/recall tools; let it
        # decide). Normalise the reason to a short key.
        _rkey = re.sub(r"[^a-z ]", "", (reason or "").lower()).strip()[:40]
        _rkey = re.sub(r"\s+", " ", _rkey)
        if _rkey and not _rkey.startswith("guard"):
            _hist = session.setdefault("reason_hist", [])
            _hist.append(_rkey)
            del _hist[:-6]
            # Count near-identical recent goals (share the first 3 words).
            _head = " ".join(_rkey.split()[:3])
            _reps = sum(1 for r in _hist if " ".join(r.split()[:3]) == _head)
            if _reps >= 3:
                session["review_nudge"] = (
                    f"You have repeated the goal '{reason[:40]}' several turns. If "
                    "it's not working, you may have already done it or be missing "
                    "info: use the 'quests' tool to review your quest log, or "
                    "'recall' to check your notes on an NPC/topic, then pick a "
                    "DIFFERENT approach. Consulting your own memory is a valid move.")
                session["reason_hist"] = []
        # Feed the FULL context to the GUI so the "Show context" button can
        # display exactly what the model saw this turn (system + user + reply).
        if window.available:
            window.set_context_dump(
                f"===== TURN {step} =====\n"
                f"----- SYSTEM PROMPT -----\n{SYSTEM_PROMPT}\n\n"
                f"----- USER (per-turn state) -----\n{_user}\n\n"
                f"----- MODEL REPLY -----\n{reply}\n")
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
            # Inline PLOT SUMMARY: the LLM maintains a running "story so far" it
            # keeps updated. Replaces the bounded summary that's always in
            # context (for detail it uses recall/quests). Accept a few key names.
            _ps = _obj.get("plot_summary") or _obj.get("story_so_far") or _obj.get("summary")
            if isinstance(_ps, str) and _ps.strip():
                kb.set_plot_summary(_ps)
                print(f"[{step:03d}] plot summary updated ({len(_ps)} chars)")
            # Inline current-quest declaration: the agent tells us which quest it
            # is working on. Tagged onto the action log (shows quest switches).
            _cq = _obj.get("set_current_quest") or _obj.get("current_quest")
            if isinstance(_cq, str) and _cq.strip():
                if _cq.strip() != (kb.current_quest or ""):
                    kb.set_current_quest(_cq)
                    print(f"[{step:03d}] current quest -> {_cq}")
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
    if isinstance(action, dict) and action.get("type") == "quests":
        # Pull the FULL quest log (with notes/prereqs) on demand - it's kept
        # compact in the always-on state to save context, so this lets the
        # agent review details when planning. Shown next turn as "quest_detail".
        session["quest_detail"] = kb.quest_view(max_open=20)
        kb.record_action("reviewed quest log")
        if window.available:
            window.set_action("[quests] reviewed full quest log")
        print(f"[{step:03d}] quests: reviewed full log")
        action = {"type": "wait"}
        reason = "(reviewed my quest log)"

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
    MAX_CONVO_TURNS = 30       # hard cap; a thorough NPC (ask every topic, each
                               # = answer+space) needs many turns. The topic-
                               # aware end-guard closes sooner when topics are
                               # exhausted, so this is just a runaway backstop.
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
    # Range-based oscillation: positions bouncing within a tiny bounding box
    # (e.g. a goto to an UNREACHABLE target that always stops 1-3 tiles short,
    # ping-ponging 1086<->1088). stuck_count misses this (position changes each
    # turn), so detect a small span over recent history and treat as wedged.
    if (len(hist) >= 5 and pos_now[0] is not None and not in_convo):
        _hx = [h[0] for h in hist if h[0] is not None]
        _hy = [h[1] for h in hist if h[1] is not None]
        if _hx and max(max(_hx) - min(_hx), max(_hy) - min(_hy)) <= 3:
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
        session["same_answer_n"] = 0
        session["last_answer_idx"] = None

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
        return {"type": "dismiss"}

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
                action = {"type": "continue"}
                reason = "(guard) leaving: clearing dialog text"
            # Safety: if we've been "leaving" too many turns, force escape.
            session["leaving_n"] = session.get("leaving_n", 0) + 1
            if session["leaving_n"] > 8:
                action = {"type": "dismiss"}
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
        # SAME-ANSWER-REPEAT breaker: if the model keeps picking the SAME index
        # and the conversation isn't progressing (menu keeps re-appearing), it's
        # stuck (e.g. re-'accept'ing an offer that loops). After a few repeats,
        # pick a DIFFERENT available answer - prefer an unasked non-generic
        # topic, else the bye/leave option to exit cleanly.
        if chosen_idx is not None:
            if chosen_idx == session.get("last_answer_idx"):
                session["same_answer_n"] = session.get("same_answer_n", 0) + 1
            else:
                session["same_answer_n"] = 0
            session["last_answer_idx"] = chosen_idx
            if session.get("same_answer_n", 0) >= 2 and answers:
                _gen = {"name", "job", "bye", "yes", "no", "leave", "farewell",
                        "goodbye", "nothing", "hello"}
                _cn = session.get("current_npc", "?")
                _ask = set((kb.recall_npc(_cn) or {}).get("topics_asked", []))
                _alt = [(i, a) for i, a in enumerate(answers)
                        if i != chosen_idx and a.lower() not in _gen and a not in _ask]
                if _alt:
                    action = {"type": "answer", "index": _alt[0][0]}
                    reason = f"(guard) same answer looping; trying '{_alt[0][1]}' instead"
                    session["same_answer_n"] = 0
                    chosen_idx = _alt[0][0]
                    print(f"[{step:03d}] answer-loop: switch to idx {_alt[0][0]} '{_alt[0][1]}'")
                else:
                    _bye = next((i for i, a in enumerate(answers)
                                 if a.lower() in ("bye", "leave", "farewell", "goodbye", "nothing")), None)
                    if _bye is not None:
                        action = {"type": "answer", "index": _bye}
                        reason = "(guard) same answer looping; exiting conversation"
                        session["same_answer_n"] = 0
                        chosen_idx = _bye
                        print(f"[{step:03d}] answer-loop: no new topics; leaving via idx {_bye}")
        too_long = session.get("convo_turns", 0) >= MAX_CONVO_TURNS
        # Are there still useful (non-generic, unasked) topics on the CURRENT
        # menu? If so, don't force-end - let the agent exhaust the tree first.
        _generic0 = {"name", "job", "bye", "yes", "no", "leave", "farewell",
                     "goodbye", "nothing", "hello"}
        _cnpc0 = session.get("current_npc", "?")
        _asked0 = set((kb.recall_npc(_cnpc0) or {}).get("topics_asked", []))
        _useful_left = [a for a in answers
                        if a.lower() not in _generic0 and a not in _asked0]
        # Only treat "repeating" as done if there's nothing useful left to ask.
        repeating = (chosen_idx is not None and chosen_idx in picked
                     and len(picked) >= max(1, len(answers) - 1)
                     and not _useful_left)
        if too_long or repeating:
            # Engage the leaving latch so we drive to a clean close, not just
            # a single bye that may re-open choices.
            session["leaving"] = True
            session["leaving_n"] = 0
            action = _bye_action()
            reason = f"(guard) ending conversation ({'too long' if too_long else 'looping'})"
        elif chosen_idx is not None:
            picked.add(chosen_idx)

    # 0b) Don't LEAVE a conversation while meaningful topics are unasked - a
    #     thorough player exhausts the dialogue tree. If the model picks a
    #     bye/leave answer but there are still useful unasked topics, redirect
    #     to ask one of them instead. Skip generic closers.
    if (state.get("conversation_active") and answers
            and isinstance(action, dict) and action.get("type") == "answer"
            and not session.get("leaving")):
        _ci = action.get("index")
        _choice = answers[_ci].lower() if isinstance(_ci, int) and 0 <= _ci < len(answers) else ""
        _leaving_choice = any(w in _choice for w in ("bye", "leave", "farewell", "goodbye", "nothing"))
        if _leaving_choice:
            _generic = {"name", "job", "bye", "yes", "no", "leave", "farewell",
                        "goodbye", "nothing", "hello"}
            # Unasked, non-generic topics still available in the CURRENT menu.
            cur_npc = session.get("current_npc", "?")
            asked = set((kb.recall_npc(cur_npc) or {}).get("topics_asked", []))
            unasked = [(i, a) for i, a in enumerate(answers)
                       if a.lower() not in _generic and a not in asked]
            # Also allow 'name'/'job' if not yet asked (a thorough player asks).
            basic = [(i, a) for i, a in enumerate(answers)
                     if a.lower() in ("name", "job") and a not in asked]
            pick = unasked or basic
            if pick and session.get("topic_push", 0) < 12:
                session["topic_push"] = session.get("topic_push", 0) + 1
                action = {"type": "answer", "index": pick[0][0]}
                reason = f"(guard) exhausting dialogue: asking '{pick[0][1]}' before leaving"
    if not state.get("conversation_active"):
        session["topic_push"] = 0

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
            action = {"type": "dismiss"}
            reason = "(guard) conversation hung with no choices; escaping"
            session["stuck_convo_n"] = 0
        else:
            action = {"type": "continue"}
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
        cx, cy = _grid_center(rows)
        def _cell(dx, dy):
            x, y = cx + dx, cy + dy
            if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                return rows[y][x]
            return "#"
        WALK = ".*&C/xbnE+"  # walkable-ish: open, items, actors, doors, bodies,
        #                      containers, exits. NOT '~' water, '=' barrier, '#'.
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
        R = max(cx, cy) if (cx or cy) else 12
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
        # Detect a SMALL-AREA oscillation trap: track recent positions; if the
        # last several turns span only 1-2 distinct tiles, we're bouncing in a
        # pocket even though position "changes" each turn (so a same-tile stall
        # counter would never fire). Force a committed escape then.
        _pp_now = (px, py)
        _wr = session.setdefault("wedge_recent", [])
        _wr.append(_pp_now)
        del _wr[:-8]
        # Range-based oscillation: if the last several positions all fall within
        # a tiny bounding box (<=3 tiles across each axis), we're stuck bouncing
        # even if that's 3-4 distinct tiles (a same-count check missed this).
        if len(_wr) >= 6:
            _xs = [t[0] for t in _wr]
            _ys = [t[1] for t in _wr]
            _span = max(max(_xs) - min(_xs), max(_ys) - min(_ys))
            _oscillating = _span <= 3
        else:
            _oscillating = False
        _force_step = _oscillating
        if best and best_d >= 3 and _force_step:
            # We're trapped in a tiny pocket but the flood-fill sees a far open
            # tile: commit a goto straight to it and clear the recent buffer so
            # we don't immediately re-trigger.
            session["wedge_recent"] = []
            action = {"type": "goto", "tx": px + best[0], "ty": py + best[1]}
            reason = f"(guard) oscillation trap; committing to far tile ({px+best[0]},{py+best[1]})"
        elif _force_step:
            # Trapped and the visible flood-fill can't find a far tile (a tortuous
            # pocket). Escape via the engine's FULL pathfinder: goto a remembered
            # open place (which uses A* over the whole map, not just the visible
            # grid). Rotate through known places so we don't retry a bad one.
            session["wedge_recent"] = []
            _places = kb.places_view(px, py, limit=8) if kb else []
            _far = [pl for pl in _places
                    if abs(pl.get("dx", 0)) + abs(pl.get("dy", 0)) >= 6]
            _idx = session.get("trap_place_idx", 0)
            if _far:
                _pl = _far[_idx % len(_far)]
                session["trap_place_idx"] = _idx + 1
                _ptx = px + _pl.get("dx", 0)
                _pty = py + _pl.get("dy", 0)
                action = {"type": "goto", "tx": _ptx, "ty": _pty}
                reason = f"(guard) trap; routing to known place '{_pl.get('name')}' via full A*"
            else:
                # No far place known: step toward the single open neighbor.
                _dd = {"n": (0,-1),"s": (0,1),"e": (1,0),"w": (-1,0),
                       "ne": (1,-1),"nw": (-1,-1),"se": (1,1),"sw": (-1,1)}
                _pk = next((d for d, (ddx, ddy) in _dd.items()
                            if _cell(ddx, ddy) in WALK), "w")
                action = {"type": "move", "dir": _pk, "speed": 120}
                reason = f"(guard) trap; stepping {_pk} toward the only opening"
        elif best and best_d >= 2 and not _force_step:
            action = {"type": "goto", "tx": px + best[0], "ty": py + best[1]}
            reason = f"(guard) wedged; flood-fill escape to open tile ({px+best[0]},{py+best[1]})"
        elif (has_door or (state.get("doors") or [])) and not _force_step:
            action = {"type": "open"}
            reason = "(guard) wedged; opening a nearby door to escape"
        else:
            # Truly boxed in on the visible grid - step toward any adjacent
            # walkable cell, rotating by attempt to avoid oscillating.
            deltas = {"n": (0,-1),"s": (0,1),"e": (1,0),"w": (-1,0),
                      "ne": (1,-1),"nw": (-1,-1),"se": (1,1),"sw": (-1,1)}
            pref = list(deltas.keys())
            # Rotate by recent-oscillation size so repeated forced steps try
            # DIFFERENT directions instead of re-picking the same blocked one.
            off = (n + len(session.get("wedge_recent", []))) % len(pref)
            pref = pref[off:] + pref[:off]
            # Prefer a truly walkable neighbor; fall back to the rotated first.
            picked = next((d for d in pref if _cell(*deltas[d]) in WALK), pref[0])
            action = {"type": "move", "dir": picked, "speed": 120}
            reason = f"(guard) wedged; forcing step {picked}"
    else:
        # 1b) A "continue"/"dismiss"/key press outside a conversation does
        #     nothing useful -> redirect to something productive.
        if (isinstance(action, dict)
                and action.get("type") in ("key", "continue", "dismiss")
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
        # Non-lootable corpse short-circuit: if the closest body-like thing is a
        # corpse the engine flagged as NOT a container (nothing to take) and no
        # real lootable body/container is nearby, searching it is pointless -
        # examine it instead and tell the agent. (The ritually-murdered human
        # body vs the lootable gargoyle - general, not plot-specific.)
        _objs_all = state.get("objects") or []
        _lootable_near = [o for o in _objs_all
                          if o.get("body")
                          and abs(o.get("dx", 99)) + abs(o.get("dy", 99)) <= 4]
        _corpse_near = [o for o in _objs_all
                        if o.get("corpse")
                        and abs(o.get("dx", 99)) + abs(o.get("dy", 99)) <= 4]
        if _corpse_near and not _lootable_near:
            session["last_look"] = describe_scene(state, kb)
            kb.record_action("examined a corpse (not lootable)")
            action = {"type": "wait"}
            reason = "(guard) corpse not lootable; examined instead"
            session["last_bump"] = (
                "That corpse is NOT a container - it has nothing to take, so "
                "'search' does nothing on it. You've noted the scene; move on "
                "(look for a real container/body, loose items, or a person "
                "with information).")
            print(f"[{step:03d}] search-guard: corpse not lootable; examine instead")

    # Remaining search-guard logic only applies if the action is STILL a search
    # (the corpse short-circuit above may have turned it into a wait).
    if (isinstance(action, dict) and action.get("type") == "search"
            and not state.get("conversation_in_progress")):
        _pp = state.get("player") or {}
        _here = (_pp.get("tx"), _pp.get("ty"))
        # Consider "already looted here" if we're within 2 tiles of any spot we
        # emptied (the agent re-searches slightly different adjacent tiles).
        _looted = session.get("looted_spots", set())
        _near_looted = any(abs(_here[0]-lx) + abs(_here[1]-ly) <= 2
                           for (lx, ly) in _looted)
        # Already looted from this exact spot? Don't re-search - the body is
        # empty. Move on to the next objective instead of looping.
        if _near_looted:
            session["last_bump"] = ("You already searched and emptied the body here - "
                                    "there is nothing left to take. Move on: grab any "
                                    "loose items you can see, or pursue your other goals.")
            # First, if there is notable UNOWNED loot visible nearby, grab it -
            # the body being empty doesn't mean the SCENE is empty (e.g. the
            # Gargoyle jewelry lying next to the body). This breaks the fruitless
            # re-search loop by doing something productive.
            _pp2 = state.get("player") or {}
            _NOT_LOOT = ("blood", "trap", "lever", "switch", "grave", "coffin",
                         "altar", "shrine", "cauldron", "skeleton", "locked",
                         "rune", "chest", "body", "corpse")  # scenery/containers
            _loot = [o for o in (state.get("objects") or [])
                     if o.get("name") and not o.get("owned") and not o.get("body")
                     and kb.is_notable_object(o.get("name"))
                     and not any(w in o.get("name", "").lower() for w in _NOT_LOOT)
                     and o.get("name") not in session.get("picked", set())
                     and abs(o.get("dx", 99)) + abs(o.get("dy", 99)) <= 10]
            if _loot:
                _it = min(_loot, key=lambda o: abs(o["dx"]) + abs(o["dy"]))
                if abs(_it["dx"]) <= 1 and abs(_it["dy"]) <= 1:
                    action = {"type": "pickup", "name": _it["name"]}
                    session.setdefault("picked", set()).add(_it["name"])
                    reason = f"(guard) body empty; grabbing nearby {_it['name']}"
                else:
                    action = {"type": "goto", "tx": _pp2.get("tx", 0) + _it["dx"],
                              "ty": _pp2.get("ty", 0) + _it["dy"]}
                    reason = f"(guard) body empty; going to loot {_it['name']}"
                print(f"[{step:03d}] search-guard: body empty; loot '{_it['name']}'")
            else:
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
                    # garbage' etc). search only opens BODIES/CONTAINERS. The
                    # look HANDLER already ran earlier this turn, so emitting a
                    # look action here would fall through to the engine (unknown
                    # action). Instead produce the examine description directly
                    # and wait, and tell the agent what search is for.
                    session["last_look"] = describe_scene(state, kb)
                    kb.record_action("looked around")
                    action = {"type": "wait"}
                    reason = "(guard) nothing to search here; examined instead"
                    session["last_bump"] = (
                        "'search' only opens a nearby BODY or CONTAINER (chest, "
                        "barrel, bag). There is none within reach, so it does "
                        "nothing on scenery like garbage/tables. To inspect the "
                        "area use 'look'; to grab a loose item use 'pickup'.")
                    print(f"[{step:03d}] search-guard: no searchable target; examined instead")

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

    # --- Open guard: "open" only works on a real DOOR within a few tiles. If
    #     there is no door nearby, don't waste the turn - explain (and note the
    #     town gate/portcullis is NOT opened this way; it needs the password /
    #     winch). Redirect toward a real nearby door if one exists.
    if (isinstance(action, dict) and action.get("type") == "open"
            and not state.get("conversation_in_progress")):
        _doors = state.get("doors") or []
        # Track consecutive open attempts with no door in view; only intervene
        # after a repeat so we never block a legitimate first attempt.
        if not _doors:
            session["open_fails"] = session.get("open_fails", 0) + 1
        else:
            session["open_fails"] = 0
        if not _doors and session.get("open_fails", 0) >= 2:
            session["open_fails"] = 0
            _exit_near = any(o.get("town_exit") for o in (state.get("objects") or []))
            if _exit_near:
                session["last_bump"] = (
                    "There is no ordinary door here. The town GATE/portcullis is "
                    "not opened with 'open' - you need the gate PASSWORD (from the "
                    "Mayor) and it is operated at the gate. Pursue the password.")
            else:
                session["last_bump"] = (
                    "'open' keeps finding no door. Doors show in the 'doors' list "
                    "and as '+'/'/' on the grid. Move next to a door, or 'goto' "
                    "through it (goto opens it for you).")
            action = _explore_far(state, session, wedged)
            reason = "(guard) repeated open with no door; moving on"
            print(f"[{step:03d}] open-guard: repeated no-door open")
        elif _doors:
            # A door IS here. If the agent keeps issuing 'open' from the SAME
            # tile, the door is already open and standing there does nothing -
            # the goal is to pass THROUGH. INFORM it (self-sufficiency: give the
            # fact, let it act) to goto a tile on the far side of the door.
            _pp = state.get("player") or {}
            _here = (_pp.get("tx"), _pp.get("ty"))
            if session.get("last_open_pos") == _here:
                session["open_same"] = session.get("open_same", 0) + 1
            else:
                session["open_same"] = 0
            session["last_open_pos"] = _here
            if session.get("open_same", 0) >= 1:
                # nearest door -> suggest the tile just beyond it as a goto
                _d = min(_doors, key=lambda d: abs(d.get("dx", 9)) + abs(d.get("dy", 9)))
                _bx = _here[0] + 2 * _d.get("dx", 0)
                _by = _here[1] + 2 * _d.get("dy", 0)
                session["last_bump"] = (
                    "The door is already open - standing here re-opening it does "
                    f"nothing. To ENTER, 'goto' a tile on the far SIDE of the door "
                    f"(e.g. goto ({_bx},{_by})); goto walks you through an open or "
                    "closed door. Don't repeat 'open'.")
                print(f"[{step:03d}] open-info: door already open; suggested goto through")

    # --- Wall-aware move guard: never walk into a '#'. ------------------
    # The grid is centered on the avatar (radius 12 -> center [12][12]).
    # A '.' or open door '/' or an item '*' or NPC is walkable; '#' and a
    # closed door '+' are not directly walkable (a closed door needs goto,
    # which opens it). If the model's move heads into a wall, pick the closest
    # walkable direction toward the same heading, else pathfind/turn.
    if isinstance(action, dict) and action.get("type") == "move":
        grid = state.get("grid") or ""
        rows = grid.split("\n")
        if len(rows) >= 9 and rows[0] and len(rows[0]) >= 9:
            cx, cy = _grid_center(rows)
            deltas = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0),
                      "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1)}
            def cell(dx, dy):
                x, y = cx + dx, cy + dy
                if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                    return rows[y][x]
                return "#"
            def walkable(ch):
                # open, item, npc, companion, body, container, open/closed door,
                # EXIT/STAIRS ('E'), self. Must include 'E' or this guard will
                # redirect a step onto stairs and prevent climbing.
                return ch in ".*&CxbnE/+@"
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
        _chosen = None
        if isinstance(idx, int) and 0 <= idx < len(answers):
            _chosen = answers[idx]
            kb.record_my_reply(_chosen)
        elif action.get("text"):
            _chosen = str(action.get("text"))
            kb.record_my_reply(_chosen)
        if _chosen:
            # Log the actual answer TEXT (not the opaque index) to the temporal
            # action log, and stash it so the GUI turn display shows it too.
            kb.record_action(f"answered: \"{_chosen[:60]}\"")
            session["answer_text"] = _chosen

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
                    f"To find a PERSON, enter buildings (goto through doors '+'/'/' "
                    f"and look inside). Pick a KNOWN place or a visible target, or "
                    f"explore a NEW direction - don't repeat goto '{nm}'.")
                print(f"[{step:03d}] goto unresolved: '{nm}' (known: {names})")
                # Self-sufficiency: INFORM (bump above) and let the model choose.
                # Track how often it repeats an unresolvable goto for the SAME
                # name; escalate the intervention only if it keeps doing it, and
                # never ping-pong. First time: just 'look' (a no-move examine) so
                # the model re-decides with the bump. Then try ONE fresh door.
                # Only after persistent repeats fall back to explore-far.
                _un = session.setdefault("unresolved_goto", {})
                _un[nm] = _un.get(nm, 0) + 1
                _cnt = _un[nm]
                if _cnt <= 1:
                    # Give the model the info and a no-op examine; it decides next.
                    session["last_look"] = describe_scene(state, kb)
                    action = {"type": "wait"}
                    reason = f"(guard) '{nm}' unknown; informed, letting agent choose"
                else:
                    # It ignored the info and repeated. Try ONE unvisited door
                    # (buildings hide people/chests), else a single explore step.
                    door = None
                    for d in (state.get("doors") or []):
                        key = (_pp.get("tx", 0) + d.get("dx", 0),
                               _pp.get("ty", 0) + d.get("dy", 0))
                        if key not in session.setdefault("entered_doors", set()):
                            door = (key, d)
                            break
                    if door is not None:
                        (dtx, dty), _d = door
                        session["entered_doors"].add((dtx, dty))
                        action = {"type": "goto", "tx": dtx, "ty": dty}
                        reason = f"(guard) '{nm}' unknown; trying a new door @({dtx},{dty})"
                    else:
                        action = _explore_far(state, session, wedged)
                        reason = f"(guard) '{nm}' unknown; exploring new ground"
                    # reset so we inform again next time rather than looping here
                    if _cnt >= 4:
                        _un[nm] = 0

    # --- Search/goto conflation informer: the agent often WRITES "search bag
    #     for key" but EMITS a goto (it conflates arriving with searching). If
    #     its reasoning mentions search/open/look and there is a searchable
    #     container/body/bag adjacent RIGHT NOW, tell it to use the 'search'
    #     action - arriving is not searching. Inform-only (self-sufficiency);
    #     we do NOT convert the action for it. ---------------------------------
    if (isinstance(action, dict) and action.get("type") == "goto"
            and not state.get("conversation_in_progress")):
        _rl = (reason or "").lower()
        if any(w in _rl for w in ("search", "open", "look inside", "loot")):
            _adj = [o for o in (state.get("objects") or [])
                    if (o.get("body") or "bag" in (o.get("name") or "").lower()
                        or "chest" in (o.get("name") or "").lower()
                        or "barrel" in (o.get("name") or "").lower())
                    and abs(o.get("dx", 9)) <= 1 and abs(o.get("dy", 9)) <= 1]
            if _adj:
                _nm = _adj[0].get("name", "it")
                session["last_bump"] = (
                    f"You are standing next to the {_nm}. Arriving there is NOT "
                    "searching it. To look inside, emit the 'search' action now "
                    "(not goto). If you already searched it and it was empty, stop "
                    "returning - update your plot_summary and try another lead.")
                print(f"[{step:03d}] conflation-info: adjacent {_nm}; suggested search")

    # --- Emptied-body magnet: the agent sometimes keeps issuing goto toward a
    #     spot it ALREADY searched empty. PRINCIPLE: make the agent self-
    #     sufficient - INFORM it (don't seize its action) so it can decide, and
    #     only override as a last resort if it's truly stuck in an infinite
    #     loop despite being told. We never edit its memory to move it along.
    #
    # --- Already-at-ground guard: the agent repeatedly tries to "descend to
    #     ground level" while it is ALREADY at ground (tz 0), gotoing a stairs
    #     tile at the same elevation. This is a provably-false premise. Inform
    #     it plainly and, after a couple of repeats, redirect to explore so it
    #     stops the loop. (Purely correcting a false belief about a physical
    #     fact - not choosing content for it.)
    # --- Fortress-wall escape LATCH: the descend/ground guards can ping-pong
    #     the avatar around the wall (down->explore->reclimb->down...) because
    #     the wall is right there and the LLM keeps re-targeting it. When we
    #     detect a tight confined loop with heavy descend churn, LATCH a
    #     committed journey to a far ground destination and honor it for several
    #     turns REGARDLESS of the LLM, physically pulling the avatar away from
    #     the wall's pull. This is a last-resort anti-stall, not content steering.
    _cur_tz_now = (state.get("player") or {}).get("tz", 0) or 0
    if session.get("escape_latch", 0) > 0:
        session["escape_latch"] -= 1
        _et = session.get("escape_target", (1065, 2180))
        # If we've arrived (near target, ground level), drop the latch.
        _pp = state.get("player") or {}
        if (abs(_pp.get("tx", 0) - _et[0]) + abs(_pp.get("ty", 0) - _et[1]) <= 3
                and (_pp.get("tz", 0) or 0) == 0):
            session["escape_latch"] = 0
        else:
            action = {"type": "goto", "tx": _et[0], "ty": _et[1], "tz": 0}
            reason = f"(latch) escaping to ({_et[0]},{_et[1]}) - leaving the wall area"
            print(f"[{step:03d}] escape-latch: committed goto {_et} ({session['escape_latch']} left)")
            # skip the rest of the guards this turn; execute the latched move
            _atype = action.get("type")
            result = exult.act(action)
            kb.record_tool(_atype, result.get("ok") if isinstance(result, dict) else None)
            session["last_action_type"] = _atype
            if window.available:
                window.update_turn(step)
                window.set_action(json.dumps(action) + "\n\n-> " + json.dumps(result))
                window.set_thinking(reason)
            return
    # Arm the latch on a confined + descend-heavy loop.
    _recent_pos = session.get("wedge_recent", [])
    if (len(_recent_pos) >= 6
            and (max(t[0] for t in _recent_pos) - min(t[0] for t in _recent_pos)) <= 6
            and (max(t[1] for t in _recent_pos) - min(t[1] for t in _recent_pos)) <= 6):
        _descend_recent = sum(1 for r in (session.get("reason_hist") or [])[-6:]
                              if "descend" in r or "fortress" in r or "stairs" in r)
        if _descend_recent >= 3 and session.get("escape_latch", 0) == 0:
            # Head to a far GROUND destination (a known place far away, else the
            # murder-scene/start area to the NW of the fortress).
            _pp = state.get("player") or {}
            _far = None
            for _pl in (kb.places_view(_pp.get("tx", 0), _pp.get("ty", 0), limit=12) if kb else []):
                if abs(_pl.get("dx", 0)) + abs(_pl.get("dy", 0)) >= 14:
                    _far = (_pp.get("tx", 0) + _pl.get("dx", 0), _pp.get("ty", 0) + _pl.get("dy", 0))
                    break
            session["escape_target"] = _far or (1065, 2180)
            session["escape_latch"] = 10
            # also demote the fortress cluster now
            kb.deprioritize_matching_quest("fortress descend stairs wall ground")
            print(f"[{step:03d}] escape-latch: ARMED -> {session['escape_target']}")

    # Arm the latch when recent reasoning is DOMINATED by descend/fortress/
    # stairs (the fixation), regardless of exact box size - the avatar may do a
    # WIDE oscillation (fortress<->start) that a tiny-box check misses.
    _rh6 = (session.get("reason_hist") or [])[-6:]
    _fixate = sum(1 for r in _rh6
                  if "descend" in r or "fortress" in r or "stairs" in r or "climb" in r)
    if _fixate >= 4 and session.get("escape_latch", 0) == 0:
        _pp = state.get("player") or {}
        _far = None
        for _pl in (kb.places_view(_pp.get("tx", 0), _pp.get("ty", 0), limit=12) if kb else []):
            _nm = (_pl.get("name", "") or "").lower()
            if ("fortress" in _nm or "wall" in _nm or "stair" in _nm):
                continue   # don't escape TO the fortress
            if abs(_pl.get("dx", 0)) + abs(_pl.get("dy", 0)) >= 10:
                _far = (_pp.get("tx", 0) + _pl.get("dx", 0), _pp.get("ty", 0) + _pl.get("dy", 0))
                break
        session["escape_target"] = _far or (1065, 2180)
        session["escape_latch"] = 12
        kb.deprioritize_matching_quest("fortress descend stairs wall ground climb")
        print(f"[{step:03d}] escape-latch: ARMED (fixation) -> {session['escape_target']}")

    _at_ground = _cur_tz_now == 0
    _wants_descend = any(w in (reason or "").lower()
                         for w in ("descend", "climb down", "go down", "down to ground",
                                   "to ground level", "down the stairs", "down the fortress",
                                   "down from"))
    if _at_ground and _wants_descend:
        n = session.get("false_descend", 0) + 1
        session["false_descend"] = n
        session["last_bump"] = (
            "FACT: you are ALREADY at ground level (elevation 0). You cannot "
            "'descend' - there is no lower level here. Stop trying to go down. "
            "Whatever you're looking for at ground level, you are already on it - "
            "walk to the PERSON or PLACE you want (e.g. the stables) directly.")
        if n >= 2:
            session["false_descend"] = 0
            action = _explore_far(state, session, wedged)
            reason = "(guard) already at ground; stop descending, explore"
            print(f"[{step:03d}] ground-guard: already at tz0; redirect from descend")
    elif _cur_tz_now > 0 and _wants_descend:
        # Elevated and wanting down: goto a GROUND tile with tz:0. The engine
        # descends 1 level/step, so targeting a known ground destination (a
        # remembered place, else the world-start area) at tz 0 walks the avatar
        # back down reliably (verified tz4->0). This is the descend analog of
        # the climb - use goto with an explicit ground Z rather than poking the
        # stairs tile.
        _pp0 = state.get("player") or {}
        _here0 = (_pp0.get("tx", 0), _pp0.get("ty", 0))
        _gp = None
        for _pl in (kb.places_view(_here0[0], _here0[1], limit=12) if kb else []):
            if abs(_pl.get("dx", 0)) + abs(_pl.get("dy", 0)) >= 12:  # must be FAR
                _gp = (_here0[0] + _pl.get("dx", 0), _here0[1] + _pl.get("dy", 0))
                break
        if _gp is None:
            _gp = (1079, 2214)   # Trinsic start / murder-scene area (ground)
        action = {"type": "goto", "tx": _gp[0], "ty": _gp[1], "tz": 0}
        reason = f"(guard) descending: goto ground tile ({_gp[0]},{_gp[1]}) tz0"
        print(f"[{step:03d}] descend-guard: goto ground {_gp} tz0 from tz{_cur_tz_now}")
    elif not _wants_descend:
        session["false_descend"] = 0

    if (isinstance(action, dict) and action.get("type") == "goto" and "tx" in action
            and not state.get("conversation_in_progress")):
        _tgt = (action.get("tx"), action.get("ty"))
        _empties = [(v.get("tx"), v.get("ty"))
                    for v in (kb.searched_empty or {}).values()
                    if v.get("tx") is not None]
        _to_empty = any(_tgt[0] is not None
                        and abs(_tgt[0] - ex) + abs(_tgt[1] - ey) <= 2
                        for (ex, ey) in _empties)
        # A fresh lootable body or unowned notable loot at the target is a
        # legitimate reason - never interfere with that.
        _worth = any(
            (o.get("body") or (o.get("name") and not o.get("owned")
                               and kb.is_notable_object(o.get("name"))
                               and not o.get("corpse")))
            and _tgt[0] is not None
            and abs((state.get("player") or {}).get("tx", 0) + o.get("dx", 0) - _tgt[0])
                + abs((state.get("player") or {}).get("ty", 0) + o.get("dy", 0) - _tgt[1]) <= 2
            for o in (state.get("objects") or []))
        if _to_empty and not _worth:
            n = session.get("empty_magnet", 0) + 1
            session["empty_magnet"] = n
            # INFORM every time: give the agent the fact so it can self-correct.
            session["last_bump"] = (
                "Heads up: that destination is a body/spot you ALREADY searched "
                "and found EMPTY (see already_searched_empty). There is nothing "
                "to gain there. If your plot summary still lists it as a goal, "
                "rewrite \"plot_summary\" to drop it and choose a real lead "
                "(a person to ask, a place to explore). Your call.")
            # LAST-RESORT override only if it ignores the info repeatedly (hard
            # loop) - purely to prevent an infinite stall, not to steer content.
            if n >= 4:
                session["empty_magnet"] = 0
                action = _explore_far(state, session, wedged)
                reason = "(guard) stuck looping on an emptied body; forcing explore"
                print(f"[{step:03d}] goto-guard: hard loop on emptied body -> explore")
            else:
                print(f"[{step:03d}] goto-guard: informed about emptied spot (x{n})")
        else:
            session["empty_magnet"] = 0


        _pp = state.get("player") or {}
        here = (_pp.get("tx"), _pp.get("ty"))
        tgt = (action["tx"], action["ty"])
        # Did we move at all since the previous turn?
        moved = here != session.get("prev_pos_for_goto")
        session["prev_pos_for_goto"] = here
        # --- Stairs climb (empirical): a staircase is a RAMP - you climb by
        #     repeatedly MOVING in the ascending direction and tz rises one per
        #     step (verified: walking east across the ramp took tz 0->5 onto the
        #     wall). goto stalls at tz1 and oscillates, so we drive MOVEs. We
        #     learn the ascending direction from feedback: remember our last tz;
        # NOTE: the special-case stairs guards below are DISABLED. Testing showed
        # Exult's own pathfinder climbs multi-level ramps correctly when goto is
        # given a destination at the right elevation (the engine steps up 1 level
        # per tile via is_blocked's max_rise). Our goto Z-resolution now sets the
        # destination to the standable height, so 'goto a wall-top/upper tile'
        # climbs the whole staircase (verified tz 0->5 in 2 gotos). The guards
        # were solving the wrong problem (targeting the stairs TILE, where goto
        # arrives at tz1 and stops) and fought the agent, so we let goto do it.
        _reason_stairs = False
        rows = (state.get("grid") or "").split("\n")
        cx, cy = _grid_center(rows)
        _e_cells = [(x - cx, y - cy)
                    for y in range(len(rows)) for x in range(len(rows[y]))
                    if rows[y][x] == "E"]
        _cur_tz = (state.get("player") or {}).get("tz", 0) or 0
        _sc = session.get("stairs_attempts", 0)
        # Only ASSIST the climb when: the model itself is trying to move/goto
        # (don't hijack talk/search/etc), it's reasoning about stairs, stairs
        # are CLOSE (within ~4 tiles), and it isn't already up high. This keeps
        # the guard from commandeering the agent every turn.
        _near_stairs = _e_cells and min(abs(c[0]) + abs(c[1]) for c in _e_cells) <= 4
        _act_type = action.get("type") if isinstance(action, dict) else None
        if (_reason_stairs and _near_stairs and _sc < 25
                and _act_type in ("move", "goto") and _cur_tz < 5):
            session["stairs_attempts"] = _sc + 1
            _nm = {(0,-1):"n",(0,1):"s",(1,0):"e",(-1,0):"w",
                   (1,-1):"ne",(1,1):"se",(-1,1):"sw",(-1,-1):"nw"}
            _prev_tz = session.get("climb_prev_tz")
            _prev_dir = session.get("climb_dir")
            session["climb_prev_tz"] = _cur_tz
            # If our last move raised tz, we're on the ramp ascending - keep going.
            if _prev_dir is not None and _prev_tz is not None and _cur_tz > _prev_tz:
                action = {"type": "move", "dir": _prev_dir, "speed": 120}
                reason = f"(guard) climbing ramp {_prev_dir} (tz {_prev_tz}->{_cur_tz})"
                print(f"[{step:03d}] stairs-climb: continue {_prev_dir} tz={_cur_tz}")
            else:
                # Pick a direction toward the nearest E cell and try it; the
                # ramp base is reached by heading at the stairs. Cycle the
                # candidate directions across attempts until one raises tz.
                _near = min(_e_cells, key=lambda c: abs(c[0]) + abs(c[1]))
                _tdx = (1 if _near[0] > 0 else -1 if _near[0] < 0 else 0)
                _tdy = (1 if _near[1] > 0 else -1 if _near[1] < 0 else 0)
                # candidate step dirs, biased toward the stairs, rotating by attempt
                _cands = [(_tdx, _tdy), (_tdx, 0), (0, _tdy),
                          (1, 0), (-1, 0), (0, 1), (0, -1)]
                _cands = [c for c in _cands if c != (0, 0)]
                _pick = _cands[_sc % len(_cands)]
                session["climb_dir"] = _nm[_pick]
                action = {"type": "move", "dir": _nm[_pick], "speed": 120}
                reason = f"(guard) approaching/climbing stairs: try {_nm[_pick]}"
                print(f"[{step:03d}] stairs-climb: try {_nm[_pick]} toward E{_near} tz={_cur_tz}")
        elif _reason_stairs and _e_cells and _sc >= 25:
            # Gave the climb enough tries and it hasn't worked - stop forcing it.
            # Inform the agent this staircase is a dead end so it stops trying to
            # go up here and pursues its goal (the Mayor) by another route.
            session["stairs_attempts"] = 0     # allow a fresh future attempt elsewhere
            session["last_bump"] = (
                "You've repeatedly tried to climb these stairs with no success - "
                "treat this route as a DEAD END. The person you seek is likely "
                "NOT up here; stop trying to climb and look for them elsewhere in "
                "town (they may be in a building or wandering the streets).")
            action = _explore_far(state, session, True)
            reason = "(guard) stairs unclimbable; abandoning and exploring elsewhere"
            print(f"[{step:03d}] stairs: attempts exhausted; abandoning dead-end")
        # Reset the attempt counter once we actually climbed (tz>0) or wandered
        # away from any stairs, so a legitimate future staircase isn't pre-capped.
        if (p_now := (state.get("player") or {})).get("tz", 0) > 0 or not _e_cells:
            session["stairs_attempts"] = 0
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
            cx, cy = _grid_center(rows)
            def _cell(ddx, ddy):
                x, y = cx + ddx, cy + ddy
                if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                    return rows[y][x]
                return "#"
            WALK = ".*&C/xbnE+"   # open, item, npc, comp, door, body, container, exit
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
    # Blocked-move / unreachable-target breaker: a move that comes back
    # blocked:true (e.g. stairs that can't be climbed from this angle) means the
    # avatar physically can't go there. If this keeps happening, ABANDON the
    # target: record a strong bump telling the agent this route is impassable,
    # and force a committed explore AWAY so it stops wasting turns and pursues
    # its goal another way (the target NPC/place may be reachable elsewhere).
    if _atype == "move" and isinstance(result, dict) and result.get("blocked"):
        session["blocked_moves"] = session.get("blocked_moves", 0) + 1
    elif _atype == "move" and isinstance(result, dict) and not result.get("blocked"):
        session["blocked_moves"] = 0
    if session.get("blocked_moves", 0) >= 3:
        session["blocked_moves"] = 0
        session["last_bump"] = (
            "That route is IMPASSABLE - your last several steps were blocked "
            "(e.g. stairs you cannot climb from this side, or a wall). Stop "
            "trying to go that exact way. Pick a DIFFERENT approach or a "
            "different objective; the person/place you want is reachable by "
            "another route or will come to you. Walk away and try elsewhere.")
        # Overwrite the just-executed (blocked) action's follow-up by nudging
        # exploration next turn via a flag the top-of-turn logic can honor.
        session["force_explore_next"] = True
        print(f"[{step:03d}] blocked-breaker: route impassable; abandon target")
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
        _disp = dict(action) if isinstance(action, dict) else action
        # Show the chosen ANSWER TEXT instead of the opaque index.
        if isinstance(_disp, dict) and _disp.get("type") == "answer" and session.get("answer_text"):
            _disp = {"type": "answer", "chose": session["answer_text"]}
        window.set_action(json.dumps(_disp) + "\n\n-> " + json.dumps(result))
        session.pop("answer_text", None)
    if (isinstance(action, dict) and action.get("type") == "move"
            and isinstance(result, dict) and result.get("blocked")):
        d = action.get("dir", "")
        session["last_bump"] = f"Your last move {d} was BLOCKED - something (a wall/obstacle) is that way. Try a different direction or use goto to route around it."
        kb.record_action(f"bumped a wall moving {d}")
    else:
        session.pop("last_bump", None)

    # Log meaningful actions (not routine moves/waits) to the short action
    # history so the agent can avoid repeating itself. Record the OUTCOME, not
    # just the action - a bare list of gotos taught the model nothing (it kept
    # re-searching the same empty body because it never recorded "empty"). Skip
    # routine goto/move so meaningful events (searches, talks, findings) aren't
    # crowded out.
    atype = action.get("type") if isinstance(action, dict) else None
    _ok = result.get("ok") if isinstance(result, dict) else None
    if atype in ("talk", "open", "pickup", "search", "combat", "feed", "take", "read",
                 "equip", "unequip", "drop"):
        p0 = state.get("player") or {}
        detail = action.get("name") or action.get("dir") or ""
        outcome = ""
        if atype == "search":
            if _ok:
                # The engine now LOOTS the body/container and reports exactly
                # what it took ("looted": "gold, bread, torch", "count": N) or
                # "empty": true. Trust that over the stale gump snapshot.
                looted_str = result.get("looted") if isinstance(result, dict) else None
                took_n = result.get("count") if isinstance(result, dict) else None
                is_empty = result.get("empty") if isinstance(result, dict) else None
                if took_n:
                    outcome = f" -> LOOTED {looted_str} ({took_n})"
                elif is_empty or not looted_str:
                    outcome = " -> EMPTY (nothing to take)"
                    # Remember this spot as already-searched-and-empty so the
                    # agent stops returning to it (short-term memory).
                    pp = state.get("player") or {}
                    kb.mark_searched_empty(pp.get("tx", 0), pp.get("ty", 0),
                                           action.get("name") or "body")
                else:
                    outcome = f" -> LOOTED {looted_str}"
            else:
                outcome = " -> nothing to search here"
        elif atype in ("pickup", "take"):
            outcome = f" -> got {result.get('item')}" if _ok else " -> could not take"
        elif atype == "equip":
            outcome = (f" -> equipped {result.get('item') or action.get('name')}"
                       if _ok else " -> could not equip")
        elif atype == "unequip":
            outcome = (f" -> removed {result.get('item') or action.get('name')}"
                       if _ok else " -> could not unequip")
        elif atype == "drop":
            outcome = (f" -> dropped {result.get('item') or action.get('name')}"
                       if _ok else " -> could not drop")
        elif atype == "talk":
            outcome = " (conversed)" if _ok else " -> could not talk"
        elif atype == "read":
            if _ok:
                _txt = (result.get("text") or "").strip() if isinstance(result, dict) else ""
                _tgt = result.get("target", "sign") if isinstance(result, dict) else "sign"
                if _txt:
                    outcome = f" -> reads: \"{_txt[:120]}\""
                    # Durable: record the sign's text as an observation and show
                    # it to the model this turn via last_look.
                    kb.note_observation(f'read {_tgt}: "{_txt[:160]}"', kind="read",
                                        step=step)
                    session["last_look"] = f'The {_tgt} reads: "{_txt}"'
                else:
                    outcome = " -> (sign had no readable text)"
            else:
                outcome = " -> nothing to read here"
        kb.record_action(f"{atype} {detail}".strip()
                         + f" @({p0.get('tx')},{p0.get('ty')})" + outcome)
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
    # Tool-call stats are per-RUN health metrics: reset them so a run's error
    # rates (e.g. 'open') reflect THIS run, not all history. Durable memory
    # (quests, notes, places, plot summary) is preserved by load().
    kb.reset_tool_stats()
    if args.memory_file:
        print(f"[+] Loaded journal: {len(kb.quests)} quests, "
              f"{len(kb.npcs)} NPCs, {len(kb.journal)} notes")
        # Seed the agent's context on load with a brief orientation of where it
        # left off (open quests, known places, what it's learned), so it resumes
        # with continuity instead of re-discovering everything. Shown as the
        # first alert; cleared after a few turns.
        _orient = kb.orientation_summary()
        if _orient:
            session["hint"] = _orient
            session["hint_ttl"] = 3
            print("[+] Seeded orientation summary into context")
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
    ap.add_argument("--think", action="store_true",
                    help="Keep the reasoning model's thinking channel ON (for "
                         "troubleshooting - captured to raw_comms.log). Off by "
                         "default for stability/speed.")
    args = ap.parse_args()

    ollama = None
    if not args.dry_run:
        ollama = OllamaClient(model=args.model, host=args.ollama_host,
                              num_ctx=args.num_ctx, allow_think=args.think)
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
