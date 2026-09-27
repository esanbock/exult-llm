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

# --- Independently live-editable prompt sections -------------------------
# These three constants hold the DEFAULT text for the MISSION, INTERACTION and
# WISDOM sections of the system prompt. They are spliced into SYSTEM_PROMPT in
# their original positions so the rendered prompt is byte-identical to before.
# At runtime each section can be overridden (without restart) via the GUI, which
# writes prompt_sections.json next to this module; see get_effective_system_prompt.
DEFAULT_MISSION = """\
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
    it as a lead worth pursuing. Use your journal to remember what you learned."""

DEFAULT_INTERACTION = """\
INTERACTION RULES (how the world works - know these so you don't waste turns):
  * PROXIMITY: to take, pickup, open, or loot something you must be RIGHT NEXT
    to it (about 1 tile away). If you are farther, first "goto" the target's tile
    (or its dx,dy), THEN act. A failed take will tell you if the item is
    too far and give the tile to goto. To TALK you only need the person in view,
    but being close is more reliable - goto them first.
  * KEYS OPEN LOCKS: locked DOORS and locked CHESTS/containers cannot just be
    opened - you need the right KEY. Keys are items you find (on the ground, on
    bodies, in other containers) - PICK THEM UP and keep them. To open a lock:
    stand next to it and use "unlock" (it tries your keys); if you have the
    matching key it opens, then "open" it. If unlock says no key fits,
    go FIND the key (often near who/what the lock belongs to) and come back. Do
    not abandon a locked chest/door - the key is usually findable.
  * DOORS: '+' = closed (a passage, not a wall - "open" it or "goto" beyond it),
    '/' = open. Locked doors need a key as above.
  * CONTAINERS: chests, desks, drawers, cabinets, bags, barrels, crates, sacks
    are all openable ("open" a specific one to reveal contents; "take" pulls
    items out; "loot" takes all). A slain creature/person's body is openable too.
  * COMBAT: enemies show HOSTILE. "attack" (by name or nearest hostile) engages
    one; "combat" toggles auto-fight. You must be near a foe to hit it (melee)
    or have a ranged weapon. Flee fights you cannot win."""

DEFAULT_WISDOM = """\
RPG PLAYER WISDOM (seasoned-player habits):
  * EXHAUST DIALOGUE: ask every topic (names, jobs, proper nouns) - clues hide in
    skipped topics.
  * EXPLORE EVERYWHERE: enter every building; open doors ('+') - rooms hold
    people, loot, and clues.
  * OPEN AND LOOT: OPEN every container and body, then TAKE what is useful;
    try levers/switches. SELF-CHECK "what_i_have_actually_done": if opened/
    picked-up are ~0 while you keep talking, the answer you need is a PHYSICAL
    thing to find - go open containers and bodies.
  * GATHER USEFUL THINGS: take gold, food, keys, weapons, armour, potions,
    scrolls, reagents, tools - and any ODD item (may be needed for a quest).
  * STEALING HAS CONSEQUENCES: "owned" items are property; taking them if
    witnessed angers people/summons guards. Unowned/loot from the dead is free.
  * FIGHT TO GROW: winnable fights give XP/levels ("combat"/"attack"); flee ones
    that would kill you.
  * MAKE PROGRESS: act purposefully; if you've exhausted a person/place, move on.
  * IF A QUEST STALLS, SWITCH: after many turns with no progress (check
    action_log/turns_since_progress), park it and work a DIFFERENT quest.
  * DON'T LINGER: don't re-interview people (esp. party) with nothing new; once
    local leads are gathered, LEAVE the area to advance the story.
  * ONLY CLAIM WHAT YOU HAVE: never say you hold/know something unless it's in
    your inventory/notes/transcript NOW; answer NPCs only from what you actually
    have - else go get it. Don't guess or invent.
  * RESOLVE ONLY WHEN DONE: mark a quest done ONLY when its concrete goal is
    truly met; if you can't point to the result, keep it open.
  * KEEP YOUR SUMMARY TRUE: plot_summary/notes must match reality - don't record
    a step ("left town", "got X") until it actually happened.
  * CHECK FOR IMPOSSIBLE LOOPS: if a plan needs the thing it's meant to produce,
    it's circular - find what UNLOCKS the blocker first.
  * FINISH THE CHAIN: do multi-step paths A->B->C step by step; don't drift back
    to a blocked action before its prerequisite is met.
  * SURVIVE: "feed" when food is low, heal when hurt; avoid needless danger."""

SYSTEM_PROMPT = """\
You are an autonomous agent playing Ultima VII: The Black Gate as the Avatar.
You interact ONLY through the JSON tool interface described below. You cannot
click, use a mouse, or do anything not listed as a TOOL. Think of this as an
API: the only way to affect the game is to call one TOOL per turn.

""" + DEFAULT_MISSION + """

""" + DEFAULT_INTERACTION + """

""" + DEFAULT_WISDOM + """
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
"<quest id or title>". To REMOVE a redundant, duplicate, or obsolete quest that
you did NOT actually complete, add "drop_quest": "<title or id>" (or a list) -
this is how you CONSOLIDATE and clean up your log.

YOU OWN AND CURATE YOUR QUEST LOG. Whenever you ADD or COMPLETE a quest, briefly
RE-EVALUATE the WHOLE list (it's all in "quests"): merge or drop_quest any
duplicates/near-duplicates, drop goals that no longer make sense, and re-set
priorities so the most important open quest is P1. Keep the list lean and
accurate - a tidy quest log is your plan; a cluttered one wastes your focus.

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
  player: {tx,ty (your absolute tile on the world map), elevation, hp, max_hp, hp_pct, gold, food}
                                GOLD: how much money you have. Check it before
                                agreeing to BUY something - you cannot afford a
                                price higher than your gold (e.g. a 600-gold ship
                                if you have 50). Don't commit to purchases you
                                can't pay for.
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
                                COORDINATES: everything uses ONE ABSOLUTE frame,
                                in 3D. Your tile is (tx,ty,tz). Every nearby
                                person/object also gives ABSOLUTE (tx,ty,tz) plus
                                a short "dir" hint (e.g. "NE 5"). tz is the LEVEL
                                (0=ground, >0=up, <0=underground): something with
                                the SAME tz as you is on your level and reachable;
                                a different tz means you must change levels first.
                                To walk to something on your level, "goto" its
                                (tx,ty) directly. north=up, east=right.
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
                                each gives ABSOLUTE {tx,ty} + a "dir" hint; goto
                                its (tx,ty) to reach it.
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
  objects (list)              - ground items: {name, tx, ty (ABSOLUTE), dir}.
                                goto (tx,ty) to reach. "owned":true = someone's
                                property (taking = stealing). "body":true =
                                LOOTABLE (open it); "corpse_not_lootable":true
                                = empty corpse (nothing inside). "town_exit":true =
                                the gate OUT of town.
  doors (list)                - nearby doors: {name, dx, dy, closed}

# YOUR JOURNAL (you maintain this - it persists across turns)
  quests: {open[] (priority-sorted, 1=highest), recently_resolved[], resolved,
      unresolved}. Your plan - choose which to work on; RESOLVE the moment a goal
      is truly done. Each: {id, title, priority, status, npc, depends_on[],
      prereqs_unmet[]}. depends_on/prereqs_unmet are ordering INFO, not a block.
      A quest with "npc" is pursued by going to/talking to that NPC.
  npc_notes: your always-shown notes per NPC (their leads/wants). REVIEW before
      (re)talking someone. If long, CONSOLIDATE via
      "consolidate_notes":{"npc":"<name>","notes":"<merged>"}. "recall" gives an
      NPC's full transcript + a "dialogue_tree" (conversations are a TREE:
      each node has reached_by_asking / open_here / already_asked_here; to ask a
      subtopic, first ask its parent).
  known_places: your MENTAL MAP (landmarks, buildings, containers, NPCs you met),
      nearest first, each {name, kind, dx, dy}. "goto" any BY NAME to return.
  known_topics: your notebook of subjects (each {topic, notes}); authored via the
      "topic" field. "recall" a topic to re-read your notes on it.
  recent_dialogue: transcript of the last exchanges ({"npc","said"} / {"me"}).
  action_log: YOUR TEMPORAL MEMORY - recent actions, one per line as
      "[T<turn>] <action> -> <outcome>  (I intended: \"<reason>\")", repeats
      collapsed like "[T88-T130] move toward X x22 (NO progress)". If the same
      action/reason repeats with no result, you are LOOPING - do something else.
      CRITICAL: "(I intended: ...)" is only what you CLAIMED, NOT proof it
      happened. Only the "-> outcome" (and your inventory) is real - never treat
      a past intention as a done fact.
  current_quest: the quest you set via set_current_quest; if it stalls, switch.
  already_searched_empty (list): bodies/containers already searched empty - skip.
  story_so_far: your running plot summary (via "plot_summary") - big picture,
      key clues, current objective. Keep it accurate.
  operator_hints: standing guidance from your operator (newest last) - honor them.
  observed: notable things SEEN/OVERHEARD (deduped), e.g. "seen: chest at (x,y)".

# TOOLS (the complete list of things you can do - nothing else is possible)
  move    - Walk using GAME TILES (never screen coords). params: EITHER
            {"dir": one of n,s,e,w,ne,nw,se,sw, "steps": <n, default 1>} to walk
            n tiles that compass way, OR {"tx": <x>, "ty": <y>} to walk to an
            absolute game tile (the engine pathfinds there).
            Use the room description + objects/nearby lists to choose a
            direction or tile. To reach an NPC/object, move toward its (dx,dy)
            or use its absolute {tx,ty}.
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
  descend - Get DOWN off an elevated surface (wall/roof/stairs) to the ground.
            params: none. Use when your elevation (tz) is >0; walks to the
            nearest reachable lower ground. (already_ground = you're at tz 0.)
  talk    - START a conversation with a nearby NPC. params: {"name":"<NPC>"}.
            The only way to begin dialog. Prefer to be CLOSE first. NPC must be
            AWAKE (sleeping ones need wait_until morning). Party = nothing new.
  open    - Open a specific DOOR, CONTAINER, or BODY. params: {"name":"<chest/
            bag/barrel/desk/body/door>"} or {"tx":X,"ty":Y} (or omit to open the
            nearest door). A door toggles open/closed. A container/body opens to
            REVEAL its contents (returned as "contents": [...]) - it does NOT
            take anything. Stand within a few tiles. THEN use "take <item>" to
            pull out what you want. '+' closed door / '/' open on the map.
  loot    - Take ALL items from a container/body at once. params: {"name":...}
            or {"tx,ty"} or nearest. Convenience only - PREFER "take <item>" to
            pick specific things deliberately; use loot to empty a chest fast.
  close   - Close an open container/body gump. params: none. Do this to move again.
  unlock  - Use your KEYS on the nearest locked door/container. params: none.
            Stand adjacent. Works only if you carry a matching key; else find
            the key first. Then "open" it.
  use     - INTERACT with a world object OR a carried item (the generic
            double-click). params: {"name":"<object>"} or nearest. Stand next
            to a world object first. Covers: well, winch/lever/switch (gates/
            bridges/puzzles), sextant (your coords), carriage/boat (board), bed
            (sleep), plaque/book, moongate; and carried potions/scrolls/tools.
  read    - Read a nearby SIGN/readable object OR a book/scroll/document you are
            CARRYING. params: omit=nearest sign, or {"name":"<obj>"} (matches a
            world object or an item in your pack). Returns "text". Use this to
            read documents/books/letters you looted.
  take    - Take an item from a nearby body/container (also grabs a LOOSE nearby
            item if not in a container). params: {"name":"<item>"} or omit=first.
  pickup  - Take a LOOSE nearby item (ground/table/shelf). params:
            {"name":"<item>"} optional. (take and pickup both work for loose
            items; take also reaches inside containers.)
  give    - Hand a carried ITEM to a nearby PERSON (may advance a quest). params:
            {"item":"<item>","to":"<npc>"} ("to" optional=nearest). Stand next
            to them. Distinct from "drop" (puts item on the ground).
  inventory - Report worn/carried items. params: none. (Rarely needed - already
            shown each turn in player.worn / player.carrying.)
  annotate - Mark a spot on your map to return to. params: {"label":"<name>"}
            (current pos) or add {"tx","ty"} and optional {"note"}; then "goto"
            it by name.
  equip   - Wear/wield an item (auto-placed in its slot). params: {"name":"<item>"}.
  unequip - Take a worn item off, back into your pack. params: {"name":"<item>"}.
  drop    - Drop a carried/worn item at your feet. params: {"name":"<item>"}.
  look    - Detailed description of your surroundings (people, ground items,
            doors, terrain). params: none. Use on entering a new area.
  map     - Town-scale overview: your position, explored area (fog-of-war), and
            labeled landmarks. params: none. Orient toward unexplored areas or
            known landmarks.
  recall  - Retrieve your full saved knowledge about a CHARACTER or TOPIC.
            params: {"name":"<character or topic>"} (fuzzy). Person = transcript
            + asked/unasked topics; topic = every line said about it + who said
            it. Shown next turn as "recalled". Use before (re)talking someone.
  quests  - Review your full quest log (notes/prereqs). params: none, or
            {"finished": true} to review COMPLETED quests. Shown as "quest_detail".
  answer  - Reply during a conversation. params: {"index":<int>} (into "answers")
            or {"text":"..."}. Only when conversation_active.
            *** When answer choices are shown (conversation_active / the
            "answers" list is non-empty) you MUST reply: pick an "answer" index
            (or "continue" if the NPC is still talking). Do NOT "wait", "move",
            "goto", or "think" while a conversation is open - those do nothing
            and waste the turn. Choose the topic that best advances your goal;
            if none help, pick a "bye/leave" option to end the conversation. ***
  set_number - Answer a numeric slider. params: {"value":<int>} (clamped to
            number_min..number_max). Only when number_prompt.
  continue - Advance an NPC's speech to the next page (npc_text showing but not
            conversation_active). params: none.
  dismiss - Close/cancel a menu, sign, or popup. params: none.
  combat  - Toggle auto-fight mode on/off. params: none.
  attack  - Focus-attack a creature: params: {"name":"<creature>"} or omit =
            nearest HOSTILE. For monsters OUTSIDE town; do NOT attack townsfolk.
  set_combat_mode - How you fight. params: {"mode":
            "nearest"|"weakest"|"strongest"|"berserk"|"defend"|"flank"|"flee"|
            "protect"|"random"|"manual"} ("flee" to retreat, "defend" when hurt).
  feed    - Eat food to refill your food level (prevents starving). params: none.
  heal    - Use a bandage from your pack to restore HP when hurt. params: none.
            Do this when your hp is well below max_hp and you are safe (not mid-
            fight if avoidable). Needs a bandage in the party's inventory.
  combat_pause - Pause the current combat (stop auto-fighting momentarily).
            params: none. Rarely needed; use "combat" to toggle combat off.
  save    - Save the game so progress is not lost. params: none. (The driver
            also auto-saves periodically; you rarely need this.)
  wait    - Do nothing this turn. params: none. NOT valid during a conversation
            (when answer choices are shown) - use "answer"/"continue" instead.
  wait_until - Pass time until a target hour (0-23), e.g. wait for morning so
            sleeping NPCs wake. params: {"hour": 7} (default 7 = morning). Use
            this when the people you need are "sleeping" and it is night.

# JOURNAL TOOLS (manage your own quest log & notes - do NOT affect the game)
#
# QUEST vs SUB-TASK - organise your goals into TWO levels (this keeps your log
# readable and stops it filling with dozens of tiny near-duplicate entries):
#   * A QUEST is a BIG objective - a whole storyline or mission, e.g.
#     "Solve the Trinsic murder", "Reach Britain", "Clear the dungeon". You
#     should have only a FEW open quests at once (roughly 1-5).
#   * A SUB-TASK is a concrete STEP toward a quest, e.g. "search the body",
#     "read the sign", "ask Petre about the murder", "find the key". Record a
#     sub-task as a quest whose depends_on = ["<parent quest id>"], so it nests
#     UNDER its quest instead of becoming another top-level goal.
# Before adding a quest, check the open list: if your new goal is a STEP toward
# an existing quest, add it as a sub-task of that quest (or just DO it) - do NOT
# create a second top-level quest for the same objective. Reserve add_quest
# top-level entries for genuinely NEW big objectives.
  add_quest    - Record a goal. params: {"title": "...", "priority": 1-9
                 (1=highest), "notes": "...", "depends_on": ["<parent quest
                 id>", ...]}. For a SUB-TASK, set depends_on to the parent
                 quest's id (its title lower-cased with underscores). For a new
                 big QUEST, omit depends_on. Don't log every micro-intention as
                 its own top-level quest - nest steps under their quest.
  update_quest - Change a quest. params: {"id": "<quest id>", "status":
                 "active|blocked|done", "priority": n, "notes": "...",
                 "depends_on": [...]}. Mark a quest/sub-task "done" when you
                 complete it, and add notes as you learn more about it.
  note_npc     - Save a note about an NPC. params: {"name": "...", "note": "..."}.
                 Record leads, what they want, or what they told you.
  add_topic    - Record a TOPIC/subject you're tracking across the game (a
                 place, person, mystery, or clue), with a note. params:
                 {"topic": "...", "note": "..."}. Use this to build your
                 evolving understanding of a subject (e.g. topic "the murder"
                 with notes as you learn more). Topics persist and group related
                 clues, unlike per-NPC notes.
  (These journal tools do not advance the game. Prefer capturing a quest the
   moment you decide on a goal - a well-kept quest log is how you stay strategic
   across many turns. After using one, take a game action the same or next turn.)

# HOW TO DECIDE (policy)
  1. If conversation_active -> "answer". Ask EVERY topic in
     "not_yet_asked_this_npc" (esp. name, job, any proper noun) - asking often
     unlocks new topics; don't leave with useful topics unasked; skip
     "already_asked_this_npc"; "bye" only when nothing useful remains.
  2. Else if conversation_in_progress (open, no choices yet) -> "continue".
     Don't "talk"/"move" during a conversation.
  3. Your QUEST LOG is your plan - own it. Each turn consult "quests", pick the
     most important open one, act on it (talk/goto its npc, or act on its notes).
     RESOLVE a quest only when TRULY achieved (item in hand, door open, person
     dead) via "resolve_quest" - deciding HOW to pursue is NOT completing it;
     use drop_quest if it becomes impossible/irrelevant. Record lasting insights
     in your topic notebook so they persist after a quest closes.
  3a. Talk to NEW people (status "new", then "talked") to discover quests.
     Before re-approaching someone, "recall" them to see asked/open topics;
     re-talk an "exhausted" NPC only with a genuine new reason.
  4. If no actionable quest and no new NPC -> explore a NEW area. Travel via
     "goto" (pathfinds around walls/doors); if stranded far away, goto a known
     town/place. Use "move" only for tiny steps; never onto '#'.
  4b. Interact with relevant "objects": goto then OPEN bodies+containers, take/pickup
     useful items - don't pace past them.
  5. Low food -> "feed". Threatened -> "combat".

OUTPUT RULES (critical - follow exactly):
- Output ONLY one JSON object. Start your reply with '{' as the very first
  character. No preamble, no thinking, no explanation, no markdown, no code
  fences before or after.
- Keep it short: the object is just {"action":{...},"reason":"..."} (plus an
  optional "new_quest"). Do NOT write anything outside the JSON.
"""


# --- Live-editable operator guidance -------------------------------------
# Operators can add/tweak GUIDANCE at runtime WITHOUT restarting: whatever is in
# prompt_extra.txt (next to this module, or set via the GUI "Edit prompt"
# window) is appended to the system prompt each turn. We re-read it only when
# the file's mtime changes, so it's cheap. The core protocol/schema/tool docs
# stay in SYSTEM_PROMPT (immutable) so live edits can never break JSON parsing.
_PROMPT_EXTRA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "prompt_extra.txt")
_prompt_extra_cache = {"mtime": None, "text": ""}

# Per-section OVERRIDES: operators can replace the MISSION / INTERACTION / WISDOM
# sections of SYSTEM_PROMPT at runtime WITHOUT restarting by editing (via the
# GUI) prompt_sections.json next to this module. Shape:
#   {"mission": "...", "interaction": "...", "wisdom": "..."}
# Any present key replaces that section's default text; absent keys keep the
# DEFAULT_ constant. Like prompt_extra, we re-read only when the file's mtime
# changes, so it's cheap. This is applied on top of SYSTEM_PROMPT, and THEN the
# prompt_extra.txt operator-guidance block is appended (as before).
_PROMPT_SECTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "prompt_sections.json")
_prompt_sections_cache = {"mtime": None, "data": {}}
# Sentinel stored in the cache's mtime to FORCE a reload on next access. It can
# never equal a real mtime (a float) nor None (the "file absent" marker), so a
# reload always happens even when the file was just deleted.
_FORCE_RELOAD = object()

# Maps section name -> (DEFAULT_ constant). Order matters only for tidiness.
_SECTION_DEFAULTS = {
    "mission": DEFAULT_MISSION,
    "interaction": DEFAULT_INTERACTION,
    "wisdom": DEFAULT_WISDOM,
}


def _load_prompt_sections() -> dict:
    """Return the current per-section overrides dict (cached by mtime)."""
    try:
        mt = os.path.getmtime(_PROMPT_SECTIONS_PATH)
    except OSError:
        mt = None
    if mt != _prompt_sections_cache["mtime"]:
        _prompt_sections_cache["mtime"] = mt
        data = {}
        if mt is not None:
            try:
                with open(_PROMPT_SECTIONS_PATH, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    # Keep only known sections with non-empty string values.
                    for name in _SECTION_DEFAULTS:
                        val = loaded.get(name)
                        if isinstance(val, str) and val.strip():
                            data[name] = val
            except (OSError, ValueError):
                data = {}
        _prompt_sections_cache["data"] = data
    return _prompt_sections_cache["data"]


def get_prompt_section(name: str) -> str:
    """Current text for 'mission'|'interaction'|'wisdom' (override if set)."""
    if name not in _SECTION_DEFAULTS:
        raise KeyError(f"unknown prompt section: {name!r}")
    override = _load_prompt_sections().get(name)
    return override if isinstance(override, str) and override.strip() \
        else _SECTION_DEFAULTS[name]


def get_prompt_section_default(name: str) -> str:
    """The DEFAULT_ constant for a section (for GUI 'revert to default')."""
    if name not in _SECTION_DEFAULTS:
        raise KeyError(f"unknown prompt section: {name!r}")
    return _SECTION_DEFAULTS[name]


def set_prompt_section(name: str, text: "str | None") -> None:
    """Set (or clear) a section override in prompt_sections.json.

    text=None or empty/whitespace REMOVES the override (revert to default).
    Robust if the file does not exist or is unreadable. The driver picks up the
    change automatically on the next turn via the mtime cache.
    """
    if name not in _SECTION_DEFAULTS:
        raise KeyError(f"unknown prompt section: {name!r}")
    # Read whatever is currently on disk (ignore corruption).
    data = {}
    try:
        with open(_PROMPT_SECTIONS_PATH, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        data = {}
    if text is None or not str(text).strip():
        data.pop(name, None)
    else:
        data[name] = str(text).strip()
    try:
        if data:
            with open(_PROMPT_SECTIONS_PATH, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        else:
            # Nothing left to override - remove the file entirely.
            try:
                os.remove(_PROMPT_SECTIONS_PATH)
            except OSError:
                pass
    except OSError:
        pass
    # Invalidate cache so the change is visible immediately.
    _prompt_sections_cache["mtime"] = _FORCE_RELOAD


def get_effective_system_prompt() -> str:
    """SYSTEM_PROMPT with any live per-section overrides applied, then any
    operator guidance from prompt_extra.txt appended (both live)."""
    # 1) Apply per-section overrides by substituting each section's DEFAULT_
    #    text (which appears exactly once) with its override. With no overrides
    #    present this is a no-op, so the result is byte-identical to SYSTEM_PROMPT.
    prompt = SYSTEM_PROMPT
    overrides = _load_prompt_sections()
    for name, default_text in _SECTION_DEFAULTS.items():
        override = overrides.get(name)
        if isinstance(override, str) and override.strip():
            prompt = prompt.replace(default_text, override, 1)

    # 2) Append operator guidance from prompt_extra.txt (unchanged behavior).
    try:
        mt = os.path.getmtime(_PROMPT_EXTRA_PATH)
    except OSError:
        mt = None
    if mt != _prompt_extra_cache["mtime"]:
        _prompt_extra_cache["mtime"] = mt
        try:
            _prompt_extra_cache["text"] = (
                open(_PROMPT_EXTRA_PATH, encoding="utf-8").read().strip()
                if mt is not None else "")
        except OSError:
            _prompt_extra_cache["text"] = ""
    extra = _prompt_extra_cache["text"]
    if extra:
        return (prompt
                + "\n\n# OPERATOR GUIDANCE (added live - honor these)\n" + extra)
    return prompt


def _action_phrase(action) -> str:
    """Render an action dict as a short human-readable phrase for the GUI turn
    log (the raw JSON + result is noise there; the LLM still gets the result
    in-game). E.g. {'type':'goto','name':'stables'} -> 'goto stables'."""
    if not isinstance(action, dict):
        return str(action)
    t = action.get("type", "?")
    # Some models double-nest ({"type":"action","action":"move",...}) or wrap a
    # target ({"target":{"x":..,"y":..}}). Normalize common cases so the Action
    # column shows a phrase, not a raw dict.
    if t in ("action", "tool", "command") and isinstance(action.get("action"), str):
        t = action["action"]
    _tgt = action.get("target")
    if isinstance(_tgt, dict) and action.get("tx") is None:
        if _tgt.get("x") is not None:
            action = dict(action, tx=_tgt.get("x"), ty=_tgt.get("y"))
        elif _tgt.get("name"):
            action = dict(action, name=_tgt.get("name"))
    if t == "move":
        return f"move {action.get('dir', '')}".strip()
    if t == "goto":
        if action.get("name"):
            return f"goto {action['name']}"
        if action.get("tx") is not None:
            z = action.get("tz")
            return f"goto ({action.get('tx')},{action.get('ty')}" + (f",z{z})" if z is not None else ")")
        return "goto"
    if t == "talk":
        return f"talk to {action.get('name', '?')}"
    if t == "answer":
        return (f"say \"{action['text']}\"" if action.get("text")
                else f"answer #{action.get('index', '?')}")
    if t in ("take", "pickup", "use", "equip", "unequip", "drop", "attack"):
        return f"{t} {action.get('name', '')}".strip()
    if t == "give":
        return f"give {action.get('item', '')} to {action.get('to', 'someone')}".strip()
    if t == "set_number":
        return f"set number {action.get('value', '')}"
    # search/open/close/read/unlock/continue/dismiss/wait/look/map/recall/etc.
    _n = action.get("name")
    return f"{t} {_n}" if _n else t


def _dir_word(dx: int, dy: int) -> str:
    """Compass word for a tile delta (north = -y). '' if on top of it."""
    ns = "north" if dy < 0 else ("south" if dy > 0 else "")
    ew = "east" if dx > 0 else ("west" if dx < 0 else "")
    return (ns + ew) or "right here"


# Mass nouns / already-plural that read wrong with "A " (a hay -> just "hay").
_NO_ARTICLE = ("hay", "straw", "water", "gold", "grass", "sand", "equipment",
               "furniture", "clothing", "food")


def _article(name: str) -> str:
    """Return the object phrase with a correct article: 'a chest', 'an anvil',
    'hay' (mass noun, no article)."""
    low = name.lower()
    if any(low == m or low.startswith(m + " ") or low.endswith(" " + m)
           for m in _NO_ARTICLE):
        return name
    art = "an" if low[:1] in "aeiou" else "a"
    return f"{art} {name}"


def describe_room(state: dict, place: str = "") -> str:
    """Translate the visual environment into Zork-style prose, which LLMs read
    far more fluently than a flat coordinate table. Composes: where you are,
    the people present, and the notable/ACTIONABLE objects with their state
    (a door 'to the west, closed'; a body 'to the south, searchable'). Grounded
    entirely in real state fields (name, dx/dy, closed/body/container/owned/
    town_exit) - it never invents anything. Scenery is de-prioritized in favor
    of things you can act on. Returns a short paragraph.
    """
    p = state.get("player") or {}
    px, py = p.get("tx", 0), p.get("ty", 0)
    objs = state.get("objects") or []
    nearby = state.get("nearby") or []
    doors = state.get("doors") or []
    sentences: list = []

    # 1) Where you are.
    if place:
        sentences.append(f"You are in {place}.")
    elif (p.get("tz", 0) or 0) > 0:
        sentences.append("You are up high, on a wall-top or upper floor.")
    else:
        sentences.append("You are outdoors." if not objs else "You look around.")

    def near(it, r=10):
        return abs(it.get("dx", 99)) + abs(it.get("dy", 99)) <= r

    # 2) People (living, non-party first).
    people = [n for n in nearby if n.get("name") and not n.get("dead")
              and not n.get("in_party") and near(n, 12)]
    people.sort(key=lambda n: abs(n.get("dx", 0)) + abs(n.get("dy", 0)))
    for n in people[:3]:
        d = _dir_word(n.get("dx", 0), n.get("dy", 0))
        cond = ""
        if n.get("condition") == "sleeping":
            cond = ", asleep"
        sentences.append(f"{n['name'].capitalize()} is to the {d}{cond}."
                         if d != "right here" else f"{n['name'].capitalize()} is right beside you.")

    # 3) Actionable objects: doors, bodies, containers, town exits, loose items.
    #    De-dup by name+dir so repeated scenery (walls) doesn't flood.
    seen_desc = set()
    def add_obj(desc):
        if desc not in seen_desc:
            seen_desc.add(desc)
            sentences.append(desc)

    # doors (from the dedicated doors list, which carries 'closed')
    for dr in sorted(doors, key=lambda d: abs(d.get("dx", 0)) + abs(d.get("dy", 0)))[:2]:
        if near(dr, 8):
            d = _dir_word(dr.get("dx", 0), dr.get("dy", 0))
            st = "closed" if dr.get("closed") else "open"
            add_obj(f"A door leads {d}; it is {st}.")

    bodies = [o for o in objs if o.get("body") and near(o, 10)]
    for o in sorted(bodies, key=lambda o: abs(o.get("dx", 0)) + abs(o.get("dy", 0)))[:2]:
        d = _dir_word(o.get("dx", 0), o.get("dy", 0))
        add_obj(f"{_article(o.get('name','body')).capitalize()} lies to the {d} - it can be searched.")

    containers = [o for o in objs if o.get("container") and not o.get("body") and near(o, 8)]
    for o in sorted(containers, key=lambda o: abs(o.get("dx", 0)) + abs(o.get("dy", 0)))[:2]:
        d = _dir_word(o.get("dx", 0), o.get("dy", 0))
        own = " (owned - taking is theft)" if o.get("owned") else ""
        add_obj(f"{_article(o.get('name','container')).capitalize()} sits to the {d}{own}.")

    exits = [o for o in objs if o.get("town_exit") and near(o, 16)]
    for o in sorted(exits, key=lambda o: abs(o.get("dx", 0)) + abs(o.get("dy", 0)))[:2]:
        d = _dir_word(o.get("dx", 0), o.get("dy", 0))
        add_obj(f"A way out of town lies to the {d}.")

    # A couple of loose items worth grabbing (not owned, not scenery-ish).
    _scenery = ("wall", "tree", "fence", "roof", "floor", "window", "shutters",
                "curtain", "grass", "post", "pillar")
    items = [o for o in objs if near(o, 8) and not o.get("owned")
             and not o.get("body") and not o.get("container")
             and o.get("name") and not any(s in o.get("name", "").lower() for s in _scenery)]
    for o in sorted(items, key=lambda o: abs(o.get("dx", 0)) + abs(o.get("dy", 0)))[:3]:
        d = _dir_word(o.get("dx", 0), o.get("dy", 0))
        add_obj(f"{_article(o['name']).capitalize()} is to the {d}.")

    return " ".join(sentences)


def _recognize_place(objects: list, nearby: list) -> str:
    """Give the LLM the 'sense of place' a human gets at a glance. The engine
    sends a flat object list with coordinates but no synthesis, so the model
    can't tell it is standing IN a stable (hay, stalls, pitchfork, a horse) vs
    a house (bed, table, hearth). Detect signature object/creature clusters
    among what is CLOSE (within ~8 tiles) and name the location, so the agent
    knows it has arrived and can stop re-navigating to a place it is already in.
    Returns a short phrase like "the STABLES" or "a dwelling", or "" if unclear.
    """
    def close(items):
        names = []
        for it in items or []:
            if abs(it.get("dx", 99)) + abs(it.get("dy", 99)) <= 8:
                nm = (it.get("name") or "").lower()
                if nm:
                    names.append(nm)
        return names
    names = close(objects) + close(nearby)
    blob = " ".join(names)
    def has(*words):
        return sum(1 for w in words if w in blob)
    # Signature clusters (need >=2 signals to avoid false positives).
    if has("hay", "stall", "pitchfork", "trough", "horseshoe", "horse", "manger") >= 2:
        return "the STABLES (hay/stalls/horse tack around you)"
    if has("anvil", "forge", "bellows", "tongs", "smith", "furnace") >= 2:
        return "a SMITHY / forge (anvil/tongs/forge)"
    if has("altar", "pew", "candelabra", "shrine", "reliquary") >= 2:
        return "a TEMPLE / shrine"
    if has("counter", "barrel", "keg", "mug", "tankard", "bottle", "bar") >= 2:
        return "a TAVERN / inn (bar, kegs, mugs)"
    if has("easel", "artist", "paint", "canvas") >= 2:
        return "an ARTIST'S studio"
    if has("bookshelf", "book", "scroll", "desk", "lectern") >= 3:
        return "a LIBRARY / study"
    if has("bed", "table", "chair", "hearth", "chest", "cupboard", "shelf") >= 2:
        return "inside a DWELLING (a home/room)"
    return ""


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
            "tx": p.get("tx"), "ty": p.get("ty"), "tz": p.get("tz", 0),
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
            "gold": p.get("gold"),
            "food": p.get("food"),
            "dead": p.get("dead"),
            # ALWAYS-ON inventory so the agent knows what it holds without
            # spending a turn on the 'inventory' tool (a human always sees this).
            "worn": p.get("worn") or {},
            "carrying": p.get("carrying") or [],
        },
        "time_of_day": state.get("time_of_day"),
        "hour": state.get("hour"),
        "is_night": state.get("is_night"),
        "in_combat": state.get("in_combat"),
        # Synthesized "sense of place" from nearby object clusters (a human sees
        # hay+stalls+horse and knows it's a stable; the flat object list doesn't
        # convey that). Only when NOT mid-conversation, to avoid clutter.
        **({"you_appear_to_be_in": _recognize_place(objects, nearby)}
           if (not in_convo and _recognize_place(objects, nearby)) else {}),
        # Zork-style narrated room description (LLMs read prose far better than a
        # coordinate table): where you are + people + actionable objects and
        # their state. Grounded in real fields; scenery de-prioritized.
        **({"room_description": describe_room(state, _recognize_place(objects, nearby))}
           if not in_convo else {}),
        "conversation_in_progress": in_convo,
        "conversation_active": state.get("conversation_active"),
        "npc_text": state.get("npc_text") if in_convo else None,
        "answers": state.get("answers") if in_convo else [],
        "ambient_speech": state.get("ambient_speech") or [],
        "nearby": [
            {"name": n.get("name"),
             # ABSOLUTE tile (same frame as your position and goto targets), plus
             # a short compass hint. Use tx,ty directly with goto.
             "tx": (n.get("tx") if n.get("tx") is not None
                    else (p.get("tx", 0) + n.get("dx", 0))),
             "ty": (n.get("ty") if n.get("ty") is not None
                    else (p.get("ty", 0) + n.get("dy", 0))),
             "tz": (p.get("tz", 0) or 0) + (n.get("dz", 0) or 0),
             "dir": _compass(n.get("dx", 0), n.get("dy", 0)),
             "status": (kb.talk_status(n.get("name")) if kb and n.get("name") else "new"),
             **({"HOSTILE": True} if n.get("hostile") else {}),
             **({"condition": n["condition"]} if n.get("condition") else {})}
            for n in nearby[:near_n] if not n.get("in_party")
        ],
        # Party companions are shown separately - they follow you and have no
        # new information, so do NOT "talk" to them to investigate.
        "party": [n.get("name") for n in nearby if n.get("in_party") and n.get("name")],
        "objects": [
            {"name": o.get("name"),
             "tx": p.get("tx", 0) + o.get("dx", 0),
             "ty": p.get("ty", 0) + o.get("dy", 0),
             "tz": (p.get("tz", 0) or 0) + (o.get("dz", 0) or 0),
             "dir": _compass(o.get("dx", 0), o.get("dy", 0)),
             **({"body": True} if o.get("body") else {}),
             **({"searchable_container": True} if o.get("container") else {}),
             **({"contents": o.get("contents")} if o.get("contents") else {}),
             **({"corpse_not_lootable": True} if o.get("corpse") else {}),
             **({"owned": True} if o.get("owned") else {}),
             **({"town_exit": True} if o.get("town_exit") else {})}
            for o in objects[:obj_n]
        ],
        # NOTE: the ASCII "grid"/"grid_legend" are intentionally NOT sent to
        # the model. LLMs read a top-down ASCII map poorly, and it duplicated
        # the (much better) prose room_description + the structured objects/
        # nearby lists - three representations of the same space bloated the
        # prompt and added confusion. The raw state["grid"] is still used by the
        # wedge-escape logic and the inspector map; we just don't burden the
        # model's prompt with it.
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
        # If the stall is navigational, point at the always-on 'navigation'
        # block: goto a known place BY NAME rather than walking toward stairs/
        # walls. (The full landmark list with bearings is always in context.)
        view["navigation_reminder"] = (
            "If you are stuck trying to REACH somewhere: see the 'navigation' "
            "block for known places with bearings/distances and goto one BY "
            "NAME. Do NOT climb stairs/walls to reach a building - walk to it on "
            "the ground. If your target is not a known place yet, EXPLORE a "
            "direction you have not been (blank areas on the map).")
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
            "same attempt. ALSO RE-CHECK YOUR ASSUMPTIONS: if you are blocked "
            "waiting on a prerequisite, VERIFY you actually have it - review your "
            "FINISHED quests (quests finished:true). A quest you marked DONE may "
            "NOT truly be complete (e.g. you closed 'get the password' but never "
            "actually received it) - if so, that goal is still OPEN; go finish it "
            "for real. Also re-read your NPC notes: the person who blocks you "
            "usually TOLD you exactly what you still need and WHO gives it.")
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
    # Correct a STALE plot summary: if the agent's own summary says it's on the
    # wall / elevated / needs to descend, but it is ACTUALLY at ground (tz 0),
    # flag the contradiction so it rewrites its summary and stops "finding stairs
    # down". (It authored a true fact earlier and never updated it once it came
    # down - a false belief that keeps it hunting a non-existent descent.)
    if kb is not None and (p.get("tz", 0) or 0) == 0:
        _sl = (kb.episodic_summary or "").lower()
        if any(w in _sl for w in ("on the wall", "on fortress wall", "on the fortress wall",
                                  "on fortress roof", "on the roof", "fortress roof",
                                  "on the wall", "rooftop", "upper floor",
                                  "elev 1", "elev 2", "elev 3", "elev 4", "elev 5",
                                  "elevated", "descend to ground", "must descend",
                                  "find stairs down", "stairs down", "stairs down are",
                                  "descend to the ground", "get down", "climb down")):
            view["ELEVATION_CORRECTION"] = (
                "IMPORTANT: your plot summary says you are UP HIGH (on a wall/"
                "roof/upper floor) and need to DESCEND - but you are ALREADY at "
                "GROUND LEVEL (tz 0) now. That part of your summary is STALE and "
                "is making you waste turns 'descending'. REWRITE your plot_summary "
                "to remove any 'on the roof/wall / must descend / stairs down' "
                "text, and pursue your goal at ground level directly.")
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
        # Operator/viewer HINTS, but AGE-BOUNDED so chat guidance scrolls out
        # with the turn-memory window instead of accumulating forever. Anything
        # important enough to keep became a QUEST (persists in the quest log).
        _hint_age = state.get("_turn_window_override") or 300
        rh = kb.recent_hints_within(_hint_age, kb.turn_counter, limit=5)
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
        # NPC NOTES always in context (compact): the agent wasn't reviewing its
        # notes via the recall tool, just accumulating them. Surface them so it
        # SEES what it has learned about each person and can consolidate. Fewer
        # under context pressure.
        _nn = kb.npc_notes_view(limit_npcs=(8 if lvl >= 2 else 12),
                                notes_each=(4 if lvl >= 2 else 6))
        if _nn:
            view["npc_notes"] = _nn
        # Spots already searched and empty - stated IMPERATIVELY so the agent
        # stops returning to the same looted body/container/bag.
        se = kb.searched_empty_view(10)
        if se:
            view["already_searched_empty_DO_NOT_RETURN"] = se
            view["already_searched_note"] = (
                f"You have already opened {len(se)} spot(s) and found them EMPTY "
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
        # Long temporal memory - the agent's main loop-perception tool. Default
        # wide (600) since runs sit at ~55-60% context; shrink under squeeze so
        # we stay safe if the prompt ever grows toward the limit.
        _alog_n = 450 if lvl == 0 else (300 if lvl == 1 else 180)
        # Operator override from the GUI 'Turn memory' box (if set). Still capped
        # down under real context pressure (squeeze) so we never blow the limit.
        _ov = state.get("_turn_window_override")
        if _ov:
            _alog_n = _ov if lvl == 0 else min(_ov, _alog_n)
        ah = kb.action_view(_alog_n)
        # Factual digest of turns that have scrolled PAST the shown window, so
        # continuity survives independent of whether the LLM kept its
        # plot_summary current. Placed in the durable top-context (not at the
        # bottom with the raw log). Deterministic - counts + notable events.
        try:
            _digest = kb.action_digest(_alog_n)
            if _digest:
                view["earlier_this_session"] = _digest
        except Exception:
            pass
        # NOTE: action_log is intentionally added LAST (just before return) so it
        # is the final thing in the context - it is the continuously-appended
        # temporal memory and belongs at the bottom for readability + recency.
        if kb.current_quest:
            view["current_quest"] = kb.current_quest
        # ALWAYS-ON navigation summary: structured, text-first list of known
        # landmarks with tile, compass bearing, and distance from you, nearest
        # first. LLMs reason over this far better than an ASCII map, and having
        # it always present means the agent does NOT need to call the map tool
        # just to orient. goto any of these BY NAME.
        try:
            _pp_nav = state.get("player") or {}
            _nav = kb.navigation_summary(_pp_nav.get("tx", 0), _pp_nav.get("ty", 0))
            if _nav:
                # PROACTIVE arrival hint: name any known landmark(s) the agent is
                # already ON/next to (dist <= 2), so it sees "you're already here"
                # BEFORE deciding - and stops re-issuing goto to a place it's
                # standing in (the #1 wasted-turn loop). General: works for any
                # landmark in any quest.
                _at_now = [n["name"] for n in _nav
                           if abs(n.get("dx", 9)) + abs(n.get("dy", 9)) <= 2 and n.get("name")]
                nav = {
                    "you_are_at": [_pp_nav.get("tx", 0), _pp_nav.get("ty", 0)],
                    "known_places_nearest_first": _nav,
                    "how_to_use": ("goto any place BY NAME (e.g. {\"type\":\"goto\","
                                   "\"name\":\"stables\"}) - it pathfinds there. "
                                   "A building is reached on the GROUND, not by "
                                   "climbing stairs. Use the 'map' tool only for a "
                                   "visual overview of explored vs unexplored areas."),
                }
                if _at_now:
                    nav["you_are_ALREADY_AT"] = _at_now
                    nav["arrival_directive"] = (
                        f"You are ALREADY standing at: {', '.join(_at_now)}. Do "
                        f"NOT 'goto' {'/'.join(_at_now)} again - you are here. "
                        "Either INVESTIGATE right here (open a container/body and "
                        "take items, talk to someone, read a sign, go through a "
                        "door), or if you've already done that, treat this place "
                        "as DONE and 'goto' a DIFFERENT place by name.")
                view["navigation"] = nav
        except Exception:
            pass
        # ALWAYS-ON activity scorecard: shows lifetime counts of physical
        # investigation (searched / picked up / opened / read) vs talking, so a
        # stateless model can NOTICE if it has been talking without ever
        # searching the world for clues or items. This is fair self-knowledge a
        # human player has, and it does NOT reveal any puzzle solution.
        try:
            view["what_i_have_actually_done"] = kb.activity_scorecard()
            # ESCALATING NUDGE: if you've talked a lot but NEVER searched or
            # picked anything up, you are stuck in conversation-only mode. Many
            # quests need a PHYSICAL clue/item you must FIND. Push hard toward
            # exploring buildings and searching containers.
            _sc = kb.activity_scorecard()
            _talked = _sc.get("talked to people", 0)
            _opened = _sc.get("containers/bodies opened", 0)
            _took = _sc.get("items picked up / taken", 0)
            if _talked >= 8 and (_opened + _took) == 0:
                view["INVESTIGATE_NOW"] = (
                    f"You have TALKED {_talked} times but OPENED 0 containers/"
                    "bodies and TAKEN 0 items. Talking alone will NOT solve this "
                    "- the clue/item you need is a PHYSICAL thing to find. STOP asking "
                    "questions: ENTER a building, walk RIGHT UP to a chest/desk/"
                    "barrel (adjacent, ~1 tile), and 'open' it, then 'take' items. If a door is "
                    "locked, 'unlock' it. Do this for SEVERAL turns, trying "
                    "DIFFERENT buildings you haven't entered.")
        except Exception:
            pass
        # CONVERSATION FOCUS: tag each answer option as already-explored vs open,
        # so the agent pursues OPEN topics and avoids re-picking a branch it has
        # fully explored (the churn where it re-asks the same topics). It MAY
        # still revisit an asked topic if it believes quest progress unlocked
        # something new - but by default, skip the [asked] ones.
        if in_convo:
            _ans = state.get("answers") or []
            if _ans:
                # Conversing NPC = nearest non-party person (the one adjacent).
                _cnpc = None
                _best = 1e9
                for _n in nearby:
                    if _n.get("in_party") or not _n.get("name"):
                        continue
                    _d = abs(_n.get("dx", 0)) + abs(_n.get("dy", 0))
                    if _d < _best:
                        _best, _cnpc = _d, _n.get("name")
                _asked = set()
                if _cnpc and kb:
                    _asked = {str(a).strip().lower()
                              for a in (kb.recall_npc(_cnpc) or {}).get("topics_asked", [])}
                _tagged = []
                _open_left = 0
                for _i, _a in enumerate(_ans):
                    _is_asked = str(_a).strip().lower() in _asked
                    if not _is_asked:
                        _open_left += 1
                    _tagged.append(f"{_i}: {_a}" + ("  [asked - branch explored]" if _is_asked else ""))
                view["conversation_options"] = _tagged
                view["CONVERSATION_TIP"] = (
                    "Pick an OPEN option (not marked [asked]). Options marked "
                    "[asked] are branches you already fully explored - do NOT "
                    "re-pick them unless you think a quest you advanced unlocked "
                    "new dialogue there. "
                    + ("No open topics remain - say goodbye/leave and move on."
                       if _open_left == 0 else
                       f"{_open_left} open topic(s) remain."))
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
    if state.get("area_map") is not None:
        # Town-scale explored-area overview the agent requested via the map tool.
        view["area_map"] = state["area_map"]
    # ACTION LOG LAST: the continuously-appended temporal memory goes at the very
    # BOTTOM of the context so the newest turns are the final thing the model
    # reads (best for recency/attention) and nothing static sits below it.
    try:
        if ah:
            view["action_log"] = ah
    except NameError:
        pass
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
    _pp = state.get("player") or {}
    _px, _py, _pz = _pp.get("tx", 0), _pp.get("ty", 0), (_pp.get("tz", 0) or 0)
    nearby = state.get("nearby") or []
    if nearby:
        lines.append("")
        lines.append("Characters nearby (absolute tile):")
        for n in nearby[:12]:
            _ax = n.get("tx") if n.get("tx") is not None else _px + n.get("dx", 0)
            _ay = n.get("ty") if n.get("ty") is not None else _py + n.get("dy", 0)
            _az = _pz + (n.get("dz", 0) or 0)
            lines.append(f"  {n.get('name')}  ({_ax},{_ay},{_az})  {_compass(n.get('dx',0), n.get('dy',0))}")
    objects = state.get("objects") or []
    if objects:
        lines.append("")
        lines.append("Objects nearby (absolute tile):")
        for o in objects[:12]:
            _ax = _px + o.get("dx", 0)
            _ay = _py + o.get("dy", 0)
            _az = _pz + (o.get("dz", 0) or 0)
            lines.append(f"  {o.get('name')}  ({_ax},{_ay},{_az})  {_compass(o.get('dx',0), o.get('dy',0))}")
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
    # SALVAGE a regurgitated action-log reply: some turns the model echoes its
    # own temporal memory back as output - a JSON object whose keys/values are
    # past log lines like "[T658] move toward (1062,2145)" - and runs to the
    # token cap, producing no valid action (seen live as repeated
    # "could not parse reply" -> wait). Rather than waste the turn, pull the
    # FIRST concrete intent out of that text: a "move/go toward (x,y)" tile, or
    # a recognizable action verb. This keeps the agent progressing.
    if not isinstance(obj, dict) or ("action" not in obj and "type" not in obj):
        mt = re.search(r'(?:move|go|goto|walk|head)\w*\s+(?:toward\s+)?\(?\s*(\d{3,5})\s*,\s*(\d{3,5})\s*\)?',
                       text, re.I)
        if mt:
            return ("(guard) salvaged goto from a garbled reply",
                    {"type": "goto", "tx": int(mt.group(1)), "ty": int(mt.group(2))})
        for _verb, _act in (("open", {"type": "open"}),
                            ("loot", {"type": "loot"}),
                            ("look", {"type": "look"}),
                            ("talk", {"type": "talk"}),
                            ("continue", {"type": "continue"})):
            if re.search(rf'\b{_verb}\b', text, re.I):
                return (f"(guard) salvaged {_verb} from a garbled reply", _act)
    if not isinstance(obj, dict):
        return ("(could not parse reply)", {"type": "wait"})
    # Accept the model's rationale under any of the common field names it emits
    # (some models use "reasoning"/"thought"/"rationale" instead of "reason").
    reason = ""
    for _rk in ("reason", "reasoning", "thought", "rationale", "explanation"):
        _rv = obj.get(_rk)
        if isinstance(_rv, str) and _rv.strip():
            reason = _rv.strip()
            break
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

    # ---- Action ALIASING -------------------------------------------------
    # Small models drift to near-miss synonyms for valid actions (e.g.
    # "look_around" instead of "search", or a "direction" key instead of
    # "dir"). Rather than reject these (which wastes a turn and, worse, leaves
    # the agent with no working recovery tool when it's cornered), map the
    # common variants onto the canonical action the engine accepts. This
    # mirrors the existing dx/dy and press_key normalization.
    _t = action.get("type")
    if isinstance(_t, str):
        _TYPE_ALIASES = {
            # "look around / observe" -> the engine's search (it reports what's
            # "look around / observe" -> the driver's own "look" handler (it
            # describes the scene). NOT "search" - search was removed; opening
            # is now the deliberate "open" action.
            "look_around": "look", "lookaround": "look", "look_at": "look",
            "observe": "look", "examine": "look", "survey": "look", "scan": "look",
            # movement synonyms
            "walk": "move", "go": "move", "step": "move", "travel": "goto",
            "navigate": "goto", "goto_tile": "goto", "move_to": "goto",
            # item synonyms
            "grab": "take", "get": "take", "pick_up": "pickup", "pick": "pickup",
            # container synonyms: open a specific container/body (reveals
            # contents); "loot" is its own take-all action (do NOT alias it).
            "open_container": "open", "unlock_and_open": "open",
            "take_all": "loot", "empty": "loot", "loot_all": "loot",
            # conversation synonyms
            "reply": "answer", "respond": "answer", "choose": "answer",
            "select": "answer", "ask": "answer",
            # "say"/"speak"/"think"/"note" out of a conversation aren't engine
            # actions - the model is thinking out loud. Treat as a no-op wait so
            # it doesn't burn the turn on an error and can re-plan next turn.
            "say": "wait", "speak": "wait", "think": "wait", "note": "wait",
            "idle": "wait", "rest": "wait", "pause": "wait",
            # misc
            "descend_stairs": "descend", "go_down": "descend",
        }
        _canon = _TYPE_ALIASES.get(_t.lower())
        if _canon:
            action["type"] = _canon

    # Field alias: the engine reads "dir"; models often send "direction" or
    # "heading". Fold them in (without clobbering an explicit "dir").
    for _dkey in ("direction", "heading", "facing"):
        if _dkey in action and "dir" not in action:
            action["dir"] = action.pop(_dkey)
    # Direction VALUE aliases: accept spelled-out compass words.
    _d = action.get("dir")
    if isinstance(_d, str):
        _DIR_WORDS = {
            "north": "n", "south": "s", "east": "e", "west": "w",
            "northeast": "ne", "northwest": "nw",
            "southeast": "se", "southwest": "sw",
            "up": "n", "down": "s", "left": "w", "right": "e",
        }
        action["dir"] = _DIR_WORDS.get(_d.lower(), _d.lower())
    # Shape alias: a "move"/"walk"/"go" with a NAMED target but no dir/tx is
    # really a goto-by-name (the engine resolves the name to a tile and its
    # z-aware pathfinder walks/climbs there). Seen live: {"type":"move",
    # "name":"stairs"} was rejected as a bad direction. Route it to goto.
    if (action.get("type") == "move" and action.get("name")
            and not action.get("dir")
            and action.get("tx") is None and action.get("ty") is None):
        action["type"] = "goto"
    return (reason, action)


def _delta_to_dir(dx: int, dy: int) -> str:
    """Map a tile delta to a compass dir for the move action (north = -y).
    Fallback only; the normal path sends absolute tile tx/ty."""
    ns = "n" if dy < 0 else ("s" if dy > 0 else "")
    ew = "e" if dx > 0 else ("w" if dx < 0 else "")
    return (ns + ew) or "n"


def _compass(dx: int, dy: int) -> str:
    """Short direction+distance hint (e.g. 'NE 5') to accompany an absolute tile,
    for spatial intuition. Coordinates are the source of truth; this is a hint."""
    ns = "N" if dy < 0 else ("S" if dy > 0 else "")
    ew = "E" if dx > 0 else ("W" if dx < 0 else "")
    d = (ns + ew) or "here"
    return f"{d} {abs(dx) + abs(dy)}" if d != "here" else "here"


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
            body = " - a body you can open" if o.get("body") else ""
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


META_TOOLS = {"add_quest", "update_quest", "note_npc", "add_topic"}


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
    if t == "add_topic":
        _tn = action.get("topic") or action.get("name") or ""
        kb.add_topic(_tn, str(action.get("note", "")),
                     getattr(kb, "current_turn", 0))
        return f"recorded topic '{_tn}'"
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
    # EXPLORE TOWARD THE UNKNOWN: if we have a fog-of-war bearing toward the
    # least-explored direction AND that direction is reasonably open on the
    # visible grid, prefer it and commit a LONGER hop (15 tiles) - this actually
    # leaves the current pocket to discover new buildings, instead of a timid
    # 6-tile shuffle in the most-open VISIBLE direction (which just circles a
    # walled enclosure). Falls back to the openness choice otherwise.
    _bearing = session.get("explore_bearing")
    _hop = 6
    if _bearing:
        _bdx, _bdy = _bearing
        # Only take the unexplored bearing if it isn't walled off immediately.
        if openness(_bdx, _bdy) >= 2 and not _lands_on_recent(_bdx, _bdy):
            dx, dy = _bdx, _bdy
            _hop = 15
    target = {"type": "goto", "tx": tx + dx * _hop, "ty": ty + dy * _hop}
    recent_targets.append((target["tx"], target["ty"]))
    if len(recent_targets) > 5:
        del recent_targets[0]
    return target


def _do_turn(args, window, ollama, exult, step, recent_positions, kb, session) -> None:
    state = exult.observe()
    # Per-turn frame capture for streaming. The bridge accepts ONE client at a
    # time, so a separate stream process cannot pull screenshots while the
    # driver is connected. Instead the driver (which holds the connection) asks
    # the engine to write a PNG each turn; stream.py --from-file just serves that
    # file. Cheap and contention-free. Enabled with --screenshot.
    if getattr(args, "screenshot", False):
        try:
            # screenshot is a CMD-level request ({"cmd":"screenshot"}), not an
            # act action type - send it via request(), not act().
            exult.request({"cmd": "screenshot"})
        except Exception:
            pass
    # Advance a PERSISTENT, monotonic turn counter (kb.turn_counter) that
    # survives restarts, so the action log's turn numbers always INCREASE across
    # runs (the per-process 'step' resets to 0 each launch, which made log turns
    # jump backwards after a relaunch). record_action/record_move tag entries
    # with this counter.
    kb.turn_counter = getattr(kb, "turn_counter", 0) + 1
    kb.current_turn = kb.turn_counter
    # Fog-of-war: mark the avatar's current cell explored (for the map tool).
    _pp0 = state.get("player") or {}
    if _pp0.get("tx") is not None:
        kb.record_visit(_pp0["tx"], _pp0["ty"],
                        region=str(state.get("map_num", "world")))
    # Stat-change logging: notice hp/food/gold changes turn-to-turn and record
    # them in the action log so the stateless agent KNOWS an event happened
    # (took damage, got hungry, gained/spent money) rather than only seeing the
    # new number in isolation.
    _prev_stats = session.get("prev_stats") or {}
    for _k, _label, _dir in (("hp", "HP", "dmg"), ("food", "food", "hunger"),
                             ("gold", "gold", "gold")):
        _cur = _pp0.get(_k)
        _old = _prev_stats.get(_k)
        if isinstance(_cur, int) and isinstance(_old, int) and _cur != _old:
            _delta = _cur - _old
            if _k == "gold":
                kb.record_action(f"gold {'+' if _delta>0 else ''}{_delta} (now {_cur})")
            elif _delta < 0:  # hp/food dropping is the notable event
                _why = "took damage" if _k == "hp" else "got hungrier"
                kb.record_action(f"{_label} {_old}->{_cur} ({_why})")
            elif _k == "hp" and _delta > 0:
                kb.record_action(f"HP {_old}->{_cur} (healed)")
    session["prev_stats"] = {"hp": _pp0.get("hp"), "food": _pp0.get("food"),
                             "gold": _pp0.get("gold")}
    if window.available:
        window.update_turn(kb.turn_counter)   # persistent monotonic turn, not per-run step
        window.set_map(state.get("grid") or "(no map)")
        window.set_dialog(format_dialog(state))
        # Zork-style room description for the Room panel (guarded: tkinter window
        # may not implement set_room).
        if hasattr(window, "set_room"):
            try:
                _objs = state.get("objects") or []
                _nb = state.get("nearby") or []
                window.set_room(describe_room(state, _recognize_place(_objs, _nb)))
            except Exception:
                pass
        # Inspector panels: quests, NPC knowledge, and stats.
        try:
            nearby_names = [n.get("name") for n in (state.get("nearby") or []) if n.get("name")]
            window.set_quests(kb.quests_pretty())
            if hasattr(window, "set_quests_data"):
                try:
                    window.set_quests_data(kb.quest_view(max_open=30))
                except Exception:
                    pass
            window.set_resolved_quests(kb.resolved_quests_list())
            window.set_npc_tree(kb.npcs_tree_data())
            window.set_topics_tree(kb.topics_tree_data())
            # Aggregated notes/journal for a dedicated panel (the notes were
            # only visible buried in the NPC tree before). Guarded: the tkinter
            # window may not implement set_notes.
            if hasattr(window, "set_notes"):
                try:
                    window.set_notes(kb.npc_notes_view(limit_npcs=20, notes_each=8))
                except Exception:
                    pass
            window.set_plot(kb.episodic_summary or "(no plot summary yet - the LLM builds this)")
            p = state.get("player") or {}
            # Feed the Show-inventory button from the always-on player state
            # (no extra engine query, no agent turn spent).
            try:
                _worn = p.get("worn") or {}
                _carry = p.get("carrying") or []
                _inv = "WORN:\n" + ("\n".join(f"  {slot}: {item}"
                                              for slot, item in _worn.items())
                                    if _worn else "  (nothing worn)")
                _inv += "\n\nCARRYING (" + str(len(_carry)) + "):\n" + (
                    "\n".join(f"  - {c}" for c in _carry) if _carry
                    else "  (pack empty)")
                window.set_inventory(_inv)
            except Exception:
                pass
            # Always-visible game status line.
            window.set_gstatus(
                f"{state.get('time_of_day','?')} (h{state.get('hour','?')})  "
                f"pos({p.get('tx')},{p.get('ty')},z{p.get('tz',0)})  hp {p.get('hp')}/{p.get('max_hp')}  gold {p.get('gold')}  food {p.get('food')}  "
                f"str {p.get('str')} dex {p.get('dex')} int {p.get('int')}  "
                f"{'IN COMBAT' if state.get('in_combat') else ''}")
            # Keep the GUI's "Show map" button supplied with a fresh area map
            # each turn (this is for the OPERATOR; it does NOT go into the LLM
            # context - that only happens when the LLM calls the map tool).
            try:
                window.set_area_map(kb.render_area_map(
                    p.get("tx", 0), p.get("ty", 0),
                    region=str(state.get("map_num", "world"))))
            except Exception:
                pass
            # Structured stats: labelled key/value pairs for distinct boxes.
            _resolved = sum(1 for q in kb.quests.values() if q.get("status") == "done")
            _ts = kb.tool_stats or {}
            stats_kv = {
                # Position is already shown in the top game-status row, so it is
                # intentionally NOT repeated here.
                "HP": f"{p.get('hp')}/{p.get('max_hp')}",
                "Gold": p.get("gold"),
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
            window.set_tool_stats(kb.tool_stats_pretty(top=20))
            if hasattr(window, "set_guard_stats"):
                try:
                    window.set_guard_stats(kb.guard_stats_data(top=30))
                except Exception:
                    pass
            if hasattr(window, "set_action_digest"):
                try:
                    # 450 = the default shown action-log window; the digest is
                    # everything older that has scrolled off it.
                    window.set_action_digest(kb.action_digest(450))
                except Exception:
                    pass
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

    # FUNCTIONAL BUILDING INFERENCE: many buildings are not labeled with their
    # purpose - a human recognizes a stable by its horses/trough/hay, a smithy
    # by its anvil/forge/bellows, etc. Detect characteristic contents in view
    # and record the building as a navigable landmark so the agent can "goto
    # stables" instead of wandering. Keyed on distinctive item sets.
    _FUNCTION_MARKERS = {
        "stables": ("horse", "water trough", "horseshoe", "hay", "trough",
                    "stall", "saddle"),
        "smithy": ("anvil", "forge", "bellows", "tongs"),
        "kitchen": ("oven", "hearth", "cauldron", "cooking"),
    }
    _people = state.get("nearby") or []
    for _bld, _markers in _FUNCTION_MARKERS.items():
        _hits = [_o for _o in (state.get("objects") or [])
                 if any(m in (_o.get("name") or "").lower() for m in _markers)]
        # A single stray horseshoe isn't a stable; require 2+ distinct markers
        # (or a horse, which is decisive for a stable).
        _distinct = {next(m for m in _markers if m in (_o.get("name") or "").lower())
                     for _o in _hits}
        _decisive = _bld == "stables" and any(
            "horse" in (n.get("name") or "").lower() for n in _people + _hits)
        if len(_distinct) >= 2 or _decisive:
            # Center the landmark on the marker cluster (average offset).
            _dx = sum(_o.get("dx", 0) for _o in _hits) // max(1, len(_hits))
            _dy = sum(_o.get("dy", 0) for _o in _hits) // max(1, len(_hits))
            kb.record_place(_bld, _ptx + _dx, _pty + _dy, kind="building",
                            note=f"recognized by its {', '.join(sorted(_distinct))}")

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
        _is_container = _o.get("container") or _o.get("body")
        if _is_container or kb.is_notable_object(_onm):
            _ox, _oy = _ptx + _o.get("dx", 0), _pty + _o.get("dy", 0)
            kb.note_observation(f"{_onm} at ({_ox},{_oy})", kind="seen", step=step)
            # ALSO record it as a goto-able PLACE so the agent can RETURN to it
            # later ("goto chest", "goto parrot") instead of losing its location
            # when it scrolls off the per-turn "objects nearby" list. This turns
            # ephemeral sightings into durable, navigable memory. Skip ultra-
            # common furniture that would clutter the map.
            _low = _onm.lower()
            _skip = any(w in _low for w in ("wall", "floor", "roof", "window",
                                            "fence", "post", "garbage", "tree",
                                            "table", "chair", "light source"))
            if _onm and not _skip:
                kb.record_place(_onm, _ox, _oy,
                                kind=("container" if _is_container else "seen"))

    # Record NPC dialog into the journal + dialogue history (with speaker).
    npc_text = state.get("npc_text")
    if state.get("conversation_in_progress") and npc_text:
        kb.add_journal(npc_text)
        cur_npc = session.get("current_npc", "?")
        # Speaker attribution rule: we ONLY trust a speaker that was set
        # deliberately - when the agent issued a `talk` action (current_npc set
        # to that NPC) or a name was revealed mid-conversation. If a conversation
        # somehow produced NPC text without a known partner, we do NOT guess by
        # proximity: guessing is exactly what misfiled companion dialogue under a
        # nearby "dog". Per policy, unknown attribution defaults to a clearly
        # labelled "unknown speaker" so nothing is ever blamed on the wrong
        # character. (Better a truthful "unknown" than a confident wrong guess.)
        if cur_npc in ("?", "", None):
            cur_npc = "unknown speaker"
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
                              "child", "sailor", "monk", "healer", "innkeeper",
                              "unknown speaker")
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
        # OPERATOR SAVE: if the GUI Save-game button was pressed, save the game
        # now (the driver owns the engine socket) and ack back to the GUI.
        if window.available and window.consume_save_request():
            try:
                _sr = exult.act({"type": "save"})
                _ok = bool(isinstance(_sr, dict) and _sr.get("ok"))
                print(f"[{step:03d}] operator SAVE -> {_sr}")
                window.set_save_ack(_ok)
            except Exception as _e:
                print(f"[{step:03d}] operator SAVE failed: {_e}")
                window.set_save_ack(False)
            return   # saving is a non-game side-effect; don't advance this cycle
        # OPERATOR INTERVIEW: if the human asked the agent a question, answer it
        # out-of-band using the CURRENT game context - WITHOUT advancing the game
        # or altering the agent's memory. This lets us probe its reasoning
        # ("why are you carrying a pitchfork?") separately from play.
        _ask = window.get_ask() if window.available else None
        _ask_from = "operator"
        if not _ask:
            # File hooks the external Twitch bridge writes to:
            #  - ask_queue.txt : one QUESTION per line (viewers via !ask). We pop
            #    the FIRST line each turn (FIFO) so many viewers can queue up.
            #  - ask.txt       : single-shot (legacy / scripted).
            _dir = os.path.dirname(os.path.abspath(__file__))
            try:
                qf = os.path.join(_dir, "ask_queue.txt")
                if os.path.isfile(qf):
                    _lines = [l for l in open(qf, encoding="utf-8").read().splitlines() if l.strip()]
                    if _lines:
                        _ask = _lines[0].strip()
                        _ask_from = "twitch"
                        open(qf, "w", encoding="utf-8").write("\n".join(_lines[1:]))
            except OSError:
                pass
            if not _ask:
                try:
                    af = os.path.join(_dir, "ask.txt")
                    if os.path.isfile(af):
                        _q = open(af, encoding="utf-8").read().strip()
                        if _q:
                            _ask = _q
                            open(af, "w", encoding="utf-8").close()
                except OSError:
                    pass
        if _ask:
            try:
                # Compact, PROSE-friendly context (not the full state JSON, which
                # the model echoed). Give it what it needs to explain itself.
                _pp = state.get("player") or {}
                _recent = kb.action_view(12) if kb else []
                _cur_q = kb.current_quest if kb else None
                _ctx = (
                    "Plot so far: " + (kb.episodic_summary or "(none yet)") + "\n"
                    "Current focus quest: " + str(_cur_q) + "\n"
                    "Position: " + f"({_pp.get('tx')},{_pp.get('ty')}) elevation {_pp.get('tz',0)}" + "\n"
                    "Wearing: " + str(_pp.get("worn") or {}) + "\n"
                    "Carrying: " + str(_pp.get("carrying") or []) + "\n"
                    "Recent actions:\n  " + "\n  ".join(str(a) for a in _recent))
                _iprompt = (
                    "You are the character/agent playing Ultima VII. A viewer sent "
                    "you a MESSAGE. Do TWO things:\n"
                    "1) ANSWER in plain English (under 3 sentences, honest and "
                    "specific, first person). If unsure or mistaken, say so.\n"
                    "2) Decide if the message is GUIDANCE to act on in gameplay "
                    "vs. just a question.\n"
                    "Reply EXACTLY as:\nANSWER: <reply>\nACTIONABLE: <yes|no>\n\n"
                    "=== YOUR SITUATION ===\n" + _ctx +
                    "\n\n=== THEIR MESSAGE ===\n" + _ask + "\n")
                _iprompt = (
                    "You are the character/agent playing Ultima VII. A viewer sent "
                    "you a MESSAGE. Do TWO things:\n"
                    "1) ANSWER in plain English (under 3 sentences, honest, first "
                    "person). If unsure or mistaken, say so.\n"
                    "2) CLASSIFY the message as exactly one of:\n"
                    "   EPHEMERAL - just a question about your thoughts; change "
                    "nothing.\n"
                    "   HINT - a short-term tip to act on soon (e.g. 'try the "
                    "north door'); you'll keep it in mind for a few turns.\n"
                    "   QUEST - a NEW lasting goal or major direction (e.g. 'your "
                    "real objective is to find Batlin in Britain'); worth adding "
                    "to your quest log permanently.\n"
                    "Reply EXACTLY as:\nANSWER: <reply>\nKIND: <EPHEMERAL|HINT|QUEST>"
                    "\nTITLE: <if QUEST, a short quest title; else ->>\n\n"
                    "=== YOUR SITUATION ===\n" + _ctx +
                    "\n\n=== THEIR MESSAGE ===\n" + _ask + "\n")
                _ans = ollama.chat_ex("You are a game-playing agent explaining "
                                      "your reasoning in plain English.",
                                      _iprompt, force_json=False)
                _raw = (_ans.get("content") or "").strip() if isinstance(_ans, dict) else str(_ans)
                # Parse ANSWER / KIND / TITLE (fall back to whole text as answer).
                _atext = _raw
                _kind = "EPHEMERAL"
                _qtitle = ""
                import re as _re
                _ma = _re.search(r"ANSWER:\s*(.+?)(?:\nKIND:|$)", _raw, _re.S | _re.I)
                if _ma:
                    _atext = _ma.group(1).strip()
                _mk = _re.search(r"KIND:\s*(EPHEMERAL|HINT|QUEST)", _raw, _re.I)
                if _mk:
                    _kind = _mk.group(1).upper()
                _mt = _re.search(r"TITLE:\s*(.+)", _raw, _re.I)
                if _mt:
                    _qtitle = _mt.group(1).strip().strip("->").strip()
                print(f"[{step:03d}] MESSAGE: {_ask}\n         A: {_atext[:200]}"
                      f"  [kind={_kind}{(' title='+_qtitle) if _qtitle else ''}]")
                # Route the answer to the ORIGINATING channel only, keeping the
                # operator's private GUI chat separate from public Twitch chat.
                # Only tag the answer when something actually CHANGED (a hint
                # noted or a quest added). A pure question needs no tag.
                _tag = {"HINT": "[noted as a HINT - I'll act on it soon]",
                        "QUEST": f"[added a QUEST: {_qtitle or _ask[:40]}]"}.get(_kind, "")
                _shown = (_atext or "(no answer)") + ("\n" + _tag if _tag else "")
                if _ask_from == "operator":
                    if window.available:
                        window.set_answer(_shown)
                # Persist based on the agent's OWN classification:
                #  EPHEMERAL -> nothing stored (pure interview).
                #  HINT      -> active hint for a few turns (transient steering).
                #  QUEST     -> a durable new quest in the log (long-term goal).
                if _kind == "HINT":
                    session["hint"] = _ask
                    session["hint_ttl"] = 3
                    kb.record_hint(_ask, step)
                    print(f"[{step:03d}] (message -> transient HINT)")
                elif _kind == "QUEST":
                    _title = _qtitle or _ask[:60]
                    kb.add_quest(title=_title, priority=2,
                                 notes=f"From operator/viewer message: {_ask}")
                    kb.record_hint(_ask, step)   # also note it as guidance
                    print(f"[{step:03d}] (message -> new QUEST: {_title})")
                # For TWITCH-sourced messages only, write the Q&A to a file so the
                # Twitch bridge can post it back to chat + OBS. Operator/GUI chat
                # stays private and is NOT written here.
                if _ask_from == "twitch":
                    try:
                        _dir = os.path.dirname(os.path.abspath(__file__))
                        with open(os.path.join(_dir, "agent_answer.txt"), "w",
                                  encoding="utf-8") as _af:
                            _af.write(f"Q: {_ask}\nA: {_shown}")
                    except OSError:
                        pass
            except Exception as _e:
                _fail = f"(interview failed: {_e})"
                if _ask_from == "operator" and window.available:
                    window.set_answer(_fail)
                elif _ask_from == "twitch":
                    try:
                        _dir = os.path.dirname(os.path.abspath(__file__))
                        open(os.path.join(_dir, "agent_answer.txt"), "w",
                             encoding="utf-8").write(f"Q: {_ask}\nA: {_fail}")
                    except OSError:
                        pass
            return   # interview iteration: do NOT advance the game this cycle
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
        _am = session.pop("area_map", None)
        if _am is not None:
            state["area_map"] = _am
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
        # Operator-adjustable temporal-memory window (GUI 'Turn memory' box).
        if window.available:
            state["_turn_window_override"] = window.get_turn_window()
        # Compute the least-explored compass direction (fog of war) so the
        # explore guard can head toward GENUINELY NEW territory - discovering
        # new buildings - instead of shuffling within the already-explored
        # pocket (seen live: it circled the stable pen for dozens of turns).
        try:
            _pp_ex = state.get("player") or {}
            session["explore_bearing"] = kb.unexplored_bearing(
                _pp_ex.get("tx", 0), _pp_ex.get("ty", 0)) if kb else None
        except Exception:
            session["explore_bearing"] = None
        _user = summarize_state(state, kb, session.pop("last_look", ""), alert, squeeze)
        # Operator THROTTLE: wait N ms BEFORE sending the request to ollama, to
        # cool the LLM box when it runs hot (spaces out the GPU-heavy inference
        # calls). Default 0 (no effect); live-adjustable 1-1000 ms via the
        # inspector "throttle ms" control.
        if window is not None and hasattr(window, "get_throttle_ms"):
            try:
                _throttle = window.get_throttle_ms()
                if _throttle and _throttle > 0:
                    time.sleep(min(int(_throttle), 5000) / 1000.0)
            except Exception:
                pass
        res = ollama.chat_ex(get_effective_system_prompt(), _user)
        reply = res["content"]
        reason, action = parse_reply(reply)
        # Movement normalizer: the "move" action takes EITHER a compass "dir"
        # (n/s/e/w/ne/nw/se/sw, optional "steps") OR game-tile "tx"/"ty".
        # Models often emit a delta {"dx":..,"dy":..} instead; translate that to
        # an absolute game-tile move (current tile + delta) - purely tile-based,
        # no screen coordinates involved.
        if isinstance(action, dict) and action.get("type") == "move" \
                and "dir" not in action and "tx" not in action \
                and ("dx" in action or "dy" in action):
            try:
                dx = int(action.get("dx", 0))
                dy = int(action.get("dy", 0))
            except (TypeError, ValueError):
                dx = dy = 0
            p = state.get("player") or {}
            if dx == 0 and dy == 0:
                action = {"type": "wait"}
                reason = "(guard) move with zero delta -> wait"
            elif p.get("tx") is not None:
                action = {"type": "move",
                          "tx": int(p["tx"]) + dx, "ty": int(p["ty"]) + dy}
                reason = f"(guard) move dx/dy -> tile ({action['tx']},{action['ty']})"
            else:
                action = {"type": "move", "dir": _delta_to_dir(dx, dy)}
                reason = f"(guard) move dx/dy -> dir {action['dir']}"
        # Conversation normalizer: while an NPC's answer menu is on screen, the
        # ONLY action the engine accepts is "answer" (by index or text). Some
        # models emit invented shapes ({"type":"say",...}, {"type":"talk",
        # "option":...}, or a "move" with dx/dy) which the bridge rejects, so the
        # dialog never advances and the agent gets stuck. Coerce those into a
        # valid "answer" against the live answer list.
        if state.get("conversation_active") and isinstance(action, dict):
            answers = state.get("answers") or []
            atype = action.get("type")
            # Coerce when NOT an answer, OR when it's an "answer" that lacks a
            # usable index/text but carries a topic/option string (models often
            # send {"type":"answer","topic":"name"} which the engine rejects as
            # "answer out of range" since it wants an integer index).
            answer_needs_fix = (
                atype == "answer"
                and not isinstance(action.get("index"), int)
                and not action.get("text"))
            if atype != "answer" or answer_needs_fix:
                # Pull whatever text the model intended to say/pick.
                want = (action.get("text") or action.get("option")
                        or action.get("topic") or action.get("target_topic") or "")
                want = str(want).strip().lower()
                chosen = None
                if want and answers:
                    # Exact, then substring match against the offered options.
                    for i, a in enumerate(answers):
                        if str(a).strip().lower() == want:
                            chosen = i
                            break
                    if chosen is None:
                        for i, a in enumerate(answers):
                            if want in str(a).strip().lower():
                                chosen = i
                                break
                if chosen is not None:
                    action = {"type": "answer", "index": chosen}
                    reason = f"(guard) coerced '{atype}' -> answer #{chosen} ({answers[chosen]})"
                elif answers:
                    # Unknown/none matched: if the model was trying to leave
                    # (say/bye/leave/goto/move away), pick an exit option if one
                    # is offered, else advance/first option to keep dialog moving.
                    leave_words = ("bye", "leave", "goodbye", "farewell")
                    exit_idx = next(
                        (i for i, a in enumerate(answers)
                         if any(w in str(a).strip().lower() for w in leave_words)),
                        None)
                    if atype in ("say", "goto", "move", "stop") and exit_idx is not None:
                        action = {"type": "answer", "index": exit_idx}
                        reason = f"(guard) '{atype}' in conversation -> leaving via answer #{exit_idx} ({answers[exit_idx]})"
                    else:
                        action = {"type": "answer", "index": 0}
                        reason = f"(guard) invalid '{atype}' in conversation -> answer #0 ({answers[0]})"
                else:
                    # No answer list yet (NPC still talking) -> advance text.
                    action = {"type": "continue"}
                    reason = f"(guard) '{atype}' but no answers yet -> continue"
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
                f"===== TURN {kb.turn_counter} =====\n"
                f"----- SYSTEM PROMPT -----\n{get_effective_system_prompt()}\n\n"
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
                    kb.record_tool("new_quest", True)
                    print(f"[{step:03d}] inline add_quest: {q['title']!r} -> {qid}")
            _rq = _obj.get("resolve_quest")
            if isinstance(_rq, str) and _rq:
                kb.resolve_quest(_rq)
                kb.reset_talk_gate()
                kb.record_tool("resolve_quest", True)
                # Log the resolution WITH the reason so a premature/false
                # "done" is visible in the temporal memory and can be noticed.
                kb.record_action(f"marked quest DONE: {_rq}", reason=reason or "")
                print(f"[{step:03d}] inline resolve_quest: {_rq}")
            # Inline drop_quest: REMOVE a redundant/obsolete quest (not the same
            # as resolving - use this to consolidate duplicates or discard a goal
            # that's no longer relevant). Accepts a title/id or a list.
            _dq = _obj.get("drop_quest") or _obj.get("remove_quest")
            _dq_items = _dq if isinstance(_dq, list) else ([_dq] if _dq else [])
            for _d in _dq_items:
                if isinstance(_d, str) and _d.strip():
                    if kb.drop_quest(_d):
                        kb.record_tool("drop_quest", True)
                        print(f"[{step:03d}] inline drop_quest: {_d}")
            # Inline consolidate_notes: the LLM rewrites an NPC's notes to a
            # pruned/merged version. {"consolidate_notes": {"npc": "...",
            # "notes": "..."|[...]}} or a list of such. Lets it clean up notes it
            # only ever appended to.
            _cn = _obj.get("consolidate_notes")
            _cn_items = _cn if isinstance(_cn, list) else ([_cn] if _cn else [])
            for _c in _cn_items:
                if isinstance(_c, dict) and _c.get("npc"):
                    if kb.consolidate_notes(str(_c["npc"]), _c.get("notes", "")):
                        kb.record_tool("consolidate_notes", True)
                        print(f"[{step:03d}] inline consolidate_notes: {_c['npc']}")
            # Inline LLM-authored TOPICS: {"topic":"Fellowship","note":"..."} or a
            # list. Topics now come ONLY from the LLM, so the topic list shows
            # its own thinking accumulating over time.
            _tp = _obj.get("topic") or _obj.get("topics") or _obj.get("add_topic")
            _tp_items = _tp if isinstance(_tp, list) else ([_tp] if _tp else [])
            for t in _tp_items:
                if isinstance(t, dict) and (t.get("name") or t.get("topic")):
                    tname = str(t.get("name") or t.get("topic"))
                    tkey = kb.add_topic(tname, str(t.get("note", "")), step)
                    kb.record_tool("topic", True)
                    print(f"[{step:03d}] inline add_topic: {tname!r} -> {tkey}")
                elif isinstance(t, str) and t.strip():
                    kb.add_topic(t.strip(), "", step)
                    kb.record_tool("topic", True)
            # Inline PLOT SUMMARY: the LLM maintains a running "story so far" it
            # keeps updated. Replaces the bounded summary that's always in
            # context (for detail it uses recall/quests). Accept a few key names.
            _ps = _obj.get("plot_summary") or _obj.get("story_so_far") or _obj.get("summary")
            if isinstance(_ps, str) and _ps.strip():
                kb.set_plot_summary(_ps)
                kb.record_tool("plot_summary", True)
                print(f"[{step:03d}] plot summary updated ({len(_ps)} chars)")
            # Inline current-quest declaration: the agent tells us which quest it
            # is working on. Tagged onto the action log (shows quest switches).
            _cq = _obj.get("set_current_quest") or _obj.get("current_quest")
            if isinstance(_cq, str) and _cq.strip():
                if _cq.strip() != (kb.current_quest or ""):
                    kb.set_current_quest(_cq)
                    kb.record_tool("set_current_quest", True)
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
            # Reasoning column: ONLY the parsed short reason (never the raw JSON
            # reply, which is noise). Empty reason -> blank, not the reply dump.
            window.set_thinking(reason if reason else "")
            # Raw model thinking (reasoning-model chain-of-thought) if present -
            # goes to the Thought column; empty for think-off models like qwen.
            _thk = res.get("thinking") if isinstance(res, dict) else None
            if _thk:
                window.set_thought(_thk)
    # Stash the model's raw thinking (if any) for the stream overlay. Empty for
    # think-off models (e.g. our Granite with think=false); populated for
    # reasoning models. Read by _write_overlay at the end of the turn.
    session["last_thinking"] = (res.get("thinking") if isinstance(res, dict) else "") or ""

    if window.available and args.dry_run:
        window.set_thinking(reason)

    # --- "recall": retrieve the FULL saved dialogue tree for a character -
    #     everything they said (their transcript), topics already asked, and
    #     topics still unasked. Does not advance the game; the result is shown
    #     in next turn's observation as "recalled" so the agent can remember,
    #     e.g., the Mayor's instructions from a past run.
    if isinstance(action, dict) and action.get("type") == "quests":
        # Pull the FULL quest log on demand (kept compact in the always-on
        # state). With {"finished": true} review COMPLETED quests instead of
        # open ones - so you can see what you've already accomplished, avoid
        # re-adding done goals, and curate your log. Shown next turn as
        # "quest_detail".
        if action.get("finished") or action.get("done"):
            session["quest_detail"] = {"finished": kb.finished_quests_detail(40)}
            kb.record_action("reviewed FINISHED quests")
            _lbl = "[quests] reviewed finished quests"
            reason = "(reviewed my finished quests)"
        else:
            session["quest_detail"] = kb.quest_view(max_open=20)
            kb.record_action("reviewed quest log")
            _lbl = "[quests] reviewed full quest log"
            reason = "(reviewed my quest log)"
        if window.available:
            window.set_action(_lbl)
        print(f"[{step:03d}] quests: reviewed ({'finished' if action.get('finished') else 'open'})")
        kb.record_tool("quests", True)
        action = {"type": "wait", "_counted": True}

    # MAP tool: a town-scale overview of where you are and what you've explored,
    # with labeled landmarks - for orientation relative to the whole town/area
    # (the screen grid only shows a small radius). Shown next turn as area_map.
    if isinstance(action, dict) and action.get("type") == "map":
        _pp = state.get("player") or {}
        session["area_map"] = kb.render_area_map(
            _pp.get("tx", 0), _pp.get("ty", 0),
            region=str(state.get("map_num", "world")),
            radius_cells=16, cell_override=4)   # finer detail; we have context room
        kb.record_action("checked the area map")
        if window.available:
            window.set_action("[map] reviewed the area map")
        print(f"[{step:03d}] map: rendered area overview")
        kb.record_tool("map", True)
        action = {"type": "wait", "_counted": True}
        reason = "(checked the area map to orient myself)"

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
        kb.record_tool("recall", True)
        action = {"type": "wait", "_counted": True}
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
        kb.record_tool("annotate", True)
        action = {"type": "wait", "_counted": True}
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
            kb.record_tool("look", True)
            action = {"type": "wait", "_counted": True}
            reason = "(looked around; see description)"

    # --- Journal meta-tools: update the KB, then take a game action too. ---
    if isinstance(action, dict) and action.get("type") in META_TOOLS:
        note = _apply_meta(action, kb)
        kb.record_tool(str(action.get("type")), True)
        if window.available:
            window.set_action(f"[journal] {note}\n{json.dumps(action)}")
        print(f"[{step:03d}] journal: {note} :: {reason!r}")
        # Meta-tools don't advance the game; fall through with a light game
        # action so the turn still does something useful.
        action = {"type": "wait", "_counted": True}
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
        _gc = _guard_category(reason)
        if _gc:
            kb.record_guard(_gc)
        if window.available:
            window.set_action(_action_phrase(action))
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
            _gc = _guard_category(reason)
            if _gc:
                kb.record_guard(_gc)
            if window.available:
                window.set_action(_action_phrase(action))
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
        # NOTE: no special elevated/descend escape here. goto is z-aware - it
        # resolves the destination's elevation (probing down AND up for the
        # nearest standable surface) and the pathfinder climbs/descends stairs
        # as needed. So a plain goto to a ground tile from a wall-top resolves
        # correctly; we don't need a band-aid descend action in the wedge guard.
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
                    action = {"type": "open"}
                    reason = "(guard) stuck near a body; opening it to see contents"
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

    # --- Narrate-but-WAIT guard: the model often writes a paragraph about
    #     moving ("I will head north to the stables") but then emits a bare
    #     {"type":"wait"}, which does nothing - so it narrates progress while
    #     standing still, burning turns (seen live: many consecutive waits with
    #     movement-intent reasoning). A single wait is fine (it may be pausing
    #     for a threat to pass); but 2+ bare waits in a row outside a
    #     conversation/gump/combat means it's idling. Redirect to a real action:
    #     pursue a quest lead, else explore. Legitimate waits (wait_until for
    #     night, combat) are excluded. --------------------------------------
    if (isinstance(action, dict) and action.get("type") == "wait"
            and not state.get("conversation_in_progress")
            and not state.get("gump_open")
            and not state.get("in_combat")
            and not action.get("_counted")):   # _counted = a guard-issued wait
        n_wait = session.get("bare_wait_streak", 0) + 1
        session["bare_wait_streak"] = n_wait
        if n_wait >= 2:
            session["bare_wait_streak"] = 0
            qa, qr = _pursue_focus_quest(state, kb)
            if qa:
                action, reason = qa, qr
            else:
                action = _explore_far(state, session, wedged)
                reason = "(guard) narrated but waited repeatedly; acting (explore) instead"
            print(f"[{step:03d}] wait-guard: {n_wait} bare waits -> {action.get('type')}")
    elif isinstance(action, dict) and action.get("type") != "wait":
        session["bare_wait_streak"] = 0   # reset once a real action is taken

    # --- Open/Loot reachability guard: "open" and "loot" act on a body or
    #     container within reach (~few tiles). If the model targets one that is
    #     visible but a few tiles away, walk to it first so the action lands.
    #     Also break the re-loot loop: if we already emptied the spot we're at,
    #     nudge the agent to move on. (search was removed; opening is deliberate
    #     now, so this is much simpler than the old search guard.) -------------
    if (isinstance(action, dict) and action.get("type") in ("open", "loot")
            and not state.get("conversation_in_progress")):
        _pp = state.get("player") or {}
        _here = (_pp.get("tx"), _pp.get("ty"))
        # Already looted this spot? Don't re-loot an empty body - move on.
        _looted = session.get("looted_spots", set())
        if any(abs(_here[0]-lx) + abs(_here[1]-ly) <= 2 for (lx, ly) in _looted):
            session["last_bump"] = (
                "You already emptied the container/body here - nothing left to "
                "take. Grab any loose items you can see, or pursue another goal.")
            qa, qr = _pursue_focus_quest(state, kb)
            if qa and qa.get("type") not in ("open", "loot"):
                action, reason = qa, qr
            else:
                action = _explore_far(state, session, wedged)
                reason = "(guard) already emptied here; moving on to explore"
            print(f"[{step:03d}] loot-guard: {_here} already emptied; moving on")
        else:
            # If a target name/tx was given, trust it. Otherwise, if no
            # body/container is adjacent (<=1 tile) but one is visible farther
            # off, walk there first so open/loot hits something.
            _named = action.get("name") or (action.get("tx") is not None)
            _adj = any(abs(o.get("dx", 9)) <= 1 and abs(o.get("dy", 9)) <= 1
                       for o in (state.get("objects") or [])
                       if o.get("body") or o.get("container"))
            _adj = _adj or any(abs(n.get("dx", 9)) <= 1 and abs(n.get("dy", 9)) <= 1
                               for n in (state.get("nearby") or []) if n.get("dead"))
            if not _named and not _adj:
                cands = [(abs(o.get("dx", 99)) + abs(o.get("dy", 99)),
                          _pp.get("tx", 0) + o.get("dx", 0),
                          _pp.get("ty", 0) + o.get("dy", 0), o.get("name", "container"))
                         for o in (state.get("objects") or [])
                         if o.get("body") or o.get("container")]
                cands += [(abs(n.get("dx", 99)) + abs(n.get("dy", 99)),
                           _pp.get("tx", 0) + n.get("dx", 0),
                           _pp.get("ty", 0) + n.get("dy", 0), n.get("name", "body"))
                          for n in (state.get("nearby") or []) if n.get("dead")]
                if cands:
                    cands.sort(key=lambda c: c[0])
                    _, btx, bty, bnm = cands[0]
                    action = {"type": "goto", "tx": btx, "ty": bty}
                    reason = f"(guard) walking to '{bnm}' @({btx},{bty}) before {action.get('type','open')}"
                    print(f"[{step:03d}] loot-guard: goto '{bnm}' before open/loot")
                else:
                    session["last_bump"] = (
                        "There is no container or body within reach to open/loot. "
                        "'open' works on a door, chest, desk, drawer, bag, barrel, "
                        "crate, or body (name it or give its tx,ty). To grab a "
                        "loose item use 'pickup'; to look around use 'look'.")

    # --- Reach-the-item guard: "take"/"pickup" need you within ~1 tile. If the
    #     model targets a VISIBLE item that's farther, walk to its exact tile
    #     first (using the object's own coords), then it can take it next turn.
    #     This fixes "'key' is TOO FAR (7 tiles)" loops where the engine told the
    #     agent to goto the tile but it didn't follow through. General: any item,
    #     any quest. After repeated failures on the same item, give up on it.
    if (isinstance(action, dict) and action.get("type") in ("take", "pickup")
            and not state.get("conversation_in_progress")):
        _pn = (action.get("name") or "").lower()
        _pp = state.get("player") or {}
        match = next((o for o in (state.get("objects") or [])
                      if _pn and (o.get("name") or "").lower() == _pn), None)
        _fails = session.get("pickup_fails", {}).get(_pn, 0)
        _dz = (match.get("dz", 0) or 0) if match else 0
        if match and (abs(match.get("dx", 0)) > 1 or abs(match.get("dy", 0)) > 1):
            if _fails >= 3:
                # Tried to reach it several times without success - give up.
                action = _explore_far(state, session, wedged)
                reason = f"(guard) '{_pn}' unreachable after retries; moving on"
            else:
                # Walk to the item's EXACT tile, then take next turn.
                action = {"type": "goto",
                          "tx": _pp.get("tx", 0) + match.get("dx", 0),
                          "ty": _pp.get("ty", 0) + match.get("dy", 0)}
                reason = f"(guard) '{_pn}' is {abs(match.get('dx',0))+abs(match.get('dy',0))} tiles away; walking to it to take it"
                session.setdefault("pickup_fails", {})[_pn] = _fails + 1
                print(f"[{step:03d}] reach-guard: goto '{_pn}' @({action['tx']},{action['ty']}) before take")
        elif match and _dz < 0 and (_pp.get("tz", 0) or 0) > 0:
            # Right x,y but the item is BELOW us (we're on a ledge/wall-top above
            # it, e.g. standing on the wall over a ground-level key). take needs
            # the SAME level - descend to the item's floor first.
            if _fails >= 3:
                action = _explore_far(state, session, wedged)
                reason = f"(guard) '{_pn}' below us, can't reach after retries; moving on"
            else:
                action = {"type": "descend"}
                reason = f"(guard) '{_pn}' is below you (you're up high); descending to its level to take it"
                session.setdefault("pickup_fails", {})[_pn] = _fails + 1
                print(f"[{step:03d}] reach-guard: descend to '{_pn}' (dz={_dz}) before take")
        elif match:
            # Adjacent AND same level - clear the fail counter; let take proceed.
            session.setdefault("pickup_fails", {}).pop(_pn, None)
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

    # --- Anti-THRASH guard: the model can ping-pong between TWO goto goals a
    #     few tiles apart (e.g. "enter stables" <-> "read the sign"), each goto
    #     moving a little then switching, so it never ARRIVES at either. This
    #     bounces across a ~6-10 tile box - too big for the <=3 oscillation
    #     detector, too small to be progress - and uses raw {tx,ty} so the
    #     named-arrival guard misses it. Detect it directly: if recent turns are
    #     dominated by gotos confined to a small bounding box, stop navigating
    #     and COMMIT to investigating right here. -----------------------------
    if (isinstance(action, dict) and action.get("type") == "goto"
            and not state.get("conversation_in_progress")):
        _pp = state.get("player") or {}
        _th = session.setdefault("thrash_pos", [])
        _th.append((_pp.get("tx"), _pp.get("ty")))
        del _th[:-10]
        if len(_th) >= 8:
            _xs = [t[0] for t in _th if t[0] is not None]
            _ys = [t[1] for t in _th if t[1] is not None]
            if _xs and max(max(_xs) - min(_xs), max(_ys) - min(_ys)) <= 10:
                # Confined to a small area for 8+ goto turns = thrashing between
                # goals while standing in the place. Commit to a real local
                # action instead of another goto.
                session["thrash_pos"] = []
                _body = next((o for o in (state.get("objects") or [])
                              if (o.get("body") or o.get("container"))
                              and abs(o.get("dx", 99)) + abs(o.get("dy", 99)) <= 3),
                             None)
                if _body and abs(_body.get("dx", 9)) <= 1 and abs(_body.get("dy", 9)) <= 1:
                    action = {"type": "open", "name": _body.get("name", "")}
                    reason = f"(guard) thrashing; opening the {_body.get('name','container')} right here"
                elif _body:
                    action = {"type": "goto",
                              "tx": (state.get("player") or {}).get("tx", 0) + _body.get("dx", 0),
                              "ty": (state.get("player") or {}).get("ty", 0) + _body.get("dy", 0)}
                    reason = f"(guard) thrashing; walking to the {_body.get('name','container')} to open it"
                else:
                    # NOTE: emit a real no-op wait, NOT {"type":"look"} - the
                    # driver's look-handler runs EARLIER in the turn, so a look
                    # set here would fall through to the engine as 'unknown
                    # action type'. Describe the scene ourselves and wait.
                    session["last_look"] = describe_scene(state, kb)
                    kb.record_action("looked around")
                    action = {"type": "wait", "_counted": True}
                    reason = "(guard) thrashing between nearby goals; re-assessing (looked around)"
                    session["last_bump"] = (
                        "You have been walking back and forth between spots a few "
                        "tiles apart without arriving at anything - you ARE in the "
                        "area you keep trying to reach. STOP issuing goto to nearby "
                        "coordinates. Instead: open/take an item or container "
                        "you can SEE, enter a door, or pick ONE distant NEW area "
                        "and commit to it.")
                print(f"[{step:03d}] thrash-guard: bounded goto loop -> {action['type']}")

    # If the model asks to "goto" a named target that isn't visible but IS a
    # remembered place or NPC, resolve it to coordinates from the mental map.
    if (isinstance(action, dict) and action.get("type") == "goto"
            and action.get("name") and "tx" not in action):
        nm = action["name"]
        nearby_names = {n.get("name") for n in (state.get("nearby") or [])}
        if nm not in nearby_names:
            pos = kb.place_pos(nm) or kb.npc_last_pos(nm)
            if pos and pos[0] is not None:
                _pp0 = state.get("player") or {}
                _dist = (abs(pos[0] - _pp0.get("tx", 0))
                         + abs(pos[1] - _pp0.get("ty", 0)))
                # ARRIVAL CHECK: if we're already essentially AT the place we
                # keep trying to goto, the model is looping "goto <place>" while
                # standing in it, reading the no-op as "can't reach it" (seen
                # live: at the stables talking to Petre, yet plot says "failing
                # to reach the stables"). Tell it plainly it has ARRIVED so it
                # investigates here (search/enter the building/talk) instead of
                # re-routing to a phantom coordinate.
                if _dist <= 6:
                    session["last_bump"] = (
                        f"You have ARRIVED at '{nm}' - you are standing in it "
                        f"right now (within {_dist} tiles). STOP trying to "
                        f"'goto {nm}'; that is why it feels like a loop. "
                        f"INVESTIGATE here instead: 'open' bodies/containers then 'take' items, "
                        f"go THROUGH a door ('+'/'/') to enter the building, or "
                        f"'talk' to someone present. Update your plot_summary to "
                        f"note you have reached '{nm}'.")
                    # Prefer a productive local action over the no-op goto.
                    _door = next((d for d in (state.get("doors") or [])
                                  if abs(d.get("dx", 99)) + abs(d.get("dy", 99)) <= 5),
                                 None)
                    _cont = next((o for o in (state.get("objects") or [])
                                  if (o.get("body") or o.get("container"))
                                  and abs(o.get("dx", 99)) + abs(o.get("dy", 99)) <= 4),
                                 None)
                    if _cont and abs(_cont.get("dx", 9)) <= 1 and abs(_cont.get("dy", 9)) <= 1:
                        # Adjacent openable -> open it BY NAME (bare open scans
                        # only ~1 tile and was failing).
                        action = {"type": "open", "name": _cont.get("name", "")}
                        reason = f"(guard) arrived at '{nm}'; opening the {_cont.get('name','container')} here"
                    elif _cont:
                        # Openable in view but not adjacent -> walk onto its tile.
                        action = {"type": "goto",
                                  "tx": _pp0.get("tx", 0) + _cont.get("dx", 0),
                                  "ty": _pp0.get("ty", 0) + _cont.get("dy", 0)}
                        reason = f"(guard) arrived at '{nm}'; walking to the {_cont.get('name','container')} to open it"
                    elif _door:
                        action = {"type": "goto",
                                  "tx": _pp0.get("tx", 0) + _door.get("dx", 0),
                                  "ty": _pp0.get("ty", 0) + _door.get("dy", 0)}
                        reason = f"(guard) arrived at '{nm}'; entering through the door"
                    else:
                        # Nothing openable here at all -> do NOT emit a bare
                        # 'open' (it just fails 'nothing to open there'). Mark
                        # this landmark exhausted and move on so we don't loop.
                        session.setdefault("exhausted_landmarks", set()).add(
                            (_pp0.get("tx", 0), _pp0.get("ty", 0)))
                        action = _explore_far(state, session, wedged)
                        reason = f"(guard) arrived at '{nm}'; nothing to open here, exploring on"
                    print(f"[{step:03d}] arrival-guard: already at '{nm}' ({_dist} tiles)")
                # FIXATION BREAKER: if this landmark was already reached and had
                # NOTHING searchable (marked exhausted), stop re-going there -
                # the model loops 'goto stables' forever. Redirect to a NEW area.
                elif (session.get("exhausted_landmarks")
                      and any(abs(pos[0] - ex) + abs(pos[1] - ey) <= 2
                              for (ex, ey) in session.get("exhausted_landmarks"))):
                    session["last_bump"] = (
                        f"'{nm}' is a DEAD END for searching - you already went "
                        "there and there was nothing to search/loot. STOP going "
                        "back. Go to a DIFFERENT building you have NOT entered and "
                        "search inside it. The chest you need is in ANOTHER house.")
                    action = _explore_far(state, session, True)
                    reason = f"(guard) '{nm}' already searched-empty; exploring a NEW building"
                else:
                    action = {"type": "goto", "tx": pos[0], "ty": pos[1]}
                    reason = f"{reason} [mapped '{nm}' -> ({pos[0]},{pos[1]})]"
            else:
                # Before treating the name as unresolvable, try to match it to
                # something VISIBLE right now - the model often gotos an abstract
                # label ("victim", "the body", "chest") that IS in view as a real
                # object/NPC. Fuzzy-match the name (and a few common synonyms)
                # against nearby objects and NPCs; if one matches, route there.
                _pp = state.get("player") or {}
                _nml = nm.lower()
                # GENERIC name resolution (no puzzle-specific synonyms): match
                # the requested name as a substring of a visible object/NPC name.
                # Additionally, if the name refers to a "body"/"corpse", match any
                # object the ENGINE flagged body=true (its own generic flag - not
                # a curated list). This resolves things like "the body", "chest",
                # "guard" to what's actually in view, for ANY quest.
                _body_ref = ("body" in _nml or "corpse" in _nml)
                _cands = []
                for o in (state.get("objects") or []):
                    _on = (o.get("name") or "").lower()
                    if _on and (_nml in _on or (o.get("name","").lower() in _nml)
                                or (_body_ref and o.get("body"))):
                        _cands.append((abs(o.get("dx", 99)) + abs(o.get("dy", 99)),
                                       _pp.get("tx", 0) + o.get("dx", 0),
                                       _pp.get("ty", 0) + o.get("dy", 0), o.get("name")))
                for n2 in (state.get("nearby") or []):
                    _nn = (n2.get("name") or "").lower()
                    if _nn and (_nml in _nn or _nn in _nml):
                        _cands.append((abs(n2.get("dx", 99)) + abs(n2.get("dy", 99)),
                                       _pp.get("tx", 0) + n2.get("dx", 0),
                                       _pp.get("ty", 0) + n2.get("dy", 0), n2.get("name")))
                if _cands:
                    _cands.sort(key=lambda c: c[0])
                    _, _rtx, _rty, _rnm = _cands[0]
                    action = {"type": "goto", "tx": _rtx, "ty": _rty}
                    reason = f"(guard) '{nm}' -> visible '{_rnm}' @({_rtx},{_rty})"
                    print(f"[{step:03d}] goto-resolve: '{nm}' -> visible '{_rnm}'")
                else:
                  # Unresolvable target: it's not visible and not in our mental map
                  # or NPC memory. Don't hand the engine a goto it can't route
                  # (that makes the agent flail). Tell the model it's unknown and
                  # list what IS known so it can pick a real destination or explore.
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
                    "opening it. To look inside, emit the 'open' action now "
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
    # (Removed: the escape-latch subsystem was dead code - both arming blocks
    # were `if False and ...` after it was found to churn ~94x/run and worsen
    # the stairs loop. We rely on the action log + quest log + GROUND-LEVEL
    # label + the lighter descend/ground guards instead.)

    _at_ground = _cur_tz_now == 0
    _wants_descend = any(
        w in (reason or "").lower()
        for w in ("descend", "climb down", "go down", "down to ground",
                  "to ground level", "down the stairs", "down from"))
    # DESCEND-GUARD REMOVED: it caused more harm than good (fought the agent's
    # own stair navigation, dragging it tz->0 and causing endless up/down
    # oscillation). The agent now handles elevation itself via move/goto/descend
    # tools. We only leave a gentle FYI if it tries to "descend" while already at
    # ground (a no-op it should stop trying) - but we do NOT force any action.
    if _at_ground and _wants_descend:
        session["last_bump"] = (
            "FYI: you are already at GROUND level (elevation 0) - there is no "
            "lower level to descend to here. If you want to go UP, step onto the "
            "stairs; the floor you seek may be UPSTAIRS.")

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
        # (Removed: dead stairs-climb assist. It was gated on `_reason_stairs =
        # False` after Exult's own pathfinder was found to climb ramps when goto
        # targets the right elevation - the z-aware goto handles it now.)
        near_here = abs(tgt[0]-here[0]) + abs(tgt[1]-here[1]) <= 1
        last_was_goto = session.get("last_action_type") == "goto"
        if near_here:
            # Target reached. If there's a SEARCHABLE container/body adjacent,
            # SEARCH it (the agent came here to investigate). Else if there's a
            # nearby DOOR, go THROUGH it (the loot is usually INSIDE the
            # building, not at the outdoor landmark tile). Else, mark this spot
            # exhausted and explore a genuinely NEW direction - this breaks the
            # "goto landmark -> already here -> re-goto same landmark" loop that
            # traps the agent OUTSIDE a building it wants to search.
            _sc_obj = state.get("objects") or []
            _adj_cont = next((o for o in _sc_obj
                              if (o.get("container") or o.get("body"))
                              and abs(o.get("dx", 9)) <= 1 and abs(o.get("dy", 9)) <= 1), None)
            _near_door = next((o for o in _sc_obj
                               if "door" in (o.get("name", "").lower())
                               and abs(o.get("dx", 9)) + abs(o.get("dy", 9)) <= 4), None)
            if _adj_cont:
                action = {"type": "open"}
                reason = "(guard) at target with a container here; opening it"
            elif _near_door:
                _pp = state.get("player") or {}
                action = {"type": "goto",
                          "tx": _pp.get("tx", 0) + _near_door.get("dx", 0),
                          "ty": _pp.get("ty", 0) + _near_door.get("dy", 0)}
                reason = "(guard) at landmark; going THROUGH the door to search inside"
            else:
                # Nothing to search AT this landmark - it's a dead end for
                # looting. Remember it so we don't keep re-targeting it, and
                # explore somewhere NEW.
                session.setdefault("exhausted_landmarks", set()).add(tuple(tgt))
                action = _explore_far(state, session, True)
                reason = "(guard) nothing to search at this landmark; exploring a NEW area"
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
            # Remember WHERE we talked to them, so a later "goto <npc>" can
            # navigate BACK instead of failing 'unresolved' and looping. Use the
            # NPC's own tile if it's actually in view; else our tile (we talk to
            # people we're next to). Only records a real, present NPC's spot.
            _ptt = state.get("player") or {}
            _match = next((n for n in (state.get("nearby") or [])
                           if (n.get("name") or "").lower() == tname.lower()), None)
            if _match and _match.get("tx") is not None:
                kb.see_npc(tname, _match.get("tx"), _match.get("ty"))
            elif _match and _ptt.get("tx") is not None:
                kb.see_npc(tname, _ptt.get("tx") + _match.get("dx", 0),
                           _ptt.get("ty") + _match.get("dy", 0))
            # New conversation: reset the dialogue-tree pointer to the top so the
            # first menu is a root, not nested under a prior conversation's topic.
            kb.begin_npc_conversation(tname)
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
        # SAFETY NET: a driver-only pseudo-action (e.g. "look", which the
        # look-handler EARLIER in the turn normally converts, but a later guard
        # can re-introduce) must never reach the engine as 'unknown action
        # type'. Convert any such leftover to a harmless wait right before
        # dispatch. (Engine-valid types pass through untouched.)
        if isinstance(action, dict) and action.get("type") == "look":
            action = {"type": "wait", "_counted": True}
        # Guard-firing stats: if the final reason for this turn is a guard
        # intervention, tally its category for the inspector.
        _gc = _guard_category(reason)
        if _gc:
            kb.record_guard(_gc)
        result = exult.act(action)
    _ok = result.get("ok") if isinstance(result, dict) else None
    # Canonical action type for the post-dispatch outcome tracking below. Assign
    # here so it's always bound regardless of which dispatch branch ran (a
    # missing assignment on the talk/answer path caused UnboundLocalError,
    # crashing every turn before the overlay/stats were written).
    _atype = action.get("type") if isinstance(action, dict) else "?"
    # DESCEND outcome tracking: if the engine reports no REACHABLE way down from
    # this spot, remember it so the guard walks to a different platform edge and
    # retries, instead of the wedge guard thrashing in place (fortress-gateway
    # trap). Clear the counter on a successful descent.
    if _atype == "descend":
        if isinstance(result, dict) and result.get("ok") and not result.get("already_ground"):
            session["descend_fail"] = 0
        elif isinstance(result, dict) and result.get("ok") is False:
            session["descend_fail"] = session.get("descend_fail", 0) + 1
    # Skip if this was a memory/info tool (recall/quests/map/look/annotate) that
    # was already recorded under its real name before being converted to a wait.
    if not (isinstance(action, dict) and action.get("_counted")):
        kb.record_tool(_atype, _ok)
        # DURABLE lifetime tally of physical-investigation actions (survives
        # runs) so the agent can see whether it ever actually searches the world
        # vs only talking. Only count genuine (ok) engine actions.
        if _ok is not False and _atype in (
                "loot", "pickup", "take", "open", "read", "talk",
                "close", "equip", "unequip", "drop", "attack", "combat",
                "unlock", "use_key", "use", "give"):
            kb.record_activity(_atype)
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
        # Short, human result for the turn-log table (not the full JSON).
        _shortres = ""
        if isinstance(result, dict):
            if result.get("error"):
                _shortres = "blocked: " + str(result["error"])[:40]
            elif result.get("looted"):
                _shortres = "looted " + str(result["looted"])[:30]
            elif result.get("item"):
                _shortres = "got " + str(result["item"])[:24]
            elif result.get("did"):
                _shortres = "ok" if result.get("ok", True) else "failed"
        window.set_action({"action": _action_phrase(_disp), "result": _shortres})
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
                 "equip", "unequip", "drop", "use", "unlock", "attack", "give"):
        p0 = state.get("player") or {}
        detail = action.get("name") or action.get("dir") or ""
        outcome = ""
        if atype == "open":
            if _ok:
                _kind = result.get("kind") if isinstance(result, dict) else None
                if _kind == "door":
                    outcome = " -> opened the door"
                else:
                    _contents = result.get("contents") if isinstance(result, dict) else None
                    _cn = result.get("count") if isinstance(result, dict) else None
                    if _cn:
                        outcome = f" -> opened; inside: {', '.join(_contents)}"
                    else:
                        outcome = " -> opened; EMPTY (nothing inside)"
                        pp = state.get("player") or {}
                        kb.mark_searched_empty(pp.get("tx", 0), pp.get("ty", 0),
                                               action.get("name") or "container")
            else:
                _e = result.get("error", "") if isinstance(result, dict) else ""
                outcome = f" -> could not open: {_e}" if _e else " -> nothing to open here"
        elif atype == "loot":
            if _ok:
                looted_str = result.get("looted") if isinstance(result, dict) else None
                took_n = result.get("count") if isinstance(result, dict) else None
                is_empty = result.get("empty") if isinstance(result, dict) else None
                if took_n:
                    outcome = f" -> LOOTED ALL: {looted_str} ({took_n})"
                    pp = state.get("player") or {}
                    session.setdefault("looted_spots", set()).add(
                        (pp.get("tx", 0), pp.get("ty", 0)))
                elif is_empty or not looted_str:
                    outcome = " -> EMPTY (nothing to take)"
                    pp = state.get("player") or {}
                    kb.mark_searched_empty(pp.get("tx", 0), pp.get("ty", 0),
                                           action.get("name") or "container")
                else:
                    outcome = f" -> LOOTED ALL: {looted_str}"
            else:
                _e = result.get("error", "") if isinstance(result, dict) else ""
                outcome = f" -> could not loot: {_e}" if _e else " -> nothing to loot here"
        elif atype in ("pickup", "take"):
            if _ok:
                outcome = f" -> got {result.get('item')}"
                # THEFT signal from the engine: the item was someone's property.
                # Surface it prominently so the agent can weigh the consequence
                # (angered owners/guards; possible quest/nav item). We do NOT
                # block or undo it - the agent owns the decision.
                if isinstance(result, dict) and result.get("stolen"):
                    outcome += " (STOLEN - owned property!)"
                    _w = result.get("warning") or (
                        "That item was someone's property - taking it is theft.")
                    session["last_bump"] = _w
            else:
                _err = result.get("error", "") if isinstance(result, dict) else ""
                outcome = f" -> could not take: {_err}" if _err else " -> could not take"
                # If the item is just TOO FAR, surface the guidance prominently
                # so the agent walks over on its own (no auto-nav puppeteering).
                if _err and "too far" in _err.lower():
                    session["last_bump"] = _err
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
        elif atype == "use":
            if _ok:
                outcome = f" -> used {result.get('object', detail)}"
            else:
                _e = result.get("error", "") if isinstance(result, dict) else ""
                outcome = f" -> could not use: {_e}" if _e else " -> could not use"
                if _e and "too far" in _e.lower():
                    session["last_bump"] = _e
        elif atype == "unlock":
            if _ok:
                outcome = " -> UNLOCKED it"
            else:
                _e = result.get("error", "") if isinstance(result, dict) else ""
                outcome = f" -> could not unlock: {_e}" if _e else " -> could not unlock"
                if _e:
                    session["last_bump"] = _e
        elif atype == "attack":
            outcome = (f" -> attacking {result.get('target', detail)}"
                       if _ok else " -> no target to attack")
        elif atype == "give":
            if _ok:
                outcome = f" -> gave {result.get('item', '')} to {result.get('to', '')}"
            else:
                _e = result.get("error", "") if isinstance(result, dict) else ""
                outcome = f" -> could not give: {_e}" if _e else " -> could not give"
                if _e:
                    session["last_bump"] = _e
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
                         + f" @({p0.get('tx')},{p0.get('ty')})" + outcome,
                         reason=reason)
    elif atype in ("goto", "move"):
        # Log movement too, but COLLAPSED per target so a movement LOOP shows as
        # a single counted line (e.g. 'move toward stairs x22 (NO progress)')
        # rather than flooding the log. Target = the goto name/coords or move dir;
        # 'progressed' = whether the avatar actually changed tile.
        _p1 = state.get("player") or {}
        _tgt = (action.get("name")
                or (f"({action.get('tx')},{action.get('ty')})" if "tx" in action else None)
                or action.get("dir") or "somewhere")
        _movedok = (isinstance(result, dict) and result.get("ok")
                    and not result.get("blocked"))
        # progressed only if position actually changed since last turn
        _progressed = (_p1.get("tx"), _p1.get("ty")) != session.get("prev_xy_for_log")
        session["prev_xy_for_log"] = (_p1.get("tx"), _p1.get("ty"))
        kb.record_move(str(_tgt), moved=(_movedok and _progressed), reason=reason)
    # A successful pickup/take/loot/open changes the world -> NPCs may now have
    # new dialogue, so allow revisiting them.
    if atype in ("pickup", "take", "loot", "open") and isinstance(result, dict) and result.get("ok"):
        kb.reset_talk_gate()
    # A looted/opened body/container OR a spot where we picked something up is a
    # notable location -> auto-mark it so the agent can find its way back even
    # if it never annotates on its own (e.g. returning to a crime scene).
    if atype in ("loot", "open", "pickup", "take") and isinstance(result, dict) and result.get("ok"):
        pp = state.get("player") or {}
        tgt = result.get("target") or result.get("item") or action.get("name") or "spot"
        verb = "opened" if atype == "open" else ("looted" if atype == "loot" else "found items at")
        kb.record_place(f"where I {verb} {tgt}", pp.get("tx", 0), pp.get("ty", 0),
                        kind="marked")
        # Auto-resolve 'investigate/search/find <subject>' quests now that we've
        # actually examined/collected that subject.
        for _t in kb.auto_resolve_examine_quests(str(tgt)):
            print(f"[{step:03d}] auto-resolved quest (examined {tgt}): {_t}")

    p = state.get("player") or {}
    # Clean, aligned TURN LOG line for the driver console window (mirrors the
    # GUI table columns: Turn | Action | Reasoning | Result). One line, capped
    # at ~200 chars. The verbose raw line below still goes to the log file.
    try:
        _act = _action_phrase(action)
    except Exception:
        _act = str(action)
    _act = str(_act).replace("\n", " ").strip()
    _rsn = (reason or "").replace("\n", " ").strip()
    _res = result
    if isinstance(result, dict):
        _res = (result.get("did") or result.get("error")
                or result.get("ok") or "")
        for _k in ("target", "looted", "item"):
            if result.get(_k):
                _res = f"{_res}: {result.get(_k)}"
                break
    _res = str(_res).replace("\n", " ").strip()
    # Column budget so the whole line stays within ~100 chars:
    # "TURN " + 5 (turn) + " | " + 18 (action) + " | " + 45 (reason)
    # + " | " + 18 (result) = ~98.
    _A, _R, _S = 18, 45, 18

    def _fit(s, n):
        s = s if len(s) <= n else (s[: n - 1] + "\u2026")
        return f"{s:<{n}}"
    line = (f"TURN {step:>5} | {_fit(_act, _A)} | {_fit(_rsn, _R)} | "
            f"{_res[:_S]}")
    print(line[:100], flush=True)
    print(f"[{step:03d}] pos=({p.get('tx')},{p.get('ty')}) "
          f"conv={state.get('conversation_active')} "
          f"reason={reason!r} action={action} -> {result}"
          + (f"  [{session.get('last_ctx')}]" if session.get('last_ctx') else ""))
    # Overlay text for the video stream (ffmpeg drawtext reads this file live).
    # Shows what the LLM is doing + why, so viewers see the reasoning.
    if getattr(args, "overlay_file", None):
        _write_overlay(args.overlay_file, step, action, reason,
                       thinking=session.get("last_thinking", ""))
    time.sleep(args.delay)


def _guard_category(reason: str) -> str:
    """Map a '(guard) <text>' reason to a short, STABLE category so guard-firing
    counts group sensibly (many guards share a theme). Returns '' for
    non-guard reasons (the model's own reasoning), which are not counted."""
    r = (reason or "").strip()
    if not r.startswith("(guard)"):
        return ""
    r = r[len("(guard)"):].strip().lower()
    _CATS = [
        ("thrash", ("thrashing",)),
        ("arrival", ("arrived at", "at landmark", "at target with a container")),
        ("wedged/oscillation", ("wedged", "oscillation trap", "flood-fill",
                                 "trap;", "committing to far tile")),
        ("goto-not-moving", ("goto not moving", "goto stuck", "route impassable")),
        ("stuck-in-place", ("stuck near", "stuck looping", "stuck;", "already looked here")),
        ("narrate-but-wait", ("narrated but waited",)),
        ("conversation-coerce", ("invalid ", "coerced ", "same answer looping",
                                  "ending conversation", "exhausting dialogue",
                                  "leaving", "no answers yet", "conversation hung",
                                  "advancing npc dialog", "no conversation open")),
        ("greet-new-npc", ("greeting new person",)),
        ("search-guard", ("nothing to search", "corpse not lootable",
                           "body already looted", "body empty", "emptied body")),
        ("container/loot", ("container open", "done looting", "container here")),
        ("stairs/elevation", ("stairs unclimbable", "climbing ramp",
                               "approaching", "elevated", "descend")),
        ("open-guard", ("repeated open", "no door")),
        ("move-normalize", ("move dx", "move with zero delta")),
        ("salvage", ("salvaged",)),
        ("explore", ("exploring", "no active quest lead")),
    ]
    for cat, keys in _CATS:
        if any(k in r for k in keys):
            return cat
    return "other-guard"


def _write_overlay(path: str, step: int, action: dict, reason: str,
                   thinking: str = "") -> None:
    """Write the stream overlay: the agent's REASONING (and raw THINKING if the
    model emits it) - NOT the action. Viewers already see the action happen in
    the game; the interesting part is *why*. Sanitized so ffmpeg drawtext
    (textfile=...:reload=1) renders it safely; pixel-wrapped to fill the frame."""
    try:
        import re as _re
        def clean(s: str) -> str:
            s = _re.sub(r"[\r\n]+", " ", str(s))
            s = _re.sub(r"[:%\\']", " ", s)       # chars drawtext treats specially
            s = _re.sub(r"\s+", " ", s).strip()
            return s

        rsn = (reason or "").strip()
        if rsn.startswith("(guard)"):
            rsn = rsn[len("(guard)"):].strip()
        thk = clean(thinking or "")
        rsn = clean(rsn)

        # The overlay font (LiberationSans) is PROPORTIONAL, so wrapping by a
        # fixed character count either overflows (many caps) or wastes width
        # (mostly lowercase). Wrap by MEASURED pixel width instead, using the
        # real font metrics, so each line fills the frame. Falls back to a
        # conservative char estimate if the font can't be measured.
        # Frame is 512px wide; drawtext box at x=10, boxborder 8; leave a small
        # right margin -> usable text width ~490px.
        _PX_BUDGET = 490
        _FONT_PATH = os.environ.get(
            "OVERLAY_FONT",
            "/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf")
        _FONT_SIZE = int(os.environ.get("OVERLAY_FONTSIZE", "26"))
        _measure = None
        try:
            from PIL import ImageFont  # metrics only; no image is rendered
            _ttf = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
            _measure = lambda s: _ttf.getlength(s)
        except Exception:
            # ~0.55*fontsize avg advance for Liberation Sans lowercase text.
            _avg = max(1.0, 0.55 * _FONT_SIZE)
            _measure = lambda s: len(s) * _avg

        def wrap_px(s: str, budget: int, maxlines: int) -> list:
            out, cur = [], ""
            for w in s.split(" "):
                trial = (cur + " " + w).strip()
                if cur and _measure(trial) > budget:
                    out.append(cur)
                    cur = w
                    if len(out) == maxlines:
                        return out
                else:
                    cur = trial
            if cur and len(out) < maxlines:
                out.append(cur)
            return out

        parts = []
        # Reasoning first (the "why"), up to 3 lines filling the frame width.
        if rsn:
            parts.extend(wrap_px(rsn, _PX_BUDGET, 3))
        # If the model emitted raw thinking (reasoning models), show a line or
        # two of it below the reasoning, prefixed so viewers can tell them apart.
        if thk:
            _room = max(0, 4 - len(parts))
            if _room:
                parts.extend(wrap_px("(thinking) " + thk, _PX_BUDGET, _room))
        text = "\n".join(parts) if parts else " "
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)    # atomic so drawtext never reads a half file
    except Exception:
        pass






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
        # Start looping background music for the stream. The driver holds the
        # single-client bridge, so an external play.py can't get in once we're
        # connected - starting it here is the reliable path. A fresh --newgame
        # never auto-starts the map theme.
        if getattr(args, "music", True):
            try:
                r = exult.act({"type": "play_music",
                               "track": args.music_track, "repeat": 1})
                print(f"[+] Music: {r}")
            except Exception as e:
                print(f"[!] Music start failed: {e}")
        _last_game_save = time.time()    # wall-clock gate for periodic saves
        for step in range(args.steps):
            try:
                _do_turn(args, window, ollama, exult, step, recent_positions, kb, session)
                if args.memory_file and step % 5 == 0:
                    kb.save(args.memory_file)
                # Periodically save the GAME so progress survives a crash/close.
                # Wall-clock gated (default 30 min) - NOT per-turn - because an
                # in-game save is a synchronous full-savegame disk write on the
                # engine's main thread and hitches video/gameplay; doing it often
                # is both costly and pointless.
                if time.time() - _last_game_save >= args.save_interval:
                    try:
                        r = exult.act({"type": "save"})
                        saved_this_run = True
                        _last_game_save = time.time()
                        print(f"[{step:03d}] game saved (periodic) -> {r}")
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


def _load_agent_env() -> None:
    """Load operator/deployment config from a gitignored 'agent.env' (KEY=VALUE)
    next to this module, into os.environ (without overriding already-set vars).
    Keeps deployment-specific values (LLM host, model, bridge host/port) OUT of
    the tracked source. Real env vars take precedence."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.env")
    if not os.path.isfile(path):
        return
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


def main() -> int:
    _load_agent_env()
    ap = argparse.ArgumentParser(description="Drive Exult with an Ollama LLM.")
    # Defaults come from agent.env / environment (deployment-specific), with
    # only GENERIC loopback fallbacks in code - no server names, model, or
    # secrets are baked into the tracked source.
    ap.add_argument("--host", default=os.environ.get("AGENT_BRIDGE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("AGENT_BRIDGE_PORT", "45999")))
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", ""))
    ap.add_argument("--ollama-host",
                    default=os.environ.get("AGENT_OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between turns")
    ap.add_argument("--dry-run", action="store_true", help="skip Ollama; scripted moves")
    ap.add_argument("--show-thoughts", action="store_true", help="open the local LLM thinking window (tkinter)")
    ap.add_argument("--inspector", action="store_true",
                    help="serve the inspector over HTTP so a browser on another "
                         "machine (e.g. a Windows box) can watch all agent data")
    ap.add_argument("--inspector-port", type=int, default=8092,
                    help="port for the remote inspector (default 8092)")
    ap.add_argument("--inspector-host", default="0.0.0.0",
                    help="bind address for the remote inspector (default all interfaces)")
    ap.add_argument("--screenshot", action="store_true",
                    help="write a PNG each turn (for stream.py --from-file)")
    ap.add_argument("--music", action="store_true", default=True,
                    help="start looping background music on connect (default on)")
    ap.add_argument("--no-music", dest="music", action="store_false",
                    help="do not auto-start music")
    ap.add_argument("--music-track", type=int, default=9,
                    help="music track number to loop (default 9)")
    ap.add_argument("--overlay-file", default="/tmp/exult_overlay.txt",
                    help="write current action+reasoning here each turn for the "
                         "stream's on-screen text overlay (ffmpeg drawtext reads it)")
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
    ap.add_argument("--save-interval", type=float, default=1800.0,
                    help="minimum seconds between in-game saves (default 1800 "
                         "= 30 min); an in-game save is a synchronous disk write "
                         "that hitches the stream, so keep it infrequent")
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
    ap.add_argument("--log-file", default=None,
                    help="also mirror all console output to this file (a "
                         "built-in tee) so the turn log is visible LIVE in the "
                         "console AND saved. Robust and headless-friendly.")
    args = ap.parse_args()

    # Built-in tee: mirror stdout/stderr to --log-file while still printing to
    # the console window, so the turn log is visible live (and works headless).
    if args.log_file:
        try:
            _lf = open(args.log_file, "w", encoding="utf-8", buffering=1)

            class _Tee:
                def __init__(self, *streams):
                    self._streams = streams

                def write(self, s):
                    for st in self._streams:
                        try:
                            st.write(s)
                        except Exception:
                            pass
                    return len(s)

                def flush(self):
                    for st in self._streams:
                        try:
                            st.flush()
                        except Exception:
                            pass

            sys.stdout = _Tee(sys.__stdout__, _lf)
            sys.stderr = _Tee(sys.__stderr__, _lf)
        except Exception as _e:
            print(f"[!] could not open --log-file {args.log_file}: {_e}",
                  file=sys.stderr)

    ollama = None
    if not args.dry_run:
        if not args.model:
            print("[!] No model specified. Set AGENT_MODEL in agent.env, or pass "
                  "--model <name>, or use --dry-run.", file=sys.stderr)
            return 2
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

    if args.inspector:
        # Remote inspector: same interface as the tkinter window, but served
        # over HTTP so a browser on another machine (e.g. a Windows box) can
        # watch. Runs the server on its own thread; the game loop runs normally.
        from inspector_server import InspectorServer
        window = InspectorServer(host=args.inspector_host, port=args.inspector_port)
        window.start()
        run_loop(args, window, ollama)
    else:
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
