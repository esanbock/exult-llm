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
  * FOLLOW LEADS: when someone mentions a person, place, item, or event, treat
    it as a lead worth pursuing. Use your journal to remember what you learned.
  * EXAMINE THE WORLD: investigate notable objects, bodies, and containers you
    come across; collect items that look important (keys, notes, valuables).
  * MAKE PROGRESS: prefer purposeful action over aimless wandering or repeating
    yourself. If you have exhausted a person or place, move on to somewhere new.
  * SURVIVE: keep fed and stay alive; avoid needless danger.
You are not told the solution to anything - reason from what you observe and are
told, as a curious, capable adventurer would.

# PROTOCOL
Each turn you receive a STATE object (schema below) and must reply with EXACTLY
one JSON object, nothing else. Put the tool's parameters at the TOP LEVEL of
"action" (do NOT nest them under a "params" key):
  {"reason": "<one short sentence>", "action": {"type": "move", "dir": "n"}}
  {"reason": "greet the nearby NPC", "action": {"type": "talk", "name": "Iolo"}}
  {"reason": "pick first reply", "action": {"type": "answer", "index": 0}}

# STATE SCHEMA (what you receive each turn)
  world_loaded (bool)         - is a game world loaded
  player: {tx,ty (your tile), hp, dead, food}
  in_combat (bool)            - are you in combat mode
  conversation_in_progress (bool) - true while a conversation is open (faces shown)
  conversation_active (bool)  - true when NPC answer choices are on screen NOW
  npc_text (string|null)      - the last thing an NPC said to you
  answers (list[string])      - the reply choices you may pick (only when
                                conversation_active is true)
  nearby (list)               - NPCs you can see: {name, dx, dy, in_party, status}
                                dx>0 = east, dx<0 = west, dy>0 = south, dy<0 = north
                                status = new (never talked) | talked (may have more)
                                | exhausted (you already asked all they know for now -
                                talking again wastes turns until the situation changes)
  objects (list)              - items on the ground: {name, dx, dy}
  grid (string)               - top-down ASCII map centered on you (@):
                                  @ you   C companion   & other NPC   x body
                                  * object   + closed door (PASSAGE - openable)
                                  / open door   # wall/impassable   . open ground
                                north=up, south=down, east=right, west=left.
                                To enter a building/room head for its door
                                ('+' or '/'), NOT the '#' walls around it.
  doors (list)                - nearby doors: {name, dx, dy, closed}

# YOUR JOURNAL (you maintain this - it persists across turns)
  quests: {focus, actionable[], blocked[]}
      - focus: the single highest-priority quest you can act on now (or null)
      - actionable: quests you can work on now, sorted by priority (1=highest)
      - blocked: quests waiting on a prerequisite (see each quest's depends_on)
      Each quest: {id, title, priority, status(active|blocked|done), notes, depends_on[]}
  npc_notes: what you have recorded about nearby/known NPCs (their leads, wants)
  recent_dialogue: a running transcript of the last conversation exchanges
      ({"npc":name,"said":...} for NPC lines, {"me":...} for your replies).
      Use this to remember what you have learned and what was said earlier.
  recent_actions: your last few meaningful actions (talk/open/pickup/etc).
      Use this to avoid repeating something you just did.

# TOOLS (the complete list of things you can do - nothing else is possible)
  move    - Walk one step. params: {"dir": one of n,s,e,w,ne,nw,se,sw}
            Use the grid: step onto '.' tiles, never into '#'. To reach an
            NPC/object, move toward its (dx,dy).
  goto    - PATHFIND to a destination and walk there automatically, routing
            around walls and THROUGH doorways. params: {"name": "<NPC/object>"}
            to go to the nearest thing with that name, OR {"tx":<int>,"ty":<int>}
            for an absolute tile. PREFER "goto" over many "move" steps when you
            want to reach a specific NPC, item, or building entrance.
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
  search  - Open the nearest body or container to see/take what is inside.
            params: none. Works on any body ('x' on the grid) or container/chest
            to reveal its contents. After searching, use "pickup" to take items.
  pickup  - Take a nearby item off the ground into your inventory.
            params: {"name": "<item name>"} (optional; omit to grab the closest
            takeable item). Use this to collect any useful item that appears as
            '*' on the grid or in "objects" (keys, weapons, food, gold, etc.).
  answer  - Choose a reply during a conversation. params: {"index": <int>} (0-based
            into the "answers" list) OR {"text": "<answer text>"}.
            Only valid when conversation_active is true.
  key     - Press a key. params: {"key": "space"|"escape"|"a".."z"|"0".."9"}.
            Use "space" to advance NPC text when there is npc_text but no answers.
  combat  - Toggle combat/attack mode on or off. params: none.
  feed    - Eat food to refill your food level (prevents starving). params: none.
  save    - Save the game so progress is not lost. params: none. (The driver
            also auto-saves periodically; you rarely need this.)
  wait    - Do nothing this turn. params: none.

# JOURNAL TOOLS (manage your own quest log & notes - do NOT affect the game)
  add_quest    - Record a goal you discovered. params: {"title": "...",
                 "priority": 1-9 (1=highest), "notes": "...",
                 "depends_on": ["<quest id>", ...] (optional prerequisites)}.
                 Use when an NPC gives you a task or you infer a goal. If quest B
                 requires finishing quest A first, set B.depends_on=["<A id>"].
  update_quest - Change a quest. params: {"id": "<quest id>", "status":
                 "active|blocked|done", "priority": n, "notes": "...",
                 "depends_on": [...]}. Mark a quest "done" when you complete it.
  note_npc     - Save a note about an NPC. params: {"name": "...", "note": "..."}.
                 Record leads, what they want, or what they told you.
  (These journal tools do not advance the game, so after using one, keep taking
   game actions. Use them sparingly - only to capture genuinely new information.)

# HOW TO DECIDE (policy)
  1. If conversation_active is true -> use "answer" (pick the index of the reply
     you want). Explore genuinely NEW topics, but once you have asked the useful
     ones (or you see the same choices again), END the conversation by choosing
     the "bye"/"leave" reply. Do NOT keep re-picking the same topics in a loop.
  2. Else if conversation_in_progress is true (a conversation is open but no
     choices yet) -> use "key" with "space" to advance the NPC's text until the
     answer choices appear. Do NOT "talk" again or "move" during a conversation.
  3. Else if you want to talk to someone in "nearby" -> "talk" with their name.
     Choose by their "status": prefer NPCs marked "new", then "talked". Do NOT
     "talk" to an NPC marked "exhausted" - you have already learned what they
     know for now, and asking again just wastes turns (a smart adventurer moves
     on). It is fine to revisit someone AFTER real progress (you completed a task
     they mentioned, found an item) - their status resets when things change.
     If everyone nearby is "exhausted", explore to a NEW area to find fresh
     people/places (use "goto" toward unexplored parts of the map).
  3b. USE YOUR JOURNAL: consult "quests" - work on the "focus" quest (highest
     priority you can act on now). When you learn a new goal, "add_quest"; when
     you finish one, mark it "done" with "update_quest"; record leads with
     "note_npc". Respect prerequisites: a "blocked" quest needs its depends_on
     quests done first, so complete those first.
  4. Else explore. To reach anywhere more than a step or two away (an NPC, an
     item, a building/entrance, a new part of town) ALWAYS use "goto" - it
     pathfinds around walls and through doors for you. Only use single "move"
     steps for tiny local adjustments, and NEVER move onto a '#' wall: on the
     grid you (@) can only step onto '.', items '*', or an open door '/'. If you
     keep bumping the same spot, you are against a wall - use "goto" to route
     around it.
  4b. PAY ATTENTION TO OBJECTS: the "objects" list names what is on the ground
     around you; bodies also show as 'x' on the grid. When something looks
     relevant to your goals or curiosity, interact with it rather than pacing:
     "goto"/move adjacent, "search" bodies and containers to see their contents,
     and "pickup" useful items. Read object names to understand your surroundings.
  5. If your food is low, use "feed". If threatened, "combat".

Reply with ONLY the single JSON object. No prose, no markdown.
"""


def summarize_state(state: dict, kb: "KnowledgeBase | None" = None) -> str:
    """Compact the observation to keep the prompt small and focused."""
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
        "in_combat": state.get("in_combat"),
        "conversation_in_progress": in_convo,
        "conversation_active": state.get("conversation_active"),
        "npc_text": state.get("npc_text") if in_convo else None,
        "answers": state.get("answers") if in_convo else [],
        "nearby": [
            {"name": n.get("name"), "dx": n.get("dx"), "dy": n.get("dy"),
             "in_party": n.get("in_party"),
             "status": (kb.talk_status(n.get("name")) if kb and n.get("name") else "new")}
            for n in nearby[:8]
        ],
        "objects": [
            {"name": o.get("name"), "dx": o.get("dx"), "dy": o.get("dy"),
             **({"body": True} if o.get("body") else {})}
            for o in objects[:14]
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
        view["already_talked_to"] = sorted(kb.npcs.keys())
        # NPC notes: focus on those currently nearby, plus recently noted.
        nearby_names = [n.get("name") for n in nearby[:8] if n.get("name")]
        notes = kb.npc_view(nearby_names)
        if notes:
            view["npc_notes"] = notes
        # Growing window of recent CONVERSATION (story/clues live here) and a
        # short window of recent ACTIONS (to avoid repeating yourself).
        dh = kb.dialogue_view(30)
        if dh:
            view["recent_dialogue"] = dh
        ah = kb.action_view(10)
        if ah:
            view["recent_actions"] = ah
    return json.dumps(view)


def format_dialog(state: dict) -> str:
    """Human-readable dialog/characters/objects panel for the thoughts window."""
    lines = []
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


def _do_turn(args, window, ollama, exult, step, recent_positions, kb, session) -> None:
    state = exult.observe()
    if window.available:
        window.update_turn(step)
        window.set_map(state.get("grid") or "(no map)")
        obs_compact = {k: v for k, v in state.items() if k != "grid"}
        window.set_observation(json.dumps(obs_compact, indent=2))
        window.set_dialog(format_dialog(state))

    # Safety net: keep the party fed so a long run can't starve.
    if args.auto_feed and step % args.auto_feed_every == 0:
        exult.act({"type": "feed", "level": 30})

    if not state.get("world_loaded"):
        if window.available:
            window.set_thinking("World not loaded yet; waiting...")
        time.sleep(args.delay)
        return

    # Record NPC dialog into the journal + dialogue history (with speaker).
    npc_text = state.get("npc_text")
    if state.get("conversation_in_progress") and npc_text:
        kb.add_journal(npc_text)
        cur_npc = session.get("current_npc", "?")
        kb.record_npc_line(cur_npc, npc_text)
        # Passively capture durable knowledge (NPC notes + task-like quests)
        # from each NEW line, so the structured memory builds up even though
        # the model rarely calls the journal tools itself.
        if npc_text != session.get("last_captured_line"):
            session["last_captured_line"] = npc_text
            kb.auto_note_from_dialogue(cur_npc, npc_text)

    if args.dry_run:
        reason, action = scripted_reply(step, state)
    else:
        reply = ollama.chat(SYSTEM_PROMPT, summarize_state(state, kb))
        reason, action = parse_reply(reply)
        if window.available:
            window.set_thinking(reply)

    if window.available and args.dry_run:
        window.set_thinking(reason)

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

    in_convo = bool(state.get("conversation_in_progress"))
    answers = state.get("answers") or []

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

    # 0) Auto-end a conversation that has gone on too long or is repeating the
    #    same answer choices (the model won't pick 'bye' on its own).
    if state.get("conversation_active") and answers:
        picked = session.setdefault("picked_answers", set())
        chosen_idx = action.get("index") if isinstance(action, dict) and action.get("type") == "answer" else None
        too_long = session.get("convo_turns", 0) >= MAX_CONVO_TURNS
        repeating = chosen_idx is not None and chosen_idx in picked and len(picked) >= max(1, len(answers) - 1)
        if too_long or repeating:
            action = _bye_action()
            reason = f"(guard) ending conversation ({'too long' if too_long else 'looping'})"
        elif chosen_idx is not None:
            picked.add(chosen_idx)

    # 1) If a conversation is open but no choices are shown yet, advance text.
    if state.get("conversation_in_progress") and not state.get("conversation_active"):
        action = {"type": "key", "key": "space"}
        reason = "(guard) advancing NPC dialog"
    else:
        # Helper: commit to exploring a distant tile so the agent leaves an
        # exhausted area instead of oscillating between adjacent NPCs.
        def _explore_far():
            p = state.get("player") or {}
            tx, ty = p.get("tx", 0), p.get("ty", 0)
            # Rotate heading every ~8 turns so it sweeps the map over time.
            headings = [(0, -18), (18, 0), (0, 18), (-18, 0),
                        (14, -14), (-14, 14), (14, 14), (-14, -14)]
            hx, hy = headings[(step // 8) % len(headings)]
            return {"type": "goto", "tx": tx + hx, "ty": ty + hy}

        # 1b) A "space"/"key" press outside a conversation does nothing.
        if (isinstance(action, dict) and action.get("type") == "key"
                and not state.get("conversation_in_progress")):
            fresh = [n for n in (state.get("nearby") or [])
                     if not n.get("dead") and talked(n.get("name")) < MAX_TALKS]
            if fresh:
                target = min(fresh, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
                action = {"type": "talk", "name": target["name"]}
                reason = f"(guard) no conversation open; talking to {target['name']}"
            else:
                action = _explore_far()
                reason = "(guard) NPCs exhausted; exploring a new area"
        # 2) Redirect re-talk to an exhausted NPC toward a fresh one.
        if isinstance(action, dict) and action.get("type") == "talk":
            nm = action.get("name", "")
            if nm and talked(nm) >= MAX_TALKS:
                fresh = [n for n in (state.get("nearby") or [])
                         if not n.get("dead") and talked(n.get("name")) < MAX_TALKS]
                if fresh:
                    target = min(fresh, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
                    action = {"type": "talk", "name": target["name"]}
                    reason = f"(guard) already talked to {nm}; trying {target['name']}"
                else:
                    action = _explore_far()
                    reason = "(guard) all nearby NPCs exhausted; exploring a new area"
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
                     and abs(n.get("dy", 99)) <= 4 and talked(n.get("name")) < MAX_TALKS]
            if stuck and fresh:
                target = min(fresh, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
                action = {"type": "talk", "name": target["name"]}
                reason = f"(guard) stuck near {target['name']}; talking instead of moving"
                recent_positions.clear()
            elif stuck:
                objs = [o for o in (state.get("objects") or [])
                        if abs(o.get("dx", 99)) <= 3 and abs(o.get("dy", 99)) <= 3
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

    # "talk" is a top-level command, not an act() action.
    if isinstance(action, dict) and action.get("type") == "talk":
        tname = action.get("name", "")
        if tname:
            kb.mark_talked(tname)
            session["current_npc"] = tname
        result = exult.talk(tname)
    else:
        result = exult.act(action)
    if window.available:
        window.set_action(json.dumps(action) + "\n\n-> " + json.dumps(result))

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

    p = state.get("player") or {}
    print(f"[{step:03d}] pos=({p.get('tx')},{p.get('ty')}) "
          f"conv={state.get('conversation_active')} "
          f"reason={reason!r} action={action} -> {result}")
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
    args = ap.parse_args()

    ollama = None
    if not args.dry_run:
        ollama = OllamaClient(model=args.model, host=args.ollama_host)
        if not ollama.is_up():
            print(
                f"[!] Ollama not reachable at {args.ollama_host}. "
                f"Start it (ollama serve) or use --dry-run.",
                file=sys.stderr,
            )
            return 2

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
