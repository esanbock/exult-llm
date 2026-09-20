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
import sys
import threading
import time

from exult_client import ExultClient
from ollama_client import OllamaClient
from thoughts_window import ThoughtsWindow

SYSTEM_PROMPT = """\
You are an autonomous agent playing Ultima VII: The Black Gate as the Avatar.
You interact ONLY through the JSON tool interface described below. You cannot
click, use a mouse, or do anything not listed as a TOOL. Think of this as an
API: the only way to affect the game is to call one TOOL per turn.

# MISSION (your purpose - let this drive every decision)
Ultima VII is a mystery and adventure. You have just arrived in the town of
Trinsic and discovered that a gruesome murder has taken place. Your long-term
goal is to investigate this mystery, follow the clues, and ultimately uncover
and defeat the hidden enemy behind it (a sinister organization called the
Fellowship and its master, the Guardian).

Play with intent, in roughly this order:
  1. INVESTIGATE: Talk to the townspeople of Trinsic (Finnigan the mayor, Petre,
     guards, the gargoyle Spark's father, etc.). Ask them about the murder, the
     victim, and anything suspicious. Exhaust their useful dialog topics.
  2. GATHER CLUES: Examine the crime scene (the stables) and note names, places,
     and leads people mention.
  3. PROGRESS: Once you have learned what Trinsic can tell you, travel onward to
     pursue the investigation (e.g. toward Britain) - but early on, focus on
     Trinsic.
Prefer actions that advance this investigation over aimless wandering. Do not
repeat the same conversation or pace back and forth. If you have already learned
what an NPC has to say, move on to a new NPC or a new place.

# PROTOCOL
Each turn you receive a STATE object (schema below) and must reply with EXACTLY
one JSON object, nothing else. Put the tool's parameters at the TOP LEVEL of
"action" (do NOT nest them under a "params" key):
  {"reason": "<one short sentence>", "action": {"type": "move", "dir": "n"}}
  {"reason": "greet Iolo", "action": {"type": "talk", "name": "Iolo"}}
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
  nearby (list)               - NPCs you can see: {name, dx, dy, in_party, dead}
                                dx>0 = east, dx<0 = west, dy>0 = south, dy<0 = north
  objects (list)              - items on the ground: {name, dx, dy}
  grid (string)               - top-down ASCII map centered on you (@):
                                  @ you   & NPC   x body   * object
                                  + closed door   / open door
                                  # blocked/impassable   . open ground
                                north=up, south=down, east=right, west=left
  doors (list)                - nearby doors: {name, dx, dy, closed}

# TOOLS (the complete list of things you can do - nothing else is possible)
  move    - Walk one step. params: {"dir": one of n,s,e,w,ne,nw,se,sw}
            Use the grid: step onto '.' tiles, never into '#'. To reach an
            NPC/object, move toward its (dx,dy).
  stop    - Stop walking. params: none.
  talk    - START a conversation with a nearby NPC. params: {"name": "<NPC name>"}
            This is the ONLY way to begin dialog. Walking next to an NPC does
            NOT start dialog. You do NOT need to be adjacent - it finds the
            named NPC in your view and opens the conversation. Works for both
            townspeople and party companions (in_party:true).
  open    - Open (or close) the nearest door within a few tiles. params: none.
            Doors show as '+' (closed) or '/' (open) on the grid and in "doors".
            A closed door ('+') blocks you - walk adjacent to it, "open" it, then
            "move" through the now-open ('/') doorway.
  answer  - Choose a reply during a conversation. params: {"index": <int>} (0-based
            into the "answers" list) OR {"text": "<answer text>"}.
            Only valid when conversation_active is true.
  key     - Press a key. params: {"key": "space"|"escape"|"a".."z"|"0".."9"}.
            Use "space" to advance NPC text when there is npc_text but no answers.
  combat  - Toggle combat/attack mode on or off. params: none.
  feed    - Eat food to refill your food level (prevents starving). params: none.
  wait    - Do nothing this turn. params: none.

# HOW TO DECIDE (policy)
  1. If conversation_active is true -> use "answer" (pick the index of the reply
     you want; prefer moving the conversation forward, use the "bye" reply to end).
  2. Else if conversation_in_progress is true (a conversation is open but no
     choices yet) -> use "key" with "space" to advance the NPC's text until the
     answer choices appear. Do NOT "talk" again or "move" during a conversation.
  3. Else if you want to talk to someone in "nearby" -> "talk" with their name.
     Do NOT repeatedly "move" toward them expecting dialog to auto-start.
  4. Else explore with "move", using the grid to avoid '#' and head toward
     interesting NPCs/objects. If a closed door '+' blocks your path, move next
     to it, use "open", then move through the '/' opening.
  5. If your food is low, use "feed". If threatened, "combat".

Reply with ONLY the single JSON object. No prose, no markdown.
"""


def summarize_state(state: dict, memory: dict | None = None) -> str:
    """Compact the observation to keep the prompt small and focused."""
    p = state.get("player") or {}
    nearby = state.get("nearby") or []
    objects = state.get("objects") or []
    view = {
        "player": {
            "tx": p.get("tx"), "ty": p.get("ty"),
            "hp": p.get("hp"), "dead": p.get("dead"),
        },
        "in_combat": state.get("in_combat"),
        "conversation_in_progress": state.get("conversation_in_progress"),
        "conversation_active": state.get("conversation_active"),
        "npc_text": state.get("npc_text"),
        "answers": state.get("answers"),
        "nearby": [
            {"name": n.get("name"), "dx": n.get("dx"), "dy": n.get("dy")}
            for n in nearby[:8]
        ],
        "objects": [
            {"name": o.get("name"), "dx": o.get("dx"), "dy": o.get("dy")}
            for o in objects[:8]
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
    if memory:
        view["already_talked_to"] = sorted(memory.get("talked", []))
        if memory.get("journal"):
            view["journal"] = memory["journal"][-8:]  # recent notes
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


def _do_turn(args, window, ollama, exult, step, recent_positions, memory) -> None:
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

    # Record NPC dialog into the journal (dedup consecutive dups).
    npc_text = state.get("npc_text")
    if npc_text and (not memory["journal"] or memory["journal"][-1] != npc_text):
        memory["journal"].append(npc_text)

    if args.dry_run:
        reason, action = scripted_reply(step, state)
    else:
        reply = ollama.chat(SYSTEM_PROMPT, summarize_state(state, memory))
        reason, action = parse_reply(reply)
        if window.available:
            window.set_thinking(reply)

    if window.available and args.dry_run:
        window.set_thinking(reason)

    # --- Driver-side guards to keep behavior sane ------------------------
    # 1) If a conversation is open but no choices are shown yet, advance text.
    if state.get("conversation_in_progress") and not state.get("conversation_active"):
        action = {"type": "key", "key": "space"}
        reason = "(guard) advancing NPC dialog"
    # 2) Anti-chase: if the model keeps trying to MOVE toward a talkable NPC
    #    that is already close, just talk to it instead of pacing.
    elif (isinstance(action, dict) and action.get("type") == "move"
          and not state.get("conversation_in_progress")):
        pos = (state.get("player") or {}).get("tx"), (state.get("player") or {}).get("ty")
        recent_positions.append(pos)
        if len(recent_positions) > 6:
            recent_positions.pop(0)
        stuck = len(recent_positions) >= 4 and len(set(recent_positions)) <= 2
        close_npcs = [n for n in (state.get("nearby") or [])
                      if not n.get("dead") and abs(n.get("dx", 99)) <= 4
                      and abs(n.get("dy", 99)) <= 4]
        if stuck and close_npcs:
            target = min(close_npcs, key=lambda n: abs(n["dx"]) + abs(n["dy"]))
            action = {"type": "talk", "name": target["name"]}
            reason = f"(guard) stuck near {target['name']}; talking instead of moving"
            recent_positions.clear()

    # "talk" is a top-level command, not an act() action.
    if isinstance(action, dict) and action.get("type") == "talk":
        tname = action.get("name", "")
        if tname:
            memory["talked"].add(tname)
        result = exult.talk(tname)
    else:
        result = exult.act(action)
    if window.available:
        window.set_action(json.dumps(action) + "\n\n-> " + json.dumps(result))

    p = state.get("player") or {}
    print(f"[{step:03d}] pos=({p.get('tx')},{p.get('ty')}) "
          f"conv={state.get('conversation_active')} "
          f"reason={reason!r} action={action} -> {result}")
    time.sleep(args.delay)


def run_loop(args, window: ThoughtsWindow, ollama) -> None:
    recent_positions: list = []
    memory: dict = {"talked": set(), "journal": []}
    exult = ExultClient(args.host, args.port)
    try:
        exult.connect()
        print(f"[+] Connected to Exult bridge: {exult.ping()}")
        for step in range(args.steps):
            try:
                _do_turn(args, window, ollama, exult, step, recent_positions, memory)
            except (ConnectionError, OSError) as e:
                # Lost the game connection (Exult closed, or a probe stole the
                # socket). Try to reconnect and keep going.
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

    window = ThoughtsWindow()
    if args.show_thoughts:
        window.start()

    if window.available and args.show_thoughts:
        # Tkinter must own the main thread; run the game loop in a worker.
        worker = threading.Thread(target=run_loop, args=(args, window, ollama), daemon=True)
        worker.start()
        window.mainloop()
    else:
        run_loop(args, window, ollama)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
