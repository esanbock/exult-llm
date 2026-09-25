"""Non-intrusive progress monitor for the long observation run.

Reads agent_memory.json + driver_log.txt only (never touches the TCP socket the
driver holds). Prints a compact status snapshot and appends it to
monitor_history.tsv so we can see trends across the multi-hour run.

Also detects likely-stuck conditions (few distinct positions over the last N
turns, a spike in parse-fails, or a guard firing repeatedly) and prints a
DIAGNOSIS line so the operator knows whether a driver change is warranted.
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MEM = os.path.join(HERE, "agent_memory.json")
LOG = os.path.join(HERE, "driver_log.txt")
HIST = os.path.join(HERE, "monitor_history.tsv")


def _read_log():
    try:
        with open(LOG, encoding="utf-8", errors="ignore") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def _read_mem():
    try:
        with open(MEM, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    lines = _read_log()
    mem = _read_mem()
    pos_lines = [l for l in lines if re.match(r"^\[\d{3,}\] pos=", l)]
    turns = len(pos_lines)
    parse_fails = sum(1 for l in lines if "parse-fail" in l)

    def pos_of(l):
        m = re.search(r"pos=\(([\d,]+)\)", l)
        return m.group(1) if m else None

    all_pos = [pos_of(l) for l in pos_lines if pos_of(l)]
    distinct = len(set(all_pos))
    # Movement over the last 20 turns (stuck detector).
    recent = all_pos[-20:]
    recent_distinct = len(set(recent))
    # Guard-fire counts in the last 40 lines (loop detector).
    tail = lines[-80:]
    guard_fires = sum(1 for l in tail if "(guard)" in l or "guard:" in l)

    q = mem.get("quests", {})
    resolved = sum(1 for v in q.values() if v.get("status") == "done")
    topics = len(mem.get("topics", {}))
    notes = sum(len(v.get("notes", [])) for v in mem.get("npcs", {}).values())
    summ = len(mem.get("episodic_summary", ""))
    npcs = len(mem.get("npcs", {}))

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    status = (f"[{ts}] turns={turns} pfail={parse_fails} distinct={distinct} "
              f"recent20_distinct={recent_distinct} guard80={guard_fires} "
              f"quests={len(q)} resolved={resolved} topics={topics} "
              f"npcs={npcs} notes={notes} summary={summ}c")
    print(status)

    # Diagnosis heuristics (advisory only).
    diag = []
    # In-conversation turns are legitimately stationary; don't flag those as
    # stuck. Count how many of the recent lines are dialogue-advancing.
    convo_tail = sum(1 for l in lines[-40:]
                     if "advancing NPC dialog" in l or "conv=True" in l
                     or "answer" in l or "conversation" in l)
    if recent_distinct <= 3 and turns > 20 and convo_tail < 8:
        diag.append("STUCK? <=3 distinct positions in last 20 turns")
    if parse_fails > max(3, turns * 0.05):
        diag.append(f"PARSE-FAILS high ({parse_fails})")
    if guard_fires > 30:
        diag.append(f"GUARDS firing a lot ({guard_fires}/80 lines) - possible loop")
    if turns == 0:
        diag.append("NO TURNS logged - driver may be down or just started")
    print("DIAGNOSIS: " + ("; ".join(diag) if diag else "healthy"))

    # ---- CHECKPOINT PROGRESS (operator-only; TextQuests-style objective read) -
    # Trinsic escape sequence, inferred from OBSERVABLE evidence (notes,
    # transcripts, looted events, action log) - NOT from the LLM's self-reported
    # quests (which inflate on false 'done'). This gives a truer progress read
    # than quest counts. It is diagnostic and never shown to the LLM.
    hay = []
    for r in mem.get("npcs", {}).values():
        hay += [t.get("said", "") for t in r.get("transcript", [])]
        hay += list(r.get("notes", []))
    hay += [e.get("text", "") if isinstance(e, dict) else str(e)
            for e in mem.get("action_history", [])]
    hay_s = " ".join(hay).lower()
    log_s = "\n".join(lines).lower()
    looted = "looted" in log_s
    checkpoints = [
        ("met the Mayor / got the task", "finnigan" in hay_s or "investigate" in hay_s),
        ("gathered a murder clue (hook/gargoyle)", "hook" in hay_s or "gargoyle" in hay_s),
        ("actually LOOTED a container/body", looted),
        ("found the report clue in a chest", "chest" in hay_s and looted),
        ("obtained the gate PASSWORD", "password" in hay_s and
            ("received" in hay_s or "got the password" in hay_s or "proper password" in hay_s)),
        ("obtained the ship DEED", "deed" in hay_s and
            ("received" in hay_s or "have the deed" in hay_s or "got the deed" in hay_s)),
        ("LEFT Trinsic", "left trinsic" in hay_s or "outside trinsic" in hay_s
            or "britain" in hay_s and "outside" in hay_s),
    ]
    done_n = sum(1 for _, ok in checkpoints if ok)
    print(f"CHECKPOINTS {done_n}/{len(checkpoints)} (Trinsic escape):")
    for name, ok in checkpoints:
        print(f"  [{'x' if ok else ' '}] {name}")

    # Last few reasons for context.
    reasons = []
    for l in pos_lines[-6:]:
        m = re.search(r"pos=\(([\d,]+)\).*reason='([^']{0,44})", l)
        if m:
            reasons.append(f"  ({m.group(1)}) {m.group(2)}")
    if reasons:
        print("recent:")
        print("\n".join(reasons))

    with open(HIST, "a", encoding="utf-8") as f:
        f.write(status + "\t" + ("|".join(diag) if diag else "healthy") + "\n")


if __name__ == "__main__":
    main()
