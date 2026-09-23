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
    if recent_distinct <= 3 and turns > 20:
        diag.append("STUCK? <=3 distinct positions in last 20 turns")
    if parse_fails > max(3, turns * 0.05):
        diag.append(f"PARSE-FAILS high ({parse_fails})")
    if guard_fires > 30:
        diag.append(f"GUARDS firing a lot ({guard_fires}/80 lines) - possible loop")
    if turns == 0:
        diag.append("NO TURNS logged - driver may be down or just started")
    print("DIAGNOSIS: " + ("; ".join(diag) if diag else "healthy"))

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
