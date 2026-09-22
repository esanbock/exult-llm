"""knowledge.py - Structured, queryable knowledge base for the Exult agent.

Maintains three things the LLM builds up as it plays:

  * QUESTS   - goals the agent discovers. Each quest has an id, title, status,
               priority (1 = highest), free-form notes, and optional
               dependencies (ids of quests that must be 'done' first). The KB
               computes which quests are *actionable* (active and not blocked by
               unfinished prerequisites) and sorts them by priority.
  * NPCS     - per-NPC notes (what they said, what they want, leads they gave)
               plus how many times we've spoken to them.
  * JOURNAL  - a chronological log of noteworthy dialog/observations.

The LLM manipulates this via meta-tools (add_quest/update_quest/note_npc/query),
and the driver feeds a compact, priority-sorted summary back into each prompt so
the agent works on quests opportunistically but with awareness of priority and
prerequisites.  Nothing here is game-specific.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:40] or "quest"


class KnowledgeBase:
    def __init__(self) -> None:
        # id -> {id,title,status,priority,notes,depends_on:[ids]}
        self.quests: dict[str, dict] = {}
        # name -> {name, times_talked, notes:[str]}
        self.npcs: dict[str, dict] = {}
        self.journal: list[str] = []
        # Persistent "mental map": discovered places (towns, buildings, caves,
        # landmarks) as name -> {name, tx, ty, kind, notes}. Grows over time so
        # the agent can navigate a large multi-region world.
        self.places: dict[str, dict] = {}
        # Rolling window of recent conversation exchanges (kept fairly long -
        # dialogue carries the story/clues). Each: {"npc":str,"said":str} or
        # {"me":str} for the answer the agent chose.
        self.dialogue_history: list[dict] = []
        # Short rolling window of meaningful actions taken (open/pickup/etc).
        self.action_history: list[str] = []
        # Persistent HINT history: every operator hint ever given, with the
        # step it was given at. Hints are guidance from the human and are high
        # value - we keep them all (cheap) so the agent can recall earlier
        # steering even long after a hint's short "active" window expires.
        # Each: {"step":int, "text":str}
        self.hints: list[dict] = []
        # Persistent OBSERVATION memory: notable things seen and ambient speech
        # overheard, deduped and filtered so it is not just a dump of every tile
        # every turn. Each: {"step":int, "kind":str, "text":str}
        self.observations: list[dict] = []
        # Rolling EPISODIC SUMMARY: a compact running gist of older events that
        # have scrolled out of the raw dialogue/action windows. This is how we
        # keep clues/story alive within a bounded token budget - old detail is
        # compressed into this text instead of being dropped outright.
        self.episodic_summary: str = ""

    # ----- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"quests": self.quests, "npcs": self.npcs, "journal": self.journal,
                "dialogue_history": self.dialogue_history,
                "action_history": self.action_history,
                "places": self.places,
                "hints": self.hints,
                "observations": self.observations,
                "episodic_summary": self.episodic_summary}

    @classmethod
    def load(cls, path: Optional[str]) -> "KnowledgeBase":
        kb = cls()
        if not path or not os.path.isfile(path):
            return kb
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            kb.quests = dict(data.get("quests", {}))
            kb.npcs = dict(data.get("npcs", {}))
            kb.journal = list(data.get("journal", []))
            kb.dialogue_history = list(data.get("dialogue_history", []))
            kb.action_history = list(data.get("action_history", []))
            kb.places = dict(data.get("places", {}))
            kb.hints = list(data.get("hints", []))
            kb.observations = list(data.get("observations", []))
            kb.episodic_summary = str(data.get("episodic_summary", "") or "")
        except (OSError, ValueError):
            pass
        return kb

    def save(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        except OSError:
            pass

    # ----- quests --------------------------------------------------------
    def add_quest(self, title: str, priority: int = 5, notes: str = "",
                  depends_on: Optional[list] = None, qid: Optional[str] = None,
                  status: str = "active", npc: Optional[str] = None) -> str:
        qid = qid or _slug(title)
        # Merge if it already exists (update in place).
        q = self.quests.get(qid, {"id": qid, "notes": ""})
        q["id"] = qid
        q["title"] = title or q.get("title", qid)
        q["priority"] = int(priority) if priority is not None else q.get("priority", 5)
        q["status"] = status or q.get("status", "active")
        if npc:
            q["npc"] = npc
        if notes:
            q["notes"] = (q.get("notes", "") + ("\n" if q.get("notes") else "") + notes).strip()
        if depends_on is not None:
            q["depends_on"] = [d for d in depends_on if d]
        else:
            q.setdefault("depends_on", [])
        self.quests[qid] = q
        return qid

    def update_quest(self, qid: str, **fields: Any) -> bool:
        # Allow lookup by id or by fuzzy title match.
        q = self.quests.get(qid)
        if q is None:
            for cand in self.quests.values():
                if _slug(cand.get("title", "")) == _slug(qid):
                    q = cand
                    break
        if q is None:
            # Unknown quest: create it so nothing is lost.
            self.add_quest(fields.get("title", qid), qid=_slug(qid))
            q = self.quests[_slug(qid)]
        for k, v in fields.items():
            if k == "notes" and v:
                q["notes"] = (q.get("notes", "") + ("\n" if q.get("notes") else "") + str(v)).strip()
            elif k == "priority" and v is not None:
                q["priority"] = int(v)
            elif k == "depends_on" and v is not None:
                q["depends_on"] = [d for d in v if d]
            elif k in ("status", "title") and v:
                q[k] = v
        return True

    def _is_blocked(self, q: dict) -> bool:
        for dep in q.get("depends_on", []):
            d = self.quests.get(dep)
            if d and d.get("status") != "done":
                return True
        return False

    def quest_view(self, max_actionable: int = 8, max_blocked: int = 6) -> dict:
        """Priority-sorted quests split into actionable vs blocked, plus the
        single top recommended quest to work on now.  Caps the number shown so
        the prompt stays bounded even after a long playthrough with many quests
        (the full set is always kept on disk)."""
        active = [q for q in self.quests.values() if q.get("status") not in ("done",)]
        for q in active:
            q_blocked = self._is_blocked(q)
            q["_actionable"] = (q.get("status") == "active" and not q_blocked)
        actionable = sorted([q for q in active if q["_actionable"]],
                            key=lambda q: q.get("priority", 5))
        blocked = sorted([q for q in active if not q["_actionable"]],
                        key=lambda q: q.get("priority", 5))

        def brief(q: dict) -> dict:
            b = {"id": q["id"], "title": q.get("title"), "priority": q.get("priority", 5),
                "status": q.get("status", "active"),
                "resolved": q.get("status") == "done"}
            if q.get("npc"):
                b["npc"] = q["npc"]
            if q.get("depends_on"):
                b["depends_on"] = q["depends_on"]
            if q.get("notes"):
                b["notes"] = q["notes"][-240:]
            return b

        return {
            "focus": brief(actionable[0]) if actionable else None,
            "actionable": [brief(q) for q in actionable[:max_actionable]],
            "blocked": [brief(q) for q in blocked[:max_blocked]],
            "total_open": len(active),
            "unresolved": len(active),
            "resolved": sum(1 for q in self.quests.values() if q.get("status") == "done"),
        }

    def resolve_quest(self, qid: str) -> bool:
        """Mark a quest resolved (done). Accepts id or fuzzy title."""
        return self.update_quest(qid, status="done")

    def has_unresolved(self) -> bool:
        return any(q.get("status") != "done" for q in self.quests.values())

    # ----- human-readable summaries (for the inspector GUI) --------------
    def quests_pretty(self) -> str:
        qv = self.quest_view(max_actionable=20, max_blocked=20)
        lines = []
        f = qv.get("focus")
        lines.append("FOCUS: " + (f["title"] if f else "(none)"))
        lines.append(f"resolved {qv.get('resolved',0)} / unresolved {qv.get('unresolved',0)}")
        lines.append("")
        lines.append("ACTIONABLE:")
        for q in qv.get("actionable", []):
            npc = f" [{q['npc']}]" if q.get("npc") else ""
            lines.append(f"  P{q.get('priority',5)} {q['title']}{npc}")
        if qv.get("blocked"):
            lines.append("")
            lines.append("BLOCKED (needs prereq):")
            for q in qv["blocked"]:
                dep = ",".join(q.get("depends_on", []))
                lines.append(f"  P{q.get('priority',5)} {q['title']} <- {dep}")
        return "\n".join(lines)

    def npcs_pretty(self, nearby_names=None) -> str:
        lines = []
        near = set(n.lower() for n in (nearby_names or []))
        # Show nearby NPCs first, then the rest, with talk status + a note.
        items = list(self.npcs.items())
        items.sort(key=lambda kv: (kv[0].lower() not in near, -kv[1].get("times_talked", 0)))
        for name, r in items[:20]:
            tag = "*here* " if name.lower() in near else ""
            st = self.talk_status(name)
            note = (r.get("notes") or [""])[-1]
            note = (note[:60] + "...") if len(note) > 60 else note
            lines.append(f"{tag}{name} ({st}, x{r.get('times_talked',0)}): {note}")
        return "\n".join(lines) if lines else "(no NPCs met yet)"

    # ----- npcs ----------------------------------------------------------
    def note_npc(self, name: str, note: str = "") -> None:
        if not name:
            return
        rec = self.npcs.setdefault(name, {"name": name, "times_talked": 0, "notes": []})
        if note:
            if not rec["notes"] or rec["notes"][-1] != note:
                rec["notes"].append(note)
                rec["notes"] = rec["notes"][-12:]

    def mark_talked(self, name: str) -> None:
        if not name:
            return
        rec = self.npcs.setdefault(name, {"name": name, "times_talked": 0, "notes": []})
        rec["times_talked"] = rec.get("times_talked", 0) + 1
        # Per-progress-epoch counter (used to decide "talked enough for NOW").
        rec["talked_this_epoch"] = rec.get("talked_this_epoch", 0) + 1

    def times_talked(self, name: str) -> int:
        return self.npcs.get(name, {}).get("times_talked", 0)

    def see_npc(self, name: str, tx: int, ty: int) -> None:
        """Remember where we last saw an NPC, so a quest that involves them can
        be navigated to even after we walk away."""
        if not name:
            return
        rec = self.npcs.setdefault(name, {"name": name, "times_talked": 0, "notes": []})
        rec["last_pos"] = [int(tx), int(ty)]

    def npc_last_pos(self, name: str):
        return self.npcs.get(name, {}).get("last_pos")

    def talked_recently(self, name: str) -> int:
        """How many times we've talked to this NPC since the last meaningful
        progress. Resets when the world changes (quest/item/area), so NPCs
        become worth revisiting after you have accomplished something."""
        return self.npcs.get(name, {}).get("talked_this_epoch", 0)

    def reset_talk_gate(self) -> None:
        """Meaningful progress happened - allow revisiting NPCs (they may now
        have new dialogue)."""
        for rec in self.npcs.values():
            rec["talked_this_epoch"] = 0

    def talk_status(self, name: str, exhausted_at: int = 3) -> str:
        """Classify how worthwhile talking to this NPC is right now:
        'new' (never talked), 'talked' (spoken to but may have more), or
        'exhausted' (asked enough since last progress - unlikely to offer new
        info until the situation changes)."""
        rec = self.npcs.get(name)
        if not rec or rec.get("times_talked", 0) == 0:
            return "new"
        if rec.get("talked_this_epoch", 0) >= exhausted_at:
            return "exhausted"
        return "talked"

    def npc_view(self, names: Optional[list] = None) -> dict:
        """NPC notes; if names given, only those, else all known."""
        src = self.npcs
        if names:
            lname = {n.lower() for n in names}
            src = {k: v for k, v in self.npcs.items() if k.lower() in lname}
        return {n: {"times_talked": r.get("times_talked", 0),
                    "notes": r.get("notes", [])[-6:]}
                for n, r in src.items()}

    # ----- journal -------------------------------------------------------
    def add_journal(self, text: str) -> None:
        if text and (not self.journal or self.journal[-1] != text):
            self.journal.append(text)
            self.journal = self.journal[-500:]

    # ----- dialogue history (long window - story/clues live here) --------
    DIALOGUE_WINDOW = 30

    def record_npc_line(self, npc: str, said: str) -> None:
        if not said:
            return
        entry = {"npc": npc or "?", "said": said}
        if self.dialogue_history and self.dialogue_history[-1] == entry:
            return
        self.dialogue_history.append(entry)
        self.dialogue_history = self.dialogue_history[-self.DIALOGUE_WINDOW:]

    def record_my_reply(self, text: str) -> None:
        if not text:
            return
        self.dialogue_history.append({"me": text})
        self.dialogue_history = self.dialogue_history[-self.DIALOGUE_WINDOW:]

    def dialogue_view(self, limit: int = 30) -> list:
        return self.dialogue_history[-limit:]

    # ----- action history (short window - avoid repetition) --------------
    ACTION_WINDOW = 10

    def record_action(self, text: str) -> None:
        if not text:
            return
        self.action_history.append(text)
        self.action_history = self.action_history[-self.ACTION_WINDOW:]

    def action_view(self, limit: int = 10) -> list:
        return self.action_history[-limit:]

    # ----- hint history (operator guidance - persistent, high value) -----
    def record_hint(self, text: str, step: int = 0) -> None:
        """Record an operator hint permanently. Deduped against the immediately
        previous hint so a hint held 'active' for several turns is stored once."""
        text = (text or "").strip()
        if not text:
            return
        if self.hints and self.hints[-1].get("text") == text:
            return
        self.hints.append({"step": int(step), "text": text[:300]})
        # Keep a generous archive; these are cheap and valuable.
        self.hints = self.hints[-100:]

    def recent_hints(self, limit: int = 5) -> list:
        """The most recent operator hints (newest last), for the prompt so the
        agent remembers steering it was given earlier - not just this turn."""
        return [h.get("text", "") for h in self.hints[-limit:]]

    def hints_pretty(self, limit: int = 12) -> str:
        if not self.hints:
            return "(no hints given yet)"
        return "\n".join(f"  @{h.get('step',0)}: {h.get('text','')}"
                         for h in self.hints[-limit:])

    # ----- observation memory (notable things seen / overheard) ----------
    # Words that make an on-screen object worth remembering as "seen" (general,
    # not tied to any puzzle). Doors/signs are already mapped as places.
    _NOTABLE_SEEN = (
        "body", "corpse", "chest", "key", "book", "scroll", "note", "letter",
        "gold", "gem", "ring", "sword", "shield", "armor", "potion", "wand",
        "lever", "switch", "grave", "coffin", "altar", "shrine", "cauldron",
        "skeleton", "blood", "trap", "locked", "magic", "rune",
    )

    def note_observation(self, text: str, kind: str = "seen", step: int = 0) -> bool:
        """Log a notable observation (something seen or overheard), deduped so
        repeated sightings of the same thing don't flood memory. Returns True if
        a new observation was actually recorded."""
        text = (text or "").strip()
        if not text:
            return False
        # Dedup against anything recorded recently (last 40) regardless of step.
        recent = {o.get("text") for o in self.observations[-40:]}
        if text in recent:
            return False
        self.observations.append({"step": int(step), "kind": kind, "text": text[:200]})
        self.observations = self.observations[-300:]
        return True

    def is_notable_object(self, name: str) -> bool:
        low = (name or "").lower()
        return any(w in low for w in self._NOTABLE_SEEN)

    def observations_view(self, limit: int = 12) -> list:
        """Recent notable observations (newest last) for the prompt."""
        return [f"{o.get('kind','')}: {o.get('text','')}"
                for o in self.observations[-limit:]]

    def observations_pretty(self, limit: int = 20) -> str:
        if not self.observations:
            return "(nothing notable logged yet)"
        return "\n".join(f"  @{o.get('step',0)} [{o.get('kind','')}] {o.get('text','')}"
                         for o in self.observations[-limit:])

    # ----- episodic summary + context budget -----------------------------
    def append_summary(self, text: str) -> None:
        """Fold a gist line into the rolling episodic summary (bounded)."""
        text = (text or "").strip()
        if not text:
            return
        if self.episodic_summary:
            self.episodic_summary += " " + text
        else:
            self.episodic_summary = text
        # Bound the summary so it can't grow without limit; keep the tail
        # (most recent gist). ~1500 chars is a few hundred tokens.
        if len(self.episodic_summary) > 1500:
            self.episodic_summary = "..." + self.episodic_summary[-1500:]

    # Phrases whose presence makes an old dialogue line worth preserving as gist
    # when it scrolls out of the raw window (clues/tasks), vs. dropping chit-chat.
    _KEEP_GIST = (
        "must", "need", "should", "find", "bring", "fetch", "seek", "go to",
        "password", "key", "quest", "task", "search", "look for", "deliver",
        "rescue", "retrieve", "secret", "hidden", "murder", "kill", "steal",
        "beware", "danger", "north", "south", "east", "west", "gold", "reward",
    )

    def fold_dialogue_into_summary(self, keep_recent: int) -> int:
        """Compress dialogue older than the most-recent `keep_recent` entries
        into the episodic summary (extractive: keep clue/task-like NPC lines),
        then drop them from the raw window. Returns how many were folded.

        This is the core of the context-budget strategy: instead of merely
        dropping old lines, their gist survives in a compact running summary."""
        if len(self.dialogue_history) <= keep_recent:
            return 0
        old = self.dialogue_history[:-keep_recent] if keep_recent > 0 else self.dialogue_history[:]
        kept = []
        for e in old:
            said = e.get("said") or e.get("me") or ""
            low = said.lower()
            who = e.get("npc") if "npc" in e else "you"
            if any(k in low for k in self._KEEP_GIST) and len(said) > 15:
                kept.append(f"{who}: {said[:100]}")
        if kept:
            # Cap how much we add at once so a big fold doesn't bloat the summary.
            self.append_summary(" | ".join(kept[-12:]))
        n = len(old)
        self.dialogue_history = self.dialogue_history[-keep_recent:] if keep_recent > 0 else []
        return n

    # ----- mental map / places -------------------------------------------
    def record_place(self, name: str, tx: int, ty: int,
                     kind: str = "place", note: str = "") -> None:
        """Remember a discovered location so it can be navigated to later."""
        if not name:
            return
        key = _slug(name)
        rec = self.places.setdefault(key, {"name": name, "kind": kind})
        rec["name"] = name
        rec["tx"] = int(tx)
        rec["ty"] = int(ty)
        if kind and kind != "place":
            rec["kind"] = kind
        if note:
            rec["note"] = note[:120]

    def place_pos(self, name: str):
        key = _slug(name)
        rec = self.places.get(key)
        if rec:
            return [rec.get("tx"), rec.get("ty")]
        # fuzzy: substring match on names
        low = (name or "").lower()
        for r in self.places.values():
            if low and low in r.get("name", "").lower():
                return [r.get("tx"), r.get("ty")]
        return None

    def places_view(self, here_tx: int = 0, here_ty: int = 0, limit: int = 12) -> list:
        """Known places, nearest first, with rough direction from 'here'."""
        out = []
        for r in self.places.values():
            dx = r.get("tx", 0) - here_tx
            dy = r.get("ty", 0) - here_ty
            out.append((abs(dx) + abs(dy),
                        {"name": r.get("name"), "kind": r.get("kind", "place"),
                         "dx": dx, "dy": dy}))
        out.sort(key=lambda t: t[0])
        return [o for _, o in out[:limit]]

    # ----- automatic knowledge capture -----------------------------------
    # Phrases that suggest an NPC is giving a task/lead worth remembering as a
    # quest.  General across the game - not tied to any specific puzzle.
    _QUEST_HINTS = (
        "must", "need to", "should", "find", "bring", "fetch", "seek",
        "go to", "travel to", "help me", "please", "task", "quest", "mission",
        "password", "key", "search", "look for", "deliver", "rescue", "retrieve",
        "speak to", "talk to", "ask ", "tell ", "return to", "report",
    )

    def auto_note_from_dialogue(self, npc: str, said: str) -> Optional[str]:
        """Passively capture durable knowledge from an NPC line without relying
        on the model to call meta-tools: always save it as an NPC note, and if
        it looks like a task/lead, record a lightweight quest.  Returns a quest
        id if one was created/updated, else None."""
        if not npc or npc == "?" or not said:
            return None
        # Trim game markup and whitespace.
        clean = said.replace("*", " ").strip()
        if not clean:
            return None
        # 1) Always remember what this NPC said (bounded, deduped in note_npc).
        self.note_npc(npc, clean[:200])
        # 2) Heuristic quest capture from task-like lines.
        low = clean.lower()
        if any(h in low for h in self._QUEST_HINTS) and len(clean) > 25:
            qid = "lead_" + _slug(npc)
            title = f"Follow up on what {npc} said"
            # Keep the most recent task-like line as the quest's notes.
            self.add_quest(title=title, priority=4,
                          notes=f"{npc}: {clean[:180]}", qid=qid, status="active",
                          npc=npc)
            return qid
        return None
