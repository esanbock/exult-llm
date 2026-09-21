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
        # Rolling window of recent conversation exchanges (kept fairly long -
        # dialogue carries the story/clues). Each: {"npc":str,"said":str} or
        # {"me":str} for the answer the agent chose.
        self.dialogue_history: list[dict] = []
        # Short rolling window of meaningful actions taken (open/pickup/etc).
        self.action_history: list[str] = []

    # ----- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"quests": self.quests, "npcs": self.npcs, "journal": self.journal,
                "dialogue_history": self.dialogue_history,
                "action_history": self.action_history}

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
                  status: str = "active") -> str:
        qid = qid or _slug(title)
        # Merge if it already exists (update in place).
        q = self.quests.get(qid, {"id": qid, "notes": ""})
        q["id"] = qid
        q["title"] = title or q.get("title", qid)
        q["priority"] = int(priority) if priority is not None else q.get("priority", 5)
        q["status"] = status or q.get("status", "active")
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
                "status": q.get("status", "active")}
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
        }

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

    def times_talked(self, name: str) -> int:
        return self.npcs.get(name, {}).get("times_talked", 0)

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
