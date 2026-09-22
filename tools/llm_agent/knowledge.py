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
        # Shared TOPIC knowledge base: cross-cutting subjects the world talks
        # about (e.g. "Fellowship", "Batlin", "the gargoyles", "murder"), keyed
        # by a slug. Each aggregates what MULTIPLE npcs said about it, so the
        # agent can understand a theme across everyone, not just per-person.
        # slug -> {name, mentions:[{npc,said}], npcs:[names], first_step,last_step}
        self.topics: dict[str, dict] = {}
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
        # TOOL-CALL STATS: how the agent is spending its turns, so we can see
        # health at a glance (e.g. many parse failures = model output problem,
        # many errors on a tool = a stuck pattern). Structure:
        #   {"turns":int, "parse_fail":int,
        #    "tools": {name: {"calls":int, "ok":int, "err":int}}}
        self.tool_stats: dict = {"turns": 0, "parse_fail": 0, "tools": {}}

    # ----- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"quests": self.quests, "npcs": self.npcs, "journal": self.journal,
                "dialogue_history": self.dialogue_history,
                "action_history": self.action_history,
                "places": self.places,
                "hints": self.hints,
                "observations": self.observations,
                "episodic_summary": self.episodic_summary,
                "tool_stats": self.tool_stats,
                "topics": self.topics}

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
            kb.topics = dict(data.get("topics", {}))
            kb.journal = list(data.get("journal", []))
            kb.dialogue_history = list(data.get("dialogue_history", []))
            kb.action_history = list(data.get("action_history", []))
            kb.places = dict(data.get("places", {}))
            kb.hints = list(data.get("hints", []))
            kb.observations = list(data.get("observations", []))
            kb.episodic_summary = str(data.get("episodic_summary", "") or "")
            ts = data.get("tool_stats") or {}
            kb.tool_stats = {"turns": int(ts.get("turns", 0)),
                             "parse_fail": int(ts.get("parse_fail", 0)),
                             "tools": dict(ts.get("tools", {}))}
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
    @staticmethod
    def _title_words(title: str) -> set:
        stop = {"the","a","an","to","of","and","on","in","with","about","for",
                "from","at","investigate","follow","up","gather","leads","find",
                "speak","talk","ask","get","go"}
        words = re.findall(r"[a-z0-9]+", (title or "").lower())
        return {w for w in words if w not in stop and len(w) > 2}

    def _find_similar_quest(self, title: str) -> Optional[str]:
        """Return the id of an existing OPEN quest that is a near-duplicate of
        `title` (shares most significant words), else None. Prevents the log
        filling with 'Investigate X' / 'Find X' / 'X in Britain' variants."""
        want = self._title_words(title)
        if not want:
            return None
        for qid, q in self.quests.items():
            if q.get("status") == "done":
                continue
            have = self._title_words(q.get("title", ""))
            if not have:
                continue
            overlap = want & have
            # Near-duplicate if the significant words substantially overlap.
            if overlap and len(overlap) >= max(1, min(len(want), len(have)) - 0):
                if len(overlap) / max(len(want), len(have)) >= 0.6:
                    return qid
        return None

    def add_quest(self, title: str, priority: int = 5, notes: str = "",
                  depends_on: Optional[list] = None, qid: Optional[str] = None,
                  status: str = "active", npc: Optional[str] = None) -> str:
        # Merge into a near-duplicate open quest if one exists (unless an
        # explicit qid was given), so we don't accumulate redundant variants.
        if qid is None:
            sim = self._find_similar_quest(title)
            if sim is not None:
                qid = sim
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

    def quest_view(self, max_open: int = 12) -> dict:
        """One priority-sorted list of open quests (priority 1 = highest). We do
        NOT partition into actionable/blocked - the agent decides what it can
        work on. Prerequisites are surfaced as info (depends_on + which are
        unmet) so the agent can reason about ordering itself. The full set is
        always kept on disk; only the display is capped."""
        active = [q for q in self.quests.values() if q.get("status") not in ("done",)]
        active.sort(key=lambda q: q.get("priority", 5))

        def brief(q: dict) -> dict:
            b = {"id": q["id"], "title": q.get("title"), "priority": q.get("priority", 5),
                 "status": q.get("status", "active")}
            if q.get("npc"):
                b["npc"] = q["npc"]
            dep = q.get("depends_on") or []
            if dep:
                b["depends_on"] = dep
                # Which prerequisites are not yet done (info only, not a gate).
                unmet = [d for d in dep
                         if (self.quests.get(d) or {}).get("status") != "done"]
                if unmet:
                    b["prereqs_unmet"] = unmet
            if q.get("notes"):
                b["notes"] = q["notes"][-240:]
            return b

        done = [q for q in self.quests.values() if q.get("status") == "done"]
        return {
            # Single prioritized list; the agent chooses what to pursue.
            "open": [brief(q) for q in active[:max_open]],
            # So the agent can refer back to what it finished and not redo it.
            "recently_resolved": [q.get("title") for q in done[-6:]],
            "unresolved": len(active),
            "resolved": len(done),
        }

    def resolve_quest(self, qid: str) -> bool:
        """Mark a quest resolved (done). Accepts id or fuzzy title."""
        return self.update_quest(qid, status="done")

    def auto_resolve_talk_quests(self, npc: str) -> list:
        """Conservatively resolve open quests that are simply 'speak to / talk to
        / ask <npc>' once we've actually had a substantive conversation with that
        npc. Returns titles resolved. The LLM still owns richer quests; this only
        closes the trivial 'go talk to X' ones it reliably forgets to close."""
        if not npc or npc == "?":
            return []
        low_npc = npc.lower()
        verbs = ("speak to", "talk to", "ask ", "meet ", "find and speak",
                 "consult ", "report to", "return to")
        resolved = []
        for q in self.quests.values():
            if q.get("status") == "done":
                continue
            title = (q.get("title") or "").lower()
            qnpc = (q.get("npc") or "").lower()
            starts_verb = any(title.startswith(v) for v in verbs)
            names_npc = low_npc in title or (qnpc and low_npc in qnpc)
            if starts_verb and names_npc:
                q["status"] = "done"
                resolved.append(q.get("title"))
        return resolved

    def has_unresolved(self) -> bool:
        return any(q.get("status") != "done" for q in self.quests.values())

    # ----- human-readable summaries (for the inspector GUI) --------------
    def quests_pretty(self) -> str:
        qv = self.quest_view(max_open=30)
        lines = [f"resolved {qv.get('resolved',0)} / unresolved {qv.get('unresolved',0)}", ""]
        for q in qv.get("open", []):
            npc = f" [{q['npc']}]" if q.get("npc") else ""
            prereq = ""
            if q.get("prereqs_unmet"):
                prereq = "  (needs: " + ",".join(q["prereqs_unmet"]) + ")"
            lines.append(f"  P{q.get('priority',5)} {q['title']}{npc}{prereq}")
        rr = qv.get("recently_resolved") or []
        if rr:
            lines.append("")
            lines.append("DONE: " + "; ".join(rr))
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
            pos = r.get("last_pos")
            loc = f" @({pos[0]},{pos[1]})" if pos and pos[0] is not None else ""
            lines.append(f"{tag}{name} ({st}, x{r.get('times_talked',0)}){loc}: {note}")
        return "\n".join(lines) if lines else "(no NPCs met yet)"

    # Low-value lines not worth storing as NPC knowledge (greetings/closers and
    # generic look descriptions add noise and crowd out real clues).
    _FILLER = ("goodbye", "good day", "farewell", "hello", "greetings",
               "you see a", "you see an", "yes?", "what dost thou want")

    def note_npc(self, name: str, note: str = "") -> None:
        if not name:
            return
        rec = self.npcs.setdefault(name, {"name": name, "times_talked": 0, "notes": []})
        note = (note or "").strip()
        if not note:
            return
        low = note.lower().strip(' "*')
        # Skip pure filler (greetings, closers, stock look descriptions).
        if any(low.startswith(f) or low == f for f in self._FILLER):
            return
        # Dedup against ALL recent notes (not just the last one) so repeated
        # lines from re-talking don't pile up.
        if note in rec["notes"][-20:]:
            return
        rec["notes"].append(note)
        rec["notes"] = rec["notes"][-20:]

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
        """Meaningful progress happened - allow revisiting ALL NPCs (they may
        now have new dialogue). This fully clears the per-epoch counter so even
        an NPC we spoke to many times becomes worth revisiting after the story
        advances (e.g. a companion who only joins once a quest is underway)."""
        for rec in self.npcs.values():
            rec["talked_this_epoch"] = 0

    def talk_status(self, name: str, exhausted_at: int = 5) -> str:
        """Classify how worthwhile talking to this NPC is right now:
        'new' (never talked), 'talked' (spoken to but may have more), or
        'exhausted' (asked enough - unlikely to offer new info).

        Two exhaustion signals:
          * epoch: asked `exhausted_at` times since the last meaningful progress
            (resets on progress, so NPCs become worth revisiting after you
            accomplish something).
          * hard cap: asked `hard_cap`+ times TOTAL. This does NOT reset - if
            you've hammered an NPC many times overall, they really are tapped
            out and progress resets shouldn't keep sending you back to them."""
        rec = self.npcs.get(name)
        if not rec or rec.get("times_talked", 0) == 0:
            return "new"
        # Exhaustion is driven by talks SINCE the last meaningful progress
        # (the epoch counter), which resets on progress so an NPC becomes
        # worth revisiting after the story advances (e.g. a companion who only
        # joins once a quest is underway). We do NOT permanently blacklist an
        # NPC by lifetime talk count - that wrongly ignored recruits/quest NPCs
        # forever. The epoch gate alone stops tight re-talk loops.
        if rec.get("talked_this_epoch", 0) >= exhausted_at:
            return "exhausted"
        return "talked"

    def npc_view(self, names: Optional[list] = None, note_limit: int = 12) -> dict:
        """NPC notes for the prompt; if names given, only those, else all known.
        Each entry includes last_seen coords. note_limit caps how many recorded
        lines per NPC are included (lowered under context pressure)."""
        src = self.npcs
        if names:
            lname = {n.lower() for n in names}
            src = {k: v for k, v in self.npcs.items() if k.lower() in lname}
        out = {}
        for n, r in src.items():
            entry = {"times_talked": r.get("times_talked", 0),
                     "status": self.talk_status(n),
                     "notes": r.get("notes", [])[-note_limit:]}
            pos = r.get("last_pos")
            if pos and pos[0] is not None:
                # last_seen = tile where we most recently observed this NPC.
                entry["last_seen"] = {"tx": pos[0], "ty": pos[1]}
            out[n] = entry
        return out

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
        # Also append to the per-NPC full transcript (the dialogue TREE), which
        # persists across runs so the agent can recall exactly what each
        # character told it - including the Mayor's instructions.
        self._npc_transcript_add(npc, {"said": said})

    # ----- per-NPC dialogue tree (persistent, retrievable via recall) ----
    def _npc_rec(self, npc: str) -> dict:
        return self.npcs.setdefault(
            npc or "?", {"name": npc or "?", "times_talked": 0, "notes": []})

    def _npc_transcript_add(self, npc: str, entry: dict) -> None:
        if not npc or npc == "?":
            return
        rec = self._npc_rec(npc)
        tr = rec.setdefault("transcript", [])
        # Dedup consecutive identical spoken lines (the game repeats the current
        # line across turns until a choice is picked).
        if entry.get("said") and tr and tr[-1].get("said") == entry["said"]:
            return
        tr.append(entry)
        rec["transcript"] = tr[-80:]    # keep a generous per-NPC history

    def record_npc_choices(self, npc: str, choices: list) -> None:
        """Record the answer TOPICS the game offered with this NPC (the tree
        branches), so the agent knows what it can still ask and what it has
        covered. Accumulates the union of all topics ever seen for this NPC."""
        if not npc or npc == "?" or not choices:
            return
        rec = self._npc_rec(npc)
        seen = rec.setdefault("topics_offered", [])
        for c in choices:
            c = str(c).strip()
            if c and c not in seen:
                seen.append(c)
        rec["topics_offered"] = seen[-60:]

    def record_npc_choice_taken(self, npc: str, topic: str) -> None:
        """Record that the agent asked this topic (a branch it explored)."""
        if not npc or npc == "?" or not topic:
            return
        rec = self._npc_rec(npc)
        asked = rec.setdefault("topics_asked", [])
        topic = str(topic).strip()
        if topic and topic not in asked:
            asked.append(topic)
        rec["topics_asked"] = asked[-60:]
        self._npc_transcript_add(npc, {"me": topic})

    # ----- shared topic knowledge base -----------------------------------
    _GENERIC_TOPICS = {
        "name", "job", "bye", "goodbye", "yes", "no", "leave", "farewell",
        "hello", "thanks", "thank you", "nothing", "who", "what", "why", "how",
    }

    def add_topic(self, name: str, note: str = "", step: int = 0) -> str:
        """LLM-AUTHORED topic. The model creates topics and attaches its own
        notes/insights, so the topic list reflects the agent's evolving
        understanding over time (not a mechanical copy of game dialogue). Each
        note is timestamped by step so you can watch its thinking develop.
        Returns the topic key."""
        name = (name or "").strip()
        if not name:
            return ""
        key = _slug(name)
        if not key:
            return ""
        rec = self.topics.setdefault(
            key, {"name": name, "notes": [], "first_step": step})
        rec["last_step"] = step
        note = (note or "").strip()
        if note:
            entry = {"step": int(step), "note": note[:400]}
            # Dedup against the most recent note.
            if not rec["notes"] or rec["notes"][-1].get("note") != entry["note"]:
                rec["notes"].append(entry)
                rec["notes"] = rec["notes"][-40:]
        return key

    def recall_topic(self, subject: str) -> dict:
        """Everything the LLM has recorded about a topic - all its notes over
        time (its evolving understanding)."""
        if not subject:
            return {"name": subject, "known": False}
        key = _slug(subject)
        rec = self.topics.get(key)
        if rec is None:
            low = subject.lower()
            for v in self.topics.values():
                if low in v.get("name", "").lower():
                    rec = v
                    break
        if rec is None:
            return {"name": subject, "known": False}
        return {"name": rec.get("name", subject), "known": True,
                "notes": rec.get("notes", [])[-20:]}

    def topics_view(self, limit: int = 16) -> list:
        """Compact list of LLM-authored topics for the prompt: name + how many
        notes it has recorded, most-recently-updated first."""
        recs = sorted(self.topics.values(),
                      key=lambda r: -r.get("last_step", 0))
        out = []
        for r in recs[:limit]:
            out.append({"topic": r.get("name"), "notes": len(r.get("notes", []))})
        return out

    def topics_pretty(self, limit: int = 20) -> str:
        recs = sorted(self.topics.values(), key=lambda r: -r.get("last_step", 0))
        if not recs:
            return "(no topics yet)"
        lines = []
        for r in recs[:limit]:
            last = (r.get("notes") or [{}])[-1].get("note", "")
            last = (last[:60] + "...") if len(last) > 60 else last
            lines.append(f"  {r.get('name')} ({len(r.get('notes', []))}): {last}")
        return "\n".join(lines)

    def topics_tree_data(self, limit: int = 40) -> list:
        """Structured topic data for the GUI tree: each topic with its
        timestamped LLM notes (most-recently-updated topics first)."""
        recs = sorted(self.topics.values(),
                      key=lambda r: -r.get("last_step", 0))
        out = []
        for r in recs[:limit]:
            out.append({"name": r.get("name", "?"),
                        "notes": r.get("notes", [])[-40:]})
        return out

    def npcs_tree_data(self, limit: int = 40) -> list:
        """Structured per-character data for the GUI tree: transcript, topics
        asked/unasked, notes. Most-talked-to first."""
        recs = sorted(self.npcs.values(),
                      key=lambda r: -r.get("times_talked", 0))
        out = []
        for r in recs[:limit]:
            offered = r.get("topics_offered", [])
            asked = set(r.get("topics_asked", []))
            out.append({
                "name": r.get("name", "?"),
                "times_talked": r.get("times_talked", 0),
                "transcript": r.get("transcript", [])[-40:],
                "topics_asked": r.get("topics_asked", []),
                "topics_unasked": [t for t in offered if t not in asked],
                "notes": r.get("notes", []),
            })
        return out

    def recall_npc(self, npc: str) -> dict:
        """Full retrievable record of a character: everything they said (their
        transcript), the topics offered, and which topics we already asked.
        Fuzzy-matches the name so 'mayor' finds 'Finnigan' if noted, etc."""
        rec = self.npcs.get(npc)
        if rec is None:
            low = (npc or "").lower()
            # 1) name substring match
            for k, v in self.npcs.items():
                if low and low in k.lower():
                    rec = v
                    break
            # 2) match against what they said / notes (e.g. "mayor" -> Finnigan
            #    who said "I am the Mayor"), so titles/roles resolve too.
            if rec is None and low:
                for v in self.npcs.values():
                    hay = " ".join(
                        [t.get("said", "") for t in v.get("transcript", [])]
                        + list(v.get("notes", []))).lower()
                    if low in hay:
                        rec = v
                        break
        if rec is None:
            return {"name": npc, "known": False}
        offered = rec.get("topics_offered", [])
        asked = set(rec.get("topics_asked", []))
        return {
            "name": rec.get("name", npc),
            "known": True,
            "times_talked": rec.get("times_talked", 0),
            "transcript": rec.get("transcript", [])[-60:],
            "topics_asked": rec.get("topics_asked", []),
            "topics_unasked": [t for t in offered if t not in asked],
            "notes": rec.get("notes", [])[-20:],
        }

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
    # ----- tool-call stats (turn accounting / health) --------------------
    def record_turn(self, parse_ok: bool) -> None:
        """Count one agent turn; note whether the model's reply parsed."""
        self.tool_stats["turns"] = self.tool_stats.get("turns", 0) + 1
        if not parse_ok:
            self.tool_stats["parse_fail"] = self.tool_stats.get("parse_fail", 0) + 1

    def record_tool(self, name: str, ok: Optional[bool]) -> None:
        """Count a tool/action invocation and its outcome (ok/err/unknown)."""
        if not name:
            name = "?"
        t = self.tool_stats.setdefault("tools", {})
        rec = t.setdefault(name, {"calls": 0, "ok": 0, "err": 0})
        rec["calls"] += 1
        if ok is True:
            rec["ok"] += 1
        elif ok is False:
            rec["err"] += 1

    def tool_stats_pretty(self, top: int = 12) -> str:
        ts = self.tool_stats
        turns = ts.get("turns", 0)
        pf = ts.get("parse_fail", 0)
        pf_pct = (100 * pf / turns) if turns else 0
        lines = [f"turns: {turns}   parse-fail: {pf} ({pf_pct:.0f}%)"]
        tools = ts.get("tools", {})
        for name, r in sorted(tools.items(), key=lambda kv: -kv[1].get("calls", 0))[:top]:
            c, ok, err = r.get("calls", 0), r.get("ok", 0), r.get("err", 0)
            lines.append(f"  {name}: {c}  (ok {ok} / err {err})")
        return "\n".join(lines)

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
        """Passively capture what an NPC said as an NPC NOTE (deduped/filtered).
        We deliberately do NOT auto-create quests here anymore: the generic
        'Follow up on what X said' stubs flooded the quest log with noise and
        weren't strategic. Real quests should come from the LLM's own reasoning
        via the add_quest meta-tool. Returns None (kept for call compatibility)."""
        if not npc or npc == "?" or not said:
            return None
        clean = said.replace("*", " ").strip()
        if not clean:
            return None
        self.note_npc(npc, clean[:200])
        return None
