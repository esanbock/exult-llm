"""thoughts_window.py - A live "LLM agent inspector" window (Tkinter, stdlib).

Two columns:
  LEFT  (live play):   map, LLM reasoning, chosen action, dialog/characters
  RIGHT (inspector):   context-usage gauge, quest log (with status),
                       NPC knowledge, and misc stats

Runs Tkinter on the main thread; the driver pushes updates from a worker thread
via a thread-safe queue.
"""

from __future__ import annotations

import queue
from typing import Optional

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


class ThoughtsWindow:
    def __init__(self, title: str = "Exult LLM Agent - Inspector"):
        self.available = _TK_AVAILABLE
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._root: Optional["tk.Tk"] = None
        self._title = title
        self._panes = {}
        # Hints the user types are queued here for the driver to consume.
        self._hints: "queue.Queue[str]" = queue.Queue()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if not self.available:
            print("[thoughts] tkinter not available; running without a window.")
            return
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.geometry("1180x900")

        top = tk.Frame(self._root)
        top.pack(fill="x", padx=8, pady=(8, 2))
        tk.Label(top, text="Ultima VII - LLM Agent",
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        self._status = tk.Label(top, text="waiting...", font=("Segoe UI", 10))
        self._status.pack(side="right")

        # Context-usage gauge row.
        ctxrow = tk.Frame(self._root)
        ctxrow.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(ctxrow, text="Context:", font=("Segoe UI", 9, "bold")).pack(side="left")
        self._ctx_bar = ttk.Progressbar(ctxrow, length=300, maximum=100)
        self._ctx_bar.pack(side="left", padx=6)
        self._ctx_lbl = tk.Label(ctxrow, text="- tok", font=("Consolas", 9))
        self._ctx_lbl.pack(side="left")

        body = tk.Frame(self._root)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        def _pane(parent, key, label, height, mono=False, wrap="word"):
            frame = tk.LabelFrame(parent, text=label, font=("Segoe UI", 10, "bold"))
            frame.pack(fill="both", expand=True, padx=4, pady=3)
            font = ("Consolas", 10) if mono else ("Segoe UI", 10)
            txt = scrolledtext.ScrolledText(frame, height=height, wrap=wrap, font=font)
            txt.pack(fill="both", expand=True)
            self._panes[key] = txt

        # LEFT: live play
        _pane(left, "map", "Map (@ you, C companion, & npc, x body, * item, + door, # wall)",
              14, mono=True, wrap="none")
        _pane(left, "think", "LLM reasoning (this turn)", 6)
        _pane(left, "action", "Chosen action -> result", 4, mono=True)
        _pane(left, "dialog", "Dialog / characters / objects on screen", 8)

        # RIGHT: inspector
        _pane(right, "quests", "Quest log (focus / actionable / blocked)", 12, mono=True)

        # Knowledge notebook: expandable trees for Topics and Characters.
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, padx=4, pady=3)

        # Topics tree: topic -> mentions (npc: line)
        topic_frame = tk.Frame(nb)
        self._topic_tree = ttk.Treeview(topic_frame, columns=("meta",), show="tree headings")
        self._topic_tree.heading("#0", text="Topic / mention")
        self._topic_tree.heading("meta", text="npcs / lines")
        self._topic_tree.column("meta", width=90, anchor="e")
        _tsb = ttk.Scrollbar(topic_frame, orient="vertical", command=self._topic_tree.yview)
        self._topic_tree.configure(yscrollcommand=_tsb.set)
        self._topic_tree.pack(side="left", fill="both", expand=True)
        _tsb.pack(side="right", fill="y")
        nb.add(topic_frame, text="Topics")

        # Characters tree: npc -> {transcript, topics asked/unasked, notes}
        char_frame = tk.Frame(nb)
        self._char_tree = ttk.Treeview(char_frame, columns=("meta",), show="tree headings")
        self._char_tree.heading("#0", text="Character / dialogue")
        self._char_tree.heading("meta", text="info")
        self._char_tree.column("meta", width=90, anchor="e")
        _csb = ttk.Scrollbar(char_frame, orient="vertical", command=self._char_tree.yview)
        self._char_tree.configure(yscrollcommand=_csb.set)
        self._char_tree.pack(side="left", fill="both", expand=True)
        _csb.pack(side="right", fill="y")
        nb.add(char_frame, text="Characters")

        _pane(right, "stats", "Stats & memory", 6, mono=True)

        # Hint bar: type a hint and Send it to the agent for the next turn(s).
        hintrow = tk.Frame(self._root)
        hintrow.pack(fill="x", padx=8, pady=(2, 8))
        tk.Label(hintrow, text="Hint:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self._hint_entry = tk.Entry(hintrow, font=("Segoe UI", 10))
        self._hint_entry.pack(side="left", fill="x", expand=True, padx=6)
        self._hint_entry.bind("<Return>", lambda e: self._send_hint())
        tk.Button(hintrow, text="Send hint", command=self._send_hint).pack(side="left")
        self._hint_status = tk.Label(hintrow, text="", font=("Segoe UI", 8), fg="green")
        self._hint_status.pack(side="left", padx=6)

        self._root.after(100, self._drain)

    def _send_hint(self) -> None:
        txt = self._hint_entry.get().strip()
        if txt:
            self._hints.put(txt)
            self._hint_entry.delete(0, "end")
            self._hint_status.config(text="sent")
            if self._root is not None:
                self._root.after(1500, lambda: self._hint_status.config(text=""))

    def get_hint(self) -> Optional[str]:
        """Return the next queued user hint (or None). Called by the driver."""
        try:
            return self._hints.get_nowait()
        except queue.Empty:
            return None

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "turn":
                    self._status.config(text=f"Turn {payload}")
                elif kind == "ctx":
                    pct, label = payload
                    try:
                        self._ctx_bar["value"] = max(0, min(100, pct))
                    except Exception:
                        pass
                    self._ctx_lbl.config(text=label)
                elif kind == "topics_tree":
                    self._rebuild_topics(payload)
                elif kind == "npc_tree":
                    self._rebuild_chars(payload)
                elif kind in self._panes:
                    w = self._panes[kind]
                    w.delete("1.0", "end")
                    w.insert("end", payload)
        except queue.Empty:
            pass
        if self._root is not None:
            self._root.after(100, self._drain)

    def mainloop(self) -> None:
        if self._root is not None:
            self._root.mainloop()

    # -- tree rebuilders (main thread) ----------------------------------------
    def _expanded_ids(self, tree) -> set:
        """Top-level item ids (by their text) currently expanded, so a rebuild
        keeps the user's open nodes open."""
        out = set()
        for iid in tree.get_children(""):
            if tree.item(iid, "open"):
                out.add(tree.item(iid, "text"))
        return out

    def _rebuild_topics(self, topics: list) -> None:
        """topics: list of {name, npcs, mentions:[{npc,said}]} sorted by the
        driver (most-discussed first)."""
        tree = self._topic_tree
        keep_open = self._expanded_ids(tree)
        tree.delete(*tree.get_children(""))
        for t in topics or []:
            name = t.get("name", "?")
            ments = t.get("mentions", [])
            meta = f"{t.get('npcs', 0)}n / {len(ments)}l"
            parent = tree.insert("", "end", text=name, values=(meta,),
                                 open=(name in keep_open))
            for m in ments:
                who = m.get("npc", "?")
                said = (m.get("said", "") or "")[:120]
                tree.insert(parent, "end", text=f"{who}: {said}", values=("",))

    def _rebuild_chars(self, chars: list) -> None:
        """chars: list of {name, times_talked, transcript:[{said|me}],
        topics_asked:[], topics_unasked:[], notes:[]}"""
        tree = self._char_tree
        keep_open = self._expanded_ids(tree)
        tree.delete(*tree.get_children(""))
        for c in chars or []:
            name = c.get("name", "?")
            meta = f"x{c.get('times_talked', 0)}"
            parent = tree.insert("", "end", text=name, values=(meta,),
                                 open=(name in keep_open))
            # Transcript subtree
            tr = c.get("transcript", [])
            if tr:
                tnode = tree.insert(parent, "end", text=f"transcript ({len(tr)})", values=("",))
                for e in tr:
                    if "said" in e:
                        line = f"{name}: {e['said'][:110]}"
                    else:
                        line = f"you asked: {e.get('me', '')[:60]}"
                    tree.insert(tnode, "end", text=line, values=("",))
            # Topics asked / unasked
            ua = c.get("topics_unasked", [])
            if ua:
                unode = tree.insert(parent, "end", text=f"not yet asked ({len(ua)})", values=("",))
                for topic in ua:
                    tree.insert(unode, "end", text=topic, values=("",))
            asked = c.get("topics_asked", [])
            if asked:
                anode = tree.insert(parent, "end", text=f"asked ({len(asked)})", values=("",))
                for topic in asked:
                    tree.insert(anode, "end", text=topic, values=("",))
            for note in c.get("notes", [])[-6:]:
                tree.insert(parent, "end", text=f"note: {note[:110]}", values=("",))

    def close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    # -- updates (thread-safe) ------------------------------------------------

    def update_turn(self, turn: int) -> None:
        self._q.put(("turn", turn))

    def set_context(self, pct: int, label: str) -> None:
        self._q.put(("ctx", (pct, label)))

    def set_map(self, text: str) -> None:
        self._q.put(("map", text))

    def set_thinking(self, text: str) -> None:
        self._q.put(("think", text))

    def set_action(self, text: str) -> None:
        self._q.put(("action", text))

    def set_dialog(self, text: str) -> None:
        self._q.put(("dialog", text))

    def set_quests(self, text: str) -> None:
        self._q.put(("quests", text))

    def set_npcs(self, text: str) -> None:
        # Superseded by the Characters tree; kept as a no-op for compatibility.
        pass

    def set_topics_tree(self, topics: list) -> None:
        """topics: list of {name, npcs, mentions:[{npc,said}]}."""
        self._q.put(("topics_tree", topics))

    def set_npc_tree(self, chars: list) -> None:
        """chars: list of {name, times_talked, transcript, topics_asked,
        topics_unasked, notes}."""
        self._q.put(("npc_tree", chars))

    def set_stats(self, text: str) -> None:
        self._q.put(("stats", text))

    # observation now folded into stats/dialog; keep for compatibility
    def set_observation(self, text: str) -> None:
        pass
