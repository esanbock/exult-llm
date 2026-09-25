"""thoughts_window.py - A live "LLM agent inspector" window (Tkinter, stdlib).

Two columns:
  LEFT  (live play):   map, LLM reasoning, chosen action, dialog/characters
  RIGHT (inspector):   context-usage gauge, quest log (with status),
                       NPC knowledge, and misc stats

Runs Tkinter on the main thread; the driver pushes updates from a worker thread
via a thread-safe queue.
"""

from __future__ import annotations

import os
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
        # Operator QUESTIONS to the agent (out-of-band interview, not game turns).
        self._asks: "queue.Queue[str]" = queue.Queue()
        self._last_answer = ""          # latest agent interview answer
        self._chat_log: list = []       # persistent operator<->agent transcript
        self._chat_win = None           # the Chat Toplevel (created on demand)
        self._chat_text = None          # the Chat transcript widget
        # Rolling combined turn log (last N turns). Each entry pairs the
        # reasoning with the action/result it produced.
        self._MAX_TURNS = 12
        self._cur_turn = 0
        self._turn_log: list = []       # [(turn, reason, action_result)]
        self._pending_reason = ""       # reason awaiting its action this turn
        self._last_context = ""         # full prompt for the Show-context window
        self._last_area_map = ""        # latest area map for the Show-map window
        self._last_inventory = ""       # latest inventory for the Show-inventory window
        self._stat_labels = {}          # key -> value Label widget

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if not self.available:
            print("[thoughts] tkinter not available; running without a window.")
            return
        self._root = tk.Tk()
        self._root.title(self._title)
        # Restore the last window size/position from a small ini next to this
        # module, falling back to a sensible default.
        self._geom_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".gui_geometry")
        self._sash_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".gui_sashes")
        _geo = "1500x1000"
        try:
            with open(self._geom_file, encoding="utf-8") as _gf:
                _saved = _gf.read().strip()
            if _saved:
                _geo = _saved
        except OSError:
            pass
        self._root.geometry(_geo)
        # Persist geometry on close.
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        # Game-status row: time of day, position, hp/food - always visible.
        statusrow = tk.Frame(self._root)
        statusrow.pack(fill="x", padx=8, pady=(0, 4))
        self._gstatus = tk.Label(statusrow, text="", font=("Consolas", 9), anchor="w")
        self._gstatus.pack(side="left", fill="x", expand=True)

        # ---- Resizable layout: nested ttk.PanedWindows so EVERY box border is
        #      draggable (both directions). Outer split = live play | inspector.
        outer = ttk.PanedWindow(self._root, orient="horizontal")
        outer.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.PanedWindow(outer, orient="vertical")
        right = ttk.PanedWindow(outer, orient="vertical")
        outer.add(left, weight=3)
        outer.add(right, weight=4)
        # Register paned windows so we can persist/restore their sash (divider)
        # positions across sessions (see _save_sashes/_restore_sashes).
        self._paneds = {"outer": outer, "left": left, "right": right}

        def _text_pane(parent, key, label, mono=False, wrap="word", weight=1):
            frame = tk.LabelFrame(parent, text=label, font=("Segoe UI", 9, "bold"))
            font = ("Consolas", 10) if mono else ("Segoe UI", 10)
            txt = scrolledtext.ScrolledText(frame, height=6, wrap=wrap, font=font)
            txt.pack(fill="both", expand=True)
            self._panes[key] = txt
            parent.add(frame, weight=weight)
            return frame

        # LEFT: live play (map / turn log / dialog) - all draggable-resizable.
        # Map uses no-wrap + horizontal scrollbar so a wide map isn't clipped.
        map_frame = tk.LabelFrame(
            left, text="Map (@ you  C comp  & person  b body  x corpse  n cont  "
            "* item  E exit  ~ water  = barrier  +/ doors  # wall)",
            font=("Segoe UI", 9, "bold"))
        map_txt = scrolledtext.ScrolledText(map_frame, height=10, wrap="none",
                                            font=("Consolas", 10))
        _mxsb = ttk.Scrollbar(map_frame, orient="horizontal", command=map_txt.xview)
        map_txt.configure(xscrollcommand=_mxsb.set)
        _mxsb.pack(side="bottom", fill="x")
        map_txt.pack(side="top", fill="both", expand=True)
        self._panes["map"] = map_txt
        left.add(map_frame, weight=3)
        _text_pane(left, "turnlog",
                   f"Turn log - reasoning -> action -> result (last {self._MAX_TURNS})",
                   weight=3)
        _text_pane(left, "dialog", "Dialog / characters / objects on screen", weight=2)

        # RIGHT: inspector.
        _text_pane(right, "plot", "Story so far (LLM's running plot summary)", weight=1)

        # Quests: Open | Finished side by side (draggable).
        quest_pw = ttk.PanedWindow(right, orient="horizontal")
        self._paneds["quest_pw"] = quest_pw
        oq_frame = tk.LabelFrame(quest_pw, text="Open quests (priority-sorted)",
                                 font=("Segoe UI", 9, "bold"))
        oq_txt = scrolledtext.ScrolledText(oq_frame, height=8, wrap="word",
                                           font=("Consolas", 9))
        oq_txt.pack(fill="both", expand=True)
        self._panes["quests"] = oq_txt
        quest_pw.add(oq_frame, weight=1)
        rq_frame = tk.LabelFrame(quest_pw, text="Finished quests",
                                 font=("Segoe UI", 9, "bold"))
        self._resolved_list = tk.Listbox(rq_frame, font=("Segoe UI", 9))
        _rsb = ttk.Scrollbar(rq_frame, orient="vertical",
                             command=self._resolved_list.yview)
        self._resolved_list.configure(yscrollcommand=_rsb.set)
        self._resolved_list.pack(side="left", fill="both", expand=True)
        _rsb.pack(side="right", fill="y")
        quest_pw.add(rq_frame, weight=1)
        right.add(quest_pw, weight=2)

        # Knowledge: Topics ABOVE Characters, each FULL WIDTH (both trees have
        # wide content when branches expand, so side-by-side caused horizontal
        # scrolling). Stack them vertically in a draggable sub-pane.
        know_pw = ttk.PanedWindow(right, orient="vertical")
        self._paneds["know_pw"] = know_pw
        topic_frame = tk.LabelFrame(know_pw, text="Topics (LLM's notebook)",
                                    font=("Segoe UI", 9, "bold"))
        self._topic_tree = ttk.Treeview(topic_frame, columns=("meta",), show="tree headings")
        self._topic_tree.heading("#0", text="Topic / note")
        self._topic_tree.heading("meta", text="notes")
        self._topic_tree.column("#0", width=380, anchor="w", stretch=True)
        self._topic_tree.column("meta", width=60, anchor="e", stretch=False)
        _tsb = ttk.Scrollbar(topic_frame, orient="vertical", command=self._topic_tree.yview)
        self._topic_tree.configure(yscrollcommand=_tsb.set)
        self._topic_tree.pack(side="left", fill="both", expand=True)
        _tsb.pack(side="right", fill="y")
        know_pw.add(topic_frame, weight=1)

        char_frame = tk.LabelFrame(know_pw, text="Characters (dialogue/notes)",
                                   font=("Segoe UI", 9, "bold"))
        self._char_tree = ttk.Treeview(char_frame, columns=("meta",), show="tree headings")
        self._char_tree.heading("#0", text="Character / dialogue")
        self._char_tree.heading("meta", text="info")
        self._char_tree.column("#0", width=380, anchor="w", stretch=True)
        self._char_tree.column("meta", width=60, anchor="e", stretch=False)
        _csb = ttk.Scrollbar(char_frame, orient="vertical", command=self._char_tree.yview)
        self._char_tree.configure(yscrollcommand=_csb.set)
        self._char_tree.pack(side="left", fill="both", expand=True)
        _csb.pack(side="right", fill="y")
        know_pw.add(char_frame, weight=1)
        right.add(know_pw, weight=3)

        # Stats: its OWN full-width row so all rows are visible without scrolling
        # (laid out as multi-column label:value pairs in _rebuild_stats_grid).
        stats_frame = tk.LabelFrame(right, text="Stats & memory",
                                    font=("Segoe UI", 9, "bold"))
        self._stats_grid = tk.Frame(stats_frame)
        self._stats_grid.pack(fill="both", expand=True, padx=4, pady=2)
        right.add(stats_frame, weight=1)

        # Tool calls: its OWN full-width row so all columns are visible without
        # horizontal scrolling.
        tool_frame = tk.LabelFrame(right, text="Tool calls",
                                   font=("Segoe UI", 9, "bold"))
        tsum = tk.Frame(tool_frame)
        tsum.pack(fill="x", padx=4, pady=(2, 0))
        self._tool_turns = tk.Label(tsum, text="turns: -", font=("Consolas", 9, "bold"))
        self._tool_turns.pack(side="left", padx=(0, 12))
        self._tool_pfail = tk.Label(tsum, text="parse-fail: -", font=("Consolas", 9, "bold"))
        self._tool_pfail.pack(side="left")
        cols = ("calls", "ok", "err", "errpct")
        self._tool_tree = ttk.Treeview(tool_frame, columns=cols, show="tree headings",
                                       height=6)
        self._tool_tree.heading("#0", text="tool")
        self._tool_tree.column("#0", width=110, anchor="w", stretch=True)
        for c, txt, w in (("calls", "calls", 55), ("ok", "ok", 45),
                          ("err", "err", 45), ("errpct", "err%", 55)):
            self._tool_tree.heading(c, text=txt)
            self._tool_tree.column(c, width=w, anchor="e", stretch=False)
        _ttsb = ttk.Scrollbar(tool_frame, orient="vertical", command=self._tool_tree.yview)
        self._tool_tree.configure(yscrollcommand=_ttsb.set)
        self._tool_tree.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=2)
        _ttsb.pack(side="right", fill="y")
        self._tool_tree.tag_configure("err", foreground="#c0392b")
        right.add(tool_frame, weight=2)

        # Consolidated OPERATOR row: one input used for BOTH hints (steer the
        # agent) and questions (interview it). The full threaded conversation
        # lives in a separate persistent Chat window (Open chat) so it is not
        # lost in the turn-log noise.
        oprow = tk.Frame(self._root)
        oprow.pack(fill="x", padx=8, pady=(2, 4))
        tk.Label(oprow, text="To agent:", font=("Segoe UI", 10, "bold"),
                 fg="#1a5276").pack(side="left")
        self._op_entry = tk.Entry(oprow, font=("Segoe UI", 10))
        self._op_entry.pack(side="left", fill="x", expand=True, padx=6)
        self._op_entry.bind("<Return>", lambda e: self._send_ask())     # Enter = Ask
        tk.Button(oprow, text="Ask", command=self._send_ask).pack(side="left")
        tk.Button(oprow, text="Hint", command=self._send_hint).pack(side="left", padx=(4, 0))
        tk.Button(oprow, text="Open chat", command=self._show_chat,
                  font=("Segoe UI", 9, "bold")).pack(side="left", padx=(8, 0))
        self._op_status = tk.Label(oprow, text="", font=("Segoe UI", 8), fg="#1a5276")
        self._op_status.pack(side="left", padx=6)

        # Utility + turn-memory row.
        utilrow = tk.Frame(self._root)
        utilrow.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(utilrow, text="Show context",
                  command=self._show_context).pack(side="left")
        tk.Button(utilrow, text="Show map",
                  command=self._show_map).pack(side="left", padx=(6, 0))
        tk.Button(utilrow, text="Show inventory",
                  command=self._show_inventory).pack(side="left", padx=(6, 0))
        tk.Label(utilrow, text="  Turn memory:", font=("Segoe UI", 9, "bold")
                 ).pack(side="left")
        self._turnmem_var = tk.StringVar(value="default")
        ttk.Combobox(utilrow, textvariable=self._turnmem_var, width=8,
                     state="readonly",
                     values=("default", "100", "200", "300", "450",
                             "600", "800", "1000")).pack(side="left", padx=4)

        self._root.after(100, self._drain)
        # Restore the user's saved divider (sash) positions once the window has
        # been laid out and sized (sash coords are pixel-based, so the panes
        # need real dimensions first). A short delay lets geometry settle.
        self._root.after(400, self._restore_sashes)

    def _chat_add(self, who: str, text: str) -> None:
        """Append a line to the persistent operator<->agent chat transcript and,
        if the chat window is open, render it live."""
        self._chat_log.append((who, text))
        del self._chat_log[:-200]
        self._render_chat()

    def _render_chat(self) -> None:
        w = getattr(self, "_chat_text", None)
        if w is None:
            return
        try:
            w.configure(state="normal")
            w.delete("1.0", "end")
            for who, text in self._chat_log:
                tag = {"you-hint": "hint", "you-ask": "ask",
                       "agent": "agent"}.get(who, "")
                label = {"you-hint": "YOU (hint)", "you-ask": "YOU (ask)",
                         "agent": "AGENT"}.get(who, who)
                w.insert("end", f"{label}: ", (tag,))
                w.insert("end", f"{text}\n\n")
            w.tag_configure("hint", foreground="#117a2b", font=("Segoe UI", 10, "bold"))
            w.tag_configure("ask", foreground="#1a5276", font=("Segoe UI", 10, "bold"))
            w.tag_configure("agent", foreground="#7d3c98", font=("Segoe UI", 10, "bold"))
            w.see("end")
            w.configure(state="disabled")
        except Exception:
            pass

    def _send_hint(self) -> None:
        txt = self._op_entry.get().strip()
        if txt:
            self._hints.put(txt)
            self._op_entry.delete(0, "end")
            self._op_status.config(text="hint sent")
            self._chat_add("you-hint", txt)
            if self._root is not None:
                self._root.after(1500, lambda: self._op_status.config(text=""))

    def _send_ask(self) -> None:
        txt = self._op_entry.get().strip()
        if txt:
            self._asks.put(txt)
            self._op_entry.delete(0, "end")
            self._op_status.config(text="asking... (answer next turn)")
            self._chat_add("you-ask", txt)
            self._show_chat()   # surface the chat so the answer isn't missed

    def _show_chat(self) -> None:
        """Open (or focus) the persistent operator<->agent CHAT window: a
        threaded transcript of hints sent and questions asked + the agent's
        answers, separate from the turn-log noise. You can also type here."""
        if self._root is None:
            return
        w = getattr(self, "_chat_win", None)
        if w is not None:
            try:
                w.deiconify(); w.lift(); self._render_chat(); return
            except Exception:
                self._chat_win = None
        win = tk.Toplevel(self._root)
        self._chat_win = win
        win.title("Agent chat (hints + interview)")
        win.geometry("640x560")
        self._chat_text = scrolledtext.ScrolledText(win, wrap="word",
                                                    font=("Segoe UI", 10))
        self._chat_text.pack(fill="both", expand=True)
        entry_row = tk.Frame(win)
        entry_row.pack(fill="x")
        ce = tk.Entry(entry_row, font=("Segoe UI", 10))
        ce.pack(side="left", fill="x", expand=True, padx=4, pady=4)

        def _ask_here():
            t = ce.get().strip()
            if t:
                self._asks.put(t); ce.delete(0, "end"); self._chat_add("you-ask", t)

        def _hint_here():
            t = ce.get().strip()
            if t:
                self._hints.put(t); ce.delete(0, "end"); self._chat_add("you-hint", t)
        ce.bind("<Return>", lambda e: _ask_here())
        tk.Button(entry_row, text="Ask", command=_ask_here).pack(side="left")
        tk.Button(entry_row, text="Hint", command=_hint_here).pack(side="left", padx=4)

        def _on_close():
            self._chat_win = None
            self._chat_text = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)
        self._render_chat()

    def _show_context(self) -> None:
        """Open a separate window showing the full context sent to the model
        this turn (system prompt + per-turn state + reply)."""
        if self._root is None:
            return
        win = tk.Toplevel(self._root)
        win.title("Full context (this turn)")
        win.geometry("900x800")
        txt = scrolledtext.ScrolledText(win, wrap="word", font=("Consolas", 9))
        txt.pack(fill="both", expand=True)
        txt.insert("end", self._last_context or "(no context captured yet)")
        txt.configure(state="disabled")
        btnrow = tk.Frame(win)
        btnrow.pack(fill="x")
        tk.Label(btnrow, text=f"{len(self._last_context)} chars",
                 font=("Segoe UI", 8)).pack(side="left", padx=6)
        tk.Button(btnrow, text="Close", command=win.destroy).pack(side="right", padx=6, pady=4)

    def _show_map(self) -> None:
        """Open a separate window showing the latest town-scale AREA MAP
        (explored cells + labeled landmarks + your position), monospaced so the
        ASCII grid aligns. Refreshable to see the map update as play continues."""
        if self._root is None:
            return
        win = tk.Toplevel(self._root)
        win.title("Area map (explored + landmarks)")
        win.geometry("760x760")
        txt = scrolledtext.ScrolledText(win, wrap="none", font=("Consolas", 11))
        txt.pack(fill="both", expand=True)

        def _fill():
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            txt.insert("end", self._last_area_map or "(no map yet - it fills in as the agent explores)")
            txt.configure(state="disabled")
        _fill()
        btnrow = tk.Frame(win)
        btnrow.pack(fill="x")
        tk.Button(btnrow, text="Refresh", command=_fill).pack(side="left", padx=6, pady=4)
        tk.Button(btnrow, text="Close", command=win.destroy).pack(side="right", padx=6, pady=4)

    def _show_inventory(self) -> None:
        """Open a window showing what the Avatar is WEARING (per slot) and
        CARRYING. Sourced from a periodic out-of-band inventory query the driver
        pushes to the GUI (does NOT cost the agent a turn). Refreshable."""
        if self._root is None:
            return
        win = tk.Toplevel(self._root)
        win.title("Inventory (worn + carried)")
        win.geometry("520x640")
        txt = scrolledtext.ScrolledText(win, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True)

        def _fill():
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            txt.insert("end", getattr(self, "_last_inventory", "")
                       or "(no inventory captured yet - it refreshes every few turns)")
            txt.configure(state="disabled")
        _fill()
        btnrow = tk.Frame(win)
        btnrow.pack(fill="x")
        tk.Button(btnrow, text="Refresh", command=_fill).pack(side="left", padx=6, pady=4)
        tk.Button(btnrow, text="Close", command=win.destroy).pack(side="right", padx=6, pady=4)

    def _append_turn_entry(self, action_result=None) -> None:
        """Merge reasoning + action/result into ONE entry per turn. Reasoning
        (think) usually arrives first and creates/updates the entry; the action
        arrives next and completes it. If they land out of order we still pair
        them by the current turn number."""
        log = self._turn_log
        # Find an existing entry for this turn to complete, else make one.
        entry = None
        if log and log[-1][0] == self._cur_turn:
            entry = log[-1]
        if entry is None:
            entry = [self._cur_turn, self._pending_reason, ""]
            log.append(entry)
            del log[:-self._MAX_TURNS]
        else:
            # update reason if we just got one
            if self._pending_reason:
                entry[1] = self._pending_reason
        if action_result is not None:
            entry[2] = action_result
            self._pending_reason = ""
        self._render_turn_log()

    def _render_turn_log(self) -> None:
        w = self._panes.get("turnlog")
        if w is None:
            return
        w.delete("1.0", "end")
        for turn, reason, act in self._turn_log:
            w.insert("end", f"[{turn}] ", ("turnnum",))
            w.insert("end", f"{reason}\n" if reason else "(no reasoning)\n")
            if act:
                w.insert("end", f"      \u2192 {act}\n", ("act",))
            w.insert("end", "\n")
        # Style tags (configure once).
        try:
            w.tag_configure("turnnum", foreground="#2c3e50", font=("Segoe UI", 10, "bold"))
            w.tag_configure("act", foreground="#2e7d32", font=("Consolas", 9))
        except Exception:
            pass
        w.see("end")

    def _rebuild_stats_grid(self, kv: dict) -> None:
        """Render distinct labelled value boxes in a 2-column grid."""
        grid = self._stats_grid
        # Create labels once; update values on subsequent calls.
        if not self._stat_labels:
            keys = list(kv.keys())
            _NCOL = 3   # 3 columns keeps ~15 stats to ~5 short rows (no scroll)
            for i, k in enumerate(keys):
                r, c = divmod(i, _NCOL)
                cell = tk.Frame(grid, bd=1, relief="groove")
                cell.grid(row=r, column=c, sticky="ew", padx=2, pady=1)
                grid.grid_columnconfigure(c, weight=1)
                tk.Label(cell, text=k, font=("Segoe UI", 8), fg="#555",
                         anchor="w").pack(side="left", padx=(4, 2))
                val = tk.Label(cell, text=str(kv[k]), font=("Consolas", 9, "bold"),
                               anchor="e")
                val.pack(side="right", padx=(2, 4))
                self._stat_labels[k] = val
        else:
            for k, v in kv.items():
                if k in self._stat_labels:
                    self._stat_labels[k].config(text=str(v))

    def _rebuild_resolved(self, items: list) -> None:
        lb = self._resolved_list
        lb.delete(0, "end")
        for it in items or []:
            lb.insert("end", "\u2713 " + str(it))

    def _rebuild_tool_stats(self, data: dict) -> None:
        """data: {turns, parse_fail, parse_fail_pct, rows:[{tool,calls,ok,err,err_pct}]}"""
        self._tool_turns.config(text=f"turns: {data.get('turns', 0)}")
        pf = data.get("parse_fail", 0)
        self._tool_pfail.config(text=f"parse-fail: {pf} ({data.get('parse_fail_pct', 0):.0f}%)")
        tree = self._tool_tree
        tree.delete(*tree.get_children(""))
        for r in data.get("rows", []):
            err = r.get("err", 0)
            tags = ("err",) if err else ()
            tree.insert("", "end", text=r.get("tool", "?"),
                        values=(r.get("calls", 0), r.get("ok", 0), err,
                                f"{r.get('err_pct', 0):.0f}%"), tags=tags)

    def get_hint(self) -> Optional[str]:
        """Return the next queued user hint (or None). Called by the driver."""
        try:
            return self._hints.get_nowait()
        except queue.Empty:
            return None

    def get_ask(self) -> Optional[str]:
        """Return the next queued operator QUESTION (or None). The driver
        answers it out-of-band (interview), not as a game action."""
        try:
            return self._asks.get_nowait()
        except queue.Empty:
            return None

    def get_turn_window(self) -> Optional[int]:
        """Operator-selected action-log window size (turns of temporal memory),
        or None to use the driver default. Set via the GUI 'Turn memory' box."""
        try:
            v = self._turnmem_var.get()
            return int(v) if v and v != "default" else None
        except Exception:
            return None

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "turn":
                    self._cur_turn = payload
                    self._status.config(text=f"Turn {payload}")
                elif kind == "ctx":
                    pct, label = payload
                    try:
                        self._ctx_bar["value"] = max(0, min(100, pct))
                    except Exception:
                        pass
                    self._ctx_lbl.config(text=label)
                elif kind == "gstatus":
                    self._gstatus.config(text=payload)
                elif kind == "think":
                    # Reasoning arrives first; hold it until its action lands.
                    self._pending_reason = payload
                    self._append_turn_entry()
                elif kind == "action":
                    # Pair the action/result with the reasoning from this turn.
                    self._append_turn_entry(action_result=payload)
                elif kind == "stats_kv":
                    self._rebuild_stats_grid(payload)
                elif kind == "tool_stats":
                    self._rebuild_tool_stats(payload)
                elif kind == "resolved":
                    self._rebuild_resolved(payload)
                elif kind == "context_dump":
                    self._last_context = payload
                elif kind == "area_map":
                    self._last_area_map = payload
                elif kind == "inventory":
                    self._last_inventory = payload
                elif kind == "answer":
                    self._last_answer = payload
                    self._chat_add("agent", payload)
                    if hasattr(self, "_op_status"):
                        self._op_status.config(text="answered")
                        if self._root is not None:
                            self._root.after(2000,
                                             lambda: self._op_status.config(text=""))
                    self._show_chat()
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
            # Save geometry PERIODICALLY (not just on close): the driver process
            # is often force-killed on relaunch, which skips the close handler,
            # so persist the current size/position every few seconds while
            # running to survive a hard kill.
            self._drain_ticks = getattr(self, "_drain_ticks", 0) + 1
            if self._drain_ticks % 30 == 0:   # ~every 3s (drain runs each 100ms)
                self._save_geometry()
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
        """topics: list of {name, notes:[{step,note}]} - LLM-authored, most
        recently updated first (the agent's evolving understanding)."""
        tree = self._topic_tree
        keep_open = self._expanded_ids(tree)
        tree.delete(*tree.get_children(""))
        for t in topics or []:
            name = t.get("name", "?")
            notes = t.get("notes", [])
            meta = f"{len(notes)} notes"
            parent = tree.insert("", "end", text=name, values=(meta,),
                                 open=(name in keep_open))
            for n in notes:
                step = n.get("step", "")
                note = (n.get("note", "") or "")[:140]
                tree.insert(parent, "end", text=f"@{step}: {note}", values=("",))

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

    def _save_geometry(self) -> None:
        """Persist the current window size+position so the next launch restores
        it (like a typical app storing window state in an ini)."""
        try:
            if self._root is not None and getattr(self, "_geom_file", None):
                with open(self._geom_file, "w", encoding="utf-8") as _gf:
                    _gf.write(self._root.geometry())
        except Exception:
            pass
        # Also persist the divider (sash) positions alongside geometry.
        self._save_sashes()

    def _save_sashes(self) -> None:
        """Persist each PanedWindow's sash (divider) positions so the user's
        chosen box proportions survive close/reopen. Stored as
        '<name>=<pos0>,<pos1>,...' lines (pixel coords along the pane's orient
        axis)."""
        try:
            if self._root is None or not getattr(self, "_sash_file", None):
                return
            paneds = getattr(self, "_paneds", {})
            lines = []
            for name, pw in paneds.items():
                try:
                    n = len(pw.panes())
                except Exception:
                    continue
                coords = []
                for i in range(max(0, n - 1)):
                    try:
                        # ttk.PanedWindow.sashpos(i) returns the i-th sash offset
                        coords.append(str(pw.sashpos(i)))
                    except Exception:
                        pass
                if coords:
                    lines.append(f"{name}={','.join(coords)}")
            if lines:
                with open(self._sash_file, "w", encoding="utf-8") as _sf:
                    _sf.write("\n".join(lines))
        except Exception:
            pass

    def _restore_sashes(self) -> None:
        """Restore saved sash positions. Retries a couple of times because the
        pane sizes may not be final on the first pass right after show."""
        try:
            if self._root is None or not getattr(self, "_sash_file", None):
                return
            if not os.path.isfile(self._sash_file):
                return
            saved = {}
            with open(self._sash_file, encoding="utf-8") as _sf:
                for line in _sf:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        saved[k] = [int(x) for x in v.split(",") if x.strip()]
            paneds = getattr(self, "_paneds", {})
            for name, pw in paneds.items():
                for i, pos in enumerate(saved.get(name, [])):
                    try:
                        pw.sashpos(i, pos)
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_geometry()
        self.close()

    def close(self) -> None:
        self._save_geometry()
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

    def set_gstatus(self, text: str) -> None:
        self._q.put(("gstatus", text))

    def set_plot(self, text: str) -> None:
        self._q.put(("plot", text))

    def set_map(self, text: str) -> None:
        self._q.put(("map", text))

    def set_inventory(self, text: str) -> None:
        self._q.put(("inventory", text))

    def set_answer(self, qa: str) -> None:
        """Show the agent's answer to an operator interview question."""
        self._q.put(("answer", qa))

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
        # Superseded by the labelled stats grid + tool-stats box. No-op kept
        # for backward compatibility.
        pass

    def set_stats_kv(self, kv: dict) -> None:
        self._q.put(("stats_kv", dict(kv)))

    def set_tool_stats(self, text: str) -> None:
        self._q.put(("tool_stats", text))

    def set_resolved_quests(self, items: list) -> None:
        self._q.put(("resolved", list(items)))

    def set_context_dump(self, text: str) -> None:
        self._q.put(("context_dump", text))

    def set_area_map(self, text: str) -> None:
        self._q.put(("area_map", text))

    # observation now folded into stats/dialog; keep for compatibility
    def set_observation(self, text: str) -> None:
        pass
