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
        _pane(right, "quests", "Quest log (focus / actionable / blocked)", 14, mono=True)
        _pane(right, "npcs", "NPC knowledge (who, notes)", 10)
        _pane(right, "stats", "Stats & memory", 6, mono=True)

        self._root.after(100, self._drain)

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
        self._q.put(("npcs", text))

    def set_stats(self, text: str) -> None:
        self._q.put(("stats", text))

    # observation now folded into stats/dialog; keep for compatibility
    def set_observation(self, text: str) -> None:
        pass
