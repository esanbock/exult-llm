"""thoughts_window.py - A live "LLM thinking" window (Tkinter, stdlib only).

Shows, for each turn:
  * the game observation the LLM was given (player, dialog, characters, objects)
  * the model's raw reply / reasoning
  * the parsed action that was sent to the game

Runs Tkinter on the main thread and lets the driver push updates from a
worker thread via a thread-safe queue.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

try:
    import tkinter as tk
    from tkinter import scrolledtext
    _TK_AVAILABLE = True
except Exception:  # tkinter may be missing on headless installs
    _TK_AVAILABLE = False


class ThoughtsWindow:
    """A simple three-pane viewer updated from a background thread."""

    def __init__(self, title: str = "Exult LLM Agent - Thoughts"):
        self.available = _TK_AVAILABLE
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._root: Optional["tk.Tk"] = None
        self._title = title
        self._turn = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if not self.available:
            print("[thoughts] tkinter not available; running without a window.")
            return
        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.geometry("760x980")

        header = tk.Label(self._root, text="Ultima VII - LLM Agent",
                          font=("Segoe UI", 14, "bold"))
        header.pack(pady=(8, 0))
        self._status = tk.Label(self._root, text="waiting for first turn...",
                                font=("Segoe UI", 10))
        self._status.pack(pady=(0, 6))

        self._panes = {}
        for key, label in (
            ("map", "Map (top-down, @ = you)"),
            ("obs", "Observation (what the LLM sees)"),
            ("dialog", "Dialog / characters / objects"),
            ("think", "LLM reasoning"),
            ("action", "Chosen action"),
        ):
            frame = tk.LabelFrame(self._root, text=label, font=("Segoe UI", 10, "bold"))
            frame.pack(fill="both", expand=True, padx=8, pady=4)
            # The map needs a fixed-width font and no word wrap to stay aligned.
            wrap = "none" if key == "map" else "word"
            txt = scrolledtext.ScrolledText(frame, height=(12 if key == "map" else 6),
                                            wrap=wrap, font=("Consolas", 10))
            txt.pack(fill="both", expand=True)
            self._panes[key] = txt

        self._root.after(100, self._drain)

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "turn":
                    self._turn = payload
                    self._status.config(text=f"Turn {payload}")
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

    def set_observation(self, text: str) -> None:
        self._q.put(("obs", text))

    def set_map(self, text: str) -> None:
        self._q.put(("map", text))

    def set_dialog(self, text: str) -> None:
        self._q.put(("dialog", text))

    def set_thinking(self, text: str) -> None:
        self._q.put(("think", text))

    def set_action(self, text: str) -> None:
        self._q.put(("action", text))
