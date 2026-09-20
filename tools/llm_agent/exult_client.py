"""exult_client.py - Thin client for the Exult LLM agent bridge.

The engine (built with USE_LLM_AGENT and launched with --llmagent) listens on
127.0.0.1:45999 and speaks newline-delimited JSON:

    -> {"cmd":"observe"}
    <- {"world_loaded":true,"player":{...},...}

    -> {"cmd":"act","action":{"type":"move","dir":"n"}}
    <- {"ok":true,"did":"move","dir":"n"}

    -> {"cmd":"ping"}
    <- {"ok":true,"pong":true}

This module wraps that protocol in a small synchronous client.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Optional


class ExultClient:
    """Synchronous client for the Exult agent bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = 45999, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    # -- connection management ------------------------------------------------

    def connect(self, retries: int = 30, delay: float = 1.0) -> None:
        """Connect, retrying while Exult finishes starting up."""
        last_err: Optional[Exception] = None
        for _ in range(retries):
            try:
                self._sock = socket.create_connection((self.host, self.port), self.timeout)
                self._sock.settimeout(self.timeout)
                return
            except OSError as e:  # not listening yet
                last_err = e
                time.sleep(delay)
        raise ConnectionError(
            f"could not connect to Exult agent at {self.host}:{self.port}: {last_err}"
        )

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "ExultClient":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- low-level request/response ------------------------------------------

    def _send_line(self, obj: dict) -> None:
        assert self._sock is not None, "not connected"
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self._sock.sendall(data)

    def _recv_line(self) -> dict:
        assert self._sock is not None, "not connected"
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed by Exult")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line.decode("utf-8"))

    def request(self, obj: dict) -> dict:
        self._send_line(obj)
        # Skip any asynchronous event notifications (e.g. conversation_ended)
        # so they don't desync request/response pairing.
        while True:
            resp = self._recv_line()
            if isinstance(resp, dict) and "event" in resp and "ok" not in resp:
                continue
            return resp

    # -- high-level helpers ---------------------------------------------------

    def ping(self) -> dict:
        return self.request({"cmd": "ping"})

    def observe(self) -> dict:
        return self.request({"cmd": "observe"})

    def act(self, action: dict) -> dict:
        return self.request({"cmd": "act", "action": action})

    def move(self, direction: str, speed: int = 200) -> dict:
        return self.act({"type": "move", "dir": direction, "speed": speed})

    def stop(self) -> dict:
        return self.act({"type": "stop"})

    def key(self, name: str) -> dict:
        return self.act({"type": "key", "key": name})

    def answer(self, index: Optional[int] = None, text: Optional[str] = None) -> dict:
        action: dict = {"type": "answer"}
        if index is not None:
            action["index"] = index
        if text is not None:
            action["text"] = text
        return self.act(action)

    def wait(self) -> dict:
        return self.act({"type": "wait"})

    def talk(self, name: str = "") -> dict:
        """Start a conversation with a nearby NPC (by name, or nearest)."""
        return self.request({"cmd": "talk", "name": name})
