"""test_bridge.py - Verify the Exult agent bridge end to end (no Ollama).

Connects to a running Exult (launched with --llmagent), pings it, prints an
observation, then issues a move and confirms the engine accepted it and that
the avatar position changes over a couple of observations.

Run:
    python test_bridge.py
"""

from __future__ import annotations

import json
import sys
import time

from exult_client import ExultClient


def main() -> int:
    with ExultClient() as ex:
        print("ping:", ex.ping())

        obs = ex.observe()
        print("observe:", json.dumps(obs, indent=2)[:1200])

        if not obs.get("world_loaded"):
            print("[!] World not loaded. Start a game in Exult first "
                  "(load/continue past the menu), then re-run.")
            return 1

        start = obs.get("player") or {}
        sx, sy = start.get("tx"), start.get("ty")
        print(f"start pos = ({sx},{sy})")

        # Walk east for a few turns.
        for i in range(6):
            r = ex.move("e", speed=150)
            print(f"move e -> {r}")
            time.sleep(0.6)

        ex.stop()
        time.sleep(0.4)
        end = (ex.observe().get("player") or {})
        ex_, ey = end.get("tx"), end.get("ty")
        print(f"end pos   = ({ex_},{ey})")

        moved = (ex_ != sx or ey != sy)
        print("MOVED" if moved else "NO MOVEMENT DETECTED")
        return 0 if moved else 3


if __name__ == "__main__":
    raise SystemExit(main())
