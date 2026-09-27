"""Remote inspector for the Exult LLM agent.

The old inspector was a tkinter window baked INTO the agent process, so you
could only watch the agent from the machine it ran on. This module provides the
same view over the network instead: a small HTTP server (stdlib only, its own
thread) that any browser on another machine can open.

Design
------
`InspectorServer` implements the *same* interface as `ThoughtsWindow`
(`available`, `start`, `mainloop`, `close`, every `set_*`, plus `get_ask`,
`get_hint`, `consume_save_request`, `get_turn_window`). That makes it a drop-in:
the driver keeps calling `window.set_map(...)`, `window.get_ask()`, etc. exactly
as before. Instead of drawing widgets, each `set_*` stores the value in a
thread-safe snapshot and pushes a delta to every connected browser over
Server-Sent Events (SSE).

Endpoints (all on one port, bound to 0.0.0.0 so remote machines can reach it):
  GET  /              -> the self-contained HTML/JS inspector client
  GET  /state         -> full JSON snapshot of every panel (for initial load)
  GET  /events        -> SSE stream: one `data: {"key":...,"value":...}` per update
  POST /ask           -> {"question": "..."}  queue an interview question (!ask)
  POST /save          -> request an in-game save (driver executes it next turn)
  POST /turn-window   -> {"turns": N|null}  set temporal-memory window

No third-party dependencies: http.server + threading + json from the stdlib.
No credentials are ever read or served by this module.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


# Panels the driver pushes as plain text (key -> latest string).
_TEXT_KEYS = (
    "room", "action_digest",
    "gstatus", "plot", "map", "inventory", "dialog", "quests",
    "tool_stats", "context_dump", "area_map", "answer", "thinking",
)
# Panels pushed as structured data (lists/dicts) rendered specially by the client.
_STRUCT_KEYS = ("topics_tree", "npc_tree", "resolved", "stats_kv", "turn_log", "notes", "guard_stats")


def _primary_lan_ip() -> Optional[str]:
    """Best-effort discovery of this machine's primary LAN IP (no traffic sent).
    Used only to print a friendly URL for the remote (Windows) client."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # doesn't actually send packets
        return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return None
    finally:
        s.close()


class InspectorServer:
    """Network-facing drop-in for ThoughtsWindow.

    Every method the driver calls on the tkinter window exists here with the
    same signature; the data is stored and broadcast instead of drawn.
    """

    def __init__(self, title: str = "Exult LLM Agent - Inspector",
                 host: str = "0.0.0.0", port: int = 8092):
        # The driver checks `window.available` before calling set_*/get_*.
        # We're always available (no display needed).
        self.available = True
        self._title = title
        self._host = host
        self._port = port

        self._lock = threading.RLock()
        # The full snapshot a newly-connected client gets from GET /state.
        self._state: Dict[str, Any] = {
            "title": title,
            "turn": 0,
            "context_pct": 0,
            "context_label": "",
            "turn_window": None,   # operator override (None = driver default 450)
            "throttle_ms": 0,      # operator per-turn cooldown ms (0 = off)
            "quests_data": {},     # structured open-quests for the readable panel
        }
        for k in _TEXT_KEYS:
            self._state[k] = ""
        for k in _STRUCT_KEYS:
            self._state[k] = [] if k != "stats_kv" else {}

        # Rolling turn log pairing reasoning with the action/result it produced
        # (mirrors the tkinter version's combined log).
        self._MAX_TURNS = 20
        self._cur_turn = 0
        self._turn_log: List[Dict[str, Any]] = []
        self._pending_reason = ""

        # SSE subscribers: each connected browser gets its own Queue of deltas.
        self._subscribers: "List[queue.Queue]" = []

        # Driver-consumed inputs (same queues/flags as ThoughtsWindow).
        self._hints: "queue.Queue[str]" = queue.Queue()
        self._asks: "queue.Queue[str]" = queue.Queue()
        self._save_requested = False
        self._turn_window: Optional[int] = None
        self._throttle_ms: int = 0   # operator per-turn cooldown (0 = off)

        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the HTTP server on a daemon thread and return immediately."""
        handler = _make_handler(self)
        try:
            self._httpd = ThreadingHTTPServer((self._host, self._port), handler)
        except OSError as e:
            print(f"[inspector] could not bind {self._host}:{self._port}: {e}")
            self.available = False
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="inspector-http", daemon=True)
        self._thread.start()
        # Print a concrete URL the (often Windows) client can open. When bound
        # to all interfaces, resolve the primary LAN IP so the user doesn't have
        # to guess it.
        url_host = self._host
        if self._host in ("", "0.0.0.0"):
            url_host = _primary_lan_ip() or "<this-machine-ip>"
        print(f"[inspector] serving on http://{url_host}:{self._port}/  "
              f"(open this in a browser on the Windows box)")

    def mainloop(self) -> None:
        """Block forever (server runs on its own thread). Provided so the
        driver can treat this like the tkinter window if it ever calls it."""
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None

    # -- broadcast helpers ----------------------------------------------------

    def _broadcast(self, key: str, value: Any) -> None:
        """Store the latest value and push a delta to every connected client."""
        with self._lock:
            self._state[key] = value
            subs = list(self._subscribers)
        msg = json.dumps({"key": key, "value": value})
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # a slow client just misses a delta; it can re-fetch /state

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))  # deep copy, JSON-safe

    def _subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(q)
        return q

    def _unsubscribe(self, q: "queue.Queue") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # -- turn-log pairing (reasoning <-> action) ------------------------------

    def _append_turn_entry(self, action_result: Optional[str] = None) -> None:
        with self._lock:
            log = self._turn_log
            if action_result is None:
                # New reasoning arrived; open a fresh entry for this turn.
                log.append({"turn": self._cur_turn,
                            "reason": self._pending_reason,
                            "action": "", "thought": ""})
            else:
                # Action landed; attach it to the latest open entry (or make one).
                if log and log[-1]["turn"] == self._cur_turn and not log[-1]["action"]:
                    log[-1]["action"] = action_result
                else:
                    log.append({"turn": self._cur_turn,
                                "reason": self._pending_reason,
                                "action": action_result, "thought": ""})
            del log[:-self._MAX_TURNS]
            snapshot = list(log)
        self._broadcast("turn_log", snapshot)

    # -- updates (thread-safe): mirror ThoughtsWindow signatures --------------

    def update_turn(self, turn: int) -> None:
        with self._lock:
            self._cur_turn = turn
        self._broadcast("turn", turn)

    def set_context(self, pct: int, label: str) -> None:
        with self._lock:
            self._state["context_pct"] = max(0, min(100, pct))
            self._state["context_label"] = label
        self._broadcast("context", {"pct": max(0, min(100, pct)), "label": label})

    def set_gstatus(self, text: str) -> None:
        self._broadcast("gstatus", text)

    def set_plot(self, text: str) -> None:
        self._broadcast("plot", text)

    def set_map(self, text: str) -> None:
        self._broadcast("map", text)

    def set_inventory(self, text: str) -> None:
        self._broadcast("inventory", text)

    def set_answer(self, qa: str) -> None:
        self._broadcast("answer", qa)

    def set_thinking(self, text: str) -> None:
        # Reasoning arrives first; open a turn entry so the action can pair to it.
        with self._lock:
            self._pending_reason = text
        self._broadcast("thinking", text)
        self._append_turn_entry()

    def set_action(self, text: str) -> None:
        # The driver calls set_action either with a plain phrase string OR with
        # a dict {"action": <phrase>, "result": <short result>} (the tkinter
        # window paired those into its combined log). Normalize to a string here
        # so the web client never renders a raw object as "[object Object]".
        if isinstance(text, dict):
            _act = str(text.get("action", "")).strip()
            _res = str(text.get("result", "")).strip()
            text = f"{_act} -> {_res}" if _res else _act
        elif not isinstance(text, str):
            text = str(text)
        self._append_turn_entry(action_result=text)

    def set_thought(self, text: str) -> None:
        with self._lock:
            log = self._turn_log
            if log and log[-1]["turn"] == self._cur_turn:
                log[-1]["thought"] = str(text)
            snapshot = list(log)
        self._broadcast("turn_log", snapshot)

    def set_dialog(self, text: str) -> None:
        self._broadcast("dialog", text)

    def set_room(self, text: str) -> None:
        """Zork-style narrated room description."""
        self._broadcast("room", text)

    def set_action_digest(self, text: str) -> None:
        """Factual digest of turns that scrolled off the shown action log."""
        self._broadcast("action_digest", text)

    def set_quests(self, text: str) -> None:
        self._broadcast("quests", text)

    def set_npcs(self, text: str) -> None:
        pass  # superseded by npc_tree (kept for compatibility)

    def set_topics_tree(self, topics: list) -> None:
        self._broadcast("topics_tree", list(topics or []))

    def set_notes(self, notes: list) -> None:
        """Aggregated per-NPC notes/journal for the dedicated Notes panel."""
        self._broadcast("notes", list(notes or []))

    def set_guard_stats(self, rows: list) -> None:
        """Per-run guard-firing counts for the dedicated Guard-stats panel."""
        self._broadcast("guard_stats", list(rows or []))

    def set_quests_data(self, data: dict) -> None:
        """Structured quest data (open[] + counts) for a readable quests panel."""
        self._broadcast("quests_data", dict(data or {}))

    def set_npc_tree(self, chars: list) -> None:
        self._broadcast("npc_tree", list(chars or []))

    def set_stats(self, text: str) -> None:
        pass  # superseded by stats_kv (kept for compatibility)

    def set_stats_kv(self, kv: dict) -> None:
        self._broadcast("stats_kv", dict(kv or {}))

    def set_tool_stats(self, text: str) -> None:
        self._broadcast("tool_stats", text)

    def set_resolved_quests(self, items: list) -> None:
        self._broadcast("resolved", list(items or []))

    def set_context_dump(self, text: str) -> None:
        self._broadcast("context_dump", text)

    def set_area_map(self, text: str) -> None:
        self._broadcast("area_map", text)

    def set_observation(self, text: str) -> None:
        pass  # folded into stats/dialog (kept for compatibility)

    def set_save_ack(self, ok: bool) -> None:
        self._broadcast("save_ack", bool(ok))

    # -- inputs consumed by the driver ----------------------------------------

    def get_hint(self) -> Optional[str]:
        try:
            return self._hints.get_nowait()
        except queue.Empty:
            return None

    def get_ask(self) -> Optional[str]:
        try:
            return self._asks.get_nowait()
        except queue.Empty:
            return None

    def consume_save_request(self) -> bool:
        with self._lock:
            if self._save_requested:
                self._save_requested = False
                return True
            return False

    def get_turn_window(self) -> Optional[int]:
        with self._lock:
            return self._turn_window

    def get_throttle_ms(self) -> int:
        """Operator per-turn cooldown in ms (0 = no extra delay). Lets you slow
        the agent down to cool the LLM box when it runs hot."""
        with self._lock:
            return self._throttle_ms

    # -- called by the HTTP handler (remote client actions) -------------------

    def _remote_ask(self, question: str) -> None:
        if question and question.strip():
            self._asks.put(question.strip())

    def _remote_hint(self, hint: str) -> None:
        if hint and hint.strip():
            self._hints.put(hint.strip())

    def _remote_save(self) -> None:
        with self._lock:
            self._save_requested = True

    def _remote_turn_window(self, turns: Optional[int]) -> None:
        with self._lock:
            self._turn_window = turns
        # Store in the snapshot and broadcast so the UI can confirm the value
        # actually registered (and a page refresh shows the current setting).
        self._broadcast("turn_window", turns)

    def _remote_throttle(self, ms: int) -> None:
        ms = max(0, min(int(ms or 0), 5000))
        with self._lock:
            self._throttle_ms = ms
        self._broadcast("throttle_ms", ms)


def _make_handler(server: "InspectorServer"):
    """Build a request handler bound to a specific InspectorServer instance."""

    class Handler(BaseHTTPRequestHandler):
        # Quieter logs (the driver's stdout is precious).
        def log_message(self, *args):  # noqa: N802
            pass

        # -- helpers ----------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str,
                  extra: Optional[dict] = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def _read_json(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n > 0 else b""
                return json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                return {}

        # -- routes -----------------------------------------------------------
        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._json(200, server._snapshot())
            elif self.path == "/events":
                self._serve_sse()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path == "/ask":
                q = (self._read_json() or {}).get("question", "")
                server._remote_ask(str(q))
                self._json(200, {"ok": True})
            elif self.path == "/hint":
                h = (self._read_json() or {}).get("hint", "")
                server._remote_hint(str(h))
                self._json(200, {"ok": True})
            elif self.path == "/save":
                server._remote_save()
                self._json(200, {"ok": True})
            elif self.path == "/turn-window":
                body = self._read_json() or {}
                t = body.get("turns", None)
                try:
                    t = int(t) if t not in (None, "", "default") else None
                except (TypeError, ValueError):
                    t = None
                server._remote_turn_window(t)
                self._json(200, {"ok": True, "turns": t})
            elif self.path == "/throttle":
                body = self._read_json() or {}
                try:
                    ms = int(body.get("ms", 0) or 0)
                except (TypeError, ValueError):
                    ms = 0
                server._remote_throttle(ms)
                self._json(200, {"ok": True, "ms": max(0, min(ms, 5000))})
            else:
                self._send(404, b"not found", "text/plain")

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q = server._subscribe()
            try:
                # Prime the client with a full snapshot as one event.
                self.wfile.write(b"event: snapshot\n")
                self.wfile.write(b"data: " +
                                 json.dumps(server._snapshot()).encode("utf-8") +
                                 b"\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(b"data: " + msg.encode("utf-8") + b"\n\n")
                    except queue.Empty:
                        # keep-alive comment so proxies/browsers hold the stream
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError):
                pass
            finally:
                server._unsubscribe(q)

    return Handler


# The entire inspector client: one self-contained HTML page (no external deps).
_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Exult LLM Agent - Inspector</title>
<style>
  :root { --bg:#0f1216; --panel:#181d24; --panel2:#212832; --fg:#dfe6ee;
          --muted:#8a95a3; --accent:#5ab0ff; --good:#4ec97a; --warn:#e0b34a;
          --err:#ff5c5c; --think:#b58bff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  /* Header */
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
           padding:6px 12px; background:var(--panel2); border-bottom:1px solid #000;
           position:sticky; top:0; z-index:10; }
  header b { font-size:13px; }
  .chip { font-size:11px; padding:2px 8px; border-radius:10px; background:#0c0f14; color:var(--muted); }
  .chip.hp { color:var(--good); } .chip.err { color:var(--err); }
  #conn { background:#3a2323; color:#e88; } #conn.ok { background:#213a24; color:var(--good); }
  .ctxwrap{display:inline-flex;align-items:center;gap:8px;}
  .ctxbar{position:relative;width:220px;height:18px;background:#0c0f14;border:1px solid #2a323d;
          border-radius:4px;overflow:hidden;display:inline-block;vertical-align:middle;}
  .ctxfill{position:absolute;left:0;top:0;height:100%;width:0;background:var(--accent);transition:width .3s;}
  .ctxpct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
          font-size:11px;font-weight:600;color:#fff;text-shadow:0 0 3px #000;}
  .ctxlbl{font-size:10.5px;color:var(--muted);}
  input,button,select { font:inherit; background:var(--panel); color:var(--fg);
         border:1px solid #333; border-radius:4px; padding:3px 7px; }
  button{cursor:pointer;} button:hover{border-color:var(--accent);}
  /* Layout: big left (the turn + feed), right rail (plot/quest/notes/stats), map + debug below */
  main { display:grid; grid-template-columns: 1.6fr 1fr; gap:8px; padding:8px; }
  .stack{display:flex;flex-direction:column;gap:8px;min-width:0;}
  .card { background:var(--panel); border:1px solid #000; border-radius:6px; min-width:0;}
  .card>h2{margin:0;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;
           color:var(--muted);padding:5px 10px;border-bottom:1px solid #000;background:var(--panel2);
           cursor:default;display:flex;justify-content:space-between;}
  .card .body{padding:8px 10px;overflow:auto;}
  /* THE TURN - centerpiece */
  #now .body{padding:10px 12px;}
  .now-act{font-size:17px;font-weight:600;color:var(--fg);margin-bottom:2px;}
  .now-act .out{font-size:13px;font-weight:400;color:var(--good);}
  .now-act .out.err{color:var(--err);}
  .now-reason{color:#cdd6e0;margin:6px 0;white-space:pre-wrap;}
  .now-think{color:var(--think);font-size:12px;white-space:pre-wrap;opacity:.85;border-left:2px solid var(--think);padding-left:8px;margin-top:6px;}
  /* live feed */
  #feed .body{max-height:340px;padding:4px 0;}
  .frow{padding:3px 10px;border-bottom:1px solid #14181e;display:flex;gap:8px;}
  .frow .ft{color:var(--warn);flex:0 0 46px;}
  .frow .fa{flex:1;min-width:0;}
  .frow .fa .fr{color:var(--muted);font-size:11px;}
  .frow .out{color:var(--good);} .frow .out.err{color:var(--err);font-weight:600;}
  /* right rail */
  .card.tall .body{max-height:220px;}
  .name{color:var(--accent);cursor:pointer;} .child{margin-left:14px;} .child.dim{color:var(--muted);}
  table.kv{width:100%;border-collapse:collapse;} table.kv td{padding:1px 6px;border-bottom:1px solid #14181e;}
  table.kv td.k{color:var(--muted);}
  .mono{white-space:pre;font-size:11.5px;line-height:1.15;}
  .pre{white-space:pre-wrap;}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
  /* collapsible */
  details{background:var(--panel);border:1px solid #000;border-radius:6px;}
  details>summary{padding:6px 10px;color:var(--muted);font-size:10.5px;text-transform:uppercase;
                  letter-spacing:.6px;cursor:pointer;background:var(--panel2);border-radius:6px;}
  details[open]>summary{border-bottom:1px solid #000;border-radius:6px 6px 0 0;}
  details .body{padding:8px 10px;overflow:auto;max-height:300px;}
  #wide{grid-column:1 / -1;display:flex;flex-direction:column;gap:8px;}
</style>
</head>
<body>
<header>
  <b>Exult LLM Inspector</b>
  <span id="conn" class="chip">connecting…</span>
  <span id="c-turn" class="chip">turn —</span>
  <span id="c-loc" class="chip"></span>
  <span id="c-hp" class="chip hp"></span>
  <span id="c-pos" class="chip"></span>
  <span class="ctxwrap" title="Context window usage">
    <span class="ctxbar"><span class="ctxfill" id="ctxfill"></span><span class="ctxpct" id="ctxpct">0%</span></span>
    <span class="ctxlbl" id="ctxlbl"></span>
  </span>
  <span id="c-model" class="chip"></span>
  <span style="flex:1"></span>
  <input id="askbox" placeholder="Ask the agent…" size="22"/>
  <button id="askbtn">Ask</button>
  <button id="savebtn">Save</button>
  <select id="turnwin" title="Action-log memory window (default 450; auto-shrinks under context pressure)">
    <option value="default">mem: default (450)</option>
    <option value="100">100</option><option value="200">200</option>
    <option value="300">300</option><option value="450">450</option>
    <option value="600">600</option><option value="800">800</option><option value="1000">1000</option>
  </select>
  <span id="turnwin-ok" style="color:var(--good);font-size:11px;"></span>
  <label style="font-size:11px;color:var(--muted);" title="Extra delay before each LLM request (ms) to cool the box. 0=full speed; a Granite turn is ~1.2s compute, so use 2000-4000 to meaningfully lower GPU duty cycle. Max 5000.">
    throttle <input id="throttle" type="number" min="0" max="5000" step="100" value="0" style="width:64px;"/> ms
  </label>
  <span id="throttle-ok" style="color:var(--good);font-size:11px;"></span>
</header>

<main>
  <!-- LEFT: the turn (centerpiece) + live feed -->
  <div class="stack">
    <div class="card" id="now"><h2>Now — action · reasoning · thinking · outcome</h2>
      <div class="body">
        <div class="now-act" id="now-act">—</div>
        <div class="now-reason" id="now-reason"></div>
        <div class="now-think" id="now-think" style="display:none"></div>
      </div>
    </div>
    <div class="card" id="feed"><h2>Live activity (recent turns)</h2><div class="body" id="feed-body"></div></div>
    <div class="card"><h2>Room</h2><div class="body" id="p-room"></div></div>
    <div class="card"><h2>Dialog</h2><div class="body" id="p-dialog"></div></div>
  </div>

  <!-- RIGHT rail: plot, quests, notes, stats (priority order) -->
  <div class="stack">
    <div class="card tall"><h2>Plot</h2><div class="body" id="p-plot"></div></div>
    <div class="card tall"><h2>Earlier this session (scrolled-off digest)</h2><div class="body pre" id="p-action_digest"></div></div>
    <div class="card tall"><h2>Open quests</h2><div class="body" id="p-quests"></div></div>
    <div class="card tall"><h2>Resolved quests</h2><div class="body" id="p-resolved"></div></div>
    <div class="card tall"><h2>Notes / Journal</h2><div class="body" id="p-notes"></div></div>
    <div class="card tall"><h2>Stats</h2><div class="body" id="p-stats_kv"></div></div>
    <div class="card tall"><h2>Guard firings (this run)</h2><div class="body" id="p-guard_stats"></div></div>
    <div class="card tall"><h2>Inventory</h2><div class="body pre" id="p-inventory"></div></div>
  </div>

  <!-- FULL WIDTH BELOW: map + tool stats side by side, then collapsible debug -->
  <div id="wide">
    <div class="row2">
      <div class="card"><h2>Tool stats</h2><div class="body pre" id="p-tool_stats"></div></div>
      <div class="card"><h2>Map</h2><div class="body mono" id="p-map"></div></div>
    </div>
    <details><summary>NPCs met</summary><div class="body" id="p-npc_tree"></div></details>
    <details><summary>Topics</summary><div class="body" id="p-topics_tree"></div></details>
    <details><summary>Area map</summary><div class="body mono" id="p-area_map"></div></details>
    <details><summary>Full context (prompt sent to the model)</summary><div class="body mono" id="p-context_dump"></div></details>
    <details><summary>Game status (raw)</summary><div class="body" id="p-gstatus"></div></details>
  </div>
</main>

<script>
const $=id=>document.getElementById(id);
function esc(s){ if(s==null)s=""; else if(typeof s==="object"){try{s=JSON.stringify(s);}catch(e){s=String(s);}} else s=String(s);
  return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function setText(id,v){const e=$(id); if(e)e.textContent=(v==null?"":String(v));}
// An action string like "move -> blocked: ..." is an error if it mentions these.
function isErr(s){ return /(?:->|:)\s*(blocked|failed|error|no path|unknown|bad )/i.test(String(s||"")); }
function splitAct(s){ // "goto (x,y) -> ok"  =>  {act, out}
  const m=String(s||"").split(/\s*->\s*/); return {act:m[0]||"", out:m.slice(1).join(" -> ")}; }

function renderNow(entries){
  const e=(entries&&entries.length)?entries[entries.length-1]:null;
  if(!e){return;}
  const {act,out}=splitAct(e.action);
  const err=isErr(e.action);
  $("now-act").innerHTML='#'+esc(e.turn)+'  '+esc(act||"(thinking…)")+
    (out?' <span class="out'+(err?' err':'')+'">→ '+esc(out)+'</span>':'');
  $("now-reason").textContent=e.reason||"";
  const th=$("now-think");
  if(e.thought){th.style.display="";th.textContent="thinking: "+e.thought;} else {th.style.display="none";}
}
function renderFeed(entries){
  const box=$("feed-body"); box.innerHTML="";
  (entries||[]).slice().reverse().forEach(e=>{
    const {act,out}=splitAct(e.action); const err=isErr(e.action);
    const row=document.createElement("div"); row.className="frow";
    row.innerHTML='<span class="ft">#'+esc(e.turn)+'</span><span class="fa">'+
      esc(act)+(out?' <span class="out'+(err?' err':'')+'">→ '+esc(out)+'</span>':'')+
      (e.reason?'<div class="fr">'+esc(e.reason.slice(0,160))+'</div>':'')+'</span>';
    box.appendChild(row);
  });
}
function renderKV(kv){const box=$("p-stats_kv");box.innerHTML="";const t=document.createElement("table");t.className="kv";
  Object.entries(kv||{}).forEach(([k,v])=>{const tr=document.createElement("tr");
    tr.innerHTML='<td class="k">'+esc(k)+'</td><td>'+esc(v)+'</td>';t.appendChild(tr);});box.appendChild(t);}
function renderList(id,items,fmt){const box=$(id);box.innerHTML="";
  (items||[]).forEach(it=>{const d=document.createElement("div");d.className="child";d.textContent=fmt(it);box.appendChild(d);});}
function renderNotes(groups){const box=$("p-notes");box.innerHTML="";
  (groups||[]).forEach(g=>{const h=document.createElement("div");h.className="name";
    h.textContent=(g.npc||"?")+" ("+(g.total_notes||(g.notes||[]).length)+")";box.appendChild(h);
    (g.notes||[]).forEach(n=>{const d=document.createElement("div");d.className="child dim";d.textContent="• "+n;box.appendChild(d);});});}
function renderQuests(data){
  const box=$("p-quests"); if(!box) return; box.innerHTML="";
  const open=(data&&data.open)||[];
  if(!open.length){box.innerHTML='<span class="child dim">(no open quests)</span>';return;}
  function pcol(p){return p<=2?"#ff7a7a":p<=5?"var(--warn)":"var(--muted)";}
  const byId={}; open.forEach(q=>byId[q.id]=q);
  // Partition into top-level quests vs sub-tasks (those whose depends_on points
  // at another OPEN quest). Sub-tasks render nested under their parent.
  const isSub=q=>(q.depends_on||[]).some(d=>byId[d]);
  const tops=open.filter(q=>!isSub(q));
  const subsOf=id=>open.filter(q=>(q.depends_on||[]).includes(id));
  function qrow(q,indent){
    const row=document.createElement("div");
    row.style.cssText="padding:3px 0;"+(indent?"margin-left:16px;":"border-bottom:1px solid #14181e;");
    let h=(indent?'<span class="child dim">↳</span> ':'')+
      '<span style="display:inline-block;min-width:26px;font-weight:700;color:'+pcol(q.priority||9)+';">P'+esc(q.priority||"?")+'</span> '+
      '<span'+(indent?' class="child dim"':'')+'>'+esc(q.title||q.id||"?")+'</span>';
    if(q.npc) h+=' <span class="child dim">['+esc(q.npc)+']</span>';
    if(!indent && q.prereqs_unmet && q.prereqs_unmet.length)
      h+='<div class="child dim" style="color:#e0857a;">⛔ blocked on: '+esc(q.prereqs_unmet.join(", "))+'</div>';
    if(q.notes) h+='<div class="child dim">'+esc(String(q.notes).slice(0,110))+'</div>';
    row.innerHTML=h; box.appendChild(row);
  }
  tops.forEach(q=>{ qrow(q,false); subsOf(q.id).forEach(s=>qrow(s,true)); });
  const f=document.createElement("div"); f.className="child dim"; f.style.marginTop="4px";
  f.textContent=(data.unresolved||open.length)+" open · "+(data.resolved||0)+" resolved";
  box.appendChild(f);
}
function renderGuards(rows){
  const box=$("p-guard_stats"); if(!box) return; box.innerHTML="";
  if(!rows||!rows.length){box.innerHTML='<span class="child dim">(none yet)</span>';return;}
  const max=Math.max(...rows.map(r=>r.count||0),1);
  const t=document.createElement("table"); t.className="kv";
  rows.forEach(r=>{
    const tr=document.createElement("tr");
    const pct=Math.round(100*(r.count||0)/max);
    tr.innerHTML='<td class="k" style="white-space:nowrap;">'+esc(r.guard)+'</td>'+
      '<td style="width:99%;"><span style="display:inline-block;height:10px;width:'+pct+'%;'+
      'background:var(--accent);border-radius:2px;vertical-align:middle;"></span> '+
      '<b>'+esc(r.count)+'</b></td>';
    t.appendChild(tr);
  });
  box.appendChild(t);
}
function renderTree(id,nodes,spec){const box=$(id);
  // Preserve which top-level nodes are expanded so a live update doesn't
  // collapse the one the user just opened.
  const openNames=new Set();
  box.querySelectorAll("[data-node]").forEach(el=>{
    if(el.dataset.open==="1") openNames.add(el.dataset.node);
  });
  box.innerHTML="";
  (nodes||[]).forEach(n=>{const nm=spec.name(n);
    const wrap=document.createElement("div");wrap.dataset.node=nm;
    const head=document.createElement("div");
    head.innerHTML='<span class="name">▸ '+esc(nm)+'</span> <span class="child dim">'+esc(spec.meta(n))+'</span>';
    const kids=document.createElement("div");
    spec.children(n).forEach(c=>{const k=document.createElement("div");k.className="child dim";k.textContent=c;kids.appendChild(k);});
    const isOpen=openNames.has(nm);
    kids.style.display=isOpen?"block":"none"; wrap.dataset.open=isOpen?"1":"0";
    head.querySelector(".name").onclick=()=>{const o=kids.style.display==="none";
      kids.style.display=o?"block":"none"; wrap.dataset.open=o?"1":"0";};
    wrap.appendChild(head);wrap.appendChild(kids);box.appendChild(wrap);});}
const npcSpec={name:n=>n.name||"?",meta:n=>"x"+(n.times_talked||0),children:n=>{let o=[];
  (n.transcript||[]).forEach(e=>o.push(e.said?((n.name||"")+": "+e.said):("you: "+(e.me||""))));
  (n.topics_unasked||[]).forEach(t=>o.push("not asked: "+t));(n.topics_asked||[]).forEach(t=>o.push("asked: "+t));
  (n.notes||[]).slice(-8).forEach(x=>o.push("note: "+x));return o;}};
const topicSpec={name:t=>t.name||"?",meta:t=>((t.notes||t.mentions||[]).length)+" notes",children:t=>{let o=[];
  (t.notes||[]).forEach(n=>o.push("@"+(n.step||"")+": "+(n.note||"")));
  (t.mentions||[]).forEach(m=>o.push((m.npc||"")+": "+(m.said||"")));return o;}};

function apply(key,value){
  switch(key){
    case "turn": setText("c-turn","turn "+value); break;
    case "context":{const p=value.pct||0;$("ctxfill").style.width=p+"%";
      $("ctxfill").style.background=p>90?"#ff5c5c":p>75?"#e0b34a":"#5ab0ff";
      setText("ctxpct",(p||0)+"%"); setText("ctxlbl",value.label||"");break;}
    case "turn_log": renderNow(value); renderFeed(value); break;
    case "stats_kv": renderKV(value); break;
    case "notes": renderNotes(value); break;
    case "guard_stats": renderGuards(value); break;
    case "quests_data": renderQuests(value); break;
    case "quests": break;  // superseded by quests_data (structured renderer)
    case "npc_tree": renderTree("p-npc_tree",value,npcSpec); break;
    case "topics_tree": renderTree("p-topics_tree",value,topicSpec); break;
    case "resolved": renderList("p-resolved",value,x=>typeof x==="string"?x:(x.name||JSON.stringify(x))); break;
    case "room": setText("p-room",value); break;
    case "gstatus": setText("p-gstatus",value);
      // pull HP/pos/location chips out of the status line for the header
      { const t=String(value||""); const hp=t.match(/hp\s+(\d+\/\d+)/i); const pos=t.match(/pos\(([^)]+)\)/i);
        if(hp){$("c-hp").textContent="HP "+hp[1]; $("c-hp").className="chip hp"+(/(^|\D)[0-3]\//.test(hp[1])?" err":"");}
        if(pos)$("c-pos").textContent=pos[1]; } break;
    case "save_ack":{const b=$("savebtn");b.textContent=value?"Saved!":"Save failed";setTimeout(()=>b.textContent="Save",2500);break;}
    case "turn_window":{const s=$("turnwin");s.value=(value==null?"default":String(value));
      const st=$("turnwin-ok");if(st){st.textContent="✓ "+(value==null?"default":value);setTimeout(()=>st.textContent="",2500);}break;}
    case "throttle_ms":{const t=$("throttle");if(t&&document.activeElement!==t)t.value=(value||0);
      const st=$("throttle-ok");if(st){st.textContent=(value>0?("✓ "+value+"ms"):"✓ off");setTimeout(()=>st.textContent="",2500);}break;}
    default: setText("p-"+key,value);
  }
}
function applySnapshot(s){
  setText("c-turn","turn "+(s.turn||0));
  setText("c-model", s.title? "" : "");
  apply("context",{pct:s.context_pct||0,label:s.context_label||""});
  ["gstatus","plot","map","area_map","inventory","dialog","tool_stats","context_dump","room","action_digest"].forEach(k=>apply(k,s[k]));
  apply("turn_log",s.turn_log||[]);apply("stats_kv",s.stats_kv||{});apply("notes",s.notes||[]);
  apply("npc_tree",s.npc_tree||[]);apply("topics_tree",s.topics_tree||[]);apply("resolved",s.resolved||[]);
  apply("quests_data",s.quests_data||{});
  apply("turn_window",s.turn_window);
  apply("throttle_ms",s.throttle_ms||0);
  apply("guard_stats",s.guard_stats||[]);
}
function connect(){
  const es=new EventSource("/events");
  es.addEventListener("snapshot",ev=>{try{applySnapshot(JSON.parse(ev.data));}catch(e){}});
  es.onmessage=ev=>{try{const m=JSON.parse(ev.data);apply(m.key,m.value);}catch(e){}};
  es.onopen=()=>{const c=$("conn");c.textContent="live";c.className="chip ok";};
  es.onerror=()=>{const c=$("conn");c.textContent="reconnecting…";c.className="chip";es.close();setTimeout(connect,2000);};
}
async function post(p,b){try{await fetch(p,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})});}catch(e){}}
$("askbtn").onclick=()=>{const q=$("askbox").value.trim();if(q){post("/ask",{question:q});$("askbox").value="";}};
$("askbox").addEventListener("keydown",e=>{if(e.key==="Enter")$("askbtn").click();});
$("savebtn").onclick=()=>post("/save",{});
$("turnwin").onchange=e=>post("/turn-window",{turns:e.target.value});
$("throttle").onchange=e=>{let v=parseInt(e.target.value||"0",10);if(isNaN(v))v=0;v=Math.max(0,Math.min(v,5000));e.target.value=v;post("/throttle",{ms:v});};
connect();
</script>
</body>
</html>
"""
