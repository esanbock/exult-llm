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
    "gstatus", "plot", "map", "inventory", "dialog", "quests",
    "tool_stats", "context_dump", "area_map", "answer", "thinking",
)
# Panels pushed as structured data (lists/dicts) rendered specially by the client.
_STRUCT_KEYS = ("topics_tree", "npc_tree", "resolved", "stats_kv", "turn_log")


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

    def set_quests(self, text: str) -> None:
        self._broadcast("quests", text)

    def set_npcs(self, text: str) -> None:
        pass  # superseded by npc_tree (kept for compatibility)

    def set_topics_tree(self, topics: list) -> None:
        self._broadcast("topics_tree", list(topics or []))

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
  :root { --bg:#12151b; --panel:#1b2029; --panel2:#232a35; --fg:#d7dde5;
          --muted:#8b96a5; --accent:#5ab0ff; --good:#4ec97a; --warn:#e0b34a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { display:flex; align-items:center; gap:14px; padding:8px 14px;
           background:var(--panel2); border-bottom:1px solid #000; position:sticky; top:0; z-index:5; }
  header h1 { font-size:14px; margin:0; font-weight:600; }
  #conn { font-size:11px; padding:2px 8px; border-radius:10px; background:#3a2323; color:#e88; }
  #conn.ok { background:#213a24; color:var(--good); }
  #turn { color:var(--muted); }
  .ctxwrap { flex:1; max-width:340px; }
  .ctxbar { height:10px; background:#0c0f14; border-radius:5px; overflow:hidden; }
  .ctxfill { height:100%; width:0; background:var(--accent); transition:width .3s; }
  .ctxlbl { font-size:10px; color:var(--muted); margin-top:2px; }
  .controls { display:flex; gap:6px; align-items:center; }
  input,button,select { font:inherit; background:var(--panel); color:var(--fg);
         border:1px solid #333; border-radius:4px; padding:4px 7px; }
  button { cursor:pointer; }
  button:hover { border-color:var(--accent); }
  #grid { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; padding:8px; }
  .panel { background:var(--panel); border:1px solid #000; border-radius:6px;
           display:flex; flex-direction:column; min-height:120px; max-height:420px; }
  .panel h2 { margin:0; font-size:11px; text-transform:uppercase; letter-spacing:.5px;
              color:var(--muted); padding:6px 10px; border-bottom:1px solid #000; background:var(--panel2); }
  .panel .body { padding:8px 10px; overflow:auto; white-space:pre-wrap; flex:1; }
  .panel.wide { grid-column:span 2; }
  .panel.tall .body { max-height:380px; }
  .mono { white-space:pre; font-size:12px; }
  .tree .node { margin:2px 0; }
  .tree .name { color:var(--accent); cursor:pointer; }
  .tree .meta { color:var(--muted); font-size:11px; }
  .tree .child { margin-left:16px; color:var(--fg); }
  .tree .child.dim { color:var(--muted); }
  table.kv { width:100%; border-collapse:collapse; }
  table.kv td { padding:2px 6px; border-bottom:1px solid #000; }
  table.kv td.k { color:var(--muted); }
  .turnentry { border-bottom:1px solid #000; padding:5px 0; }
  .turnentry .t { color:var(--warn); }
  .turnentry .r { color:var(--fg); }
  .turnentry .a { color:var(--good); }
  .turnentry .th { color:var(--muted); font-size:11px; }
  #answer { color:var(--good); }
</style>
</head>
<body>
<header>
  <h1>Exult LLM Inspector</h1>
  <span id="conn">connecting…</span>
  <span id="turn">turn —</span>
  <div class="ctxwrap">
    <div class="ctxbar"><div class="ctxfill" id="ctxfill"></div></div>
    <div class="ctxlbl" id="ctxlbl"></div>
  </div>
  <div class="controls">
    <input id="askbox" placeholder="Ask the agent…" size="26"/>
    <button id="askbtn">Ask</button>
    <button id="savebtn">Save game</button>
    <select id="turnwin" title="Temporal memory window">
      <option value="default">turn mem: default</option>
      <option value="5">5</option><option value="10">10</option>
      <option value="20">20</option><option value="40">40</option>
    </select>
  </div>
</header>

<div id="grid">
  <div class="panel"><h2>Game status</h2><div class="body" id="p-gstatus"></div></div>
  <div class="panel"><h2>Dialog</h2><div class="body" id="p-dialog"></div></div>
  <div class="panel"><h2>Latest answer</h2><div class="body" id="answer"></div></div>

  <div class="panel wide tall"><h2>Reasoning / actions (recent turns)</h2><div class="body" id="p-turnlog"></div></div>
  <div class="panel tall"><h2>Plot summary</h2><div class="body" id="p-plot"></div></div>

  <div class="panel tall"><h2>Map</h2><div class="body mono" id="p-map"></div></div>
  <div class="panel tall"><h2>Area map</h2><div class="body mono" id="p-area_map"></div></div>
  <div class="panel tall"><h2>Inventory</h2><div class="body" id="p-inventory"></div></div>

  <div class="panel tall"><h2>Open quests</h2><div class="body" id="p-quests"></div></div>
  <div class="panel tall"><h2>Resolved quests</h2><div class="body" id="p-resolved"></div></div>
  <div class="panel tall"><h2>NPCs</h2><div class="body tree" id="p-npc_tree"></div></div>

  <div class="panel tall"><h2>Topics</h2><div class="body tree" id="p-topics_tree"></div></div>
  <div class="panel"><h2>Stats</h2><div class="body" id="p-stats_kv"></div></div>
  <div class="panel tall"><h2>Tool stats</h2><div class="body mono" id="p-tool_stats"></div></div>

  <div class="panel wide tall"><h2>Full context (prompt)</h2><div class="body mono" id="p-context_dump"></div></div>
</div>

<script>
const $ = id => document.getElementById(id);
function setText(id, v){ const e=$(id); if(e) e.textContent = (v==null?"":String(v)); }

function renderTurnLog(entries){
  const box=$("p-turnlog"); box.innerHTML="";
  (entries||[]).slice().reverse().forEach(e=>{
    const d=document.createElement("div"); d.className="turnentry";
    let h='<span class="t">#'+e.turn+'</span> ';
    if(e.reason) h+='<span class="r">'+esc(e.reason)+'</span>';
    if(e.action) h+='<div class="a">→ '+esc(e.action)+'</div>';
    if(e.thought) h+='<div class="th">'+esc(e.thought)+'</div>';
    d.innerHTML=h; box.appendChild(d);
  });
}
function renderKV(kv){
  const box=$("p-stats_kv"); box.innerHTML="";
  const t=document.createElement("table"); t.className="kv";
  Object.entries(kv||{}).forEach(([k,v])=>{
    const tr=document.createElement("tr");
    tr.innerHTML='<td class="k">'+esc(k)+'</td><td>'+esc(v)+'</td>';
    t.appendChild(tr);
  });
  box.appendChild(t);
}
function renderList(id, items, fmt){
  const box=$(id); box.innerHTML="";
  (items||[]).forEach(it=>{ const d=document.createElement("div");
    d.className="child"; d.textContent=fmt(it); box.appendChild(d); });
}
function renderTree(id, nodes, spec){
  const box=$(id); box.innerHTML="";
  (nodes||[]).forEach(n=>{
    const wrap=document.createElement("div"); wrap.className="node";
    const head=document.createElement("div");
    head.innerHTML='<span class="name">▸ '+esc(spec.name(n))+'</span> <span class="meta">'+esc(spec.meta(n))+'</span>';
    const kids=document.createElement("div"); kids.style.display="none";
    spec.children(n).forEach(c=>{ const k=document.createElement("div");
      k.className="child dim"; k.textContent=c; kids.appendChild(k); });
    head.querySelector(".name").onclick=()=>{ kids.style.display = kids.style.display==="none"?"block":"none"; };
    wrap.appendChild(head); wrap.appendChild(kids); box.appendChild(wrap);
  });
}
function esc(s){
  if(s==null) s="";
  else if(typeof s==="object"){ try{ s=JSON.stringify(s); }catch(e){ s=String(s); } }
  else s=String(s);
  return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

const npcSpec = {
  name:n=>n.name||"?",
  meta:n=>"x"+(n.times_talked||0),
  children:n=>{ let out=[];
    (n.transcript||[]).forEach(e=> out.push(e.said?((n.name||"")+": "+e.said):("you: "+(e.me||""))));
    (n.topics_unasked||[]).forEach(t=> out.push("not asked: "+t));
    (n.topics_asked||[]).forEach(t=> out.push("asked: "+t));
    (n.notes||[]).slice(-6).forEach(x=> out.push("note: "+x));
    return out; }
};
const topicSpec = {
  name:t=>t.name||"?",
  meta:t=>((t.notes||t.mentions||[]).length)+" notes",
  children:t=>{ let out=[];
    (t.notes||[]).forEach(n=> out.push("@"+(n.step||"")+": "+(n.note||"")));
    (t.mentions||[]).forEach(m=> out.push((m.npc||"")+": "+(m.said||"")));
    return out; }
};

function apply(key, value){
  switch(key){
    case "turn": setText("turn","turn "+value); break;
    case "context": {
      const pct=value.pct||0; $("ctxfill").style.width=pct+"%";
      $("ctxfill").style.background = pct>90?"#e0574a":pct>75?"#e0b34a":"#5ab0ff";
      setText("ctxlbl", value.label); break; }
    case "turn_log": renderTurnLog(value); break;
    case "stats_kv": renderKV(value); break;
    case "npc_tree": renderTree("p-npc_tree", value, npcSpec); break;
    case "topics_tree": renderTree("p-topics_tree", value, topicSpec); break;
    case "resolved": renderList("p-resolved", value, x => typeof x==="string"?x:(x.name||JSON.stringify(x))); break;
    case "save_ack": { const b=$("savebtn"); b.textContent=value?"Saved!":"Save failed";
      setTimeout(()=>b.textContent="Save game",2500); break; }
    case "answer": setText("answer", value); break;
    default: setText("p-"+key, value);
  }
}
function applySnapshot(s){
  setText("turn","turn "+(s.turn||0));
  apply("context",{pct:s.context_pct||0,label:s.context_label||""});
  ["gstatus","plot","map","area_map","inventory","dialog","quests",
   "tool_stats","context_dump","answer","thinking"].forEach(k=>apply(k,s[k]));
  apply("turn_log", s.turn_log||[]);
  apply("stats_kv", s.stats_kv||{});
  apply("npc_tree", s.npc_tree||[]);
  apply("topics_tree", s.topics_tree||[]);
  apply("resolved", s.resolved||[]);
}

function connect(){
  const es=new EventSource("/events");
  es.addEventListener("snapshot", ev=>{ try{applySnapshot(JSON.parse(ev.data));}catch(e){} });
  es.onmessage = ev => { try{ const m=JSON.parse(ev.data); apply(m.key,m.value);}catch(e){} };
  es.onopen = ()=>{ const c=$("conn"); c.textContent="live"; c.className="ok"; };
  es.onerror = ()=>{ const c=$("conn"); c.textContent="reconnecting…"; c.className="";
    es.close(); setTimeout(connect,2000); };
}

async function post(path,body){ try{
  await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body||{})}); }catch(e){} }

$("askbtn").onclick = ()=>{ const q=$("askbox").value.trim();
  if(q){ post("/ask",{question:q}); $("askbox").value=""; } };
$("askbox").addEventListener("keydown", e=>{ if(e.key==="Enter") $("askbtn").click(); });
$("savebtn").onclick = ()=> post("/save",{});
$("turnwin").onchange = e=> post("/turn-window",{turns:e.target.value});

connect();
</script>
</body>
</html>
"""
