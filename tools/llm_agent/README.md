# Exult LLM Agent

Let a local LLM (via [Ollama](https://ollama.com)) play *Ultima VII: The Black
Gate* through a headless Exult build, streamed to Twitch (and locally) with the
model's reasoning shown on-screen and a live web inspector. No X server needed.

This directory contains the agent driver, the game bridge client, the A/V
streaming stack, the web inspector, and the Twitch chat bridge.

---

## Architecture at a glance

```
                         ┌─────────────────────────────────────────┐
   Ollama (LLM) ◄────────┤ driver.py  (the agent loop)             │
   qwen/granite/…        │  observe → think → act, one turn/step    │
                         │  + guards, memory (knowledge.py)         │
                         └───┬───────────────────────────┬─────────┘
                             │ TCP bridge (:45999)        │ HTTP (:8092)
                    ┌────────▼─────────┐         ┌────────▼──────────┐
                    │ exult --llmagent │         │ inspector_server  │
                    │ (headless game)  │         │ (web GUI + REST)  │
                    │  ├ video → FIFO  │         └───────────────────┘
                    │  └ audio → FIFO  │
                    └────────┬─────────┘
                             │ raw RGB24 + PCM FIFOs
                    ┌────────▼─────────┐   HLS :8090 → VLC / browser
                    │ ffmpeg (go_live) ├───┤
                    │  H.264/AAC       │   RTMPS → Twitch
                    └──────────────────┘
                             ▲
                    twitch_bridge.py  (viewers `!ask` → agent answers in chat)
```

---

## Prerequisites

- **Build Exult with the LLM agent enabled** (from repo root):
  ```bash
  ./autogen.sh                 # if configure doesn't exist yet
  ./configure --enable-llm-agent
  make -j$(nproc)
  ```
  This adds the `--llmagent`, `--llmstream`, and `--llmaudio` runtime options and
  compiles `llm/*.cc`. All LLM/stream code is guarded by `USE_LLM_AGENT`, so a
  normal build without the flag is unaffected.

- **Ollama** running and reachable, with a model pulled, e.g.:
  ```bash
  ollama pull granite4.2:30b
  ```
- **ffmpeg**, **python3** (stdlib only — no pip deps required for the core loop;
  `PIL`/Pillow optional, used only to pixel-fit the on-screen overlay text).
- Game data: a legally-owned Black Gate `STATIC` extracted where `~/.exult.cfg`
  points its `blackgate` path.

---

## Configuration

### `agent.env` (committed defaults; safe, no secrets)
```
AGENT_MODEL=granite4.2:30b
AGENT_OLLAMA_HOST=http://alien1.esanbock.com:11434
AGENT_BRIDGE_HOST=127.0.0.1
AGENT_BRIDGE_PORT=45999
```

### `twitch.env` (gitignored — holds secrets; create locally)
```
TWITCH_STREAM_KEY=live_xxxxxxxxxxxxxxxxx     # required to go live on Twitch
TWITCH_OAUTH=oauth:xxxxxxxxxxxxxxxxxxxxxxxx  # chat bridge (viewers can !ask)
TWITCH_NICK=your_bot_nick
TWITCH_CHANNEL=your_channel
TWITCH_INGEST=rtmps://live.twitch.tv/app     # optional; a nearer ingest is fine
```
Leave `twitch.env` absent to run **local-HLS-only** (no Twitch push, no chat).

---

## Quick start

Two commands, in order:

```bash
cd tools/llm_agent

# 1) Start the game + A/V stack (headless Exult, ffmpeg → HLS + Twitch, chat bridge)
./go_live.sh

# 2) Start the agent (connects to the running game; opens the web inspector)
python3 -u driver.py --no-launch --inspector \
    --model granite4.2:30b --num-ctx 24576 \
    --steps 5000 --delay 0.5 --music --music-track 9
```

Then:
- **Web inspector:** `http://<this-host>:8092/` (works in any browser, incl. Windows)
- **Local video (VLC):** `http://<this-host>:8090/stream.m3u8`
- **Twitch:** live if `TWITCH_STREAM_KEY` is set.

> **num-ctx note:** the system prompt is large and memory grows over a run.
> Granite (and other strict models) return HTTP 400 if the prompt exceeds the
> context window, so run with **`--num-ctx 24576`**. (qwen-family silently
> truncates instead, but 24576 is the safe value for all.)

---

## The tools

### `go_live.sh` — the A/V + game stack launcher
Starts headless Exult (writing raw video/audio to FIFOs), then ffmpeg to encode
one H.264/AAC stream fanned (via `tee`) to **local HLS** (served on `:8090`) and
**Twitch RTMPS**, plus the Twitch chat bridge if creds are present. Restarts the
mux if it drops; cleans up all children on Ctrl-C.

Tunable variables near the top of the script:

| Var | Default | Meaning |
|---|---|---|
| `FPS` | `10` | stream framerate |
| `SIZE` | `512x384` | raw capture size from the engine |
| `PORT` | `8090` | local HLS HTTP port |
| `TWITCH_INGEST` | `rtmps://live.twitch.tv/app` | Twitch ingest endpoint |
| `FONT` | LiberationSans | overlay font for the reasoning text |

The output frame is the game area padded to **512×512** with a bottom strip that
shows the agent's live **reasoning** (and raw thinking if the model emits it),
at ~1000 kbps. (Chosen to not waste bits upscaling U7's ~320×200 native art.)

### `driver.py` — the agent loop (main program)
Runs the observe→think→act loop: pulls game state over the bridge, builds the
prompt, asks Ollama, parses the action, applies safety guards, executes it, and
maintains persistent memory. Also hosts the web inspector.

**Key command-line options:**

| Flag | Default | Description |
|---|---|---|
| `--model <name>` | `$AGENT_MODEL` | Ollama model (e.g. `granite4.2:30b`) |
| `--num-ctx <n>` | `8192` | context window; **use `24576`** for Granite |
| `--steps <n>` | `50` | max turns to run |
| `--delay <sec>` | `1.5` | pause between turns |
| `--no-launch` | off | connect to an already-running Exult (use with `go_live.sh`) |
| `--inspector` | off | serve the web GUI over HTTP |
| `--inspector-port <n>` | `8092` | inspector port |
| `--inspector-host <h>` | `0.0.0.0` | inspector bind address (LAN-reachable) |
| `--think` | off | allow the model's chain-of-thought (needed for gpt-oss; off = faster/terser) |
| `--music` / `--no-music` | on | loop background music on connect |
| `--music-track <n>` | `9` | which track to loop |
| `--memory-file <path>` | `agent_memory.json` | persistent agent memory (quests, NPCs, notes, places) |
| `--save-interval <sec>` | `1800` | in-game save cadence (30 min) |
| `--num-ctx` | `8192` | (see note above) |
| `--overlay-file <path>` | `/tmp/exult_overlay.txt` | file ffmpeg's drawtext reads for the on-screen reasoning |
| `--ollama-host <url>` | `$AGENT_OLLAMA_HOST` | Ollama endpoint |
| `--host` / `--port` | bridge host/port | Exult bridge (`127.0.0.1:45999`) |
| `--show-thoughts` | off | open the local tkinter thinking window (vs `--inspector`) |
| `--dry-run` | off | scripted moves, no Ollama (plumbing test) |
| `--raw-log` | off | append every prompt+reply to `raw_comms.log` |

### `inspector_server.py` — web GUI (served by `--inspector`)
A zero-install, single-page web dashboard (open in any browser). Shows, in
priority order: the current **action · reasoning · thinking · outcome** (errors
in red), a **live activity feed**, the **plot**, an **"earlier this session"**
digest of scrolled-off turns, **open quests** (with sub-tasks nested),
**notes/journal**, **stats**, **guard-firing counts**, **map**, and (collapsed)
NPCs/topics/inventory/tool-stats/full-prompt. Header shows turn, location, HP,
position, a graphical **context gauge**, plus live controls.

Live controls / REST endpoints (bound to `0.0.0.0:<inspector-port>`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | the HTML dashboard |
| `/state` | GET | full JSON snapshot |
| `/events` | GET | SSE stream of live updates |
| `/ask` | POST | `{"question": "..."}` — ask the agent (answered out-of-band) |
| `/save` | POST | request an in-game save |
| `/turn-window` | POST | `{"turns": N|"default"}` — action-log memory window (default 450) |
| `/throttle` | POST | `{"ms": 0-5000}` — extra delay **before each LLM request** to cool the GPU |

The **throttle** is the cost/heat lever: a Granite turn is ~1.2 s of compute, so
set 2000–4000 ms to meaningfully lower GPU duty cycle. Default 0 (full speed).

### `play.py` — manual bridge client (hands-on testing)
Send single actions/observations to a running game without the LLM:
```bash
python3 play.py observe            # pretty observation JSON
python3 play.py observe --compact  # observation minus the big grid
python3 play.py act '{"type":"open","name":"chest"}'
python3 play.py talk Petre
python3 play.py raw '{"type":"inventory"}'   # any raw bridge command
```

### `twitch_bridge.py` — chat integration
Launched automatically by `go_live.sh` when chat creds are in `twitch.env`.
Viewers type `!ask <question>` in Twitch chat; the driver answers in chat
(via `ask_queue.txt` / `agent_answer.txt`). No on-video overlay for answers
(Twitch has its own chat UI).

### `stream.py` — standalone streamer (alternative to `go_live.sh`)
Screenshot-based streamer to Twitch or a file/HTTP. Flags: `--out`, `--serve`,
`--serve-host/-port`, `--fps`, `--width/--height`, `--vbitrate`, `--from-file`,
`--seconds`. Mostly superseded by the in-engine capture in `go_live.sh`.

### `monitor.py` — non-intrusive progress monitor
Samples the run and prints a periodic progress summary (does not touch the game
socket, so it can't disturb the agent).

### Supporting modules (imported, not run directly)
- **`knowledge.py`** — the agent's persistent memory: quests (+ sub-tasks),
  per-NPC notes & dialogue trees, topics, known places, fog-of-war visited
  cells, the action log (with wander-run consolidation + a factual digest), and
  tool/guard stats.
- **`ollama_client.py`** — minimal Ollama HTTP client (stdlib only). Handles
  per-model quirks (`think=false` for terse JSON except gpt-oss) and surfaces
  HTTP 400s (e.g. context overflow) as real errors.
- **`exult_client.py`** — thin TCP client for the Exult bridge (`:45999`).
- **`thoughts_window.py`** — the original local tkinter inspector (`--show-thoughts`);
  the web `inspector_server.py` is the network-capable replacement.
- **`avstream.sh`** — older single-client VLC mux (superseded by `go_live.sh`'s
  HLS approach).

---

## Agent actions (what the LLM can do)

Movement: `move` (dir or tx,ty), `goto` (by name or tile; z-aware), `descend`,
`stop`. Investigation: `open` (door/container/body by name or location — reveals
contents), `take` / `pickup` (specific item), `loot` (take-all from a
container), `use` (double-click a world object **or a carried item** — this is
how you *eat*: `use` a food item), `read`, `unlock`/`use_key`. Combat: `attack`,
`combat`, `set_combat_mode`, `heal`. Conversation: `talk`, `answer`, `continue`,
`dismiss`. Party/inventory: `inventory` (yours + each party member's), `give`
(to an NPC **or a party member** — transfer/feed a companion), `equip`,
`unequip`, `drop`. Journal (persist across turns): `add_quest`, `update_quest`,
`note_npc`, `add_topic`. Misc: `wait`, `wait_until`, `feed` (emergency refill
only), `play_music`, `save`, `stats`, `screenshot`.

Note: there is **no generic `search`** — investigation is deliberate: `open` a
specific thing, then `take`/`loot`.

---

## Safety guards

The driver wraps the model with automatic guards that catch loops and invalid
actions (conversation coercion, wedge/oscillation escape, arrival detection,
narrate-but-wait, reach-the-item walking, elevation descend, action aliasing,
etc.). Their firing counts are shown in the inspector's **Guard firings** panel
— treat a high count as a signal of where the model is struggling, not as
"working well."

---

## Shutting down

```bash
# stop the agent first (it saves memory on exit), then the stack
pkill -TERM -f '[d]river.py'
pkill -TERM -f '[g]o_live.sh'
pkill -x exult ; pkill -x ffmpeg
```
Twitch takes ~30–90 s to detect the disconnect and mark the channel offline.

---

## Troubleshooting

- **`HTTP Error 400 … exceeds context size`** — raise `--num-ctx` (use `24576`).
- **`http://enoch:8092` not responding** — the inspector only runs when the
  driver is launched with `--inspector`; check `driver.py` is running.
- **Stream stuck on "waiting for the agent…"** — the driver isn't writing the
  overlay (it errored before its turn completed); check the driver log.
- **Avatar starving** — the agent must `use` a food item from its pack; there is
  intentionally no auto-feed (the model should manage this itself).
