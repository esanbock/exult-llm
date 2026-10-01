#!/usr/bin/env bash
# go_live.sh - Start the full headless A/V stack in the correct order:
#   1) Exult (headless video + ALSA audio, writing raw frames to a FIFO)
#   2) the A/V mux (reads FIFO video + loopback audio -> H.264/AAC -> HTTP MPEG-TS)
#
# Exult MUST start before ffmpeg opens the FIFO (a FIFO read-open blocks until a
# writer exists). This script enforces that ordering and keeps both alive.
#
# Watch in VLC:  http://<this-host>:8090/
set -u
cd "$(dirname "$0")/../.."          # repo root (tools/llm_agent -> repo)
REPO="$(pwd)"
FIFO=/tmp/exult_video.raw
AFIFO=/tmp/exult_audio.pcm
OVERLAY=/tmp/exult_overlay.txt
FONT=/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf
PORT=8090
FPS=10
SIZE=512x384
# Optional Twitch output. Set TWITCH_STREAM_KEY (env or tools/llm_agent/twitch.env,
# gitignored). Leave unset for local-HLS-only (default). Ingest is Twitch's
# recommended RTMPS endpoint; pick a nearer server if you like.
# LOCAL_ONLY=1 skips twitch.env entirely (no broadcast, no chat bridge).
if [ "${LOCAL_ONLY:-0}" = 1 ]; then
  unset TWITCH_STREAM_KEY TWITCH_OAUTH TWITCH_NICK TWITCH_CHANNEL
elif [ -f tools/llm_agent/twitch.env ]; then
  . tools/llm_agent/twitch.env
fi
TWITCH_INGEST="${TWITCH_INGEST:-rtmps://live.twitch.tv/app}"
TWITCH_STREAM_KEY="${TWITCH_STREAM_KEY:-}"

echo "[go_live] repo=$REPO"
rm -f "$FIFO" "$AFIFO"
# Seed the overlay text file so ffmpeg's drawtext has something to read at
# startup (drawtext errors if textfile is missing); the driver rewrites it live.
printf 'Exult LLM agent\nwaiting for the agent...' > "$OVERLAY"

# 1) Exult (writer): video frames -> $FIFO, mixed PCM -> $AFIFO. SDL still opens
# ALSA (so the audio callback that produces the PCM keeps firing), but we no
# longer CAPTURE from snd-aloop - the PCM is tapped in-engine and written to
# $AFIFO, perfectly paced. This sidesteps the whole loopback-clock problem.
# RESUME by default: Exult continues from its gamedat working copy (the driver
# saves into it periodically and on a clean stop). NEWGAME=1 starts the world
# over - and then the agent's memory of the old world is archived too, so a
# fresh world never meets stale memory ("I already took the key").
NEWGAME_ARG=""
if [ "${NEWGAME:-0}" = 1 ]; then
  NEWGAME_ARG="--newgame"
  MEM=tools/llm_agent/agent_memory.json
  if [ -f "$MEM" ]; then
    ARCH="tools/llm_agent/agent_memory.before-newgame.$(date +%s).json"
    mv "$MEM" "$ARCH"
    echo "[go_live] NEW GAME: archived agent memory -> $ARCH"
  fi
else
  echo "[go_live] resuming the saved game (NEWGAME=1 to start over)"
fi
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  ./exult --bg --nomenu --llmagent $NEWGAME_ARG \
  --llmstream "$FIFO" --llmstream-fps "$FPS" \
  --llmaudio "$AFIFO" \
  >/tmp/exult_stream.log 2>&1 &
EXULT=$!
echo "[go_live] exult pid $EXULT; waiting for video+audio FIFOs..."

for i in $(seq 20); do
  sleep 1
  [ -p "$FIFO" ] && [ -p "$AFIFO" ] || continue
  grep -q "raw RGB24 video FIFO" /tmp/exult_stream.log 2>/dev/null || continue
  break
done
if ! kill -0 "$EXULT" 2>/dev/null; then
  echo "[go_live] ERROR: exult exited early; see /tmp/exult_stream.log"; tail -5 /tmp/exult_stream.log; exit 1
fi
echo "[go_live] exult up (video+audio FIFOs ready)."

# Start looping music so the stream always has audio (fresh --newgame doesn't
# auto-start the map theme).
for i in 1 2 3 4 5; do
  if python3 tools/llm_agent/play.py act '{"type":"play_music","track":9,"repeat":1}' 2>/dev/null | grep -q '"ok"'; then
    echo "[go_live] music started (looping track 9)"; break
  fi
  sleep 1
done

# 2) A/V mux -> HLS served by a always-on HTTP server. HLS avoids ffmpeg's
# fragile "-listen 1" single-client model (which broke on VLC reconnects): the
# encoder runs continuously writing segments, and a plain HTTP server serves
# them to any number of clients that can connect/disconnect/reconnect freely.
HLS_DIR=/tmp/exult_hls
rm -rf "$HLS_DIR"; mkdir -p "$HLS_DIR"

# Always-on HTTP server for the HLS directory (serves to the whole LAN).
python3 -m http.server "$PORT" --directory "$HLS_DIR" --bind 0.0.0.0 >/tmp/hls_http.log 2>&1 &
HTTPD=$!

echo "[go_live] HLS -> http://0.0.0.0:$PORT/stream.m3u8"
echo "[go_live] VLC: http://<this-host>:$PORT/stream.m3u8   (Ctrl-C to stop)"

# Twitch chat integration: if chat creds are present, run the bridge so viewers
# can '!ask <question>' and the agent answers on chat + on the stream's top
# banner (we have no OBS). Bridge reads ask_queue.txt / writes agent_answer.txt;
# driver.py consumes/answers. Values come from twitch.env (already sourced).
BRIDGE=""
if [ -n "${TWITCH_OAUTH:-}" ] && [ -n "${TWITCH_NICK:-}" ] && [ -n "${TWITCH_CHANNEL:-}" ]; then
  TWITCH_OAUTH="$TWITCH_OAUTH" TWITCH_NICK="$TWITCH_NICK" TWITCH_CHANNEL="$TWITCH_CHANNEL" \
    python3 tools/llm_agent/twitch_bridge.py >/tmp/twitch_bridge.log 2>&1 &
  BRIDGE=$!
  echo "[go_live] Twitch chat: bridge running (viewers can !ask); joined #$TWITCH_CHANNEL"
else
  echo "[go_live] Twitch chat: OFF (set TWITCH_OAUTH/NICK/CHANNEL in twitch.env)"
fi
cleanup() { echo "[go_live] stopping"; kill "${MUX:-0}" "$HTTPD" "${BRIDGE:-0}" "$EXULT" 2>/dev/null; exit 0; }
trap cleanup INT TERM

# Build the output: always local HLS; also Twitch (FLV/RTMPS) if a key is set.
# Using the 'tee' muxer means we ENCODE ONCE and fan the result to both, so the
# Twitch push adds negligible CPU. Twitch wants H.264 + AAC + ~2s keyframes
# (we have -g = 2*fps) in an FLV container over RTMPS.
HLS_OUT="[f=hls:hls_time=2:hls_list_size=15:hls_flags=delete_segments+omit_endlist:hls_segment_filename=$HLS_DIR/seg%05d.ts]$HLS_DIR/stream.m3u8"
if [ -n "$TWITCH_STREAM_KEY" ]; then
  echo "[go_live] Twitch: LIVE -> $TWITCH_INGEST/<key>"
  TEE_OUT="${HLS_OUT}|[f=flv:onfail=ignore]${TWITCH_INGEST}/${TWITCH_STREAM_KEY}"
else
  echo "[go_live] Twitch: OFF (set TWITCH_STREAM_KEY in twitch.env to go live)"
  TEE_OUT="${HLS_OUT}"
fi

while kill -0 "$EXULT" 2>/dev/null; do
  ffmpeg -hide_banner -loglevel warning \
    -thread_queue_size 1024 \
    -f rawvideo -pixel_format rgb24 -video_size "$SIZE" -framerate "$FPS" -i "$FIFO" \
    -thread_queue_size 4096 \
    -f s16le -ar 48000 -ac 2 -i "$AFIFO" \
    -vf "pad=512:512:0:0:color=black,drawtext=fontfile=$FONT:textfile=$OVERLAY:reload=1:fontcolor=white:fontsize=26:line_spacing=6:x=10:y=392:box=1:boxcolor=black@0.6:boxborderw=8,format=yuv420p" \
    -c:v libx264 -preset ultrafast -threads 4 -pix_fmt yuv420p -g $((FPS*2)) -r "$FPS" \
    -b:v 1000k -maxrate 1000k -bufsize 2000k \
    -c:a aac -b:a 128k -ar 48000 -ac 2 \
    -f tee -map 0:v -map 1:a "$TEE_OUT" &
  MUX=$!
  wait "$MUX"
  echo "[go_live] mux exited; restarting in 1s..."
  sleep 1
done
echo "[go_live] exult exited; stopping."
kill "$HTTPD" 2>/dev/null
