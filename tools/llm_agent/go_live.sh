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
PORT=8090
FPS=15
SIZE=1024x768

echo "[go_live] repo=$REPO"
rm -f "$FIFO" "$AFIFO"

# 1) Exult (writer): video frames -> $FIFO, mixed PCM -> $AFIFO. SDL still opens
# ALSA (so the audio callback that produces the PCM keeps firing), but we no
# longer CAPTURE from snd-aloop - the PCM is tapped in-engine and written to
# $AFIFO, perfectly paced. This sidesteps the whole loopback-clock problem.
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  ./exult --bg --nomenu --llmagent --newgame \
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
  if python3 "$(dirname "$0")/play.py" act '{"type":"play_music","track":9,"repeat":1}' 2>/dev/null | grep -q '"ok"'; then
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
cleanup() { echo "[go_live] stopping"; kill "${MUX:-0}" "$HTTPD" "$EXULT" 2>/dev/null; exit 0; }
trap cleanup INT TERM

while kill -0 "$EXULT" 2>/dev/null; do
  ffmpeg -hide_banner -loglevel warning \
    -thread_queue_size 1024 \
    -f rawvideo -pixel_format bgr0 -video_size "$SIZE" -framerate "$FPS" -i "$FIFO" \
    -thread_queue_size 16384 \
    -f s16le -ar 48000 -ac 2 -i "$AFIFO" \
    -vf "format=yuv420p" \
    -c:v libx264 -preset ultrafast -pix_fmt yuv420p -g $((FPS*2)) -r "$FPS" \
    -b:v 2500k -maxrate 2500k -bufsize 5000k \
    -c:a aac -b:a 128k -ar 48000 -ac 2 \
    -af "aresample=async=1" \
    -f hls -hls_time 2 -hls_list_size 6 -hls_flags delete_segments+omit_endlist \
    -hls_segment_filename "$HLS_DIR/seg%05d.ts" \
    "$HLS_DIR/stream.m3u8" &
  MUX=$!
  wait "$MUX"
  echo "[go_live] mux exited; restarting in 1s..."
  sleep 1
done
echo "[go_live] exult exited; stopping."
kill "$HTTPD" 2>/dev/null
