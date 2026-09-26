#!/usr/bin/env bash
# avstream.sh - Mux Exult's raw video FIFO + ALSA loopback audio into a live
# H.264/AAC MPEG-TS stream served over HTTP for VLC (and later Twitch).
#
# ffmpeg's HTTP "-listen 1" serves exactly one client then exits; this wrapper
# relaunches it so you can connect/disconnect/reconnect in VLC freely.
#
# Usage: ./avstream.sh [FIFO] [WxH] [PORT] [FPS]
set -u
FIFO="${1:-/tmp/exult_video.raw}"
SIZE="${2:-1024x768}"
PORT="${3:-8090}"
FPS="${4:-15}"
ALSA_CAP="${ALSA_CAP:-plughw:2,1}"   # loopback capture side

echo "[avstream] video=$FIFO ($SIZE @ ${FPS}fps)  audio=$ALSA_CAP  ->  http://0.0.0.0:$PORT/"
echo "[avstream] open in VLC:  http://<this-host>:$PORT/   (Ctrl-C to stop)"

trap 'echo "[avstream] stopping"; kill "${FF_PID:-0}" 2>/dev/null; exit 0' INT TERM

while true; do
  ffmpeg -hide_banner -loglevel warning \
    -f rawvideo -pixel_format rgba -video_size "$SIZE" -framerate "$FPS" -i "$FIFO" \
    -f alsa -i "$ALSA_CAP" \
    -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p -g $((FPS*2)) -r "$FPS" \
    -b:v 2500k -maxrate 2500k -bufsize 5000k \
    -c:a aac -b:a 128k -ar 44100 -ac 2 \
    -f mpegts -listen 1 "http://0.0.0.0:$PORT/" &
  FF_PID=$!
  wait "$FF_PID"
  echo "[avstream] viewer disconnected; re-listening in 1s..."
  sleep 1
done
