#!/usr/bin/env bash
# twitch_relay.sh - Push the running local HLS stream (from go_live.sh) to
# Twitch, plus the chat bridge, WITHOUT re-encoding or restarting the game.
# Lets you go live / go dark while a run continues. Credentials come from the
# gitignored twitch.env (TWITCH_STREAM_KEY, TWITCH_OAUTH/NICK/CHANNEL); the
# stream key is masked in any output.
#
# Normally started/stopped by start_all.sh --twitch / stop_all.sh.
set -u
cd "$(dirname "$0")"
[ -f twitch.env ] || { echo "[relay] no twitch.env - nothing to do"; exit 1; }
. ./twitch.env
: "${TWITCH_STREAM_KEY:?TWITCH_STREAM_KEY missing from twitch.env}"
INGEST="${TWITCH_INGEST:-rtmps://live.twitch.tv/app}"
HLS="${HLS_URL:-http://127.0.0.1:8090/stream.m3u8}"

if [ -n "${TWITCH_OAUTH:-}" ] && [ -n "${TWITCH_NICK:-}" ] && [ -n "${TWITCH_CHANNEL:-}" ]; then
  TWITCH_OAUTH="$TWITCH_OAUTH" TWITCH_NICK="$TWITCH_NICK" TWITCH_CHANNEL="$TWITCH_CHANNEL" \
    python3 twitch_bridge.py >/tmp/twitch_bridge.log 2>&1 &
  echo "[relay] chat bridge started"
fi

while true; do
  # -c copy: no re-encode; the AAC-in-TS audio needs the ADTS->ASC filter for FLV.
  ffmpeg -hide_banner -loglevel warning -live_start_index -1 \
    -i "$HLS" -c copy -bsf:a aac_adtstoasc \
    -f flv "$INGEST/$TWITCH_STREAM_KEY" 2>&1 | sed "s#$TWITCH_STREAM_KEY#<key>#g"
  echo "[relay] ffmpeg exited; restarting in 2s"
  sleep 2
done
