#!/usr/bin/env bash
# start_all.sh - Start a full run: game + local stream (go_live.sh), the agent
# driver, and optionally the Twitch relay. Each piece runs detached in its own
# process group (setsid) with a PID file in $RUN, so it survives the shell that
# started it and stop_all.sh can stop it reliably by group - no name matching.
#
#   ./start_all.sh                 resume the saved game, local stream only
#   ./start_all.sh --twitch        ...and go live on Twitch (twitch.env)
#   ./start_all.sh --new-game      start the world over (archives agent memory)
#   ./start_all.sh -- --model qwen3.6:latest   extra args go to driver.py
#   ./start_all.sh --driver-only -- --model X  restart just the agent
#
# Env: THROTTLE_MS (default 100), NUM_CTX (24576), DELAY (0.5).
# Logs: $RUN/{game,driver,relay}.log; driver turn log: tools/llm_agent/driver_log.txt
set -u
cd "$(dirname "$0")"
RUN=/tmp/exult_llm
mkdir -p "$RUN"

TWITCH=0 NEWGAME=0 DRIVER_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --twitch)   TWITCH=1 ;;
    --new-game) NEWGAME=1 ;;
    --driver-only) DRIVER_ONLY=1 ;;
    --)         shift; break ;;
    *)          echo "unknown option: $1 (driver args go after --)"; exit 2 ;;
  esac
  shift
done

running() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
launch() {   # name, command... -> detached session leader; its pid = its pgid
  local name=$1; shift
  # The new session leader records its OWN pid. ($! is wrong when setsid has
  # to fork - e.g. when called from a shell with job control - and a stale
  # pid file left the old driver running next to the new one.)
  setsid bash -c 'echo $$ > "$0"; exec "$@"' "$RUN/$name.pid" "$@" \
    > "$RUN/$name.log" 2>&1 < /dev/null &
  for i in $(seq 20); do [ -s "$RUN/$name.pid" ] && break; sleep 0.1; done
}

start_driver() {   # model/host come from agent.env unless overridden after --
  rm -f "$RUN/driver.pid"
  launch driver python3 -u driver.py --no-launch --inspector --no-guards \
    --throttle-ms "${THROTTLE_MS:-100}" --num-ctx "${NUM_CTX:-24576}" \
    --delay "${DELAY:-0.5}" --steps 1000000 --log-file driver_log.txt "$@"
  echo "[start] driver started, pid $(cat "$RUN/driver.pid") (turn log tools/llm_agent/driver_log.txt)"
}

if [ "$DRIVER_ONLY" = 1 ]; then
  # Restart just the agent (after a driver.py change); game, stream, memory
  # and Twitch keep going. The old driver saves on its way out.
  running game || { echo "[start] game isn't running - use ./start_all.sh"; exit 1; }
  ./stop_all.sh driver
  [ -f driver_log.txt ] && mv driver_log.txt "$RUN/driver_log.$(date +%s).txt"
  start_driver "$@"
  exit 0
fi

for p in game driver relay; do
  if running "$p"; then echo "[start] $p already running - run ./stop_all.sh first"; exit 1; fi
done

# 1) Game + local HLS. Wait until the stream is being served.
LOCAL_ONLY=1 NEWGAME=$NEWGAME launch game ./go_live.sh
echo "[start] game starting (log $RUN/game.log)..."
for i in $(seq 60); do
  grep -q "HLS ->" "$RUN/game.log" 2>/dev/null && break
  running game || { echo "[start] game exited:"; tail -5 "$RUN/game.log"; exit 1; }
  sleep 1
done
grep -q "HLS ->" "$RUN/game.log" || { echo "[start] game not ready after 60s"; exit 1; }
grep -E "NEW GAME|resuming" "$RUN/game.log" | sed 's/^/[start] /'

# 2) Agent driver.
start_driver "$@"

# 3) Optional Twitch relay (+ chat bridge).
if [ "$TWITCH" = 1 ]; then
  launch relay ./twitch_relay.sh
  echo "[start] Twitch relay started (log $RUN/relay.log)"
fi

echo "[start] local stream: http://<this-host>:8090/stream.m3u8"
echo "[start] inspector:    http://<this-host>:8092/"
