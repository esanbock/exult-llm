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
#
# Env: THROTTLE_MS (default 100), NUM_CTX (24576), DELAY (0.5).
# Logs: $RUN/{game,driver,relay}.log; driver turn log: tools/llm_agent/driver_log.txt
set -u
cd "$(dirname "$0")"
RUN=/tmp/exult_llm
mkdir -p "$RUN"

TWITCH=0 NEWGAME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --twitch)   TWITCH=1 ;;
    --new-game) NEWGAME=1 ;;
    --)         shift; break ;;
    *)          echo "unknown option: $1 (driver args go after --)"; exit 2 ;;
  esac
  shift
done

running() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
launch() {   # name, command... -> detached session leader; its pid = its pgid
  local name=$1; shift
  setsid "$@" > "$RUN/$name.log" 2>&1 < /dev/null &
  echo $! > "$RUN/$name.pid"
}

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

# 2) Agent driver (model/host come from agent.env unless overridden after --).
launch driver python3 -u driver.py --no-launch --inspector --no-guards \
  --throttle-ms "${THROTTLE_MS:-100}" --num-ctx "${NUM_CTX:-24576}" \
  --delay "${DELAY:-0.5}" --steps 1000000 --log-file driver_log.txt "$@"
echo "[start] driver started (turn log tools/llm_agent/driver_log.txt)"

# 3) Optional Twitch relay (+ chat bridge).
if [ "$TWITCH" = 1 ]; then
  launch relay ./twitch_relay.sh
  echo "[start] Twitch relay started (log $RUN/relay.log)"
fi

echo "[start] local stream: http://<this-host>:8090/stream.m3u8"
echo "[start] inspector:    http://<this-host>:8092/"
