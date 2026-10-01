#!/usr/bin/env bash
# stop_all.sh - Stop a run started by start_all.sh, in the order that keeps
# progress: the driver first (its SIGTERM handler saves the game + memory while
# the game is still up), then the Twitch relay, then the game and stream.
# Each piece is stopped by its process group (PID files in $RUN). Exult ignores
# SIGTERM, so anything still alive after a grace period gets SIGKILL.
#
#   ./stop_all.sh            stop everything
#   ./stop_all.sh relay      stop only the Twitch relay (go dark, keep playing)
set -u
cd "$(dirname "$0")"
RUN=/tmp/exult_llm

stop() {   # name, grace-seconds
  local name=$1 grace=$2 f="$RUN/$1.pid"
  [ -f "$f" ] || return 0
  local pg; pg=$(cat "$f")
  if kill -0 "$pg" 2>/dev/null; then
    echo "[stop] $name (group $pg)"
    kill -TERM -- "-$pg" 2>/dev/null
    for i in $(seq $((grace * 2))); do
      kill -0 "$pg" 2>/dev/null || break
      sleep 0.5
    done
  fi
  # Leftovers in the group (exult ignores TERM; ffmpeg children).
  kill -KILL -- "-$pg" 2>/dev/null
  rm -f "$f"
}

case "${1:-all}" in
  relay) stop relay 5 ;;
  all)
    stop driver 60     # final game + memory save happens here
    stop relay 5
    stop game 5
    ;;
  *) echo "usage: $0 [all|relay]"; exit 2 ;;
esac
echo "[stop] done"
