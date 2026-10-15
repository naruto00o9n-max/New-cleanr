#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data
running() {
  [ -f data/bot.pid ] || return 1
  pid=$(cat data/bot.pid)
  kill -0 "$pid" 2>/dev/null || return 1
  case "$(ps -o stat= -p "$pid")" in Z*|'') return 1 ;; esac
}
case "${1:-status}" in
  start)
    if running; then
      echo 'Bot is already running'; exit 0
    fi
    nohup setsid sh -c 'echo $$ > data/bot.pid; exec .venv/bin/python -u -m manga_cleaner.cli bot' </dev/null >> data/bot.log 2>&1 &
    echo 'Bot launch requested; see data/bot.log for connection status'
    ;;
  stop)
    if [ -f data/bot.pid ]; then kill -TERM "$(cat data/bot.pid)" 2>/dev/null || true; rm -f data/bot.pid; fi
    ;;
  status)
    if running; then echo 'Bot process is running'; else echo 'Bot is stopped'; fi
    ;;
  *) echo 'Usage: scripts/bot.sh start|stop|status'; exit 2 ;;
esac
