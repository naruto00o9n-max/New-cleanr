#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p data
case "${1:-status}" in
  start)
    if [ -f data/bot.pid ] && kill -0 "$(cat data/bot.pid)" 2>/dev/null; then
      echo 'Bot is already running'; exit 0
    fi
    nohup .venv/bin/python -u -m manga_cleaner.cli bot >> data/bot.log 2>&1 &
    echo $! > data/bot.pid
    echo "Started bot PID $(cat data/bot.pid)"
    ;;
  stop)
    if [ -f data/bot.pid ]; then kill -TERM "$(cat data/bot.pid)" 2>/dev/null || true; rm -f data/bot.pid; fi
    ;;
  status)
    if [ -f data/bot.pid ] && kill -0 "$(cat data/bot.pid)" 2>/dev/null; then echo 'Bot process is running'; else echo 'Bot is stopped'; fi
    ;;
  *) echo 'Usage: scripts/bot.sh start|stop|status'; exit 2 ;;
esac
