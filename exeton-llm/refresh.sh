#!/usr/bin/env bash
# Re-scrape exeton.com, rebuild the index, restart the chatbot so it loads the new data.
# Scheduled by cron (see README "Keep it running"), e.g. every night at 02:00:
#   0 2 * * * $HOME/vipera-ai-bot/exeton-llm/refresh.sh >> $HOME/exeton-refresh.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=== refresh started $(date) ==="
$PY scrape_exeton.py            # exits without touching data if the scrape looks broken
$PY build_index.py
systemctl --user restart exeton-chat.service
echo "=== refresh done $(date) ==="
