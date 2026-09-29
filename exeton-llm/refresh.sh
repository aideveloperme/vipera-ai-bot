#!/usr/bin/env bash
# Nightly refresh: re-scrape exeton.com and rebuild the index, then restart the API.
# crontab -e →  0 2 * * * /path/to/exeton-llm/refresh.sh >> /var/log/exeton-refresh.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
python scrape_exeton.py
python build_index.py
systemctl --user restart exeton-chat.service 2>/dev/null || echo "Restart chat_server.py to load the new index."
