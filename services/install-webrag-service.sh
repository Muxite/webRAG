#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DIR="$(pwd)"
if [[ ! -f webrag.service ]]; then
  echo "webrag.service not found in $DIR" >&2
  exit 1
fi
tmp="$(mktemp)"
while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" == WorkingDirectory=* ]]; then
    echo "WorkingDirectory=$DIR"
  else
    echo "$line"
  fi
done < webrag.service >"$tmp"
sudo cp "$tmp" /etc/systemd/system/webrag.service
rm -f "$tmp"
sudo systemctl daemon-reload
sudo systemctl enable webrag.service
echo "Enabled webrag.service for boot. Start now: sudo systemctl start webrag.service"
