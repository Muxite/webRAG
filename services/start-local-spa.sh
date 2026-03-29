#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f ../frontend/dist/index.html ]]; then
  echo "No frontend/dist. Run from repo root: python scripts/deploy_local_stack.py build-frontend" >&2
  exit 1
fi
docker compose -f docker-compose.yml -f docker-compose.local.yml --profile local-spa up -d --build --scale agent=3
sleep 5
tailscale funnel --bg --yes 80
