#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build --scale agent=3
sleep 5
tailscale funnel --bg --yes 18080
