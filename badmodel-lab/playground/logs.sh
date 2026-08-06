#!/usr/bin/env bash
# Follow container logs for a tier's stack — useful for watching first-boot model-pull
# progress or diagnosing a stuck container.
#   ./logs.sh 8gb
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

TIER="$(require_tier_arg "${1:-}")"

exec docker compose logs -f ollama searxng "badmodel-${TIER}"
