#!/usr/bin/env bash
# Open (or reopen) an interactive chat session with a tier's badmodel.
#   ./chat.sh 8gb
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

TIER="$(require_tier_arg "${1:-}")"
CONTAINER="$(container_name_for_tier "$TIER")"
check_container_running "$CONTAINER" "$TIER"

exec docker exec -it "$CONTAINER" python -m playground.chat_entrypoint
