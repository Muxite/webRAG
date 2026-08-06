#!/usr/bin/env bash
# Open a second, independent shell into a running tier's container — usable alongside an
# active chat.sh session in another terminal (Docker supports multiple concurrent `exec`
# sessions into one container). Handy for tailing logs or poking at files mid-chat.
#   ./shell.sh 8gb
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

TIER="$(require_tier_arg "${1:-}")"
CONTAINER="$(container_name_for_tier "$TIER")"
check_container_running "$CONTAINER" "$TIER"

exec docker exec -it "$CONTAINER" bash
