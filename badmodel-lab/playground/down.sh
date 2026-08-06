#!/usr/bin/env bash
# Bring a tier's stack down cleanly (weights + memory persist — no -v).
#   ./down.sh 8gb
#
# A bare `docker compose down` silently does NOTHING here: every service in
# docker-compose.yml is profile-gated (see the file's header comment), and Compose
# computes an empty "active" service set when no profile is selected — confirmed via a
# live test, not assumed. `--profile <tier>` makes `down` see the right services, mirroring
# how `up badmodel-<tier>` already selects them by name.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

TIER="$(require_tier_arg "${1:-}")"

exec docker compose --profile "$TIER" down
