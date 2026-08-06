#!/usr/bin/env bash
# Shared helpers for chat.sh / shell.sh / logs.sh. Sourced, not run directly.

resolve_tier() {
  case "$1" in
    4|4gb|badmodel-4gb) echo "4gb" ;;
    6|6gb|badmodel-6gb) echo "6gb" ;;
    8|8gb|badmodel-8gb) echo "8gb" ;;
    12|12gb|badmodel-12gb) echo "12gb" ;;
    *) echo "" ;;
  esac
}

container_name_for_tier() {
  echo "badmodel-playground-$1"
}

require_tier_arg() {
  if [[ -z "${1:-}" ]]; then
    echo "Usage: $(basename "$0") <tier>   (tier: 4gb, 6gb, 8gb, or 12gb)" >&2
    exit 1
  fi
  local tier
  tier="$(resolve_tier "$1")"
  if [[ -z "$tier" ]]; then
    echo "Unknown tier '$1'. Expected one of: 4gb, 6gb, 8gb, 12gb." >&2
    exit 1
  fi
  echo "$tier"
}

check_container_running() {
  local container="$1" tier="$2" running
  running="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || echo false)"
  if [[ "$running" != "true" ]]; then
    echo "Container '$container' isn't running." >&2
    echo "Bring it up first:  docker compose up badmodel-${tier} --build -d" >&2
    exit 1
  fi
}
