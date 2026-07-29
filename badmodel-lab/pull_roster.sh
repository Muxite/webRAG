#!/usr/bin/env bash
# Pull the subject roster into the dedicated badmodel-ollama container.
# Anchors that live on OpenRouter (tags with a "/") are skipped — no pull needed.
set -euo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CTR="${BADMODEL_OLLAMA_CTR:-badmodel-ollama}"
docker inspect "$CTR" >/dev/null 2>&1 || {
  echo "container '$CTR' not up. Run: docker compose -f $LAB_DIR/docker-compose.yml up -d" >&2
  exit 1
}

# subject + local-anchor tags from roster.yaml (crude YAML read: 'tag:' lines without a '/')
mapfile -t TAGS < <(grep -E '^\s*-\s*tag:' "$LAB_DIR/roster.yaml" \
  | sed -E 's/.*tag:\s*//' | awk '{print $1}' | grep -v '/')

for t in "${TAGS[@]}"; do
  echo ">> ollama pull $t"
  docker exec "$CTR" ollama pull "$t" || echo "!! pull failed for $t (bad tag?) — skipping" >&2
done
echo ">> resident models:"
docker exec "$CTR" ollama list
