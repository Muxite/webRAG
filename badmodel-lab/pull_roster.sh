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

# subject tags ONLY from roster.yaml's `subjects:` block (crude YAML read: 'tag:' lines without
# a '/'). The `anchors:` block is deliberately excluded here -- its local entry (qwen2.5:7b) is
# served from the SEPARATE yappers-ollama container (:11434, per run_matrix.sh/run_format.sh's
# BADMODEL_OLLAMA_URL override for anchor cells), never from badmodel-ollama (:11435) -- pulling
# it here wastes bandwidth/disk on a multi-GB model this container will never actually serve.
mapfile -t TAGS < <(sed -n '/^subjects:/,/^anchors:/p' "$LAB_DIR/roster.yaml" \
  | grep -E '^\s*-\s*tag:' \
  | sed -E 's/.*tag:\s*//' | awk '{print $1}' | grep -v '/')

for t in "${TAGS[@]}"; do
  echo ">> ollama pull $t"
  docker exec "$CTR" ollama pull "$t" || echo "!! pull failed for $t (bad tag?) — skipping" >&2
done
echo ">> resident models:"
docker exec "$CTR" ollama list
