#!/usr/bin/env bash
# Launch ONE ephemeral, hardened sandbox container that runs a single task and exits.
# The model runs on the HOST ollama; the sandbox holds only the agent loop + tools and
# can read/write/move files and run allow-listed commands ONLY inside its tmpfs workdir.
#
#   ./run_sandbox.sh <model> "<goal>"
#   MODEL_API_URL=http://host.docker.internal:11435/v1 ./run_sandbox.sh gemma2:2b "count the files here"
set -euo pipefail

MODEL="${1:?model tag, e.g. gemma2:2b}"
GOAL="${2:?the task goal, quoted}"
IMAGE="${LOCALAGENT_IMAGE:-localagent-sandbox}"
NET="${LOCALAGENT_NET:-localagent_sandbox}"

# A DEDICATED bridge network, segmented off euglena_enet — the sandbox cannot reach
# Redis/RabbitMQ/Chroma/Ollama-internal by name or subnet. (Egress to the wider internet
# for web fetch stays on; for stricter deployments route it through an allow-listing
# proxy and block 169.254.169.254 — see README.)
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null

# Durable memory rides a named volume (survives the ephemeral container); the task
# scratch workdir is tmpfs (fresh per run, nothing persists to disk).
docker volume inspect localagent_mem >/dev/null 2>&1 || docker volume create localagent_mem >/dev/null

exec docker run --rm \
  --network "$NET" \
  --add-host host.docker.internal:host-gateway \
  --read-only \
  --tmpfs /work:rw,size=64m,mode=1777 \
  --mount type=volume,src=localagent_mem,dst=/mem \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 512m --cpus 1.0 \
  -e MODEL_API_URL="${MODEL_API_URL:-http://host.docker.internal:11435/v1}" \
  -e SEARXNG_URL="${SEARXNG_URL:-http://host.docker.internal:8888}" \
  "$IMAGE" \
    --model "$MODEL" --goal "$GOAL" \
    --workdir /work --memory /mem/mem.jsonl --identity "${IDENTITY:-default}"
