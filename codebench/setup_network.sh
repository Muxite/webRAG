#!/usr/bin/env bash
# Idempotent one-time setup: create the codebench sandbox network and give it its two
# intentional holes — the existing badmodel-lab/playground SearXNG container, and an
# allowlist proxy in front of the LLM (never badmodel-ollama directly — see below). Run
# this once before any run_agent_sandbox.sh invocation.
#
# codebench_sandbox is a FOURTH network, distinct from euglena_enet / localagent_sandbox /
# badmodel_playground_net, so this never interferes with those running systems. It's
# created --internal: Docker drops the outbound MASQUERADE rule for that bridge, so
# containers on it have no path to the real internet at all — that's the actual mechanism
# blocking Aider's update-check pings, not just a CLI flag we're trusting it to honor.
#
# SECURITY (fixed after an adversarial review found this live-exploitable): an EARLIER
# version of this script joined the real badmodel-ollama container to codebench_sandbox
# directly. Ollama's HTTP API has no auth, so that gave the sandbox unrestricted access to
# its ENTIRE admin surface over the same TCP connection — verified live: GET /api/tags
# enumerated the shared instance's full model roster, DELETE /api/delete actually
# processed deletion requests (404 for an unknown name, not 401/403 — no auth at all), and
# POST /api/pull made Ollama's own internet-connected registry client fetch whatever tag
# was requested. A malicious or merely confused agent could DoS or corrupt this SHARED
# instance, which other sessions/work depend on. codebench-ollama-proxy (see
# ollama_proxy/) fixes this: it forwards ONLY POST /v1/chat/completions and returns 403
# for everything else, and is the ONLY thing joined to codebench_sandbox — badmodel-ollama
# itself never is.
set -euo pipefail
CB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CB_DIR/.." && pwd)"
LAB_DIR="$REPO_ROOT/badmodel-lab"
NET="${CODEBENCH_NET:-codebench_sandbox}"
SEARXNG_CTR="${CODEBENCH_SEARXNG_CTR:-badmodel-playground-searxng}"
SEARXNG_PROFILE="${CODEBENCH_SEARXNG_PROFILE:-4gb}"   # any playground profile works — searxng has no depends_on
OLLAMA_NET="${CODEBENCH_OLLAMA_NET:-euglena_enet}"     # the network badmodel-ollama is actually on
PROXY_CTR="${CODEBENCH_PROXY_CTR:-codebench-ollama-proxy}"
PROXY_IMAGE="${CODEBENCH_PROXY_IMAGE:-codebench-ollama-proxy:latest}"

if docker network inspect "$NET" >/dev/null 2>&1; then
  echo "network $NET already exists"
else
  docker network create --internal "$NET"
  echo "created --internal network $NET"
fi

if ! docker container inspect "$SEARXNG_CTR" >/dev/null 2>&1; then
  echo "starting $SEARXNG_CTR via playground compose (profile=$SEARXNG_PROFILE, searxng only — no ollama/agent pulled in)"
  docker compose -f "$LAB_DIR/playground/docker-compose.yml" --profile "$SEARXNG_PROFILE" up -d searxng
fi

if ! docker image inspect "$PROXY_IMAGE" >/dev/null 2>&1; then
  echo "building $PROXY_IMAGE"
  docker build -f "$CB_DIR/ollama_proxy/Dockerfile" -t "$PROXY_IMAGE" "$CB_DIR/ollama_proxy"
fi

if ! docker container inspect "$PROXY_CTR" >/dev/null 2>&1; then
  echo "starting $PROXY_CTR on $OLLAMA_NET (reaches badmodel-ollama by name; not on $NET yet)"
  docker run -d --name "$PROXY_CTR" \
    --network "$OLLAMA_NET" \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,mode=1777 \
    --cap-drop=ALL --security-opt no-new-privileges --pids-limit 32 \
    --memory 64m --cpus 0.5 \
    --restart unless-stopped \
    "$PROXY_IMAGE"
fi

for ctr in "$SEARXNG_CTR" "$PROXY_CTR"; do
  if docker network inspect "$NET" -f '{{range .Containers}}{{.Name}} {{end}}' | grep -qw "$ctr"; then
    echo "$ctr already joined to $NET"
  else
    docker network connect "$NET" "$ctr"
    echo "joined $ctr to $NET"
  fi
done

# Defensive: if an older setup left badmodel-ollama itself joined to codebench_sandbox
# (the vulnerable topology this script used to create), disconnect it every run.
if docker network inspect "$NET" -f '{{range .Containers}}{{.Name}} {{end}}' | grep -qw "badmodel-ollama"; then
  echo "removing stale direct badmodel-ollama <-> $NET join (superseded by $PROXY_CTR)"
  docker network disconnect "$NET" badmodel-ollama
fi

echo "setup_network.sh: OK"
