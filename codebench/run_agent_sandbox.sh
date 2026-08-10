#!/usr/bin/env bash
# Launch ONE ephemeral, hardened sandbox container that runs a single (model, task) cell
# to completion and exits. Adapted from badmodel-lab/localagent/docker/run_sandbox.sh:
# same hardening posture (read-only rootfs, dropped caps, no-new-privileges, non-root),
# but /work is now writable+exec (real code execution, not the old read-only shell DSL),
# and network is codebench_sandbox instead of localagent_sandbox (run setup_network.sh
# first). The whole write->test->revise loop for the task runs INSIDE this one process
# lifetime — no external command-dispatch needed, so this stays within the same one-shot-
# per-container model the existing sandbox already proved out.
#
#   ./run_agent_sandbox.sh --agent-kind badmodel --model gemma2:2b \
#       --task-dir tasks/c01_nth_even_fib_sum/public --out-dir /tmp/c01_badmodel_gemma2
set -euo pipefail

if [ "$(id -u)" = "0" ]; then
  echo "run_agent_sandbox.sh: refusing to run as root — --user overrides the image's" >&2
  echo "  non-root user with the invoking host user's own uid; running this as root" >&2
  echo "  would put the agent's sandbox container at uid 0 too." >&2
  exit 1
fi

AGENT_KIND=""
MODEL=""
TASK_DIR=""
OUT_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --agent-kind) AGENT_KIND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --task-dir) TASK_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${AGENT_KIND:?--agent-kind badmodel|aider is required}"
: "${MODEL:?--model is required}"
: "${TASK_DIR:?--task-dir is required (the public/ task fixture, never private/)}"
: "${OUT_DIR:?--out-dir is required}"

case "$AGENT_KIND" in
  badmodel) IMAGE="${CODEBENCH_BADMODEL_IMAGE:-codebench-badmodel}" ;;
  aider)    IMAGE="${CODEBENCH_AIDER_IMAGE:-codebench-aider}" ;;
  *) echo "unknown --agent-kind: $AGENT_KIND (want badmodel|aider)" >&2; exit 2 ;;
esac

NET="${CODEBENCH_NET:-codebench_sandbox}"
# codebench-ollama-proxy by CONTAINER NAME, not host.docker.internal (--internal removes
# the bridge's route to the host gateway entirely, verified live) and NOT badmodel-ollama
# directly (that gave the sandbox unauthenticated access to Ollama's whole admin API — see
# setup_network.sh's SECURITY comment). The proxy forwards only POST /v1/chat/completions.
MODEL_API_URL="${MODEL_API_URL:-http://codebench-ollama-proxy:8090/v1}"
SEARXNG_URL="${SEARXNG_URL:-http://badmodel-playground-searxng:8080}"
TIMEOUT_S="${CODEBENCH_TASK_TIMEOUT_S:-900}"

docker network inspect "$NET" >/dev/null 2>&1 || {
  echo "network $NET missing — run ./setup_network.sh first" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# /work is a HOST bind mount, not --tmpfs: verified live that `docker cp` after the
# container's own process exits returns an EMPTY tree even though the container hasn't
# been `docker rm`'d yet — tmpfs backing is torn down with the process/mount namespace, not
# with removal. A bind mount is already host-resident, needs no post-exit extraction step,
# and sidesteps the timing race entirely. It also needs to be world-writable up front: the
# container runs as a non-root, non-host uid (agent, 10001), and this dir is created here
# by whatever host user runs the harness.
#
# --user overrides the image's baked-in uid (agent, 10001) with the HOST user's own —
# files the container writes under /work then come out already owned by whoever ran the
# harness, so a later `rm -rf` on $OUT_DIR never hits "Permission denied" on a nested dir
# the container created (verified live: it does, without this). Doesn't weaken the sandbox
# — the isolation here comes from --cap-drop=ALL/--read-only/--network/the resource caps,
# not from which specific non-root uid number the process happens to run as.
RAW_DIR="$OUT_DIR/raw"
mkdir -p "$RAW_DIR"
chmod 0777 "$RAW_DIR"

CID=$(docker run -d \
  --network "$NET" \
  --user "$(id -u):$(id -g)" \
  --read-only \
  --tmpfs /tmp:rw,exec,size=64m,mode=1777 \
  -v "${RAW_DIR}:/work" \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --memory 1024m --cpus 2.0 \
  -e MODEL_API_URL="$MODEL_API_URL" \
  -e SEARXNG_URL="$SEARXNG_URL" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}" \
  -v "${TASK_DIR}:/task:ro" \
  "$IMAGE" --model "$MODEL" --task-dir /task --workdir /work)

echo "$CID" > "$OUT_DIR/container_id"
START_T=$EPOCHSECONDS

if timeout "$TIMEOUT_S" docker wait "$CID" > "$OUT_DIR/exit_code" 2>"$OUT_DIR/wait_stderr"; then
  :
else
  echo "codebench: task exceeded ${TIMEOUT_S}s wall clock, killing $CID" >&2
  docker kill "$CID" >/dev/null 2>&1 || true
  echo "124" > "$OUT_DIR/exit_code"   # 124 == coreutils timeout's own convention
fi
echo "$((EPOCHSECONDS - START_T))" > "$OUT_DIR/duration_s"

docker logs "$CID" > "$OUT_DIR/container.log" 2>&1 || true
docker rm -f "$CID" >/dev/null 2>&1 || true

echo "codebench: $AGENT_KIND/$MODEL run complete, exit=$(cat "$OUT_DIR/exit_code" 2>/dev/null || echo '?') out=$OUT_DIR"
