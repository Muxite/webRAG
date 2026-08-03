#!/usr/bin/env bash
# One-time proof exercise (not run per-task): launch a throwaway container with the exact
# same hardening flags run_agent_sandbox.sh uses, but running containment_check.sh instead
# of the real agent entrypoint. Run this after setup_network.sh, before trusting any live
# matrix run.
set -euo pipefail

if [ "$(id -u)" = "0" ]; then
  echo "verify_containment.sh: refusing to run as root (see run_agent_sandbox.sh for why)" >&2
  exit 1
fi

CB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:?usage: verify_containment.sh <codebench-badmodel|codebench-aider>}"
NET="${CODEBENCH_NET:-codebench_sandbox}"

# Mirrors run_agent_sandbox.sh's real mount posture exactly (host bind mount for /work,
# not --tmpfs — see that script's comment on why tmpfs doesn't survive past container exit;
# irrelevant to THIS script specifically since nothing gets extracted here, but this exists
# to verify the posture actually used in production, so it must match it). No
# --add-host host.docker.internal — --internal removes that route entirely regardless
# (verified live, and containment_check.sh's own §3b re-checks this every run).
WORK_SCRATCH=$(mktemp -d)
chmod 0777 "$WORK_SCRATCH"
trap 'rm -rf "$WORK_SCRATCH"' EXIT

docker run --rm \
  --network "$NET" \
  --user "$(id -u):$(id -g)" \
  --read-only \
  --tmpfs /tmp:rw,exec,size=64m,mode=1777 \
  -v "$WORK_SCRATCH:/work" \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --memory 1024m --cpus 2.0 \
  -v "$CB_DIR/containment_check.sh:/tmp/containment_check.sh:ro" \
  --entrypoint bash \
  "$IMAGE" /tmp/containment_check.sh
