#!/usr/bin/env bash
# Entrypoint for the "conventional agent" baseline (aider), matching the same
# --model/--task-dir/--workdir CLI contract run_agent_sandbox.sh uses for the badmodel
# image, so the matrix runner can dispatch either agent kind identically.
#
# Reads the task's public fixture (starter files + spec) from --task-dir (read-only bind
# mount, never contains canonical/hidden tests — see codebench/materialize_task.py),
# copies it into a fresh git repo at --workdir, then runs aider non-interactively pointed
# at MODEL_API_URL (the local Ollama OpenAI-compatible endpoint via host.docker.internal).
#
# Flag names verified live against `aider --help` for the exact pinned version (0.60.1) —
# the docs-derived first draft had --yes (should be --yes-always) and --analytics-disable
# (does not exist in this version at all); re-verify again if ../Dockerfile's pin changes.
set -euo pipefail

MODEL=""
TASK_DIR="/task"
WORKDIR="/work"
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --task-dir) TASK_DIR="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    *) echo "codebench-aider: unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${MODEL:?--model is required}"
: "${MODEL_API_URL:?MODEL_API_URL env var is required}"

if [ -d "$TASK_DIR/repo" ]; then
  cp -r "$TASK_DIR/repo/." "$WORKDIR/"
fi

# HOME=$WORKDIR: the container is --read-only outside /work (only WORKDIR is writable), so
# the agent user's real home (baked in at build time, part of the read-only rootfs) has
# nowhere to write `git config --global` or aider's own state files. Verified live: without
# this, `git config --global` silently no-ops against an unwritable path and every git
# command then fails with "detected dubious ownership in repository at '$WORKDIR'" — the
# tmpfs mountpoint is root-owned at container start, and git refuses to operate there as a
# non-root user without an explicit safe.directory exception.
export HOME="$WORKDIR"
cd "$WORKDIR"
git init -q
git config --global --add safe.directory "$WORKDIR"
git config user.email "aider@codebench.local"
git config user.name "codebench-aider"
git add -A
git commit -q -m "starter fixture" --allow-empty

export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export OPENAI_API_BASE="$MODEL_API_URL"

aider \
  --model "openai/${MODEL}" \
  --openai-api-base "$MODEL_API_URL" \
  --openai-api-key "$OPENAI_API_KEY" \
  --yes-always \
  --no-check-update \
  --no-stream \
  --message-file "$TASK_DIR/prompt.md" \
  --exit

echo "codebench-aider: run complete"
