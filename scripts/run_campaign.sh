#!/usr/bin/env bash
# Launch a benchmark campaign so it OUTLIVES whatever started it.
#
# Why this exists. On 2026-09-02 two campaign runs died mid-flight with no trace on the machine:
# no OOM (kernel log clean, systemd-oomd inactive), no timer, no service action, nothing in the
# journal at the moment of death. They were started with an agent harness's own background-task
# facility, so they inherited that task's lifetime -- when the task went away, so did the run.
# `docs/DEV_CYCLE.md` gate 5 already said to do it this way ("setsid nohup ... < /dev/null &
# disown"); this script is that gate made hard to skip.
#
# It also implements gate 3 (singleton by lockfile, not by memory): two campaigns sharing one
# single-threaded Ollama backend do not merely run slowly, they contend for VRAM and produce cells
# whose timings mean nothing.
#
# Usage:
#   scripts/run_campaign.sh <run_id> <<'ENVEOF'
#   IDEA_TEST_MODELS=qwen2.5:7b
#   IDEA_TEST_IDS=210,211
#   ...one KEY=VALUE per line...
#   ENVEOF
#
# The run is detached immediately; this script prints the PID and log path and returns. Watch with
#   tail -f <log>            and stop with        scripts/run_campaign.sh --stop <run_id>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT}/agent/idea_test_results/_campaigns"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--stop" ]]; then
  run_id="${2:?--stop needs a run_id}"
  pidfile="${RUN_DIR}/${run_id}.pid"
  [[ -f "$pidfile" ]] || { echo "no pidfile for ${run_id}"; exit 1; }
  pgid="$(cat "$pidfile")"
  # Kill the PROCESS GROUP by id, never a name pattern. `pkill -f` matches any command line
  # containing the string -- including other agent sessions' shells, and including the very shell
  # running the pkill. That is not hypothetical: it happened, and it killed an unrelated session's
  # job as collateral.
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  rm -f "$pidfile"
  echo "stopped ${run_id} (pgid ${pgid})"
  exit 0
fi

run_id="${1:?usage: run_campaign.sh <run_id> [env on stdin]}"
lock="${RUN_DIR}/campaign.lock"
pidfile="${RUN_DIR}/${run_id}.pid"
log="${RUN_DIR}/${run_id}.log"
envfile="${RUN_DIR}/${run_id}.env"

# Singleton: one campaign at a time against a single-threaded backend.
if [[ -f "$lock" ]]; then
  held="$(cat "$lock" 2>/dev/null || true)"
  if [[ -n "$held" ]] && kill -0 "$held" 2>/dev/null; then
    echo "REFUSING: campaign already running (pid ${held}). Stop it first." >&2
    exit 1
  fi
  echo "note: clearing stale lock from pid ${held:-?}" >&2
fi

cat > "$envfile"   # caller pipes KEY=VALUE lines in
{
  echo "# launched $(date -Is)  run_id=${run_id}  commit=$(git -C "$ROOT" rev-parse --short HEAD)"
  cat "$envfile"
} > "${envfile}.stamped" && mv "${envfile}.stamped" "$envfile"

# setsid gives the run its OWN process group and session, so it survives the death of whatever
# launched it and can later be signalled precisely by that group id.
setsid nohup bash -c '
  set -a; source "'"$envfile"'"; set +a
  cd "'"$ROOT"'"
  export PYTHONPATH=.:services:agent
  exec ./.venv/bin/python -m agent.app.idea_test_runner
' < /dev/null > "$log" 2>&1 &
pgid=$!
disown || true
echo "$pgid" > "$pidfile"
echo "$pgid" > "$lock"

echo "launched ${run_id}"
echo "  pgid : ${pgid}"
echo "  log  : ${log}"
echo "  stop : scripts/run_campaign.sh --stop ${run_id}"
