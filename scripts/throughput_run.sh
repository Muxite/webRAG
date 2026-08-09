#!/usr/bin/env bash
# THROUGHPUT-MODE driver — run by the `benchmark` agent (live $ API calls).
#
# Purpose: minimize wall-clock for a multi-model / multi-cell run by (a) preflighting ONCE
# across all models (fanned out over the pooled connectors), overlapped with the ChromaDB
# warmup, and (b) executing cells CONCURRENTLY (IDEA_TEST_CONCURRENCY=N), guarded against
# provider 429s by the process-global in-flight LLM limiter (IDEA_TEST_LLM_MAX_INFLIGHT).
#
# This is the throughput counterpart to scripts/retest_055_grounding.sh, which is a
# clean-ATTRIBUTION A/B driver (one invocation per model, concurrency=1). Do NOT use this
# script when you need isolated per-model timing — the concurrency mixes cells' wall-clock.
#
# Key differences from the attribution driver:
#   * ONE invocation with IDEA_TEST_MODELS="m1,m2,..."  -> a single shared preflight.
#   * IDEA_TEST_CONCURRENCY=N                            -> N connector slots; cells run in parallel.
#   * IDEA_TEST_LLM_MAX_INFLIGHT caps total concurrent LLM wire calls across all slots (429 guard).
#   * IDEA_TEST_LLM_TIMEOUT lowers the per-call LLM timeout for faster-fail (optional).
#
# Launch (survives terminal close):
#   setsid nohup scripts/throughput_run.sh > /tmp/throughput_run.log 2>&1 &
set -uo pipefail   # NOT -e: a single failing cell must not abort the whole run.

REPO=/home/muk/projects/webRAG
cd "$REPO" || exit 1

PY=./.venv/bin/python
export PYTHONPATH=.:services:agent

RUN_ID="${IDEA_TEST_RUN_ID:-throughput_run}"
RESULTS_DIR=agent/idea_test_results
OUT_DIR="$RESULTS_DIR/_throughput"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ---- env: keys (CRLF-stripped, surrounding quotes removed) — same recipe as the barrage rig ----
keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"

export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5

# ---- THROUGHPUT knobs (tune to taste) ---------------------------------------------------------
export IDEA_TEST_CONCURRENCY="${IDEA_TEST_CONCURRENCY:-6}"          # N parallel cells + pooled connectors
export IDEA_TEST_LLM_MAX_INFLIGHT="${IDEA_TEST_LLM_MAX_INFLIGHT:-12}"  # global wire-call ceiling (429 guard)
export IDEA_TEST_LLM_TIMEOUT="${IDEA_TEST_LLM_TIMEOUT:-45}"         # per-call LLM timeout (faster-fail)
export IDEA_TEST_PARALLEL_ACTION_LIMIT="${IDEA_TEST_PARALLEL_ACTION_LIMIT:-2}"

export IDEA_TEST_FIXTURES="${IDEA_TEST_FIXTURES:-off}"
export IDEA_TEST_RUN_ID=$RUN_ID
export IDEA_TEST_EFFORT_TIERS="${IDEA_TEST_EFFORT_TIERS:-0}"
export IDEA_TEST_RUNS="${IDEA_TEST_RUNS:-1}"
export IDEA_TEST_IDS="${IDEA_TEST_IDS:-055}"
export IDEA_TEST_EXECUTION_VARIANTS="${IDEA_TEST_EXECUTION_VARIANTS:-graph_compiled}"
export IDEA_TEST_USD_CEILING="${IDEA_TEST_USD_CEILING:-2.00}"

# SINGLE invocation, comma-separated models -> one shared preflight, cells overlap.
MODELS="${IDEA_TEST_MODELS:-openai/gpt-4.1-nano,openai/gpt-5-mini}"
export IDEA_TEST_MODELS="$MODELS"

log "throughput_run — models=$MODELS concurrency=$IDEA_TEST_CONCURRENCY max_inflight=$IDEA_TEST_LLM_MAX_INFLIGHT ids=$IDEA_TEST_IDS R=$IDEA_TEST_RUNS"
"$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
log "throughput_run complete (exit $?) — see $LOG"
