#!/usr/bin/env bash
# Native adaptive-engine A/B benchmark driver.
#
# Reforms the compiled-era retest_055 pattern for the NATIVE engine and its research questions:
#   - arms selected by name via IDEA_TEST_ARM (baseline / good_adaptive / reexpand_only /
#     confidence_only / backtrack_only / kvote_only / full) — no hand-written settings files;
#   - connector-retry ON in EVERY arm (infra-fairness: transient empty/timeout fetches don't
#     confound the accuracy delta — the lesson from the Group-2 flakiness window);
#   - arms INTERLEAVED per (task, repeat) so both arms share the same network window;
#   - NO fixtures (they don't fit a variably-exploring engine — record-on-one-arm misses on another);
#   - one IDEA_TEST_RUN_ID per arm (<RUN_ID>_<arm>) so scripts/adaptive_ab_analyze.py can roll up.
#
# Why not fixtures / why interleave / why connector-retry: see scripts/BENCHMARK_NATIVE.md.
#
# Usage:
#   ARMS="baseline good_adaptive" TASKS="122 128 134 140" MODEL="openai/gpt-5-mini" \
#   R=5 RUN_ID=nativeab USD_CEILING=8.00 scripts/native_ab_run.sh
#
# Launch unattended (survives terminal close):
#   setsid nohup env ARMS="baseline good_adaptive" TASKS="122 128 134 140" R=5 RUN_ID=nativeab \
#     scripts/native_ab_run.sh > /tmp/nativeab.log 2>&1 &
set -uo pipefail   # NOT -e: one failing cell must not abort the matrix.

REPO=/home/muk/projects/webRAG
cd "$REPO" || exit 1
PY=./.venv/bin/python
export PYTHONPATH=.:services:agent

# ---- params (env-overridable) ----
ARMS="${ARMS:-baseline good_adaptive}"
TASKS="${TASKS:-122 125 128 130 134 138 140 144}"
MODEL="${MODEL:-openai/gpt-5-mini}"
R="${R:-5}"
RUN_ID="${RUN_ID:-nativeab}"
USD_CEILING="${USD_CEILING:-8.00}"

# ---- keys (services/keys.env has CRLF; strip it) ----
keyval() { grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"
export LLM_PROVIDER=openrouter
export MODEL_API_URL=https://openrouter.ai/api/v1
export CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5

# ---- fixed harness controls ----
export IDEA_TEST_CONCURRENCY=1          # shared connectors per process; keep serial for clean attribution
export IDEA_TEST_PARALLEL_ACTION_LIMIT=1
export IDEA_TEST_FIXTURES=off           # native engine explores variably; fixtures don't serve it
export IDEA_TEST_EXECUTION_VARIANTS=graph
export IDEA_TEST_MODELS="$MODEL"
export IDEA_TEST_CONNECTOR_RETRY=1      # infra-fairness: absorb transient tool failures in ALL arms
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_USD_CEILING="$USD_CEILING"

OUT_DIR="agent/idea_test_results/_${RUN_ID}"
mkdir -p "$OUT_DIR"
echo "[native_ab_run] arms=[$ARMS] tasks=[$TASKS] model=$MODEL R=$R run_id=$RUN_ID ceiling=\$$USD_CEILING"

# ---- interleave: for each task, each repeat, run every arm back-to-back (shared network window) ----
for task in $TASKS; do
  for rep in $(seq 1 "$R"); do
    for arm in $ARMS; do
      # each (task,arm) invocation does exactly 1 run; the rep index lives in the run-id suffix so
      # files don't collide, and arms for the same (task,rep) are adjacent in wall-clock time.
      export IDEA_TEST_ARM="$arm"
      export IDEA_TEST_IDS="$task"
      export IDEA_TEST_RUNS=1
      export IDEA_TEST_RUN_ID="${RUN_ID}_${arm}_rep${rep}"
      echo "[native_ab_run] task=$task rep=$rep arm=$arm ..."
      "$PY" -m agent.app.idea_test_runner >> "$OUT_DIR/driver.log" 2>&1
      echo "[native_ab_run]   exit=$? (task=$task rep=$rep arm=$arm)"
    done
  done
done

echo "[native_ab_run] matrix complete. Analyzing..."
# Roll up every rep-suffixed run-id per arm (comma-joined) through the analysis layer.
adaptive_ids=""; baseline_ids=""
for rep in $(seq 1 "$R"); do
  baseline_ids="${baseline_ids:+$baseline_ids,}${RUN_ID}_baseline_rep${rep}"
  adaptive_ids="${adaptive_ids:+$adaptive_ids,}${RUN_ID}_good_adaptive_rep${rep}"
done
"$PY" scripts/adaptive_ab_analyze.py \
  --baseline-run-id "$baseline_ids" --adaptive-run-id "$adaptive_ids" \
  --out "$OUT_DIR/analysis" 2>&1 | tee -a "$OUT_DIR/driver.log"
echo "[native_ab_run] done — see $OUT_DIR/analysis/"
