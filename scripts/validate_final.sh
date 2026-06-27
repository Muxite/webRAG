#!/usr/bin/env bash
# Final validation: the 2 last new tests (077,078 hand plan) + the 4 fold-in
# candidates (040,042,044,047 auto plan). nano, parametric vs graph_compiled,
# R=2, record fixtures, tier 0, hard ceilings. Then gate_report each run-id.
set -u
cd /home/muk/projects/webRAG
PY=./.venv/bin/python
export PYTHONPATH=services:services/agent
OUT=services/agent/idea_test_results
LOG="$OUT/_validate_final/driver.log"
mkdir -p "$OUT/_validate_final"

keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_MODELS=openai/gpt-4.1-nano
export IDEA_TEST_EXECUTION_VARIANTS=parametric,graph_compiled
export IDEA_TEST_RUNS=2
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_FIXTURES=record
export IDEA_TEST_RENDER_DAG=0

echo "[$(date -u +%FT%TZ)] ==== VALIDATE-FINAL START ====" | tee -a "$LOG"

# 1) new tests, hand plan
IDEA_TEST_IDS=077,078 IDEA_TEST_COMPILED_PLAN_SOURCE=hand IDEA_TEST_USD_CEILING=1.00 \
  IDEA_TEST_RUN_ID=validate_new3 "$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] new3 exit $?" | tee -a "$LOG"

# 2) fold-ins, auto plan (they have no hand get_compiled_plan)
IDEA_TEST_IDS=040,042,044,047 IDEA_TEST_COMPILED_PLAN_SOURCE=auto IDEA_TEST_USD_CEILING=2.50 \
  IDEA_TEST_RUN_ID=validate_foldins "$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] foldins exit $?" | tee -a "$LOG"

echo "---- gate_report: NEW (077,078) want parametric LOW, compiled HIGH ----" | tee -a "$LOG"
"$PY" scripts/gate_report.py --run-id validate_new3 2>>"$LOG" | tee -a "$LOG"
echo "---- gate_report: FOLD-INS (040,042,044,047) ----" | tee -a "$LOG"
"$PY" scripts/gate_report.py --run-id validate_foldins 2>>"$LOG" | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ==== VALIDATE-FINAL DONE ====" | tee -a "$LOG"
