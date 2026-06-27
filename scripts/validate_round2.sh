#!/usr/bin/env bash
# Live-validate the push-to-30 batch (079-085). nano, parametric vs graph_compiled,
# R=2, record fixtures, tier 0, hand plans, hard ceiling. Then gate_report.
set -u
cd /home/muk/projects/webRAG
PY=./.venv/bin/python
export PYTHONPATH=services:services/agent
OUT=services/agent/idea_test_results
LOG="$OUT/_validate_round2/driver.log"
mkdir -p "$OUT/_validate_round2"

keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_MODELS=openai/gpt-4.1-nano
export IDEA_TEST_IDS=079,080,081,082,083,084,085
export IDEA_TEST_EXECUTION_VARIANTS=parametric,graph_compiled
export IDEA_TEST_COMPILED_PLAN_SOURCE=hand
export IDEA_TEST_RUNS=2
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_FIXTURES=record
export IDEA_TEST_RENDER_DAG=0
export IDEA_TEST_USD_CEILING=2.00
export IDEA_TEST_RUN_ID=validate_round2

echo "[$(date -u +%FT%TZ)] ==== VALIDATE-ROUND2 START (079-085) ====" | tee -a "$LOG"
"$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] runner exit $?" | tee -a "$LOG"
echo "---- result files by test ----" | tee -a "$LOG"
for t in 079 080 081 082 083 084 085; do
  n=$(ls "$OUT"/validate_round2_${t}_*.json 2>/dev/null | grep -v summary | wc -l)
  echo "  test $t: $n result file(s)" | tee -a "$LOG"
done
echo "---- gate_report (want parametric LOW <=0.35, graph_compiled HIGH >=0.80) ----" | tee -a "$LOG"
"$PY" scripts/gate_report.py --run-id validate_round2 2>>"$LOG" | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ==== VALIDATE-ROUND2 DONE ====" | tee -a "$LOG"
