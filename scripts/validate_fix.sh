#!/usr/bin/env bash
# Re-validate the 5 FIXED tier tests LIVE after the Sonnet repairs.
# nano, parametric (anti-leak floor) + graph_compiled (should be HIGH), R=2,
# record fixtures, tier 0, hard USD ceiling. Then gate_report.
set -u
cd /home/muk/projects/webRAG
PY=./.venv/bin/python
export PYTHONPATH=.:services:agent
OUT=agent/idea_test_results
LOG="$OUT/_validate_fix/driver.log"
mkdir -p "$OUT/_validate_fix"

keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"
export SERPER_KEY="$(keyval SERPER_KEY)"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_MODELS=openai/gpt-4.1-nano
export IDEA_TEST_IDS=063,066,067,068,071
export IDEA_TEST_EXECUTION_VARIANTS=parametric,graph_compiled
export IDEA_TEST_COMPILED_PLAN_SOURCE=hand
export IDEA_TEST_RUNS=2
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_FIXTURES=record
export IDEA_TEST_RENDER_DAG=0
export IDEA_TEST_USD_CEILING=1.50
export IDEA_TEST_RUN_ID=validate_fix

echo "[$(date -u +%FT%TZ)] ==== VALIDATE-FIX START (5 fixed tests) ====" | tee -a "$LOG"
"$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] ==== runner exit $? ====" | tee -a "$LOG"
echo "---- result files by test ----" | tee -a "$LOG"
for t in 063 066 067 068 071; do
  n=$(ls "$OUT"/validate_fix_${t}_*.json 2>/dev/null | grep -v summary | wc -l)
  echo "  test $t: $n result file(s)" | tee -a "$LOG"
done
echo "---- gate_report (want parametric LOW <=0.35, graph_compiled HIGH >=0.80) ----" | tee -a "$LOG"
"$PY" scripts/gate_report.py --run-id validate_fix 2>>"$LOG" | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ==== VALIDATE-FIX DONE ====" | tee -a "$LOG"
