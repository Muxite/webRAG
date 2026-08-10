#!/usr/bin/env bash
set -u; cd /home/muk/projects/webRAG; PY=./.venv/bin/python
export PYTHONPATH=.:services:agent
OUT=agent/idea_test_results; LOG="$OUT/_validate_round3/driver.log"; mkdir -p "$OUT/_validate_round3"
keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)" SEARCH_API_KEY="$(keyval SEARCH_API_KEY)" SERPER_KEY="$(keyval SERPER_KEY)"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1 IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_MODELS=openai/gpt-4.1-nano IDEA_TEST_IDS=086,087,088,089
export IDEA_TEST_EXECUTION_VARIANTS=parametric,graph_compiled IDEA_TEST_COMPILED_PLAN_SOURCE=hand
export IDEA_TEST_RUNS=2 IDEA_TEST_EFFORT_TIERS=0 IDEA_TEST_FIXTURES=record IDEA_TEST_RENDER_DAG=0
export IDEA_TEST_USD_CEILING=1.50 IDEA_TEST_RUN_ID=validate_round3
echo "[$(date -u +%FT%TZ)] ==== VALIDATE-ROUND3 START ====" | tee -a "$LOG"
"$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] runner exit $?" | tee -a "$LOG"
"$PY" scripts/gate_report.py --run-id validate_round3 2>>"$LOG" | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] ==== VALIDATE-ROUND3 DONE ====" | tee -a "$LOG"
