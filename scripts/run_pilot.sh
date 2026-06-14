#!/usr/bin/env bash
# Bounded pilot for the cost-recovery benchmark.
# Stage 0: reference model (records complete web fixtures + gold reference lines).
# Stage 1: cheap models replay those fixtures (fair, network-free).
# Stage 2: build the recovery curve / Pareto over the 6-test subset.
set -uo pipefail
cd "$(dirname "$0")/.."

export OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/')"
export SEARCH_API_KEY="$(grep -E '^SEARCH_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/')"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1
export CHROMA_URL=http://localhost:8001 DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5

SUBSET="026,025,019,037,038,036"
PY="PYTHONPATH=services:services/agent ./.venv/bin/python -m agent.app.idea_test_runner"
COMMON="IDEA_TEST_IDS=$SUBSET IDEA_TEST_CONCURRENCY=1 IDEA_TEST_RUNS=1 IDEA_TEST_MAX_STEPS=40 IDEA_TEST_REPORT_VERBOSITY=1 IDEA_TEST_EXECUTION_VARIANTS=graph,parametric,naive_rag"

echo "===== STAGE 0: reference (record fixtures) ====="
env $COMMON \
  MODEL_NAME=google/gemini-3.1-pro-preview \
  IDEA_TEST_MODELS="google/gemini-3.1-pro-preview" \
  IDEA_TEST_EFFORT_TIERS="0" \
  IDEA_TEST_FIXTURES=record \
  PYTHONPATH=services:services/agent ./.venv/bin/python -m agent.app.idea_test_runner

echo "===== STAGE 1: cheap models (replay fixtures) ====="
env $COMMON \
  MODEL_NAME=google/gemini-2.5-flash \
  IDEA_TEST_MODELS="google/gemini-2.5-flash,openai/gpt-5-mini,openai/gpt-4.1-nano,google/gemini-2.5-flash-lite" \
  IDEA_TEST_EFFORT_TIERS="0,20" \
  IDEA_TEST_FIXTURES=replay \
  PYTHONPATH=services:services/agent ./.venv/bin/python -m agent.app.idea_test_runner

echo "===== STAGE 2: recovery curve ====="
./.venv/bin/python scripts/recovery_curve.py --since "$(date +%Y%m%d)" --tests "$SUBSET"
echo "PILOT DONE"
