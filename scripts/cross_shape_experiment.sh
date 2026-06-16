#!/usr/bin/env bash
# Cross-shape A/B/C experiment for the compiled-scaffold thesis.
#
# THESIS: an expensive-model-compiled DAG + cheap execution beats a cheap-model-built DAG
# across task SHAPES — pure fan-out (052/053), dependent chains (050/051) and a mixed DAG (054).
#
# MATRIX: cheap model  x  {graph, sequential_react, graph_compiled}  x  {050,051,052,053,054}
#   - A  = graph            (cheap model BUILDS the GoT graph at runtime)
#   - B  = graph_compiled   (cheap model only EXECUTES a strong-model-authored DAG)
#          - B-hand : plan from the test module's get_compiled_plan()   (IDEA_TEST_COMPILED_PLAN_SOURCE=hand)
#          - B-auto : plan from the offline compiler                    (IDEA_TEST_COMPILED_PLAN_SOURCE=auto)
#   - C  = sequential_react (strong linear comparator, same toolset)
#   - reference model across A/B/C-hand = the quality CEILING.
#
# COSTS REAL MONEY (OpenRouter) + needs ChromaDB on :8001. User-triggered; not run automatically.
# Concurrency=1 is MANDATORY (shared connectors). See COST_BENCHMARK_HANDOFF.md.
#
# NOTE on fixtures: 050-054 are URL-free (search-driven), so pages are discovered live. We RECORD
# on the reference pass and REPLAY (replay-or-record, fills misses live) for cheap models. That is
# not byte-identical evidence across models — acceptable for a first cross-shape pass; tighten to
# replay_strict only after a prewarm that captures the discovered queries/URLs.
set -uo pipefail
cd "$(dirname "$0")/.."

export OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/')"
export SEARCH_API_KEY="$(grep -E '^SEARCH_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/')"
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1
export CHROMA_URL=http://localhost:8001 DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5

# --- knobs ---------------------------------------------------------------------------------
TESTS="${TESTS:-050,051,052,053,054}"
REFERENCE_MODEL="${REFERENCE_MODEL:-google/gemini-3.1-pro-preview}"
AUTHOR_MODEL="${AUTHOR_MODEL:-$REFERENCE_MODEL}"   # strong model that authors B-auto plans offline
CHEAP_MODELS="${CHEAP_MODELS:-google/gemini-2.5-flash,openai/gpt-5-mini,openai/gpt-4.1-nano,google/gemini-2.5-flash-lite}"
REPEATS="${REPEATS:-3}"                            # error bars on the headline cells
RUN_ID="${RUN_ID:-xshape_$(date -u +%Y%m%d_%H%M%S)}"
PY=(./.venv/bin/python -m agent.app.idea_test_runner)

export IDEA_TEST_IDS="$TESTS"
export IDEA_TEST_CONCURRENCY=1                     # MANDATORY (shared connectors)
export IDEA_TEST_PARALLEL_ACTION_LIMIT=1           # breadth fix for the native GoT graph arm
export IDEA_TEST_COMPILED_CONCURRENCY="${IDEA_TEST_COMPILED_CONCURRENCY:-3}"  # parallel leaves; tune vs starvation
export IDEA_TEST_MAX_STEPS=40
export IDEA_TEST_REPORT_VERBOSITY=1
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_RUN_ID="$RUN_ID"
export PYTHONPATH=services:services/agent

echo "===== run_id=$RUN_ID  tests=$TESTS  author=$AUTHOR_MODEL ====="

echo "===== STAGE A: author B-auto plans offline (paid once) ====="
# 4096 tokens: the 6-item breadth plan (052) decomposes into ~12 leaves and overflows 2048.
./.venv/bin/python scripts/compile_plans.py --tests "$TESTS" --author-model "$AUTHOR_MODEL" --max-tokens 4096 || exit 1
./.venv/bin/python scripts/compile_plans.py --tests "$TESTS" --dry-run

echo "===== STAGE B: reference ceiling + record fixtures (A / C / B-hand) ====="
env IDEA_TEST_MODELS="$REFERENCE_MODEL" MODEL_NAME="$REFERENCE_MODEL" \
    IDEA_TEST_EXECUTION_VARIANTS="graph,sequential_react,graph_compiled" \
    IDEA_TEST_COMPILED_PLAN_SOURCE=hand \
    IDEA_TEST_RUNS=1 IDEA_TEST_FIXTURES=record \
    "${PY[@]}"

echo "===== STAGE C1: cheap models, A / C / B-hand (replay) ====="
env IDEA_TEST_MODELS="$CHEAP_MODELS" MODEL_NAME="${CHEAP_MODELS%%,*}" \
    IDEA_TEST_EXECUTION_VARIANTS="graph,sequential_react,graph_compiled" \
    IDEA_TEST_COMPILED_PLAN_SOURCE=hand \
    IDEA_TEST_RUNS="$REPEATS" IDEA_TEST_FIXTURES=replay \
    "${PY[@]}"

echo "===== STAGE C2: cheap models, B-auto (compiler plans, replay) ====="
# Distinct run-id suffix: B-auto's graph_compiled files must NOT overwrite C1's B-hand
# graph_compiled files (same variant name). "$RUN_ID" is still a prefix of "${RUN_ID}_auto",
# so --run-id "$RUN_ID" analysis picks up both passes.
env IDEA_TEST_MODELS="$CHEAP_MODELS" MODEL_NAME="${CHEAP_MODELS%%,*}" \
    IDEA_TEST_EXECUTION_VARIANTS="graph_compiled" \
    IDEA_TEST_COMPILED_PLAN_SOURCE=auto \
    IDEA_TEST_RUN_ID="${RUN_ID}_auto" \
    IDEA_TEST_RUNS="$REPEATS" IDEA_TEST_FIXTURES=replay \
    "${PY[@]}"

echo "===== STAGE D: analysis (by run_id=$RUN_ID) ====="
./.venv/bin/python scripts/level_ladder.py --run-id "$RUN_ID" || true
./.venv/bin/python scripts/recovery_curve.py --run-id "$RUN_ID" --tests "$TESTS" \
    --reference-models "$REFERENCE_MODEL" || true
echo "CROSS-SHAPE EXPERIMENT DONE  (run_id=$RUN_ID)"
