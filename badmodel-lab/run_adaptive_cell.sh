#!/usr/bin/env bash
# Run ONE bad-model-lab cell = (subject model) x (native-mode profile) x (task tier),
# through the NATIVE ADAPTIVE engine (execution_variant=graph) instead of the compiled
# scaffold (graph_compiled). This is the harness gap the compiled-only run_cell.sh left:
# it hardcodes graph_compiled and strips the compiled-only knobs (leaf mode, plan
# source, single-leaf passthrough) that don't apply to native — see ADAPTIVE_ENGINE.md.
#
# In native mode the model itself proposes/expands GoT leaves each turn (idea_policies/
# expansion.py) instead of executing a plan a strong model authored offline. That is a
# materially harder ask for a small/mid local model — this script exists to measure it.
#
#   ./badmodel-lab/run_adaptive_cell.sh <model_tag> <profile> <tier|ids> [runs]
#   ./badmodel-lab/run_adaptive_cell.sh qwen2.5:7b a0_native_baseline micro 3
#
# profiles: see badmodel-lab/profiles/a*.env   tiers: sanity|micro|reachable|format|hard (or literal ids)
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$LAB_DIR/.." && pwd)"
cd "$REPO"

MODEL="${1:?model tag, e.g. qwen2.5:7b or openai/gpt-4.1-nano}"
PROFILE="${2:?profile name, e.g. a0_native_baseline (see badmodel-lab/profiles/a*.env)}"
TIER="${3:?tier: sanity|micro|reachable|format|hard, or a comma-separated id list}"
RUNS="${4:-1}"

PROFILE_FILE="$LAB_DIR/profiles/${PROFILE}.env"
[ -f "$PROFILE_FILE" ] || { echo "no such profile: $PROFILE_FILE" >&2; exit 2; }

case "$TIER" in
  sanity)    IDS="048" ;;
  micro)     IDS="m01,m02,m03" ;;
  reachable) IDS="062,064,069,070,072,076,078" ;;
  format)    IDS="f01,f02,f03" ;;
  hard)      IDS="063,071,073,075,077" ;;
  *)         IDS="$TIER" ;;
esac

while IFS='=' read -r k v; do
  case "$k" in SEARCH_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY) export "$k=$v" ;; esac
done < <(tr -d '\r' < services/keys.env)

_KEY_CTR="${BADMODEL_SEARCH_KEY_FROM_CTR:-euglena-agent-1}"
_ck="$(docker exec "$_KEY_CTR" printenv SEARCH_API_KEY 2>/dev/null || true)"
if [ -n "$_ck" ]; then export SEARCH_API_KEY="$_ck"; echo ">> search key sourced from $_KEY_CTR" >&2; fi

export PYTHONPATH=services:services/agent
export IDEA_TEST_CONCURRENCY=1
export CHROMA_URL="${CHROMA_URL:-http://localhost:8001}"

if [[ "$MODEL" == */* ]]; then
  PLACE=remote
  export LLM_PROVIDER=openrouter
  export MODEL_API_URL=https://openrouter.ai/api/v1
else
  PLACE=local
  export LLM_PROVIDER=openai_compatible
  export MODEL_API_URL="${BADMODEL_OLLAMA_URL:-http://localhost:11435/v1}"
  export OPENAI_API_KEY=ollama
  export IDEA_TEST_VALIDATION_MODEL="${IDEA_TEST_VALIDATION_MODEL:-$MODEL}"
fi

export IDEA_TEST_PREFLIGHT_JSON=0
export IDEA_TEST_JSON_TELEMETRY=1
export IDEA_TEST_MODELS="$MODEL"
export IDEA_TEST_EXECUTION_VARIANTS=graph        # <-- the native adaptive engine, NOT graph_compiled
export IDEA_TEST_RUNS="$RUNS"
export IDEA_TEST_USD_CEILING="${BADMODEL_USD_CEILING:-5}"
export IDEA_TEST_IDS="$IDS"

SLUG="$(echo "$MODEL" | tr '/:' '--')"
TIERTAG="$(echo "$TIER" | tr '/,: ' '----' | cut -c1-16)"
export IDEA_TEST_RUN_ID="bmladapt__${SLUG}__${PROFILE}__${TIERTAG}"

# native-mode profile overrides (IDEA_TEST_GOT_*) — last word wins
set -a; . "$PROFILE_FILE"; set +a

mkdir -p "$LAB_DIR/results"
./.venv/bin/python - "$LAB_DIR/results/cells.jsonl" <<PY
import json, sys, time
row = {"run_id": "$IDEA_TEST_RUN_ID", "model": "$MODEL", "place": "$PLACE",
       "profile": "$PROFILE", "tier": "$TIER", "ids": "$IDS", "runs": $RUNS, "t": time.time(),
       "variant": "graph"}
open(sys.argv[1], "a").write(json.dumps(row) + "\n")
PY

echo ">> native cell  model=$MODEL  profile=$PROFILE  tier=$TIER  ids=$IDS  runs=$RUNS  run_id=$IDEA_TEST_RUN_ID  place=$PLACE" >&2
./.venv/bin/python -m agent.app.idea_test_runner
echo ">> gate_report:" >&2
./.venv/bin/python scripts/gate_report.py --run-id "$IDEA_TEST_RUN_ID" || true
