#!/usr/bin/env bash
# Run ONE bad-model-lab cell = (subject model) x (mitigation profile) x (task tier).
# It points the EXISTING benchmark runner at a bad local model and scores it with the
# EXISTING gate_report — the lab adds only env wiring + parse-failure telemetry.
#
#   ./badmodel-lab/run_cell.sh <model_tag> <profile> <tier|ids> [runs]
#   ./badmodel-lab/run_cell.sh qwen2.5:0.5b m1_thin micro 1
#   ./badmodel-lab/run_cell.sh openai/gpt-4.1-nano m1_thin micro 3     # anchor (OpenRouter, $)
#
# profiles: see badmodel-lab/profiles/*.env   tiers: sanity|micro|reachable|hard  (or a literal id list)
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$LAB_DIR/.." && pwd)"
cd "$REPO"

MODEL="${1:?model tag, e.g. qwen2.5:0.5b or openai/gpt-4.1-nano}"
PROFILE="${2:?profile name, e.g. m1_thin (see badmodel-lab/profiles/)}"
TIER="${3:?tier: sanity|micro|reachable|hard, or a comma-separated id list}"
RUNS="${4:-1}"

PROFILE_FILE="$LAB_DIR/profiles/${PROFILE}.env"
[ -f "$PROFILE_FILE" ] || { echo "no such profile: $PROFILE_FILE" >&2; exit 2; }

case "$TIER" in
  sanity)    IDS="048" ;;
  micro)     IDS="m01,m02,m03" ;;                     # authored, live-verified lab tests
  reachable) IDS="062,064,069,070,072,076,078" ;;     # single count/argmax/entity (memory-known simple)
  format)    IDS="f01,f02,f03" ;;                      # format-stress: micro fact, hard multi-field JSON shape
  hard)      IDS="063,071,073,075,077" ;;             # known to floor; anchors + merge only
  *)         IDS="$TIER" ;;                            # literal comma list
esac

# --- keys from services/keys.env (CRLF per HANDOFF); only the three we need ---
# Values may be double-quoted in keys.env; strip the quotes the same way every other
# key-sourcing script in this repo does (scripts/*.sh's `keyval()` helper), or a
# remote-provider key ends up literally including the quote characters and every
# OpenRouter call 401s. Found live via this exact failure (E1, 2026-08-03) --
# openai/gpt-4.1-nano had never been successfully run through this script before.
while IFS='=' read -r k v; do
  case "$k" in SEARCH_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|SERPER_KEY) export "$k=$(sed -E 's/^"(.*)"$/\1/' <<< "$v")" ;; esac
done < <(tr -d '\r' < services/keys.env)

# 2026-08-08: default search provider is now Serper (Brave's SEARCH_API_KEY ran out of quota,
# confirmed live HTTP 402). Set SEARCH_PROVIDER=brave explicitly to opt back into the Brave path
# below once/if its quota resets.
export SEARCH_PROVIDER="${SEARCH_PROVIDER:-serper}"

# The Brave SEARCH_API_KEY in keys.env is stale/invalid in this environment
# (SUBSCRIPTION_TOKEN_INVALID); the running euglena agent holds the valid key.
# Prefer the container's key so retrieval actually works; keys.env is the fallback.
# Only relevant when SEARCH_PROVIDER=brave — left in place (not removed) so switching back needs
# no script changes, per this repo's additive-not-destructive convention.
if [ "$SEARCH_PROVIDER" = "brave" ]; then
  _KEY_CTR="${BADMODEL_SEARCH_KEY_FROM_CTR:-euglena-agent-1}"
  _ck="$(docker exec "$_KEY_CTR" printenv SEARCH_API_KEY 2>/dev/null || true)"
  if [ -n "$_ck" ]; then export SEARCH_API_KEY="$_ck"; echo ">> search key sourced from $_KEY_CTR" >&2; fi
fi

export PYTHONPATH=services:services/agent
export IDEA_TEST_CONCURRENCY=1                 # MANDATORY — shared connectors
export CHROMA_URL="${CHROMA_URL:-http://localhost:8001}"

# --- provider routing: OpenRouter for "vendor/model" ids, else local ollama ---
if [[ "$MODEL" == */* ]]; then
  PLACE=remote
  export LLM_PROVIDER=openrouter
  export MODEL_API_URL=https://openrouter.ai/api/v1
else
  PLACE=local
  export LLM_PROVIDER=openai_compatible
  export MODEL_API_URL="${BADMODEL_OLLAMA_URL:-http://localhost:11435/v1}"
  export OPENAI_API_KEY=ollama                 # ollama ignores the key; must be non-empty
  # The runner also PRE-FLIGHTS the validation model against the same endpoint and
  # ABORTS if it 404s. The lab's micro/reachable tests use deterministic grep
  # validators (no LLM judge, rubric off), so the validation model is a never-called
  # label — point it at the local subject so its preflight passes. Do NOT enable
  # IDEA_TEST_RUBRIC on local runs (a 0.5B judge is worthless + would 404 remotely).
  export IDEA_TEST_VALIDATION_MODEL="${IDEA_TEST_VALIDATION_MODEL:-$MODEL}"
fi

# --- bad-model essentials ---
export IDEA_TEST_PREFLIGHT_JSON=0              # do NOT drop a model that can't JSON — it's the subject
export IDEA_TEST_JSON_TELEMETRY=1              # capture parse-failure classes (analyze.py reads them)
export IDEA_TEST_MODELS="$MODEL"
export IDEA_TEST_EXECUTION_VARIANTS=graph_compiled
export IDEA_TEST_COMPILED_PLAN_SOURCE=hand
export IDEA_TEST_RUNS="$RUNS"
export IDEA_TEST_USD_CEILING="${BADMODEL_USD_CEILING:-5}"   # hard $ guardrail across the whole cell
export IDEA_TEST_IDS="$IDS"
# Single-leaf micro tasks: return the leaf fact directly (keeps "value — source: url" so grounding
# is credited; skips a weak-model aggregation call that drops the URL). No-op on multi-leaf tiers.
export IDEA_TEST_COMPILED_SINGLE_LEAF_PASSTHROUGH=1

SLUG="$(echo "$MODEL" | tr '/:' '--')"
TIERTAG="$(echo "$TIER" | tr '/,: ' '----' | cut -c1-16)"   # keep tier distinct in the run_id
export IDEA_TEST_RUN_ID="bml__${SLUG}__${PROFILE}__${TIERTAG}"

# mitigation profile overrides (leaf mode / votes / token budgets) — last word wins
set -a; . "$PROFILE_FILE"; set +a

# attribution sidecar: run_id -> (model, place, profile, tier) for analyze.py
mkdir -p "$LAB_DIR/results"
./.venv/bin/python - "$LAB_DIR/results/cells.jsonl" <<PY
import json, sys, time
row = {"run_id": "$IDEA_TEST_RUN_ID", "model": "$MODEL", "place": "$PLACE",
       "profile": "$PROFILE", "tier": "$TIER", "ids": "$IDS", "runs": $RUNS, "t": time.time()}
open(sys.argv[1], "a").write(json.dumps(row) + "\n")
PY

echo ">> cell  model=$MODEL  profile=$PROFILE  tier=$TIER  ids=$IDS  runs=$RUNS  run_id=$IDEA_TEST_RUN_ID  place=$PLACE" >&2
./.venv/bin/python -m agent.app.idea_test_runner
echo ">> gate_report:" >&2
./.venv/bin/python scripts/gate_report.py --run-id "$IDEA_TEST_RUN_ID" || true
