#!/usr/bin/env bash
# ladder03 driver. Resumable: any campaign that already has cells is skipped, so re-running this
# continues a sweep that was stopped or died. Serialised by run_campaign.sh's singleton lock
# (one ollama, concurrency 1) — it waits for the lock rather than racing it.
#
#   scripts/run_ladder03.sh agent/idea_test_results/prereg/ladder03_order.json
#
# Launch it DETACHED, or it inherits the caller's lifetime and dies with it:
#   setsid nohup bash scripts/run_ladder03.sh <order.json> < /dev/null > ladder03.log 2>&1 & disown
#
# Do NOT edit agent/app/** while this runs: each campaign is a fresh process, so a mid-sweep edit
# runs different code for later campaigns than earlier ones. Analysis scripts are safe.
#
# ladder03: 28 campaigns x 12 tasks = 336 cells, corpus replay, seeded, $0.
# Ordered so the langgraph/off block runs first — 15 of its cells have known byte-exact values
# from probe3, giving continuous drift detection at no extra cost.
set -u
ROOT=/home/muk/projects/webRAG
cd "$ROOT" || exit 1
CAMP="$ROOT/agent/idea_test_results/_campaigns"
ORDER="$1"

# Wait for any in-flight campaign to release the singleton lock.
while :; do
  held=$(cat "$CAMP/campaign.lock" 2>/dev/null || true)
  [ -n "$held" ] && kill -0 "$held" 2>/dev/null || break
  sleep 10
done

python3 - "$ORDER" <<'PY' > /tmp/ladder03_order.txt
import json, sys
for rid, model, host, state in json.load(open(sys.argv[1])):
    print(f"{rid}\t{model}\t{host}\t{state}")
PY

while IFS=$'\t' read -r rid model host state; do
  [ -n "$rid" ] || continue
  if ls "$ROOT"/agent/idea_test_results/${rid}_*_r1.json >/dev/null 2>&1; then
    echo "=== skipping $rid (already has cells)"; continue
  fi
  echo "=== launching $rid  model=$model host=$host module=$state  at $(date -Is)"
  modules=""; [ "$state" = "derive" ] && modules="derive"
  scripts/run_campaign.sh "$rid" <<ENVEOF
SEARCH_PROVIDER=corpus
LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/numeric22
LEDGER_MAX_LIVE_FALLBACKS=0
LEDGER_CONTEXT_FIT=1
LEDGER_ZERO_VISIT_GATE=1
LEDGER_HOST_MODULES=$modules
LLM_SEED=12345
LLM_PROVIDER=ollama
MODEL_API_URL=http://127.0.0.1:11435/v1
OPENAI_API_KEY=dummy
IDEA_TEST_MODELS=$model
IDEA_TEST_VALIDATION_MODEL=qwen2.5:7b
IDEA_TEST_IDS=210,211,212,213,214,215,216,217,218,219,220,221
IDEA_TEST_EXECUTION_VARIANTS=$host
IDEA_TEST_RUNS=1
IDEA_TEST_CONCURRENCY=1
IDEA_TEST_RUN_ID=$rid
IDEA_TEST_JSON_TELEMETRY=1
IDEA_TEST_KEEP_TRACES=1
IDEA_TEST_CAPTURE_LLM_IO=1
IDEA_TEST_REPORT_VERBOSITY=3
ENVEOF
  [ -f "$CAMP/$rid.pid" ] || { echo "!!! $rid no pidfile - abort"; exit 1; }
  pgid=$(cat "$CAMP/$rid.pid"); waited=0
  while kill -0 "-$pgid" 2>/dev/null; do
    sleep 15; waited=$((waited+15))
    [ "$waited" -gt 5400 ] && { echo "!!! $rid over 90m - stopping"; scripts/run_campaign.sh --stop "$rid"; exit 1; }
  done
  echo "=== $rid done in ${waited}s, cells: $(ls "$ROOT"/agent/idea_test_results/${rid}_*_r1.json 2>/dev/null | wc -l)/12  [total: $(ls "$ROOT"/agent/idea_test_results/ladder03_*_r1.json 2>/dev/null | wc -l)/336]"
done < /tmp/ladder03_order.txt
echo "=== LADDER03 ALL DONE $(date -Is)"
