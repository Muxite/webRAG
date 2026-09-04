#!/usr/bin/env bash
# Paid ladder extension: gpt-4.1-nano and gpt-5-mini, OpenRouter, same frozen corpus.
# Waits for the local ladder's singleton lock, then runs the determinism probe FIRST (cheapest
# check, ~$0.02) so a seed problem is known before the bulk spend.
set -u
ROOT=/home/muk/projects/webRAG
cd "$ROOT" || exit 1
CAMP="$ROOT/agent/idea_test_results/_campaigns"
ORDER="$1"
BUDGET_STOP="${2:-4.50}"      # hard stop across all paid campaigns, in USD

# keys.env wraps values in double quotes. ConnectorConfig._clean_secret strips those for SEARCH
# keys, but _resolve_llm_api_key only .strip()s whitespace — a quoted LLM key reaches the provider
# verbatim and 401s. Strip here so the campaign env carries a bare key regardless.
# IDEA_TEST_VALIDATION_MODEL must be a VALID OPENROUTER SLUG, not a local one: LLM_PROVIDER is
# global, so the runner pre-flights the validation model against OpenRouter too and aborts the
# whole campaign on "qwen2.5:7b is not a valid model ID". Costing nothing is what makes this safe
# rather than a paid-grader trap: all 22 numeric tasks return None from
# get_llm_validation_function(), and validation.llm_validation is null in 144/144 stored ladder
# cells, so overall_score is deterministic grep either way and stays comparable to the local run.
KEY=$(grep '^OPENROUTER_API_KEY=' services/keys.env | cut -d= -f2- | tr -d '\r' | sed -e 's/^"//' -e 's/"$//')
case "$KEY" in sk-or-v1-*) ;; *) echo "!!! key does not look like an OpenRouter key"; exit 1;; esac
[ -n "$KEY" ] || { echo "!!! no OPENROUTER_API_KEY"; exit 1; }

spent() {   # total USD across every paid cell written so far
  python3 - <<'PY'
import json, glob, os
tot = 0.0
for f in glob.glob('agent/idea_test_results/ladder03_openai_*_r1.json') + \
         glob.glob('agent/idea_test_results/apidet_*_r1.json'):
    if '_report_' in os.path.basename(f):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    tot += ((d.get('execution') or {}).get('observability') or {}).get('cost', {}).get('usd') or 0
print(f"{tot:.4f}")
PY
}

# Wait for the LOCAL LADDER DRIVER to exit, not merely for the lock to be free. The local driver
# releases the lock between its 28 campaigns; grabbing it in that gap makes the local driver's
# next run_campaign.sh call fail and abort the whole sweep. Observed once, at 144/336.
while pgrep -f "run_ladder03.sh" >/dev/null 2>&1; do sleep 30; done
while :; do
  held=$(cat "$CAMP/campaign.lock" 2>/dev/null || true)
  [ -n "$held" ] && kill -0 "$held" 2>/dev/null || break
  sleep 20
done

python3 - "$ORDER" <<'PY' > /tmp/api_order.txt
import json, sys
rows = json.load(open(sys.argv[1]))
rows.sort(key=lambda r: 0 if r[0].startswith("apidet") else 1)   # probe first
for rid, model, host, state in rows:
    print(f"{rid}\t{model}\t{host}\t{state}")
PY

while IFS=$'\t' read -r rid model host state; do
  [ -n "$rid" ] || continue
  if ls "$ROOT"/agent/idea_test_results/${rid}_*_r1.json >/dev/null 2>&1; then
    echo "=== skipping $rid (already has cells)"; continue
  fi
  so_far=$(spent)
  if awk "BEGIN{exit !($so_far >= $BUDGET_STOP)}"; then
    echo "!!! BUDGET STOP: \$$so_far >= \$$BUDGET_STOP — refusing to launch $rid"; exit 2
  fi
  echo "=== launching $rid  model=$model host=$host module=$state  (spent so far \$$so_far)"
  modules=""; [ "$state" = "derive" ] && modules="derive"
  tasks="210,211,212,213,214,215,216,217,218,219,220,221"
  [ "${rid#apidet}" != "$rid" ] && tasks="210,211,212"

  scripts/run_campaign.sh "$rid" <<ENVEOF
SEARCH_PROVIDER=corpus
LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/numeric22
LEDGER_MAX_LIVE_FALLBACKS=0
LEDGER_CONTEXT_FIT=1
LEDGER_ZERO_VISIT_GATE=1
LEDGER_HOST_MODULES=$modules
LLM_SEED=12345
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=$KEY
LLM_API_KEY=$KEY
IDEA_TEST_MODELS=$model
IDEA_TEST_VALIDATION_MODEL=openai/gpt-4.1-nano
MODEL_API_URL=
IDEA_TEST_IDS=$tasks
IDEA_TEST_EXECUTION_VARIANTS=$host
IDEA_TEST_RUNS=1
IDEA_TEST_CONCURRENCY=1
IDEA_TEST_USD_CEILING=0.45
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
    [ "$waited" -gt 3600 ] && { echo "!!! $rid over 60m - stopping"; scripts/run_campaign.sh --stop "$rid"; exit 1; }
  done
  echo "=== $rid done in ${waited}s, cells: $(ls "$ROOT"/agent/idea_test_results/${rid}_*_r1.json 2>/dev/null | wc -l)  [spent \$$(spent)]"
done < /tmp/api_order.txt
echo "=== API LADDER ALL DONE total \$$(spent)"
