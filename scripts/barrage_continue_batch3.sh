#!/usr/bin/env bash
# Piecemeal continuation of the cost-recovery barrage (run_id=barrage24b), batch 2.
#
# Survives terminal close (launch via: setsid nohup scripts/barrage_continue_batch3.sh &).
# - concurrency=1 (shared connectors) — MANDATORY for the benchmark rig.
# - replay fixtures (serve cache hits; live-fetch + SAVE misses) -> robust + self-warming.
# - Hard kill-switches: per-invocation IDEA_TEST_USD_CEILING + driver-enforced global cap.
# - Logs everything (driver.log) and rebuilds the aggregated reports after EVERY test.
#
# Scope: full matrix on 5 confirmed-discriminating tier-5 tests that so far only had a
# cheap nano-vs-parametric spot check (validate_round2.sh), never sequential_react/mini/gemini.
# Matrix per test:
#   nano       : graph, graph_compiled, naive_rag, parametric, sequential_react   (R=3)
#   gpt-5-mini : graph_compiled, sequential_react                                  (R=3)
#   gemini ref : graph_compiled, sequential_react                                  (R=3)
set -uo pipefail   # NOT -e: a single failing cell must not kill the whole barrage.

REPO=/home/muk/projects/webRAG
cd "$REPO" || exit 1

RUN_ID=barrage24b                     # shared run_id -> aggregates with the existing 050-062 data
RESULTS_DIR=agent/idea_test_results
DRIVER_DIR="$RESULTS_DIR/_driver_${RUN_ID}_batch3"
AGG_DIR="$DRIVER_DIR/agg"
LOG="$DRIVER_DIR/driver.log"
STATUS="$DRIVER_DIR/STATUS.md"
LOCK="$DRIVER_DIR/driver.pid"
PY=./.venv/bin/python
export PYTHONPATH=.:services:agent

# barrage24b spent $16.11 through batch 2; cap = that + ~$8 approved for batch 3 (test 069 was rescored offline, no extra spend).
GLOBAL_USD_CAP=25.60
PER_INV_CEILING=3.00      # backstop inside a single runner invocation (runaway guard)

NEW_TESTS="070 072 074 076 078"       # full matrix — confirmed discriminators, never full-matrix tested

mkdir -p "$AGG_DIR"

# ---- single-instance lock -------------------------------------------------
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "Another driver is already running (pid $(cat "$LOCK")). Aborting." >&2
  exit 1
fi
echo $$ > "$LOCK"
rm -f "$DRIVER_DIR/DONE" "$DRIVER_DIR/STOPPED"

log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ---- env: keys (CRLF-stripped, surrounding quotes removed) ----------------
keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"
export SERPER_KEY="$(keyval SERPER_KEY)"

export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=20 DEFAULT_DELAY=2 JITTER_SECONDS=0.5  # F19: 45 was a dead knob (outer 20s action budget always bound first)
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1
export IDEA_TEST_FIXTURES=replay
export IDEA_TEST_RENDER_DAG=1
export IDEA_TEST_RUN_ID=$RUN_ID
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_RUNS=3
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_USD_CEILING=$PER_INV_CEILING

# ---- cumulative spend for this run_id -------------------------------------
cum_spend(){ "$PY" - "$RUN_ID" "$RESULTS_DIR" <<'PYEOF'
import json,glob,sys,os
run,rd=sys.argv[1],sys.argv[2]
tot=0.0
for f in glob.glob(os.path.join(rd,run+'_*.json')):
    if f.endswith('summary.json'): continue
    try: tot+=float(json.load(open(f))['execution']['observability']['cost']['usd'])
    except Exception: pass
print(f"{tot:.4f}")
PYEOF
}

over_budget(){ awk -v c="$(cum_spend)" -v cap="$GLOBAL_USD_CAP" 'BEGIN{exit !(c+0>=cap+0)}'; }

file_count(){ ls "$RESULTS_DIR/${RUN_ID}_"*.json 2>/dev/null | grep -v summary | wc -l; }

cell_complete(){  # model variants ids
  local slug="${1//\//-}" variants="$2" t="$3" v r
  IFS=',' read -ra _V <<< "$variants"
  for v in "${_V[@]}"; do
    for r in $(seq 1 "$IDEA_TEST_RUNS"); do
      [ -f "$RESULTS_DIR/${RUN_ID}_${t}_${slug}_${v}_r${r}.json" ] || return 1
    done
  done
  return 0
}

run_cell(){  # label models variants ids
  local label="$1" models="$2" variants="$3" ids="$4"
  local before after
  if cell_complete "$models" "$variants" "$ids"; then
    log "SKIP $label (all R$IDEA_TEST_RUNS files present)"; LAST_NEW=0; LAST_RAN=0; return 0
  fi
  before="$(file_count)"
  log "RUN  $label | models=$models | variants=$variants | ids=$ids | spent=\$$(cum_spend)"
  IDEA_TEST_MODELS="$models" IDEA_TEST_EXECUTION_VARIANTS="$variants" IDEA_TEST_IDS="$ids" \
    "$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
  after="$(file_count)"
  LAST_NEW=$(( after - before )); LAST_RAN=1
  log "DONE $label (exit $?) | new_files=$LAST_NEW | spent=\$$(cum_spend)"
}

RIG_VERIFIED=0
check_rig(){
  [ "${LAST_RAN:-0}" = 1 ] || return 0
  [ "$RIG_VERIFIED" = 1 ] && return 0
  if [ "${LAST_NEW:-0}" -lt 1 ]; then
    log "FATAL: first executed cell produced 0 result files — preflight/auth failure. Aborting."
    CURRENT_STAGE="ABORTED: rig/auth failure (see driver.log)"; write_status
    touch "$DRIVER_DIR/STOPPED"; rm -f "$LOCK"; exit 1
  fi
  RIG_VERIFIED=1
}

write_status(){
  local now; now="$(date -u +%FT%TZ)"
  local spent; spent="$(cum_spend)"
  {
    echo "# Barrage batch 2 — live status"
    echo
    echo "- run_id: \`$RUN_ID\` (batch3 driver)"
    echo "- updated: $now"
    echo "- cumulative spend (whole run_id): **\$$spent** / cap \$$GLOBAL_USD_CAP"
    echo "- stage: ${CURRENT_STAGE:-starting}"
    echo "- fixtures: replay · concurrency: 1 · R=3 · tier 0"
    echo "- tests this batch: $NEW_TESTS"
    echo "- log: \`$LOG\`  · aggregates: \`$AGG_DIR/\`"
    echo
    echo "## Latest level ladder"
    echo '```'
    cat "$AGG_DIR/level_ladder.txt" 2>/dev/null | head -80
    echo '```'
  } > "$STATUS"
}

aggregate(){
  log "AGG  rebuilding reports for $RUN_ID"
  "$PY" scripts/level_ladder.py --run-id "$RUN_ID"  > "$AGG_DIR/level_ladder.txt" 2>>"$LOG" || log "  level_ladder FAILED"
  "$PY" scripts/gate_report.py  --run-id "$RUN_ID"  > "$AGG_DIR/gate_report.txt"  2>>"$LOG" || log "  gate_report FAILED"
  "$PY" scripts/recovery_curve.py --run-id "$RUN_ID" --size 1920 \
        --out "$AGG_DIR/recovery_curve.png" --csv "$AGG_DIR/recovery_curve.csv" >>"$LOG" 2>&1 || log "  recovery_curve FAILED"
  write_status
}

finish(){ local why="$1"; log "BARRAGE BATCH3 END ($why) | cumulative=\$$(cum_spend)"; CURRENT_STAGE="ended: $why"; aggregate; rm -f "$LOCK"; }

# ===========================================================================
log "==== BARRAGE BATCH3 START ===="
log "run_id=$RUN_ID  global_cap=\$$GLOBAL_USD_CAP  per_inv_ceiling=\$$PER_INV_CEILING  tests=$NEW_TESTS"
log "starting cumulative spend = \$$(cum_spend)"
CURRENT_STAGE="Batch 3: full matrix on 070,072,074,076,078"
write_status

for t in $NEW_TESTS; do
  if over_budget; then finish "global cap hit before $t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Batch2: $t (nano full)"
  run_cell "B2:$t-nano-full" "openai/gpt-4.1-nano" "graph,graph_compiled,naive_rag,parametric,sequential_react" "$t"
  check_rig

  if over_budget; then finish "global cap hit mid-$t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Batch2: $t (gpt-5-mini pair)"
  run_cell "B2:$t-mini-pair" "openai/gpt-5-mini" "graph_compiled,sequential_react" "$t"
  check_rig

  if over_budget; then finish "global cap hit mid-$t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Batch2: $t (gemini ref pair)"
  run_cell "B2:$t-gemini-pair" "google/gemini-3.1-pro-preview" "graph_compiled,sequential_react" "$t"
  check_rig

  aggregate
done

touch "$DRIVER_DIR/DONE"
finish "all tests complete"
exit 0
