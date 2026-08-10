#!/usr/bin/env bash
# Unattended continuation of the cost-recovery barrage (run_id=barrage24b).
#
# Survives terminal close (launch via: setsid nohup scripts/barrage_continue.sh &).
# - concurrency=1 (shared connectors) — MANDATORY for the benchmark rig.
# - replay fixtures (serve cache hits; live-fetch + SAVE misses) -> robust + self-warming.
# - Hard kill-switches: per-invocation IDEA_TEST_USD_CEILING + driver-enforced global cap.
# - Logs everything (driver.log) and rebuilds the aggregated reports after EVERY test.
#
# Scope (continuing barrage24b):
#   Phase A  complete the reference (gemini) graph_compiled CEILING on the 6 done tests.
#   Phase B  full matrix on 7 more previously-validated tests.
# Matrix per test:
#   nano       : graph, graph_compiled, naive_rag, parametric, sequential_react   (R=3)
#   gpt-5-mini : graph_compiled, sequential_react                                  (R=3)
#   gemini ref : graph_compiled, sequential_react                                  (R=3)
set -uo pipefail   # NOT -e: a single failing cell must not kill the whole barrage.

REPO=/home/muk/projects/webRAG
cd "$REPO" || exit 1

RUN_ID=barrage24b
RESULTS_DIR=agent/idea_test_results
DRIVER_DIR="$RESULTS_DIR/_driver_${RUN_ID}"
AGG_DIR="$DRIVER_DIR/agg"
LOG="$DRIVER_DIR/driver.log"
STATUS="$DRIVER_DIR/STATUS.md"
LOCK="$DRIVER_DIR/driver.pid"
PY=./.venv/bin/python
export PYTHONPATH=.:services:agent

GLOBAL_USD_CAP=15.00      # driver stops between cells once cumulative barrage24b spend crosses this
PER_INV_CEILING=4.00      # backstop inside a single runner invocation (runaway guard)

DONE_TESTS="050 052 055 057 061 062"            # need ONLY gemini graph_compiled (ceiling gap-fill)
NEW_TESTS="051 053 054 056 058 059 060"         # full matrix

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
# keys.env stores values wrapped in literal double-quotes; strip them or the
# Authorization header is malformed (401). Mirrors scripts/run_pilot.sh.
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

# Expected result filenames for a single-model invocation (variants x repeats).
# The runner maps "/" -> "-" in the model slug.
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

# ---- run one runner invocation (one model-group x variant-set x test) ------
# Idempotent/resumable: skips when every expected result file already exists.
# Sets LAST_NEW (new files produced) and LAST_RAN (1 if the runner was invoked).
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

# Fatal rig/auth guard: the first cell that ACTUALLY RUNS must produce files.
# Zero new files from a real invocation => preflight/auth failure -> abort loudly.
RIG_VERIFIED=0
check_rig(){
  [ "${LAST_RAN:-0}" = 1 ] || return 0          # skipped cell: nothing to verify
  [ "$RIG_VERIFIED" = 1 ] && return 0
  if [ "${LAST_NEW:-0}" -lt 1 ]; then
    log "FATAL: first executed cell produced 0 result files — preflight/auth failure. Aborting."
    CURRENT_STAGE="ABORTED: rig/auth failure (see driver.log)"; write_status
    touch "$DRIVER_DIR/STOPPED"; rm -f "$LOCK"; exit 1
  fi
  RIG_VERIFIED=1
}

# ---- write a human-readable STATUS.md -------------------------------------
write_status(){
  local now; now="$(date -u +%FT%TZ)"
  local spent; spent="$(cum_spend)"
  {
    echo "# Barrage continuation — live status"
    echo
    echo "- run_id: \`$RUN_ID\`"
    echo "- updated: $now"
    echo "- cumulative spend: **\$$spent** / cap \$$GLOBAL_USD_CAP"
    echo "- stage: ${CURRENT_STAGE:-starting}"
    echo "- fixtures: replay · concurrency: 1 · R=3 · tier 0"
    echo "- log: \`$LOG\`  · aggregates: \`$AGG_DIR/\`"
    echo
    echo "## Latest level ladder"
    echo '```'
    cat "$AGG_DIR/level_ladder.txt" 2>/dev/null | head -80
    echo '```'
  } > "$STATUS"
}

# ---- rebuild aggregated reports (called after every test) ------------------
aggregate(){
  log "AGG  rebuilding reports for $RUN_ID"
  "$PY" scripts/level_ladder.py --run-id "$RUN_ID"  > "$AGG_DIR/level_ladder.txt" 2>>"$LOG" || log "  level_ladder FAILED"
  "$PY" scripts/gate_report.py  --run-id "$RUN_ID"  > "$AGG_DIR/gate_report.txt"  2>>"$LOG" || log "  gate_report FAILED"
  "$PY" scripts/recovery_curve.py --run-id "$RUN_ID" --size 1920 \
        --out "$AGG_DIR/recovery_curve.png" --csv "$AGG_DIR/recovery_curve.csv" >>"$LOG" 2>&1 || log "  recovery_curve FAILED"
  write_status
}

finish(){ local why="$1"; log "BARRAGE END ($why) | cumulative=\$$(cum_spend)"; CURRENT_STAGE="ended: $why"; aggregate; rm -f "$LOCK"; }

# ===========================================================================
log "==== BARRAGE CONTINUE START ===="
log "run_id=$RUN_ID  global_cap=\$$GLOBAL_USD_CAP  per_inv_ceiling=\$$PER_INV_CEILING"
log "starting cumulative spend = \$$(cum_spend)"
CURRENT_STAGE="Phase A — reference graph_compiled ceiling gap-fill"
write_status

# ---- Phase A: complete reference ceiling on the 6 done tests ---------------
for t in $DONE_TESTS; do
  if over_budget; then finish "global cap hit before $t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Phase A: gemini graph_compiled $t"
  run_cell "A:$t-gemini-compiled" "google/gemini-3.1-pro-preview" "graph_compiled" "$t"
  check_rig
  aggregate
done

# ---- Phase B: full matrix on 7 previously-validated tests ------------------
for t in $NEW_TESTS; do
  if over_budget; then finish "global cap hit before $t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Phase B: $t (nano full)"
  run_cell "B:$t-nano-full" "openai/gpt-4.1-nano" "graph,graph_compiled,naive_rag,parametric,sequential_react" "$t"
  check_rig

  if over_budget; then finish "global cap hit mid-$t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Phase B: $t (gpt-5-mini pair)"
  run_cell "B:$t-mini-pair" "openai/gpt-5-mini" "graph_compiled,sequential_react" "$t"
  check_rig

  if over_budget; then finish "global cap hit mid-$t"; touch "$DRIVER_DIR/STOPPED"; exit 0; fi
  CURRENT_STAGE="Phase B: $t (gemini ref pair)"
  run_cell "B:$t-gemini-pair" "google/gemini-3.1-pro-preview" "graph_compiled,sequential_react" "$t"
  check_rig

  aggregate
done

touch "$DRIVER_DIR/DONE"
finish "all tests complete"
exit 0
