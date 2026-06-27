#!/usr/bin/env bash
# "More options -> better cheap-model accuracy?" experiment.
# nano on its headroom reasoning tests (059/060/062), R=3, replay fixtures.
# Four arms, each its OWN run_id (graph_compiled filenames would otherwise
# collide), each with a hard per-invocation USD ceiling. Global stop near $1.
set -u
cd /home/muk/projects/webRAG
PY=./.venv/bin/python
export PYTHONPATH=services:services/agent

OUT=/home/muk/projects/webRAG/services/agent/idea_test_results
LOG="$OUT/_optscan/driver.log"
mkdir -p "$OUT/_optscan"

# ---- keys (CRLF-stripped, surrounding quotes removed) ----
keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"

export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1
export IDEA_TEST_FIXTURES=replay
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_RUNS=3
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_MODELS=openai/gpt-4.1-nano
export IDEA_TEST_IDS=059,060,062
export IDEA_TEST_EXECUTION_VARIANTS=graph_compiled
export IDEA_TEST_COMPILED_PLAN_SOURCE=hand        # apples-to-apples leak-free hand plan
export IDEA_TEST_USD_CEILING=0.40                 # per-invocation backstop
export IDEA_TEST_RENDER_DAG=0

log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

spend_for(){ "$PY" - "$1" "$OUT" <<'PYEOF'
import json,glob,sys,os
run,rd=sys.argv[1],sys.argv[2]; tot=0.0
for f in glob.glob(os.path.join(rd,run+'_*.json')):
    if f.endswith('summary.json'): continue
    try: tot+=float(json.load(open(f))['execution']['observability']['cost']['usd'])
    except Exception: pass
print(f"{tot:.4f}")
PYEOF
}
total_spend(){ awk 'BEGIN{s=0}{s+=$1}END{printf "%.4f",s}' <<EOF
$(spend_for optscan_a_base)
$(spend_for optscan_b_dg9)
$(spend_for optscan_c_dg21)
$(spend_for optscan_d_thin21)
EOF
}

GLOBAL_CAP=1.00
run_arm(){  # runid  human-desc  [extra env assignments...]
  local runid="$1"; shift
  local desc="$1"; shift
  local cur; cur="$(total_spend)"
  if awk -v c="$cur" -v cap="$GLOBAL_CAP" 'BEGIN{exit !(c+0>=cap+0)}'; then
    log "STOP before $runid ($desc): cumulative \$$cur >= cap \$$GLOBAL_CAP"; return 0
  fi
  log "RUN  $runid ($desc) | cumulative=\$$cur | extra: $*"
  env IDEA_TEST_RUN_ID="$runid" "$@" "$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
  log "DONE $runid (exit $?) | arm_spend=\$$(spend_for "$runid") | cumulative=\$$(total_spend)"
}

log "==== OPTIONS-SCAN START (nano, tests 059/060/062, R=3, replay) ===="

# A — baseline: react leaf, single aggregation (== the barrage config)
run_arm optscan_a_base   "react/single (baseline)"           IDEA_TEST_COMPILED_LEAF_MODE=react  IDEA_TEST_COMPILED_AGG_MODE=single
# B — diverse_ground aggregation, 9 candidate derivations
run_arm optscan_b_dg9    "diverse_ground AGG_N=9"             IDEA_TEST_COMPILED_LEAF_MODE=react  IDEA_TEST_COMPILED_AGG_MODE=diverse_ground IDEA_TEST_COMPILED_AGG_N=9
# C — diverse_ground aggregation, 21 candidate derivations (HUGE options @ aggregation)
run_arm optscan_c_dg21   "diverse_ground AGG_N=21"            IDEA_TEST_COMPILED_LEAF_MODE=react  IDEA_TEST_COMPILED_AGG_MODE=diverse_ground IDEA_TEST_COMPILED_AGG_N=21
# D — thin leaf, 21-sample voting (HUGE options @ extraction)
run_arm optscan_d_thin21 "thin leaf VOTES=21"                 IDEA_TEST_COMPILED_LEAF_MODE=thin   IDEA_TEST_COMPILED_AGG_MODE=single         IDEA_TEST_COMPILED_VOTES=21

log "==== OPTIONS-SCAN END | total spend=\$$(total_spend) ===="
log "---- per-arm score grids (gate_report) ----"
for rid in optscan_a_base optscan_b_dg9 optscan_c_dg21 optscan_d_thin21; do
  echo "===== $rid =====" | tee -a "$LOG"
  "$PY" scripts/gate_report.py --run-id "$rid" 2>>"$LOG" | tee -a "$LOG"
done
log "==== REPORT DONE ===="
