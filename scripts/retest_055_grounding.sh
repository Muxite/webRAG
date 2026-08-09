#!/usr/bin/env bash
# Turnkey live re-test for the test_055 thin-leaf wrong-grounding fix.
# HANDOFF: run by the `benchmark` agent — NOT by the engine agent. Uses live $ API calls.
#
# What it verifies
# ----------------
# Two engine fixes landed in agent/app/testing/execution_compiled.py:
#   1. _target_entity indirect-pointer guard — a thin leaf whose instruction says
#      "find the author of the novel 'X', then open THAT AUTHOR's page and read ..." no longer
#      grounds its search/page-pick on the NOVEL 'X'; it defers to the LLM query (which names the
#      author) so it lands on the author's page. Fixes chain A/B collapsing to novel/film/template
#      pages (the 100%-repro failure on nano and gpt-5-mini).
#   2. reasoning-model token/effort fix — gpt-5-mini (priced "mid" but a reasoning model) gets the
#      premium thin budget (128) + reasoning_effort=minimal, so hidden reasoning no longer starves
#      the completion budget to content=None / finish_reason=length (the all-UNKNOWN cascade).
#
# Success (per model) = across R runs, the keystone \b119\b appears in the deliverable AND the
# leaves cite the RIGHT pages (Stephen_King / University_of_Maine / F._Scott_Fitzgerald /
# Princeton_University), NOT The_Shining_(film) / The_Great_Gatsby / Template:Infobox_organization.
#
# Runs graph_compiled with thin leaves HARD-PINNED (IDEA_TEST_COMPILED_LEAF_MODE=thin) on BOTH
# models, so nano is tested on the exact thin path that failed (auto-routing would send cheap nano
# to react and hide the regression). Compare against the pre-fix baselines in
# agent/idea_test_results/consol_gpt5mini_*_055_*.json and
# p2_consol_pilot_*_055_openai-gpt-4.1-nano_*.json.
#
# Launch (survives terminal close):
#   setsid nohup scripts/retest_055_grounding.sh > /tmp/retest_055.log 2>&1 &
set -uo pipefail   # NOT -e: a single failing cell must not abort the whole re-test.

REPO=/home/muk/projects/webRAG
cd "$REPO" || exit 1

PY=./.venv/bin/python
export PYTHONPATH=.:services:agent

RUN_ID=retest055grounding
RESULTS_DIR=agent/idea_test_results
OUT_DIR="$RESULTS_DIR/_retest_055"
LOG="$OUT_DIR/driver.log"
mkdir -p "$OUT_DIR"

log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ---- env: keys (CRLF-stripped, surrounding quotes removed) — same recipe as the barrage rig ----
keyval(){ grep -E "^$1=" services/keys.env | cut -d= -f2- | tr -d '\r\n' | sed -E 's/^"(.*)"$/\1/'; }
export OPENROUTER_API_KEY="$(keyval OPENROUTER_API_KEY)"
export SEARCH_API_KEY="$(keyval SEARCH_API_KEY)"

export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1   # MANDATORY: shared connectors
export IDEA_TEST_FIXTURES=off                                    # always live, no cache — validating live grounding
export IDEA_TEST_RUN_ID=$RUN_ID
export IDEA_TEST_EFFORT_TIERS=0
export IDEA_TEST_RUNS="${IDEA_TEST_RUNS:-3}"
export IDEA_TEST_IDS=055
export IDEA_TEST_EXECUTION_VARIANTS=graph_compiled
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
export IDEA_TEST_USD_CEILING="${IDEA_TEST_USD_CEILING:-2.00}"     # backstop per invocation

# The fix under test: thin leaves on BOTH models (pin so cheap nano exercises the failing path).
export IDEA_TEST_COMPILED_LEAF_MODE=thin
export IDEA_TEST_COMPILED_PLAN_SOURCE=hand

MODELS=("openai/gpt-4.1-nano" "openai/gpt-5-mini")

log "retest_055 grounding — models=${MODELS[*]} variant=graph_compiled leaf_mode=thin R=$IDEA_TEST_RUNS fixtures=live"

# NOTE (attribution mode, intentionally kept): this per-model loop runs a SEPARATE runner
# invocation for each model, so it pays a FULL preflight (~40s: plain + JSON gate for the
# model + validation model) PER model — 2x here. That's the price of clean, isolated per-model
# timing attribution (IDEA_TEST_CONCURRENCY=1). For THROUGHPUT (not attribution), prefer a
# SINGLE invocation with IDEA_TEST_MODELS="m1,m2" + IDEA_TEST_CONCURRENCY=N: it preflights
# once (fanned out across the pooled connectors) and overlaps cells. See scripts/throughput_run.sh.
for m in "${MODELS[@]}"; do
  log "RUN model=$m"
  IDEA_TEST_MODELS="$m" "$PY" -m agent.app.idea_test_runner >>"$LOG" 2>&1
  log "DONE model=$m (exit $?)"
done

# ---- post-check: keystone + correct grounding per result JSON ---------------------------------
log "==== GROUNDING CHECK ===="
"$PY" - "$RESULTS_DIR" "$RUN_ID" <<'PYEOF' | tee -a "$LOG"
import glob, json, os, re, sys
rd, run = sys.argv[1], sys.argv[2]
KEYSTONE = re.compile(r"\b119\b")
GOOD = re.compile(r"wiki/(University_of_Maine|Princeton_University|Stephen_King|F\._Scott_Fitzgerald)", re.I)
BAD  = re.compile(r"wiki/(The_Shining|The_Great_Gatsby|Template:)", re.I)
rows = []
for f in sorted(glob.glob(os.path.join(rd, f"{run}_055_*graph_compiled_r*.json"))):
    if f.endswith("summary.json"):
        continue
    try:
        d = json.load(open(f))
    except Exception as exc:
        rows.append((os.path.basename(f), "PARSE_ERR", str(exc))); continue
    ex = d.get("execution", d)
    deliv = str((ex.get("output") or {}).get("final_deliverable") or "")
    model = d.get("model", "?")
    key = bool(KEYSTONE.search(deliv))
    good = bool(GOOD.search(deliv))
    bad = bool(BAD.search(deliv))
    verdict = "PASS" if (key and good and not bad) else "FAIL"
    rows.append((os.path.basename(f), model, f"keystone119={key} good_cite={good} bad_cite={bad} -> {verdict}"))
if not rows:
    print("NO RESULT FILES FOUND — did the runner produce output? Check driver.log.")
for name, model, info in rows:
    print(f"  [{model}] {name}: {info}")
passes = sum(1 for _, _, i in rows if isinstance(i, str) and i.endswith("PASS"))
print(f"\nSUMMARY: {passes}/{len(rows)} runs PASS (keystone 119 present, cites right pages, no novel/template pages).")
print("Expected after the fix: all runs PASS. Any FAIL with bad_cite=True means the leaf still grounded on the novel page.")
PYEOF

log "retest_055 complete — see the GROUNDING CHECK block above and $LOG"
