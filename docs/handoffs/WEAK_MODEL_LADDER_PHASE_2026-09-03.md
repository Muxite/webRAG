# Weak-model floor + ladder phase — handoff (2026-09-03)

Branch `dagv2-evidence-ledger`, 12 commits on top of `19c28ad2`, tree clean, offline suite
**9079 passed / 19 skipped / 0 failed** (`PYTHONPATH=.:services:agent ./.venv/bin/python -m
pytest -q agent/tests`), run by the coordinator.

Read `docs/DEV_CYCLE.md` first if you are new. This is a phase report.

**A 336-cell campaign was still running when this was written.** §6 tells you how to check on it,
resume it, and analyse it. Nothing else is in flight.

---

## 1. The one thing to carry forward

**`LLM_SEED` never reached `langgraph_react`.** That arm built its own `ChatOpenAI` in
`LangGraphSolver._build_llm` and bypassed `llm_backends`, where the seed is applied. Every other
arm routes through `ConnectorLLM` and was genuinely seeded. Fixed in `adeeffa5`.

Why it matters more than any mechanism in this phase: the repo's standing rule is "set `LLM_SEED`
for any local A/B, verified byte-identical, so `reps=1` suffices". For one arm that was false, and
`ladder02`'s own preregistration cites that justification. Evidence, not inference: two identical
runs of a native-transport model were **0/3 byte-identical before the fix and 15/15 after**.

Statistical shape, which decides what survives: unseeded sampling **inflates variance without
biasing**. So a significant positive survives (it cleared a higher bar than intended); a **null or
negative may be a false negative**; a categorical outcome is robust either way.

**The audit is `docs/analysis/SEEDING_AUDIT_2026-09-03.md`.** Its headline: the affected nulls
include the measurement cited as the rationale for the **project's scope pivot**
(`LEDGER.md:44-52`, blocks `gpu0831`/`gpu0831b`). The pivot's other leg — baselines cannot draw a
risk-coverage curve at all, an output-contract fact — is untouched, and the strategic argument
stands on its own. But the parity premise needs a seeded rerun before it is quoted again.
`project_euglena_ledger_pivot.md` in the memory directory now carries that caveat, because memory
outlives docs.

---

## 2. What shipped

| commit | what | status |
|---|---|---|
| `135068ba` | repeat-refused search now reports `executed=False`, so the existing `max_tool_errors` streak bounds it | live, validated |
| `adeeffa5` | `_build_llm` sends `LLM_SEED` | live, validated 15/15 |
| `15b9bc40` | floor taxonomy + probe-round analysis | docs |
| `7ca915ff` | ladder03 design | docs |
| `46ae4b75`, `e6c0ccda`, `1085f4c0` | seeding audit; search-count re-check; trajectory-chaos resolution | docs |
| `42d32e77` | `compare_arms`/`kpi_dashboard` no longer count the verbosity render as a cell | live |
| `7e1fd75d` | a run that never searched can prove its live-fallback count is 0 | live |
| `6d29b66a` | `cell_mechanism.py`, `run_diff.py`, `ladder_curve.py` + tests | tooling |

**Two planned fixes were wrong as specified, which is why probing first mattered.** "Fix A"
(success must mean executed) was **already shipped** in `0fa6e733` and no stored cell had ever
exercised it — `bughunt01`'s files are newer than the fix but its code was five hours older, which
only the campaign git-SHA stamp could settle. "Fix B" (send `num_ctx` on the langgraph arm) **must
not be built**: ollama's `/v1` shim provably drops every spelling of it, and the shipped
context-fit trim already gives 0/30 capped gemma cells against 12/21 without. If you want real
headroom, the lever is `OLLAMA_CONTEXT_LENGTH` server-side.

---

## 3. The floor is several mechanisms, not one

163 stored cells classified (`scripts/cell_mechanism.py`). **34/34 concordance reproduces
exactly**: reading >=1 status-200 page and scoring above zero agree in every `ladder02` cell.
Retrieval is necessary and **not sufficient** — `BUGHUNT01_RESULT_2026-09-03.md` has phi3 at 7/12
visited, keystone **1/12**.

The cell worth internalising is `zv01_qwen2_5_0_5b_211`: 0 searches, guesses
`wikipedia.org/wiki/<Entity>` from the mandate, both guesses return 200, reads Lake Baikal
**1,642 m** and Lake Tanganyika **1,470 m** correctly — and answers **"the computed sum is
576.5 meters"** (true: 3,112). It scores **0.5**, because coverage credits both operands and only
the keystone fails, and the audit-derived confidence channel says **ANSWER**. The model retrieves
correctly and fabricates the arithmetic, and both the score and the confidence channel reward it.
That is precisely what `derive` recomputes, and it had never been measured on a model that fails
that way.

`tinyllama` is excluded from the ladder as a documented capability boundary: 20 prose / 1
valid_json across a whole run, 0 actions under three prompt shapes and a second loop entirely.

---

## 4. EXTRA THINGS TO CHECK — traps that produced wrong numbers here

Full list with numbers in `docs/analysis/WEAK_MODEL_FLOOR_PROBE_2026-09-03.md` §5. The four that
will bite you again:

1. **"Claimed successful minus timings" is not a bug count.** It counts the search tool's
   unconditional repeat-query dedup, which behaves correctly. My headline "52% of tool calls never
   executed" was wrong; dedup was **80%** of that gap and **100%** of gemma's. Post-fix, count
   `payload.executed is False`.
2. **`execution` is a nesting level, and visit status is at `payload.status`.** Reading either at
   the top level returns nothing and prints as a finding ("the ladder captured no diagnostics").
   I hit both in one session.
3. **Bare `pytest` from the repo root is not the suite.** `.claude/worktrees/` holds **10 full
   repo copies**; a root collection runs the suite eleven times and does not finish in 50 minutes.
   Scope it to `agent/tests` (`DEV_CYCLE.md:38`). Scoped, it takes 2m30s.
4. **Absent is never zero — including when you are the one inferring.** I "fixed" the live-fallback
   gate to treat a search-free cell as a real 0 and broke three tests, because absent telemetry is
   not proof that no search ran. A genuine zero needs a PRESENT, search-free timings block.

### Correction found after the ladder started: the give-up is per INVOCATION, not per cell

The repeat-refusal give-up bounds `max_tool_errors` (3) CONSECUTIVE failed dispatches **within one
`run_tool_loop` invocation**. A cell runs the main pass plus up to three `_run_extension` passes
(`langgraph_solver.py:1661,1688,1710`), and each calls `transport.run(...)` -> a fresh
`run_tool_loop` with `tool_error_streak = 0`. So a CELL can legitimately show more than 3
consecutive failures: `ladder03_phi3_mini_off_lg` task 219 shows a run of **6**, which is two
invocations of 3.

This is not a defect in the fix and not a reason to change it, but it corrects two things stated
earlier: the probe phase's C2 criterion ("worst streak 3 vs bound 3") is a per-invocation bound
that passed on probe3 only because those cells did not take an extension with failures in it, and
the improvement is "at most 3 per pass" rather than "at most 3 per cell". The original pathology —
burning the ENTIRE step budget on refused calls — is still closed, since each pass is bounded.

Open question for whoever revisits this: should the streak persist across extension passes? An
argument either way. A corrective extension is a deliberate second chance, so resetting is
defensible; but a model that fails three calls, gets nudged, and fails three more has not been
bounded in any way a user would recognise.

### Infra failures land as files, so completion is not the same as data

`ladder03_phi3_mini_off_lg` task 218: `infra_failed: True`, one `llm_call` that failed after
**1555s** with zero LLM turns — an ollama stall. The harness classified it correctly, and
`prereg.audit`'s `min_completion_rate` still passes because the FILE landed. A cell can therefore
be complete and scientifically absent at the same time. Watch `max_infra_failed_rate` (set to 0.34
here), not completion alone. Rate at 38 cells: **2.6%**, so this looks like an outlier rather than
a systemic stall; projected wall time at that pace is ~6.7h.

---

## 5. EXTRA THINGS TO LOOK FOR — open and unverified

**Needs a seeded rerun before being cited (all `langgraph_react` nulls):**

- `gpu0831` / `gpu0831b` (`EVIDENCE_STACK_NIGHT_2026-08-31.md:29-31,135-137`) — the pivot's parity
  premise. **Highest priority**, because a standing scope decision rests on it.
- `stall_recovery_gate` "trends positive, n.s." (t=1.10), currently shelved as opt-in.
- `require_finish_tool` "tested NEGATIVE on step-budget cost", currently ruled out.
- `LEDGER.md:171` "treat breadth parity as **established**" — doubly weak: already underpowered
  per-shape (n=6, per-task reliability ~0.11) and built on the unseeded arm. Note the *retirement*
  of "graph collapses on fan-out" is unaffected (that was graph vs `sequential_react`).

**Unresolved, flagged rather than answered:** whether the earlier "author 5-8 genuinely
breadth-shaped tasks" recommendation was ever formally cancelled. Its status is uncertain.

**Corrections to existing numbers:** `search.count` counted result DOCUMENTS at `search_k=6`, so
every pre-`0fa6e733` figure is 6x the real call count. `LEDGER.md:207`'s "4.00 vs 21.85
searches/cell" is really ~0.67 vs ~3.64 (conclusion survives, label does not);
`DAG_V2_HONEST_SCOREBOARD...:193`'s "79.5 searches per cell [...] worth attacking next" is really
~13 and that priority should be re-derived.

**Residual, unfixed by design:** gemma2:2b still spends 26 turns with ~20 dedup observations
interleaved among ~20 real calls. Alternating a real call with a repeat resets the give-up streak
and is unbounded on purpose. Whether it costs anything is unmeasured.

---

## 6. The ladder, and how to pick it up

Design and its six changes over `ladder02`: `docs/analysis/LADDER03_DESIGN_2026-09-03.md`.
7 models x 12 tasks x 2 hosts x 2 module states = **336 cells**, 28 campaigns, corpus replay,
seeded, `reps=1`, **$0**.

```bash
# progress
ls agent/idea_test_results/ladder03_*_r1.json | wc -l           # of 336
ls -t agent/idea_test_results/_campaigns/ladder03_*.log | head  # per-campaign logs, newest first

# is it alive?  (empty or a dead pid means it is not)
pid=$(cat agent/idea_test_results/_campaigns/campaign.lock 2>/dev/null); kill -0 "$pid" 2>/dev/null \
    && echo "running (pgid $pid)" || echo "not running"

# resume from wherever it stopped — campaigns with cells are skipped
setsid nohup bash scripts/run_ladder03.sh \
    agent/idea_test_results/prereg/ladder03_order.json < /dev/null > ladder03.log 2>&1 & disown

# drift self-check — 15 ladder cells duplicate probe3 exactly and must stay byte-identical
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/run_diff.py probe3_ ladder03_ \
    --tasks 210,211,212 --candidate-suffix _off_lg

# completion audit, per campaign (prereg.audit is model-blind, hence one run_id each)
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/prereg.py audit --run-id ladder03_<model>_<state>_<host>

# the result
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/ladder_curve.py        # --split dev is the default
```

**Resuming.** `scripts/run_ladder03.sh` skips any campaign that already has cells, so re-running
it continues from where it stopped, and it waits for the singleton lock rather than racing it. Its
campaign order is `agent/idea_test_results/prereg/ladder03_order.json`; the 28 preregs alongside
carry the full design and `_campaigns/ladder03_*.env` records the exact launch environment (and
git SHA) of every campaign that started. **Never `pkill -f`** — it matches the shell running it;
use `scripts/run_campaign.sh --stop <run_id>`, or kill the driver by process group.

**Do not edit code while it runs.** Round 2 of the probe phase was aborted for exactly this: a
mid-sweep edit would have run different code for a campaign's first models than its last. Analysis
scripts are safe (the runner imports none of them); `agent/app/**` is not. Note also that the
campaign provenance stamp records `git rev-parse --short HEAD` and **cannot see uncommitted work**.

**Reading the result.** Endpoints are categorical — fabrication rate (UNKNOWN -> measured),
provenance coverage, availability, `derive` adoption, replayability — with `overall_score` as an
accuracy guard only. `arm_ranking_claim_permitted` is false at this n; no mean-score ranking.
Holdout tasks 213/217/221 are executed to keep the grid complete but **must not be reported** —
`ladder_curve.py --split dev` drops them, and `--split all` prints a banner saying so.

Known confound, recorded before the run: `sequential_react` receives `num_ctx=32768` and
`langgraph_react` receives none, so the hosts differ in context window as well as loop design. The
primary comparison is within-host and unaffected; any cross-host statement inherits it.

---

## 7. Standing rules that earned their place here

- One `run_id` per condition **and per model**: `LEDGER_HOST_MODULES` never reaches the cfg hash,
  and `prereg.audit` never iterates models, so a shared run_id either overwrites cells or reports
  a 7-model sweep complete when one model landed.
- `IDEA_TEST_VALIDATION_MODEL` defaults to **paid `gpt-5-mini`**. Override it on every "$0" run.
- `risk_coverage.py` silently defaults to `--run-ids barrage24b`. Always pass it, plus
  `--rule graded`.
- Prefer a **schema** guard over a filename guard when loading cells. `module_ab.load_run` requires
  `"execution"` and a parseable `test_id` and was immune to the `_report_v3` bug that hit the two
  filename-guarded loaders.
- An agent reproducing your numbers is not verification. The one used here was told to re-derive
  from scratch, and its decisive finding was then re-derived independently from the raw cells
  before being acted on.
