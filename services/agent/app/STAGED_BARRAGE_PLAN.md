# Staged Barrage Plan — slow start → cost estimate → full run → stat report

> **DONE (2026-07-08).** This plan's phases were executed; the campaign closed out as `barrage24b`
> (38 tests, 1,026 runs, ≈$38). See the banner at the top of `COST_BENCHMARK_HANDOFF.md` and
> [`linkedin_package_38tests_2026-07-08/`](../../../linkedin_package_38tests_2026-07-08/README_LINKEDIN.md)
> for final numbers. Kept below for the phase-by-phase methodology, not current status.

Goal: don't fire the whole matrix blind. Walk up in cheap, abortable stages; confirm the
**early-phase runs say what we expect**; derive a **real cost projection** before committing
the expensive runs; then run the rest and come back with **one full statistical report**.

Live-$ throughout (real OpenRouter). Every stage runs through the **benchmark agent**
(singleton, `concurrency=1`, shared connectors) with a hard spend ceiling. Run recipe:
`project_benchmark_run_recipe` + `COST_BENCHMARK_HANDOFF.md` §4. Measurement layer:
`PRE_BARRAGE_AUDIT.md`.

**The kill-switch on every stage:** `IDEA_TEST_USD_CEILING=<usd>` aborts the matrix once
cumulative measured cost (runtime + offline compiler) crosses it. Set it on EVERY live stage
so a bug can't overspend. `0`/unset = no ceiling (never use unset for unattended runs).

Common env (all live stages):
```
export OPENROUTER_API_KEY=... SEARCH_API_KEY=...   # from services/keys.env (CRLF -> tr -d '\r')
export LLM_PROVIDER=openrouter MODEL_API_URL=https://openrouter.ai/api/v1 CHROMA_URL=http://localhost:8001
export DEFAULT_TIMEOUT=45 DEFAULT_DELAY=2 JITTER_SECONDS=0.5
export IDEA_TEST_CONCURRENCY=1 IDEA_TEST_PARALLEL_ACTION_LIMIT=1   # MANDATORY (shared connectors)
export IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096                        # MANDATORY w/ reference: it emits
                                                                  # reasoning before JSON -> false-dropped otherwise
PYTHONPATH=services:services/agent ./.venv/bin/python -m agent.app.idea_test_runner
```
**Variant tokens (use the `IDEA_TEST_EXECUTION_VARIANTS` axis, NOT bare tooling):** `graph_compiled`,
`sequential_react` (the ReAct baseline — **NOT `sequential`**, which is the dead 0-visit legacy path),
`graph` (native GoT, the "before"), `naive_rag`, `parametric`. `sequential_react` + all baselines are
tier-0-forced by the runner, so the tier sweep only ever applies to `graph_compiled` and `graph`.
Subset under test (discriminating, drop saturated 026/019):
- **Cross-shape (headline):** `050,051,052,053,054` (navigation / chain / breadth / argmax / mixed)
- **Hard tier:** `040,041,042,043,044` (dependent chain / breadth / contradiction / capstone / CVE)

---

## Phase 0 — Free preflight ($0, offline + tiny probe)

No matrix spend. Confirm the rig before paying for anything.
- `PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest -q services/agent/tests services/shared/tests` → green.
- Chroma reachable on `:8001`; `keys.env` loads (CRLF stripped).
- JSON-capability preflight on the roster (`IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096`) — drops any
  model that can't emit `json_mode`. (One tiny probe call/model; effectively free.)
- **Prewarm fixtures** for the subset so every arm later reads identical evidence:
  `scripts/prewarm_fixtures.py` (web fetch only). Then runs use `IDEA_TEST_FIXTURES=replay_strict`
  (miss = fail) so variance is model behavior, not evidence drift. (050–054 are URL-free — prewarm
  on a reference pass, see handoff §ROUND-3 fixture caveat.)
- Author the compiled plans once (cache): `scripts/compile_plans.py --tests 050,...,054`
  (reference author model; cache hits cost nothing thereafter).

GATE 0 → proceed only if suite green + chroma up + fixtures warmed.

---

## Phase 1 — One cheapest cell (~cents). "Does early-phase stuff say what we want?"

Smallest possible live run: the cheapest model, the subset, tier 0, **1 repeat**, the three
variants that must separate.
```
IDEA_TEST_MODELS=openai/gpt-4.1-nano
IDEA_TEST_IDS=050,051,052,053,054
IDEA_TEST_TOOLING=compiled,sequential,minimal           # graph_compiled / sequential_react / parametric
IDEA_TEST_EFFORT_TIERS=0  IDEA_TEST_RUNS=1
IDEA_TEST_FIXTURES=replay_strict
IDEA_TEST_RENDER_DAG=1                                   # emit a DAG png per run to eyeball
IDEA_TEST_USD_CEILING=0.50                               # hard stop
IDEA_TEST_RUN_ID=cal1_<stamp>
```
**Sanity gates (this is the "says what we want" check):**
1. Pipeline completes; every run has a `cost.usd` block (instrumentation live).
2. Discrimination ordering holds: `graph_compiled ≥ sequential_react > parametric` on the chain/
   mixed tests (050/051/054). If parametric ties graph_compiled → the task leaks parametrically
   (revisit, don't spend more).
3. Validators are **not constant** (no 0.44 / constant-0 traps); keystone gates pass on the
   known-good answers.
4. `plan_structure` matches the expected shape per test (chain = singleton waves; 052 = [6,6];
   053 = one wave of 6); DAG PNGs render and look right.
5. Groundedness sane (compiled grounds; native graph notoriously doesn't — that's expected).

Deliverable: **per-run $ and wall-time by (variant, test)** for the cheap model. Read with
`scripts/level_ladder.py --run-id cal1_<stamp>` (now mean±ci) and the per-run JSON `cost.usd`.

GATE 1 → all five sanity checks pass. If any fail, fix offline before spending more.

---

## Phase 2 — Cost calibration ($1–2). Project the full matrix before committing.

Measure the two unknowns Phase 1 didn't: the **reference model's** per-run $ (it's the ~40×
cost driver) and the **tier multiplier**.
```
# 2a: reference, 2 tests, tier 0, 1 repeat  (measures reference per-run $)
IDEA_TEST_MODELS=google/gemini-3.1-pro-preview  IDEA_TEST_IDS=051,052
IDEA_TEST_TOOLING=compiled,sequential  IDEA_TEST_EFFORT_TIERS=0  IDEA_TEST_RUNS=1
IDEA_TEST_USD_CEILING=2.00  IDEA_TEST_RUN_ID=cal2_<stamp>
# 2b: cheap model, 1 test, all tiers  (measures the tier-0→40 cost ramp)
IDEA_TEST_MODELS=openai/gpt-4.1-nano  IDEA_TEST_IDS=052
IDEA_TEST_TOOLING=compiled,sequential  IDEA_TEST_EFFORT_TIERS=0,10,20,40  IDEA_TEST_RUNS=1
IDEA_TEST_USD_CEILING=0.50
```
**Projection worksheet** (fill from cal1/cal2 measured $):
```
full_matrix_$ ≈ Σ_runs  mean_$(variant, model, tier)
runs = engine_variants(3) × tiers(T) × tests(N) × models(M) × repeats(R)
     + baselines(2)       × tier0   × tests(N) × models(M) × repeats(R)
```
Final config (cross-shape N=5, **M=3** models, R=5, T=4): engine(3)×T(4)×N(5)×M(3)×R(5) +
baselines(2)×tier0×N(5)×M(3)×R(5) = **900 + 150 = ~1050 runs** at `concurrency=1` — so wall-time
matters as much as $. Use cal2b's tier ramp to decide whether all four tiers earn their cost or
whether `0,20` suffices (the biggest single multiplier). The 3-model set roughly halves the
~2100-run 5-model matrix.

**DECISION GATE 2** → present the projected `$` and wall-clock to the user. Pick a budget
ceiling and trim dims (tiers, repeats, model count, test count) to fit. Nothing big runs until
this number is approved.

---

## Final model set (3) — the narrative

The full test uses **three** models chosen to dramatize "prune the bad parts of a cheap model,
keep the good": native (self-built) quality is uncorrelated with price and uniformly mediocre,
while compiled quality is flat-high. From the proven 2026-06-15 matrix:

| model | $/task | native graph | graph_compiled | role |
|---|---|---|---|---|
| `openai/gpt-4.1-nano` | $0.0016 (1×) | 0.46 | 0.96 | cheap hero: +0.50, ≈ premium at 1/42 cost |
| `openai/gpt-5-mini` | $0.0144 (9×) | 0.29 | 0.95 | dramatic rescue: worst native, +0.66 |
| `google/gemini-3.1-pro-preview` | $0.0655 (41×) | 0.38 | 0.97 | premium ceiling the cheap models match |

Clean 1×→9×→41× cost ladder; compiled flat 0.95–0.97, native scattered 0.29–0.46. nano +
gemini-3.1-pro are measured directly in calibration; only gpt-5-mini is extrapolated.

## Phase 3 — Tier-0 mini-barrage ($$). Confirm the result holds at small scale.

The three final models, full subset, **tier 0 only**, `RUNS=3`. Cheapest run that still produces
the headline table + significance — a dress rehearsal that de-risks Phase 4.
```
IDEA_TEST_MODELS=openai/gpt-4.1-nano,openai/gpt-5-mini,google/gemini-3.1-pro-preview
IDEA_TEST_IDS=050,051,052,053,054
IDEA_TEST_TOOLING=compiled,sequential,full,partial,minimal
IDEA_TEST_EFFORT_TIERS=0  IDEA_TEST_RUNS=3
IDEA_TEST_FIXTURES=replay_strict  IDEA_TEST_RENDER_DAG=1
IDEA_TEST_USD_CEILING=<from projection>  IDEA_TEST_RUN_ID=mini_<stamp>
```
GATE 3 → `level_ladder.py --run-id mini_<stamp>` shows `graph_compiled` beating
`sequential_react` with **CI-disjoint = sig** on the graph level (as it did on the 2026-06-15
run: 0.923±0.043 vs 0.755±0.055). If the direction or significance flips, stop and diagnose —
don't pay for the tier sweep.

---

## Phase 3b — new-task discrimination + diverse-ground A/B (combined, ~$2)

The new reasoning/math tasks (055–060) and the new framework arm (`diverse_ground` aggregation)
both need a live check before the barrage. One run does both, on the two cheap models where the
lift should show. Run four invocations sharing `IDEA_TEST_RUN_ID=disc_<stamp>`:
```
IDEA_TEST_MODELS=openai/gpt-4.1-nano,openai/gpt-5-mini  IDEA_TEST_IDS=055,056,057,058,059,060
IDEA_TEST_RUNS=3  IDEA_TEST_EFFORT_TIERS=0  IDEA_TEST_FIXTURES=record  IDEA_TEST_RENDER_DAG=1
# 1 parametric (anti-leak floor) · 2 graph (native "before") · 3 graph_compiled single (default) ·
# 4 graph_compiled + diverse_ground (the new arm):
(1) IDEA_TEST_EXECUTION_VARIANTS=parametric
(2) IDEA_TEST_EXECUTION_VARIANTS=graph
(3) IDEA_TEST_EXECUTION_VARIANTS=graph_compiled  IDEA_TEST_COMPILED_AGG_MODE=single
(4) IDEA_TEST_EXECUTION_VARIANTS=graph_compiled  IDEA_TEST_COMPILED_AGG_MODE=diverse_ground  IDEA_TEST_RUN_ID=disc_<stamp>_dg
```
GATES: (a) **discrimination** — each task shows cheap `parametric`/`graph` POOR and
`graph_compiled` HIGH (else fix the task: 055's King/Princeton may leak parametrically → swap
entities). (b) **diverse-ground lift** — `graph_compiled+diverse_ground` ≥ `single` on the
math/reasoning tasks (055/059/060) at acceptable extra $; if it helps, make it the barrage default
for those tasks. `diverse_ground` does N+1 aggregation calls (N scattered candidates + 1 grounded
reranker), so it costs more at the aggregation step — the A/B decides if the lift earns it.

Knob: `IDEA_TEST_COMPILED_AGG_MODE=single|diverse_ground`, candidate count
`IDEA_TEST_COMPILED_AGG_N` (default price-aware, min 3).

---

## Phase 4 — Full matrix ($$$, only after GATE 2 budget approval)

Calibration showed the reference model is ~88% of a naive full-matrix's $48. So run it ASYMMETRICALLY
as two invocations sharing one `IDEA_TEST_RUN_ID` (the report aggregates them as one run): cheap models
get the full variant + tier sweep (cheap, and the native `graph` arm is the differential-lift "before");
the reference gets a trimmed set (no native `graph`, R=1, tier 0) — its ceiling line needs only
`graph_compiled` + `sequential_react` + baselines. This keeps Phase 4 at **~$5–6** instead of ~$48.

```
# 4a — cheap models: full variant set; tier sweep on compiled, native graph tier-0 only
IDEA_TEST_MODELS=openai/gpt-4.1-nano,openai/gpt-5-mini
IDEA_TEST_IDS=<curated reasoning set, see below>
IDEA_TEST_EXECUTION_VARIANTS=graph_compiled  IDEA_TEST_EFFORT_TIERS=0,10,20,40  IDEA_TEST_RUNS=3
  # then a 2nd cheap pass for the tier-0-only arms (native graph + baselines):
IDEA_TEST_EXECUTION_VARIANTS=graph,sequential_react,naive_rag,parametric  IDEA_TEST_EFFORT_TIERS=0 IDEA_TEST_RUNS=3
# 4b — reference: ceiling + decent-baseline only, R=1, tier 0 (NO native graph)
IDEA_TEST_MODELS=google/gemini-3.1-pro-preview
IDEA_TEST_EXECUTION_VARIANTS=graph_compiled,sequential_react,naive_rag,parametric
IDEA_TEST_EFFORT_TIERS=0  IDEA_TEST_RUNS=1  IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096
# all invocations share:
IDEA_TEST_FIXTURES=replay_strict  IDEA_TEST_RENDER_DAG=1  IDEA_TEST_RUN_ID=barrage_<stamp>
IDEA_TEST_USD_CEILING=<approved budget>
```
Resumable in slices: any invocation can be re-run; all share `IDEA_TEST_RUN_ID` (prefix-matched by
`--run-id`). If the ceiling trips, the partial run is still valid — note the missing cells.

**Curated reasoning set (lean toward complex reasoning):** `040,042,051,054,055,056,057,058` (dependent
chains, contradiction/verify, mixed DAG, + the four new reasoning tasks). Keep `050,052,053` as optional
coverage. Drop saturated `026,019`.

---

## Phase 5 — Full statistical report (one deliverable)

Aggregate the single `barrage_<stamp>` run:
- `scripts/level_ladder.py --run-id barrage_<stamp>` — per-level success **mean±ci95** + the
  significance block (`graph_compiled` vs each baseline: Δ, Cohen's d, CI-disjoint verdict).
- `scripts/recovery_curve.py --run-id barrage_<stamp> --size 1920` — square 1920² cost-recovery
  curve with CI error bars + Pareto frontier + crossing report + CSV.
- `scripts/gate_report.py --run-id barrage_<stamp>` — per-model/test score + USD grids
  (hand vs auto compiled).
- DAG gallery from `IDEA_TEST_RENDER_DAG` (one `<result>.dag.png` per run).
- Written summary: headline ($/quality vs reference), per-shape table, where it wins/ties,
  variance, and the honest caveats (parametric leak on capable cheap models; verify-node usage;
  fixture parity).

---

## Cost levers (cheapest → most expensive to add)

| Lever | Multiplier | Note |
|---|---|---|
| Reference model | ~40× cheap | Biggest single $; n=1 may suffice for a ceiling line |
| Effort tiers (0/10/20/40) | up to 4× engine runs | Calibrate in 2b; `0,20` often enough |
| Repeats (R) | linear | 3 → defensible, 5 → tighter CI; baselines need fewer |
| Test count (N) | linear | Drop saturated (026/019); keep discriminators |
| Model count (M) | linear | 4 cheap covers the cheap-tier story |

Stop rules: any `IDEA_TEST_USD_CEILING` trip; discrimination ordering inverts; a validator goes
constant; fixture miss-rate > 0 under `replay_strict` (evidence parity broken).

---

## Calibration results (2026-06-26, $0.37 spent, cap $3, all gates PASS)

Phase 0 PASS (rig sound). Phase 1 PASS: cost instrumentation live on all 15 runs; discrimination holds
(compiled > sequential_react > baseline on 051/053); plan shapes correct; 10 DAG PNGs emitted. Two rig
fixes folded in above: (1) `IDEA_TEST_PREFLIGHT_JSON_TOKENS=4096` mandatory with the reference (else
false-dropped); (2) use `sequential_react`, never bare `sequential` (the latter made 0 page visits and
scored ~0 — a dead arm).

Measured per-run $ (tier 0): nano `graph_compiled` **$0.00118** / `sequential_react`-class $0.00046 /
baseline $0.0001; reference `graph_compiled` **$0.0855** / sequential $0.091. Tier ramp: compiled cost
is ~tier-flat (1.14× across 0→40) and fast (~13s); the tier wall-clock blow-up the calibration saw was
the dead `sequential` arm and disappears once `sequential_react` (tier-0-forced) replaces it.

Cost reality: naive full matrix (all variants × all tiers × R=5, both arms symmetric) ≈ **$48** with the
reference = 88%. The asymmetric Phase 4 above (cheap = full set + tier sweep on compiled; reference =
trimmed, R=1, no native graph) lands at **~$5–6** for the curated reasoning set — within the $10
envelope alongside the new-task discrimination check (~$1) and any framework tuning (~$2–3).
