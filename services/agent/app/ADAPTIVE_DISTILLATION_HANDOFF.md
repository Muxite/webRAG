# Adaptive Engine + Opus-Exemplar Distillation — Handoff (2026-07-09/10)

Plan: `/home/muk/.claude/plans/a-lot-of-work-gleaming-hejlsberg.md`. Companion:
`COST_BENCHMARK_HANDOFF.md` (the compiled-scaffold campaign this track follows up on).

**Goal:** the closed-out compiled-scaffold campaign proved cheap-model + a pre-authored DAG
plan ≈ premium quality at a fraction of the cost — but on a benchmark suite that's mostly
parallel fan-out/merge. This track asked: can the *native* (non-compiled) engine reason
adaptively mid-run — "explored A, B, C; now I need D" — instead of relying on either an upfront
plan or pure single-shot expansion? And can a reasoning pattern distilled from a strong model
(Claude Code's own Opus, zero API $) lift a cheap model's behavior on genuine multi-hop tasks?

**Bottom line: partially built, honestly inconclusive on the headline hope, one useful negative
finding, nothing promoted to a default.** Total live spend: **$0.70 of the $12 ceiling.**

---

## What shipped (all offline-verified, zero behavior change to defaults)

| Mechanism | Toggle | Where |
|---|---|---|
| Adaptive leaf re-expansion | `IDEA_TEST_GOT_REEXPAND=1` (JSON default stays `false`) | `idea_test_runner.py` env override → `got_reexpand_enabled`; logic in `idea_engine.py:477-585`, `got_operations.py:206-310` (`check_needs_followup`) |
| Reasoning-exemplar injection | `IDEA_TEST_REASONING_EXEMPLAR=chain\|mixed\|parallel` (unset = no-op, byte-identical prompt) | `idea_policies/expansion.py` (`_load_reasoning_exemplar`, cached, injected into the expansion system prompt ahead of the ancestor-path history) |
| Exemplar content | 3 files, fact-free abstract reasoning demonstrations | `services/agent/app/reasoning_exemplars/{chain,mixed,parallel}.md` |

Cleanup done alongside (Phase 0): consolidated 4 overlapping price-tiering functions in
`testing/execution_compiled.py` into shared `_price_tier` calls, fixed 3 `GoTConfig` dataclass
defaults that silently disagreed with the shipped JSON (`improve_enabled`, `backtrack_enabled`,
`backtrack_dead_end_threshold`), deleted a dead `idea_dag_settings.multiround_experiment.json`,
tidied misplaced settings keys. Offline suite: **742 passed / 18 skipped / 3 pre-existing
unrelated failures** (`test_063_strict_csv_validators_test.py`) throughout — no regressions from
any change in this track.

**The native reexpand mechanism itself was audited and found sound** — no bugs, correctly
bounded by `got_reexpand_max_iterations`, doesn't fight the existing light-touch
`expansion_planning_addendum` foresight nudge, doesn't interact badly with (unwired) self-refine.
**Speed gate: GO** — when it fires, it adds a real but small, single-iteration-bounded chunk of
work (test 065: +57% wall-clock, +$0.009), never approached a timeout/step budget across 4 live
smoke runs.

## The Opus-exemplar pipeline works exactly as designed

Three Claude Code Opus subagents (not OpenRouter — **$0 API cost**) each solved one task shape
for real via live WebSearch/WebFetch, verified against ground truth, then distilled a
fact-free/leak-checked abstract reasoning demonstration:
- `chain.md` — sequential dependency-chain hop-following (carry a disambiguating detail forward).
- `mixed.md` — branch-to-eliminate (check every candidate, don't guess the famous one) then
  forward-chain from whichever candidate survives.
- `parallel.md` — independent parallel sub-chains merged by a final computation.

All three were leak-checked (grepped for the source task's actual entities/numbers — zero hits)
and their underlying research was verified correct (Neruda/Parral/162m; the Hampshire Avon →
Dorset Stour chain / 1,240 km²; Stephen King/Fitzgerald university years / 119).

## Experimentation results — the honest part

| Cycle | What | Model | R | Result |
|---|---|---|---|---|
| 1 | All 5 tasks, exemplar matched to shape vs none | gpt-4.1-nano | 1 | 051 chain +0.125; 065 chain flat but 4x costlier; 055/061 parallel +0.70/+0.20 (looked like a clean win); **095 mixed −0.10, WORSE** — model under-decomposed even further (fewer nodes/visits than baseline) |
| 2 | Revised `mixed.md` (added explicit anti-early-stop instruction) on 095 alone | gpt-4.1-nano | 1 | **Made it worse still** (0.30→0.09; 25→1 visits) — the model latched onto the exemplar's phase language as a template to pattern-match superficially, not internalize. Stopped iterating rather than keep guessing. |
| 3 | R=3 confirmation of the parallel "win" on 055/061 | gpt-4.1-nano | 3 | **Dissolved to noise.** 055: baseline mean 0.45 (scores 0.80/0.25/0.30) vs exemplar mean 0.667 (1.00/0.80/0.20) — directionally higher but overlapping, worst exemplar run below 2 of 3 baseline runs. 061: both near-ceiling, flat (0.867 vs 0.883). The R=1 "win" was very likely a favorable draw in both directions, not a real effect at this model. |
| 4 | Does the parallel exemplar look better on a stronger cheap model? | gpt-5-mini | 3 | Baseline 0.00/0.15/1.00 (mean 0.383) vs exemplar 0.25/1.00/0.30 (mean 0.517). A **plausible floor-raising pattern** — exemplar arm never fully collapsed to 0.00 — but n=3/arm is too small to call it confirmed; both arms still span 0.25–1.00. |

**Net verdict: not a confirmed win, but not nothing.** The one shape (parallel-chains-merged) that
showed any signal shows a *plausible floor-raising effect* (fewer catastrophic 0.00 failures) on a
moderately capable cheap model, not a mean-lifting effect, and not on the very weakest model. The
mixed/branch-eliminate shape — the task this whole effort most wanted to fix — got *worse* with
exemplar injection on nano, twice, with two different exemplar wordings. Do not adopt either
mechanism as a default based on current evidence.

## Why the mixed exemplar likely backfired (worth reading before retrying)

`test_095`'s task statement itself asks for a 3-part `(a)/(b)/(c)` report. A weak model given an
abstract "Situation/Thought/Action" exemplar on top of that structure appears to pattern-match the
exemplar's *shape* (a confident-sounding staged answer) rather than internalize its *content*
(check every candidate first). Adding a blunter "do not stop early" instruction made this worse,
not better — more meta-instruction gave the weak model more surface structure to mimic instead of
more actual research discipline. This suggests abstract few-shot reasoning demonstrations may not
transfer well to a model this weak on a task this hard; a **concrete worked example** (with a
different, non-overlapping entity so it can't be copied verbatim) or testing on a stronger cheap
model first are the two most promising untried directions.

## Recommended next steps from Phase 1 (superseded in part by Phase 2 below)

1. ~~Raise R to 5-8 on `test_055` with `gpt-5-mini`~~ — deprioritized; Phase 2 pivoted away from
   narrative exemplars toward deterministic gates (see below), per user direction.
2. ~~Try a concrete worked exemplar~~ — not pursued; same pivot.
3. **Do not** re-enable `got_reexpand_enabled` or set a default `IDEA_TEST_REASONING_EXEMPLAR` in
   `idea_dag_settings.json` — both stay opt-in pending stronger evidence. Still true after Phase 2.
4. ~$11.3 of the $12 ceiling remained at the end of Phase 1.

Result JSONs: `services/agent/idea_test_results/cycle{1,2,3,4}_*`.

---

# Phase 2 (2026-07-10): rule mining + code-enforced completeness gates

Plan: same file, updated in place for this phase (`a-lot-of-work-gleaming-hejlsberg.md`).
External research notes consulted during this phase: `RESEARCH_NOTES.md`.

**Reframing:** the user pivoted the target from "cheap OpenRouter tier" to **local-LLM
deployment — no strong model ever in the runtime path**, which changes the objective from
$/quality to reliability-per-weak-model given free/cheap compute (large prompts, more retries,
more self-consistency are all "free" locally). Since Phase 1 showed narrative few-shot exemplars
are unreliable (weak models copy surface structure, not intent), Phase 2 built **deterministic,
code-enforced mechanisms** instead of prompt-only ones: a candidate-coverage completeness gate, a
flat imperative rule checklist (replacing narrative), and a first-draft task-shape classifier.

**Bottom line: infrastructure built and (after real bug-hunting) verified correct; the mechanism
itself did NOT show a statistically consistent score improvement at R=3 on `gpt-4.1-nano`.
Total additional live spend: ~$1.02** (cumulative campaign total ≈$1.02 of $12 — see per-step
breakdown below). **Not adopted as a default.**

## What shipped (all offline-verified)

| Mechanism | Toggle | Where |
|---|---|---|
| Candidate-coverage completeness gate | `got_candidate_coverage_enabled` (JSON default `false`) | `idea_policies/candidate_coverage.py` (new); hooked into `idea_engine.py`'s `_grounding_replan` AND `testing/execution.py`'s parallel benchmark-loop reimplementation (see bug #4 below) |
| Flat imperative rule checklist | `IDEA_TEST_REASONING_RULES=branch_eliminate` (unset = no-op; auto-classifies when unset, see classifier) | `idea_policies/expansion.py`; content at `reasoning_rules/branch_eliminate.md` |
| Deterministic shape classifier | `classify_shape(mandate)` — no toggle, always active as a fallback when `IDEA_TEST_REASONING_RULES` is unset | `idea_policies/shape_classifier.py` (new) — 7/7 accuracy on the validation set (095→branch_eliminate, 065/051→chain, 055/061→parallel_merge, 052/059→None); only `branch_eliminate` has a matching rule file, so chain/parallel_merge classification is correct but currently a no-op |
| New model in roster | `deepseek/deepseek-v4-flash` | `testing/config.py` — confirmed real OpenRouter slug, passed the JSON-mode preflight gate, cheaper than `gpt-4.1-nano` ($0.09/$0.18 per M vs. nano's pricing) |

Offline suite: **785 passed / 18 skipped / 3 pre-existing unrelated failures**
(`test_063_strict_csv_validators_test.py`) at the end of Phase 2 — no regressions introduced.

## Four real bugs found and fixed this phase (worth reading even if the mechanism isn't adopted)

All four are instances or near-misses of the same general class: **a verifier/validator
satisfied by evidence that looks right but isn't proof of actual execution** — see
`RESEARCH_NOTES.md`'s "implicit-as-explicit verification failure" section for the literature name.

1. **Coverage gate matched against the root node's title**, which embeds the full task mandate
   (which names all candidates) — so it was trivially "satisfied" with **zero page visits**,
   fully inert on first live test. Fixed to only credit visit-backed evidence.
2. **The gate could be starved of its own second chance to fire** — the engine's main loop has
   two exit paths (`_grounding_replan` consulted vs. step-budget exhaustion), and the gate was
   only wired into one. Fixed with an unconditional pre-finalize check (annotates
   `candidate_coverage_incomplete`/`candidate_coverage_missing` when it can't force more work).
3. **`test_095`'s own pre-existing `validate_branch_exploration` validator** (authored months
   before this phase) had the identical bug in miniature: credited "branch coverage" from
   narrative text alone, no visit cross-check — a model could score 3/4 with zero visits. Fixed
   to cap credited breadth at actual visit count.
4. **Split-brain between two loop implementations**: the benchmark harness (`testing/execution.py`
   `run_test_execution`) does NOT call `IdeaEngine.run()` — it reimplements the loop inline with
   its own `_grounding_replan` call and its own `build_final_payload`. Bug #2's fix only patched
   `idea_engine.run()`; the benchmark path never got the pre-finalize annotation, so live tests
   showed the gate detecting incompleteness in its logs while the saved result JSON showed nothing.
   Fixed by porting the same annotation block into `execution.py`.

None of these four bugs were found by static review — all four surfaced only by tracing an actual
live result against the code, which is itself the practical argument for the deterministic-gate
approach over trusting an LLM's self-report: **even our own deterministic gate needed three rounds
of bug-fixing before it was trustworthy**, and a fourth bug was a pre-existing benchmark-validator
defect it happened to illuminate.

## Live validation results — the honest part

| Step | What | Model(s) | R | Result |
|---|---|---|---|---|
| 3 (v1, buggy) | Coverage gate on/off, +rules, on test_095 | nano + deepseek-v4-flash | 1 | Looked promising for nano (+0.15) — but the gate was bug #1's inert no-op; result discarded. |
| 3 (v2, bug #1 fixed) | Same matrix | nano + deepseek-v4-flash | 2 | Nano: real gate engagement confirmed (12→28 visits, 0.25→0.40). Deepseek: gate had no effect / possibly hurt; found bug #1's search-snippet loophole affecting deepseek specifically. Flagged as still needing R=3. |
| 4 | Coverage gate on/off, +rules, on test_095, all 3 bugs (#1-3) fixed | nano only | **3** | **Null/noisy.** Scores: baseline mean 0.163 (0.09/0.00/0.40), +gate mean 0.177 (0.35/0.09/0.09), +gate+rules mean 0.190 (0.09/0.18/0.30). No condition consistently beat another per-repeat; visit counts showed no monotonic gate effect (a=[1,0,20], b=[17,1,1], c=[1,2,11]). Baseline's single best run (r3, 20 visits from natural reexpansion) outscored every gated run. `candidate_coverage_incomplete` never appeared in any of the 9 result JSONs despite mid-loop logs showing the gate detecting incompleteness in 2 of them — this is bug #4, found and fixed AFTER this run (not yet re-validated live). |

**Net verdict: the gate infrastructure is now believed correct (four real bugs fixed, all
offline-verified with regression tests), but does not yet show a consistent score improvement for
`gpt-4.1-nano` on the hardest task shape at R=3.** The most likely reason, based on the visit-count
data: forcing a replan when the step/token budget is nearly exhausted doesn't give the model enough
*room* to actually visit more candidates — the gate can detect and now correctly annotate the
problem, but detecting isn't the same as fixing it if there's no budget left to act on the
detection. This was not tested (increasing the step budget specifically when coverage is
incomplete, rather than just annotating) — see next steps.

## Recommended next steps (Phase 2)

1. **Re-validate at R=3 now that bug #4 (annotation) is fixed** — this doesn't change scores, but
   makes it possible to see, per-run, whether `candidate_coverage_incomplete` correlates with low
   scores (confirming the "not enough budget left" hypothesis) before investing in a fix for it.
2. **If the budget-exhaustion hypothesis holds:** try granting extra step/token budget
   specifically when the coverage gate detects incompleteness (rather than just annotating) —
   this is the untested mechanism that could turn "detect the problem" into "fix the problem."
3. **Adopt E-valuator-style sequential testing** (see `RESEARCH_NOTES.md`) instead of brute-force
   R=3 — its headline result (90% of task accuracy recovered at 80% of token budget via early
   termination) is directly applicable to this harness's evaluation cost, and it would have let us
   detect the null result in fewer than 9 runs.
4. **Try `deepseek/deepseek-v4-flash` again** at R=3 (only nano was tested at R=3 this phase,
   prioritized for statistical confidence on one model) — its bug-#1-confounded R=2 data suggested
   a different, search-snippet-based failure mode worth a clean re-test.
5. **Chain and parallel_merge rule files don't exist yet** — the shape classifier correctly
   identifies these shapes but has nothing to inject. Only worth authoring if the branch_eliminate
   gate approach is eventually validated (no point porting an unproven mechanism to more shapes).
6. Do **not** flip `got_candidate_coverage_enabled` to `true` by default — stays opt-in.

**Remaining budget: ~$10.98 of the original $12 ceiling** (Phase 1 spent $0.70; Phase 2 spent
~$0.32 across the DeepSeek preflight probe + 3 rounds of live validation).

Result JSONs: `services/agent/idea_test_results/step3_coverage_gate_*`,
`step3b_coverage_fixed_*`, `step4_r3_fixed_*`.

---

# Phase 3 (2026-07-10): budget extension — RETIRED, decision-gate result

Plan: same file, Priority 1 (`a-lot-of-work-gleaming-hejlsberg.md`). Following external research
(see `RESEARCH_NOTES.md` update) confirming a fixed/capped, evidence-only-triggered budget
extension was the right *design* to try (no incentive gradient toward gaming), Phase 2's next-step
#2 was implemented: `got_candidate_coverage_budget_extension` (default `10` steps, ~20-25% of
`max_steps=50`), granted exactly once when the gate detects incomplete coverage, wired into both
loop implementations (`idea_engine.py` and `testing/execution.py`).

**R=3 result on `test_095`/`gpt-4.1-nano`: still a null result.**

| Condition | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| baseline (reused) | 0.09 | 0.00 | 0.40 | 0.163 |
| +gate +extension | 0.18 | 0.18 | 0.18 | 0.180 |
| +gate +rules +extension | 0.13 | 0.09 | 0.18 | 0.133 |

Neither gated condition consistently beat baseline per-repeat (both lose badly on r3, where
baseline's natural reexpansion happened to drive 20 visits); c's mean is now *worse* than baseline.

**Root cause of why the extension didn't help, found by inspecting the actual run logs**: the
budget grant mechanically worked exactly as designed — all 6 runs logged
`[COVERAGE] Granting one-time +10-step budget extension (max_steps now 60)` — but it was
**functionally inert**. The extension's "re-activate root" trigger doesn't cause the engine to
actually re-expand missing branches, because `idea_engine.py`'s forced-expansion path only fires
when a node has `children == 0`; test_095's root already had all-`done` children, so
reactivation was a no-op and the loop exited on the very next step, consuming only 1 of the 10
granted steps. The lever was pulled but connected to nothing.

**Decision (per the plan's own stop rule — this mechanism has now had 4 bug-fix rounds + 1 design
change without success): retired.** `got_candidate_coverage_enabled` and
`got_candidate_coverage_budget_extension` remain in the codebase, offline-tested, opt-in, JSON
defaults `false`/harmless — but no further engineering investment planned. A real fix would need a
different re-expansion trigger (e.g. force new children even when the node already has
all-`done` children, if coverage is unsatisfied) — out of scope per the stop rule; note here for
whoever revisits this mechanism, don't rediscover the same dead end.

**Total campaign spend: ~$1.05 of $12.** Remaining priorities (2: ConSol/E-valuator adoption, 3:
architecture consolidation, 5: cleanup debt, 6: more chain/mixed tasks) proceed independently of
this mechanism's fate — see the plan file for current status.

---

# Phase 4 (2026-07-10): ConSol sampling pilot — validated with caveats, not broadly wired

Plan: Priority 2 of `a-lot-of-work-gleaming-hejlsberg.md`.

Built an opt-in (`IDEA_TEST_USE_CONSOL=1`) ConSol-based early-stopping wrapper around
`execution_compiled.py`'s `_vote_extract` leaf-extraction voting (`testing/consol_pilot.py`, real
PyPI package `consol` 0.3.0, LLM-agnostic `confidence_models` core — `MsprtConfidenceModel.test(first,
second)` over top-two answer counts). Offline: 802 tests passing, no regressions, graceful fallback
if ConSol isn't installed/errors. `consol` intentionally kept OUT of `requirements.txt` (heavy
langchain/langgraph deps) pending a decision on broader adoption.

**Live pilot (test_055, gpt-4.1-nano, `graph_compiled`+thin leaves, R=5 each condition, ~$0.03
spent):**
- **Answer agreement: trustworthy.** Same winning answers, same score distribution (mean 0.28
  both conditions), same per-leaf failure pattern — ConSol neither fixed nor introduced errors.
  (Incidentally reproduced a pre-existing, unrelated bug: Chain A's thin-leaf extraction
  mis-grounds to a wrong Stephen King alma-mater year in 100% of runs under BOTH conditions — not
  a ConSol issue, noted for whoever picks up thin-leaf extraction quality next.)
- **Cost savings: real but modest.** ~18% fewer LLM calls, ~27% cheaper per run ($0.0023 vs
  $0.0031/run) — early stopping is genuinely pruning convergent leaves below the fixed k=5.
- **Wall-clock: a real trade-off, not a free win.** ~60% SLOWER per run (13.1s vs 8.2s) because
  `consol_vote` samples sequentially to inspect the running tally, while the existing fixed-k path
  uses `asyncio.gather` in parallel. This is a $-for-latency swap.

**Verdict: worth keeping available (opt-in) for offline/batch benchmarking where $ matters more
than wall-clock — NOT recommended for a blanket default or any latency-sensitive path** as built.
A parallel-with-early-cutoff variant (batch a few samples concurrently, check convergence between
batches rather than after every single sample) would fix the wall-clock trade-off but wasn't built
this pass — worth doing before any broader rollout.

**E-valuator not yet piloted** (per the plan, it's gated on its own calibration-transfer validation
before trusting it — bigger lift, deferred to a future session).

**Campaign spend: ~$1.08 of $12.**

---

# Phase 5 (2026-07-10): control-loop consolidation, cleanup debt, E-valuator pilot

Plan: `/home/muk/.claude/plans/plan-next-steps-and-functional-dusk.md` (post-consolidation roadmap;
supersedes the "Priority 1/3" framing in the original plan file — Priority 1 is retired above,
Priority 3 landed this phase).

**Control-loop consolidation (Priority 3, was deferred, now landed):** `idea_engine.py`'s `run()`
and `testing/execution.py`'s `run_test_execution()` — previously two independent reimplementations
of the same step/prune/backtrack/finalize loop (the root cause of Phase 2's bug #4) — are now one
implementation via Strangler Fig: shared `IdeaDagEngine._run_loop()` and `.finalize()`, an explicit
`fail_soft` parameter replacing what was a silent fail-fast/fail-soft divergence between the two
paths, and a dedicated parity suite (`services/agent/tests/control_loop_parity_test.py`). Committed
as `bbea37b`.

**Cleanup debt (Priority 5):**
- Fixed the 3 pre-existing `test_063_strict_csv_validators_test.py` failures — a day-one authoring
  bug (the validators test was written against a phantom element set that never matched the source
  file's real `ENTRIES`), not drift over time. Offline suite: 3 failed → 0.
- Added a config-drift guard test (`services/agent/tests/config_drift_test.py`) asserting every
  `*Config` dataclass group in `idea_policies/config.py` can't silently disagree with
  `idea_dag_settings.json`'s shipped values — the bug class that's hit 3 times previously. No 4th
  real drift found; only benign `None`/`""` "no override" sentinel non-differences.
- Deleted the dead `got_improve_enabled`/`try_improve_node` mechanism entirely (code, settings
  keys, config fields, docs) — confirmed zero call sites, per the campaign's stop-rule discipline
  against carrying unwired/unproven mechanisms indefinitely.

**E-valuator pilot (Priority 2, second half):** piloted the real PyPI package (`e-valuator` v1.0.0,
confirmed to match the paper's math exactly) against this repo's archived benchmark data
(`linkedin_package_38tests_2026-07-08/` + local `idea_test_results/`), $0, no live runs needed.
Used `validation.grep_validations` (an ordered per-check score list + `overall_passed` outcome) as
the best available substitute for the paper's per-timestep verifier-score trace — this repo has no
literal per-execution-step verifier signal recorded anywhere.

**Result: the machinery works, but the substrate can't prove it earns its keep.** On a pooled nano
cell (570 runs) at α=0.2, held-out false-alarm rate was **0.000 across all 4 seeds** (well under
target) with 0.79-0.90 power — but FAR being *exactly* zero, not just under target, is itself the
finding: `grep_validations` scores are what *compute* `overall_score`, which *sets*
`overall_passed`, so the "verifier" is near-deterministic of the label. The paper's actual value
proposition (controlling FAR against a noisy, partially-informative judge, where naive
calibrated-score thresholding fails) is never stressed by data this clean. A single-agent cell
(nano `sequential_react`, 114 runs) hit the paper's own min-calibration-size wall at tighter α
(threshold → +∞, zero power) — a real, useful illustration of the method's stated limitation, not a
bug. Cross-model transfer (nano→gpt-5-mini) also held FAR=0, but for the same weak-substrate reason.

**Verdict: not worth production integration on the current data.** Would only become genuinely
useful if the harness first recorded a real, partially-informative per-step signal decorrelated
from the final grep outcome (e.g. a lightweight LLM-judge confidence per GoT step, logged into
`execution.observability`) — that's an instrumentation task, not a calibration one, and not
attempted this phase. Consistent with the campaign's discipline: sound method, wrong data shape to
prove value here. Code kept for reference (`testing/evaluator_pilot.py`,
`tests/evaluator_pilot_test.py`), package deps in a throwaway scratch venv only, not added to
`services/agent/requirements.txt`.

**`deepseek/deepseek-v4-flash` clean R=3 baseline (Priority 2 leftover):** ran `test_095`
(`tier5_branch_eliminate_chain`), native `graph` variant, coverage gate off, now that the harness is
consolidated and bug-#1/#4-fixed. Scores: **1.00 / 0.00 / 0.27 (mean 0.423)**, ~$0.0148 spent. No
sign of the old coverage-gate confound — the R2 zero-visit run correctly scored 0 straight down the
line (the earlier confound let 0-visit runs pass via root-title text matching; that's gone). R3
illustrates a dependent-check cascade: a partial run (3/4 Avon candidates resolved) still zeroed 3
of 5 checks because the one required keystone-page visit (Dorset Stour) was skipped. Qualitatively
higher than nano's Phase 2 R=3 baseline (mean 0.163) — n=3 only, data-gathering, no decision drawn.
Result JSONs: `services/agent/idea_test_results/20260710_120830_deepseek095_*` (note: this repo has
result JSONs under both `services/agent/idea_test_results/` and a repo-root `idea_test_results/`
path referenced in some older docs — worth reconciling, flagged here rather than silently ignored).

**ConSol batched-early-cutoff variant (Priority 2 wall-clock fix):** added an opt-in
`IDEA_TEST_CONSOL_BATCH` env var to `testing/consol_pilot.py`'s `consol_vote()` — when `>1`, draws
`batch_size` samples concurrently per round via `asyncio.gather` and checks the SPRT stopping
condition once per batch instead of once per sample (default stays `1`, byte-identical to the
existing sequential path, no behavior change unless explicitly opted into). 8 new offline tests
(true-concurrency check, early-stop-across-batches, cap-never-exceeded, error tolerance). Live
3-condition pilot (fixed-k / sequential ConSol / batched ConSol, test_055, gpt-4.1-nano,
`graph_compiled`+thin leaves, R=5 each, ~$0.041 spent): **wall-clock fix validated.** Batched
(batch=2) cut ConSol's overhead over the non-ConSol baseline roughly in half — sequential was 74%
slower than baseline (18.91s vs 10.87s), batched was only 37% slower (14.89s) — a 21.3% wall-clock
reduction from sequential→batched. Cost held: batched was actually the cheapest of the three
conditions this pilot ($0.00246/run vs baseline's $0.00297 and sequential's $0.00283). Answer
agreement/score distribution: sequential and batched statistically indistinguishable (avg score
0.280 both, same shape). **Verdict: validated for opt-in use** (`IDEA_TEST_CONSOL_BATCH=2` alongside
`IDEA_TEST_USE_CONSOL=1`) — single test/model cell, n=5, so directional not a broad statistical
confirmation; a wider matrix would be needed before considering a default-on flip, not recommended
at this evidence level. Default stays sequential (`batch_size=1`) unless explicitly set.

**Offline suite at end of Phase 5 (pre-ConSol-live-pilot): 827 passed, 18 skipped, 0 failed** (up
from 809/18/3 at the start of this phase — the 3 pre-existing failures are now fixed, plus 18 new
tests across the drift guard, E-valuator pilot, and ConSol batching).

**Campaign spend: ~$1.11 of $12** (Phase 5 so far: ~$0.0148 on the deepseek run + ~$0.0173 on the
ConSol cross-model addendum below; everything else this phase was $0).

**Addendum — ConSol batching cross-model check (gpt-5-mini):** to address the "single cell, n=5"
caveat above, re-ran sequential-vs-batched ConSol on `test_055`/`gpt-5-mini` (5 runs each, $0.017
spent). **Wall-clock speedup generalized**: batching cut mean duration 189.0s→133.9s (**−29%**),
consistent in direction with nano's finding, with no cost regression. **But this cell can't validate
answer-agreement**: both conditions hit an identical 0/5 wrong-grounding failure floor (score 0.20
every run, same cascade signature — thin leaves grounded to irrelevant pages, keystone step
correctly reported "unknown" rather than hallucinating) unrelated to ConSol — a pre-existing
gpt-5-mini/thin-leaf issue, not something this pilot introduced. gpt-5-mini also hit a known
`finish_reason=length` retry-storm pathology (reasoning tokens consuming the completion budget)
that inflated both conditions' absolute wall-clock roughly proportionally, so the relative
speedup is probably real but the absolute numbers aren't comparable to nano's baseline. **Net:
directionally consistent second data point for the speedup, but answer-agreement generalization
remains unconfirmed** — would need a cell with actual passing/varying outcomes on a second model to
test that specifically. Still opt-in, default stays sequential.
