# The Native Adaptive Engine — How It Works, What Changed, Lessons Learnt

_Last updated 2026-07-11. Companion docs: `ADAPTIVE_DISTILLATION_HANDOFF.md` (research log),
`COST_BENCHMARK_HANDOFF.md` (the compiled-scaffold campaign), the plan
`~/.claude/plans/quiet-scribbling-barto.md` (dev program + research agenda)._

---

## 1. The thesis in one paragraph

The compiled-scaffold campaign **proved** that a cheap model executing a well-structured,
pre-authored DAG plan (`graph_compiled`) reaches ≈ premium quality at a fraction of the cost. But
`graph_compiled` authors the entire plan upfront and executes it blindly — it cannot react to what a
step reveals. **The goal is the native (non-compiled) engine reasoning adaptively mid-run:**
plan → act → **observe the step** → decide the next move (re-expand, backtrack, or stop). `graph_compiled`
is the *teacher* whose lessons we port into the native loop; it is **not** the goal. This document
describes the native engine as it stands after that porting work.

---

## 2. How the interleaved loop works now

The native engine is a Graph-of-Thoughts loop in `idea_engine.py`. One turn:

1. **Expand** (`idea_policies/expansion.py`): an LLM proposes typed candidate leaves
   (search / visit / think / merge / verify), steered by the `expansion_planning_addendum` and an
   auto-selected reasoning-rule (`_auto_reasoning_rules` → `branch_eliminate` today).
2. **Execute a leaf** (`idea_policies/actions.py`, dispatched by `idea_engine.py:_execute_action`):
   `SearchLeafAction` (query → URLs), `VisitLeafAction` (fetch + parse + link-select),
   `ThinkLeafAction`, `MergeLeafAction` (synthesis — where the answer is actually extracted),
   `VerifyLeafAction`.
3. **Observe the step** — every completed leaf funnels through `_apply_action_result`, which runs, in
   order: `_maybe_judge_step_confidence` (a decorrelated LLM judge scoring the step 0–1, sees only the
   task + resolved content, never validators/ground-truth), then `_maybe_confidence_reexpand_batch`,
   then `_maybe_reexpand_leaf`.
4. **Decide the next move** — two independent triggers feed the single, trigger-agnostic
   `_apply_reexpand` mutation:
   - **follow-up detector** (`got_operations.check_needs_followup`): "did this result reveal a
     concrete new thing to investigate?"
   - **confidence trigger** (`_confidence_triggers_reexpand`): "was this step untrustworthy
     (confidence < threshold)?"
   Either can grow real child leaves on the same lineage. A separate **backtrack** mechanism
   (`_run_loop` + `got_operations.should_backtrack`) can abandon a low-scoring dead-end chain,
   and a third outcome — **calibrated early exit** (`_run_loop` +
   `got_operations.should_exit_early`, A6) — stops expanding entirely and goes straight to
   finalize when the run's confidence history clears a threshold *derived* from labelled
   trajectories. Backtrack is checked first (a dead end must be abandoned before the run can
   be called done), then early exit, then the step hook.
5. **Merge / finalize** (`MergeLeafAction`, `idea_finalize.py`): synthesize a provenance-preserving
   answer.

The re-expansion is what makes the engine *adaptive* — it is the concrete "observe then decide to
take another step" mechanism, bounded so it can't run away.

---

## 3. The adaptive mechanisms — all opt-in, all default-OFF

Every mechanism below is gated by a flag whose JSON default preserves current behavior
**byte-for-byte**. The typed views live in `idea_policies/config.py`; the defaults in
`idea_dag_settings.json`. Nothing here changes a default run — they exist to be turned on by the
research agenda and measured.

| Flag (JSON key) | Default | What it does | Added |
|---|---|---|---|
| `got_reexpand_enabled` | `false` | Follow-up-detector re-expansion of a completed leaf | pre-session |
| `got_reexpand_max_iterations` | `1` | Bounds re-expansion cycles **per lineage** (see A2) | pre-session |
| `got_step_confidence_judge_enabled` | `false` | Logs a decorrelated per-step confidence signal (E-valuator substrate) | Phase 6 |
| `got_step_confidence_reexpand_enabled` | `false` | Lets a low-confidence step **drive** re-expansion (A1) | this session |
| `got_step_confidence_reexpand_threshold` | `0.5` | Confidence below this triggers re-expansion | this session |
| `got_backtrack_enabled` | `false` | Abandon a low-scoring dead-end chain | pre-session |
| `expansion_expect_contract_enabled` | `false` | Leaves declare a structured `expect` (measurable output + source) (A4) | this session |
| `native_reasoning_effort_discipline_enabled` | `false` | Reasoning-model micro-prompts get `effort=minimal` + token floor (A3b) | this session |
| `native_reasoning_min_tokens_floor` | `2048` | The anti-starvation token floor | this session |
| `price_tier_param_tiering_enabled` | `false` | Executor token budgets scale by model price tier (A5) | this session |
| `native_vote_k_enabled` | `false` | k-vote finalize: k independent extractions, majority vote (A3c) | C1b |
| `native_vote_k` | `1` | Vote count when the flag above is on (`good_adaptive` uses 3) | C1b |
| `native_confidence_early_exit_enabled` | `false` | **Calibrated high-confidence early exit (A6)** — stop expanding and finalize when the accumulated step-confidence prefix clears a *derived* threshold | A6 |
| `native_confidence_early_exit_margin` | `0.05` | Extra conservatism added on top of the calibrated threshold | A6 |
| `native_confidence_early_exit_min_judged_steps` | `2` | Hard floor: no rule may stop a run on fewer judged steps than this | A6 |

**The "good adaptive agent" configuration** (the thing the research agenda tests) = native `graph`
variant with `got_reexpand_enabled` + `got_step_confidence_judge_enabled` +
`got_step_confidence_reexpand_enabled` on, `got_reexpand_max_iterations ≥ 2`, optionally backtrack,
plus the foresight nudge. Toggle per-run with `IDEA_TEST_GOT_REEXPAND=1`,
`IDEA_TEST_GOT_STEP_CONFIDENCE_JUDGE=1`, `IDEA_TEST_GOT_CONFIDENCE_REEXPAND=1`.

---

## 4. The benchmark & experiment system

- **Variants** (`IDEA_TEST_EXECUTION_VARIANTS`): `graph` (native adaptive), `graph_compiled` (the
  teacher), `sequential_react`, `naive_rag`, `parametric`, `minimal`.
- **Task suite** — 145 tasks in `idea_tests/`:
  - 001–097 baseline/research/branching/tier-5.
  - 098–121 the "extra 24" general mixed (branch-eliminate + chain) breadth tasks.
  - **122–145 the adaptive-targeted 24** (`category="adaptive_targeted"`), purpose-built to
    discriminate the adaptive engine on **accuracy + decision-making at low context usage**, across
    four decision archetypes: **A** branch-eliminate/survivor (122–127), **B** conflicting-source
    reconciliation (128–133), **C** minimal-hop stop/continue chain (134–139), **D** under-grounded
    re-expansion trigger (140–145). Each has a bimodal `keystone` gate (rejects the archetype's trap
    answer <0.75, accepts the correct value) and a leak-free compiled-plan stub; ground truth was
    live-verified at authoring.
- **Metrics per run** (result JSON `observability`): accuracy `validation.overall_score`, cost
  `cost.usd`, **context usage `visit.chars` + `llm.prompt.tokens`**, `decisions`, `grounding`,
  `step_confidence`, `timings`.
- **Analysis**: `scripts/recovery_curve.py` (Pareto + square plot + CROSSES), `scripts/level_ladder.py`
  (per-level CI95 + Cohen's d + strict CI-disjoint significance), `scripts/gate_report.py`,
  `evaluator_pilot.py --source confidence` (FAR/power/PAC on the confidence signal, offline, free).
- **Attribution vs throughput mode** (this session): the runner builds a `connector_pool` sized to
  `IDEA_TEST_CONCURRENCY`, each slot with its own ConnectorLLM/Search/Http/Chroma. So **concurrency=1
  is now a choice, not a hard limit** — keep it for clean per-run timing on a timing-sensitive A/B;
  use `IDEA_TEST_CONCURRENCY=N` (via `scripts/throughput_run.sh`) for raw throughput. Throughput mode
  adds parallel preflight, overlapped chroma warmup, an in-flight LLM semaphore
  (`IDEA_TEST_LLM_MAX_INFLIGHT`, 429 guard), and faster-fail (`IDEA_TEST_LLM_TIMEOUT`, non-retryable
  `finish_reason=length`).

---

## 5. What changed this session (commits on `compiled-scaffold-dag`, `f19ef82..HEAD`)

73 files, +10,637/−88. Offline suite grew 1231 → **1547 passed / 18 skipped**, green throughout.

| Commit | What |
|---|---|
| `dc92a31` | **24 adaptive-targeted tasks 122–145** (48 files, 265 tests, 4 decision archetypes) |
| `62c5c84` | **A1** — close the confidence→action loop (opt-in) + `IDEA_TEST_GOT_CONFIDENCE_REEXPAND` |
| `4353de3` | **A2** — iterative re-expansion actually governs lineage depth (real bug fixed) |
| `b5f4eb6` | **A4** — optional typed `expect`/measurable-output decomposition contract |
| `1829475` | **reasoning-model starvation fix + benchmark throughput mode** |
| `1408f99` | **A3b+A5** — native reasoning-effort discipline + price-tier param tiering |
| `e48736b`, `3153884` | honest retest_055 grounding verdicts (pre-fix FAIL, post-fix PARTIAL PASS) |

### 5.1 A6 — calibrated high-confidence early exit (2026-08-02)

**The gap.** Every confidence mechanism above only ever *adds* compute: a distrusted step
re-expands (A1). Nothing short-circuits an easy, high-confidence mandate — which is where most
of the compute-optimal literature's actual efficiency gain lives. A6 is the symmetric half.

**What shipped.**
- `idea_policies/confidence_early_exit.py` — the shared, pure statistics: prefix statistics
  (`running_min` / `running_mean` / `last`), an exact one-sided **Clopper–Pearson** lower bound
  on stop-set precision, per-timestep threshold certification with a **selectivity guard**, a
  **sequential-consistent** fit (a trajectory stopped at *t* is removed before *t+1* is
  certified) with **Bonferroni** correction across timesteps, rule replay, and a fail-closed
  artifact loader.
- `scripts/calibrate_confidence_early_exit.py` — the driver: scans
  `idea_test_results/*.json`, filters to the regular roster (badmodel-lab runs excluded),
  labels each run `overall_score >= 0.75`, splits deterministically 70/30 by filename hash,
  walks a target ladder, and writes the versioned artifact.
- `confidence_early_exit_calibration.json` — the committed artifact.
- `got_operations.should_exit_early` + the `_run_loop` call site + the three flags in §3.
- `IDEA_TEST_NATIVE_EARLY_EXIT` / `IDEA_TEST_NATIVE_EARLY_EXIT_MARGIN` per-run toggles.

**Why a simplification of E-valuator (arXiv 2512.03109).** Its per-timestep logistic
classifier + density ratio + PAC order-statistic threshold needs *hundreds of trajectories per
split*; we have 354 in total. So the classifier collapses to one scalar prefix statistic chosen
on the fit split only, and the PAC order statistic becomes an exact Clopper–Pearson bound —
the same distribution-free finite-sample guarantee on the false-stop rate, computed the way a
small sample allows. The "when nothing certifies, never stop" behavior is E-valuator's own
`c_α = ∞` degenerate case, kept deliberately.

**The result is a negative one, and it is the honest one.** n = **354** regular-roster
trajectories (fit 260 / holdout 94), base pass rate **0.511**, mean 4.75 judged steps.
**No rung of the target ladder (0.95 → 0.65) certifies a rule.** The highest stop precision
*any* admissible threshold can certify is **0.553** — a +4.2pt lift on a 0.511 base rate,
against a preferred target of 0.90. The shipped artifact therefore has `thresholds: {}` and
`should_exit_early` cannot fire even with the flag on. Raw (uncertified) precision tops out
around 0.70–0.77 at usable coverage, so this is not a bound-tightness problem: **the
step-confidence judge simply is not predictive of eventual success** — the same anti-calibration
that motivated F33's contract-based re-expansion, now measured (prefix-statistic AUC ≈ 0.58 at
t=1, ≤ 0.5 by t=5).

**What that buys.** The mechanism, its flags, its tests and its call site are all in place;
re-running the calibration script is the *only* step needed for a certified rule to go live —
no code change. Re-run it when the corpus grows (post-barrage) or when a better-calibrated
per-step signal exists (e.g. contract satisfaction rather than the judge's number). Two tests
pin the current "certifies nothing" state deliberately, so a future recalibration that *does*
certify has to be acknowledged rather than slipping in silently.

---

## 6. Lessons learnt

1. **One root cause, two symptoms.** The single highest-value change of the session was discovering
   that `llm_backends.simplify_payload()` stripped `reasoning_effort` before the wire. That one bug
   caused *both* the thin-leaf grounding failure *and* the benchmark slowness: reasoning models
   (gpt-5-mini) starved to `content=None`/`finish_reason=length`, wasting 40 of 42 LLM calls at ~13.5s
   each. Fixing it (shared `accepts_reasoning_effort` predicate) delivered **3.8× speedup AND lifted
   grounding 0/6 → 1/3**. *Lesson: when a reasoning model is slow, look for silently-failing calls
   before assuming inference latency.*

2. **Brittle heuristics don't port; principles do.** The compiled path grounds via `_target_entity`,
   a regex that guesses the target entity from instruction text. It broke three different ways on
   phrasing (possessive apostrophe, quoted field-name, indirect pointer). *Lesson: do not port the
   regex to native. Port the reasoning-effort discipline (general) and let the native adaptive loop
   (re-expand when a step lands on the wrong page) do robustly what the upfront regex did fragilely.*

3. **An inert knob is worse than no knob.** `got_reexpand_max_iterations` looked configurable but
   `=2` behaved exactly like `=1` — the childless-guard capped per-node re-expansion at 1, so the
   knob governed nothing. *Lesson: test that a bound actually binds (A2 now proves two real cycles
   occur), not just that the loop terminates.*

4. **Statistical honesty.** Phase-1's R=1 "wins" dissolved to noise at R=3. Nothing here is promoted
   to default. *Lesson: a headline claim needs R≥5, the full score vector (not just the mean), and a
   CI-disjoint separation — and reproduction under fixture-replay.*

5. **Opt-in + default-byte-identical is the discipline that let this move fast.** Ten new mechanisms
   landed without a single default-behavior change, each with a "flag-off preserves behavior" test.
   That is why the suite stayed green across 8 commits and why none of this is risky to keep
   un-promoted while the research runs.

6. **Nothing adaptive is proven yet.** A1–A5 build the machinery; whether the adaptive loop actually
   beats non-adaptive (and whether it closes the graph_compiled gap) is an **open empirical question**
   — that is exactly what Part B of the plan (the research agenda) exists to answer, and it needs live
   $ to run.

7. **A calibration that refuses to certify is a result, not a failure** (A6, §5.1). The tempting move
   was to lower the target until *something* came back — which the data would have obliged, by
   certifying a threshold of 0.0 that stops every run and merely inherits the base rate. The
   selectivity guard exists specifically to make that outcome impossible, and the artifact records
   the measured ceiling (0.553 vs. a 0.511 base rate) so the shortfall is a number rather than a
   shrug. *Lesson: build the mechanism, let the calibration decide whether it may fire, and pin the
   "cannot fire" state in a test so turning it on later is a decision somebody makes on purpose.*

---

## 7. The process (how this was executed)

- **Plan-first, then fan out.** A plan mode pass mapped the native loop, the compiled lessons, and the
  experiment harness (three parallel Explore agents), then wrote a dev program (Part A) + a ~40-question
  research agenda (Part B) for an assistant to run.
- **Parallel authoring, sequential engine edits.** The 24 tasks were written by 4 `task-author` agents
  in parallel (disjoint files); engine phases went through a single `engine-dev` agent sequentially to
  avoid edit races, each phase gated by the full offline suite.
- **Every change through a gate.** Reviewer pass (leak-check, byte-compile, suite) → separate,
  well-scoped commit per phase. No Claude trailer on this branch; `.coverage`/result-JSONs excluded.
- **Live spend was bounded and honest.** Grounding re-verifications ran under a hard USD ceiling
  ($0.041 total), and negative results were recorded faithfully — the grounding fix's first verdict
  was 0/6 FAIL and it was written up as such, then corrected to PARTIAL PASS after the real bugs were
  found and fixed.

---

## 8. Backlog / next

- ~~**A3c** — native k-vote + `strip_approximators` answer aggregation~~ **DONE** (commit `1e7ee2d`,
  "C1b"). Corrected 2026-08-02 — this section was stale; see §3's flag table and
  `RESEARCH_LIBRARY.md`'s `answer_vote.py` entry.
- **Compiled `_target_entity` third bug** (quoted field-name as target) — fix only if the Theme-2
  compiled baseline needs test_055 fully green; otherwise a documented compiled-path limitation.
- **Run the research agenda** (plan Part B) — the actual validation of everything built here. Needs a
  live-$ budget; concurrency>1 throughput mode makes the matrix cheap in wall-clock (the empirical
  speedup number still needs one confirming live run).
