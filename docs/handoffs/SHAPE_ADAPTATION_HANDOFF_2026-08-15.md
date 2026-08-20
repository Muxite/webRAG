# Handoff: shape adaptation — diagnosis, and why the sensor is the blocker (2026-08-15)

**Read `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` first** (the live benchmark this grew out of),
then this. `CAPABILITY_SPECTRUM_PREREG_2026-08-15.md` holds the design written before any cell ran.

Spend this session: **$1.35 of $5 authorized**, ~525 live cells. Offline suite **4708 passed /
18 skipped**. Nothing committed — 45 files, clean reviewable diff.

---

## 1. Where this session ended up

It started as "run more benchmarks, find where DAG v2 has an edge." It ended somewhere more useful:
a **causal diagnosis of why DAG v2 loses on sequential tasks**, grounded in code and telemetry
rather than score tables.

The goal that emerged is a **recursive, shape-adaptive engine** — each node detects the shape of
its own sub-problem and picks an execution strategy, at any level of the tree, with overhead held
down by automated detection. The 30 open questions framing that work are in
`docs/handoffs/SHAPE_ADAPTATION_OPEN_QUESTIONS.md`, with Q1/Q2/Q18/Q20 already answered inline and
Q12 marked in-flight.

**The blocker is not the strategy layer. It is the sensor.** Two independent shape detectors exist
and neither works on the shape that matters.

---

## 2. The diagnosis (all verified this session, evidence inline)

### 2.1 The static classifier exists and is ~unused

`agent/app/idea_policies/shape_classifier.py` exposes `classify_shape(mandate)` →
`chain` / `parallel_merge` / `branch_eliminate`, and `classify_answer_shape()` →
`single_value` / `argmax` / `count` / `computation` / `disambiguation`. Deterministic, free,
keyword-driven.

It feeds exactly one consumer: selecting a reasoning-rules prompt block. `expansion.py:506` states
*"Today only `branch_eliminate`"* has a rule file. **A task classified `chain` has that verdict
discarded.** Nothing schedules on it.

**Measured accuracy** (ground truth from docstring headers, 157 tasks):

| ground truth ↓ / prediction → | chain | parallel_merge | branch_eliminate | None | recall |
|---|---|---|---|---|---|
| chain (31) | 6 | 3 | 6 | 16 | **19%** |
| parallel_merge (25) | 0 | 1 | 20 | 4 | **4%** |
| branch_eliminate (29) | 0 | 0 | 11 | 18 | 38% |
| fanout (16) | 0 | 0 | 0 | 16 | — |

It returns `None` for **70% of all tasks** (fails open by design). On the **10 hand-verified chain
tasks** used in the live chain experiment it labelled **1/10** correctly (only 065).

*Caveat:* the docstring-derived ground truth is itself heuristic. The **1/10** figure is the
trustworthy one — those ten docstrings were read individually and confirmed chain-without-fanout.

### 2.2 The runtime detector cannot detect a chain

`idea_engine.py:1579` → `idea_sequencing.detect_state_dependencies()`. Reading the source, it
returns True on exactly two conditions:

- **A:** a `search` sibling exists **and** a `visit` sibling whose URL is not a valid `http(s)` string.
- **B:** a visit node carries `requires_data: {source_node_id: <a sibling id>}`.

Condition A is a **tooling** dependency ("I need a URL before I can visit"). It is not a
**semantic** one ("I need hop 1's *answer* before I can even formulate hop 2's query").
Condition B is the real dataflow mechanism, and it requires the *model* to emit a `requires_data`
dict naming a sibling's UUID.

**Measured across 476 cell logs this session:**

| path | firings |
|---|---|
| Condition B (`requires_data` dataflow) | **0** |
| the string `requires_data` appearing in any log | **0** |
| Condition A (search + visit + missing URL) | 117 |

**The general dataflow mechanism is dead code in practice.**

### 2.3 The consequence, measured

On the 54 chain-task cells: `AUTO-PARALLEL` fired **58** times, `Forcing sequential` **20** times;
dependency detection fired in **11 / 54** cells. On tasks that are definitionally strict sequential
chains, the engine parallel-batches hops into one step, so hop 2's query cannot contain hop 1's
answer.

**Causal chain:** static classifier misses the shape (1/10) → runtime detector's dataflow path never
fires (0/476) → `AUTO-PARALLEL` batches the hops → depth-1 execution → chain fails.

This is the mechanism behind the benchmark result that `graph:baseline` loses to a plain linear
ReAct loop **6W/10T/13L on chains** while being level everywhere else (4W/3T/2L parallel,
6W/3T/6L other).

### 2.4 Cost attribution (Q18)

Decisions per cell by stage, 245 cells with telemetry:

| stage | baseline | good_adaptive | added |
|---|---|---|---|
| action | 2.9 | 7.8 | **+4.9** |
| reexpand | 0.0 | 3.2 | **+3.2** |
| selection | 2.0 | 4.4 | +2.5 |
| enforce | 1.1 | 3.4 | +2.3 |
| expansion | 1.0 | 3.2 | +2.2 |
| evaluation | 2.0 | 3.5 | +1.5 |
| grounding | 1.2 | 1.2 | −0.1 |
| finalize | 1.0 | 1.0 | +0.0 |

Totals: 8.6 → 22.7 decisions/cell, 23,317 → 68,148 prompt tokens. `grounding` and `finalize` are
flat; everything else roughly triples. **This is the overhead budget any shape-router must fit
inside.**

### 2.5 The inert-mechanism list, tested (Q20)

Confirmed **genuinely inert** — 0 mentions across 476 cell logs: `backtrack`, plan-library
retrieval, early-exit, narrative exemplars.

**Partly refuted:** the inherited list conflated two mechanisms. The confidence-triggered
*re-grounding* is very much alive — across **580 judgements**, mean 0.664, min 0.0, max 1.0, and
**32.8% fall below** the `got_step_confidence_reexpand_threshold` of 0.5, so it fires regularly.
The recorded **AUC 0.571** is a claim about whether that signal is *informative*, which this data
cannot address (it needs per-step ground truth). Open question, separate experiment.

**Untested:** the candidate-coverage gate. An earlier grep appearing to show it firing 212 times was
actually matching `chain_coverage` **validator** output, which is a different thing.

---

## 3. Q12 — RUN-COMPLETE: the flag is a real cause, not the whole cause

`auto_parallel_siblings: false` on the 9 chain tasks (`csnopar_g`, 54/54 cells, $0.32). Override
verified live: settings source = the override file, `AUTO-PARALLEL` firings = **0**.

**Primary metric is `chain_coverage`**, not `overall_score` — it is a validator field
(`validation.grep_validations[].check == "chain_coverage"`) reporting *waypoints traversed*
("2/3 chain waypoints traversed"). It isolates sequential execution instead of routing through a
keystone gate that makes scores near-binary.

**Paired on (model, task, arm), n=36:**

| metric | OFF | ON | delta | W/T/L |
|---|---|---|---|---|
| `chain_coverage` | 0.495 | 0.417 | **+0.079** | 9 / 24 / 3 |
| `overall_score` | 0.397 | 0.331 | +0.066 | 15 / 13 / 8 |

| by arm (`chain_coverage`) | OFF | ON | W/T/L |
|---|---|---|---|
| `baseline` | 0.398 | 0.315 | **3 / 15 / 0** — zero losses |
| `good_adaptive` | 0.593 | 0.519 | 6 / 9 / 3 |

**Paired against the linear loop on chains, n=17:**

| config | `chain_coverage` | vs `seq_react` 0.493 | `overall` | W/T/L (cov) |
|---|---|---|---|---|
| `graph_ON:good_adaptive` | 0.510 | tie | 0.402 | 4 / 9 / 4 |
| **`graph_OFF:good_adaptive`** | **0.608** | **wins** | 0.452 | **7 / 8 / 2** |
| `graph_ON:baseline` | 0.333 | loses | 0.286 | 0 / 9 / 8 |
| `graph_OFF:baseline` | 0.422 | still loses | 0.382 | 2 / 9 / 6 |

`graph_OFF:good_adaptive` vs the current default `graph_ON:baseline`: **11W / 7T / 0L**.

### What this establishes

1. **The flag is causally implicated.** One boolean moves chain traversal +0.079 paired, with zero
   losses in the baseline arm.
2. **It flips the comparison to linear — on traversal.** Batching on, adaptive tied `seq_react`
   (0.510 vs 0.493). Batching off, adaptive **wins** (0.608 vs 0.493, 7W/2L).
3. **Traversal does not convert to answer quality.** Overall score still trails linear
   (0.452 vs 0.464) despite walking materially more of the chain. **The remaining deficit is not
   scheduling — it is extraction/finalize**, turning traversed waypoints into the keystone value.
4. The bare scaffold still loses to linear with the flag off (2W/6L), so the flag was never the
   whole story.

**Do not simply flip the default.** This was measured on 9 chain tasks only. Chains are ~10 of the
active 59; the flag exists to parallelize fan-out, where batching should help and was not tested
here. A shape-conditional setting is what the evidence supports, which is exactly what §A's broken
sensor cannot currently provide.

---

## 4. What to do next, in order

1. **Chase the traversal-vs-score gap (new, from §3).** The engine now walks more of the chain and
   still doesn't answer better. Isolate where it's lost: are the waypoints present in the finalize
   prompt but not in the answer, or absent from the prompt entirely? Dump a `graph_OFF` cell's
   finalize input and check whether the traversed values are in it. Free, and it is now the largest
   unexplained deficit.
2. **Instrument `detect_state_dependencies`** to log what it inspected and why it returned False on
   known-chain tasks (§2.2). Free. This is the direct path to a working sensor.
3. **Decide what a real dependency signal looks like.** Condition B's design (model emits
   `requires_data`) has 0/476 empirical support. The alternative worth evaluating is *dataflow on
   the emitted plan* — does candidate B's query text contain a slot only fillable by A's output —
   which is analysis, not a model judgement. Workflow engines (Airflow/Dagster/Prefect) resolve DAG
   edges from declared inputs/outputs rather than from prose; that is the comparison class.
4. **Only then** consider recursion (per-node re-planning). It is blocked on §2: a recursive node
   has nothing reliable to route on until the sensor works.

---

## 5. Corrections made this session — read these before trusting older docs

Five measurement traps and one framing error, all recorded in
`CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md`:

1. A clean thesis-supporting result (adaptive lift decreasing with model capability) published at
   n=6 and **retracted at n=10** — the whole reversal was one task where the keystone gate zeroed a
   run that had visited both correct pages.
2. `seq_react` appearing to beat the graph engine outright — actually task-coverage mismatch.
3. `langgraph@60` appearing to regress — actually completion-order skew from instant 404s.
4. A local capability-crossover hypothesis — did not survive task-matching.
5. Asymmetric infra-quarantine rates by arm — tested, came back symmetric, conclusions unaffected.
6. **Framing error, and the most dangerous:** chains were called "the graph engine's best case."
   A chain is a *path* — a degenerate graph. It is the **linear loop's** best case. Every number was
   correct; the frame was not, and no amount of pairing would have caught it.

**Standing contract adopted:** every arm comparison is **paired by (model, task) and
run-complete**. Unpaired or partial means pointed the wrong way four times in one session.

---

## 6. Code changed (uncommitted)

- **`agent/app/idea_policies/expansion.py`** — real bugfix. A truthy-but-wrong-shaped `details`
  (weak models emit a list where an object belongs) reached `dict(details)` and raised, killing the
  **whole** expansion step → action-less fallback node → 0 tool calls → grounding gate → score 0,
  reported as `ok` with no error anywhere. Same bug class already guarded 8 lines above for `meta`.
  Regression test `agent/tests/expansion_malformed_details_test.py` (7 cases) fails 6/7 before, passes after.
- **`scripts/adaptive_ladder_run.py`** — three new axes: `capspec_api`, `capspec_local`,
  `capspec_chain`.
- **`scripts/capspec_report.py`**, **`capspec_chain_report.py`**, **`capspec_tool_probe.py`** — new
  analysis scripts, byte-compiled and verified from repo root.

**Note for any live run:** `cell_env()` sets `IDEA_TEST_PREFLIGHT_JSON=0` **only for local cells**;
API cells keep the gate on, which silently drops weak API models before inference (rc=0, no result
JSON, logged as `ok`). Export `IDEA_TEST_PREFLIGHT_JSON=0` for both axes or the axes are not
comparable. This is unfixed in the driver.

---

## 7. Addendum 2026-08-16 — the scope-mismatch bug, and a strategy for edge beyond badmodels

Full detail and citations for everything below are in
`SHAPE_ADAPTATION_OPEN_QUESTIONS.md` §I–§M (Q36–Q50), added this pass after a full-file code audit of
the classifier, both detectors, the auto-parallel gate, the confidence judge, reasoning-rule injection,
and the LangGraph comparison arm.

### 7.1 The bug §2.2/§3 item 3 didn't quite name

`detect_state_dependencies` Condition B was previously described as "the model doesn't emit
`requires_data`." The precise mechanism is narrower and more actionable: the one writer that's always
in the code path (`expansion.py:1396-1424`, URL back-fill) always writes an **ancestor**-scoped
`source_node_id` (it walks `path_to_root`, and explicitly excludes self-reference to avoid deadlock),
while Condition B only checks **sibling** membership (`source_node_id in candidate_ids`). An ancestor
can never be a sibling — Condition B fails by construction on this writer's output, independent of
task, model, or prompt. This isn't "the signal is noisy," it's "the signal and the check don't share a
domain." See §I / Q36 for the one-line candidate fix (widen the check to ancestor-inclusive) and Q37
for a same-cost alternative (a coreference/slot heuristic over candidate query text, sitting in the
already-existing per-candidate parse loop).

### 7.2 Detection is free; the budget is spent elsewhere

Both `classify_shape` and `detect_state_dependencies` are pure string/structural checks — no LLM call,
no extra traversal, ~\$0 marginal cost, confirmed by this pass's code audit. The overhead this doc's
target architecture is bounded by is spent in the surrounding machinery: `good_adaptive` runs
8.6 → 22.7 decisions/cell vs. baseline (§2.4), and every one of those added decisions is re-expansion,
confidence-judging, selection, or enforcement — never the detectors themselves. Practical consequence:
there is real headroom to make detection *more accurate* (per-node instead of root-only, wider
vocabulary, a coreference heuristic) without touching the overhead budget at all — see Q38/Q48. The
harder, better-targeted lever is making the calls *already* being paid for (the step-confidence judge)
carry a second signal, rather than adding new call sites — Q39/Q40.

### 7.3 The framing correction this pass surfaced

§5 item 6 already corrected "chains are the graph's best case" once. This pass surfaces a second,
related framing risk: **"DAG v2 loses everywhere except badmodels" is currently supported by 9 paired
cells behind its architecturally strongest case** (branch/parallel-merge shapes with real fan-out and a
join — Q29, restated as Q44). That is under any reasonable power threshold. The honest current state is
narrower than the framing suggests: DAG v2 has a well-characterized loss on chains (its worst case, a
degenerate graph with nothing to exploit — §5 item 6) and an unmeasured, not measured-and-losing, best
case. "Having an edge beyond badmodels" is therefore two separable jobs, not one:
1. **Stop losing on the worst case.** §7.1's fix plus Q38/Q41 (per-node classification feeding the
   reasoning-rules injection that currently only exists for `branch_eliminate`) target this directly —
   the goal is chains degrading gracefully to sequential-loop parity, not chains becoming a strength.
2. **Actually measure the best case.** Q44's properly-powered branch/parallel run is unrun. Q45 sharpens
   it further: `branch_eliminate` is the one shape where the classifier's recall (38%, best of three),
   the candidate-coverage verification gate, and the reasoning-rules content all already exist and
   already line up — if DAG v2 has an edge anywhere, this is the most likely place to find it first, and
   it hasn't been isolated from the pooled results yet.
3. **A third, structurally distinct edge candidate:** recovery from a bad early step (the step-confidence
   judge + re-expansion) is a mechanism the LangGraph comparison arm structurally cannot do at all
   (`create_react_agent` has no mid-run restructuring — confirmed, no conditional-edge or `Send` usage
   anywhere in `langgraph_solver.py`). This has never been isolated as its own benchmark category
   (Q46) — today it's folded into shape-pooled scores where a chain task's loss can mask it.

### 7.4 What "high adaptability, limited overhead" concretely means, revised

Given 7.2, prioritize in this order (near-zero cost first, each gated on the previous landing before
the next is worth attempting):

1. **Fix the scope bug (Q36) or add the slot heuristic (Q37).** One-line change or a parse-time check;
   testable immediately against the existing 476-cell log corpus with no new live spend.
2. **Author `chain.md` / `parallel_merge.md` reasoning-rules content (Q41).** Currently-computed,
   currently-discarded classifier verdicts start being used. Zero new LLM calls, marginally more prompt
   tokens.
3. **Move `classify_shape` to per-node call sites (Q38), widen its vocabulary.** Same free classifier,
   more/better inputs.
4. **Extend the step-confidence judge's schema with a retrospective dependency field (Q39), feed it
   back into the batching gate for a subtree's unscheduled siblings (Q40).** Zero marginal LLM calls —
   this is the "spend the existing budget better" lever from 7.2, and the one most consistent with
   "very high adaptability, limited overhead" as a design constraint rather than a slogan.
5. **Only after 1–4 land:** run Q44 (powered branch/parallel benchmark) and Q46 (recovery-from-bad-step
   category) to find out where the edge actually is, rather than continuing to argue from the pooled,
   underpowered numbers currently in `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md`.

---

## 8. Cycle 2026-08-16: items 1–4 attempted, adversarial review changed the scope

Ran §7.4 items 1–4 through the full dev-cycle loop (`docs/DEV_CYCLE.md`, Medium tier — multi-file
subsystem change). The Plan-stage re-diff and a 2-agent adversarial review panel (per Medium tier)
caught real problems in three of the four original designs *before* any code was written, changing
what actually shipped. Offline suite green throughout: 4716 passed, 18 skipped, 0 failed
(`PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/`), all touched files
byte-compiled. No live benchmark run this cycle (not authorized, not needed — nothing here has a
live-benchmark surface on its own).

### Shipped

- **Item 2 (chain/parallel_merge reasoning-rules content).** Authored
  `agent/app/reasoning_rules/chain.md` and `parallel_merge.md`; widened `_RULES_NAMES` in
  `agent/app/idea_policies/expansion.py` to include both. Review found this would break *three*
  existing tests, not the two obviously-named ones — `test_manual_env_overrides_auto_classification`
  silently depended on `"chain"` being an invalid manual override name. All three fixed in
  `agent/tests/shape_classifier_test.py`; two new positive-injection tests added for chain/parallel_merge
  (mirroring the existing branch_eliminate one). `parallel_merge.md`'s content also directly instructs
  against premature `think`-candidate combination before both chains resolve — the same pattern
  review flagged as item 1's false-positive mechanism, so this rule is a small defensive win even with
  item 1 deferred.
- **Item 3 (per-node shape classification), shipped in an additive form, not the original "switch the
  call site" form.** Review traced that switching root-mandate classification to node-local goal text
  would silently regress `branch_eliminate` — the one shape with real supporting infrastructure —
  because per-node action-title text (`"Open the authoritative Wikipedia page for X..."`) uses a
  completely different vocabulary register than root-level task-statement prose, not a length problem
  as the original design assumed. Shipped instead: `_auto_reasoning_rules` now classifies the root
  mandate first (unchanged priority, preserves 100% of current behavior) and only falls back to the
  node's local goal text (new `_node_local_goal` helper) when the root does not resolve to a shape with
  a rule file. Also widened `_CHAIN_PHRASES` with phrases grounded in a real, previously-unmatched task
  (`test_023`'s "requires sequential steps" dialect, distinct from `test_051`/`065`'s already-covered
  phrasing) — verified live against `test_051`, `065`, `055`, `095` to confirm zero reclassification of
  already-correct cases before landing. Four new tests cover the injection paths and the
  root-never-loses-to-local-text invariant.

### Deferred, with reasons (not silently dropped)

- **Item 1 (sibling-dependency detection fix).** The stated premise — Condition B is dead because its
  only writer is ancestor-scoped — was wrong at the "confirmed dead" level, not just narrow: review
  found `agent/app/idea_policies/plan_library.py` and `post_expansion_hooks.py` *also* write
  sibling-scoped `requires_data`, by design, and it works (183 real historical firings in the wider
  `idea_test_results/` corpus, all `type="urls_from_search"`, all with a same-parent sibling
  `source_node_id`). The actual gap is narrower: organic LLM-authored candidates outside those two
  paths have no dependency signal at all. The proposed replacement (a same-batch text-cue heuristic)
  has a demonstrated false-positive mechanism against `parallel_merge` tasks specifically (an
  un-stripped `think` candidate combining two chains early reads as both cue-bearing and
  target-under-specified — exactly the shape item 2's new `parallel_merge.md` rule now discourages the
  model from producing, but that's a mitigant, not a fix to the detector). Shipping a new heuristic
  without a live measurement pass (Q36's own "needs a false-positive-rate check" requirement, never
  run) was judged too risky. **Shipped instead:** `agent/tests/idea_sequencing_test.py` (new, 6 tests) —
  zero prior coverage of this module — locking in Condition A's current behavior, Condition B's real
  *working* case (sibling-scoped `requires_data`, so the plan_library path can't silently regress), and
  Condition B's documented *non-firing* case (ancestor-scoped `requires_data`, confirmed intentional per
  the deadlock-avoidance comment in `expansion.py`, not a bug to "fix" by widening the check without
  first confirming — per Q36 — that doing so expresses real wait-for-this semantics). **Next cycle's Plan
  input:** re-run Q1's confusion-matrix methodology against the *full* historical corpus (not one
  session's 476 cells) to measure how much of the chain-task gap survives once plan_library-driven
  dependencies are accounted for, before designing a new detector for what's left.
- **Item 4 (confidence-judge sibling signal).** The most serious finding this cycle: piggybacking a new
  field onto `judge_step_confidence`'s existing call changes its prompt, and LLM confidence estimates
  are sensitive to prompt changes in unmeasured directions — silently eroding the certified false-stop
  guarantee of `confidence_early_exit.py`'s calibration (`agent/app/confidence_early_exit_calibration.json`),
  which consumes this exact signal and is already live in the very profile (`good_adaptive`) the plan
  proposed enabling this in, with nothing in the plan re-running `scripts/calibrate_confidence_early_exit.py`
  against the changed prompt. Separately, review traced that the benefit is close to absent on the path
  this was meant to target: by the time the judge fires on an auto-parallel batch, every sibling in that
  batch has already completed (`_maybe_judge_step_confidence_batch` runs after the whole
  `asyncio.gather`), so "depends on an unavailable sibling" is moot for exactly the batch it's meant to
  catch; re-expansion adds children to the *judged leaf itself*, never back onto the original suspected
  parent, closing off the other hoped-for benefit too. The only path where this would genuinely help
  (the sequential/`best_first_global` path) was not the plan's stated target. **Not shipped, no
  safety-net tests added** — there's no current behavior to lock in since nothing changed. **Next cycle's
  Plan input:** if this is revisited, it needs either a structurally separate call/field that provably
  can't perturb the existing `confidence` number, or an explicit recalibration step folded into the
  plan — and should be scoped against the sequential path specifically, not auto-parallel.

### Files changed

- `agent/app/reasoning_rules/chain.md`, `parallel_merge.md` — new.
- `agent/app/idea_policies/expansion.py` — `_RULES_NAMES` widened; `_auto_reasoning_rules` takes an
  additive `local_text` fallback param; new `_node_local_goal` helper; stale "today only
  branch_eliminate" comments updated.
- `agent/app/idea_policies/shape_classifier.py` — `_CHAIN_PHRASES` widened (4 new entries); module
  docstring updated to reflect all three shapes now having rule files.
- `agent/tests/shape_classifier_test.py` — 2 tests rewritten (no-injection → positive-injection), 1
  fixed (`test_manual_env_overrides_auto_classification`'s stale invalid-name assumption), 2 new
  (additive-fallback behavior and the root-never-loses-to-local-text invariant).
- `agent/tests/idea_sequencing_test.py` — new, 6 tests, zero prior coverage of this module closed.

Explicitly out of scope for the "limited overhead" framing: any fix that adds a new per-node LLM call
whose only job is shape/dependency detection. Every cheap option above avoids that; if none of them
work, that itself is evidence worth recording (a candidate falsifier in Q30's spirit) before reaching
for a paid detector.
