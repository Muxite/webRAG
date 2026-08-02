# What `node.score` measures, and why backtrack can never read it

_Last updated 2026-08-02. Analysis only — **nothing in the engine was changed by this
investigation**. Companion docs: `CONFIDENCE_JUDGE_MISCALIBRATION.md` (the *other* judge, same
methodology), `RESEARCH_LIBRARY.md` (the `got_operations.py` entry), `idea_policies/evaluation.py`
(where the number comes from), `got_operations.py` §`should_backtrack` (its only real consumer)._

Reproduce every number here with:

```
PYTHONPATH=services:services/agent ./.venv/bin/python \
  scripts/analyze_evaluation_score_predictive_power.py --samples 4
```

**Corpus:** 261 regular-roster trajectories that record a graph with at least one scored node
(`is_regular_roster` reused verbatim; label = `validation.overall_score >= 0.75`; base rate 0.628),
767 scored nodes, timestamps 2026-05-26 → 2026-07-07. **129** of those are
`execution_variant=graph` — the native GoT loop, the only place `should_backtrack` is reachable —
carrying 389 scored nodes at base rate 0.729; the rest are `sequential`. Intervals are
Hanley–McNeil 95% (maths *imported* from `analyze_confidence_judge_miscalibration.py` so the two
docs are comparable); node-level intervals are optimistic (nodes in one run share a label), so
run-level numbers are the ones to argue from.

**Corpus caveat, read before quoting anything.** This is **not** the same population as
`CONFIDENCE_JUDGE_MISCALIBRATION.md`'s 354 trajectories. That flat top-level
`idea_test_results/*.json` corpus is no longer on disk — only empty per-run directories remain, and
`scripts/analyze_confidence_judge_miscalibration.py` with default arguments now prints *"no usable
trajectories"*. The runs that still carry a recorded `execution.graph` live in per-run
subdirectories and are older, so this script globs recursively. The two docs' AUCs are computed
identically but describe different runs.

---

## 1. The quantified claim

**`node.score` is not a weak predictor of success. It is not a predictor of anything, because it
is not a measurement of anything that happened: it is a pre-execution rating of a *proposed*
action, hard-capped at 0.5, and the graphs it is written onto are one level deep — so the walk
`should_backtrack` performs could never, in any recorded run, reach its trigger.**

**The mechanism could not have fired.** `should_backtrack` counts consecutive nodes below 0.3
along `path_to_root` and triggers at 5. Replaying that walk on every recorded node:

| | all variants (261 runs) | `graph` only (129 runs) |
|---|---|---|
| longest `path_to_root` anywhere | **2 nodes** | **2 nodes** |
| path length of every scored node | 2 (itself + root) | 2 (itself + root) |
| per-run best consecutive-low chain | 0 or **1** | 0 or **1** |
| runs firing the shipped rule (`low<0.3`, `dead_end>=5`) | **0** | **0** |

The counter maxes out at 1 against a trigger of 5. It is not close: **no cell of the
`low_score ∈ {0.2…0.6} × dead_end ∈ {1…5}` sweep with `dead_end >= 2` fires on a single run**, at
any threshold. Every scored node is a direct child of the root, and the root is never scored, so
the walk terminates after one step by construction.

**And at maximum sensitivity it points the wrong way.** `dead_end>=1` is the only setting that
fires at all; the runs it would abandon *pass more often* than the ones it would keep:

| rule | runs fired | pass rate fired | pass rate not fired |
|---|---|---|---|
| `low<0.3, dead_end>=1` (all variants) | 163 / 261 | **0.681** | 0.541 |
| `low<0.3, dead_end>=1` (`graph` only) | 92 / 129 | **0.750** | 0.676 |
| `low<0.5, dead_end>=1` (all variants) | 206 / 261 | **0.694** | 0.382 |

**Run-level AUC vs eventual pass** (one sample per trajectory; `path_running_min` is literally the
statistic `should_backtrack` reads — the minimum score along the deepest node's path to root):

| statistic | `graph` only (n=129) | all variants (n=261) |
|---|---|---|
| `mean` of node scores | 0.444 [0.33, 0.56] | 0.391 [0.32, 0.46] |
| `min` | 0.466 [0.35, 0.58] | 0.389 [0.32, 0.46] |
| `max` | 0.436 [0.32, 0.55] | 0.396 [0.32, 0.47] |
| `last` | 0.466 [0.35, 0.58] | 0.399 [0.33, 0.47] |
| **`path_running_min`** | **0.466 [0.35, 0.58]** | 0.399 [0.33, 0.47] |
| `executed_mean` (done nodes only) | 0.441 [0.33, 0.55] | 0.385 [0.31, 0.46] |
| *free:* number of scored nodes | 0.422 [0.31, 0.54] | 0.396 [0.32, 0.47] |
| *free:* number of nodes | 0.391 [0.28, 0.50] | 0.371 [0.30, 0.44] |
| *free:* fraction of nodes `done` | 0.473 [0.36, 0.59] | **0.637 [0.57, 0.70]** |

On the variant that owns the mechanism, **every score statistic's interval straddles 0.5** and
every point estimate is on the *wrong* side of it. Pooled over both variants the inversion
tightens (`mean` 0.391 [0.32, 0.46], i.e. a 0.609 *failure* predictor) — but that is the same
confound the confidence-judge analysis found: raw node count scores 0.371 in the same direction,
so what the pooled statistic mostly measures is graph size. The only thing here that beats chance
by more than noise is an LLM-free one: **fraction of nodes that reached `done`, 0.637 [0.57,
0.70]** — descriptive, not a proposed lever, and it needs no judge call.

**By model — the spread is the same size confound, not model quality.** (`graph` only.)

| model | runs | `mean` AUC | free `n_scored_nodes` AUC |
|---|---|---|---|
| `gpt-5-mini` | 53 | 0.621 [0.42, 0.82] | 0.714 [0.54, 0.89] |
| `gemini-3-flash-preview` | 11 | 0.611 [0.18, 1.04] | 0.833 [0.56, 1.11] |
| `claude-sonnet-4.6` | 14 | **0.500 exactly** | 0.154 [-0.35, 0.65] |
| `gpt-4.1-nano` | 21 | 0.309 [0.04, 0.58] | 0.500 [0.18, 0.82] |
| `gemini-2.5-flash` | 30 | 0.255 [0.03, 0.48] | 0.475 [0.23, 0.73] |

The 0.255 → 0.621 spread looks dramatic and tracks the free structural baseline almost
rank-for-rank. Sonnet's exact 0.500 is not a coincidence or a good sign: every one of its scored
nodes carries the same value, so the statistic is a constant. **No model's evaluation score is
"fine".**

**Node level** (intervals optimistic): pooled AUC 0.426 [0.36, 0.49] on `graph`, 0.396 [0.36, 0.44]
pooled over both. By action kind on `graph`: `visit` 0.421 (n=261), `search` 0.440 (n=94),
`think` 0.185 (n=25). Nothing separates.

**Path position.** `should_backtrack` cares about dead branches, so the natural split is
"node the run actually took" vs "node the selector dropped". In the 129 `graph` runs there is no
such split to make: of 389 scored nodes, **365 are `done`, 13 `failed`, 11 `blocked`, and 0 are
`skipped`** — every candidate the evaluator scored was executed anyway. That is not an accident of
the corpus: `min_score_threshold` is **0.0**, and a score clamped to `[0, 1]` can never fall below
it, so the score gates nothing at the selection step. Zero nodes in the whole corpus carry
`_got_pruned` either. In the `sequential` variant, where dropped nodes do exist, the score is
*anti*-selective: dropped nodes mean 0.342 vs executed 0.272, AUC 0.378 for "which sibling ran".

---

## 2. Root cause

Two mechanisms, both structural, neither about prompt quality.

### 2a. The score rates an action that has not run yet, and is capped at 0.5 before anyone sees it

`idea_engine._expand_or_execute` calls `evaluation.evaluate_batch(graph, node_id, eligible)` on
the *children* — before `_execute_action` touches any of them. `evaluate`/`evaluate_batch` then
hit their own penalty path, because at that moment every candidate has an action and no result:

```python
# LlmEvaluationPolicy.evaluate; LlmBatchEvaluationPolicy.evaluate_batch does the same per node
has_action = action and not NodeDetailsExtractor.is_merge_action(node.details)
has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
...
if has_action and not has_result:
    score = min(score, float(self._cfg.evaluation.no_action_result_score_cap))   # 0.5
```

and the batch prompt independently instructs the same thing harder:

> "Nodes with actions but no action_result score <=0.2."

Both fire on **every** node the judge ever scores. The recorded distribution is exactly what that
predicts, and it is re-derived from data rather than trusted from the source:

| | all variants (767) | `graph` only (389) |
|---|---|---|
| distinct values | 14 | 14 |
| range | **[0.00, 0.50]** | **[0.00, 0.50]** |
| mean / median | 0.293 / 0.200 | 0.256 / 0.200 |
| fraction above the 0.5 cap | **0.000** | **0.000** |
| exactly on a penalty constant (0.4 or 0.5) | **0.459** | 0.355 |
| `<= 0.2` (the prompt's ceiling for unexecuted work) | 0.515 | 0.599 |
| `< 0.3` (the backtrack low-score threshold) | **0.515** | **0.599** |
| runs whose scores are all identical | **0.785** | 0.760 |
| mean within-run spread | 0.035 | 0.041 |

Two things fall out of that table. First, **the `<0.3` set and the `<=0.2` set are the same set** —
the band (0.2, 0.3) is empty, so "below the backtrack threshold" means precisely "the judge obeyed
the rubric line about unexecuted work", not "this branch is dead". Second, `node.score` is never
updated after the action runs — `graph.evaluate` is called from nowhere else — so the number a
backtrack decision would read describes a *plan*, and by the time the walk reads it the plan's
outcome is already sitting in the same node's `action_result`, unused.

The same cap distorts the score's two other consumers: `compute_dynamic_beam_width`'s
`beam_score_high` is **0.7**, above the achievable maximum, so its narrow-the-beam branch is
unreachable, while the observed means (0.293 / 0.256) sit below `beam_score_low` **0.3**, so it
returns `beam_max` almost always.

### 2b. The graphs are one level deep, so there is no chain to walk

`should_backtrack`'s premise is a *chain* of low-scoring ancestors. Recorded path lengths:
`{1: 261, 2: 1101}` (all variants) — root, plus children, and nothing below. 767 of 767 scored
nodes sit at path length 2. Decomposition emits a flat fan of leaf actions and the engine's default
path (`auto_parallel_siblings`, `allow_execute_all_children`) executes them all in one step, which
also **skips evaluation entirely** for that batch — `"PARALLEL: Executing {n} children
(limit={parallel_limit}, skipping evaluation)"`.
Depth beyond 1 exists in today's engine only through re-expansion (A1/contract/plan-library), and
no run in this corpus used it.

So the failure is not "the judge scored the dead end 0.4 instead of 0.2". It is that a rule
requiring five stacked evaluations has been shipped against graphs that never stack two.

---

## 3. Qualitative evidence

**There is none to quote, and that is itself a finding.** `LlmBatchEvaluationPolicy` records
`node.details["evaluation"] = {"score": score}` — the rationale the prompt asks for is parsed only
by the single-node `LlmEvaluationPolicy` and discarded by the batch path the engine actually uses.
Across the corpus: **0 of 767 scored nodes recorded a rationale.** Where the confidence judge left
1683 `reason` strings to mine, this judge leaves a bare float. Nothing can be audited after the
fact, including by a future calibration driver.

What can be shown verbatim is the shape the walk is given. A complete recorded graph
(`20260526_003141_014_google/gemini-2.5-flash_graph_r1.json`, run scored 0.80):

```
root  (score None)  Start at a news article about climate change …
 ├─ 0.5 done     Search for a recent news article about climate change
 ├─ 0.5 done     Visit the news article found in the search results
 ├─ 0.5 done     Search for a scientific research paper …
 ├─ 0.5 failed   Visit the scientific research paper or study
 ├─ 0.5 done     Search for a government policy document …
 └─ None skipped Merge: …
```

Five scored nodes, one distinct value, all at depth 1, one of them a *failed* action still holding
the 0.5 it was given before it ran. `path_to_root` from any of them is two nodes long. The
top-scored node inside eventually-failed runs is the same picture — e.g.
`20260526_030710_001_google/gemini-2.5-flash_graph_r1.json` (run 0.59): node score 0.50, status
`done`, title *"Gather population data from U.S. Census Bureau (government source)"*. The judge is
rating whether a proposed sub-goal sounds like a sensible thing to do. It is not wrong about that,
and it is not evidence about anything downstream.

---

## 4. NOT implemented — what the data does and does not support

**Nothing was built. No engine file was touched by this investigation.**

### The verdict on calibrating a backtrack mechanism on this signal: **no.**

The deferred twin of A6 (a calibrated backtrack) cannot be built on `node.score` as it stands, and
the blocker is not threshold choice:

* there is **no held-out signal to calibrate** — every run-level statistic's interval straddles
  0.5 on the variant that owns the mechanism, with point estimates on the wrong side, and the free
  `done_fraction` baseline (0.637) beats all of them;
* there is **nothing to trigger on** — the rule's premise (a ≥5-deep low chain) occurs zero times,
  and ≥2-deep occurs zero times;
* the number is **not a measurement of the thing being decided** — it is written before the action
  runs, capped at 0.5, identical within 78.5% of runs, and never revised afterwards;
* an A6-style driver would fail closed anyway. That is the right outcome, but running it would
  cost real effort to reproduce a conclusion already visible here.

### If the direction is ever revisited, these are prerequisites, not proposals

1. **Score outcomes, not plans.** A backtrack decision needs a post-execution number: re-evaluate
   (or evaluate only) after `action_result` exists, so the penalty cap stops firing on 100% of
   nodes and the value range stops being `[0, 0.5]`. This is a real behaviour change to selection
   and beam width as well, and needs its own A/B — it is not a free instrumentation fix.
2. **Record the rationale on the batch path.** One dict key. Without it no future calibration or
   audit of this judge is possible at all.
3. **Re-measure on graphs that are actually deep.** Every number above is conditioned on depth-1
   graphs. Re-expansion-heavy adaptive runs may produce chains; until a corpus of those exists,
   "does the score predict anything along a chain" is untested rather than answered — this doc
   answers "does the shipped rule ever get a chain to read", and that answer is no.
4. **Prefer the free statistic as the bar.** Any replacement must beat `done_fraction` (0.637),
   not 0.5.

### Explicitly not worth pursuing (measured, negative)

* Lowering `got_backtrack_dead_end_threshold` to make the rule reachable: at `dead_end>=1` the
  rule fires on 71% of `graph` runs and the runs it abandons pass *more* often than the ones it
  keeps (0.750 vs 0.676).
* Mining the judge's prose: there is no prose (0/767).
* `node.score` at the selection step: `min_score_threshold` is 0.0, so it already gates nothing,
  and in the one variant where siblings are dropped the score is anti-selective (AUC 0.378).
