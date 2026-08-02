# Why the step-confidence judge barely predicts anything

_Last updated 2026-08-02. Analysis only — **nothing in the engine was changed by this
investigation**. Companion docs: `ADAPTIVE_ENGINE.md` §2–3 (where the judge sits in the loop),
`idea_policies/confidence_early_exit.py` (A6, which found the weakness), `RESEARCH_LIBRARY.md`
(the `got_operations.py` entry)._

Reproduce every number here with:

```
PYTHONPATH=services:services/agent ./.venv/bin/python \
  scripts/analyze_confidence_judge_miscalibration.py --samples 12
```

**Corpus:** the *same* 354 regular-roster trajectories A6 calibrated on (`is_regular_roster`
reused verbatim; label = `validation.overall_score >= 0.75`; base rate 0.511), 1683 judged
steps, models `gpt-4.1-nano` (170 runs) / `gpt-5-mini` (96) / `deepseek-v4-flash` (88).
Intervals are Hanley–McNeil 95%; step-level intervals are optimistic (steps in one run share a
label), so run-level numbers are the ones to argue from.

---

## 1. The quantified claim

**The judge is not "too generous". It is non-discriminative, and on nearly half the trace it is
actively misleading.**

| Statistic (whole run, one sample per trajectory) | AUC vs eventual pass |
|---|---|
| `running_mean` of confidence, all kinds | 0.571 [0.51, 0.63] |
| `running_min`, all kinds | 0.469 [0.41, 0.53] |
| `last`, all kinds | 0.479 [0.42, 0.54] |
| `running_mean`, content-bearing kinds only | 0.551 [0.48, 0.62] |
| **number of judged steps** (free, no LLM) | **0.655 [0.60, 0.71]** |
| **fraction of judged steps that are content-bearing** (free, no LLM) | **0.634 [0.58, 0.69]** |

Two LLM-free graph statistics outrank every confidence statistic the judge produces. (They are
*descriptive*, not proposed levers — an early-exit rule that acts on step count changes step
count. They are here as the bar a judge has to clear to be worth its calls.)

**Overconfidence check.** Mean confidence on steps of eventually-**passed** runs is 0.374
(median 0.200, n=1014); on eventually-**failed** runs it is 0.384 (median 0.200, n=669) — a gap
of **−0.010**, i.e. failed trajectories' steps score marginally *higher*. Step-level AUC 0.488
[0.46, 0.52]. Restricted to the kinds the judge can actually read (`visit`/`search`): 0.614 vs
0.549, gap **+0.065**, AUC 0.561 [0.52, 0.60].

**By action kind** (`n`, mean confidence, fraction scoring ≤0.05, AUC, and the *lift* a ≥0.8
score buys over that kind's own base rate):

| kind | n | mean | ≤0.05 | AUC | AUC (non-degenerate only) | ≥0.8 rate | lift |
|---|---|---|---|---|---|---|---|
| `visit` | 719 | 0.564 | 0.051 | 0.584 [0.54, 0.63] | 0.607 [0.56, 0.65] | 0.423 | **+0.076** |
| `merge` | 385 | 0.169 | 0.616 | **0.418 [0.36, 0.47]** | **0.288 [0.21, 0.37]** | 0.099 | **−0.335** |
| `search` | 233 | 0.663 | 0.142 | 0.508 [0.43, 0.58] | 0.513 [0.43, 0.59] | 0.652 | **+0.004** |
| `think` | 168 | 0.054 | 0.917 | 0.459 [0.37, 0.55] | 0.425 [0.09, 0.76] | 0.042 | −0.262 |
| `verify` | 166 | 0.013 | 0.958 | 0.493 [0.38, 0.60] | 0.500 | 0.006 | +0.205 |
| `save` | 10 | 0.000 | 1.000 | 0.500 [0.12, 0.88] | — | 0.000 | — |

`visit` is the only kind carrying signal. A confident `search` step is worth **nothing** (+0.004
lift on 65% of search steps). A confident `merge` is a *bad* omen (−0.335) and its AUC interval
excludes 0.5 in the wrong direction. (The `think` and `verify` lifts are computed on ~7 and ~1
steps respectively — noise, listed only for completeness.)

**By model — the weakness is not a bad-model average.** E-valuator's caution that calibration is
per-model was tested and does *not* explain this:

| model | runs | run-mean AUC (all kinds) | run-mean AUC (content only) | n-judged-steps AUC |
|---|---|---|---|---|
| `deepseek-v4-flash` | 88 | 0.665 [0.55, 0.78] | 0.402 [0.22, 0.58] | 0.679 |
| `gpt-4.1-nano` | 170 | 0.394 [0.31, 0.48] | 0.481 [0.38, 0.58] | 0.602 |
| `gpt-5-mini` | 96 | 0.738 [0.64, 0.84] | 0.523 [0.40, 0.65] | 0.787 |

The all-kinds column looks like a huge per-model spread (0.394 → 0.738), but the spread tracks
the *free* structural statistics almost exactly (0.602 → 0.787): what the all-kinds run-mean
mostly measures is how many blind steps a model's graph shape produced, and that correlates with
the outcome in opposite directions per model. Once the blind kinds are removed, all three models
land in a narrow, unimpressive 0.40–0.52 band. **No model's step-confidence is "fine" — the
apparent heterogeneity is the blindness confound, not model quality.**

**Reproducing the barrage census.** `contract_satisfaction.py`'s docstring cites "conf≥0.6 runs
scored 0.475 vs 0.687 for conf<0.6 on nano/good_adaptive". That reproduces **exactly** (n=21 vs
n=53). It does *not* generalise: corpus-wide the same split gives 0.631 vs 0.622 (no inversion,
no signal), and on content-bearing steps only it is correctly *ordered* — 0.748 vs 0.662 mean
score, 0.662 vs 0.541 pass rate. So the "anti-calibrated" label is too strong; the honest
statement is **blind on 43% of the trace and near-useless on the rest**, with an inversion that
appears in slices dominated by blind steps.

**A1's exposure.** With `got_step_confidence_reexpand_threshold` at its 0.5 default, **62.2%**
of all judged steps (1047/1683) are below the trigger, and **62.9%** of those are kinds the
judge cannot see; **45.1%** of them carry a reason that says outright there was nothing to judge.
Kind mix of sub-threshold steps: `merge` 324, `visit` 317, `verify` 164, `think` 159,
`search` 71, `save` 10. `_reexpand_guards_ok` drops the merge share (merge leaves are never
re-expanded), but `think`/`verify`/`save` are not excluded by that guard — so **333 of the 721
sub-threshold steps that survive it (46.2%) are steps the judge could not see**. Roughly half of
A1's triggers are blindness rather than distrust; that A1 still helped is evidence re-expansion
is independently useful, not that the trigger is informative.

**Signals already computed and discarded** (checked so nobody re-investigates): `verify`'s own
evidence-grounded `confidence` field scores AUC 0.438 [0.32, 0.56] (step) / 0.428 (run);
`merge`'s `goal_achieved` flag scores 0.514 / 0.531. Neither is a free win.

**Reason text vs the number.** Reason length has AUC 0.443 [0.41, 0.47] — i.e. it predicts
*failure* at 0.557, weakly, and mostly because empty-payload complaints are short. Hedging
vocabulary ("appears to", "seems to", "likely", …) scores 0.502 [0.47, 0.53]: **nothing**. There
is no wasted signal in the prose to harvest.

---

## 2. Root cause

The prior hypothesis — "rates plausibility of a single step's content, no independent
verification, no cross-step consistency check" — is **confirmed but incomplete**. There are two
distinct mechanisms, and the larger one is plumbing, not prompt design.

### 2a. The judge is structurally blind on 43% of the steps it scores

`GoTOperations.judge_step_confidence` builds its payload from exactly three result fields:

```python
content = result.get("content") or result.get("content_full") or ""
results_summary = result.get("results")
```

Measured over the 1837 successful action results recorded in these runs' graphs:

| action | successful results | fraction with anything in those fields |
|---|---|---|
| `visit` | 923 | **1.000** |
| `search` | 218 | 0.936 |
| `merge` | 388 | **0.000** |
| `think` | 154 | **0.000** |
| `verify` | 144 | **0.000** |
| `save` | 10 | **0.000** |

`MergeLeafAction` returns `synthesized`/`raw_response`, `ThinkLeafAction` returns
`thinking_content`, `VerifyLeafAction` returns `verdict`/`quote`/`reasoning`/`raw_response`,
`SaveLeafAction` returns `count`. None of them touch `content`/`content_full`/`results`, so the
judge is handed `{"resolved_content": "", "resolved_results": null}` and asked how confident it
is. (`search`'s missing 6.4% are searches that genuinely returned nothing — a real, correctly
scored zero, not blindness.) **731 of 1683 judged steps (43.4%)** are outside `visit`/`search`
(`merge` 385, `think` 168, `verify` 166, `save` 10, plus two one-off `expansion`/`skip` entries);
**76.7%** of them score ≤0.05 and **63.2%** say so verbatim in the reason. **83.6%** of runs
contain at least one, **72.3%** within the first five judged steps.

This is what destroys the prefix statistic A6 measured. The `running_min` AUC by timestep:

```
all kinds      t1:0.583  t2:0.520  t3:0.442  t4:0.373  t5:0.364  t6:0.324  t7:0.254
content only   t1:0.567  t2:0.592  t3:0.616  t4:0.545  t5:0.533  t6:0.461  t7:0.432
```

One blind `merge`/`think` zeroes the running minimum for the rest of the trajectory, so the
longer a run goes the more certainly its statistic is a constant 0 — which is exactly the
"decays to chance (and below) by step 5" shape A6 reported. The blind steps do not merely
dilute: `merge`'s non-degenerate scores are *anti*-predictive at AUC 0.288 [0.21, 0.37], because
a nonzero score on an empty payload is the judge grading the **mandate text it was shown**
instead of an output that does not exist (see §3).

### 2b. Where it *can* see, it rates local plausibility, and for `search` that is worthless

`search` is the clean demonstration: 65.2% of search steps score ≥0.8, and those steps'
eventual pass rate is 0.579 against a 0.575 base rate — **+0.004 lift, zero information**. The
question the judge answers on a search step ("do these results contain a plausible page for this
sub-goal?") is almost always yes and is nearly independent of whether the trajectory ends up
correct. `visit` is the one place the judge earns its call (AUC 0.607 on non-degenerate scores,
+0.076 lift) — it can at least notice a page that does not mention the datum.

The prompt asks for "CORRECT and ON-TRACK … using ONLY what is visible below … judge from the
resolved content alone". There is no independent solve, no comparison against the other steps'
findings, and no view of the mandate's *answer* requirements beyond the mandate text itself. It
is a single-step plausibility rater by construction, and the data says that is worth ~0.58 AUC
on `visit` and nothing anywhere else.

---

## 3. Qualitative evidence (verbatim `reason` strings)

**The judge grading the instructions because there is no output** — `merge`, confidence **1.0**,
run scored 0.40 (`suite50_nano_full_rep1_122_openai-gpt-4.1-nano_graph_r1.json`):

> "The output accurately reflects the instructions and the task's requirements by detailing the
> step-by-step approach: checking each telescope's status on Wikipedia, ruling out the ring-type,
> the non-operational, and the smaller fully steerable dish… The reasoning aligns with the task's
> discipline guidelines, and the response is clear and comprehensive."

There was no output. Every clause praises the mandate that was pasted into the prompt.

**The number contradicting its own prose** — `merge`, confidence **1.0**, run scored 0.20
(`suite50_ds_good_adaptive_rep2_129_deepseek-deepseek-v4-flash_graph_r1.json`):

> "The resolved content is empty and contains no output to evaluate… the step has not produced
> any usable output, and **confidence is zero** that it is correct or on-track."

The prose is right; the emitted `confidence` field is 1.0. The JSON number and the reasoning are
not tied together, so blind steps do not even fail *consistently*.

**The normal blind case** — `think`, confidence 0.0, in a run that scored 0.80 and passed
(`adaptive_proof_g0_140_openai-gpt-5-mini_graph_r1.json`):

> "No step output or page content was provided: 'resolved_content' is empty and
> 'resolved_results' is null, so there is nothing to verify."

A correct, honest report of a plumbing gap, recorded as if it were a quality judgement — and
then fed to A1's re-expansion trigger and A6's stopping statistic.

**Plausibility, not correctness, on a step the judge could see** — `search`, confidence **0.98**,
run scored 0.40 (`honest_adaptive_134_openai-gpt-5-mini_graph_r1.json`):

> "The resolved results include the exact 'Statue of Liberty' Wikipedia article
> (https://en.wikipedia.org/wiki/Statue_of_Liberty), which matches the sub-task goal to search
> for that article… there is no visible error or mismatch in the provided output."

The same task at confidence 0.92 scored **0.00** on another repetition. Finding the right URL is
not evidence about the answer, but it is the entire content of the judgement.

**Right fact, wrong hop** — `visit`, confidence **1.0**, run scored 0.30
(`suite50_ds_good_adaptive_rep1_124_deepseek-deepseek-v4-flash_graph_r1.json`):

> "The resolved content explicitly states the Tu-144's introduction into commercial service on
> 26 December 1975, confirming its service entry status as required for the sub-task."

Correct about the page, correct about the sub-goal, and irrelevant to whether the mandate's
multi-hop answer came out right. The recurring vocabulary across the high-confidence-but-failed
sample is *"clearly states"*, *"explicitly shows"*, *"directly matches"*, *"is on-track"* —
presence-of-plausible-text language. Nothing in any sampled reason resembles an independent
attempt to answer the sub-question and compare.

---

## 4. NOT implemented — proposals, in data-supported order

**None of this was built. No engine file was touched by this investigation.** Each proposal
below states what the data actually supports and what it does not.

### P1. Stop scoring steps whose payload the judge never receives

*Scope:* either (a) extend `judge_step_confidence`'s extraction to the per-kind output fields
(`synthesized`/`raw_response` for merge, `thinking_content` for think, `verdict`+`quote`+
`reasoning` for verify), or (b) skip judging those kinds entirely and record an explicit
"not judged" entry instead of a 0.0 confidence. (b) is strictly smaller and cannot change what
the judge is asked; (a) is a real behavior change that needs its own A/B because a merge's
synthesis is a *different* object to judge than a page.

*What the data supports:* 43.4% of the trace is currently a plumbing artifact; removing it moves
step-level AUC 0.488 → 0.561, removes an actively anti-predictive term (`merge` graded AUC
0.288), un-breaks `running_min` as a statistic (0.254 → 0.432 at t7), and would stop the ~46% of
A1's guard-surviving triggers that currently fire on blindness. Every downstream consumer would
need to handle a "not judged" step instead of treating it as 0.0 — A1's threshold (where a
contract verdict is unavailable the judge alone decides), A6's prefix statistic, and the
observability trace the calibration driver reads.

*What it does not fix:* the remaining signal is still weak — content-only run-level AUC 0.551
[0.48, 0.62], which does not clear the free `n_judged_steps` baseline of 0.655. **P1 is a
correctness fix for the instrumentation, not a fix for the judge.**

### P2. Make the judge solve the sub-question before it sees the answer

*Scope:* per arXiv 2607.05904 (recorded in `RESEARCH_LIBRARY.md`'s confirmed-gaps section:
independent-solving judges cut a self-play reward-hacking false-positive rate 0.719 → 0.012 on
GSM8K), restructure the judge into two turns: (1) given only the mandate + sub-goal + action,
state what a correct output would have to contain (the expected datum/entity/URL shape); (2)
then reveal the resolved content and score the *match* between the two. Cheaper variant: keep it
one call but require the JSON to emit `expected` before `observed` before `confidence`, so the
expectation cannot be written backwards from the content.

*What the data supports:* the failure mode is specifically presence-of-plausible-text scoring —
`search` gets +0.004 lift while 65% of its steps score ≥0.8, and every sampled reason keys on
"clearly states"/"directly matches". A judge forced to write "a correct output for hop 1 names
*the French engineer who built the statue's iron armature*" before looking would have had to
notice that the Statue-of-Liberty search step it scored 0.98 names no engineer at all — it
contains a URL and a snippet. Note
this overlaps `contract_satisfaction.py`'s deterministic datum/subject rules,
which already encode a cheaper version of "what should this step have produced"; the LLM version
is only worth it if it beats that, so P2 should be measured against `contract_satisfaction`, not
against the raw judge.

*What it does not fix, and the honest risk:* the 2607.05904 result is from GSM8K self-play, not
multi-hop web research — that exact gap is why `RESEARCH_LIBRARY.md` lists it as an
*idea*, not as evidence. It lengthens the judge prompt and, in the two-turn form, roughly doubles
judge cost — and the bar it must clear is 0.655 (the free structural baseline), not 0.5.

### Explicitly not worth pursuing (measured, negative)

* `verify`'s own `confidence` field — AUC 0.438 [0.32, 0.56].
* `merge`'s `goal_achieved` flag — AUC 0.514 (step) / 0.531 (run).
* Mining the reason prose: hedge vocabulary AUC 0.502, reason length 0.443 (weak *failure*
  predictor, driven by empty-payload complaints being short).

### How any of this gets accepted

Re-run `scripts/analyze_confidence_judge_miscalibration.py` (this doc's numbers) and then
`scripts/calibrate_confidence_early_exit.py`. The bar is unchanged and unforgiving: A6 ships a
rule only if some threshold certifies a stop precision above the base rate with non-vacuous
coverage. Today nothing certifies (ceiling 0.553 vs a 0.511 base rate). A judge redesign is
worth landing when that artifact changes, not when the AUC table looks nicer.
