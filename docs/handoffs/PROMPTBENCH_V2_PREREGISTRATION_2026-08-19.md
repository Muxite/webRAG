# Pre-registration: promptbench v2 — coverage, calibration, and a statistic that can resolve

**Written 2026-08-19, before any v2 row exists.** No v2 cell has been run beyond a
6-cell plumbing smoke (`qwen2.5:0.5b`, `keystone_claim`/`calibration`, output
discarded). The offline suite is green at 5068 passed / 18 skipped.

Design rationale in `docs/superpowers/specs/2026-08-19-promptbench-expansion-design.md`.
v1's result in `PROMPTBENCH_RESULTS_2026-08-19.md`.

---

## 1. Why there is a v2

v1 established one thing cleanly and was **structurally incapable** of establishing
the other.

**Established, and not in question here:** the trailing justification costs
**16.3x the completion tokens and buys no measurable accuracy**. That follows from
`A0 >= A1` plus the token ratio and needs no significance test.

**Structurally blocked:** v1's headline aggregate was a **sign test over 5 models**.

```
smallest attainable two-sided p at k models = 2 x (1/2)^k
k = 5  ->  0.0625
```

H1 landed at **p = 0.062**. That is the floor, not a measurement. A perfect 5/0
sweep could not have cleared 0.05. Items were never the unit of the test, so
adding items to v1's design would have sharpened every cell and moved the headline
not at all.

**Out of instrument range:** v1 measures discrete accuracy. The anti-calibration in
`CONFIDENCE_JUDGE_MISCALIBRATION.md` concerns a continuous score against run
outcomes. The bench that was built to investigate it could not address it.

## 2. What is fixed, and what is therefore judged once

### Primary estimator (replaces the sign test as the headline)

Mean paired per-item delta, `arm − baseline`, with:

- **unit of analysis** the `(model, item)` pair;
- **models as fixed effects** — this estimates the effect **across this roster**,
  not across models in general, and no sentence in the write-up may imply otherwise;
- **equal weight per model** (mean of per-model means), so a model cannot dominate
  by having more surviving cells after transport errors and parse failures;
- **task module as the resampling unit** — items from one module share a topic, a
  statement and an author;
- **CI** by percentile cluster bootstrap, 10,000 draws, seed 20260819;
- **test** by cluster-level sign permutation, 10,000 draws, same seed.

The per-model sign test still prints as a **consistency display**, labelled as
such. It answers a different question (do all models agree on direction) and costs
nothing.

**Declared now, because it is the honest reading:** applying this estimator to v1's
existing local rows gives `A2 − A1` = **+0.149, CI [+0.035, +0.263], permutation
p = 0.0345** on the three non-degenerate models. That number is **post-hoc** — the
estimator was chosen after v1's result was known — and it is recorded here as a
machinery sanity check and as the reason for the expected direction, **not** as a
confirmatory result. Only the v2 run tests it.

### Baselines

| family | baseline | rationale |
|---|---|---|
| `verify`, `select`, `keystone_claim`, `followup`, `goal_achieved` | `A1` | the engine's answer-then-justify convention |
| `calibration` | `C_A1` | the engine's literal `{confidence, reason}` order |

### Families and counts, frozen

| family | items | clusters | balance | arms |
|---|---|---|---|---|
| `verify` | 38 | 19 | 19/19 | 8 |
| `select` | 19 | 19 | 1-of-4..6 | 7 |
| `keystone_claim` | 30 | 15 | 15/15 | 9 |
| `followup` | 56 | 28 | 28/28 | 8 |
| `goal_achieved` | 56 | 28 | 28/28 | 8 |
| `calibration` | 38 | 19 | 19/19 | 5 |

**237 graded items, 37 distinct task modules, 1,793 cells per model.**
Reproduce with `runner --census`.

### Models

**AMENDED 2026-08-19, before any v2 row was collected.** The amendment is recorded
rather than silently applied, and it is a reduction, which cannot manufacture an
effect — a smaller roster widens intervals.

**Run now (local tier):** `qwen2.5:0.5b`, `qwen2.5:7b`. Two models spanning a 14x
parameter range. `qwen2.5:7b` is retained specifically because it is the only model
on which v1 found an individually significant SHIPPED effect (−0.211, p = 0.008).

**Deferred:** `qwen2.5:1.5b`, `llama3.2:3b` pending a local-inference hardware
upgrade. Note for that upgrade: `runner.py` iterates model-outer, so each model is
loaded exactly once per run and swap latency is already negligible. The ~8 h figure
is inference volume, not swapping.

**Deferred:** the API tier (`openai/gpt-4.1-nano`,
`google/gemini-2.5-flash-lite`), gated behind a clean local matrix.

**The roster is deliberately not widened** — power comes from items and the
estimator, not from models, which is the whole reason the sign test was replaced.
With 2 models the sign test would floor at p = 0.5 and be useless; the pooled
estimator is unaffected in kind, only in width.

**Consequence for H4, declared now:** the size-interaction hypothesis is
*descriptive only* at 6 models and is **not reportable at all** at 2. It is dropped
from this run and returns if the deferred models are added.

### Metrics

Accuracy, paired by `(item, model)`. **Parse failure and abstention are reported as
separate columns and never folded into "incorrect"** — counting a parse failure as
wrong systematically punishes verbose arms and manufactures the effect H1 tests.
Cost as accuracy per completion token; `cached_prompt_tokens` reported alongside
`prompt_tokens`, never subtracted.

Calibration: Brier, 10-bin ECE, AUC vs correctness, calibration slope, and the
Murphy reliability/resolution/uncertainty decomposition.

## 3. Hypotheses, judged once

- **H1 (primary).** Reason-before-answer (`A2`) beats the engine's convention
  (`A1`), pooled across models and families, CI excluding 0.
- **H2 (primary).** `A0` (answer only) is no worse than `A1`, which is what
  licenses dropping unread `reason` fields for the 16.3x token saving.
- **H3 (primary, calibration).** `C_A2` and `C_expected` produce better-calibrated
  confidence than `C_A1` — lower Brier and higher AUC.
- **H4 (descriptive only).** The effect is larger on smaller models. Six models
  cannot test an interaction; reported with its interval and never called confirmed.
- **H5 (secondary, Holm-corrected).** `F_json` moves accuracy independently of
  answer position.
- **Validity controls, not hypotheses.** `G_nostatement` and `G_noevidence` must
  collapse accuracy toward chance. If withholding the thing an item is supposed to
  require does **not** hurt, that family was answerable from priors and every other
  number on it is void.

## 4. Exclusion rules, fixed now

Applied to the printed summary **and to the primary estimate** — v1 applied them
only to the summary, which this cycle fixes.

1. Parse-failure rate > 50% → reported, excluded from accuracy conclusions.
2. Fewer than 5 clusters → underpowered, excluded.
3. LOCO swing > 10 pp → underpowered, excluded.
4. **Degenerate**: one answer on ≥90% of ≥8 parsed predictions → excluded. On a
   balanced set a constant answerer scores exactly 0.500 while judging nothing.
   v1 caught six such cells, including one model's **A1 baseline**, which
   invalidated every delta measured against it.
5. **Degenerate confidence**: a cell whose stated confidence never varies is
   excluded from calibration conclusions. Its Brier can look respectable while its
   resolution is exactly 0 — the continuous analogue of rule 4.
6. When either side of a contrast is excluded, that model is dropped from that
   contrast. An arm judged against a disqualified baseline is as unusable as a
   disqualified arm.

## 5. Known limits, stated before the numbers exist

- **`keystone_claim`'s evidence is authoring prose, not retrieved page text.** The
  module docstrings are hand-written walkthroughs of the intended solution path.
  They state the datum in prose, which is what the family needs, but they are
  cleaner and more purposeful than a real page. Expect this family to sit high.
- **Families share clusters and are not independent.** `followup` and
  `goal_achieved` are built from the same 28 candidate sets. A per-family result is
  not 6 independent replications of the same question.
- **`goal_achieved` and `followup` may hit a ceiling on stronger models.** Both ask
  a question with a clean structural answer. A ceiling compresses arm differences;
  it does not bias them, and it will be reported rather than worked around.
- **One rep, T = 0.** Provider nondeterminism stays unquantified. Reps are the
  pre-registered response to an underpowered result, not a default.
- **These are offline judgement items, not agentic runs.** Transfer to end-to-end
  task score is exactly what the follow-on A/B has to establish.
- **No shipped prompt is changed by this cycle**, whatever the result. A micro-eval
  win does not transfer for free.

## 6. Cost, authorized separately at execution time

| tier | cells | basis | estimate |
|---|---|---|---|
| **local, this run (2 models)** | **3,586** | v1: 1,824 cells ≈ 2 h | **≈ 4 h GPU, $0** |
| local, deferred (2 more models) | 3,586 | pending hardware upgrade | ≈ 4 h GPU, $0 |
| API, deferred (2 models) | 3,586 | v1: 912 cells = $0.0685; v2 prompts are longer | ≈ $0.30–0.80 |

Requested ceiling: **$3**, measured as an OpenRouter **key-usage delta**, not a
local sum — a locally-summed estimate misses calls whose response never came back.
Local work takes `gpu-lock`. Per `DEV_CYCLE.md`, this plan does **not**
pre-authorize the run.

## 7. The decision gate

Written before the numbers exist, so the replan is a lookup rather than a
judgement made after seeing them.

### Answer position — on the pooled `A2 − A1`

| result | action |
|---|---|
| CI excludes 0, positive | Implement reason-first at the tabulated call sites, flagged, default OFF. Next cycle is the end-to-end transfer A/B on the mixed-shape tasks. |
| CI excludes 0, negative | Do not reorder. Record the reversal against v1's direction and against this document's post-hoc estimate. |
| CI includes 0, half-width < 0.05 | Effect absent at a useful resolution. **Stop.** Ship only the token saving, which does not depend on H1. |
| CI includes 0, half-width ≥ 0.05 | Underpowered. Expand: reps first (quantifies the unmeasured T=0 nondeterminism), roster second. |

### Token saving — on the pooled `A0 − A1`

CI's lower bound above −0.02 → drop the `reason` field at the call sites whose
reason has no consumer (`got_operations.py:309`, `:192`, `leak_gate.py:928`),
flagged, default OFF. Otherwise leave them and record the cost of the reason.

### Calibration

`C_expected` beats `C_A1` on Brier **and** clears the LLM-free structural baseline
of **AUC 0.655** (from `CONFIDENCE_JUDGE_MISCALIBRATION.md`, not 0.5) → P2 becomes
a real proposal with a measured basis. Otherwise it joins that document's
explicitly-not-worth-pursuing list, which is a genuine result and will be reported
as one.

**A negative slope on `C_A1`** would be the first direct evidence for the
anti-calibration mechanism on clean items, and is worth its own cycle regardless of
how the other gates fall.

### Per-family divergence

If `keystone_claim` and `verify` disagree in **direction** on `A2 − A1`, the
per-call-site recommendations do not generalise across decision types, and the
next cycle is scoped per call site rather than as one global reordering.

## 8. Reproduce

```bash
# fixture + census, $0
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.extract_task_specs
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.runner --census

# containerised, non-default profile
docker compose --profile promptbench run --rm promptbench \
    --models qwen2.5:0.5b qwen2.5:7b \
    --families verify select keystone_claim followup goal_achieved calibration \
    --variants A0 A1 A2 A3 A4 SHIPPED F_json G_nostatement G_noevidence \
                C_A1 C_A2 C_verbal C_expected \
    --out agent/idea_test_results/promptbench_runs_v2.jsonl

docker compose --profile promptbench run --rm promptbench-analyze \
    --runs agent/idea_test_results/promptbench_runs_v2.jsonl \
    --json-out agent/idea_test_results/promptbench_v2_analysis.json
```

Local runs take `gpu-lock` first. The host path is unchanged and still works:
`PROMPTBENCH_BASE_URL` defaults to `http://127.0.0.1:11435/v1` outside the container.
