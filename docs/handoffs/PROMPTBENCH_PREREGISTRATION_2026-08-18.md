# Pre-registration: promptbench, answer position and verbosity

**Written 2026-08-18, before any result was inspected.** The local matrix was
launched immediately before this file and no row of it has been read. The API
tier has not been run at all.

## The question

Every short-answer LLM call site in this engine emits **the answer before its
justification**. Verified against source at `HEAD` 45af6033:

| call site | answer | reasoning | order |
|---|---|---|---|
| `idea_policies/actions.py:2202` verify | 4-way enum + confidence | `"reasoning": "<one sentence>"`, last field | answer first |
| `got_operations.py` step-confidence | float 0–1 | `reason` | answer first |
| `got_operations.py` needs_followup | boolean | `reason` | answer first |
| `idea_policies/actions.py` merge `goal_achieved` | boolean | `goal_evaluation` | answer first |
| `evaluation.py` single score | float | `rationale` | answer first |
| `evaluation.py` batch score | N floats | none | n/a |
| `leak_gate.py` audit | boolean | `reason` | answer first |
| `idea_finalize.py` recompute | long | enumerated, before the answer | reasoning first |

Nobody chose this. It accreted. And `CONFIDENCE_JUDGE_MISCALIBRATION.md` records
the step-confidence judge as anti-calibrated (runs scored ≥0.6 averaged 0.475;
runs scored <0.6 averaged 0.687). Commit-then-rationalise is a plausible
mechanism for that inversion, and it has never been tested.

## Hypotheses, judged once

- **H1 (primary).** `A1` (answer-then-justify, the engine's convention) is no
  more accurate than `A0` (answer only), and both trail `A2`/`A4`
  (reason-then-answer) on items requiring a candidate to be compared against a
  stated constraint.
- **H2 (descriptive only).** The A-effect grows as models shrink. **With 4
  models this cannot be tested** — it is reported as a trend with its interval
  and never called a confirmed interaction.
- **H3 (secondary).** Format (`F_json`) and goal restatement (`G_nostatement`)
  move accuracy independently of answer position.

## Design

**Families**, both constructed from `agent/app/idea_tests/test_*.py` via the
frozen fixture `agent/tests/fixtures/promptbench/task_specs.json`:

| family | items | clusters | balance |
|---|---|---|---|
| `verify` | 38 | 19 | 19 SATISFIES / 19 VIOLATES, balanced by construction |
| `select` | 19 | 19 | one survivor among 3–5 described candidates |

The cluster is the source task module. `verify`'s balance is what makes any
precision-side number defined; the previous cycle shipped two rates that were
undefined and rendered as `0.0`.

**Why not the run corpus.** `agent/idea_test_results/` (3.3 GB) stores only
aggregates — `observability.visit` is `{"count": 1, "chars": 26142}`. There is
no page text, no URL, no link set, and the telemetry sidecars hold a ~200-char
completion prefix with no prompt. `link_select` and `extract_value` as
originally designed are **not buildable from this repo** and are dropped.

**Arms.** `A0`, `A1`, `A2`, `A3`, `A4`, `SHIPPED` (primary);
`F_json`, `G_nostatement` (secondary, Holm-corrected). `SHIPPED` imports
`VerifyAction._DEFAULT_SYSTEM_PROMPT` rather than retyping it.

**Models.** `qwen2.5:0.5b`, `qwen2.5:1.5b`, `llama3.2:3b`, `qwen2.5:7b` — the
cheap end of the roster LangGraph's `create_react_agent` can *also* run, so
findings transfer to a future DAG-v2-vs-LangGraph comparison. Tool-calling
re-verified live this session: all four PASS.

**Metric.** Accuracy, paired by `(item, model)`. Parse failure and abstention
are reported as **separate columns and never folded into "incorrect"** —
counting a parse failure as wrong would systematically punish the verbose arms
and manufacture the very effect H1 is testing.

**Cost metric.** Accuracy per completion token, as a Pareto frontier.
`cached_prompt_tokens` is reported alongside `prompt_tokens`, never subtracted.

## Exclusion rules, fixed now

1. Any (model, family) row whose **parse-failure rate exceeds 50%** is reported
   but excluded from accuracy conclusions: at that rate the accuracy figure
   describes the parser, not the model.
2. Any row with **fewer than 5 clusters**, or a **leave-one-cluster-out swing
   above 10 percentage points**, is tagged `UNDERPOWERED` and excluded from
   conclusions.
3. Significance by sign-flip test on paired per-item deltas
   (`scripts/adaptive_ab_analyze.signflip_p`), Holm correction across the
   secondary arms only. The primary comparison (`A1` vs each of `A0`/`A2`/`A4`)
   is judged once, on `verify`, which is the balanced family.
4. A false-positive rate is emitted only when
   `n_negative >= 0.5 * n_positive`; otherwise `null` with a stated reason.

## What this cycle will NOT do

Change any shipped prompt. A micro-eval win does not transfer to task score for
free. Every recommendation carries a required end-to-end A/B behind a flag,
default OFF, as its own follow-on cycle.

## Spend

Local: $0, ~80 minutes under `gpu-lock`. API tier: not yet run; ≤$1.00 planned
against the standing $5 authorization, with budget reconfirmed before it starts.
