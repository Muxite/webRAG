# Results: promptbench, answer position and verbosity (2026-08-19)

Companion to `PROMPTBENCH_PREREGISTRATION_2026-08-18.md`, which was written and
committed before any row was inspected.

**2,736 graded cells.** 1,824 local (4 models, $0, ~2 h GPU) + 912 API
(2 models). **API spend: $0.0685** against the $5 authorization, measured as an
OpenRouter key-usage delta, not a local sum.

---

## The one-paragraph version

The engine's universal convention — **commit to the answer, then justify it** —
is not the best available shape at any measured point, and it is by far the most
expensive way to be no better. Asking for the answer *alone* is statistically
indistinguishable from asking for answer-then-justification (4 of 5 models
favour answer-only, sign p = 0.375) while costing **16.3x fewer completion
tokens**. Where reasoning genuinely helps, putting it *before* the answer beats
putting it after on **5 of 5** models with a usable baseline (p = 0.062,
individually significant on `llama3.2:3b` at +0.237, p = 0.022). The engine's
literal shipped verify prompt is **significantly worse** than even the plain
answer-first shape on the strongest local model (−0.211, p = 0.008) and on
`gpt-4.1-nano` (−0.158, p = 0.071). Unbounded "think step by step" is the worst
trade in the matrix: 4–70x the tokens, more parse failures, and no gain.

---

## Two measurement artifacts caught before they became findings

Recorded first, because both would have produced confident, wrong, publishable
numbers, and one of them briefly did.

**1. The grader could not see multi-word answers.** `grade_enum` tokenised on
`\w+` before matching, so an option like `Boston Marathon` — never a single
token — could never match. A completion that answered perfectly parsed as
*nothing*. The first full run therefore reported 79–95% "parse failure" across
every prose arm of `select`, and `F_json` "winning" by a landslide.

It was caught by an asymmetry, not by inspection: `verify` (two short options)
showed **0%** parse failure on the same model and arm where `select` showed
**79%**. A real model deficit is not conditional on how long the option strings
are. The whole matrix was re-run after the fix.

**2. The SHIPPED arm was graded against the wrong vocabulary.** The engine's
verify prompt asks for `TRUE/PARTIALLY_TRUE/FALSE/UNVERIFIABLE`; the benchmark's
options are `SATISFIES/VIOLATES`. Models emitted perfectly well-formed
`{"verdict": "FALSE"}` and scored a 92% parse failure. Aliasing the two
vocabularies (declared in `analyze.py`, not applied silently) moved
`qwen2.5:0.5b` SHIPPED from 0.000 to 0.500. SHIPPED x `select` is **excluded
outright**: a four-way truth verdict cannot name one of five candidate
marathons, so that cell asks an incoherent question.

**A third check earned its place: degeneracy.** A model that always answers the
same way scores exactly 0.500 on a balanced set while having judged nothing.
Six cells were caught, including `qwen2.5:1.5b`'s **A1 baseline** (100%
SATISFIES) — which invalidates every delta measured against it, so that model is
excluded from the verify contrasts entirely. Without the balanced design this
would have been invisible; with accuracy alone it is indistinguishable from
genuine half-accuracy.

---

## H1 — answer position. Partially supported, in the predicted direction.

`verify` (38 balanced items, 19 clusters), paired per-item deltas against **A1**,
the engine's convention. `qwen2.5:1.5b` excluded: degenerate baseline.

| arm | 0.5b | 3b | 7b | nano | flash-lite | pos/neg | sign p |
|---|---|---|---|---|---|---|---|
| `A0` answer only | −0.053 | +0.105 | +0.053 | +0.026 | +0.132 | 4/1 | 0.375 |
| `A2` reason→answer | +0.158 | **+0.237*** | +0.053 | +0.026 | +0.211 | **5/0** | **0.062** |
| `A3` step-by-step | +0.184 | +0.079 | −0.079 | +0.081 | −0.132 | 3/2 | 1.000 |
| `A4` ≤40w→answer | +0.079 | +0.105 | +0.000 | +0.079 | +0.132 | 4/0 | 0.125 |
| `SHIPPED` | (degen) | +0.132 | **−0.211*** | −0.158 | −0.026 | 1/3 | 0.625 |

`*` individually significant by sign-flip: `A2` on `llama3.2:3b` p = 0.022;
`SHIPPED` on `qwen2.5:7b` p = 0.008.

**What holds.** A1 is never the best shape at any point. A2 beats it in every
non-degenerate cell measured — a perfectly consistent direction across a 14x
parameter range and two providers — though the aggregate sign test lands at
0.062 rather than under 0.05. **A1 is no better than A0**, which is the half of
H1 that matters most for the recommendation below.

**What does not hold.** No aggregate comparison reaches p < 0.05. Claiming
"reason-first is better" as an established result would overstate this; the
honest statement is *consistent direction, one significant cell, underpowered
aggregate.*

## H2 — interaction with model size. Not supported.

`A2 − A1` by size: 0.5b **+0.158**, 3b **+0.237**, 7b **+0.053**, nano
**+0.026**, flash-lite **+0.211**. No monotone ordering. Pre-registered as
descriptive-only at 5 usable models, and it does not even describe a trend.

## H3 — secondary factors, Holm-corrected.

`F_json` favours JSON on 4 of 5 (p = 0.375 aggregate; best single cell
`llama3.2:3b` +0.158, Holm-adjusted p = 0.489). **Nothing survives correction.**
The strong prior from Q27 is not reproduced at this scale on this task.

`G_nostatement` is the **validity control**, and it works: withholding the task
statement drops `qwen2.5:7b` by −0.421 (Holm p = 0.007) and pushes both API
models to chance on `verify` and to 0.211–0.316 on `select`. The items genuinely
require the supplied context and are not answerable from model priors.

---

## The cost axis — the finding with an actual decision attached

Mean completion tokens per call, `verify`:

| model | `A0` | `A1` | ratio |
|---|---|---|---|
| `qwen2.5:0.5b` | 4.2 | 170.6 | **40.5x** |
| `qwen2.5:1.5b` | 4.9 | 74.3 | 15.2x |
| `llama3.2:3b` | 4.2 | 47.6 | 11.2x |
| `qwen2.5:7b` | 4.7 | 49.6 | 10.6x |
| `gpt-4.1-nano` | 4.3 | 44.9 | 10.5x |
| `gemini-2.5-flash-lite` | 3.5 | 32.9 | 9.4x |
| **mean** | **4.3** | **70.0** | **16.3x** |

Accuracy per 1k completion tokens on `verify` separates the arms by an order of
magnitude: `A0` scores 43.8–240.6, `A1` scores 1.4–21.6, `A3` scores 1.3–2.9.

**The trailing justification costs 16.3x the completion tokens and buys no
measurable accuracy.** That is the cycle's clearest result, and it does not
depend on H1 reaching significance — it follows from A0 ≥ A1 on accuracy
combined with the token ratio.

---

## Per-call-site recommendation

**No prompt is changed this cycle.** Each of these requires an end-to-end A/B
behind a flag, default OFF, because a micro-eval win does not transfer to task
score for free.

| call site | recommendation | basis |
|---|---|---|
| `evaluation.py:200` batch score | **leave** — already emits no rationale | consistent with A0 |
| `got_operations.py:309` step-confidence | **drop the `reason` field** (A0) unless a consumer reads it | A0 ≥ A1 at 1/16 the tokens; and see the anti-calibration note below |
| `got_operations.py:192` needs_followup | **drop the `reason` field** (A0) | same |
| `leak_gate.py:928` audit | **drop the `reason` field** (A0) | same; audit output is consumed as a boolean |
| `idea_policies/actions.py:2202` verify | **highest priority — move `reasoning` before `verdict`** | SHIPPED is −0.211 (p=0.008) on 7b, −0.158 on nano |
| `idea_policies/actions.py:2101` merge `goal_achieved` | move `goal_evaluation` before the boolean | A2 direction, 5/0 |
| `evaluation.py:60` single score | move `rationale` before the score | A2 direction, 5/0 |
| `idea_policies/actions.py:1091` link select | **leave** — no justification asked | consistent with A0 |

**Where a `reason` field is genuinely consumed** (logged for diagnosis, fed into
a later node), do not drop it — move it in front of the answer. The cost is
similar to A1 and the accuracy direction is favourable on every model measured.

**On the anti-calibration that motivated the cycle:** this bench cannot confirm
or refute it. `CONFIDENCE_JUDGE_MISCALIBRATION.md` concerns a *continuous
confidence score* correlated against run outcomes; `promptbench` measures
*discrete accuracy* on constructed items. The mechanism remains plausible and
untested. Testing it needs a confidence-calibration family — noted for the next
cycle, not claimed here.

---

## Limits, stated plainly

- **Two families, both constructed.** `link_select` and `extract_value` as
  designed are **not buildable from this repo**: `agent/idea_test_results/`
  (3.3 GB) stores only aggregates (`observability.visit` is
  `{"count": 1, "chars": 26142}`) — no page text, no URLs, no link sets, and the
  telemetry sidecars hold a ~200-char completion prefix with no prompt.
- **Small n.** 38 verify / 19 select items, 19 clusters each. LOCO swings run
  1.5–5.3 pp, under the 10 pp exclusion bar, but every aggregate rests on 5
  models.
- **Ceiling effects on `select`.** `qwen2.5:7b`, `gpt-4.1-nano` and
  `flash-lite` all sit at 0.84–0.95, compressing arm differences.
- **One rep, T=0.** Provider nondeterminism is unquantified; the planned rep-2
  replicate was dropped for wall clock.
- **These are offline judgement items, not agentic runs.** They isolate prompt
  shape at one decision point. Transfer to end-to-end task score is exactly what
  the follow-on A/B has to establish.

## Reproduce

```bash
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.extract_task_specs
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.runner --census
/home/muk/projects/gpu-lock acquire "promptbench" --ttl 14400 --wait
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.runner \
    --models qwen2.5:0.5b qwen2.5:1.5b llama3.2:3b qwen2.5:7b \
    --families verify select --variants A0 A1 A2 A3 A4 SHIPPED F_json G_nostatement
PYTHONPATH=.:services:agent ./.venv/bin/python -m agent.app.promptbench.analyze
```

Raw rows: `agent/idea_test_results/promptbench_runs.jsonl` (local),
`promptbench_api_runs.jsonl` (API), `promptbench_runs_v1_badgrader.jsonl`
(the pre-fix run, kept as the artifact's evidence).
