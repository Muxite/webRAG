# Model Tier List — Approximate, Evidence-Based

A practical reference for "which model is expected to be better at what," built from two rounds
of live experiments on badmodel-lab's task suite (E1, 2026-08-03; Round 2, 2026-08-04). This is
**approximate, not a rigorous statistical study** — sample sizes are R=1-3 per cell, one task
tier at a time, and a couple of the findings below are flagged as provisional pending a follow-up
fix. Treat it as a starting expectation to update as more data comes in, not a settled ranking.

**Read the caveats section before trusting any single number here** — in particular, deepseek's
placement is likely distorted by a harness bug, not a clean capability read.

## Approximate tier list

**Tier A — frontier-adjacent, minimal mitigation needed.**
- `openai/gpt-4.1-nano` — consistently excellent everywhere tested (format 0.82–0.89, reachable
  0.97, hard 0.99). No severe weakness found in this data.
- `google/gemini-2.5-flash-lite` — excellent overall (format up to 1.00, hard 0.99, micro 0.83),
  with one sharp, isolated, task-specific miss (see below) rather than a general gap.

**Tier B — strong for a free local model, two known real gaps.**
- `qwen2.5:7b` (local, $0) — ties or nears Tier A on most reachable/hard tasks, but has two
  distinct, reproducible reasoning gaps (negation/odd-one-out; k-th-ordinal). Worth the full
  mitigation stack, not worth writing off as "weak."

**Tier C — provisional, likely under-measured, needs a harness fix before re-judging.**
- `deepseek/deepseek-v4-flash` — scored lowest across every tier tested (reachable 0.53, hard
  0.64, micro 0.61, format 0.74–0.82), but the failure signature (more visits AND lower scores
  than nano/flash-lite on identical tasks, deliverables that explicitly say facts came back
  UNKNOWN) points at a starved per-call token budget, not a capability gap — see caveats.

**Not retested this round** — the smaller `roster.yaml` subjects (tinyllama, qwen2.5:0.5b/1.5b,
llama3.2:1b/3b, gemma2:2b, phi3:mini) already have extensive historical data from prior sessions
showing they floor most of this suite without the lab's mitigation stack; not re-run here since
nothing in this round's scope would change that picture.

## Per-model strengths & weaknesses

### `openai/gpt-4.1-nano` — Tier A
No severe weakness surfaced across format/reachable/hard tiers in ~35 task-runs total (E1 +
Round 2). Cheapest of the three API models tested ($0.0002–0.0056/task-run depending on tier).
The safe default recommendation among the models tested here.

### `google/gemini-2.5-flash-lite` — Tier A (one sharp gap)
Strong across the board, including a perfect 1.00 under the `fs1_structured_strict` format
profile. **One clean, reproducible miss**: task 062 (page-only topographic-prominence argmax),
0/3 reps correct, consistently naming the wrong peak with an implausible number ("Mount Gongga...
at 11,949 m"). Every *other* argmax-shaped task (078, hard-tier 077) it nails at 0.93–1.00 — this
reads as a specific fact/page-disambiguation miss on one task, not a general argmax weakness.
Worth a targeted re-test of just 062 before concluding it's a durable pattern.

### `qwen2.5:7b` (local, free) — Tier B
The standout free option. Ties gpt-4.1-nano on 5 of 7 E1 reachable-tier tasks. Two real,
reproducible gaps, both plausibly a "smaller open-weight model" signature rather than a
price-tier effect (see below — flash-lite handles the same shape cleanly):
- **Negation/odd-one-out** (task 069): 0.33 in E1 — self-contradictory across self-consistency
  samples (one run labels every candidate "NOT landlocked"; another produces the literally
  contradictory "Austria: NOT landlocked -- coastline on none (landlocked)").
- **K-th-ordinal reasoning** (hard-tier task 075, k-th-largest): the one clear miss in this
  round's hard-tier pass (0.87 mean, dragged down specifically by 075).

### `deepseek/deepseek-v4-flash` — Tier C (provisional, see caveats)
Lowest scores across every tier tested. **The evidence points at a measurement problem, not a
clean capability read** — see the caveats section for the full reasoning. If the suspected token-
budget issue is fixed and this model re-tests similarly low, move it down with confidence; until
then, treat this placement as unconfirmed. One genuine (not budget-related) miss did surface:
a disambiguation failure on task m02, visiting the Wikipedia page for the city of Amsterdam
instead of Amsterdam Island and returning UNKNOWN — a real miss, not a scoring artifact.

## Score table

| Model | Tier | Mean score | Keystone % | $/task-run | n |
|---|---|---|---|---|---|
| gpt-4.1-nano | format (fs0/fs1/fs2) | 0.82 / 0.89 / 0.89 | 100/100/100 | $0.0002–0.0007 | 9 each |
| gpt-4.1-nano | reachable | 0.97 | 100 | $0.0056 | 21 |
| gpt-4.1-nano | hard | 0.99 | 100 | $0.0050 | 5 |
| gemini-2.5-flash-lite | format (fs0/fs1/fs2) | 0.82 / 1.00 / 0.89 | 100/100/100 | $0.0002–0.0008 | 9 each |
| gemini-2.5-flash-lite | reachable | 0.87 | 86 | $0.0061 | 21 |
| gemini-2.5-flash-lite | hard | 0.99 | 100 | $0.0055 | 5 |
| gemini-2.5-flash-lite | micro | 0.83 | 100 | $0.0007 | 9 |
| qwen2.5:7b (local) | reachable | 0.85 | — | $0 | 21 (E1) |
| qwen2.5:7b (local) | hard | 0.87 | 80 | $0 | 5 |
| deepseek-v4-flash | format (fs0/fs1/fs2) | 0.74 / 0.82 / 0.78 | 100/100/89 | $0.0003–0.0014 | 9 each |
| deepseek-v4-flash | reachable | 0.53 | 38 | $0.0037 | 21 |
| deepseek-v4-flash | hard | 0.64 | 60 | $0.0037 | 5 |
| deepseek-v4-flash | micro | 0.61 | 67 | $0.0007 | 9 |

Total live spend across both rounds: E1 $0.1245 + Round 2 $0.3343 = **$0.4588**, against a
combined $5 authorized ceiling.

## Caveats — read before trusting a number above

1. **Deepseek's low scores are likely a harness measurement problem, not (only) a capability
   gap.** `badmodel-lab/run_cell.sh` always runs the `graph_compiled` variant
   (`testing/execution_compiled.py`), whose OWN `_is_reasoning_model` (line 366) only matches
   `gpt-5*/o1/o3/o4*` — it does **not** include deepseek, unlike the native engine's
   `model_tiers.py::is_reasoning_model`, which explicitly does (added from live telemetry:
   OpenRouter bills deepseek's reasoning tokens *inside* `completion_tokens`). Deepseek prices
   into execution_compiled.py's "cheap" tier, which gives it the same 24-token thin-extraction
   budget as a 0.5B local model — with no reasoning-token floor. Circumstantial evidence is
   consistent with starvation, not incapability: deepseek burns *more* visits than nano on
   identical tasks (task 076: 17–23 visits vs. nano's steady 12) yet scores lower, and one
   deliverable explicitly says the aggregation couldn't complete because gathered facts were
   "labeled UNKNOWN." Not 100% confirmed (the result JSONs don't record raw `finish_reason`), but
   strong enough that deepseek's placement here should be treated as provisional. **Follow-up**:
   either mirror `model_tiers.is_reasoning_model`'s deepseek coverage into
   `execution_compiled._is_reasoning_model`, or re-test deepseek under `react` leaf mode instead
   of `thin`, and see if the reachable-tier score recovers.
2. **`tiers.yaml`'s "hard tier floors even nano" claim didn't hold in this data** — nano scored
   0.99 and flash-lite 0.99 on the hard tier this round; neither floored. Worth a doc update, not
   chased further here.
3. **Task 069 negation reasoning is not a general cheap-model weakness** — flash-lite scored a
   clean 1.00/1.00/1.00 on the exact task where qwen2.5:7b scored 0.33 in E1, and deepseek
   partially recovered (mean 0.56, unstable). This looks specific to qwen2.5:7b (or local
   open-weight models generally), not a price-tier or "cheap model" effect broadly.
4. **Task m02's `grounding` check has a scoring artifact**, unrelated to model capability: it
   requires an exact URL match against `PAGE_URL`, but Wikipedia's canonical redirect target
   (`/wiki/%C3%8Ele_Amsterdam`) differs from the literal string the task checks for
   (`/wiki/Amsterdam_Island`) — both flash-lite and deepseek extracted the correct fact (56.6 km²)
   from the right page but scored 0.0 on grounding anyway, capping their overall score at 0.5
   despite a correct answer. Fix candidate:
   `services/agent/app/idea_tests/test_m02_amsterdam_area.py`'s grounding check should accept the
   known redirect target.
5. **Small samples.** R=1-3 per cell. Treat single-digit-percentage differences as noise; the
   patterns called out above (069, 062, 075, deepseek's broad depression) are the ones consistent
   enough across reps to trust as real signal.

## Sources

- E1 (2026-08-03): `badmodel-lab/results/cells.jsonl` run_ids
  `bml__qwen2.5-7b__m1_thin__reachable`, `bml__openai-gpt-4.1-nano__m1_thin__reachable`. Full
  writeup: `services/agent/app/AGENT_CONTINUUM.md`'s E1 section.
- Round 2 (2026-08-04): `badmodel-lab/results/cells.jsonl` run_ids prefixed
  `bml__openai-gpt-4.1-nano__m1_thin__f*`, `bml__deepseek-deepseek-v4-flash__m1_thin__*`,
  `bml__google-gemini-2.5-flash-lite__m1_thin__*`, plus a `qwen2.5:7b` hard-tier cell. Raw result
  JSONs in `services/agent/idea_test_results/bml__*.json` (gitignored, local to whichever
  environment ran them).
