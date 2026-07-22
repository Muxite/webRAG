# Compute-ladder A/B — pre-registered analysis plan (write-before-you-see-results)

_Registered 2026-07-22, before the R=5 live matrix was analyzed. Driver: `scripts/adaptive_ladder_run.py`.
Analysis: `scripts/adaptive_ab_analyze.py` (paired sign-flip test). This file exists so the win/lose
call is a rule decided in advance, not a story fit to whatever the numbers happen to say._

## The claim under test

**One cheap "bad" model can be made materially better at agentic web-research by burning more of its
(cheap) tokens/searches** — via an adaptive agent that re-expands onto better pages, re-grounds when
step-confidence is low, and self-consistency-votes — and it can approach a premium model's quality at
a fraction of the premium's cost.

## Design (fixed before results)

- **Agent model:** `openai/gpt-5-mini` (cheap), `execution_variant=graph`, in a 3-arm compute ladder:
  `baseline` (adaptive OFF) → `good_adaptive` (re-expand + confidence re-ground + corrective context)
  → `full` (+ k-vote×3 + backtrack + expect-contract = max searches/tokens).
- **Reference bar (not used by the agent):** `google/gemini-3.1-pro-preview` + `sequential_react`.
- **Tasks:** 122 125 128 130 134 138 140 144 — all four adaptive archetypes (A survivor, B conflict,
  C chain, D re-expand), incl. the D flagship that was missing from prior data.
- **Replication:** R=5 per cheap cell (interleaved, shared network window), R=3 for the reference.
- **Fairness (held fixed across arms):** connector-retry ON, fixtures OFF, `parallel_action_limit=1`,
  each cell an isolated process at internal concurrency=1.

## Pre-registered success criteria

1. **PRIMARY — adaptivity lifts the cheap model.** Paired sign-flip permutation test (two-sided) on
   per-(task,rep) deltas, `good_adaptive − baseline` AND `full − baseline`: **win = mean Δ > 0 with
   p < 0.05.** Report the conservative per-task pairing (n = #tasks) alongside as a robustness check;
   a claim is "strong" only if both pairings agree in sign.
2. **SECONDARY — compute scales monotonically.** Overall mean score ordering `full ≥ good_adaptive ≥
   baseline`. A non-monotonic result (e.g. full < good_adaptive) is reported as a real finding
   (over-spending hurts), not hidden.
3. **SECONDARY — the cheap model is made "decent" vs the premium bar.** Report the best cheap arm's
   mean score as a % of the reference mean, and its $/run as a fraction of the reference $/run.
   Pre-registered headline threshold for "decent": **≥ 85% of reference quality at ≤ 1/3 of reference
   cost.** (Actuals reported regardless of threshold.)
4. **HONESTY.** Per-archetype deltas reported in full, **including regressions** (C-chain is the known
   soft spot). Full score vectors, $/run, and $/solved (score ≥ 0.75) reported. No task dropped
   post-hoc; asymmetric-n cells flagged.

## Known risks logged in advance

- C-chain may regress or stay flat (blocked/redirected multi-hop fetches, not reasoning) — expected;
  it is a diagnosis target, not a reason to drop the archetype.
- `full` may cost 2–4× baseline. Per the thesis that is acceptable **iff** it buys accuracy and stays
  well under the premium's cost; $/solved is the honest efficiency metric.
- gpt-5-mini is cheap but not the cheapest; nano (truly "bad") is a logged follow-up, not in this run.
