# Compute-ladder A/B — pre-registered analysis plan (write-before-you-see-results)

_Re-registered 2026-07-24, before the clean barrage is analyzed — supersedes the 2026-07-22
version (which was written for the old 8-task, single-model `gpt-5-mini` ladder and predates the
survivorship / multiple-comparison / grounding-decomposition fixes). Driver:
`scripts/adaptive_ladder_run.py`. Analysis: `scripts/adaptive_ab_analyze.py`. This file exists so the
win/lose call is a rule decided in advance, not a story fit to whatever the numbers happen to say._

## The claim under test

**One cheap "bad" model can be made materially better at agentic web-research by burning more of its
(cheap) tokens + page-visits** — via an adaptive Graph-of-Thoughts agent that re-expands onto better
pages, re-grounds when step-confidence is low, and reconciles its answer (recompute / verify / decorrelated
variations). We do **not** claim to beat the premium reference — the claim is that the cheap model
**approaches** it at a **fraction of the cost**. The 2–5× token premium is the STRATEGY, not a defect.

## Design (fixed before results)

- **Cheap agent models (the subjects):** `openai/gpt-4.1-nano` and `deepseek/deepseek-v4-flash` —
  genuinely weak/cheap, so the lift is meaningful. `execution_variant=graph`.
- **Structured arms per cheap model (the ladder):**
  `baseline` (adaptive OFF) → `good_adaptive` (re-expand + confidence re-ground + corrective context +
  tool-recovery — the proven winner) → `max_burn` (good_adaptive + deeper re-expansion depth + wider
  hop/beam + the finalize reconcile chain = the productive ~5× burn). The old `full` arm (k-vote +
  backtrack + expect-contract) was measured net-negative and is DROPPED.
- **Naive floor (context):** an iterative `naive_rag` that keeps searching/visiting/hopping until it
  answers or hits a bounded budget — the honest floor a cheap model reaches WITHOUT the graph, so the
  graph arms' lift over it isolates the value of the structure, not mere persistence.
- **Reference bar (NOT used by the agent):** `anthropic/claude-sonnet-5` + `sequential_react` — the
  quality ceiling we approach, not beat.
- **Sequential-mode arm dropped (2026-08-15):** DAG v1 used a bare `sequential`/`chain`/`cot`
  execution-variant arm for its comparison; DAG v2 drops it — `sequential` is the dead 0-visit
  legacy path (see `execution_sequential.py`'s alias table), distinct from `sequential_react`
  above which stays as the non-agent reference bar only. `sequential`/`chain`/`cot` must never
  appear in `--variant`/`IDEA_TEST_EXECUTION_VARIANTS` for this relaunch.
- **Tasks:** the 59-task validity suite (4 adaptive archetypes as the spine + diverse-shape coverage);
  see `BENCHMARK_SUITE_50.md`. Power comes from the TASK count, not reps.
- **Replication:** R=5 per cheap cell (interleaved, shared network window), R=3 for the reference.
- **`diverse_ground` A/B folded in (2026-08-15):** `STAGED_BARRAGE_PLAN.md`'s Phase 3b (staged
  2026-07-08, never run) shares this relaunch's run-id/budget bookkeeping instead of a separate
  pass: tasks 055-060, `openai/gpt-4.1-nano` + `openai/gpt-5-mini`, R=3, ~$2, four invocations
  (`parametric` / `graph` / `graph_compiled` with `IDEA_TEST_COMPILED_AGG_MODE=single` /
  `graph_compiled` with `IDEA_TEST_COMPILED_AGG_MODE=diverse_ground`). Gate: `diverse_ground >=
  single` on the math/reasoning tasks (055/059/060). Prior evidence (2026-06-27 optscan run,
  `test_058`'s pinned `agg_mode="single"`) already leans toward regression — this is a
  confirmatory re-test on different models, not a re-run of known results.

- **Preflight blockers (2026-08-15):** an adversarial review found four benchmark-invalidating
  bugs (worst: grounding evidence sourced from `result["graph"]`, which only two variants emit —
  measured 0.944 → 0.417 on a real `sequential_react` cell, enough to invert the headline) and
  retracted the "task 024's hallucination provably passes 0.75" claim. All fixed; **the relaunch
  still has open items before it runs.** See `docs/handoffs/DAG_V2_PREFLIGHT_2026-08-15.md`.
- **Required for this relaunch:** `IDEA_TEST_REPORT_VERBOSITY=3` if results must stay re-scorable
  (the default strips `telemetry_raw`, and the evidence-scored tasks cannot be re-scored without
  it — `rescore_results.py` now refuses rather than fabricating a regression).
- **Report score, cost AND visits per shape.** The smoke found DAG v2 spending 7–13× the tokens
  while making FEWER tool calls than a linear agent, and tying with LangGraph on fan-out shapes at
  1/13th the cost while winning clearly on chain shapes. Pooling shapes averages that real win
  into a wash; a score-only table hides the cost gap.

## Fairness (held fixed across arms — a fair battle)

- **Search queries sanitized to Brave's limits** (≤400 chars / 50 words, count ≤20) at the connector,
  AND the query-writing prompts tell every agent the limit — so no arm is handicapped by a 422 storm
  (the react reference previously failed 40/40 on over-long queries; the graph arm 0/40).
- **Connector-retry ON in every arm** (symmetric); fixtures OFF; `parallel_action_limit=1`; each cell an
  isolated process at internal concurrency=1 with its OWN embedded chroma (no cross-cell contention).

## Pre-registered ANALYSIS RULES (the methodology, fixed in advance)

1. **Missing = 0 over the full grid.** A cell scheduled by the interleaved design but missing in one arm
   (timeout / crash) scores **0** for that arm, over the UNION grid. No intersection-drop
   ("survivorship" inflated the pilot ~15%, because timeouts cluster on the hard tasks / high-compute
   arm). `missing="drop"` is only ever an explicit, logged sensitivity check.
   **1b. INFRA failures are QUARANTINED, not zero-filled** (amended 2026-08-09, BEFORE the barrage —
   F17). A cell whose web/LLM calls died on 402 / 422 / 429 / 5xx / transport (flagged `infra_failed`
   by the runner) measured the PROVIDER, not the model, so it is excluded from the primary
   aggregation and from the paired grid — pairwise, i.e. its partner cell in the other arm goes with
   it, so no healthy run is scored against a fabricated 0. The count is reported ("infra failures
   excluded: N") and every excluded cell is listed in `ab_infra.csv`; a run whose exclusions are not
   a small minority is not publishable and must be re-run. This carve-out applies ONLY to the
   `infra_failed` classification — an ordinary timeout or crash is still a real 0 per rule 1.
2. **PRIMARY test = TASK-LEVEL.** Paired sign-flip permutation test (two-sided) on per-**task** deltas
   (n = #tasks): **win = mean Δ > 0 with p < 0.05.** The per-(task,rep) pairing is PSEUDOREPLICATED
   (reps within a task are not independent) and is reported ONLY as a secondary robustness figure.
3. **Multiple-comparison correction (Holm) within families.** The per-archetype scan (4 tests) and any
   multi-arm-pair set are Holm-corrected; a result is "significant" only if **p_holm < 0.05**. Raw p is
   printed alongside. Per-archetype (~2–6 tasks) is UNDERPOWERED → reported as exploratory, never a
   confirmatory verdict.
4. **CIs use Student-t** (not z=1.96) for the small-n rows.
5. **Grounding decomposed via the additive Oaxaca split.** Δraw = REASONING (scores better once
   grounded) + GROUNDING-RATE (grounds more often) + ungrounded-residual, terms summing exactly to Δraw,
   using the EMPIRICAL E[score|ungrounded]. Report **whichever term dominates** per model — a model that
   rarely grounds lifts via grounding-rate; one that already grounds ~85% lifts via reasoning. The two
   stories are NOT merged into one "reasons better" headline.
6. **Cost headline = $/solved (score ≥ 0.75)**, plus the best cheap arm's mean as a % of the reference
   mean **on the shared task support, with a CI** — never the disjoint-support number, never the
   rep-level p.

## Pre-registered success criteria

- **PRIMARY:** `good_adaptive − baseline` (and, exploratory, `max_burn − baseline`) task-level Δ > 0,
  p < 0.05, on the full-grid (missing=0) pairing, direction agreeing at rep-level.
- **Compute monotonicity is a FINDING, not an assumption:** if `max_burn ≤ good_adaptive`, that is
  reported (over-spending can hurt), not hidden.
- **"Approaches the reference cheaply":** report the best cheap arm's % of reference quality on shared
  support (with CI) and its $/solved vs the reference's. Threshold framing (reported regardless):
  meaningfully closes the gap at **≤ 1/4 of reference $/solved**. We do not claim to reach 100%.

## Known risks logged in advance

- **The synthesis gap caps the ceiling.** "Right page, wrong value" (Mode-1) is the dominant residual
  and is NOT closed by any burn knob — only by the finalize recompute/verify/variations chain (shipped,
  default-off, on in `max_burn`). If it does not validate, the honest claim is narrower: **burn closes
  the GROUNDING gap, not the SYNTHESIS gap.** "Near-reference" phrasing is gated on the reconcile chain
  actually lifting score in the smoke.
- **The effect is modest** (pilot ~+0.11) and was inflated from both ends — survivorship (fixed: rule 1)
  and a handicapped reference (fixed: Brave-422 fairness). Post-fix the gap shrinks but a defensible
  modest win survives; an overclaim does not.
- **deepseek** must have its reasoning-class / price-tier config bugs fixed or its ladder is noise
  (logged follow-up).
- **C-chain** may stay flat (blocked/redirected multi-hop fetches, not reasoning) — a diagnosis target,
  not a reason to drop the archetype.
