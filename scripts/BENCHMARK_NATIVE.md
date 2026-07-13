# Benchmarking the native adaptive engine — methodology

_Why the native-engine benchmark differs from the compiled-scaffold campaign, and how to run it
defensibly. Companion: `native_ab_run.sh` (driver), `adaptive_ab_analyze.py` (analysis),
`ADAPTIVE_ENGINE.md` (the engine). Last updated 2026-07-13._

## What changed vs the compiled campaign

The compiled-scaffold benchmark measured *cheap-vs-premium on a fixed, pre-authored plan* — so it
leaned on **fixtures** (record web evidence once, replay to every model) for cross-model determinism,
and on **cost-recovery curves**. The native adaptive engine asks different questions (does adaptivity
help? which mechanism? how does the DAG grow?), and it **explores variably** — so the compiled
tooling doesn't transfer cleanly. This protocol reforms it.

## Five rules (each earned from a real failure)

1. **No fixtures.** A fixture cache is keyed by exact URL+params. The adaptive engine re-expands to
   *different* pages per run/arm, so a cache recorded on one arm misses on another (measured: a
   289 MB record pass produced ~0 effective hits on the other arm). Run **live**. `IDEA_TEST_FIXTURES=off`.

2. **Connector-retry ON in *every* arm.** The Group-2 A/B was confounded because the adaptive arm ran
   during a connector-slowdown window (empty/timeout fetches). Re-expansion recovers from an
   *insufficient* page but not a *failed* tool call, so infra noise hit it hardest. `connector_retry`
   (C1a) absorbs transient empty/timeout fetches; enabling it in **both** arms removes infra luck as a
   variable, isolating the adaptive-reasoning effect. `IDEA_TEST_CONNECTOR_RETRY=1`.

3. **Interleave arms per (task, repeat).** Run baseline and adaptive for the same task adjacent in
   wall-clock time so they share the same network window — the driver loops `task → rep → arm`.

4. **Arms are named profiles, not hand-edited settings.** `IDEA_TEST_ARM=baseline|good_adaptive|
   reexpand_only|confidence_only|backtrack_only|kvote_only|full` (R1). The A/B differs in adaptive
   *reasoning* only: `good_adaptive` = reexpand + confidence-judge + confidence-reexpand +
   corrective-context + tool-failure-recovery; k-vote and backtrack are **held out** as separate
   ablation axes so the headline delta is attributable to adaptivity, not a bundled accuracy lever.

5. **R≥5 + CI-disjoint, full vectors.** A "win" is a CI-disjoint separation at R≥5 (not a higher
   mean), reported with the full score vector. R=1 "wins" have dissolved before; R=3 variance is wide
   on these tasks (see `adaptive_ab_analyze.py`'s CI column).

## Scoring integrity

The `adaptive_targeted` keystone gates require **grounding evidence** (`visit.count>0`) for keystone
credit — an ungrounded parametric-memory guess scores ~0, not partial credit. Without this, a baseline
that guesses the right number from memory inflates and muddies the delta. (Fixed 2026-07-13.)

## Two engine fixes that reshaped the baseline (read before comparing to older numbers)

- **C1d grounding-enforcement** forces a visit for grounding mandates in *both* arms — it lifted the
  non-adaptive baseline substantially (a big chunk of the original G0/G1 "0.20" baseline was this
  *bug*, not a fundamental non-adaptive weakness). The honest delta is measured against the *fixed*
  baseline.
- **The grounding keystone gate** (above) removes ungrounded false-positives from *both* arms.
So: the defensible adaptive advantage is smaller than the pre-fix G0/G1 headline — and that's the
point of measuring it after the engine was frozen "well done."

## How to run

```bash
# honest A/B (adaptivity isolated), 8 tasks across 4 archetypes, R=5
ARMS="baseline good_adaptive" TASKS="122 125 128 130 134 138 140 144" \
  MODEL="openai/gpt-5-mini" R=5 RUN_ID=nativeab USD_CEILING=8.00 scripts/native_ab_run.sh

# Theme-3 mechanism ablation (add arms; they roll up per run-id)
ARMS="baseline reexpand_only confidence_only good_adaptive" TASKS="140 144" R=5 ... scripts/native_ab_run.sh
```

Env recipe (keys.env CRLF-strip, chroma:8001, concurrency=1, parallel_action_limit=1) is baked into
the driver. Output lands in `services/agent/idea_test_results/_<RUN_ID>/` with an `analysis/` folder
(per-archetype deltas, CI-disjoint verdict, Cohen's d, conditional lift, diagrams).

## What the analysis reports (`adaptive_ab_analyze.py`)

Per-task and per-archetype: baseline vs adaptive mean ± CI95, Δ, **CI-disjoint significance**,
**Cohen's d**; **conditional lift** (adaptive score when re-expansion fired vs didn't); DAG node
count, `visit.chars` (context), $/run. Reads any comma-joined run-ids at any R.

## Known cost/latency caveat

The `good_adaptive` config is heavy (~$0.10/run, 8–15 min on gpt-5-mini, concurrency=1). A full
24-task × many-arm × 2-model × R5 matrix is >$100 and many hours — size runs accordingly, or validate
the throughput/concurrency mode first (deferred; see the plan's C4). Interactive-serving optimization
(fast/deep tiered modes) is explicitly out of scope for the benchmark and deferred.
