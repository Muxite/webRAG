# Combined local live A/B: four newly-shipped opt-in mechanisms (2026-08-22)

## What ran

`ladder_final_20260822`, launched via the new `ladder-benchmark` docker compose profile
(`services/docker-compose.yml`) rather than the host `.venv`, per this cycle's containerization
work (see `project_docker_test_benchmark_containerization` memory / commit `ebc414a9`):

```
cd services && docker compose --profile ladder-benchmark run --rm ladder-benchmark \
  --run-id ladder_final_20260822 --axis tracemech_local --task-set core24 \
  --arms good_adaptive,good_adaptive_tracemech,good_adaptive_safetynet,good_adaptive_beamraw,good_adaptive_backtrackrel \
  --jobs 8
```

`core24` tasks x R3 x 5 arms = 360 cells, `qwen2.5:7b` via `badmodel-ollama` (local, $0 spend),
embedded per-cell Chroma. Started 09:39 UTC, completed 18:32 UTC (~8h53m wall clock — the
driver's `--jobs 8` had no effect on throughput since `badmodel-ollama` holds one loaded model
at a time; cells ran effectively serially against the GPU). All 360 cells produced a result,
$0.00 spent throughout. Analysis: `scripts/analyze_ladder_final_20260822.py`, pairing each arm
against `good_adaptive` (the shared baseline) on `(task, rep)`, matching this session's
established E1/E3/T1-4 methodology.

## Results

| Arm | Mechanism | Δ score | sd | n | t | p | W/T/L | Verdict |
|---|---|---|---|---|---|---|---|---|
| `good_adaptive_backtrackrel` | T1-6: backtrack + relative dead-end threshold | **+0.048** | 0.185 | 70 | 2.18 | **0.033** | 20/38/12 | Positive, needs confirmation before flip |
| `good_adaptive_safetynet` | D4 retry-after-skip + D6 early-exit-respects-grounding | +0.057 | 0.273 | 71 | 1.75 | 0.085 | 24/31/16 | Promising, not yet significant |
| `good_adaptive_tracemech` | numeric_provenance + race_value_agreement + chain_closure | +0.023 | 0.150 | 69 | 1.28 | 0.205 | 23/34/12 | No measured gain here |
| `good_adaptive_beamraw` | T1-5: beam spread reads raw_score | +0.005 | 0.188 | 69 | 0.22 | 0.827 | 15/40/14 | Clean null |

3 cells across the run were infra-failed (excluded from all four comparisons) and are not
double-counted anywhere above.

## Reading the results

**T1-6 (backtrack) is the standout.** It's the only mechanism to clear conventional
significance, and directionally it's the opposite of this session's earlier default caution:
E1 (dedup) and T1-4 (sibling-visit dedup) both found that *spending more compute* (wider
fan-out, more candidates) bought little at this tier. T1-6 instead un-breaks a mechanism that
was previously mechanically *inert* (0 real backtrack fires at the old absolute threshold of 5,
because no graph in the corpus ever got deep enough to reach it) — the win here isn't "spend
more," it's "let an already-designed safety net actually engage." That's a meaningfully
different, and more encouraging, story than the other three.

**Safetynet (D4+D6) trends the same direction but isn't conclusive at this n** (p=0.085). Both
component mechanisms are "decline a premature/contradictory termination" gates — plausible that
they help, plausible that 71 pairs just isn't enough to separate the effect from noise. Worth a
second run before deciding either way, not worth writing off.

**Tracemech (the 3 "READY" strong-agent-trace mechanisms) shows no measured accuracy gain**
here, despite each of numeric_provenance/race_value_agreement/chain_closure individually
clearing capture-replay + isolated live-probe + end-to-end composition validation earlier this
session. This does not contradict the earlier validation — those checks confirmed the
mechanisms detect what they claim to detect and don't double-penalize; they never claimed
detection alone would move the *task score* metric, which depends on how often the underlying
failure mode (a fabricated numeric claim, a wrong race pick, an unclosed chain) actually
appears in this specific task set at this specific tier. Recommend holding all three at their
current opt-in-off default; the individual mechanisms remain correctly implemented and may
still matter at a different task mix or model tier.

**Beamraw (T1-5) is a clean, uninteresting null** — exactly the risk flagged in the fix's own
`ASSUMPTION_AUDIT.md` entry before this run: widening the beam trades accuracy for fan-out
cost, and at qwen2.5:7b that trade doesn't pay off in either direction (W/T/L is the flattest of
the four). The fix itself is correct (it does what it says — see
`project_beam_width_raw_score_fix`); the accuracy hypothesis it was meant to test just doesn't
hold at this tier. Recommend holding the default off.

## What this does and doesn't establish

One local model (`qwen2.5:7b`), one session, `core24` only (24 tasks x R3). None of these four
verdicts should be treated as final without the same second-model/second-session confirmation
this session applied to E1 (dedup) and E3 (memory floor) before recommending a default change
in either direction. The honest next step, if anyone wants to act on the T1-6 or safetynet
signal, is exactly that confirmation run — not an immediate default flip.

**Not analyzed in this pass**: cost/efficiency deltas (LLM call count, tokens, wall-clock).
Each result JSON's `execution.duration_seconds` was captured by the analysis script but not
reported here; per-cell token/call counts live in a separate trace file this script doesn't
parse. If anyone extends this analysis, that's the natural next axis, following the T1-4
lesson that a whole-sample cost delta must be split by whether the mechanism actually fired
before crediting it to the mechanism (see `project_sibling_visit_dedup_confirmation`).

## Related memory / prior work

- `project_beam_width_raw_score_fix` — T1-5's original fix + contamination analysis
- `project_prune_backtrack_deadzone_fix` — T1-6's original fix + depth-corpus analysis
- `project_d6_early_exit_grounding_fix` — D6, half of the safetynet bundle
- `project_strong_agent_trace_guidance` — the 3 tracemech mechanisms' original validation
- `project_docker_test_benchmark_containerization` — the `ladder-benchmark` profile this run used
- `project_dedup_ablation_confirmation`, `project_sibling_visit_dedup_confirmation` — the two
  prior "spend more compute" experiments this run's null/positive results are compared against
