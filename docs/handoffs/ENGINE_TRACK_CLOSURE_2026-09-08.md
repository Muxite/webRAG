# Engine track closure (2026-09-08)

**Status:** decision record. Closes the DAG track of
`docs/superpowers/plans/2026-09-08-ledger-dag-replan.md` (Phase 4) on the pre-declared Phase 0
rules, from stored cells only. $0, offline, no GPU, no campaign launched.

**Decision:** no D1 ablation, no D2 campaign, no D3 build. DAG v2 stays as it is; the Ledger
remains bound to linear hosts (`sequential_react`, `langgraph_react`). Engine work stops here
unless a new prereg with a new hypothesis reopens it.

Reproduce: `scripts/shape_subset_ab.py` (paired, task-clustered comparison on a task subset
drawn from `agent/app/testing/task_shapes.json`; see its `--help`). Registry rule and counts:
`scripts/task_shapes.py`.

## 0b — does the −0.461 aggregation deficit survive a mechanical registry?

Yes. The registry is now a code-stated rule (`scripts/task_shapes.py`), not the lost hand
classification; it yields aggregation 40 / chain 17 / other 2 (the 2026-08-30 hand split was
23 / 33 / 3; the difference is 16 survivor-elimination and cross-source-comparison tasks the
rule counts as aggregation). Suite59 three-way run `night_a1_{g,sr,lg}` (qwen2.5:7b, 59 cells
per arm, 1 rep).

`graph − sequential_react`:

| subset | n | delta | sd | t | W/T/L | graph | seq_react |
|---|---|---|---|---|---|---|---|
| all suite59 | 59 | −0.188 | 0.356 | −4.05 | 14/6/39 | 0.400 | 0.588 |
| aggregation (rule, 40) | 40 | −0.310 | 0.312 | −6.28 | 5/2/33 | 0.322 | 0.632 |
| aggregation excl. survivor/cross-source (24) | 24 | −0.424 | 0.295 | −7.04 | **0/0/24** | 0.221 | 0.646 |
| breadth sub-flag (19) | 19 | −0.476 | 0.295 | −7.04 | **0/0/19** | 0.219 | 0.695 |
| survivor / cross-source only (16) | 16 | −0.138 | 0.259 | −2.13 | 5/2/9 | 0.473 | 0.611 |
| chain (17) | 17 | +0.075 | 0.325 | 0.94 | 8/3/6 | 0.593 | 0.519 |

`graph − langgraph_react` on the same subsets: aggregation-40 −0.378 (t −6.14), narrow-24
−0.452, breadth-19 −0.464, chain-17 −0.005.

The headline −0.188 / t −4.05 / 14/6/39 reproduces the 2026-08-30 document exactly. The
deficit is concentrated where the rule says fan-out lives: the graph arm loses **every one** of
the 24 narrow-aggregation and 19 breadth tasks. The empirical sd of paired deltas on the
aggregation subsets is 0.29–0.31; the review's estimate (0.286) was right.

Side finding: the registry flags 19 suite59 tasks as breadth-shaped (≥3 independently
fetchable arms), against the repo's prior "~1 breadth task in 59". That prior was a statement
about `core_long24` (where only 041/052 qualify), not the suite; both are correct in scope.

## 0c — would the deterministic coverage executor rescue the aggregation shape?

No. `gpu0831b` (qwen2.5:7b, `core_long24`, 4 reps, clustered to per-task means, 24 tasks) already
contains the D2 contrast. Pre-declared rule (replan Phase 0c): lower 95% bound of
`evidence_loop − sequential_react_extract` ≥ +0.20 → campaign; ≤ 0 → close.

| subset | n | delta | sd | t | p | W/T/L | evidence_loop | seq_react_extract |
|---|---|---|---|---|---|---|---|---|
| all core_long24 | 24 | +0.045 | 0.217 | 1.01 | 0.33 | 13/2/9 | 0.445 | 0.400 |
| aggregation (rule) | 15 | +0.046 | 0.260 | 0.68 | 0.50 | 7/1/7 | 0.511 | 0.465 |
| narrow aggregation | 9 | +0.054 | 0.277 | 0.59 | 0.54 | 4/1/4 | 0.515 | 0.461 |
| breadth | 9 | +0.056 | 0.276 | 0.61 | 0.51 | 4/1/4 | 0.525 | 0.469 |

`evidence_loop − langgraph_react`: all 24 −0.003 (t −0.06); aggregation-15 +0.039; breadth-9
+0.144 (t 1.61, p 0.15, 6/0/3).

The 95% CI on the aggregation-15 delta is roughly [−0.10, +0.19]: the +0.20 bar the D2 design
was built around is excluded, and the point estimate is a twentieth of the graph deficit it was
meant to close. The typed-record executor is, on this shape and this model, a wash against a
plain extraction-only linear loop. The graph deficit is therefore not a representation problem
the coverage executor fixes; the linear arms simply win.

## What this closes, and what it does not

- Closed: D1 (configuration repair — its judge target was already inert, see the review), D2
  (four-arm or otherwise), D3 (rows-as-work-items DAG v3), D4 (learned stop/continue). The
  `MechanicalEvaluationPolicy`, the six `agg_*` profiles, the page-body replay shim and the
  aggregation corpus freeze are not built.
- Not closed: the merge `goal_achieved` judge (AUC 0.288) remains default-on in the graph arm.
  Nobody runs the graph arm on the ledger line, so it is a latent defect, not a live one; it is
  recorded, not scheduled.
- Not claimed: anything about models other than qwen2.5:7b, or about breadth mechanisms on a
  breadth-only suite that does not exist. The 19-task breadth sub-population is now countable
  if someone wants to author that suite later.

## Consequence for the replan

Phase 4 is done. All remaining effort — and the GPU — goes to the Ledger track: Phase 2
(`host_derive` rebuilt on the slot parser, hand-rule ranker, argmax, unit-aware agreement) and
its offline replay over 768 stored cells before any mint03.
