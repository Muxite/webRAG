# DAG v2 Phase 1: evidence-ledger foundations

**Date:** 2026-08-25 · **Branch:** `dagv2-evidence-ledger` (not yet merged — hold for full
Phase plan completion + full benchmark pass, per explicit instruction)
**Model:** openai/gpt-4.1-nano (live A/B) · **Spend:** $0.1355 (30+30 cells, well under $5 cap)
**Tests:** 6365 → 6456 passing, 18 skipped

## 1. What this cycle was for

A design review of the DAG v2 engine concluded it calls itself a graph but executes as a
shallow tree: siblings can't see each other's findings, search and evidence are conflated,
the scheduler is node-centric, and the output contract could report `success=True` on a run
that never achieved its goal. The long-term fix is a shared evidence graph + task ledger +
budget-aware scheduler, arrived at in three phases (Phase 1: improve without a rewrite,
Phase 2: replace expansion/merge with the ledger/scheduler as control authority, Phase 3:
per-model-family scaffold tuning). This cycle did **Phase 1 in full** — six items, all
default-off except the honest-output-contract fix and the malformed-JSON repair loop (both
pure bug fixes with no new opt-in surface). Full plan: see
`/home/muk/.claude/plans/the-current-design-has-eventual-phoenix.md`.

## 2. What landed (7 commits on `dagv2-evidence-ledger`)

| Item | Commit | Summary |
|---|---|---|
| 0 | `2be652cc` | `RunPolicy` config resolver (seam for the rest) |
| 1 | `2e22bb93` | Honest output contract: `execution_completed`/`deliverable_complete`/`grounding_satisfied`/`coverage_ratio`/`claim_verification_ratio`/`finalization_status`; fixed `success=True` despite `goal_achieved=False`; fixed the `grounded` double-write |
| 2 | `d0d54214` | Observe-only `TaskLedger`, default off, agrees with existing coverage gate on every fixture |
| 3 | `4334ea79` | Search-must-yield-visit remediation (generalizes `inject_coverage_visits`), default off, flag-off proven byte-identical via full node fingerprinting |
| 4 | `3f1ad322` | Sibling-context ledger delta in expansion prompts, default off, entity-count capped |
| 5 | `cb3e3694` | One repair attempt before three silent JSON fail-open sites give up — always-on |

## 3. Live A/B on item 3 (search-must-yield-visit) — no signal, and why

6 breadth tasks (052/053/152/153/156/157) × 2 arms × 5 reps = 60 cells, nano, concurrency=1.
**The mechanism never got a live trigger**: across 84 SEARCH actions, only 1 returned zero raw
results (in the control arm, before treatment even started) — 0 of 30 treatment cells produced
a completed SEARCH node with zero visit-worthy results, so `inject_empty_search_followup`
never fired. Flag-on and flag-off ran byte-identical code in every treatment cell. Score deltas
(control 0.494 vs treatment 0.480, pooled) are well inside per-task noise (stdev 0.03–0.26) and
not attributable to the flag. `coverage_ratio` was 1.0 in all 60 cells regardless of arm or
actual visit-count shortfalls (e.g. task 052 had `visit_count` grep-check failing at 1/6 while
`coverage_ratio` read 1.0) — this field needs sharpening in a follow-up, it's currently too
coarse to discriminate real coverage gaps.

**Retry recommendation if this needs live validation later**: use a query condition or model
more likely to genuinely dead-end a SEARCH (nano's queries on Wikipedia-shaped entities almost
always return *something*, even when useless), or a synthetic/injected-empty-results fixture
rather than relying on live incidence (too rare at n=84).

## 4. New bug found, not investigated (filed for a future cycle)

**VISIT-running-before-its-SEARCH-dependency-completes**, observed 11 times across the 30
treatment-arm cells (`search_nodes_checked=0` / `(pending)` in VISIT's "No URLs extracted from
search results" warning path). This is a distinct failure mode from item 3's remediation target
(item 3 only fires *after* a SEARCH completes with nothing; this is a VISIT racing ahead of an
incomplete SEARCH) — looks like a real, reproducible scheduling/ordering gap in the native graph
engine's step sequencing, not a one-off artifact. Worth its own investigation
(`engine-dev`/`strategy-tuner`) in a later cycle: 11/30 ≈ 37% of cells hit it, which is a high
enough rate to matter for anything downstream that assumes SEARCH-before-VISIT ordering holds.
Not chased in this cycle per explicit instruction — filing only.

## 5. Status / next steps

Branch stays unmerged. Per instruction: continue development on `dagv2-evidence-ledger`
(not a fresh branch) toward the fuller plan (Phase 2 and beyond), and only open a PR/merge
once the plan is judged complete and a full benchmark pass has been run — not after each
sub-cycle.
