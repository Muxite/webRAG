# Benchmark Task Suite History

`agent/app/idea_tests/` holds 163 task files across 10 distinct authoring dates. Each date is a
batch with a different purpose. Source: `git log` first-add date per file, 2026-08-14.

## Batches

| Era | Date | Count | IDs | Theme |
|---|---|---|---|---|
| 1a | 2026-02-21 | 24 | 001-024 | Original suite, generic research prompts, mostly ungraded |
| 1b | 2026-02-24, 02-27 | 3 | 025-027 | Small one-off additions |
| 1c | 2026-03-02 | 12 | 028-039 | Security research (CVE PoC discovery) + adversarial synthesis |
| 2 | 2026-06-16 | 15 | 040-054 | tier1-tier5 difficulty ladder introduced |
| 3 | 2026-06-27 | 35 | 055-089 | Tier-5 shape taxonomy, breadth pass, many b/c/d/e replicates |
| 4 | 2026-07-10 | 32 | 090-121 | Hand-authored named-entity chain/survivor/eliminate-chain tasks |
| 5 | 2026-07-11 | 24 | 122-145 | Mechanism-targeted: 4 adaptive-engine decision archetypes |
| 6 | 2026-07-29 | 6 | f01-f03, m01-m03 | Format-stress tier, fact held constant, output schema varied |
| 7 | 2026-08-06 | 12 | 146-149, 200-207 | Stacked-axis compound tasks + self-contained reasoning suite |

Total: 163.

## Era notes

**Era 1a (001-024).** Landed with the DAG v1 rewrite. Generic fact retrieval, multi-hop search,
synthesis, wikipedia link collection. Validators are mostly regex/keyword match. Several are
single common-knowledge facts answerable with zero grounding. Later audit marked most of this
cohort invalid: single-fact, non-discriminating, memorized-trivia.

**Era 1c (028-039).** Same day as the "graph of thoughts implemented successfully" commit. Two
halves: real CPU-vulnerability research (Downfall, Retbleed, uarch fuzzing) needing independent
GitHub PoC discovery, and harder multi-source synthesis (8-source fact matrices, 5-topic
convergence). First deliberately-harder design, no formal validity bar yet.

**Era 2 (040-054).** Same day as the compiled-scaffold thesis kickoff commit. First appearance of
the `tier1`...`tier5` naming. A graded difficulty curriculum, built to show accuracy recovery as
difficulty rises.

**Era 3 (055-089).** Almost entirely tier-5. Shifts from difficulty labels to named composition
shapes: argmax, subset-sum-with-distractor, count-with-condition, ratio-argmax, odd-one-out,
kth-largest, multi-constraint-filter. Many shapes ship 2-5 replicates (`_b`, `_c`, `_d`, `_e`).
Breadth-first taxonomy pass. A few outliers (057, 058, 063) are format/needle-in-haystack probes,
not composition shapes.

**Era 4 (090-121).** 090-097 continue the shape-replicate pattern. 098-121 shift to hand-authored
named-entity chains, organized as `_chain`, `_survivor`, `_eliminate_chain`. Matches the "Phase 6,
24 new tier-5 mixed tasks" commit. Sets the hand-crafted discriminator style used from here on.

**Era 5 (122-145).** Sharpest turn in the suite. Landed with "adaptive-targeted benchmark suite: 4
decision archetypes." Tasks built to fire a specific native-adaptive-engine mechanism, not to be
hard in the abstract: survivor/branch-eliminate (6), conflicting-source reconciliation (6),
stop/continue chain (6), re-expansion trigger (6). Still the active Tier-A spine.

**Era 6 (f01-f03, m01-m03).** Format-Stress Tier. Three fact/format pairs (Quesnel, Amsterdam,
Hornindalsvatnet lake depth). `m0x` is plain text, `f0x` is the same fact in a typed multi-field
JSON schema. Isolates format failure from fact failure. Specific to weak-local-model work
(badmodel-lab), absent from every earlier era.

**Era 7 (146-149, 200-207).** Two new task families, same day.
- 146-149: stacked-axis compound tasks. Stack 2-3 distinct axis types in one task (chain plus
  cross-branch argmax; AND-filter into survivor into chain terminus). Harder in kind, not depth.
- 200-207: self-contained reasoning suite. Knapsack, subset-sum, scheduling, assignment. No web
  grounding. Procedurally varied by seeded RNG. Dual-solver-verified ground truth. Designed so
  every natural greedy heuristic lands on a specific plausible wrong answer. First tasks testing
  the engine's think step in isolation from search/visit.

## Trajectory

1. Ungraded, mostly-invalid research questions (era 1).
2. Formal difficulty curriculum and shape taxonomy, built for the compiled-scaffold thesis (eras
   2-4).
3. Pivot to mechanism-targeted instruments for the native adaptive engine (era 5).
4. Fracture into isolated single-variable axes: format vs. fact, reasoning vs. grounding, compound
   vs. single-axis (eras 6-7).

## Current state

No files added to `idea_tests/` since 2026-08-06. The 8 days since have been infra, restructure,
and mechanism-fix work, not new task content. The 146-149 stacked-axis tasks are the only existing
candidate toward a "hugely expanded, harder" DAG v2 benchmark set, and at 4 tasks they are not
enough on their own. New authoring or pool-mining is required.
