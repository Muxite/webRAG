# DAG v3 Session 1 — Breadth collapse root cause & grading asymmetry (2026-08-28)

**Branch:** `dagv2-evidence-ledger`
**Status:** analysis-only session, CPU-only, $0 (all findings drawn from existing result JSONs and offline code reading — no live cells run). A benchmark batch was running concurrently; this session touched only `docs/`, nothing under `agent/` or `scripts/` beyond adding one new read-only analysis tool.

**Verdict up front:** the previously reported "graph regression" is a task-composition artifact, not a real regression (Finding 1). The actual, MEASURED bug is that the breadth-fan-out fix from `docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md` was applied only to `langgraph_react` and never ported to the native engine that `good_adaptive` (`graph`) runs on (Finding 2) — that gap is the collapse. A separate, smaller grading-format asymmetry inflates the apparent graph-vs-langgraph gap by about 12% (Finding 3). The suite itself has no dead weight (Finding 4). Nothing here has been live-validated yet; see Next Steps.

---

## Finding 1 — The reported "graph regression" was a task-composition artifact, NOT a regression

`dagbase_20260824_*` is **8 tasks x 3 reps**, not core24. Task ids: 134, 137, 141, 144, 152, 154, 155, 156 — half of them (152/154/155/156) are the breadth/fan-out tasks from the August 23 breadth pilot, not the chain-shaped tasks core24 is built from.

The earlier "+0.142 graph over sequential_react (p=0.023)" result (`docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md` lineage) was measured on core24 (tasks 122-145), which has essentially no breadth tasks. `dagbase_20260824_*` is a different workload, not a re-measurement of the same claim — despite a run-id that reads like "core24."

Split by task type (qwen2.5:7b, infra_failed excluded, n=12 per cell):

| variant | non-breadth (134,137,141,144) | breadth (152,154,155,156) | all |
|---|---|---|---|
| graph | 0.481 | 0.189 | 0.335 |
| langgraph_react | 0.642 | 0.959 | 0.800 |
| sequential_react | 0.494 | 0.562 | 0.528 |
| sequential | 0.190 | 0.082 | 0.136 |

Per-task graph vs langgraph on breadth: 152 — 0.083/0.964; 154 — 0.292/1.000; 155 — 0.333/1.000; 156 — 0.048/0.873.

**Naming hazard:** the run-id said "core24" but the batch was 8 tasks x 3 reps. Reps sample within-task noise; tasks carry the between-task variance an A/B actually needs. Recommend run-ids encode the real task set going forward, not a label inherited from an earlier, different experiment.

---

## Finding 2 — The breadth fix was applied to the wrong engine

Commit `ba351857` (the shipped outcome of `BREADTH_STALL_ROOT_CAUSE_20260823.md`) added the candidate-coverage completion gate, context-trim, stall-recovery gate, and search-retry hygiene to **`langgraph_react` only** — `agent/app/langgraph_solver.py`, `agent/app/execution_langgraph.py`, `agent/app/idea_policies/candidate_coverage.py`. That doc explicitly investigated the native-engine equivalents, ruled them "real findings but irrelevant to that arm," and deferred them:

> "Two Explore agents initially traced hypotheses in `agent/app/idea_engine.py` ... **All three are real findings about `idea_engine.py`, but irrelevant here**" — `BREADTH_STALL_ROOT_CAUSE_20260823.md`, lines 13-23.

That deferred, never-picked-up native-engine work is what is now failing in Finding 1's `graph` column. Root causes, MEASURED from result JSONs (`dagbase_20260824_*`), cite exact locations:

1. **Branching cap not widened.** `breadth_aware_branching_enabled=False` by default (`agent/app/config.py:1058`), so `max_branching=5` (`agent/app/config.py:1051`) caps a 7-way ask at 5. Task 152 rep1/rep2 decision trace: `expansion_chosen="5 sub-problems"`. `_effective_branching` (`agent/app/idea_engine.py:2749`) implements the widening logic but is flag-gated, and `good_adaptive`'s arm profile never sets the flag.
2. **No native completion gate.** `got.candidate_coverage_enabled=False` by default (`agent/app/config.py:144`) — no per-candidate completion check before synthesis. Task 155 rep1 (N=2, well under the branching cap) created 2 search branches but only **1** visit node, then reported `success: true, warning: None`. Task 156 rep1 created 3 search branches, 2 completed a visit, and the `merge` node status was **`skipped`** — the AND-join does not fire when a sibling stalls at `search`, so synthesis proceeds on partial evidence with no error surfaced anywhere in the record.
3. **Under-decomposition even when width is available.** Task 152 rep3 chose only 2 sub-problems despite a cap of 5 — the weak model doesn't reliably use the width it's given, independent of the cap itself.

No timeout, budget-exhaustion, or novelty-guard markers appear in any of the 8 cells examined — this reads as structural (missing gate + missing widening), not a budget problem.

**Correctness note, not just a scoring gap:** `success: True`, `has_failures: False`, `warning: None` while a required candidate was never opened is a silent partial-synthesis bug in its own right, independent of the benchmark score it happens to produce.

---

## Finding 3 — Grading asymmetry between variants (fairness caveat for all past comparisons)

Measured key emission across the 4 variants (n=24 each) in `dagbase_20260824_*`:

| key | graph | langgraph_react | sequential | sequential_react |
|---|---|---|---|---|
| `output.grounded` | 24/24 | 0/24 | 24/24 | 0/24 |
| `output.sources` | 24/24 | 0/24 | 14/24 | 0/24 |
| `output.goal_achieved` | 24/24 | 0/24 | 24/24 | 0/24 |
| `obs.visit.count > 0` | 24/24 | 23/24 | 14/24 | 23/24 |

144 of 178 task modules check citations in PROSE only; only `test_130`/`test_132` use the shared `_sources_cited` helper (introduced in commit `8288f51b`). `graph` emits structured `output.sources`/`output.grounded` fields; `langgraph_react` and `sequential_react` cite inline in prose only. So on the 144 prose-only modules, `graph` is scored on structured fields it happens to populate while the other two variants are graded on prose the checker may or may not parse correctly — a FORMATTING difference, not a capability difference.

Offline re-score using a corrected, drift-controlled prose-citation check: **graph 0.335 -> 0.387 (+0.052)**; langgraph_react and sequential_react move **+0.000** (they were already being graded on the prose path they actually produce). Real and worth fixing, but it closes only ~12% of the graph-vs-langgraph gap in Finding 1 — it does not explain the collapse.

This asymmetry is a caveat on every past graph-vs-langgraph score comparison in this repo's history, including the ones cited in Finding 1. The fix must land BETWEEN benchmark runs and the baseline re-measured — never mid-batch, and not in this session given the live batch constraint.

---

## Finding 4 — Benchmark suite has no dead weight (a clean null)

New tool `scripts/task_discrimination.py` ranks tasks by discriminative power (score standard deviation, ceiling/floor hit rate, with a low-n guard so sparse cells don't masquerade as informative). Across 5264 existing result files: all 24 core24 tasks and all of suite59 classify DISCRIMINATING at meaningful sample sizes (51-256 cells per task). The only DEAD_CEILING rows in the output have n=1 and are flagged LOW_CONFIDENCE — i.e. they're single-sample artifacts, not evidence of a genuinely uninformative task.

**Implication:** the planned core24 -> suite59 move (adopted for statistical power reasons) carries no pruning tax — nothing in the wider suite needs to be dropped for being non-discriminating.

---

## Next steps

1. **A/B a new `good_adaptive_breadth` arm** that flips `breadth_aware_branching_enabled` + `got.candidate_coverage_enabled` for the native engine, run on tasks 152/154/155/156 x 3 reps. Strong prior: the langgraph twin of the coverage gate alone measured +0.227 (t=2.56, n=12) on this same task family (`BREADTH_STALL_ROOT_CAUSE_20260823.md`, "Update 2026-08-23"). **This is UNTESTED as of this writing** — no native-engine live cells were run this session.
2. Land the shared-helper citation fix (extend `_sources_cited`-style checking to the 144 prose-only modules, or normalize graph's structured emission to be graded the same way) BETWEEN runs, then re-baseline before drawing any further graph-vs-langgraph conclusions.
3. Investigate the merge AND-join `skipped` path directly: a stalled sibling should trigger a corrective extension (mirroring `_candidate_coverage_extension`'s design), not silent partial synthesis with `success: True`.
4. Address under-decomposition with a minimum-enumeration re-prompt when the model's chosen sub-problem count is well under the available cap on an explicitly N-way task.

---

# Session 1 addendum — seq→fan-out tasks, and the citation-echo failure

## Finding 5 — the native breadth mechanisms are a well-powered NULL

`good_adaptive_breadth` (flips `breadth_aware_branching_enabled` + `got_candidate_coverage_enabled`
+ observe-only `run_policy_coverage_entity_conflict_check`) vs `good_adaptive`, qwen2.5:7b,
tasks 152/154/155/156, 12 reps:

| n paired | delta | t | verdict |
|---|---|---|---|
| 16 | +0.097 | +2.10 | looked borderline-positive |
| 24 | +0.035 | +0.78 | fading |
| **48** | **+0.016** | **+0.58** | **null; 95% CI [-0.039, +0.071]** |

The mechanism demonstrably works (root fan-out 5 -> 8 after the `_effective_branching` fix); enabling
it does not buy score. Widening fan-out against a fixed `max_steps` starves each branch — coverage
falls on exactly the widest tasks. **Fan-out and budget are coupled; `candidate_coverage_budget_extension`
(default 10) is the untested knob.**

## Finding 6 — `coverage_ratio` is a false metric (definitive)

Across 48 baseline cells it reads **exactly 1.00 in 48/48**, while those same cells score **0.250**
mean. `evaluate_candidate_coverage` matches candidates against a POOLED haystack of every visited
page, so one page mentioning several candidates satisfies them all. The gated arm reads 0.815 mean
and hits 1.00 in only 29/48 — it discriminates.

Confirmed head-to-head on the new tasks: engine `coverage_ratio` = **1.00** on all four while the
task-authored, per-branch validators independently report **0.00 branches resolved**. This is no
longer an inference; it is a direct contradiction between engine self-report and verified fact.

## Finding 7 — CITATION-ECHO + HALLUCINATION-FILL (the headline failure mode)

Four new tasks (161-164) with a 3-hop sequential discovery prefix, a 6-7 way fan-out, and a 2-3 step
subtask per branch. Live, qwen2.5:7b, n=2:

| variant | mean score | mean visits |
|---|---|---|
| graph | **0.070** | 1.3-6.3 |
| sequential_react | **0.194** | 7.5-16.0 |
| langgraph_react | 0.123 | 2.0-10.5 |

**Graph does not under-perform by fanning out badly — it barely explores.** Task 163: 1.3 visits vs
sequential_react's 16.0.

Direct evidence from task 162 (graph deliverable, 5 real visits for a 7-branch task): the **same
generic URL is pasted as the citation for all 7 astronauts**, and Musgrave's DOB is fabricated
(1938-10-29 / age 55 vs the verified 1935-08-19 / 58). The model exhausts its visit budget partway
through the fan-out and completes the table from parametric memory, dressed with one reused fake
source.

## Finding 8 — model scale does NOT fix it

qwen2.5:14b (49/49 layers on GPU, 32k ctx, verified) vs the 7b reference:

| variant | 14b | 7b |
|---|---|---|
| graph | 0.054 | 0.070 |
| sequential_react | 0.172 | 0.194 |

A 2x larger model of the same family buys **nothing** — and does not explore more (graph mean 2.75
visits). The bottleneck is architectural (per-branch visit allocation / stopping condition /
willingness to fabricate rather than flag an unresolved branch), not model capability.

## Finding 9 — task calibration: hard but SOLVABLE, not broken

The validators fire correctly on partial answers — 162's `crew_coverage` credited exactly the 2
astronauts 14b got numerically right and rejected the rest. A stronger model does not trivially
solve them, so they retain discriminative headroom.

## Finding 10 — engine bug: visit leaves created with no search predecessor

Task 161, graph, 14b: three sibling `visit` leaves planned directly under root with **zero preceding
search** (`Searches: 0, Visits: 0`), each failing `error_type=InvalidURL` because the leaf's
`link_idea` was a description rather than a URL and nothing populated
`parent_search_urls_found`/`chroma_urls_found`. The 7b run on the same task searched first
(8 searches -> 4 visits), so this may be planning variance. **Unconfirmed at n>1.**

## Infra note — NUM_PARALLEL multiplies RESERVED KV

ollama sizes the KV cache for `OLLAMA_NUM_PARALLEL` sequences **at load time**, not on demand
(`4/4 seqs` in the load log). At N=4, qwen2.5:14b got only 31/49 layers on GPU at 32k ctx (47/49 at
16k). At N=1: **49/49 layers, 100% GPU, KV 3264 MiB** — matching the q8_0 arithmetic exactly.
Throughput does not need the slots: the measured 3.2x came from multi-process slicing, which works
at N=1 because extra driver processes keep ollama's queue non-empty. **N=1 is the right default.**

## Throughput (measured, matched windows)

| | mean GPU | fully idle | >=90% busy |
|---|---|---|---|
| before (1 process) | 12.2% | 80.1% | 4.1% |
| after (4 slices + flash-attn + q8_0 KV) | **64.0%** | **17.4%** | **33.8%** |

1334s of cell work completed in 414s wall = **3.2x**. Zero infra failures, no rate limiting.

---

# PHASE P — benchmark throughput (2026-08-29)

## Result

| condition | cells/hour | mean GPU | fully idle | >=90% busy |
|---|---|---|---|---|
| RTX 3060, 1 process (historical) | ~40 | 12.2% | 80.1% | 4.1% |
| RTX 5070 Ti, 1 process | 57.6 | — | — | — |
| **RTX 5070 Ti, 8 slices + flash-attn + q8_0 KV + HF-offline** | **109.5** | **70.2%** | **11.9%** | **29.6%** |

**1.90x from software config on the same GPU; ~2.7x vs the original 3060 baseline.**
core24, 24 cells in 789s. Integrity: 24/24 ledger `ok`, **0 infra_failed**, no timeouts, no DEAD.

## Changes that produced it
1. `OLLAMA_FLASH_ATTENTION=true` + `OLLAMA_KV_CACHE_TYPE=q8_0` (`badmodel-lab/docker-compose.yml`).
   Workload is ~98% prefill with a 21k-token p99 tail; attention is quadratic.
2. `--slices N` in `scripts/axis_queue_runner.py` — N concurrent driver processes over a
   round-robin task deal, each `jobs: 1` so per-process `local_busy` bounds GPU concurrency to
   exactly N. Safe because every cell already forks its own process with its own Chroma dir and
   run-id-keyed filenames.
3. `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in `cell_env()` — removes ~18 unauthenticated
   HuggingFace HEAD requests per cell from the startup critical path (verified: warning count 0).
   Also removes an unmonitored external dependency from a hermetic benchmark.

## Key findings

**Slice count is capped by TASK count, not by processes.** With a 4-task workload, `--slices 8`
yields only 4 non-empty slices — verified from the run dirs (`phaseP_s8_*` shows 4 slice dirs, one
task each). CORRECTION (2026-08-29): the "66.3 vs 82.4 cells/hour" figure was therefore **NOT** a
slices=8 vs slices=4 comparison — both conditions ran 4 effective slices, and the gap reflects
extra startup across two sequential queue entries plus run-to-run variance. The capping RULE is
confirmed; the speed comparison that appeared to demonstrate it was invalid. Slicing only pays on wide task sets:
core24 (24 tasks) at 8 slices hit 109.5 cells/hour.

**`OLLAMA_NUM_PARALLEL`: one solid claim, one CONFOUNDED claim.**

CORRECTION (2026-08-29): the earlier phrasing here — "NUM_PARALLEL is not the throughput lever" —
was **confounded and must not be cited**. NUM_PARALLEL went 4 -> 1 at the same time slices went
4 -> 8, so the utilization rise (64.0% -> 70.2%) cannot be attributed to either. A one-line A/B
(N=1 vs N=4, slices held at 8, 7b-only, identical task set) is needed to settle it.

What IS solid: N=1 is required for qwen2.5:14b to reach 49/49 layers on GPU, because ollama
reserves KV for N sequences AT LOAD TIME (`4/4 seqs` in the load log). At N=4, 14b got 31/49 layers
at 32k ctx. So N=1 must hold for any 14b campaign regardless of how the throughput A/B lands. Slots are not free: ollama reserves KV for N
sequences AT LOAD TIME, which at N=4 cost qwen2.5:14b 18 layers of GPU residency. **N=1 is correct:
it costs no throughput and unblocks the 14b capability ladder.**

**Round-robin balances cell COUNT, not cell COST.** On the 4-task workload three slices finished 4
cells while one had done 2 — task 152 is simply slower. The tail is set by the slowest single task,
so wide task sets matter more than high slice counts.

## Gate
Score distributions did not degrade. core24 graph: **0.599 (sd 0.229, n=24)** post-change vs
**0.518 (n=24)** on the single-process `s1repro_graph` reference — a +0.081 difference at ~1.7 se,
not significant, and in the *favourable* direction. **Caveat: this comparison bundles several
changes** (flash-attn, q8_0 KV, HF-offline, slicing) rather than isolating one, so it is evidence of
"no degradation", not a clean per-change measurement. Per-arm integrity counters showed no
asymmetry on the 4-task condition (truncation 4 base vs 2 breadth, n~16/arm, within noise).

---

# FINDING 11 — suite59 three-way: core24 has been flattering the graph engine

First properly-powered three-way. qwen2.5:7b, suite59, **n=59 complete triples**, 177/177 cells
`ok`, 0 infra_failed, 0 "Setup failed".

| variant | mean | sd | mean visits |
|---|---|---|---|
| graph | **0.400** | 0.271 | **7.1** |
| sequential_react | 0.588 | 0.262 | 4.7 |
| langgraph_react | **0.668** | 0.320 | 3.5 |

Paired (n=59):
| comparison | delta | t | W/T/L | verdict |
|---|---|---|---|---|
| graph vs sequential_react | **-0.188** | **-4.05** | 14/6/39 | **SIGNIFICANT** |
| graph vs langgraph_react | **-0.268** | **-5.03** | 13/6/40 | **SIGNIFICANT** |
| sequential_react vs langgraph_react | -0.080 | -1.76 | 17/11/31 | n.s. |

## The reconciliation — split suite59 by whether a task is in core24

| block | n | graph | seq_react | langgraph | graph - seq_react |
|---|---|---|---|---|---|
| **core24 subset** | 24 | 0.572 | 0.584 | 0.731 | **-0.012 (t=-0.21, a TIE)** |
| **non-core24 remainder** | 35 | **0.282** | 0.591 | 0.625 | **-0.308 (t=-5.09)** |

**core24 is precisely the region where the graph engine is competitive.** On it, graph ties
sequential_react — reproducing every prior core24 result including tonight's `s1repro_graph` 0.518
and the dagbase non-breadth split (0.481 vs 0.494). On the other 35 suite59 tasks it collapses to
0.282 against 0.591.

**Implication: every A/B this project has run on core24 was measuring the one subset that flatters
the engine under test.** That is not a bug in any single measurement — each was internally valid —
but it means the accumulated core24 evidence base systematically overstates graph's standing. The
earlier "+0.142 graph over seq_react on core24" and tonight's "-0.188 on suite59" are the same
phenomenon seen through two different task samples.

**Graph does MORE work for LESS result**: 7.1 mean visits vs 4.7 (seq_react) and 3.5 (langgraph).
It is not under-exploring on suite59 as a whole — it explores most and scores least. Combined with
Finding 7 (citation-echo: exhausting visit budget then completing the table from parametric memory
with a reused source), the picture is an engine that spends its budget without converting it into
grounded answers.

**Action: suite59 becomes the default A/B task set.** core24 is retained only as a labelled subset
for continuity with historical numbers, and any core24-only result must be reported as such.

---

# FINDING 12 — WHY graph underperforms: three engine bugs + a prompt-ordering bug

Three independent log analyses of the clean 177-cell suite59 corpus. Verdict-first.

## It is SHAPE, not difficulty — and the direction disproves difficulty

| grouping | n | mean graph - seq_react |
|---|---|---|
| core24 (all narrow "survivor" tasks) | 24 | **-0.012** |
| other35 | 35 | **-0.308** |
| **shape = aggregation** (fan-out / count / argmax / AND-filter) | **18** | **-0.425** |
| shape = chain / sequential / navigation | 10 | -0.045 |

**core24 mean `difficulty_level` is 9.58 vs other35's 8.60** — core24 is the *harder* block by the
suite's own labels, and graph ties there. Pearson(difficulty, delta) = **+0.23** corpus-wide:
harder tasks correlate with a *smaller* graph deficit. Within other35, r = -0.05 (nothing). The
difficulty-artifact story requires the opposite sign. **All 18 aggregation-shaped tasks are in
other35.** The operative attribute is *sustained-retention width* — how many facts must stay live
simultaneously — not depth, not difficulty, not weight.

**The deficit is concentrated, not diffuse:** 17 of 59 tasks carry **10.93 of the 11.09 aggregate
raw-score deficit (98.5%)**; 14 of those 17 are in other35.

## Root cause A — merge skipped for single-branch nodes, no goal check at all
`agent/app/idea_policies/merge.py:268` — `if not node or len(node.children) < 2: return False`.
The engine then logs `NO MERGE NEEDED: node {id} children done, marking DONE` and marks it DONE
**without ever evaluating the mandate**, cascading DONE to the root in one tick.
Task 059 needed 5 players, visited 1 page. Task 065's final deliverable is a bare URL.

## Root cause B — the engine detects its own failure and overrides it one step later
A failed goal check sets the merge node to `SKIPPED` (`idea_engine.py:2081-2084`), but
`idea_engine.py:899-910` computes `all_terminal` over `{DONE, FAILED, SKIPPED}` **with no
distinction for why** — so "explicitly failed the goal check" is indistinguishable from "succeeded".
Verbatim, task 094:
```
[MERGE] candidate roster incomplete -- ['Trondheimsfjord','Sognefjord',...] have no page-attributable disposition
[MERGE] Goal NOT achieved
[STEP 8] MERGE INCOMPLETE: Goal not achieved, marking as incomplete
[STEP 9] All children complete (incl. merge), marking parent done
```
Same pattern at the ROOT node on task 047.

## Root cause C — finalize prompt overflows and the MANDATE is truncated away
`idea_dag_settings.json:105` orders the finalize message `MANDATE -> MERGED RESULTS -> ...`, and
ollama truncates from the **HEAD**. Measured:

| task | finalize prompt | truncation warning | deliverable |
|---|---|---|---|
| 041 | 126,829 chars | **yes** | 188 chars |
| 042 | 110,227 | **yes** | 161 chars |
| 068 | 114,442 | **yes** | 327 chars |

Task 068's output is an unfinished hedge. **Retrieval was fine** — `coverage_ratio: 1.0`, and 041's
captured page content contains "Longest span 1,991 metres" verbatim. The engine found the answer
and lost it at synthesis.

**AMENDS §0a**: I earlier measured 153 truncations (100% in graph cells) and called the bias minor
because it "understates rather than inflates graph performance". That was complacent — it is
*causal* for the worst losses, because of WHAT gets truncated (the instructions), not how often.

## Root cause D — duplicate branch spam inflates the prompt into the ceiling
Task 041: **115 nodes for a 6-URL task**, each target visited 9-11 times (~5.5x duplication), from
17 `[EXPANSION]` events re-proposing the same targets in different phrasings. The semantic dedup
gate (0.878) catches some and misses enough to spawn fresh nodes.
Also: `[RUN] GUARDRAIL: 19 nodes still pending execution... Cannot finalize with pending nodes`
fires and then **finalize runs anyway**, root node still `active`.

## The engine's self-reported success fields are unreliable
Task 049: `goal_achieved: True`, `finalization_status: "complete"`, `coverage_ratio: 1.0` — with a
deliverable of `"comparison of Eiffel Tower and Statue of Liberty completion dates"` (a title stub).
Real score **0.00**. Any KPI built on the engine's own success flags measures optimism.

## CORRECTION — citation-echo is a LANGGRAPH failure mode here, not a graph one

On this clean 177-cell corpus (`audit_citation_echo`):

| variant | echo active | over-asserted |
|---|---|---|
| graph | 1 (1.7%) | 0 |
| sequential_react | 5 (8.5%) | 0 |
| **langgraph_react** | **25 (42.4%)** | 2 |

langgraph's echo rate is 12.5% in core24 vs **62.9%** in other35. The earlier corpus-wide 49.4%
figure does **not** reproduce for the graph variant on clean data. Graph's failure mode is
under/over-exploration plus partial finalization — not citation echo.

## Cost: graph burns ~2x in BOTH blocks; only the payoff differs

| block | variant | visits | llm calls | tokens | duration |
|---|---|---|---|---|---|
| core24 | graph | 5.83 | 42.0 | 79,514 | 177s |
| core24 | seq_react | 2.50 | 16.3 | 20,978 | 61s |
| other35 | graph | 7.91 | 77.2 | 139,106 | 284s |
| other35 | seq_react | 6.17 | 34.0 | 67,991 | 114s |

The worst-scoring graph cells are also the most expensive: 189,665 tokens / 354.7s / 99.4 llm calls
vs 89,366 / 201.1s / 50.4 for the rest.

## Telemetry gaps to close
1. `grounded` / `finalization_status` / `coverage_ratio` / `sources` are **graph-only fields** —
   seq_react and langgraph payloads carry only `final_deliverable`/`success`/`goal_achieved:null`/
   `action_summary`. **Cross-variant grounding KPIs (K1, K3) are impossible from stored JSON.**
2. `SKIPPED` is overloaded — add `skip_reason` so root cause B is detectable in data, not just logs.
3. `merge.py:268`'s `<2 children` short-circuit is not logged as a distinct decision.
4. The finalize prompt is never persisted; only char counts are logged.
5. No `pending_at_finalize` / `root_status_at_finalize` in `execution.output`.
6. No `duplicate_visit_ratio` (visits / distinct urls) in `observability.visit`.
7. Expansion logs only "Parsed N candidates", never the candidate list.

---

# FINDING 13 — Adversarial re-examination: what survived, what broke

An independent hostile review of Finding 11/12. Verdicts below supersede the earlier text where
they conflict.

## SURVIVES — graph loses on suite59
All headline figures reproduce exactly. **Budget fairness attack fails and backfires**: graph gets
`max_steps=50` (`execution.py:574`) vs 25 for both others — graph loses while spending ~2x.
`effort_tier=0` everywhere, tooling identical, config hashes balanced 57/2 across all three arms.

**RETRACT the citation-format mitigation.** Granting every failing citation check for free (a hard
upper bound) gives graph +0.0797 but seq_react **+0.1003** — the sources-aware re-score moves the
delta to about **-0.209, WIDER not narrower**. Finding 3's "+0.052 graph / +0.000 others" does NOT
reproduce on this corpus. That mitigation should not be cited.

## SURVIVES, STRONGER — shape, not difficulty
Blind re-classification of all 59 modules (source only; no results, rule fixed before classifying):

| blind shape | n | graph | seq_react | delta | t |
|---|---|---|---|---|---|
| aggregation | 23 | 0.227 | 0.688 | **-0.461** | **-7.73** |
| chain | 33 | 0.524 | 0.532 | -0.008 | -0.16 |

**The de-confounder:** other35 also contains 9 CHAIN tasks — they show delta **+0.002 (t=+0.02)**.
Same block, same era mix, opposite shape, zero deficit. **Shape beats block membership.**
Validator-strictness confound is dead: seq_react scores *higher* on aggregation (0.688) than chain
(0.532), so strictness cannot produce a variant-specific collapse.

**RETRACT my Pearson(difficulty, delta)=+0.23 argument** — it is a Simpson artifact of the block
split restated (+0.456 within core24, -0.052 within other35). Use instead: empirical difficulty is
near-equal across blocks (non-graph mean 0.657 core24 vs 0.608 other35) while graph falls
0.572 -> 0.282.

**NEW and better — dose-response:** Pearson(`n_items`, delta) = **-0.491** corpus-wide, -0.391
within other35. The 8-task survivor family (5-6 items but a single-page keystone) sits at -0.156,
exactly where a retention-width model predicts.

## SURVIVES — core24 was built around the graph engine
`docs/BENCHMARK_SUITE_HISTORY.md`: Era 5 = **122-145 = exactly core24**, authored 2026-07-11,
described as *"Tasks built to fire a specific native-adaptive-engine mechanism... survivor/
branch-eliminate (6), conflicting-source reconciliation (6), stop/continue chain (6),
re-expansion trigger (6)."* core24 is the biased set — stronger than "unrepresentative".
**Not circular:** the 25 suite59 tasks from Eras 2/3/4, predating the mechanism-targeted era and
never selected for graph weakness, show delta **-0.264 (t=-3.47)**.

## BROKEN as causal — the four "root causes"
The code claims are all verbatim correct and the bugs are genuine. **What is unsupported is that
they explain the -0.188.** Marker presence vs delta across all 59 graph cells:

| marker | fires | delta when fires | delta when absent | t |
|---|---|---|---|---|
| A `NO MERGE NEEDED` | 51/59 | -0.193 | -0.154 | **-0.22** |
| B goal-not-achieved override | 27/59 | -0.248 | -0.137 | -1.24 |
| C head truncation | 10/59 | -0.335 | -0.158 | -1.79 |
| D guardrail-pending | 13/59 | -0.295 | -0.157 | -1.36 |

**None is a significant predictor.** Root cause A fires on 86% of cells *including graph's wins*
and is LESS enriched in the losers. This is the "vivid logs" failure mode. Rewrite as: **four
confirmed bugs, none yet shown to be load-bearing.**

## ARITHMETIC ERROR — "98.5% of the deficit"
That divided by the NET (-11.07), which 14 graph wins have already shrunk. Gross loss is **-14.42
across 39 losing tasks**; the top 17 carry **75.8% of gross**, not 98.5%. The method self-refutes
one step later (top 20 = 105%, top 25 = 114%). Report **76% of gross**, or "39 of 59 tasks lose".

## RETRACT the citation-echo retraction
`audit_citation_echo`'s `active` gate requires an enumerated >=2-item run with a parseable entity
name. Inactive reason is **100% "no enumerated per-entity claims"** (58/59 graph). Graph's median
deliverable is **155 chars** vs seq_react 261, langgraph 675 — the 1.7/8.5/42.4 ordering is exactly
the deliverable-LENGTH ordering. Correct statement: **the rate is UNMEASURED for graph (n=1); the
audit cannot score terse structured output.** Neither 49.4% nor 1.7% is publishable as a rate.

## n=1 — what survives
Pooled within-cell sd **0.233** (1160 dof); per-delta noise ~0.335 vs observed between-task delta
sd 0.356 -> implied true between-task sd ~0.120, **single-rep per-task reliability ~0.11**.
- **Survives:** the aggregate (t=-4.05), the block split (t=+3.59), the shape split (t=-7.73), era splits.
- **Does NOT survive:** the "17 catastrophic tasks" and every named task id. Simulated replication
  overlap is **5.8/17 vs a chance level of 4.9/17**. Named tasks are illustrations, not evidence.

## Confirmed and understated
`coverage_ratio == 1.0` on **all 17** worst tasks, with `goal_achieved: True` and
`finalization_status: "complete"` on 11 of them.

---

# FINDING 14 — Two fixes land, neither pays. The gap is coverage.

Both measured on the 21-task aggregation block (the region where graph-vs-seq_react is -0.461).

## N1a — sibling evidence digest: NULL on score, works on duplication
`run_policy_sibling_evidence_digest_enabled` (the arm Phase 0 killed **on core24, where the gap is
-0.012**). Re-tested where the gap exists, n=32 paired:

| metric | base | shared |
|---|---|---|
| score | 0.246 | 0.236 — **null**, t=-0.32, CI [-0.067,+0.049], W/T/L 12/8/12 |
| distinct URLs | — | +0.38 (n=21, sd 0.96 — suggestive only) |
| duplicate factor | — | -0.26 |
| visits | 10.1 | 9.3 |

Duplication fell sharply on the worst offenders — task 042 **12.75 -> 3.75**, task 041
**8.67 -> 5.50**. **But the freed budget is not reinvested**: visits fell, distinct pages stayed
flat, score unchanged.

→ **Sharpened hypothesis:** the constraint is NOT that duplication crowds out coverage. It is that
**nothing drives the engine toward uncovered candidates at all.** That is what N4 (entity-tied
width) and N5 (deterministically-minted jobs) must attack; N1 alone is necessary and insufficient.

## F1+F3+N3 shipped: the inversion is fixed, the score is not
Post-fix vs pre-fix, same 21 tasks (35 pre-cells, 27 post-cells, task-mean paired):

| metric | pre | post |
|---|---|---|
| **deliverable median** | **87 chars** | **250 chars** |
| **stubs (<120 chars)** | **13/21** | **5/21** |
| score | 0.259 | 0.280 — **+0.021, t=+0.63, CI [-0.046,+0.087]** |

**The intervention demonstrably works; it does not pay.**

### The offline estimate did NOT reproduce — retract it
Offline re-scoring predicted **+0.047** (arm mean 0.400 -> 0.447). Live it is +0.021, n.s.
**Likely cause of the discrepancy, and a general caution:** the offline re-score *concatenated*
`deliverable + action_summary`, handing validators BOTH fields; F3 *swaps*, so validators gain the
summary and lose whatever the deliverable held. **Offline re-scoring measures an upper bound, not
the intervention** — do not use it as a live-effect predictor again.

## Effect decay, third instance tonight
| experiment | small-n | larger-n |
|---|---|---|
| breadth arm | +0.097 (t=2.10, n=16) | **+0.016 (t=0.58, n=48)** |
| F1+F3 | +0.055 (t=1.62, n=21 pre) | **+0.021 (t=0.63, n=35 pre)** |

At sd 0.15-0.28, nothing under t~2 at n~20 should be believed. This is why the throughput work came
first.

## What this leaves
Graph now emits substantive deliverables and duplicates far less, and still does not score better
on aggregation. **The remaining gap is coverage** — distinct pages retrieved — exactly where both
experiments independently point. Phase N4/N5 is the live path.

---

# CYCLE: "trustworthy behavior from bad models" (2026-08-30)

Thesis refined by the user: the niche is **results you can verify / evidence-based thinking** —
good behaviour from models that are individually untrustworthy. Two hard constraints: **being
wrong-but-evidenced is not a win** (accuracy is a non-regression guard), and **auditability is not a
moat by default** — the baselines must compete on the same output surface.

## Finding 15 — auditability is NOT architectural (a moat we do not have)

A variant-agnostic audit layer (`agent/app/testing/audit_layer.py`) applied post-hoc to all arms:

| variant | n | fetch-recoverable | cited URLs | cited-never-fetched | quote-auditable |
|---|---|---|---|---|---|
| evidence_loop | 21 | 21/21 | 92 | **7** | 21/21 |
| graph | 59 | 59/59 | 4 | 0 | **0/59** |
| langgraph_react | 59 | **0/59** | 170 | n/a | 0/59 |
| sequential_react | 80 | **0/80** | 186 | n/a | 0/80 |

**Verdict: mostly an implementation gap, closeable in an afternoon.** The ReAct variants already
cite URLs in prose (170/186) — the model produces the raw material. The gap exists because
**telemetry redacts URLs to char counts by design** (`{"chars": N}`, never the string) and trace
files are deleted unless `traces_retained()`. Per-claim quote verification tracks **whether an arm
does typed extraction**, which is orthogonal to graph-vs-sequential — the graph engine does not do it
either (0/59), and `evidence_loop` (a FLAT ReAct loop) does.

→ **Do not claim auditability as an architectural advantage.** The honest product is
**"ReAct plus an evidence ledger"**. `evidence_loop`'s 7 cited-never-fetched URLs are a working
fabricated-citation detector.

## Finding 16 — the ledger verdict is the project's FIRST calibrated self-report

`evidence_loop` vs `sequential_react`, 21 aggregation tasks, qwen2.5:7b:

| | score | verdict calibration |
|---|---|---|
| sequential_react | 0.558 | `success=True` in **21/21** — UNINFORMATIVE, single label |
| evidence_loop | 0.579 (t=+0.27, **tie**) | **ANSWER 0.700 > PARTIAL 0.640 > ABSTAIN 0.466** |

Monotone, correct ordering. Contrast the graph engine, which is ANTI-calibrated
(`goal_achieved=True` 0.389 vs False 0.416; `finalization_status=complete` 0.372 vs partial 0.437).
**Accuracy is a tie, so the trust signal is not bought with accuracy.**

## Finding 17 — but evidence_loop OVER-ABSTAINS, and we know why

**Bad-abstain rate 87.5%** — 7 of 8 ABSTAIN verdicts were on runs scoring >0.3. Causal chain:
wrapper-punctuation bug -> **82.3% of claims fail quote verification** -> rows never reach SUPPORTED
-> verdict falls to ABSTAIN.

`strip_quote_wrapper` (shipped after this run) recovers up to 99 of 177 failures. **Re-run required
to confirm** — the `lvl_el` data predates the fix.

Honest split of 322 extractions (234 checked): 57 verified (24.4%), **99 wrapper-affected** (ceiling
66.7%), **78 genuine** — of which **51 are plain prose paraphrases** (the model composing supporting
text for values it got right), 88 empty.

## Shipped this cycle
- **Tool-call emulation shim** — `langgraph_react` now runs tinyllama/phi3:mini/gemma2:2b (previously
  instant 0.000, ollama 400 "does not support tools"). `tool_transport` recorded per cell; native path
  behaviourally unchanged. **Verified live: phi3/tinyllama = `emulated`, tool-capable models = `native`.**
- **`evidence_loop` variant** — ReAct's loop verbatim + a non-expiring ledger (~100 chars/row vs the
  1500-char observation it replaces; ReAct's scratchpad is a FIXED 12-step window, so ~6 items live),
  typed extraction, quote-offset grounding, table-first finalization.
- **`coverage_ratio` was hardcoded `1.0`** (`idea_finalize.py:1517`) — now computed or `None`.
  This explains the anti-calibration: `complete` was gated on a constant.
- **Claim provenance** with tri-state `quote_verified`; **page persistence** so a stored cell is
  re-auditable offline (+40 KB/cell, exact re-verification by default).
- **`core_long24`** — 24 tasks, 12 shapes, **124.5s/cell vs suite59's 107.5s** (longer at 41% the
  size). At 5 reps = 120 paired obs, powered for d=0.10 across the full measured sd range;
  **suite59 at 1 rep (n=59) never cleared even the best-case bound of 61.**
- **N-sweep family 165-168** (N=4/8/16/32, nested-prefix rosters so per-item difficulty CANNOT drift
  with N) to locate ReAct's 12-step forgetting boundary.
- **Trust KPI dashboard** — reports harm avoidance beside accuracy, and marks metrics
  **uncomputable** rather than 0 for arms that lack the fields.

Suite: **7531 passed, 18 skipped, 0 failed**. `execution_sequential.py` byte-identical throughout.
