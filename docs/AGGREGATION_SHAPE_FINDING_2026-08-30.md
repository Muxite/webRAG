# Aggregation shape finding: structured DAG execution loses to sequential ReAct where fan-out doesn't convert to coverage (2026-08-30)

**Status:** results document, compiled from the analysis session recorded in
`docs/handoffs/DAG_V3_S1_BREADTH_COLLAPSE_AND_GRADING_ASYMMETRY_2026-08-28.md` (Findings 11-13
specifically). CPU-only, offline, $0 to produce this document — all numbers below are re-stated
from that handoff's live benchmark runs (qwen2.5:7b, local Ollama), not re-run here. Read the
source handoff for full per-finding derivations; this document is the publishable synthesis.

## Verdict

**Structured DAG execution (`graph` / `good_adaptive`) loses to a plain sequential ReAct loop
(`sequential_react`) on aggregation-shaped tasks, and the mechanism is measured, not inferred:
fan-out multiplies work but not coverage.** On the properly-powered three-way (suite59, n=59
complete triples, 177/177 clean cells, qwen2.5:7b):

| variant | mean score | mean visits |
|---|---|---|
| graph | **0.400** | **7.1** |
| sequential_react | 0.588 | 4.7 |
| langgraph_react | 0.668 | 3.5 |

Graph loses to sequential_react by **-0.188 (t=-4.05, n=59, W/T/L 14/6/39)** while exploring
*more*, not less — 7.1 mean visits vs 4.7. The loss is concentrated on aggregation-shaped tasks
(fan-out / count / argmax / AND-filter): delta **-0.461 (t=-7.73, n=23)** under a blind,
source-only reclassification. Chain-shaped tasks show essentially no deficit: delta **-0.008
(t=-0.16, n=33)**.

This is a diagnostic result with a measured code-level mechanism, not a proof that DAG
execution is unworkable. It corroborates, on new evidence, a phasing decision the project's own
`DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` already anticipated (§6): *"sequential_react (or the
deterministic queue) becomes the default cheap-model path, with DAG v2/v3 tree execution
retained only for task shapes with a demonstrated breadth win — not a universal replacement."*
A redesigned-primitives experiment is queued as future work (see Implications).

---

## 1. The headline comparison

qwen2.5:7b, suite59, n=59 complete triples, 177/177 cells `ok`, 0 infra_failed, 0 "Setup failed".

| comparison | delta | t | W/T/L | verdict |
|---|---|---|---|---|
| graph vs sequential_react | **-0.188** | **-4.05** | 14/6/39 | **SIGNIFICANT** |
| graph vs langgraph_react | -0.268 | -5.03 | 13/6/40 | SIGNIFICANT |
| sequential_react vs langgraph_react | -0.080 | -1.76 | 17/11/31 | n.s. |

sd: graph 0.271, sequential_react 0.262, langgraph_react 0.320 (n=59 each).

**Cost is inverted relative to score.** Graph spends roughly 2x either rival's tokens and calls
in both the block where it ties and the block where it collapses (§4) — it does not win by
spending less, and it does not lose by spending less either.

## 2. Why core24 alone would have hidden this

Every prior graph-vs-sequential_react A/B in this project's history (including the previously
reported "+0.142 graph over seq_react, p=0.023") was run on `core24` (tasks 122-145). Splitting
suite59 by core24 membership:

| block | n | graph | seq_react | langgraph | graph − seq_react |
|---|---|---|---|---|---|
| core24 subset | 24 | 0.572 | 0.584 | 0.731 | **-0.012 (t=-0.21, TIE)** |
| non-core24 remainder | 35 | **0.282** | 0.591 | 0.625 | **-0.308 (t=-5.09)** |

On core24, graph ties sequential_react — reproducing every prior core24-only result. On the
other 35 suite59 tasks it collapses to 0.282 against 0.591. The accumulated core24 evidence
base was not wrong on its own terms; it was measuring exactly the subset where graph is
competitive, because it was authored to be.

**MEASURED, not speculative:** `docs/BENCHMARK_SUITE_HISTORY.md` records Era 5 = tasks 122-145
= exactly core24, authored 2026-07-11 and described in-source as *"Tasks built to fire a
specific native-adaptive-engine mechanism."* That is the biased set, stronger than merely
"unrepresentative." This is not circular reasoning about the finding, though: 25 suite59 tasks
from Eras 2-4, predating the mechanism-targeted era and never selected for graph weakness,
independently show delta **-0.264 (t=-3.47)** — the collapse is not an artifact of comparing a
biased set against itself.

## 3. Shape, not difficulty

A blind re-classification of all 59 task modules by shape (source code only, no result data,
classification rule fixed before looking at any score) gives a cleaner split than the
core24/non-core24 block boundary:

| blind shape | n | graph | seq_react | delta | t |
|---|---|---|---|---|---|
| aggregation (fan-out / count / argmax / AND-filter) | 23 | 0.227 | 0.688 | **-0.461** | **-7.73** |
| chain / sequential / navigation | 33 | 0.524 | 0.532 | -0.008 | -0.16 |

**The de-confounder that separates shape from era/block:** the non-core24 block also contains 9
tasks that are CHAIN-shaped, not aggregation-shaped. Same block, same authoring era as the
aggregation losses sitting next to them, opposite shape — and they show delta **+0.002 (t=+0.02)**,
zero deficit. Shape predicts the loss; block membership does not, independent of shape.

**Difficulty runs the wrong way to explain the collapse as "graph just loses on hard tasks."**
core24's own difficulty_level label averages 9.58, the non-core24 remainder 8.60 — core24 is the
*harder* block by the suite's own metadata, and it's exactly where graph ties. Empirical
difficulty (mean score of the two non-graph variants, i.e. how hard the task actually proved for
capable executors) is nearly flat across the two blocks — 0.657 vs 0.608 — while graph itself
falls from 0.572 to 0.282 between them. A naive Pearson(difficulty_level, delta) = +0.23 across
the whole corpus looked supportive of a difficulty story on first pass but is a Simpson's-paradox
artifact of the block split restated (+0.456 within core24, -0.052 within non-core24 — no signal
inside either block). It does not survive and should not be cited.

**Dose-response, corpus-wide:** Pearson(n_items, delta) = **-0.491** (n_items = how many distinct
entities/facts the task's answer requires). Within the non-core24 block alone: -0.391. The more
distinct items a task needs held live simultaneously, the larger graph's deficit — a smooth
relationship, not a threshold effect, consistent with a retention/coverage mechanism rather than
a generic difficulty or scale effect.

## 4. The mechanism: fan-out multiplies work, not coverage

| block | variant | visits | llm calls | tokens | duration |
|---|---|---|---|---|---|
| core24 | graph | 5.83 | 42.0 | 79,514 | 177s |
| core24 | seq_react | 2.50 | 16.3 | 20,978 | 61s |
| non-core24 | graph | 7.91 | 77.2 | 139,106 | 284s |
| non-core24 | seq_react | 6.17 | 34.0 | 67,991 | 114s |

Graph runs at roughly 2x the visits, calls, and tokens of sequential_react in *both* blocks —
where it ties and where it loses. The spend ratio doesn't change; the payoff does. The
worst-scoring graph cells are also the most expensive (189,665 tokens / 354.7s / 99.4 calls vs
89,366 / 201.1s / 50.4 for the rest) — graph is not failing by under-exploring, it explores the
most and scores the least.

Validator-measured coverage on aggregation tasks makes the mechanism concrete: graph's
per-branch coverage check reads **0.095** vs seq_react's **0.732** on the same aggregation-shaped
tasks. Distinct-evidence throughput is roughly constant regardless of shape — chain tasks need
2-3 distinct pages and graph gets 2.6; aggregation tasks need 5-7 and graph still gets about 4.0.
Widening the fan-out increases branch count without increasing the number of distinct pages
actually retained per branch — more parallel work, not more coverage.

**The code-level cause, in the authors' own words**
(`agent/app/idea_policies/expansion.py:287-300`):

> "expansion context is root-ward only (`IdeaDag.path_to_root`), so a sibling's queries, URLs
> and outcomes are invisible."

Each branch of a fan-out plans and executes with no visibility into what its siblings have
already found or are doing — the structural precondition for genuine parallel-aggregation work
(shared awareness of what's covered) is absent.

**What sequential_react does instead** (`execution_sequential.py:286`): one flat loop, one JSON
decision per step, a single linear scratchpad that every later step reads, whole page text
carried in the observation, and `seen_queries` dedup against the same shared log. One shared
context, monotonically accumulating, with global visibility of every fetch so far — the opposite
structural property from the graph's rootward-only isolation.

## 5. What was controlled for, and what was refuted

**A budget-fairness attack was run and failed — and reverses the naive interpretation.** Graph
runs with `max_steps=50` (`execution.py:574`) against 25 for both sequential_react
(`execution_sequential.py:451`) and langgraph_react (`execution_langgraph.py:72`). Graph gets
*double* the step budget of either rival and still loses — the loss is not explained by graph
being budget-starved relative to its competitors. `effort_tier=0` is uniform across all three
arms, tooling is identical, and config hashes balanced 57/2 across arms with no asymmetry traced
to any single config difference.

**A citation-format mitigation was tested and retracted.** Graph emits structured
`sources[]`/`grounded` fields on 59/59 cells; langgraph and seq_react cite inline in prose, and
144 of 178 task modules check citations in prose only — raising a fairness concern that graph
might be graded more leniently. Granting every failing prose-citation check for free (a hard
upper bound, maximally generous to the losing arms) moves graph +0.0797 but seq_react **+0.1003**
— a sources-aware re-score would *widen* the gap to about **-0.209**, not narrow it. The earlier
finding that this asymmetry closed ~12% of the gap does not reproduce on this corpus and should
not be cited.

**Difficulty was tested and refuted as an explanation** — see §3; the sign runs the wrong way and
the apparent correlation is a Simpson's-paradox artifact of the block split.

## 6. MEASURED vs INFERRED vs UNMEASURED

| claim | status |
|---|---|
| graph loses to seq_react by -0.188 (t=-4.05) on suite59 | MEASURED |
| the loss is concentrated in aggregation shape, near-zero in chain shape | MEASURED (blind classification) |
| graph spends ~2x tokens/visits/calls in both the tying and losing blocks | MEASURED |
| coverage-check gap (0.095 vs 0.732) on aggregation tasks | MEASURED |
| dose-response Pearson(n_items, delta) = -0.491 | MEASURED |
| difficulty explains the collapse | REFUTED (wrong sign once de-confounded) |
| citation-format asymmetry explains part of the gap | REFUTED (widens gap when corrected for) |
| root-ward-only expansion context is the structural cause of the coverage gap | INFERRED (code-supported, not yet A/B'd by restoring sibling visibility) |
| the four engine bugs below (merge-skip, SKIPPED-as-success, head truncation, guardrail-pending) cause the loss | NOT SUPPORTED as causal (see §7) — real defects, no measured effect on delta |
| citation-echo / hallucination-fill rate for graph specifically | UNMEASURED (audit cannot score graph's terse structured deliverables — see §7) |
| a redesigned-primitives (shared-evidence / typed-queue) architecture would close the gap | UNMEASURED, queued as future work |

## 7. Four real engine bugs — found, reported, and explicitly NOT load-bearing

An adversarial re-examination tested whether four independently-discovered code defects causally
explain the -0.188 result. They do not, on the evidence gathered. Marker presence vs. per-task
delta across all 59 graph cells:

| marker | fires | delta when fires | delta when absent | t |
|---|---|---|---|---|
| A — merge skipped for <2 children (`merge.py:268`), no goal check at all | 51/59 | -0.193 | -0.154 | **-0.22** |
| B — goal-not-achieved override: `SKIPPED` treated same as `DONE`/success (`idea_engine.py:899-910`) | 27/59 | -0.248 | -0.137 | -1.24 |
| C — finalize prompt truncated from the head, mandate lost (`idea_dag_settings.json:105`) | 10/59 | -0.335 | -0.158 | -1.79 |
| D — `GUARDRAIL: pending nodes` fires, then finalize runs anyway | 13/59 | -0.295 | -0.157 | -1.36 |

None is a significant predictor. Marker A fires on 51/59 cells (86%) **including graph's wins**
and is actually less enriched in the losing cells than in the corpus as a whole — this is a
"vivid logs" failure mode: the bug is real and worth fixing, but its presence in the log does not
track the score outcome. Correct framing: **four confirmed, reproducible code defects, none yet
shown to be load-bearing for the -0.188 result.** They are reported here because they are real
and independently worth fixing (in particular, bug B produces `success: True` on runs where the
engine's own merge step explicitly detected goal failure one step earlier — a correctness bug in
its own right, separate from any benchmark score).

**Citation-echo could not be scored for graph and should not be cited either way.** An audit
tool's `active` gate requires an enumerated, ≥2-item deliverable with parseable per-entity claims
to score echo/hallucination-fill behavior. Graph's median deliverable is **155 characters** vs
seq_react's 261 and langgraph's 675 — almost never eligible for the audit. The audit reports
graph active in 1/59 cells; that 1.7% figure is not a rate, it is a sample-size artifact of
graph's terse output format, and a previously reported corpus-wide 49.4% citation-echo figure
also does not reproduce for graph on this clean corpus. **Report as unmeasured**, not as either
high or low.

## 8. Limitations — n=1 per task, stated prominently

**This is the single most important caveat in this document and must not be buried.** Every
number above is one rep per (task, variant) cell, not a repeated-measures design.

- Pooled within-cell standard deviation is **0.233** (1160 degrees of freedom). Implied true
  between-task standard deviation of the delta itself is ~0.120, against a per-delta noise term
  of ~0.335 — **single-rep per-task reliability is approximately 0.11.**
- **What survives at this n:** the aggregate comparison (t=-4.05), the core24/non-core24 block
  split (t=+3.59 for the reversal in direction between blocks), the blind shape split (t=-7.73),
  and the era splits (pre- vs post-Era-5 authoring). These all average across 9-35 tasks, which
  is where the statistical power lives.
- **What does NOT survive:** any claim about specific task ids as evidence, including "the 17
  catastrophic tasks" framing used in an earlier pass over this data. A simulated replication of
  which tasks would rank as worst-performing again puts the expected top-17 overlap at **5.8/17**,
  against a chance-level baseline of **4.9/17** — barely above chance. **Named tasks in this
  document, and in the source handoff, are illustrations of a mechanism, never evidence for it.**
  Task 162's fabricated Musgrave DOB and reused-citation example in the source handoff (Finding
  7) is exactly this kind of illustration — real, but not statistically representative on its own.
- **An earlier "98.5% of the deficit sits in 17 tasks" claim was an arithmetic error** and has
  been corrected in the source material: that percentage was computed against the net delta
  (-11.07), which 14 graph *wins* had already shrunk. The correct gross figure is that the top 17
  losing tasks carry **75.8% of gross loss** across 39 losing tasks (gross loss -14.42) — a real
  concentration, just not the number originally reported. Cite as "39 of 59 tasks lose, with
  losses concentrated in the top 17" — not as a specific percentage of net.
- **One confound that cannot be separated in this corpus:** task shape is near-collinear with
  authoring era. core24 (Era 5) was authored specifically to fire the native-adaptive-engine
  mechanism and is disproportionately chain/survivor-shaped; the non-core24 remainder is
  disproportionately aggregation-shaped for reasons of general suite composition, not
  deliberate balancing against this question. The chain-tasks-within-non-core24 de-confounder in
  §3 is the best available control for this, not a full resolution of it — a suite explicitly
  authored to cross shape and era orthogonally would be stronger evidence than this corpus can
  provide.

## 9. Implications

The measured result supports, without proving definitively, the phasing the project's own master
plan already anticipated before this data existed
(`docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md`, §6):

> "sequential_react (or the deterministic queue) becomes the default cheap-model path, with
> DAG v2/v3 tree execution retained only for task shapes with a demonstrated breadth win — not a
> universal replacement."

This document does not claim DAG-style structured execution is unworkable in general — it claims
the *current* implementation's fan-out mechanism (root-ward-only expansion context, no sibling
evidence visibility, four uncorrected but non-load-bearing engine bugs) fails specifically on
aggregation shape, with a measured code-level cause (§4) and a refuted alternative explanation
(difficulty, §3). Whether a redesigned primitive — shared-evidence context across siblings, a
typed deterministic queue instead of an LLM-authored tree, or the `evidence_queue_deterministic`
ablation arm named in the master plan's Phase 0 — closes this gap is **UNMEASURED** and is queued
as the next experiment, not asserted here as a prediction.

## Related material

- `docs/handoffs/DAG_V3_S1_BREADTH_COLLAPSE_AND_GRADING_ASYMMETRY_2026-08-28.md` — full findings
  1-13, live run details, per-cell derivations for every number in this document
- `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` — the phasing decision this result corroborates,
  and the Phase 0 ablation arms (`graph_shared_context` in particular) that would test the
  root-ward-only-context hypothesis directly
- `docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md` — the earlier, core24-only
  "+0.142 graph over seq_react" result this document's §2 explains as a task-selection artifact,
  not a false measurement
- `docs/BENCHMARK_SUITE_HISTORY.md` — Era 5 / core24 authoring history cited in §2

---

## Addendum (2026-08-30) — scope of the coverage metric

Building `scripts/coverage_report.py` surfaced a measurement limitation that sharpens two claims in
this document.

**`execution.graph.nodes` is empty for `sequential_react` and `langgraph_react`** — neither engine
populates the GoT DAG structure. Distinct-URL counts and the duplication factor are therefore
**computable only for the `graph` variant**. (The tool reports `n/a` rather than 0 for the other
two; an early version emitted a fabricated zero, now pinned against by
`test_extract_visits_none_when_nodes_empty`.)

Consequently:

- **"Distinct-evidence throughput is roughly constant" is a WITHIN-GRAPH claim** — graph's own
  duplication factor is stable across task shapes while its distinct-page count fails to scale with
  what the task requires. Verified: core24-only duplicate factor **2.35**, distinct URLs **2.42**;
  across the full 59-task set 2.37 / 3.05, sitting between the chain and aggregation values as a
  mixed task set should. It is **not** a graph-vs-sequential_react comparison, and must not be
  presented as one.
- **The cross-variant coverage evidence is the task-authored coverage-check score** — graph
  **0.095** vs sequential_react **0.732** on aggregation tasks. That is independent of the DAG
  structure, is computed by the tasks' own validators, and stands unaffected.

**Telemetry gap this exposes:** there is no variant-agnostic visit record, so the metric most
central to the coverage hypothesis can only observe one of the three arms. Closing that is a
prerequisite for any cross-engine coverage claim.
