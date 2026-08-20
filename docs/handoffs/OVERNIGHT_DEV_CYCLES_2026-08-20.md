# Overnight dev-cycle run: DAG v2 structural fixes + evaluator root cause (2026-08-20)

Continuation of `GPU_NIGHT_CYCLES_2026-08-20.md`. Open-ended, budget-bounded run (12h wall-clock,
$15 OpenRouter ceiling) dispatching `docs/DEV_CYCLE.md`-structured cycles via subagents, coordinated
by one low-token main session. **This is an interim checkpoint, written mid-run** — all 13 cycles
below are committed; the run continues past this point.

**Spend so far: ~$0.95 of $15.** All commits on `comment-cleanup`, clean history, offline suite at
5352 passed / 18 skipped (zero failures across the whole run).

---

## What shipped

**Cycle 0** — committed last night's ~90-file uncommitted diff into 20 scoped commits (`8c6fadc4`
through `d8e64472`). Caught a broken import in `testing/runner.py` that only worked by accident
because of uncommitted files.

**Cycle 1 — the resolved-value channel** (`d47577ec`, `4f88d872`). Completes
`RESOLVED_VALUE_CHANNEL_DESIGN_2026-08-16.md`: lets a dependent DAG node receive a prior node's
discovered value (a URL) at dispatch time via a new `requires_data.slot` field, addressing
`DAG_FORMATION_REVIEW.md` PART 0's headline finding (the graph is a tree with no fan-in edge, so
chain hop n+1 can't receive hop n's value — this is why `seq_react` beats the graph on chains).
Two adversarial review passes caught a critical gap before implementation (a third dispatch path,
the auto-parallel batch route, would have silently never fired); live smoke testing then found a
*fourth* dispatch route (best-child sequential selection) the reviews missed, fixed by consolidating
into the shared `_execute_action_guarded` wrapper. Flags: `resolved_value_channel_enabled` +
`waypoint_enabled`, both default OFF.

**Cycle 2 — visit-action timeout root cause** (`77d2791e`). Live validation of Cycle 1 hit a 20s
timeout on every visit action. Root-caused (not guessed): a link-dense page's outgoing links were
all embedded synchronously on the event loop (ChromaDB's default embedding function runs in-process
CPU-bound), stalling everything for 18+ seconds. Fixed with a capped, batched, relevance-ranked link
store (18.19s → 2.50s measured). This was silently corrupting *any* future live benchmark, not just
this one.

**Cycle 1 validation** — two live A/B passes on the design doc's 8-task mixed-shape set
(`054,085,055,061,146,147,149,122`), `gpt-4.1-nano`. First pass (no fixture parity): mean Δ −0.230,
confounded by live search non-determinism (most regressed cells never engaged the mechanism at all).
Second pass (fixture parity, record-then-replay via `scripts/cross_shape_experiment.sh` +
`scripts/prewarm_fixtures.py`): mean Δ **+0.055**, near-flat, 3 improved/4 regressed/1 tied.
**Verdict: mechanism confirmed working correctly, but no clear aggregate direction yet at n=1; stays
default OFF pending a higher-n rerun.** Engagement is stochastic — 4/8 tasks touch the mechanism
depending on how the LLM's own plan happens to shape out, since only `plan_library.py`'s
`link_page_visits` and `post_expansion_hooks.py`'s grounding-retry visits currently declare a slot.

**Cycle 4 — E1/E2 backlog items** (`e8d1f365`, `1d39c398`). E2 (offline, 594 batches / 2085
candidates across 516 stored runs): arrival-position "beam" order does correlate with score
(+0.141 raw), but ~¾ of that is action-type composition (search-first plans, search scores higher
for being search), not idea quality — residualized effect collapses to +0.034 (ns). **Surfaced the
much bigger finding below.** E1: dedup kill-switch (`got_dedup_enabled`) already existed; measured
firing rate offline (16.5% of expansions, 8.2% of candidates flagged) and wrote a live-A/B readiness
note for a future cycle.

**Cycle 5 — flat-scoring root cause** (`13f84956`). The E2 analysis found 55.6% of sibling batches
score every candidate *identically*. Root-caused: 98.2% of flat batches sit on one of three fixed
values the engine itself assigns (a 0.5 unexecuted-work cap, a `<=0.2` prompt rubric band, or a 0.4
fallback for judge-omitted candidates) — because evaluation runs *before* execution for nearly every
candidate by construction (`_expand_or_execute` drops DONE/FAILED/SKIPPED children before scoring).
**This is a structural explanation for `backtrack` firing 0/261 times ever and the evaluator's
reported AUC 0.288** — not a judge-quality problem, a scoring-order problem. Genuine judge
degeneracy on real, already-executed, differentiable candidates: 1.0% of the corpus.

**Cycle 6 — instrumentation + hypothesis test** (`9b710b91`, `8c91f519`). Added `raw_score`/`capped`
fields recording the judge's pre-clip opinion (found and fixed a real shipped-path bug in the same
commit: `LlmEvaluationPolicy`'s per-node fallback never set its own logger, so every per-node judge
call silently crashed and returned 0.0). Tested whether `evaluate_parallel_siblings` (scores after
execution) collapses the flat rate: **it doesn't** (37.5% off vs 40.0% on, p=1.0, n=31 live) — the
code-level cap lifts, but the judge's own opinion lands on the same `<=0.2` prompt rubric band
regardless, because that rubric line is written for unexecuted work and nothing stops it applying
post-execution too. **The actual lever is the batch prompt's rubric line, not the flag** — a
promptbench-family question, not yet attempted.

**Cycle 7 — Chroma/vector-DB tier** (`41096efb`, `811625e9`, `e932ba04`). Added a similarity floor
on memory retrieval (`memory_retrieval_similarity_floor`, default 0.0/off) plus `similarity`
instrumentation on every retrieved memory for future calibration. Added `final_context_rank_by_similarity`
(default off) fixing a real relevance-blind bug: `idea_finalize.py`'s pooled context kept results in
arrival-batch order, so a distance-0.9 hit could displace a distance-0.1 hit. **Killed** the
chunker-mismatch backlog item as based on a false premise (the two chunkers serve genuinely
different, non-comparable purposes — "reconciling" them would truncate ~75% of indexed content).
Confirmed `strategy_library`'s threshold is inert (empty note corpus, never reaches the comparison).

**Cycle 8 — F7/F9** (`fbc4d052`, `833f6789`). F7: tags nodes whose expansion degenerated to a single
guessed fallback candidate (`fallback_expansion` detail key), logs loudly, counts into
`degenerate_fallback_count` in the result payload — instrumentation only, no reaction yet. F9:
non-root nodes can trigger plan-library template substitution from their own possibly-low-quality
local text (never validated for that use); raised the auto-apply bar for non-root triggers to the
weakest-correct-positive threshold from the existing (root-only) calibration eval, shipped
unconditional since non-root auto-apply was never a validated baseline to begin with.

**Cycle 9 — bounded fallback re-expansion** (`a2c9676d`). The first real F6 fix: a parent whose
*entire* expansion collapsed to one F7-tagged fallback child can now get one bounded retry (flag
`got_reexpand_fallback_nodes_enabled`, default OFF), reusing the existing reexpand machinery with a
corrective prompt naming the specific failure. Structurally bounded to exactly one retry per parent
regardless of `max_iterations` (a second degeneration leaves 2 children, permanently blocking a
third attempt). Caught and fixed a real infinite-loop bug during implementation (marking the old
leaf `SKIPPED` without adjusting dispatch routing would have made the step loop re-enter it forever).

**Cycle 10 — finalize leak fix** (`2dfe3384`). Cycle 9's superseded-fallback marking wasn't actually
excluded from finalize's answer-synthesis context (6 separate selector sites, not the 2 originally
suspected) — fixed, so a future benchmark of the Cycle 9 flag won't be measuring a bad-guess-polluted
answer. Deliberately did *not* exclude the superseded node from grounding-evidence checks (it did
open a real page; excluding it there would make an otherwise-grounded run wrongly fail the gate).

**Cycle 11 — execution-aware rubric probe** (`48c37486`). Offline replay test (60 real batches, an
"anti-tie" rewrite of the evaluator's `<=0.2` rubric line) found the shipped rubric already
differentiates real executed content reasonably well when replayed cleanly (spread 0.45, spike
detection beats chance p=0.004) — the rewrite moved every metric in the right direction but nothing
cleared significance. **More important finding**: replaying recorded live batches offline gave
scores that *diverged* from what the engine actually recorded for the same nominal content,
motivating Cycle 12.

**Cycle 12 — detail-truncation bug, root cause + fix** (`64d6eb17`). The live-vs-replay divergence
turned out to be a probe artifact (different prompt reconstruction, not an engine bug) — but tracing
it surfaced a real, independent, unconditional bug: candidate `details` were JSON-serialized then
raw-character-truncated at a 5000-char budget, producing **invalid, unparseable JSON for 2888/4615
(62.6%) of executed candidates**, silently dropping fields like `visit_url` (100% loss when
triggered). Fixed with a budget-aware serializer that bisects for the largest per-field allowance
that keeps output valid — byte-identical under budget, always-parseable over it. Live A/B (40
batches, $0.09) showed directional improvement on every scoring metric, none significant — ships as
a correctness fix regardless (garbage JSON reaching a judge is bad on its own terms). **Recommended
and executed: stop the 5-cycle-deep evaluator-scoring thread here** — cap mechanics, the
parallel-siblings flag, the rubric text, and the detail budget have each been tested and moved
metrics the right direction with no result clearing significance; the remaining lever (score
outcomes via a real architecture change, not another prompt/data probe) is out of scope for a probe
cycle.

**Cycle 13 — duplicate sibling visit URLs** (`a7a17c96`). A lead surfaced in passing during Cycle 12:
16.2% (97/598) of sibling visit batches have 2+ children fetch the identical URL, and half of those
have *every* sibling on the same page (the traced example: 4 siblings meant to visit "Chuck season
1-4" all fetched the same Croatian Wikipedia page). Root-caused to two distinct mechanisms:
"fallback" (51.5% — `VisitLeafAction`'s URL-resolution cascade is sibling-blind, so near-identical
titles rank the same top hit from a shared pool) and "declared" (39.6% — sequential chain plans
where every hop's URL was written at authoring time before later hops' real target existed).
Existing dedup (title-based, pre-resolution) structurally cannot catch either, since the collision
only exists post-URL-resolution. Fixed the fallback half with an opt-in per-parent URL claim map
(`action.visit_sibling_url_dedup`, default OFF — a real routing change, not a pure efficiency fix,
so it needs its own live A/B before flipping on). The "declared" half (planner-side, needs
`requires_data` instead of a premature seed URL) and a smaller "hijacked" class (chrome/sidebar
links winning a scavenge fallback) are flagged as separate future items, not fixed this cycle.

---

## Open threads for the next cycle

1. **Resolved-value channel**: needs a higher-n (3+) fixture-parity rerun before any default-flip
   decision. Current n=1 result is directionally flat, not conclusive.
2. **Evaluator-scoring thread: CLOSED for now** (Cycles 4-6, 11-12) — cap mechanics, the
   `evaluate_parallel_siblings` flag, the rubric text, and the detail-truncation bug have all been
   tested; the detail-truncation fix shipped as a correctness fix regardless of significance. Next
   lever, if picked up again, is a real architecture change (score outcomes, not plans), not another
   prompt/data probe.
3. **F6 general case**: still open beyond the narrow fallback-parent MVP — a genuinely multi-child
   (non-degenerate) plan still can't be revised once formed.
4. **E1 dedup ablation**: groundwork done (Cycle 4), live A/B not yet run.
5. **E3 similarity floor value**: lever exists (Cycle 7), needs a histogram of the new `similarity`
   instrumentation field from a recorded run before picking a threshold — should be free.
6. **R1 (qwen2.5:7b num_ctx)**: GPU-bound, blocked tonight — Ollama CLI not installed on this host.
7. **Capability-spectrum re-run** using the LangGraph comparison arm (committed tonight in Cycle 0,
   `langgraph_solver.py`) against whichever fixes land — not yet attempted.
8. **Duplicate sibling visit URLs — declared half** (Cycle 13): sequential chain plans write every
   hop's URL at authoring time, before later hops' real target exists, causing repeat fetches of the
   seed URL. Needs a planner-side fix (declare `requires_data` instead), not the dispatch-time
   dedup Cycle 13 already shipped.
9. **Duplicate sibling visit URLs — "hijacked" class** (Cycle 13, 4 groups): a declared URL fetch
   fails and falls into a link-scavenge fallback that ranks sidebar/chrome links (e.g. a donation
   page) ahead of real content. Chrome-filtering the scavenge pool is the lever.
10. **Same truncation bug in `expansion.py`** (Cycle 12 finding, not yet fixed): the identical
    mid-JSON raw-truncation pattern exists at the plan-generation prompt site too; even after
    existing compaction, 68.7% (3886/5654) of compacted candidate details still exceed the 5000-char
    budget. Same fix pattern as Cycle 12's evaluation-side fix, but touches plan generation so wants
    its own cycle/tests.
11. **`visit_sibling_url_dedup` flag** (Cycle 13): needs a live A/B before flipping on — it's a real
    routing change (second sibling gets candidate #2 instead of the shared top hit), not a pure
    efficiency win.

## Methodology notes (holding from last night, reconfirmed)

- Subagents still don't self-resume on a background monitor firing — the parent must explicitly
  `SendMessage` back in every time. This cost real overhead again tonight; instruct every dispatched
  agent up front to block synchronously in a single tool call rather than end its turn on an
  assumption.
- Two live A/B smoke attempts tonight were burned entirely on infra (a CRLF-corrupted `SERPER_KEY`
  in `services/keys.env`, then a shell that didn't export it) before any feature signal was
  observed — worth a pre-flight check (`source` the whole keys file, verify a direct curl 200)
  before any live run, every time, not just once.
- Fixture parity (record-then-replay) materially changed a conclusion this session (Cycle 1
  validation: −0.230 confounded → +0.055 clean). Don't trust a live A/B without it when search
  determinism matters to the comparison.
