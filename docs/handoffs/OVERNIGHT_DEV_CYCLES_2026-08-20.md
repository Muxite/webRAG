# Overnight dev-cycle run: DAG v2 structural fixes + evaluator root cause (2026-08-20)

Continuation of `GPU_NIGHT_CYCLES_2026-08-20.md`. Open-ended, budget-bounded run (12h wall-clock,
$15 OpenRouter ceiling) dispatching `docs/DEV_CYCLE.md`-structured cycles via subagents, coordinated
by one low-token main session. **This is an interim checkpoint, written mid-run** — all 25 cycles
below are committed; the run continues past this point.

**Spend so far: ~$3.23 of $15.** All commits on `comment-cleanup`, clean history, offline suite at
5394 passed / 18 skipped (zero failures across the whole run).

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

**Cycle 14 — expansion-side detail truncation, same bug class as Cycle 12** (`354219fd`). The
mid-JSON raw truncation bug found on the evaluation side existed on the plan-generation side too,
worse: 60.6% of candidate details were still over budget even after existing compaction, producing
invalid JSON 100% of the time it triggered. Fixed with the same budget-aware serializer (shared into
a new `detail_serialization.py` module used by both policies). **The planner routinely could not see
which URL a prior sibling had fetched, or that sibling's outgoing link menu** — confirmed via
before/after examples recovering `visit_url` on real stored runs. Shipped unconditional, same
reasoning as Cycle 12 (no defensible reason to keep sending the planner broken JSON by default).

**Cycle 15 — live chain-task comparison after cumulative unconditional fixes** (benchmark run, no
code change). Ran the exact 9-task chain set from `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md`
(source of the historical 10W/7T/11L finding) against `graph:good_adaptive`, `sequential_react`, and
`langgraph_react`, comparing fresh numbers to the pre-tonight baseline. **Result: the gap did not
close at n=1** (2W/2T/5L vs pre-fix 3W/2T/4L) — `seq_react` improved more than `graph` did, though
day-to-day live-web noise can't be ruled out at this n. More valuably, this run **surfaced a new,
distinct, precisely-traced failure mode**: task 136 terminated after exactly 1 of 2 needed hops
despite a correctly-low (0.20) step-confidence judge score, motivating Cycle 16.

**Cycle 16 — F35: subject-only contract vetoes correct low-confidence signals** (`1b739bcd`). Traced
task 136's failure to its root: a "contract satisfaction" check (F33, pre-existing) was overriding
the step-confidence judge's re-expansion trigger by treating "opened a page whose text matches my
own goal's words" as satisfaction — true of every intermediate hop of an unfinished chain, not just
genuinely complete work. **Measured offline across 400 stored runs: 68% of low-confidence-judge
vetoes (171/251) rest on this subject-only check, not a verified datum.** Fixed with a new flag
(`got_contract_veto_requires_datum_enabled`, default OFF — a live-measurable behavior change
touching 68% of veto sites, not a pure bug fix) requiring a contract to have actually verified a
measurable datum before it can override the judge. Verified by offline replay of the exact failing
node before spending any live budget.

**Cycle 17 — live A/B of the F35 fix** (benchmark run). Confirmed the mechanism works exactly as
predicted on the traced case: task 136 re-expanded on 4/5 ON-flag reps (4-5 visits vs OFF's 1),
scoring 0.290 vs 0.183 (partial win — chain_coverage improved but a downstream keystone-extraction
gap remains unresolved on nano). Aggregate across the 9 chain tasks: directionally positive but
noisy (+0.017 unweighted / +0.066 excluding one outlier task), 3 clear wins, 1 apparent loss traced
to two OTHER pre-existing bugs unrelated to this flag (judge overconfidence at the 0.6-0.7 band, and
a wrong-page disambiguation/dedup issue) — not a regression the fix caused. **Read: promising, not
yet sufficient for a default flip — needs a larger multi-model confirmation run.**

**Cycle 18 — E3 calibration + E1 dedup live A/B** (`d14be338`, `2e3452cd`, `b85dbb43`). E3: no run
had ever recorded the `similarity` instrumentation field (nothing serializes it), so reconstructed
the distribution offline for $0 by re-issuing 60 runs' actual retrieval queries against their still-
live chroma collections (3268 rows) — median similarity 0.663, with a knee at 0.40-0.45 where
density stops being a flat low shelf. **Recommended floor for a future A/B: 0.40** (cuts 6.2% of
rows), not either of the two values that had been informally guessed (0.30 near-inert, 0.50 already
cuts the productive shoulder). E1: ran the dedup kill-switch live on 12 `core24` tasks × 4 reps
(47 paired cells). **`got_dedup_enabled` (default True) was found to be significantly HURTING
score: −0.157 on `overall_score`, p=0.0007.** Traced to mechanism: dedup fires on 29.5% of
expansions post-T1-1, and 58% of firing batches flag *every* candidate, triggering a
`candidates[:1]` collapse that destroys multi-hop plans rather than trimming redundant ones — chains
hit hardest (task 135: −0.55).

**Cycle 19 — fix the dedup all-flagged collapse** (`6b092fbd`). Implemented Cycle 18's own
recommended narrow fix: when dedup would flag every candidate in a batch, keep the batch unfiltered
instead of truncating to one (partial-flag behavior unchanged). Shipped **unconditional** — the old
branch's behavior had no defensible upside once "every sibling looks like a duplicate" is understood
as a degenerate/unreliable verdict, not a real completeness assessment, and the existing
`got_dedup_enabled` kill-switch already covers anyone who wants dedup off entirely. **Live-confirmed
on the three worst-hit tasks: recovers ~61% of the ON/OFF score gap** (task 137 fully recovers and
edges past the no-dedup baseline; task 135 recovers only partially, suggesting another cost is still
active there beyond this specific bug).

**Cycle 20 — F36: dead-URL visits skip their own recovery cascade** (`cadd4843`). Root-caused task
135's residual gap after Cycle 19's fix. `agent_io.fetch_url` **raises** on a permanent HTTP failure
(e.g. a planner-invented Wikipedia title that 404s), and that exception propagated straight out of
`VisitLeafAction.execute`'s try block — skipping the URL-recovery cascade (parent/sibling/link-index
fallback) that already exists and already runs for every OTHER kind of visit failure. The one case
with zero recovery was, unsurprisingly, the single most common failure mode: **107 permanent visit
failures across 81 of 1167 recorded `graph` runs (6.9%)**. Fixed with a default-ON kill-switch
(`visit_dead_url_fallback_enabled`) that holds the exception, runs the existing cascade with the
dead URL excluded (extended to harvest the previous hop's own link menu, since the Chroma link index
was never written for URL-declaring leaves), and only re-raises if nothing recovers. **Live-confirmed
on task 135: 0.412 → 0.762**, closing nearly all of the remaining gap to the 0.800 no-dedup baseline.

**Cycle 21 — culmination check: re-run the 9-task chain set with F35 on** (benchmark run, no code
change). With F35 enabled on top of everything else now live by default, re-ran the exact
`CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` 9-task chain set against `sequential_react`. **Honest
result: the W/T/L tally did not move — bucket-for-bucket identical 2W/2T/5L with or without F35.**
Task 136 (the traced case) reproduced its improvement independently a second time (0.183→0.333).
Two apparent regressions (065, 135) carry a strong network-artifact signature (only 1 visit
recorded, action-level 20s timeouts, short duration) rather than looking F35-caused; excluding them
the 7-task subset shows a modest positive delta (+0.053), consistent with tonight's earlier finding.
**Read: F35 stays a real, mechanistically-understood, per-task-effective fix, but is not yet strong
enough evidence for a default flip** — needs k>=2 reruns on the two ambiguous cells, a second model,
and ideally a larger chain-task set before that call, since a single win/loss flip on 9 tasks moves
the tally by more than 10 points. **The chain-vs-seq_react gap itself remains genuinely open** after
21 cycles of real, verified structural fixes — this appears to be a harder problem than any single
bug, consistent with Cycle 12's own read that the remaining lever is an architecture change (score
outcomes, not plans), not another incremental fix.

**Cycle 22 — F37 (chrome-link filter, fixed) + F38 (declared-URL seed, diagnosed only)**
(`ff22f183`). Checked whether Cycle 20's F36 fix incidentally covered Cycle 13's "hijacked" class —
it didn't (F36 only fires on a raised exception; hijacked cases involve a fetch that *succeeds* on a
wrong chrome page). Traced the real mechanism and found it's **far bigger than Cycle 13's original
4-group estimate**: donation/login/portal chrome pages win URL-selection scoring ties by list order
and keyword-overlap scoring quirks, affecting **64 of 2134 executed sibling visits (3.0%) across 35
runs**. Fixed with `action.visit_chrome_link_filter` (default OFF) applying an existing chrome test
to every URL-selection pool, not just the one path F36 covers. Separately, re-measured Cycle 13's
"declared" class and found the original estimate over-counted: only 17-39% (not "39.6% of groups")
are a genuine premature-seed bug; the rest are legitimate repeat-reads of one page for different
facts. Traced the real bug to `_match_mandate_url`'s single-URL shortcut, but **deliberately did not
fix it** — the corpus shows chain hops carry no dependency edge yet, so removing the premature seed
today would just redirect the same blind visit onto the F37 chrome pool or an interlanguage link
instead of a real target. Documented an explicit fix ordering (F37 → host/language affinity filter →
a real dependency edge between authored hops → only then withdraw the mandate-URL shortcut) as F38,
diagnosed not fixed.

**Cycle 23 — live validation of F37** (benchmark run, no code change). Task 046 (the one task in a
6-task spot-check that actually exercised chrome-page selection) confirmed the fix working exactly
as designed: 3/15 visits landing on the Wikipedia donate sidebar with the flag off, **zero** with it
on — the freed-up steps went one hop deeper into legitimate content instead, at 15% lower cost, with
no score change. The other 5 tasks showed 0 chrome hits in either arm (expected, given the 3%
corpus-wide base rate) and no regressions — no case of the filter turning a present-but-low-quality
answer into an outright failure. **Side finding**: `idea_dag_settings.good_adaptive.json` is stale,
missing ~48 keys (including `final_require_grounding`) present in the current
`idea_dag_settings.json` — the benchmark agent worked around it by building settings programmatically
rather than trusting the file; this stale file is a live footgun for any future script that loads it
directly and should get its own cheap cycle.

**Cycle 24 — delete the stale per-arm settings fossils** (`b81c4937`). Investigated the stale-file
finding from Cycle 23. Turned out worse than "missing keys": `idea_dag_settings.baseline.json` and
`.good_adaptive.json` predate the `_GOT_ARM_PROFILES` machinery that superseded them (both present in
the very first GoT commit), are loaded by **zero** code in the repo (verified by grep), and
`.good_adaptive.json` actively **contradicts** the arm it's named after — it hardcodes
`native_vote_k_enabled: true`/`k=3`, which the current, canonical arm profile explicitly excludes as
measured net-negative. That false claim had already leaked into `ADAPTIVE_ENGINE.md` and
`RESEARCH_LIBRARY.md`, both now corrected. Fixed by **deleting both files** (sync-by-diff would just
drift again) and adding a permanent regression guard (`test_no_per_arm_settings_snapshots`) that
fails if any `idea_dag_settings.*.json` sidecar file reappears, plus generalizing two other tests
that had a hardcoded 3-file-name assumption baked in.

**Cycle 25 — live A/B of `visit_sibling_url_dedup`** (benchmark run, no code change). Honest
non-result: the mechanism never fired across 10 tasks chosen for parallel-sibling structure — every
sibling-URL collision observed in this sample was the "declared" class (explicit matching URLs from
the planner), which this flag deliberately doesn't touch, not the "fallback" class it targets.
No regressions in what did run (no PASS→FAIL flips), but under-sampled for a real read — the flag's
own corpus-wide base rate implies roughly 1 fallback collision per ~12 sibling-visit batches, and
this run only accumulated 25 batches total across both arms. **Recommendation: re-test with tasks
specifically selected for URL-less fan-outs, not just parallel/argmax shape** — flag stays opt-in,
neither confirmed nor invalidated.

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
10. **Same truncation bug in `expansion.py`: FIXED** (Cycle 14, commit 354219fd) — see above.
11. **`visit_sibling_url_dedup` flag** (Cycle 13): needs a live A/B before flipping on — it's a real
    routing change (second sibling gets candidate #2 instead of the shared top hit), not a pure
    efficiency win.
12. **`got_contract_veto_requires_datum_enabled` (F35, Cycle 16-17)**: promising single-task/small-n
    signal, needs a larger multi-model confirmation run before a default flip. Also surfaced two
    independent follow-on bugs during its A/B, neither caused by the flag: judge overconfidence in
    the 0.6-0.7 step-confidence band (a task scored 0.7 confidence on a step whose own stated
    reasoning acknowledged incompleteness), and a wrong-page disambiguation/dedup issue (visited
    `La_Pedrera` instead of `La_Pedrera,_Barcelona`, extracting a wrong datum from a similar-but-
    wrong page). Both worth their own cycles.
13. **Chain-task gap did not close at n=1** (Cycle 15) despite four unconditional fixes landing —
    the residual gap traces to F35 (now addressed, pending validation) and likely judge-confidence
    calibration, not the data-visibility bugs fixed in Cycles 2/12/14. Re-run the same 9-task chain
    comparison at higher n once F35's A/B is more conclusive, to see if the combination actually
    closes the historical 10W/7T/11L pattern.
14. **Capability-spectrum re-run using LangGraph arm**: Cycle 15 confirmed `langgraph_react` runs
    cleanly as a first-class variant in `idea_test_runner.py` (no wiring needed) — this was blocking
    item 7 above, now resolved; the re-run itself is still not done.

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
