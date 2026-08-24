# Graph vs seq_react gap — does a parallelism niche exist, or is it unmeasured? (2026-08-22)

**Read-only investigation, no code changed, no live spend.** Follow-up to
`ENGINE_DESIGN_REVIEW.md` (2026-08-19), triggered by the user's question: the native graph
engine loses to this repo's own `sequential_react` baseline (0.479 vs 0.516 overall, 0.356 vs
0.360 even on chains — the graph's designed best case) at 3x the prompt tokens. The user's
hope was that parallel fan-out would let a weak model try more approaches and make up ground
via extra (cheap) tokens. This doc checks whether that hope has ever actually been tested, and
diagnoses where the token spend goes instead.

Evidence labels match `ENGINE_DESIGN_REVIEW.md`: VERIFIED (read the code this pass), MEASURED
(number from a stored artifact), REPORTED (from an existing doc, not re-checked), HYPOTHESIS
(consistent with the evidence, untested).

---

## 1. Status update on ENGINE_DESIGN_REVIEW.md's own open items

Two of the review's three "open questions" have since been closed by this session's other work
(REPORTED, cross-checked against current code this pass):

- **R1 (qwen2.5:7b context truncation)** — CONFIRMED and FIXED (`618965c1`/`3bf7f604`,
  `OllamaNativeBackend`). Pooled score moved 0.208→0.420 post-fix. This closes the review's
  top diagnostic lead; it is no longer live.
- **D1 (finalize reads truncated `content` instead of `content_full`)** — FIXED. VERIFIED this
  pass: `idea_finalize.py:131` now does `ar.get("content_full") or ar.get("content") or ""`,
  matching the review's R3 recommendation exactly.
- **D2/D3/D4 (tautological/irreversible merge goal-achievement)** — FIXED this session
  (`edc3f328` etc., see `project_merge_goal_achievement_chain` memory).

**Still open, unchanged since 2026-08-19** (VERIFIED this pass):
- **R2, the evaluation-ordering invariant** ("a decision that consumes a score must run after
  that score exists") — deliberately NOT fixed. This session's own `E2` ablation
  (`ASSUMPTION_AUDIT.md`) measured the reordering's ceiling gain at ~4% (bounded by a 44.4%
  discriminating-batch rate × 45% first-is-not-top rate) and declined to promote it as not
  worth the control-flow change. This was a considered call, not an oversight — but it means
  the root cause PART 3 names is still live, just downstream-patched (T1-3/T1-5/T1-6 all
  addressed symptoms of this same invariant violation, not the invariant itself).
- **R5, the per-hop extraction step** — NOT built. VERIFIED (grep this pass: zero hits for any
  extraction-step mechanism in `agent/app/*.py` or `idea_policies/*.py`). This is still named
  as "the best-evidenced explanation for the chain deficit" and nobody has acted on it.
- **The review's own top open question — "does the graph scaffold pay for itself anywhere?"**
  — still open. This is the user's exact question, and Part 2 below is the new finding.

---

## 2. The parallelism hypothesis has essentially never been testable — the suite has ~1 breadth task

This is the single most important finding of this pass, and it directly answers "is there an
unmeasured niche."

**MEASURED** (from `agent/app/BENCHMARK_SUITE_50.md`'s own shape-balance accounting, current
active-59 suite): **survivor 9 · chain 9 · conflicting 8 · computation 8 · count 6 ·
re-expansion 6 · argmax 4 · numeric-AND-filter 2 · CVE 2 · navigation 2 · breadth 1 ·
nearest/selection 1 · temporal 1.** Task **052** is explicitly documented as "the suite's only
breadth task" after task 024 (the only other candidate) was dropped 2026-07-25 for being
un-gated and LLM-judge-scored (a 0-visit hallucination passed its bar).

Six other tasks (122/125/126/141/142/144 — VERIFIED, `BENCHMARK_SUITE_50.md` F29) carry an
"un-gated breadth diagnostic" as a **secondary** partial-credit signal, but their primary
keystone shape is survivor/chain, not breadth — a breadth-flavored bonus bolted onto a
non-breadth task is not evidence about breadth-shaped execution.

**Consequence:** `core24` (tasks 122–145, the set used for essentially every local-model
comparison this session ran, including today's 360-cell run and `ENGINE_DESIGN_REVIEW.md`'s
own cited numbers) contains **zero** primary breadth-shaped tasks — 052 falls outside that
range entirely. Every score comparison run against `core24` this session (and, per the
review's own citations, the `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` numbers it MEASURED
from) is therefore near-blind to whatever advantage parallel fan-out might offer, by
construction of the task mix, not by the engine's design failing on breadth tasks. With n=1
possible breadth task in the whole suite, no paired comparison can produce a statistically
usable answer either way. **This is not "measured and lost," it is "structurally unmeasurable
with the current suite."** The `SHAPE_ADAPTATION_OPEN_QUESTIONS.md`-era finding that "branch/
parallel shape is unmeasured" is still exactly true, 6+ months in, and this pass confirms
nothing has changed it.

---

## 3. A second, more concrete diagnosis of why the graph loses even on chains — HYPOTHESIS, code-supported

`ENGINE_DESIGN_REVIEW.md`'s Q32/Q12 finding (no extraction step, so a chain hop's value must be
re-located inside concatenated raw text at finalize time) is one mechanism. This pass found a
second, upstream one: the sequencing gate that's supposed to PREVENT chain hops from executing
in parallel in the first place is narrower than it needs to be.

`idea_engine.py:1951-1990`'s `auto_parallel_siblings` (default **True**) parallelizes an
eligible sibling batch UNLESS `_detect_state_dependencies`/`_detect_chunk_dependencies` finds a
reason not to (VERIFIED — it does correctly check dependencies before parallelizing; the
earlier hypothesis that it parallelizes unconditionally is WRONG and is corrected here).
`idea_sequencing.detect_state_dependencies` (VERIFIED, `idea_sequencing.py:85-125`), the
governing check, only returns `True` in two narrow cases:

1. A VISIT candidate in the batch lacks a resolvable URL AND a SEARCH candidate is also present
   in the same batch (implying the visit needs the search's output).
2. A candidate's `requires_data.source_node_id` explicitly names a sibling already in the
   candidate batch — and this field, VERIFIED (`expansion.py:1499-1533`), is populated
   **mechanically**, only for a VISIT candidate whose URL was extracted by following a link
   discovered on an already-visited page (i.e. real link provenance).

**What this misses:** a chain where hop *N+1* is a SEARCH (not a VISIT) whose correct QUERY
CONTENT depends on hop *N*'s answer (e.g. "find X's birth year" → "who was president when X
was born") has no mechanism to be flagged as dependent at all. If the expansion LLM emits both
search candidates in one batch (plausible-sounding queries for both hops, written before either
executes), neither of `detect_state_dependencies`'s two rules fires — no visit-without-URL, no
`source_node_id` — and `auto_parallel_siblings` executes them **concurrently**, meaning hop
*N+1*'s query was constructed without ever seeing hop *N*'s real answer. This is a plausible,
structural, and currently untested explanation for the review's own citation ("chain deficit
survives the scheduling fix... relocated to extraction/finalize") going one layer deeper: the
scheduling fix addressed cases the dependency detector CAN see; this class of chain dependency,
it cannot see at all, regardless of scheduling logic downstream.

**This is HYPOTHESIS, not measured** — it is consistent with the evidence and code-verified
as a real gap in the detector, but nobody has confirmed a specific chain task actually hits this
path live. The cheap way to check: instrument `detect_state_dependencies`'s return value plus a
same-batch check of "do these candidates' titles/queries share an entity that only one of them's
prior answer would supply," and look at recorded `core24` chain-task runs for batches that were
parallelized despite an actual data dependency existing between their titles.

---

## 4. Where does the 3x token spend actually go, and is any of it "trying more approaches"?

MEASURED (`ENGINE_DESIGN_REVIEW.md:38-43`, re-cited, not re-derived this pass): `graph:
good_adaptive` spends 58,335 prompt tokens vs `seq_react`'s 19,255, for a *worse* score. Per
this session's own findings elsewhere (VERIFIED against code, not re-measured this pass):

- **Merge overhead was real and partially fixed.** `project_merge_compaction_bug` (this
  session, `5601309e`/`59e29494`) found merge synthesis was 36.3% of prompt tokens via dead
  compaction code that re-serialized the same content repeatedly rather than compacting it —
  fixed and live-confirmed. Some of the review's 58k-token measurement predates this fix
  (`ENGINE_DESIGN_REVIEW.md` is dated 2026-08-19; the compaction fix landed 2026-08-21), so the
  CURRENT token gap is very likely smaller than 58k vs 19k, but has not been re-measured on the
  same task set since the fix — a concrete, cheap follow-up.
- **Every node re-serializes the FULL path-to-root context**, VERIFIED
  (`idea_policies/evaluation.py::_build_messages`, `path = graph.path_to_root(node.node_id)`,
  serialized per node for scoring, and the equivalent happens again for each expansion/action
  call) — a graph structurally pays an O(depth × context-size) cost per node that a linear
  ReAct loop pays once via its running message history. This is architecture, not a bug: it is
  the price of the graph's ability to revisit/re-evaluate ANY node independent of execution
  order, and it is not obviously fixable without giving up that property.
- **None of the measured token spend correlates with "trying more approaches" in the sense the
  user hoped.** The 6 documented "inert mechanism" findings in `ENGINE_DESIGN_REVIEW.md` PART 3
  (arrival-order beam, dead backtrack, dead confidence-gating) all trace to the SAME root cause:
  `auto_parallel_siblings` (default on) collapses graphs to depth ≤3 in practice (T1-6,
  measured this session on 11,121 real nodes: depth 1=6484, 2=2138, 3=2499, max 3) by executing
  whole sibling batches in one step and skipping the per-candidate evaluation that would let the
  engine choose or discard among diverse candidates. **The tokens are being spent on
  serialization/context overhead, not on genuinely evaluated alternative reasoning paths** — the
  mechanism that would make "more tokens = more approaches tried" literally true (real
  per-candidate scoring feeding real selection) is the one T1-3/T1-5 already showed is mostly
  starved of usable signal.

---

## 5. Does a defensible niche exist, and what would exploiting it look like?

**Given #2 above, the honest answer is: unknown, because it has never been properly tested —
not "no."** The user's hope (parallel fan-out compensating for a weak model via more tried
approaches) is architecturally *plausible* on a genuinely breadth-shaped task (independent
sub-goals, no cross-dependency, real per-candidate evaluation feeding a real merge). It has
simply never had adequate task coverage to be measured. That is a real, actionable gap
independent of whatever the eventual answer turns out to be.

**What a real test would require, concretely (falsifiable, cheap, $0 local):**

1. **Author more breadth-shaped tasks.** One (052) is not enough for any paired comparison with
   usable power. A handful (5-8) of genuinely independent-subgoal tasks — e.g. "find N
   unrelated facts, one per named entity, merge into one answer, no entity's answer depends on
   another's" — would let `good_adaptive` vs `seq_react` actually be compared on the shape the
   graph is nominally built to win. This is squarely in scope for the existing `task-author`
   agent type used elsewhere this session.
2. **Fix or at least instrument the evaluation-ordering invariant (R2) on that new task
   subset specifically**, rather than suite-wide — E2's earlier "not worth it" verdict was
   computed against a suite that's 98%+ non-breadth; the ceiling-gain math (44.4% discriminating
   batches × 45% first-is-not-top) may look different restricted to a breadth-heavy population
   where candidates are genuinely more likely to differ in quality (independent subgoals, not
   near-duplicate search queries for the same fact).
3. **Re-measure the current token gap** on `core24` post-merge-compaction-fix before assuming
   the 58k-vs-19k number still holds — a full re-run is $0/local and was already going to happen
   via today's in-flight `ladder_langgraph_20260822`/`ladder_seqreact_20260822` runs; extend
   that analysis to report prompt-token deltas, not just score, once those land.
4. **If, after (1)-(3), breadth-shaped tasks still show no graph advantage even with real
   evaluation and a fair task population**, retire the parallelism-compensates-for-a-weak-model
   hypothesis explicitly and look instead for a narrower, already-partially-evidenced niche:
   structural failure modes `seq_react` cannot recover from that a graph can by construction —
   e.g. contradictory-source detection (`race_value_agreement`, shipped this session, one of
   today's four benchmarked mechanisms) requires seeing two independent answers to the SAME
   sub-question to catch a conflict; a linear ReAct loop that commits to one answer per fact
   structurally cannot run that check at all. That's a correctness-under-noisy-sources niche,
   not a raw-accuracy-via-more-tokens niche — a smaller, more honest claim than the original
   hope, but a real and currently-unexploited one (today's `good_adaptive_tracemech` result,
   +0.023 n.s. on `core24`, doesn't rule this out — `core24` mostly isn't the shape where
   conflicting sources would even arise).

**Is a structural/architectural remap warranted right now?** **No — not yet, and not on this
evidence.** The evidence for "graph loses" is real and repeated across three separate
measurement rounds. But the specific claim that "parallelism can't help a weak model" has not
actually been tested — the suite that would test it barely exists. Committing to a rewrite
before running the cheap, targeted experiment above would be measuring a decision against
inference, which is exactly the mistake this session's standing discipline (measure before
rebuilding) exists to prevent. **Recommended next step: author 5-8 genuinely breadth-shaped
tasks and run the SAME kind of paired local A/B this session has been running all day, before
deciding whether a remap is justified.** That is a half-day-to-day-scale piece of work, not a
multi-week architectural bet, and it directly answers the open question rather than guessing at
it.

---

## Open questions this pass could not resolve

- Whether the Section 3 dependency-detection gap actually fires on a specific real `core24`
  chain task — HYPOTHESIS, code-verified, not yet observed live.
- The CURRENT (post-compaction-fix) token gap on `core24` — the 58k/19k numbers predate the
  merge-compaction fix and should be re-measured, not assumed stale OR assumed unchanged.
- Whether `race_value_agreement`-style correctness niches generalize beyond the specific
  contradictory-source shape, or whether that's the ONLY defensible niche once breadth is
  properly tested and (possibly) also fails to show an advantage.

---

## Cycle 0 addendum: partial post-compaction-fix token re-measurement (script-only, $0)

The `ladder_langgraph_20260822`/`ladder_seqreact_20260822` runs referenced above as "in-flight"
were hard-killed (SIGTERM) before finishing: 40/72 `langgraph_react` cells and 21/72
`sequential_react` cells landed (`core24`, `good_adaptive`, qwen2.5:7b, 3 reps). A new script,
`scripts/analyze_ladder_langgraph_vs_seqreact_20260822.py` (adapted from
`analyze_ladder_final_20260822.py`), pairs strictly on `(task_id, rep)` present in BOTH engines
— all 19 unmatched `langgraph_react`-only cells were dropped, not compared.

**Paired n = 20** (of 21 matched keys; 1 dropped for `infra_failed`). This is far below the 144
cells (72/engine) originally planned — treat everything below as directional, not confirmatory.

- **Score delta (graph − seq_react): −0.008** (langgraph_react mean 0.625 vs sequential_react
  0.633), sd=0.406, t=−0.08. Statistically indistinguishable from zero; W/T/L = 13/1/6 in raw
  counts is misleadingly graph-favorable given the near-zero mean — the wins and losses are
  large and cancel.
- **Prompt-token delta (graph − seq_react): +3,106** (langgraph_react mean 35,464 vs
  sequential_react 32,357 prompt tokens), sd=34,359, t=0.40 — also statistically indistinguishable
  from zero.
- **Total-token delta: +2,613** (36,140 vs 33,528), same story.

**The headline finding: the previously-reported ~58k/19k (roughly 3x) token gap between graph
and seq_react is NOT visible in this partial re-measurement.** Both engines now cluster around
32-36k prompt tokens on this partial `core24` sample — the gap, if it still exists, is at most
~10% here, not 3x. This is consistent with (though does not prove) the merge-compaction fix
landed earlier this session having closed most of the previously-measured gap, since that fix
directly targeted the runaway prompt-token growth in merge steps. It could equally be sampling
noise from the specific 20 surviving task/rep cells (sd is ~10x the mean on both score and
token deltas) — n=20 cannot distinguish "gap closed" from "gap unchanged, got unlucky sampling."

**Recommendation: queue a full clean 144-cell re-run before drawing any conclusion.** This
partial result is intriguing enough (a 3x gap apparently vanishing) to be worth the $0 local
compute to confirm properly, but it is not evidence on its own — the SD relative to the mean is
too large, and the sample is neither the full task set nor evenly split across reps. Do not
update the "58k/19k" figure anywhere else in this document or in MEMORY.md based on this
partial run; only a completed, non-killed 144-cell run should be treated as authoritative.

---

## 2026-08-23 addendum: FULL 144-cell post-compaction-fix re-measurement (script-only, $0)

> **RETRACTED 2026-08-23 (later same day).** This addendum's `ladder_langgraph_full_20260823`/
> `ladder_seqreact_full_20260823` runs were later discovered to have executed against a DEAD
> Serper search key (403 Unauthorized on every call) -- verified via
> `grep -c "Setup failed\|Serper health probe failed" .../cell_logs/*.log` showing hits in all
> 72/72 cell logs for both run-ids, zero successful searches anywhere in the run. Every number
> below is therefore an artifact of a fully-ungrounded search backend, not a measurement of the
> graph vs sequential_react comparison. Do not cite the +0.167/p≈0.0003 score result or the
> -12,291/p≈0.0013 token result anywhere as current state. See the "2026-08-23 REDUCED-SCOPE
> re-run (grounding restored)" section at the end of this document for the corrected,
> properly-grounded replacement measurement.


The full clean re-run recommended in the Cycle 0 addendum above has now completed both halves:
`ladder_langgraph_full_20260823` (72/72 `langgraph_react` cells) and
`ladder_seqreact_full_20260823` (72/72 `sequential_react` cells) — `core24`, `good_adaptive`,
qwen2.5:7b, 3 reps/task, the full 144-cell matrix originally planned, no kills this time. A new
script, `scripts/analyze_ladder_langgraph_vs_seqreact_full_20260823.py` (adapted from the
Cycle 0 script, same paired-on-`(task_id, rep)` methodology), was run against both.

**Paired n = 70** (of 72 matched keys; 2 dropped for `infra_failed`, 0 missing scores, 0 unmatched
task/rep — both engines finished every planned cell). This is an adequately-powered result, not
the underpowered n=20 partial sample from Cycle 0.

- **Score delta (graph − seq_react): +0.167** (langgraph_react mean 0.645 vs sequential_react
  mean 0.479), sd=0.365, t=3.83, df=69, **p≈0.0003 — significant.** W/T/L = 44/15/11, graph-favorable
  and consistent with the mean this time (not cancelling large opposite swings as in Cycle 0).
- **Prompt-token delta (graph − seq_react): −12,291** (langgraph_react mean 20,577 vs
  sequential_react mean 32,868 prompt tokens), sd=30,706, t=−3.35, df=69, **p≈0.0013 —
  significant.**
- **Total-token delta: −13,162** (21,007 vs 34,169), t=−3.52, df=69, **p≈0.0008 — significant.**

**Updated verdict: the previously-reported ~58k/19k (≈3x) token gap does NOT hold up at full
power, and the direction has reversed from the original claim.** The graph engine
(`langgraph_react`) now uses *fewer* prompt tokens than `sequential_react` (20.6k vs 32.9k, a
~1.6x gap in the OPPOSITE direction from the original 58k/19k figure, which had graph as the
expensive engine), and scores significantly higher on `core24` under `good_adaptive`
(qwen2.5:7b) at the same time — i.e. this result is not "cheaper but worse," it's cheaper AND
better. This corroborates the Cycle 0 partial finding that the merge-compaction fix closed the
original token gap, and goes further: with full statistical power, what remains is a
significant score win and a significant token win for the graph engine, not a wash. The
original 58k/19k figure predates the merge-compaction fix and should now be treated as
superseded, not merely "unconfirmed" — do not cite it as current state anywhere in this
document, MEMORY.md, or elsewhere without this addendum attached.

**Caveats before generalizing:** this is one task set (`core24`), one model (qwen2.5:7b), one
strategy arm (`good_adaptive`) — it does not by itself settle the Section-3/breadth-shaped-task
open question above (that still requires the dedicated breadth-task authoring work), and 2
cells were dropped for infra failure rather than retried. But as a same-conditions,
same-script, fully-powered re-run of exactly the comparison Cycle 0 flagged as noise-limited,
this is now confirmatory: post-compaction-fix, the graph engine is not merely at token parity
with `sequential_react` on `core24` — it is both cheaper and higher-scoring.

---

## 2026-08-23 Cycle 2 addendum: dependency-detection-gap diagnostic on `diag_parallel_dep_20260823` (log-analysis-only, $0)

> **RETRACTED 2026-08-23 (later same day).** The `diag_parallel_dep_20260823` run analyzed
> below also executed against the same dead Serper key as the FULL 144-cell re-run above (same
> root cause, same day). The 18 cells' search calls all failed setup; any batch that appeared
> to reach `detect_state_dependencies` did so with zero real page content available, which may
> or may not have affected which batches were even reachable. Treat this diagnostic's "RULED
> OUT" verdict as unconfirmed pending a re-run under a live search backend -- it has NOT been
> re-run as part of the 2026-08-23 reduced-scope grounding-restored re-run (that re-run covers
> only the langgraph_react vs sequential_react score/token comparison, not this diagnostic).


The `diag_parallel_dep_20260823` run (18 cells: 9 `core24` chain-shaped tasks x ~2 reps, native
graph engine, `auto_parallel_siblings` instrumented) was analyzed via
`cell_logs/*.log` under `agent/idea_test_results/_diag_parallel_dep_20260823/`. Grepping for
`[STEP N] PARALLEL BATCH DIAGNOSTIC: <n> candidates, detect_state_dependencies=...` summary
lines (excluding the per-candidate breakdown lines that follow each summary) found **5 fired
diagnostic events** across the 18 cells — only 5 of 18 cells actually produced a sibling batch
that reached the parallelization check; the rest either had no parallel-eligible siblings or
never reached this step. All 21 raw grep matches (5 summaries + 16 per-candidate breakdown
lines) are accounted for.

**All 5 summary events had `detect_state_dependencies=False`** (0 had `True`) — i.e. the
existing detector never flagged a dependency on this sample, consistent with these being chain
tasks where the detector is expected to be exercised. Of those 5, **only 1 had a non-empty
`shared_novel_tokens`**: task 138 (`rep1_138`), `shared_novel_tokens=['contains']`, from two
candidates — `"Visit the page that contains information about Mount Everest name proposal"`
and `"Visit the page that contains information about the Great Trigonometrical Survey"`. This
is a **false positive**: `'contains'` is generic templated phrasing shared by both leaf
prompts' boilerplate ("Visit the page that contains information about..."), not a proper
noun/entity resolved from a prior hop. Judged against the task: the two candidates are
independently-answerable sub-facts (Everest's proposer, the Survey), not a real chain
dependency — correctly left unparallelized-safe by the existing detector.

The other 4 non-empty-token-adjacent findings across all 5 cells were `novel_tokens=['identified']`
on individual candidates (e.g. "Visit the identified bridge's page", "Visit the identified
steamship page") — but critically these never appeared as *shared* across 2+ candidates in the
same batch (`shared_novel_tokens=none` for those batches), because in each case only one
candidate in the pair referenced "the identified X" while its sibling used the concrete name
directly. No batch showed a shared entity-bearing token across candidates that the detector
missed.

**Verdict: RULED OUT.** Zero genuine dependency-detection-gap hits found in this sample — the
one qualifying hit (task 138) is a templated-phrasing false positive, not a real missed
dependency. This does not prove the detector is complete (n=5 fired events is small and 13/18
cells never exercised the parallel path at all), but it gives no evidence for a companion
scheduling/detection fix. **Recommendation: a future chain-deficit fix cycle should focus on
extraction-step work alone; do not budget scope for a parallel-dependency-detection fix
alongside it** unless a larger/differently-sampled diagnostic run surfaces a genuine hit.

---

## 2026-08-23 REDUCED-SCOPE re-run (grounding restored) — SUPERSEDES the retracted 2026-08-23 addenda above

**Why this exists**: the full 144-cell re-run and the Cycle 2 dependency-detection diagnostic
above were both discovered to have run against a DEAD Serper search key (403 Unauthorized) —
`grep -c "Setup failed\|Serper health probe failed"` hit in all 72/72 cell logs of
`ladder_langgraph_full_20260823`/`ladder_seqreact_full_20260823`, zero successful searches
anywhere. Both addenda are now marked RETRACTED in place above. The root cause was an env-var
mixup: `SEARCH_API_KEY` (an old, separately-exhausted Brave key still present in
`services/keys.env`) was live in the shell instead of `SERPER_KEY`, the key
`services/shared/connector_config.py` actually prefers when set. The user confirmed Serper
credits are restored and `SERPER_KEY` returns HTTP 200 with real results.

**Scope reduction**: rather than repeat the full 144-cell matrix, this re-run uses an 8-task
subset of `core24` (fewer tasks, same 3 reps/task) to get back to a grounded, statistically
usable result faster.

### 1. Grounding fix verified BEFORE spending any run time

A single smoke-test cell (`smoke_serper_check_20260823`, task 134, `good_adaptive`,
qwen2.5:7b, `langgraph_react`, rep1) was run first and its cell log inspected directly:

- `grep -c "Setup failed\|Serper health probe failed"` → **0** across all 3 smoke cell logs.
- The log shows `ConnectorSearchSerper: Probing Serper search API...` immediately followed by
  `ConnectorSearchSerper: Serper search API OPERATIONAL`.
- The resulting deliverable cited a real, resolved URL with real content: *"The total length of
  the Garabit viaduct, which was engineered by Gustave Eiffel and spans the Truyère river gorge
  in the Massif Central of France, is 565 metres (1,854 ft). Sources: - Wikipedia: [Garabit
  viaduct](https://en.wikipedia.org/wiki/Garabit_viaduct)"* — a real URL, not a leaf-id echo,
  score 0.65, `infra_failed: false`, visit count 1.

Only after this passed did the full 8-task run launch.

### 2. Task selection — 8 of `core24`, 2 per shape family

`core24` (`agent/app/BENCHMARK_SUITE_50.md` line 35-38) splits cleanly into 4 shape families of
6 tasks each: **A survivor/branch-eliminate** (122-127), **B conflicting-source** (128-133),
**C stop/continue chain** (134-139), **D re-expansion trigger** (140-145). Picked 2 per family,
spanning distinct domains within each family rather than adjacent task IDs:

- **122** (radio-telescopes/FAST) and **126** (handhelds/Microvision) — survivor family, two
  unrelated domains (astronomy vs consumer electronics).
- **128** (Pluto diameter) and **132** (MLB batting leader) — conflicting-source family, a
  physical-measurement conflict vs a sports-record conflict (different conflict mechanics).
- **134** (Eiffel→Garabit) and **137** (Telford→Pontcysyllte) — stop/continue chain family, two
  different engineer→structure chains.
- **141** (Curium density) and **144** (RRS Sir David Attenborough length) — re-expansion
  trigger family; both of these also carry the suite's "un-gated breadth diagnostic" secondary
  signal (`BENCHMARK_SUITE_50.md` line 119), so this pair gets a small amount of breadth-signal
  coverage as a bonus, on top of being the primary re-expansion shape.

This spans all 4 core24 shape families (not just the first 8 task IDs, which would have been
122-129 — all survivor + conflicting-source, missing chain and re-expansion entirely) while
keeping the run small enough to finish in under an hour of GPU-serialized local compute.

### 3. Run configuration

```
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/adaptive_ladder_run.py \
  --run-id ladder_reduced_20260823_graph --axis e1_dedup_local \
  --tasks 122,126,128,132,134,137,141,144 \
  --arms good_adaptive --variant langgraph_react --jobs 4

PYTHONPATH=.:services:agent ./.venv/bin/python scripts/adaptive_ladder_run.py \
  --run-id ladder_reduced_20260823_seqreact --axis e1_dedup_local \
  --tasks 122,126,128,132,134,137,141,144 \
  --arms good_adaptive --variant sequential_react --jobs 4
```

`e1_dedup_local` is reused purely for its ladder table (qwen2.5:7b via `badmodel-ollama`,
reps=3 baked into the axis) — `--arms good_adaptive` selects the single arm needed, matching
the retracted run's config. `langgraph_react` is confirmed (VERIFIED, `agent/app/
langgraph_solver.py:301`, `agent/app/idea_test_runner.py:1076-1078`) as this repo's own native
graph engine registered under that execution-variant label, the same variant the retracted
`ladder_langgraph_full_20260823` used — not off-the-shelf LangGraph's `create_react_agent`. Both
drivers ran concurrently (isolated run-ids, embedded per-cell Chroma, GPU-serialized by
`badmodel-ollama`'s single-loaded-model slot) — 24 cells/engine, 48 total, all completed with no
kills.

### 4. Grounding verification for the WHOLE run (not just the smoke cell)

```
grep -c "Setup failed\|Serper health probe failed" \
  agent/idea_test_results/_ladder_reduced_20260823_graph/cell_logs/*.log      → 0 across all 24 logs
grep -c "Setup failed\|Serper health probe failed" \
  agent/idea_test_results/_ladder_reduced_20260823_seqreact/cell_logs/*.log   → 0 across all 24 logs
```

All 48 cell logs (24 + 24) show `Serper search API OPERATIONAL`. `infra_failed` count: 0/24 on
both engines. This is a clean, fully-grounded run — no repeat of the earlier dead-key failure
mode anywhere in the 48-cell matrix.

### 5. Paired results

`scripts/analyze_ladder_reduced_20260823.py` (adapted from `analyze_ladder_langgraph_vs_
seqreact_full_20260823.py`, same paired-on-`(task_id, rep)` methodology, restricted to the
8-task subset):

- **Paired n = 24** (of 24 matched keys; 0 missing score, 0 infra-failed — every planned cell on
  both engines parsed cleanly and paired).
- **SCORE mean delta (graph − seq_react): +0.142** (langgraph_react mean 0.673 vs
  sequential_react mean 0.531), sd=0.286, t=2.43, df=23, **p≈0.023 — significant.** W/T/L =
  18/0/6, graph-favorable and directionally consistent with the retracted full-144-cell run's
  +0.167 (though this is a different, smaller task set and cannot be treated as confirming that
  exact magnitude).
- **PROMPT TOKENS mean delta (graph − seq_react): −4,448.8** (langgraph_react mean 12,043 vs
  sequential_react mean 16,492 prompt tokens), sd=19,918.0, t=−1.09, df=23, **p≈0.287 — not
  significant.**
- **TOTAL TOKENS mean delta: −4,673.3** (12,359 vs 17,033), t=−1.13, **p≈0.28 — not
  significant.**

### 6. Reading the result

The score result reproduces the qualitative direction of the retracted full-144-cell finding —
graph beats sequential_react, not the other way around — under a properly grounded search
backend, on a different (smaller, shape-balanced) 8-task subset with independent randomness.
This is corroborating, not confirmatory: n=24 here vs n=70 in the retracted run, a different
task mix, and the retracted run's own numbers cannot be cited as a magnitude reference since
they were measured with zero working searches. The token result, unlike the retracted run's
significant −12,291 finding, is NOT significant here (p≈0.29) — both engines cluster closer
together on prompt tokens on this smaller task set (12.0k vs 16.5k, a real but noisy ~27%
difference, not the ~1.6x/p<0.01 result claimed in the now-retracted addendum). Given the
retracted run's token numbers were themselves collected under total search failure (every
search call short-circuited to a "Setup failed" error rather than doing real work), that
prior number cannot be trusted as a token-cost baseline either — this reduced-scope run is the
first grounded token measurement available for this comparison.

**Recommendation**: this reduced-scope result is adequate to restore confidence that "graph
beats sequential_react on core24 under good_adaptive/qwen2.5:7b" is real (not an artifact of the
earlier dead-key run, since it reproduces under a live search backend on an independent task
sample) but is underpowered relative to the original 144-cell plan for the token-cost claim
specifically. If the token-cost magnitude matters for a downstream decision, re-run the
remaining 16 `core24` tasks (not yet covered by this 8-task subset) under the same grounded
conditions before citing a specific token-savings number.
