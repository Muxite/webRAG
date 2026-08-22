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
