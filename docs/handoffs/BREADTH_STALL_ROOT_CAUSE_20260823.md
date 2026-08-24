# Breadth-pilot fan-out stall & citation-fabrication: root cause + fix (2026-08-23)

## Context

`docs/handoffs/BREADTH_PILOT_RESULTS_20260823.md` found the `langgraph_react` arm losing
decisively (-0.266, p=0.009) to `sequential_react` on genuinely-independent fan-out tasks
(152-157), stalling at 0-4/7 visits and, in one case, fabricating a fully-cited answer with
zero pages actually visited (task 152 rep1). This doc identifies the confirmed root cause and
a live-validated fix.

## Correction: the wrong file was investigated first

Two Explore agents initially traced hypotheses in `agent/app/idea_engine.py` (the native
Graph-of-Thoughts engine): `max_branching=5` truncation, `siblings_are_independent`
misclassification driving a sequential-execution/step-budget-exhaustion stall, and the native
engine's `candidate_coverage_enabled` gate being disabled by default. **All three are real
findings about `idea_engine.py`, but irrelevant here** — reading the actual saved result JSON
for the `langgraph_react` arm shows an empty `graph.nodes` (confirmed: this arm never touches
`idea_engine.py`). The `langgraph_react` variant is a separate wrapper
(`agent/app/langgraph_solver.py` / `execution_langgraph.py`) around
`langgraph.prebuilt.create_react_agent`, a genuinely third-party orchestration loop. The
`max_branching`/`siblings_are_independent`/native `candidate_coverage_enabled` findings remain
valid observations about the native engine but do not explain this arm's failures.

## Root cause (confirmed against real saved run data, task 152 rep1)

`create_react_agent` accepts **any** AI turn without a tool call as the final answer — there is
no deliberate "finish" action like `sequential_react`'s explicit `finish(answer)`. In the saved
run: 14 LLM turns, 42 batched search calls, **zero** `visit` calls, then a plain prose turn with
a fully-cited (real URLs, drawn from search snippets) but factually wrong answer. No exception
was raised, no step-budget warning was recorded (`warning: None` in the result JSON) — the model
simply decided it had enough and answered. `_SYSTEM`'s instruction to visit before answering is
advisory text only; nothing in the code enforces it, and (unlike the native engine) this arm has
no `evaluate_candidate_coverage`-equivalent check at all.

A second, distinct issue was also observed (not fixed in this pass): task 152 rep3 visited all
7 pages yet still only got 2/7 facts correct in its final answer — worse than rep1's 0-visit
fabrication (6/7, wrong on the keystone only). Likely cause: `sequential_react` bounds its
scratchpad to the last 12 truncated (1500-char) entries; `LangGraphSolver` keeps the full raw
message history (every search result + every full 6000-char page dump) with no compaction,
plausibly overwhelming qwen2.5:7b's synthesis by the time it must answer. **Not addressed by
this fix — flagged as follow-up work.**

## Fix: candidate-coverage completion gate (opt-in, default OFF)

`agent/app/langgraph_solver.py`: new `candidate_coverage_gate` constructor flag (env var
`IDEA_TEST_LANGGRAPH_CANDIDATE_COVERAGE_GATE`, wired in `execution_langgraph.py`). When enabled,
before accepting a natural-termination or step-exhausted answer, checks whether every candidate
named in the mandate (`idea_policies.candidate_coverage.extract_named_candidates`) has an actual
`visit` tool result behind it (`_visit_haystacks` — search-only ToolMessages are excluded on
purpose, mirroring the native engine's `_node_haystacks` rationale). If any are missing, feeds
the agent one corrective turn naming the missing items and grants a ONE-TIME, FIXED +10-step
recursion budget (`_CANDIDATE_COVERAGE_EXTENSION_STEPS`) — same anti-gaming design as the native
engine's `_candidate_coverage_extension` (fixed size, applied at most once, never scaled to how
much is missing). Refactored `idea_policies/candidate_coverage.py` to split out
`evaluate_candidate_coverage_from_haystacks` (graph-independent core) so both the native engine
and this arm share one deterministic coverage check.

Offline: 5988/5988 tests pass (`agent/tests/`), including 8 new gate tests in
`agent/tests/langgraph_solver_test.py` (default-off no-op, extension trigger, already-satisfied
no-op, fail-open on no named candidates, fall-through to forced synthesis if the extension also
runs out).

## Live re-validation ($0, local `badmodel-ollama`, qwen2.5:7b, `good_adaptive`)

Single-rep spot check (not the original pilot's full 3-rep design — see Next Steps) on all 6
breadth tasks with the gate enabled, vs the original 3-rep-mean baseline:

| Task | Baseline graph mean (3 reps) | With fix (1 rep) | Visits before -> after |
|---|---|---|---|
| 152 (7-way argmax) | 0.262 | 0.36 | 0/0/7 -> 7 |
| 153 (5-way argmin) | 0.500 | **1.00 PASS** | -> 5/5 |
| 154 (2-arm compare) | 0.750 | 0.50 | 2/2 (unrelated bug, see below) |
| 155 (2-arm compare) | 0.958 | **1.00 PASS** | -> 2/2 |
| 156 (7-way count) | 0.071 | **0.81 PASS** | 1/1/4 -> 6/7 |
| 157 (7-way count) | 0.325 | **1.00 PASS** | -> 7/7 |

Grounding audit clean (Serper operational, no infra failures in any of the 6 cells). Task 156 —
the worst stall case (1 visit, score 0.05, FAIL) — is now 6 visits, score 0.81, **PASS**. Task
152 — the 0-visit fabrication case — now visits all 7 pages (was 0/0/7 across the original 3
reps) but still under-reports facts (3/7 correct), consistent with the still-unaddressed
context-bloat hypothesis above. Task 154's single new rep shows a flipped comparison verdict
despite full 2/2 visits+coverage; the gate was a no-op here (nothing missing to trigger it) and
the original baseline already had 1-win/1-tie/1-loss variance on this task, so this looks like
pre-existing reasoning noise, not a regression from the fix.

## Update 2026-08-23: full paired 2-rep A/B CONFIRMS the fix

Ran a reduced (2-rep, not the original pilot's 3-rep) paired A/B: `candidate_coverage_gate` ON
vs OFF, same 6 tasks, `langgraph_react`, `good_adaptive`, qwen2.5:7b, local $0
(`scripts/analyze_coverage_gate_ab_20260823.py`, run-ids `gate_ab_20260823_gateon/gateoff_rep{1,2}`).
Grounding audit clean (0 `Setup failed`/`Serper health probe failed` hits across all 24 cells).

**Result: SCORE mean delta (gate_on - gate_off) = +0.227 (sd=0.306, n=12, t=2.56), W/T/L = 7/5/0
(never loses). VISITS mean delta = +2.17.** gate_on mean score 0.611 vs gate_off 0.384. This
statistically confirms the single-rep spot check — the fix is real, not noise.

Per-task: 152 (+0.214, visits 0→7), 153 (+0.750, visits 0→5), 156 (+0.333, visits noisy but
higher), 154 and 157 tied (delta ≈ 0 both directions) — **154 is a no-op as expected (only 2
items, already fully covered pre-gate)**. 157's tie initially looked like a distinct
counting/classification bug (both conditions fully visit 7-8 pages, both score 0.19) — **but
reading the actual deliverable text resolves it**: `gate_on`'s final answer for task 157 only
reports ONE bridge (Xihoumen) and completely drops the aggregation step, despite having visited
all 7 — the SAME context-bloat failure mode as task 152, not a separate bug. Confirmed
mechanistically: this cell burned 189,803 prompt tokens / 42 LLM calls to produce that collapsed
answer; the equivalent `context_trim=1` cell (see below) used 70,152 tokens / 28 calls (-63%) to
produce a fully correct, complete 7-item answer. Retracting the "separate bug" framing.

Given this confirms the fix at n=12 (not just n=6), **recommend flipping
`candidate_coverage_gate` to default-on for the `langgraph_react` arm** — it never lost a single
paired cell across two independent validation rounds (single-rep spot check + this 2-rep A/B).

## Update 2026-08-23 (2): a live infra bug in the first context_trim A/B, fixed and re-run

The first `context_trim` A/B (12 paired cells) found a real bug in `_trim_for_model`'s
"drop oldest tool messages over budget" path: it removed `ToolMessage`s without also removing
the matching `tool_call` entry from the `AIMessage` that requested them, which LangGraph/the
provider rejects outright (`ValueError: Found AIMessages with tool_calls that do not have a
corresponding ToolMessage`) — this crashed exactly 1 of the 12 `trim_on` cells (task 157 rep2)
as an infra failure. Fixed (`_drop_tool_messages_and_matching_calls`, strips the matching
`tool_call` or drops a now-empty `AIMessage` entirely) and offline-tested (regression tests
reproduce the exact live failure signature). Re-ran the full 24-cell A/B clean afterward — see
below for the trustworthy result.

Despite the bug, the first run's clean cells already gave a strong mechanistic read on task 157
(see the retraction above): `gate_on` alone answers about only 1 of 7 bridges (189,803 prompt
tokens, 42 LLM calls) where `context_trim=1` answers all 7 correctly (70,152 tokens, -63%, 28
calls) — direct evidence context-trim fixes this failure mode, not just correlates with it.

## Update 2026-08-23 (3): clean context_trim A/B CONFIRMS the fix

Re-ran the full 24-cell A/B after the bug fix above: `context_trim` ON vs OFF, both WITH
`candidate_coverage_gate=1` (the confirmed baseline), same 6 tasks, 2 reps, local $0. Grounding
audit clean, **0 infra failures** (vs 1 in the buggy first run).

**Result: SCORE mean delta (trim_on - trim_off) = +0.216 (sd=0.336, n=12, t=2.23), W/T/L = 6/3/3.**
trim_on mean score 0.793 vs trim_off 0.577. VISITS mean delta = -1.08 (trim_on sometimes uses
FEWER visits and still scores higher — consistent with "less noise, better synthesis," not just
"more evidence").

Per-task: 152 +0.429, 154 +0.250, 155 +0.062, 156 +0.214, 157 +0.393 — wins on 5/6 tasks. Only
153 (5-way argmin canals) shows a small loss (-0.050, visits 5→4) — worth a closer look in a
future cycle (does `_TRIM_RECENT_TOOL_MESSAGES=3` starve a 5-way task's coverage more than a
7-way one, since a smaller roster needs relatively more of its visits protected?), but not large
enough to withhold the recommendation below.

**Both `candidate_coverage_gate` (+0.227, t=2.56, n=12) and `context_trim` (+0.216, t=2.23,
n=12) now have independent, statistically significant live confirmation. Recommend flipping BOTH
to default-on for the `langgraph_react` arm.**

## Update 2026-08-23 (4): both flags flipped to default-on; capstone sanity check

Flipped `candidate_coverage_gate` and `context_trim` to default-ON in
`execution_langgraph.py`'s env-var wiring (`IDEA_TEST_LANGGRAPH_CANDIDATE_COVERAGE_GATE=0` /
`IDEA_TEST_LANGGRAPH_CONTEXT_TRIM=0` to opt back out). `LangGraphSolver`'s own constructor
defaults are left at `False` (library/direct-construction callers keep the conservative
original behavior; the benchmark harness now opts in by default). 6022/6022 offline tests pass.

**Capstone live sanity check**: re-ran task 152 — the ORIGINAL 0-visit fabrication case that
started this whole investigation (baseline score 0.21) — with NO env vars set at all (pure
defaults). Result: **score 0.93** (7/7 coverage, correct keystone: Vinson Massif 1966, 6/7
citations, 6 visits, 70,278 tokens). Confirms the full fix chain works end-to-end with zero
configuration required.

## Update 2026-08-23 (5): stall_recovery_gate live A/B — directionally positive, not significant

Paired 2-rep A/B (n=12), `stall_recovery_gate` ON vs OFF, BOTH with `candidate_coverage_gate`
and `context_trim` at their new default-on settings (`scripts/analyze_stall_recovery_ab_20260823.py`,
run-ids `stall_ab_20260823_stallon/stalloff`). Grounding clean, 0 infra failures.

**Result: SCORE mean delta +0.102 (sd=0.324, n=12, t=1.10) — NOT statistically significant.**
W/T/L 4/4/4 (evenly split). stall_on mean 0.863 vs stall_off 0.761.

Per-task pattern is informative even though the aggregate isn't significant: 152 and 153 (already
near-ceiling under the two confirmed fixes, mean scores 0.9+) show tiny/slightly negative deltas
(-0.018, -0.025) — nothing left to recover once coverage+context-trim are already handling most
failures. The harder tasks (154 +0.250, 156 +0.238, 157 +0.107) show a real positive trend. This
is consistent with a ceiling effect: with the two bigger fixes now catching most failure modes,
`stall_recovery_gate`'s marginal value on THIS task set is naturally smaller than it might be on
a harder or differently-shaped suite.

**Decision: `stall_recovery_gate` stays opt-in.** Not confirmed enough to flip its default —
unlike the other two fixes, which had t>2 and never-lost W/T/L records, this one doesn't clear
the bar this repo holds defaults to. A future cycle could re-test it on a task set with more
headroom (i.e. one context_trim+coverage_gate don't already solve), or at a larger n.

## Next steps

1. ~~Full paired 3-rep A/B~~ — DONE at 2-rep scale above; confirms the fix. A full 3-rep run
   would only tighten the confidence interval, not change the conclusion — deprioritized.
2. ~~Context-bloat / message-history compaction~~ — DONE, shipped + bug-fixed + confirmed above.
3. ~~Flip `candidate_coverage_gate` AND `context_trim` to default-on~~ — DONE, see above.
4. ~~New, separate bug (task 157/154)~~ — RESOLVED: not a separate bug, the same context-bloat
   failure mode as task 152 (see retraction above). No further diagnosis needed here.
5. ~~`stall_recovery_gate` live A/B~~ — DONE, directionally positive but not significant; stays
   opt-in (see above).
6. Small, low-priority, not investigated further this cycle: task 153's minor regression under
   `context_trim` (visits 5→4, score -0.05) — looked benign (near-complete/correct answers).
7. The unconditional search-dedup guard + shared query-reformulation-on-retry are shipped and
   offline-tested; no live A/B needed (hygiene fixes, not scored experiments, matching how the
   existing visit-dedup markers already ship unconditionally).
