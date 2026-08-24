# Breadth-suite weakness sweep + fixes, cycle 2 (2026-08-23)

Follow-on to `docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md` (which shipped
`candidate_coverage_gate` + `context_trim`, both now default-on, and confirmed the fix chain
end-to-end: task 152 went 0.21 → 0.93). This cycle started from a forensic pass over that
cycle's ~90 saved result JSONs, found three new weaknesses, fixed two of them, and then used a
$5 paid-API budget (OpenRouter, `openai/gpt-5-mini`) to cross-validate everything against a
much stronger model and broaden coverage beyond the 6 breadth tasks.

## Part 1 — Forensic findings from the prior cycle's data

1. **`coverage` scoring higher than `visit_count`** on 20/39 sampled cells (tasks 152/153) —
   initially hypothesized as a false-positive in `candidate_coverage_gate`'s fuzzy name-matching
   (a candidate credited as "covered" via an incidental mention on a DIFFERENT visited page).
   **Confirmed by code** to be a real, separate gap (see Part 2, Fix A) — but a targeted
   full-capture rerun found the ACTUAL cause of the sampled instances was different (see #2).

2. **The coverage-gate's corrective-extension budget was too small for wide fan-outs.** Two
   independent full-capture reruns on 7-way fan-out tasks (152, 156) confirmed: the fixed
   `_CANDIDATE_COVERAGE_EXTENSION_STEPS = 10` (→ `recursion_limit=20`) cannot fit the ~14 tool
   calls (7 search + 7 visit) a 7-candidate gap from zero needs. Rep1 (task 152) completed only
   6/7 visits before running out, guessing the 7th fact correctly from memory (lucky, not
   reliable). Rep2 (task 156) hit `GraphRecursionError` INSIDE the extension itself, producing a
   **completely empty final answer** despite 4 real visits having happened — worse than not
   extending at all, since none of that work made it into the returned `messages`.

3. **Task 154's flipped-verdict comparison bug, precisely diagnosed.** `qwen2.5:7b` retrieves
   both compared dam heights correctly in every sampled run (Grande Dixence 285m, Hoover
   726.4ft/221.4m) — no hallucination, no misattribution — but flips the final VERDICT in 10/12
   cells via two consistent failure modes: comparing a raw feet figure against a meters figure
   without converting ("726.4 meters - 285 meters = 441.4 meters, so Hoover is taller"), or
   computing the correct negative delta but flipping the sign in the stated conclusion.

4. **Tasks 156/157's `classification`/`item_classification` check never reached 1.0** in 0/12
   sampled cells at the best available config — the model reliably drops or garbles 1-3 of 7
   per-item PASS/FAIL verdicts against a stated numeric threshold, despite having every correct
   raw number in hand.

## Part 2 — Fixes shipped (local, qwen2.5:7b, offline-tested)

**Fix A — Identity-priority coverage matching** (`agent/app/idea_policies/candidate_coverage.py`,
`agent/app/langgraph_solver.py`). New `Haystack(identity, body)` type splits a visited page's
title/URL from its full content. A candidate now resolves via identity match first (the page is
actually ABOUT the candidate); a body match only counts within the first 1000 chars (lede
region) as a narrower fallback — never a mention buried deep in an unrelated page's body. This
closes the identical exposure in the native GoT engine's own `_node_haystacks` for free (shared
module). New `resolved_via` field records which path resolved each candidate, so a future
forensic pass doesn't need another full-capture rerun to tell them apart.

**Fix B — Scaled coverage-extension budget** (`agent/app/langgraph_solver.py`,
`_candidate_coverage_extension_steps`). Extension size now scales with the actual missing-count
(`max(10, missing * 3)`, capped at 40) instead of a fixed 10. This is the confirmed, higher-
priority fix for the two failures in finding #2 above — the "fixed size to avoid rewarding
under-resolution" concern from the original design doesn't apply here (this is a single grant
sized to THIS run's real gap, not a repeatable reward).

**Fix C — Unit-normalization prompt guidance** (`_UNIT_NORMALIZATION_GUIDANCE`, added to both
`_SYSTEM` and `_SYNTHESIS_SYSTEM`). Direct, cheap mitigation for finding #3 — explicitly
requires converting compared values to the same unit before comparing. Tried before building any
new mechanism, per this session's "cheap fix first" discipline.

**Fix D — Explicit per-item classification guidance** (`_CLASSIFICATION_GUIDANCE`, same two
prompts). Direct, cheap mitigation for finding #4 — requires an explicit PASS/FAIL verdict
stated for every named item, one line each.

All four offline-tested; 6029/6029 tests pass (`agent/tests/candidate_coverage_test.py`,
`agent/tests/langgraph_solver_test.py`).

## Part 3 — Live validation

### Local (qwen2.5:7b), all four fixes together, 3 reps on 152/153/156/157

| Task | Baseline mean (pre-cycle-2) | With all 4 fixes |
|---|---|---|
| 152 (7-way argmax) | ~0.90 | 0.71 / 0.96 / 1.00 |
| 153 (5-way argmin) | ~0.94 | **1.00 / 1.00 / 1.00** |
| 156 (7-way count) | ~0.40-0.89 (noisy) | 0.21 / 0.14 / 0.83 |
| 157 (7-way count) | ~0.60-0.70 | 0.43 / 0.43 / 0.81 |

153 improved to a clean sweep. 152 held steady. 156/157 still show variance — root-caused to a
**fifth, separate, still-open gap**: `always_synthesize` is opt-in (default OFF), so when the
model's last turn is unfinished mid-plan narration ("Let's start with the Vajont Dam...") with
no tool call, that narration is accepted verbatim as the final answer instead of being forced
through synthesis over the real evidence already gathered (task 156 rep1: 5 real visits'
worth of dam heights sitting unused in the message history, narration text scored 0.0 on every
check that needed a real answer).

**`always_synthesize` live A/B** (local, tasks 152/153/156/157, interrupted at n=2-3 per task —
partial, not conclusive): 156 improved substantially (+0.35, 0.397→0.75 mean) but 157 got WORSE
(-0.19, 0.556→0.37 mean) in this small sample. Mixed signal — **stays opt-in**, does not clear
this session's bar for a default flip. Flagged for a properly-sized future A/B.

### Paid cross-validation ($0.98 of a $5 budget, OpenRouter `openai/gpt-5-mini`)

**30-cell sweep, 6 breadth tasks × 5 reps, 0 infra failures, $0.585:**

| Task | Scores | Mean |
|---|---|---|
| 152 | 1.0, 1.0, 1.0, 1.0, 1.0 | **1.0** |
| 153 | 1.0, 1.0, 1.0, 1.0, 1.0 | **1.0** |
| 154 | 1.0, 1.0, 1.0, 1.0, 1.0 | **1.0** |
| 155 | 0.25, 0.38, 1.0, 1.0, 1.0 | 0.725 |
| 156 | 1.0, 0.88, 0.83, 1.0, 0.31 | 0.805 |
| 157 | 0.98, 0.89, 0.98, 0.88, 0.98 | 0.939 |

**Task 154 (the arithmetic/unit-mixing bug): PERFECT with a stronger model** — decisively
confirms it was a `qwen2.5:7b` capability limitation, not a benchmark or prompt defect. Tasks
152/153 also perfect. 156/157 much improved over the local numbers but not fully saturated —
consistent with a real, if smaller, residual difficulty even for a strong model.

**Task 155 — a NEW finding, distinct from anything qwen2.5:7b surfaced**: 2/5 reps hit
`step budget (25) exhausted` after 12 visits / 19-30 searches on a task that only needs 2 items
resolved — even the SUCCESSFUL reps used 7-10 visits for a 2-item comparison. This persists with
a strong model, so it's not a reasoning-capability gap — more likely a search/disambiguation
difficulty specific to this task's two aircraft (an over-exploration pattern worth its own
follow-up), or a `max_steps=25` budget too tight for how many attempts gpt-5-mini makes on this
specific pair.

**48-cell core24 sweep** (diverse task shapes 122-145, not breadth-specific, 2 reps, 0 infra
failures, $0.397): overall mean 0.851. Two consistent standouts:

- **Task 124** (0.4, 0.4 — perfectly consistent): NOT a bug. The task is a deliberate
  "leak-resistant" trap — its keystone wants the Tu-144D sub-variant's cruising speed
  (2,125 km/h) specifically, not the page's easily-found lead figure (2,200 km/h, a different
  Tu-144 variant/context). Even the strong model consistently grabs the easy number. A genuine,
  intentional task-difficulty finding, not a system defect — worth knowing about but not
  something to "fix" in the arm.
- **Task 145** (0.6, 0.6): got the right final answer (correct disambiguation + correct length)
  but under-visited (1 of 2 expected — this task's own "visit the wrong page first, then the
  right one" check is a DIFFERENT validator concept than `candidate_coverage_gate`, which
  correctly no-ops here since this is a single-entity task with no enumerated candidate list)
  and didn't cite the specific disambiguated page precisely enough to satisfy the citation
  check. A minor citation-precision issue on an otherwise-correct answer, not a reasoning bug.

## Part 4 — `require_finish_tool`: a well-motivated fix that live-tested NEGATIVE

Per a direct request to close the gap between `langgraph_react` and `sequential_react`, added a
`finish(answer)` tool (`agent/app/langgraph_solver.py`, `require_finish_tool` opt-in flag,
`_finish_answer` helper) — imitating `sequential_react`'s explicit `finish(answer)` action,
which is the actual reason that arm never suffers `langgraph_react`'s "narration accepted as
the final answer" bug. `create_react_agent` has no native "you must deliberately submit"
concept; any tool-call-free turn ends the run and whatever it said becomes the deliverable.
Design: a natural termination that never called `finish` is NOT trusted — it's discarded and
falls through to the existing forced-synthesis safety net instead of being accepted verbatim; a
real `finish` call's text is used verbatim (never rewritten, unlike `always_synthesize`, which
live-observed can make an already-good answer worse).

**Live A/B (paid, `openai/gpt-5-mini`, n=3/task, 5 tasks, 0 infra failures, $0.702) result:
NEGATIVE.** The mechanism's logic is sound and offline-tested (11 new unit tests, all pass) —
but adding a new tool to the action space measurably reduced step efficiency on already
step-constrained tasks: task 156 (7-way, needs the most tool calls) regressed sharply, mean
0.516 vs 0.960 with the flag off, hitting the fixed `max_steps=25` ceiling more often (confirmed
via saved `warning: step budget (25) exhausted` on the regressed cells). Task 155 also
regressed (0.75 vs 1.0). Tasks 152/153/157 (already near-ceiling or less step-constrained) were
unaffected. **Net: a real, unexpected efficiency cost outweighs the correctness benefit as
currently scoped.** Stays opt-in, default OFF — not recommended without also addressing the
step-budget interaction (e.g. scaling `max_steps` itself for this flag, mirroring how the
coverage-extension budget needed scaling in Part 2). Documented directly in the code's
docstrings so this isn't silently rediscovered.

This is a good example of the session's evidence-over-assumption discipline working as intended
— a well-reasoned hypothesis, cleanly implemented and tested, that real data refuted.

## Part 5 — Broad paid-API sweep (80 tasks, wide coverage) + an infra-classification bug found

80-cell sweep (tasks 001-080, 1 rep, `openai/gpt-5-mini`, $1.351) — the runner's `infra_failed`
flag showed 11/80 cells failed, but **this was a false signal**: all 80 cells actually produced
real, validly-scored deliverables (recomputed directly from `overall_score`: mean 0.847, median
0.938 across all 80 — consistent with core24's 0.851). Root cause: concurrency=4 caused
`ChromaDB failed to initialize after retries` errors on several concurrent cells (a resource-
contention issue, likely the same family as the previously-fixed "barrage ChromaDB hang" but not
fully eliminated at this concurrency level) — and `agent/app/testing/utils.py`'s
`_is_infra_timing`/`_summarize_infra` flags ANY failed timing named `visit`/`search_query`/
`http_request` with no status code as infra, with no awareness that **`langgraph_react` never
uses Chroma at all** (confirmed all session: "Chroma stores: 0, retrieves: 0" on every cell).
A Chroma init hiccup has zero bearing on this arm's actual task execution but still taints the
cell's `infra_failed` flag, silently excluding good data from any analysis that filters on it
(as this session's own earlier per-task tallies did). **Not fixed this cycle** — the shared
classification utility affects every arm/benchmark in the repo, so a fix needs its own
dedicated, careful validation pass. Flagged clearly here so a future cycle doesn't re-derive it,
and so any past analysis that trusted `infra_failed` at concurrency>1 gets re-checked against
raw `overall_score` instead.

Of the 80 real scores, 4 were genuine zeros (not infra-flag artifacts) — three worth a closer
look in a future cycle: task 045 ("Micro: Single-Page Fact Extraction") answered "Architectural
height: 828 m" for the Burj Khalifa — the factually correct figure — yet scored 0.0
("Height 828 m missing/incorrect"), smelling like an overly-strict validator regex rather than
a model error; task 023 ("Sequential Data Gathering") had the model ask a clarifying question
back ("Do you mean your OS...?") instead of attempting the task, since the benchmark harness has
no mechanism for a model to get that question answered; task 047 ("Graph: Wiki-Race Shortest
Chain") returned a bare URL list with no connecting explanation, a genuine incomplete answer.

## Summary of open items for a future cycle

1. `always_synthesize` needs a properly-sized live A/B (this cycle's was interrupted at n=2-3)
   before any default decision.
2. Task 155's over-exploration pattern (persists even with a strong model) — worth its own
   forensic dig into why gpt-5-mini can't quickly resolve a 2-item comparison for this specific
   task, independent of the fixes shipped this cycle.
3. Task 124-style "leak-resistant obscure figure" tasks are working as intended but might
   benefit from a prompt nudge ("read the SPECIFIC sub-variant/model's own figure, not just the
   first number on the page") — low priority, this is calibrated task difficulty, not a defect.
4. The core24 sweep surfaced no other systematic weakness beyond the two above — the arm
   performs solidly (mean 0.851) on a diverse task set with a strong model.
5. `require_finish_tool` (Part 4) needs `max_steps` scaling before it can be re-tested fairly —
   do not re-attempt a plain live A/B without that change first.
6. **`_is_infra_timing`'s Chroma-init false-positive (Part 5) is a real bug worth fixing** —
   affects any arm/analysis that filters on `infra_failed` at concurrency > 1, not just this
   session's data. Higher priority than it looks, since it silently discards good data rather
   than failing loudly.
7. Task 045/047/023-style genuine failures from the 80-task sweep — worth characterizing further
   (validator-strictness audit for 045-style cases in particular) in a future cycle.

## Budget summary (this session's paid API usage)

$3.432 of a $5 authorized budget spent across 189 paid cells (`openai/gpt-5-mini`, OpenRouter):
30-cell breadth sweep ($0.585), 48-cell core24 sweep ($0.397), 30-cell finish-tool A/B ($0.702),
80-cell wide sweep ($1.351), plus one initial cost-probe cell. Zero true infra failures across
all of it (the 11 flagged in the wide sweep were the false-positive described in Part 5).
