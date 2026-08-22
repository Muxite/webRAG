# Plan: reframe existing benchmark tasks toward genuinely parallelizable shapes (not yet started)

**Status: NOTED, NOT STARTED.** This is a forward-looking plan captured per explicit user
instruction, to be picked up in a future session. No task has been modified yet. Follow-up to
`docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md`, which found the
parallelism-compensates-for-a-weak-model hypothesis has essentially never been testable: the
active-59 suite has ~1 breadth-shaped task (052) and `core24` has zero.

## The principle, in the user's own framing

We will adjust a *large fraction of the existing benchmark suite* to fit a different design
principle than "author a handful of new breadth tasks and bolt them on." The reframe is:

> Benchmark tasks should be general tasks with no single, absolute sequential route to the
> answer — the way many real systems and real problems actually work. This is **not** about
> rigging tasks to make parallel execution look good. It rests on the observation that many
> real-world problems ARE, at some point, parallelizable — multiple independent pieces of work
> can be done at once, with a merge/comparison step only at the end.

The canonical example given: a **comparison task** ("is X bigger than Y", "which of A and B
happened first", "compare the population of city A to city B"). Today this suite mostly treats
such tasks the way a sequential agent naturally would — research A, then research B, then
compare. But the actual dependency structure is: **one arm retrieves everything needed about A,
a second arm retrieves everything needed about B, and neither arm needs the other's context
until the final comparison/merge step.** That's a real, common problem shape, not a contrived
one — and it's exactly the shape the graph engine's parallel fan-out + merge machinery was
built for. If the suite doesn't contain tasks with this shape, the engine's core mechanism is
being scored against a population it was never designed to help with.

## Scope: this is a suite-wide reframe, not a "write 5-8 new tasks" patch

The investigation doc already recommended authoring 5-8 new breadth tasks as the immediate
next $0 experiment — that recommendation stands and is the right FIRST step (cheapest way to
get an initial signal). This note captures the larger, slower follow-on: once that initial
signal exists, go back through the **existing** task suite (`agent/app/idea_tests/`,
`agent/app/BENCHMARK_SUITE_50.md`'s shape taxonomy) and identify which currently-chain-or-
survivor-labeled tasks *actually* have an independent-arms-then-merge structure once you look
past their current framing, and reframe/re-author them rather than always adding net-new tasks.
Candidates to look for, in rough priority order:

1. **Existing comparison-shaped tasks** (any task whose keystone requires comparing two or
   more named entities/quantities) — audit whether their current task statement/validator
   forces a sequential "look up A, then look up B" narrative, or whether the underlying
   information need is genuinely independent per side. Most comparison tasks are the latter by
   nature; the reframe is often just making the task statement and validator agnostic to
   execution order, not changing what's being asked.
2. **Survivor-shaped tasks** (9 in the active suite) — a survivor task typically asks "which of
   N candidates is still true/active/correct as of some date," which usually means each
   candidate's status can be checked independently and merged via elimination. Check whether
   any of the 9 already implicitly require this shape but are scored/validated in a way that
   assumes sequential elimination.
3. **Count / AND-filter tasks** — "how many of N items satisfy property P" is naturally one
   independent check per item, merged by counting. Likely straightforward reframe candidates.
4. **Conflicting-source tasks** (8 in the suite) — already the closest existing analogue to
   this shape (independently visit multiple sources, then reconcile), and the natural home for
   `race_value_agreement` (shipped this session, see `project_strong_agent_trace_guidance`).
   Worth checking whether these already exercise real parallel-arm-then-merge execution or
   whether they're being solved sequentially in practice despite the shape allowing otherwise.

## What "reframe" concretely means, mechanically

For each candidate task: (a) confirm via the task's actual keystone/validator (not just its
label) that the sub-parts are genuinely independent — no sub-part's correct retrieval strategy
depends on another sub-part's answer; (b) if the task statement currently implies or requires a
specific execution order, loosen the wording so a parallel-arms-then-merge solution is not
penalized relative to a sequential one (validators should already be shape-agnostic if written
correctly — this may surface validator bugs, not just task-authoring gaps); (c) leave the
task's keystone/gate/scoring untouched unless the audit in (a) finds it silently assumed
sequential execution somewhere. This is closer to a validation-and-reframing pass than a
rewrite of the suite's content.

## Explicit non-goal

**Do not build or bias tasks to make the graph engine's score go up.** The point is to remove a
structural blind spot in what gets measured, not to engineer a favorable result. If a fairly-
reframed, genuinely-parallelizable task population STILL shows no graph advantage over
`seq_react`, that is exactly as valid and actionable a finding as if it does — see the
investigation doc's own explicit fallback (retire the raw-accuracy-via-parallelism hypothesis
and fall back to the narrower correctness-under-conflicting-sources niche).

## Sequencing relative to other open work

1. First (already recommended, not yet started): author 5-8 net-new genuinely breadth-shaped
   tasks, run the same $0 local paired A/B methodology used all session on `good_adaptive` vs
   `seq_react`.
2. Only after (1) produces a signal worth acting on: begin the larger suite-wide audit/reframe
   described above, task family by task family, each validated with its own live-calibration
   pass (matching this repo's existing `task-author` agent workflow — never re-author a task's
   validator without live-verifying the new ground truth, same discipline as every other task
   in this suite).

## Related

- `docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md` — the investigation that
  triggered this plan
- `project_graph_vs_seqreact_gap_investigation` (memory)
- `agent/app/BENCHMARK_SUITE_50.md` — current shape taxonomy and per-shape task counts
