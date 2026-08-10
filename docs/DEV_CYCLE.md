# Development cycle

webRAG's day-to-day work has run as a series of one-off `*_HANDOFF.md` documents: useful within a
session, but they go stale fast once another session lands unrelated commits, and there's no
consistent point where a plan gets challenged before code gets written. This doc replaces that habit
with a repeatable loop, sized to the change instead of applied uniformly.

## What a cycle is for

A cycle is any of four things, or a combination:

- **Feature** — new capability, new tool, new benchmark surface.
- **Bug fix** — a real, reproduced defect traced to a root cause, not a symptom patch.
- **Cleanup** — retiring dead/superseded code, de-duplicating, or relocating something to its
  natural home when it's drifted from it (cycle 2's codebench fold-in was this: infrastructure that
  had ended up living inside a lab-scoped directory it didn't conceptually belong to).
- **Branch merge** — evaluating whether work sitting on another branch (`git branch -a` is worth a
  periodic look — this repo currently also has `autoscale`/`autoscale-redux` sitting unmerged)
  should land on `master` rather than drift further from it.

The loop and sizing below apply the same way to all four — a cleanup cycle still needs a real Plan
stage (what's actually unused vs. quietly load-bearing? — see lesson 2 in Provenance below), and a
branch-merge cycle still needs adversarial review (has the incoming branch's code drifted from
what master now assumes?), not just a mechanical `git merge`.

## The loop

| Stage | What happens | Skip when |
|---|---|---|
| Plan | Restate the goal, name affected files/subsystems, size the cycle (below), propose an approach. **Re-diff any existing handoff/spec doc's claims against `HEAD` before trusting it** — docs drift within days on a branch with concurrent sessions. | Never |
| Adversarial review | Red-team the plan before writing code: hidden couplings, stale assumptions, unnecessary scope. | Micro cycles |
| Write/adjust tests | Author the offline `pytest` tests or `idea_tests` task modules that will gate the change. | Never (tests may land alongside implementation, not after) |
| Adjust benchmarks | Update benchmark scripts/task definitions/reporting if the change affects a live benchmark's shape. | Change doesn't touch benchmark surface |
| Implement | Write the code. | Never |
| Run tests | `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/...` green, touched files byte-compiled. Required before any live spend. | Never |
| Run benchmarks | Live-$ runs. Smoke (1×1) before a full matrix. Explicit budget authorization at execution time — never pre-authorized by a plan. | Change has no live-benchmark surface |
| Review results | Read raw output (`gate_report.py` / `level_ladder.py` / `recovery_curve.py` / `unified_bench_report.py`) against any pre-registration. | Benchmarks were skipped |
| Analyze | Trace failures to root cause, not just an aggregate score. Findings become the next cycle's Plan input — this is what closes the loop. | Never |

## Sizing

Match the ceremony to the change. Most of this repo's work is Micro or Small; reserve Medium/Large
for what actually needs it.

- **Micro** — mechanical, single-fact change (a roster addition, a config toggle). Plan = one
  sentence. Adversarial review skipped. No benchmark adjustment.
- **Small** — single-file/module fix or feature, modest blast radius. Plan = a few sentences +
  affected files. Review = one self-critique pass, or a single agent check. New unit tests required.
- **Medium** — multi-file subsystem change, a new tool, or the first live validation of already-built
  infrastructure. Plan = written inline or as a short doc. Review = an ad hoc 2–3 agent adversarial
  panel (see below). Full stage sequence.
- **Large** — cross-cutting architecture change or a new capability thesis test. Plan = a full
  written spec under `docs/superpowers/specs/` (via the `brainstorming` skill). Review = a full
  adversarial panel; pre-registration required before any live run (see
  `scripts/LADDER_PREREGISTRATION.md` for the precedent).

## Adversarial review, without new infrastructure

There's no dedicated "plan reviewer" subagent, and building one before the pattern's proven itself
would be premature — `reviewer.md` is a pre-commit diff gate, a different job. Instead, spawn a
small panel of `Plan` or `general-purpose` agents, briefed explicitly to argue against the plan
(devil's-advocate framing), not to rubber-stamp it. Ask each to re-verify specific claims against
current `HEAD` rather than trust the plan's own summary of them — this caught a stale fix-list in
this methodology's own first cycle (see below) and is the single highest-value thing a review pass
does on this branch.

## Parallelism

- Independent files/subsystems implement in parallel (the `Agent` tool for a handful of agents, or
  `Workflow` for larger fan-out).
- Tests may be authored in parallel with implementation, but must gate before any live run.
- Live-$ runs are constrained by **two distinct, non-interchangeable locks** — don't blanket-serialize
  everything that spends money:
  - the `benchmark` subagent's OpenRouter-backed singleton (`concurrency=1`, shared connectors) —
    see `.claude/agents/benchmark.md`;
  - a separate local `gpu-lock` for Ollama-contending local-model work.
  A cycle only needs to serialize steps that actually contend for the *same* lock.

## Existing tooling per stage

Most stages already have a tool-shaped home; the loop above is mainly what ties them together.

| Stage | Tooling |
|---|---|
| Plan / adversarial review | `Plan` / `general-purpose` agents (ad hoc panel, see above) |
| Write tests / adjust benchmarks | `.claude/agents/task-author.md`, `codebench-task-author.md`, `reasoning-task-author.md` |
| Implement | `.claude/agents/engine-dev.md` |
| Run tests | `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests` |
| Run benchmarks | `.claude/agents/benchmark.md`, `scripts/adaptive_ladder_run.py` |
| Review results / Analyze | `scripts/gate_report.py`, `scripts/level_ladder.py`, `scripts/recovery_curve.py`, `scripts/unified_bench_report.py` |
| Pre-commit gate | `.claude/agents/reviewer.md` |
| One variable at a time, live-gated | `.claude/agents/strategy-tuner.md` |

## Provenance

This structure was designed and its first cycle run on 2026-08-09. Cycle 1 caught three concrete
lessons worth keeping in mind for future cycles:

1. A handoff doc (`BARRAGE_RELAUNCH_HANDOFF.md`) turned out to be ~15% stale against `HEAD` within
   two weeks — several of its listed fixes had already shipped under unrelated commits, while its
   actual root-cause fix was still missing. Nothing will flag this automatically; the Plan stage's
   re-diff-against-HEAD step exists because of this.
2. Two threads assumed to need independent tooling (a live benchmark barrage, and a first live
   codebench run) turned out to share a reporting layer once actually looked at
   (`scripts/unified_bench_report.py`, reading both `idea_test_results/*.json` and codebench's
   `runs.jsonl`) — worth checking for this kind of overlap at Plan time before building two of
   something.
3. A clean adversarial review and a green offline suite still didn't catch a real engine bug: the
   barrage's own $3 confirmation smoke found `idea_engine.py`'s re-expansion guard can make the
   `good_adaptive` arm self-loop to zero visits and silently exhaust its step budget on a common
   task shape, producing a near-zero score with no error anywhere in the driver's own accounting.
   Nothing upstream of a live run would have surfaced this — it's why "run benchmarks" stays a real
   stage even when a change looks purely infrastructural, and why a smoke's job is to look for
   exactly this kind of silent failure, not just confirm the driver doesn't crash.
