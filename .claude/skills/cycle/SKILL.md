---
name: cycle
description: Walk a piece of webRAG work through the repeatable development-cycle methodology (plan, adversarial review, tests, benchmarks, implement, run, analyze). Use whenever the user says "start a cycle", "run a dev cycle", "/cycle", or asks how to approach a new piece of work on this repo in a structured way. Also use proactively before starting any Medium-or-larger change (a new tool, a multi-file subsystem change, a first live benchmark validation) so it gets sized and adversarially reviewed before code gets written.
---

# Dev cycle

This is a thin operational wrapper, not a process definition — `docs/DEV_CYCLE.md` in this repo is
the source of truth. **Read it now** before doing anything else; this file just tells you how to
apply it to whatever the user brought.

## Steps

1. **Read `docs/DEV_CYCLE.md`** in full if you haven't already this session.
2. **Size the work** using the doc's Micro/Small/Medium/Large tiers. Say which tier you picked and
   why in one sentence — don't ask the user to pick unless it's genuinely ambiguous (e.g. it could
   plausibly be Small or Medium depending on blast radius they'd know and you wouldn't).
3. **Before treating any existing handoff doc, design spec, or your own memory of the codebase as
   current, re-diff its concrete claims against `git log`/`git diff` on `HEAD`.** This isn't
   optional caution — it's what caught a stale, partially-already-shipped fix list the one time this
   methodology has been run so far. Docs on this branch go stale within days.
4. **Walk only the stages the sized tier calls for** (see the doc's sizing table for what's skipped
   at Micro/Small). Don't run the full ceremony on a one-line config change.
5. For each stage, use the doc's "Existing tooling per stage" table to find the right subagent or
   script rather than improvising — it maps every stage to something that already exists in this
   repo (`task-author`/`engine-dev`/`benchmark`/`reviewer` subagents, the `pytest` invocation
   convention, `gate_report.py`/`level_ladder.py`/`recovery_curve.py`/`unified_bench_report.py`).
6. For the **adversarial review** stage (Medium/Large only), spawn a small panel of `Plan` or
   `general-purpose` agents briefed to argue against the plan, per the doc's "Adversarial review,
   without new infrastructure" section — there's no dedicated review subagent, and that's
   intentional.
7. **Never treat a live benchmark run as pre-authorized by the plan alone** — confirm budget and
   go-ahead with the user at execution time, even mid-cycle.
8. When the cycle closes (Analyze stage done), summarize concretely what was found and what it
   implies for the *next* cycle's Plan stage — that hand-off is what makes this a loop instead of a
   one-off.
