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
  periodic look) should land on `master` rather than drift further from it. Cycle 1 ran this
  evaluation on `autoscale`/`autoscale-redux` (2026-08-10): both turned out fully superseded and
  were deleted, not merged — see `docs/handoffs/HANDOFF.md`'s "Git state" section for the current
  branch inventory and reasoning, and for any newer branches that show up on a later
  `git branch -a` pass.

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
| Run tests | `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/...` green, touched files byte-compiled. Required before any live spend. Venv-free equivalent: `cd services && docker compose --profile test run --rm agent-test`. | Never |
| Run benchmarks | Live-$ runs. Smoke (1×1) before a full matrix. Explicit budget authorization at execution time — never pre-authorized by a plan. | Change has no live-benchmark surface |
| Review results | Read raw output (`compare_arms.py` for an arm-vs-arm A/B, `kpi_dashboard.py` / `task_discrimination.py` for suite-wide health, or the older `gate_report.py` / `level_ladder.py` / `recovery_curve.py` / `unified_bench_report.py`) against any pre-registration. | Benchmarks were skipped |
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

## Autonomous cycles

The four tiers above assume a human is reachable to authorise a live run. When one is not — an
overnight batch, a maintainer travelling — that assumption converts every experiment into a stall.
The **Autonomous** tier does not remove the gate; it makes it *machine-checkable* rather than
human-checkable. The other four tiers keep their rules unchanged.

Autonomy is granted **per run class**, not globally:

| Run class | Cost | GPU | Autonomy |
|---|---|---|---|
| Offline tests | $0 | no | unrestricted, parallel |
| Corpus replay (`SEARCH_PROVIDER=corpus`) | $0 | no | unrestricted, parallel |
| Local inference (ollama) | $0 | **yes** | autonomous, serialised under the gpu-lock |
| Paid API | **$** | no | autonomous under a standing budget ceiling |

Corpus replay is what makes this safe. A replay costs nothing, touches no network and does not
contend for the GPU, so the dangerous category shrinks to runs that genuinely need live inference.

### The five machine gates

1. **Preregistration supplies the denominator.** Write the design before launching:
   `scripts/prereg.py write --spec <file>`, giving hypothesis, arms, tasks, reps, primary endpoint,
   budget and abort conditions. `scripts/prereg.py audit --run-id <id>` then compares what landed
   against what was designed. This closes the trap where **a dead cell writes no file** and vanishes
   from the denominator — `langgraph_react` silently lost 6–7 of 48 cells and its mean was computed
   over the survivors. A missing cell is a failure, never an absence.
2. **Budget is enforced in code, not by asking.** `IDEA_TEST_USD_CEILING`,
   `adaptive_ladder_run.py --budget`, and `LEDGER_MAX_LIVE_FALLBACKS` (live search under corpus
   replay, default 25). The standing budget lives in the prereg; exceeding it aborts the run.
3. **Singleton enforced by lockfile, not by memory.** `adaptive_ladder_run.py:acquire_pid_lock`
   already refuses to start against a held `driver.lock`. Use it. An autonomous agent cannot be
   trusted to remember that the benchmark path is a singleton; the file can.
4. **Abort conditions are pre-declared** in the prereg and checked as cells land: `infra_failed`
   rate, grounding rate, spend rate. `compare_arms.py` already refuses to print on ungrounded or
   auth-failed runs; the same predicate should abort a run, not merely decline to report it.
5. **Durability by default.** `setsid nohup … < /dev/null & disown` — backgrounded jobs die past
   roughly an hour. Resume is safe because `has_complete_result()` counts only finished
   `*_r1.json`. Keep slices ≤4 against `OLLAMA_NUM_PARALLEL=1`; 8-way slicing resynchronises and
   hammers a single-threaded backend.

### Auditability defaults must be flipped

`IDEA_TEST_KEEP_TRACES` and `IDEA_TEST_CAPTURE_LLM_IO` are both **off** by default, so raw prompts
and responses are never stored — a 96-cell baseline once yielded zero recoverable traces. Turn both
**on** for any cycle whose purpose is to explain a result rather than only to score one.
`telemetry.py` already records `t_start`/`t_end` against a monotonic anchor, so per-call timings and
overlap need retention, not new instrumentation.

### Not repeating an experiment

Result filenames already carry `cfg` + the first 8 hex of a sha256 over the sorted settings dict
(`idea_test_runner.py:1667`). Before launching, check whether `(task set, arms, cfg hash, model
digest)` has already been measured and skip those cells.

### Token discipline

Subagents read; the coordinator keeps conclusions. Handoff documents get one finding per section
with a hard cap — the 616-line append-only accretion in `DAG_V3_S1_…_2026-08-28.md` is the failure
mode to avoid. Raw numbers live in the result JSON and are linked, never pasted.

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
| Run tests | `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests`, or `docker compose --profile test run --rm agent-test` |
| Run benchmarks | `.claude/agents/benchmark.md`, `scripts/adaptive_ladder_run.py`, or `docker compose --profile ladder-benchmark run --rm ladder-benchmark` ($0 local models) |
| Review results / Analyze | `scripts/compare_arms.py` (N-way paired arm comparison, the current default), `scripts/kpi_dashboard.py` (suite-wide KPI table), `scripts/task_discrimination.py` (which tasks carry statistical power); older: `scripts/gate_report.py`, `scripts/level_ladder.py`, `scripts/recovery_curve.py`, `scripts/unified_bench_report.py` |
| Pre-commit gate | `.claude/agents/reviewer.md` |
| One variable at a time, live-gated | `.claude/agents/strategy-tuner.md` |

## Provenance

This structure was designed and its first cycle run on 2026-08-09. Cycle 1 caught three concrete
lessons worth keeping in mind for future cycles; a fourth was caught during the 2026-08-14
live-reverification pass on cycle 1's own fix set:

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
4. "Live-reverify a fix" is itself invalid unless the artifact under test was actually rebuilt from
   the fixed code. A 2026-08-14 codebench reverification's first attempt silently tested stale
   pre-fix behavior because `codebench-badmodel`'s Docker image `COPY`s the agent code in at build
   time, not a live mount, and hadn't been rebuilt since before the fix commit — caught only because
   the result looked wrong enough to investigate, not by any automated check. Nothing in this repo
   stamps an image with the git SHA it was built from or warns when that SHA is behind `HEAD`; worth
   adding if codebench live verification becomes routine rather than one-off.
