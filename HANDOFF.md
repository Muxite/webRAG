# webRAG / Euglena — Session Handoff (2026-07-10)

Start here. This is the project-wide pickup doc. It has been rewritten from scratch this session —
everything before 2026-07-08 in the old version is superseded; see git history if that era's detail
is ever needed. Deep logs live in three companion docs under `services/agent/app/`:
`COST_BENCHMARK_HANDOFF.md` (the closed-out compiled-scaffold campaign),
`ADAPTIVE_DISTILLATION_HANDOFF.md` (the current active research line, Phases 1-4), and
`SYSTEM_STATUS.md` (a cross-cutting capabilities/issues/debt snapshot). External research citations
backing several design decisions live in `RESEARCH_NOTES.md`. The active plan/roadmap is
`/home/muk/.claude/plans/a-lot-of-work-gleaming-hejlsberg.md`.

## ⚠️ Nothing is committed — read this first

**89 files changed/new, ALL uncommitted**, on branch `compiled-scaffold-dag`, 18 commits ahead of
`origin/compiled-scaffold-dag`. Last commit: `4140945` (2026-07-08, tier-5 suite expansion). Every
line of work described below — the entire closed-out compiled-scaffold campaign's late additions,
plus all of Phases 1-4 of the adaptive-engine program — is sitting in the working tree, not git
history. This has been true across multiple sessions by the user's own choice (uncommitted by
instruction, not oversight), but it's the single most important fact for anyone picking this up:
**verify `git status` before doing anything destructive, and don't assume `git log` reflects
current reality.**

## Current test baseline

`PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest services/agent/tests/` →
**802 passed, 18 skipped, 3 failed.** The 3 failures are pre-existing, unrelated, and undiagnosed:
`test_063_strict_csv_validators_test.py::{test_hallucinated_keystone_value_zero,
test_partial_coverage_scores_fraction, test_compiled_plan_is_pure_fanout_and_leaks_nothing}` — they
have been present and stable at every baseline check across this entire session (734→...→802
passed, always the same 3 failing). Not flaky; genuinely unfixed debt (see `SYSTEM_STATUS.md`).

## The thesis (proven, campaign closed out 2026-07-08)

A cheap model executing an **expensive-model-authored DAG scaffold** recovers premium-reference
accuracy at a fraction of the dollar cost. Final validated numbers (38 tasks × 3 models × R=3,
1,026 live runs, ~$38 spend): `gpt-5-mini` + compiled scaffold **ties premium** (`gemini-3.1-pro`)
at **10% of the cost**; `gpt-4.1-nano` + compiled scaffold reaches **~93%** of reference quality at
**~1/85th the cost**. Significant on the hardest tier (95% CI-disjoint, Cohen's d up to 2.7).
Packaged results: `linkedin_package_38tests_2026-07-08/` (untracked, portfolio artifacts). Full
history: `COST_BENCHMARK_HANDOFF.md`.

## The active research line: adaptive engine + local-LLM reliability

Started 2026-07-09 because the compiled-scaffold suite is mostly parallel fan-out/merge — the user
wanted genuine multi-stage sequential reasoning, and a reframe toward **local LLM deployment (no
strong model ever in the runtime path, only offline dev-time use of Claude Code's own Opus)**. Four
phases so far, all documented in depth in `ADAPTIVE_DISTILLATION_HANDOFF.md`:

**Phase 1 — Opus-exemplar distillation.** Had Claude Code's own Opus (never an API call, $0) solve
three task shapes live and distill fact-free narrative reasoning exemplars. **Result: unreliable.**
Backfired twice on the hardest shape (branch-eliminate) — weak models copy an exemplar's surface
structure, not its intent. A seeming "win" on the parallel-chains shape dissolved to noise at R=3.
Lesson that shaped everything after: **never trust an R=1 result; always confirm at R=3.**

**Phase 2 — rule mining + code-enforced gates.** Pivoted from prompt-only to deterministic
mechanisms: a candidate-coverage completeness gate (`idea_policies/candidate_coverage.py`), a flat
imperative rule checklist (`reasoning_rules/branch_eliminate.md`, replacing narrative), and a
deterministic task-shape classifier (`idea_policies/shape_classifier.py`, 7/7 validated accuracy).
**Found and fixed 4 real bugs** along the way, all the same failure class (a verifier "satisfied"
by evidence that looks right but isn't proof of real execution — literature name: "implicit-as-
explicit verification failure"). One of the four was a pre-existing bug in `test_095`'s own scoring
validator, unrelated to this session's new code, incidentally surfaced.

**Phase 3 — capped budget extension. RETIRED.** Hypothesis: the gate detects incomplete coverage
but has no budget left to act on it. Built a fixed, one-time, evidence-only-triggered budget
extension (`got_candidate_coverage_budget_extension`). At R=3 live validation it fired exactly as
designed but was **functionally inert** — its "re-activate root" trigger doesn't cause real
re-expansion when the root's children are already `done`. Per the plan's own stop rule (4 bug-fix
rounds + 1 design change without success), **the candidate-coverage mechanism is retired** — stays
in the codebase, opt-in, `false` by default, fully documented so this dead end isn't rediscovered.

**Phase 4 — ConSol sampling pilot. Validated with caveats, not broadly wired.** Built an opt-in
(`IDEA_TEST_USE_CONSOL=1`) early-stopping wrapper (real PyPI package `consol` 0.3.0, SPRT-based)
around the leaf-extraction self-consistency vote. Live pilot: **trustworthy** (identical answers,
identical score distribution to fixed-k voting) with **real but modest cost savings** (~27%/run
cheaper) and a **genuine wall-clock trade-off** (~60% slower — sequential sampling, not parallel).
Verdict: keep opt-in for offline/batch use, not a default. `consol` intentionally NOT added to
`requirements.txt` (heavy langchain/langgraph deps) pending a broader-adoption decision.

**Total campaign spend across all 4 phases: ~$1.08 of a $12 authorized ceiling.**

## What's live vs. retired vs. opt-in (the toggles that matter)

| Mechanism | Status | Toggle |
|---|---|---|
| Compiled-scaffold execution | **Proven, production path** | `IDEA_TEST_EXECUTION_VARIANTS=graph_compiled` |
| Adaptive leaf re-expansion | Sound, opt-in, not yet a proven win | `IDEA_TEST_GOT_REEXPAND=1` / `got_reexpand_enabled` (JSON default `false`) |
| Narrative reasoning exemplars | **Disproven** — actively backfired | `IDEA_TEST_REASONING_EXEMPLAR=chain\|mixed\|parallel` — avoid, kept only for reference |
| Flat rule checklist | Built, inconclusive at R=3 | `IDEA_TEST_REASONING_RULES=branch_eliminate` (only shape with a file); auto-classifies via `shape_classifier.py` when unset |
| Candidate-coverage gate | **Retired** — do not re-enable without a new re-expansion trigger design | `got_candidate_coverage_enabled` (JSON default `false`) |
| Coverage-gate budget extension | **Retired**, same reason | `got_candidate_coverage_budget_extension` (JSON default `10`, harmless since the gate itself is off) |
| ConSol early-stop voting | Opt-in, validated for offline/batch use only | `IDEA_TEST_USE_CONSOL=1` (not in `requirements.txt`) |
| `deepseek/deepseek-v4-flash` | New roster model, JSON-mode confirmed | `testing/config.py` `BENCHMARK_ROSTER["experiment"]` |

## Architecture map (`services/agent/app/`)

- **Engine:** `idea_engine.py` (`IdeaDagEngine`) + `got_operations.py`, `idea_policies/*`
  (typed config in `config.py`, actions in `actions.py`, expansion/grounding/candidate-coverage
  policies, `shape_classifier.py`). Settings: `idea_dag_settings.json` (many independent `got_*`
  toggles — see `SYSTEM_STATUS.md` for the under-tested-interactions concern).
- **⚠️ Known architecture debt**: `idea_engine.py::run()` and `testing/execution.py::
  run_test_execution()` are TWO INDEPENDENT REIMPLEMENTATIONS of the same control loop. This is
  what let Phase 2's bug #4 hide (a fix landed in one copy, not the other). Not yet consolidated —
  see Priority 3 in the plan file, gated on the Strangler Fig pattern + a dedicated parity test
  suite (NOT the benchmark suite, which is itself one of the two things being unified).
- **Compiled scaffold:** `testing/compiled_plan.py` (DAG schema v2), `testing/scaffold_compiler.py`
  (offline plan authoring), `testing/execution_compiled.py` (executor: react/thin leaves,
  price-aware voting, now also the ConSol pilot hook in `_vote_extract`).
- **Reasoning-guidance artifacts:** `reasoning_exemplars/{chain,mixed,parallel}.md` (disproven
  narrative), `reasoning_rules/branch_eliminate.md` (rule checklist, only shape covered so far).
- **Tasks:** `idea_tests/test_*.py`, 38+ tasks. Only 6 (`051/055/061/065/092/095`) are genuine
  chain/mixed shapes — the rest are parallel fan-out/breadth. Suite shape-imbalance is still open
  (plan Priority 6).
- **Benchmark harness:** `idea_test_runner.py`, `testing/{runner,execution,execution_compiled,
  execution_sequential,config,consol_pilot}.py`. Cost tracking, USD ceiling enforcement, fixture
  record/replay, statistical reporting (`scripts/{level_ladder,recovery_curve,gate_report}.py`,
  DAG visualizer).
- **Completions API:** `completions_api.py` (in-process shim) + `gateway/app/openai_router.py`
  (queue-backed) — OpenAI-compatible `/v1/chat/completions` drop-in.
- **Dev agents (`.claude/agents/*.md`):** `task-author`, `benchmark` (singleton, live $ + analysis),
  `strategy-tuner` (A/B loop discipline), `engine-dev`, `reviewer` (pre-commit gate).

## How to run (live = real $)

```bash
# keys.env is CRLF — strip \r; chroma must be on :8001; PYTHONPATH needs BOTH roots
export PYTHONPATH=services:services/agent
export IDEA_TEST_CONCURRENCY=1   # MANDATORY (shared connectors)
./.venv/bin/python -m agent.app.idea_test_runner
```
Key knobs (superset of the old list): `IDEA_TEST_{IDS,MODELS,EXECUTION_VARIANTS,RUNS,FIXTURES
(record|replay|replay_strict),RUN_ID}`, `IDEA_TEST_COMPILED_{PLAN_SOURCE(hand|auto),LEAF_MODE
(react|thin),VOTES}`, `IDEA_TEST_GOT_REEXPAND`, `IDEA_TEST_REASONING_RULES`,
`IDEA_TEST_REASONING_EXEMPLAR` (avoid), `IDEA_TEST_USE_CONSOL`. Full recipe + OPENROUTER_API_KEY
extraction: `COST_BENCHMARK_HANDOFF.md` section 4.

Offline tests (no $): `PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest services/agent/tests/`.

## What's next

The plan file (`/home/muk/.claude/plans/a-lot-of-work-gleaming-hejlsberg.md`) has the full
research-informed roadmap. Remaining priorities, in order:
- **Priority 2 (partial):** E-valuator sequential-testing pilot — bigger lift than ConSol
  (needs a labeled calibration set; cross-model transfer is unproven per the paper itself), not
  yet attempted.
- **Priority 3:** consolidate the two duplicate control-loop implementations (Strangler Fig +
  dedicated parity suite).
- **Priority 5:** config-drift fix (generate `GoTConfig`/`idea_dag_settings.json` schema from code
  — Pydantic `model_json_schema()` or the dependency-free `dc_schema` — rather than hand-maintaining
  both; this class of bug has now been hit twice), wire-or-remove dead `got_improve_enabled`,
  diagnose the 3 pre-existing test_063 failures, guardrail against the disproven narrative-exemplar
  mechanism.
- **Priority 6:** author more genuine chain/mixed benchmark tasks once the harness itself is
  trustworthy (only 6/38+ tasks are that shape today).
- **Deferred by explicit user decision:** real local-model validation (Ollama/llama.cpp with
  `Qwen3.5 4B` or `Qwen3 7B` — concrete picks from research, both have 2026 tool-calling validation
  data) — wait until Priorities 2-3 land so it tests a trustworthy harness, not one still being
  debugged.

Budget: **~$10.92 of the $12 ceiling remains** for any further live validation.

## Where the detail lives

- `services/agent/app/COST_BENCHMARK_HANDOFF.md` — closed-out compiled-scaffold campaign, full
  round-by-round history.
- `services/agent/app/ADAPTIVE_DISTILLATION_HANDOFF.md` — Phases 1-4 of the current line, every
  live-run table, every bug found/fixed, every honest null result.
- `services/agent/app/SYSTEM_STATUS.md` — cross-cutting capabilities / open issues / structural
  debt snapshot (not chronological — read this for "what's broken right now").
- `services/agent/app/RESEARCH_NOTES.md` — external research citations (MAST failure taxonomy,
  verifier-failure/reward-hacking literature, ConSol/E-valuator papers, local-model landscape,
  Strangler Fig, schema-from-code patterns) backing the design decisions above.
- `/home/muk/.claude/plans/a-lot-of-work-gleaming-hejlsberg.md` — the active plan/roadmap.
