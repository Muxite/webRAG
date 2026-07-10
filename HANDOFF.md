# webRAG / Euglena — Session Handoff (2026-07-10, Phase 5 update)

Start here. This is the project-wide pickup doc. Deep logs live in companion docs under
`services/agent/app/`: `COST_BENCHMARK_HANDOFF.md` (the closed-out compiled-scaffold campaign),
`ADAPTIVE_DISTILLATION_HANDOFF.md` (the active research line, Phases 1-5), and `SYSTEM_STATUS.md`
(a cross-cutting capabilities/issues/debt snapshot). External research citations backing several
design decisions live in `RESEARCH_NOTES.md`. The active plan/roadmap is
`/home/muk/.claude/plans/plan-next-steps-and-functional-dusk.md` (post-consolidation roadmap;
supersedes the older `a-lot-of-work-gleaming-hejlsberg.md`, whose Priority 1 is retired and
Priority 3 has now landed).

## Git state

Prior sessions' work (compiled-scaffold campaign + Phases 1-4) was committed as `bbea37b`
("Adaptive reasoning research line (phases 1-4) + control-loop consolidation") and pushed to
`origin/compiled-scaffold-dag`. Phase 5's work (below) is uncommitted as of this doc's last edit —
check `git status` before assuming otherwise.

## Current test baseline

`PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest services/agent/tests/` →
**849 passed, 18 skipped, 0 failed.** The 3 pre-existing `test_063_strict_csv_validators_test.py`
failures that persisted across the whole prior campaign are now fixed (Phase 5 — they were a
day-one authoring bug: the validators test was written against a phantom element set that never
matched the real `ENTRIES` in the source-of-truth task file).

## The thesis (proven, campaign closed out 2026-07-08)

A cheap model executing an **expensive-model-authored DAG scaffold** recovers premium-reference
accuracy at a fraction of the dollar cost. Final validated numbers (38 tasks × 3 models × R=3,
1,026 live runs, ~$38 spend): `gpt-5-mini` + compiled scaffold **ties premium** (`gemini-3.1-pro`)
at **10% of the cost**; `gpt-4.1-nano` + compiled scaffold reaches **~93%** of reference quality at
**~1/85th the cost**. Significant on the hardest tier (95% CI-disjoint, Cohen's d up to 2.7).
Packaged results: `linkedin_package_38tests_2026-07-08/` (tracked in git as of `bbea37b`). Full
history: `COST_BENCHMARK_HANDOFF.md`.

## The active research line: adaptive engine + reliability (Phases 1-5)

Started 2026-07-09 because the compiled-scaffold suite is mostly parallel fan-out/merge — the user
wanted genuine multi-stage sequential reasoning. All phases documented in depth in
`ADAPTIVE_DISTILLATION_HANDOFF.md`:

- **Phase 1 — Opus-exemplar distillation. Disproven.** Weak models copy an exemplar's surface
  structure, not its intent; backfired on the hardest shape. Lesson: never trust R=1, confirm at R=3.
- **Phase 2 — rule mining + code-enforced gates.** Built a candidate-coverage completeness gate, a
  rule checklist, and a shape classifier. Found and fixed 4 real "implicit-as-explicit verification
  failure" bugs. Gate itself showed no proven score improvement.
- **Phase 3 — capped budget extension. Retired.** Fired exactly as designed but was functionally
  inert (its re-expansion trigger doesn't fire when the root's children are already `done`). Per
  the stop rule, no further investment — candidate-coverage mechanism stays opt-in, off by default.
- **Phase 4 — ConSol sampling pilot. Validated with caveats.** Trustworthy answer-agreement, real
  ~27% cost savings, but a genuine ~60% wall-clock regression (sequential sampling).
- **Phase 5 — control-loop consolidation + cleanup debt + E-valuator pilot + ConSol batching + task
  authoring, all this session:**
  - **Control-loop consolidation (landed):** `idea_engine.py::run()` and
    `testing/execution.py::run_test_execution()` — previously two independent reimplementations of
    the step/prune/backtrack/finalize loop (root cause of Phase 2's bug #4) — now share
    `IdeaDagEngine._run_loop()`/`.finalize()`, with an explicit `fail_soft` param and a dedicated
    parity suite (`control_loop_parity_test.py`). Committed as `bbea37b`.
  - **test_063 fixed**, **config-drift guard added** (`config_drift_test.py`, no 4th real drift
    found), **dead `got_improve_enabled`/`try_improve_node` deleted** entirely.
  - **`deepseek/deepseek-v4-flash` clean R=3 baseline**: mean 0.423 on test_095 (vs. nano's 0.163),
    ~$0.0148 spent — confirms the coverage-gate confound is gone (0-visit runs correctly score 0).
  - **E-valuator piloted** (real PyPI package `e-valuator`): machinery works, but this repo's best
    available score-sequence substitute (`validation.grep_validations`) has label leakage (it
    computes the pass/fail outcome it's meant to predict), so FAR was trivially 0 — not a
    meaningful validation of the method's real value proposition. **Not adopted**; would need a
    genuinely decorrelated per-step signal (e.g. LLM-judge confidence per GoT step) to be worth
    revisiting — an instrumentation gap, not a calibration one.
  - **ConSol batched-early-cutoff variant validated**: `IDEA_TEST_CONSOL_BATCH=2` cut ConSol's
    wall-clock overhead roughly in half (37% slower than baseline vs. sequential's 74% slower)
    without losing cost savings or answer-agreement, ~$0.041 spent. Opt-in, default stays
    sequential.
  - **2 new genuine `chain`-shape tasks authored** (test_096 aviation/Earhart, test_097
    art/Goya) after a real shape recount via `shape_classifier.py` found `chain` and
    `parallel_merge` tied as most underrepresented (4/95 each, not the stale "6 of 38+" figure).

**Total campaign spend across all 5 phases: ~$1.13 of a $12 authorized ceiling.**

## What's live vs. retired vs. opt-in (the toggles that matter)

| Mechanism | Status | Toggle |
|---|---|---|
| Compiled-scaffold execution | **Proven, production path** | `IDEA_TEST_EXECUTION_VARIANTS=graph_compiled` |
| Control-loop (native engine + benchmark harness) | **Consolidated, one implementation** | n/a — `idea_engine.py::run()` is the only loop now |
| Adaptive leaf re-expansion | Sound, opt-in, not yet a proven win | `IDEA_TEST_GOT_REEXPAND=1` / `got_reexpand_enabled` (JSON default `false`) |
| Narrative reasoning exemplars | **Disproven** — actively backfired | `IDEA_TEST_REASONING_EXEMPLAR=chain\|mixed\|parallel` — avoid, kept only for reference |
| Flat rule checklist | Built, inconclusive at R=3 | `IDEA_TEST_REASONING_RULES=branch_eliminate` (only shape with a file); auto-classifies via `shape_classifier.py` when unset |
| Candidate-coverage gate | **Retired** — do not re-enable without a new re-expansion trigger design | `got_candidate_coverage_enabled` (JSON default `false`) |
| Coverage-gate budget extension | **Retired**, same reason | `got_candidate_coverage_budget_extension` (JSON default `10`, harmless since the gate itself is off) |
| `got_improve_enabled` / self-refinement | **Deleted** (was dead code) | n/a — removed entirely this session |
| ConSol early-stop voting (sequential) | Opt-in, validated for offline/batch use only | `IDEA_TEST_USE_CONSOL=1` (not in `requirements.txt`) |
| ConSol batched early-cutoff | **Opt-in, validated** — fixes sequential's wall-clock regression | `IDEA_TEST_USE_CONSOL=1 IDEA_TEST_CONSOL_BATCH=2` |
| E-valuator sequential testing | **Piloted, not adopted** — substrate has label leakage | `testing/evaluator_pilot.py` (reference/pilot code only, not wired into the harness) |
| `deepseek/deepseek-v4-flash` | Roster model, clean R=3 baseline now on record | `testing/config.py` `BENCHMARK_ROSTER["experiment"]` |

## Architecture map (`services/agent/app/`)

- **Engine:** `idea_engine.py` (`IdeaDagEngine`) — now the single control-loop implementation
  (`_run_loop()`/`finalize()`, shared by both `run()` and the benchmark harness) — plus
  `got_operations.py`, `idea_policies/*` (typed config in `config.py`, actions in `actions.py`,
  expansion/grounding/candidate-coverage policies, `shape_classifier.py`). Settings:
  `idea_dag_settings.json`, now with an automated drift guard (`config_drift_test.py`) preventing
  the dataclass-vs-JSON silent-disagreement bug class that's bitten this project 3 times.
- **Compiled scaffold:** `testing/compiled_plan.py` (DAG schema v2), `testing/scaffold_compiler.py`
  (offline plan authoring), `testing/execution_compiled.py` (executor: react/thin leaves,
  price-aware voting, ConSol pilot hook in `_vote_extract`, now with a batched-sampling option).
- **Reasoning-guidance artifacts:** `reasoning_exemplars/{chain,mixed,parallel}.md` (disproven
  narrative), `reasoning_rules/branch_eliminate.md` (rule checklist, only shape covered so far).
- **Tasks:** `idea_tests/test_*.py`, 97 tasks (test_001-test_097). Real shape-classifier recount:
  `chain` 6 (was 4, +test_096/097), `parallel_merge` 4, `branch_eliminate` 5, unclassified
  fan-out/breadth 82. Suite is still shape-imbalanced by design (most tasks are intentionally
  breadth-shaped) — further chain/mixed authoring remains a standing option, not urgent.
- **Benchmark harness:** `idea_test_runner.py`, `testing/{runner,execution,execution_compiled,
  execution_sequential,config,consol_pilot,evaluator_pilot}.py`. Cost tracking, USD ceiling
  enforcement, fixture record/replay, statistical reporting
  (`scripts/{level_ladder,recovery_curve,gate_report}.py`), DAG visualizer.
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
Key knobs: `IDEA_TEST_{IDS,MODELS,EXECUTION_VARIANTS,RUNS,FIXTURES(record|replay|replay_strict),
RUN_ID}`, `IDEA_TEST_COMPILED_{PLAN_SOURCE(hand|auto),LEAF_MODE(react|thin),VOTES}`,
`IDEA_TEST_GOT_REEXPAND`, `IDEA_TEST_REASONING_RULES`, `IDEA_TEST_REASONING_EXEMPLAR` (avoid),
`IDEA_TEST_USE_CONSOL`, `IDEA_TEST_CONSOL_BATCH` (new, opt-in within the ConSol opt-in). Full recipe
+ `OPENROUTER_API_KEY` extraction: `COST_BENCHMARK_HANDOFF.md` section 4.

Offline tests (no $): `PYTHONPATH=services:services/agent ./.venv/bin/python -m pytest services/agent/tests/`.

Note: result JSONs land under `services/agent/idea_test_results/` — some older doc references to a
repo-root `idea_test_results/` path are stale, worth a cleanup pass if it causes confusion.

## What's next

The plan file (`/home/muk/.claude/plans/plan-next-steps-and-functional-dusk.md`) has the full
Phase-5-informed roadmap; all 6 of its numbered experiments landed this session. Remaining open
items, roughly in priority order:
- Decide whether to broaden the ConSol-batched validation matrix (more tests/models) before
  considering any default-on flip — currently opt-in-only on n=5/one-cell evidence.
- If a genuinely decorrelated per-step verifier signal is ever added to the harness (e.g. LLM-judge
  confidence per GoT step, logged into `execution.observability`), E-valuator is worth revisiting —
  not before that instrumentation exists.
- Continued chain/mixed task authoring is a standing option (2 authored this session), not an
  urgent gap — most of the suite is intentionally breadth-shaped.
- Local-model validation (Ollama/llama.cpp, Qwen3.5 4B / Qwen3 7B) — explicitly dropped this
  session by user decision, not deferred; would need a fresh ask to resume.
- Guardrail against the disproven narrative-exemplar mechanism (doc comment / runtime warning) —
  not yet done, low priority.

Budget: **~$10.87 of the $12 ceiling remains** for any further live validation.

## Where the detail lives

- `services/agent/app/COST_BENCHMARK_HANDOFF.md` — closed-out compiled-scaffold campaign, full
  round-by-round history.
- `services/agent/app/ADAPTIVE_DISTILLATION_HANDOFF.md` — Phases 1-5 of the current line, every
  live-run table, every bug found/fixed, every honest null result.
- `services/agent/app/SYSTEM_STATUS.md` — cross-cutting capabilities / open issues / structural
  debt snapshot (not chronological — read this for "what's broken right now").
- `services/agent/app/RESEARCH_NOTES.md` — external research citations (MAST failure taxonomy,
  verifier-failure/reward-hacking literature, ConSol/E-valuator papers with Phase-5 primary-source
  follow-up, local-model landscape, Strangler Fig, schema-from-code patterns) backing the design
  decisions above.
- `/home/muk/.claude/plans/plan-next-steps-and-functional-dusk.md` — the active plan/roadmap.
