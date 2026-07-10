# System Status — Capabilities, Issues, Debt (2026-07-10)

A cross-cutting snapshot of the webRAG/Euglena agent, synthesized from the compiled-scaffold
campaign (`COST_BENCHMARK_HANDOFF.md`), the adaptive-engine/distillation track
(`ADAPTIVE_DISTILLATION_HANDOFF.md`), and this session's bug-hunting. Organized by what works,
what's broken, and what's merely ugly — not chronological.

---

## What the system can actually do today

**Compiled-scaffold execution (proven, statistically validated):**
- An expensive model authors a DAG execution plan once, offline; a cheap model executes it. On a
  38-task live benchmark (1,026 runs, ~$38, 3 models): `gpt-5-mini` ties premium quality
  (`gemini-3.1-pro`) at 10% of the cost; `gpt-4.1-nano` reaches ~93% at ~1/85th the cost.
  Significant on the hardest tier (95% CI-disjoint, Cohen's d up to 2.7).
- The DAG schema genuinely supports multi-hop chains and mixed branch+chain shapes
  (`depends_on` + value substitution, not just fan-out/merge) — `compiled_plan.py`. Most live
  runs use hand-authored plans (`IDEA_TEST_COMPILED_PLAN_SOURCE=hand`); the auto-compiler
  (`scaffold_compiler.py`) exists and is prompted for chains too, but its real-world tendency to
  produce genuine depth vs. degenerate to fan-out is **not empirically validated**.

**Native Graph-of-Thoughts engine:**
- Standard expand → evaluate → select → merge loop with dedup, dynamic beam width, pruning,
  auto-parallel siblings, and a grounding gate.
- **Adaptive leaf re-expansion** (`got_reexpand_enabled`, opt-in): after a leaf completes, an
  independent LLM check asks "does this reveal a genuine follow-up?" and grows new children live
  if so — the actual "explored A,B,C, now need D" mechanism. Audited sound, speed-gated (adds
  ~20s/+$0.01 when it fires, never approached a timeout).
- **Candidate-coverage completeness gate** (`got_candidate_coverage_enabled`, opt-in): a
  deterministic (non-LLM) check that extracts named candidates from a task's mandate and blocks
  finalization until each has a visit-backed resolved result. Correctly wired after 3 rounds of
  bug-fixing this session; does **not yet show a proven score benefit** (see Issues).
- **Deterministic shape classifier** (`shape_classifier.py`): regex/keyword classification of a
  mandate into `branch_eliminate`/`chain`/`parallel_merge`/`None`, 7/7 validated accuracy, fails
  open on unrecognized shapes.
- **Rule/exemplar injection into the expansion prompt** — two independent, both opt-in mechanisms:
  a narrative "Situation/Thought/Action" exemplar (`reasoning_exemplars/`, from Phase 1 — shown to
  backfire on the hardest shape) and a flat imperative rule checklist (`reasoning_rules/`, Phase 2,
  currently branch_eliminate-only).

**Benchmarking infrastructure:**
- Real USD cost per call, a hard driver-enforced spend ceiling, resumable staged batch runs.
- Model roster spanning reference (`gemini-3.1-pro`), cheap (`gemini-2.5-flash`, `gpt-5-mini`,
  `gpt-4.1-nano`), and experiment/weak tiers (`gemini-2.5-flash-lite`, `gpt-5-nano`,
  `deepseek-v4-flash` — added this session, cheaper than nano).
- Fixture record/replay/replay-strict for evidence-parity reproducibility.
- Baselines: `parametric` (no tools), `naive_rag` (one search+visit round), `sequential_react`
  (ReAct scratchpad loop) — all comparators against the graph/compiled variants.
- Statistical reporting: `level_ladder.py` (mean±CI95, Cohen's d, CI-disjoint verdicts),
  `recovery_curve.py` (square 4K Pareto cost/quality plots), `gate_report.py`, a DAG visualizer
  (compiled-plan and runtime-graph PNGs).
- 38+ curated benchmark tasks with function-based validators and live-verified ground truth; a
  handful (051/055/061/065/092/095) are genuine multi-hop chain/mixed shapes, the rest are
  parallel fan-out/breadth.

**Other shipped capability:**
- OpenAI-compatible `/v1/chat/completions` + `/v1/models` (gateway, queue-backed + auth; and an
  in-process shim) — any OpenAI SDK client can point at the engine as a drop-in.
- An offline, $0 distillation pipeline: Claude Code's own Opus (never an API call) solves a task
  shape live and distills a reasoning artifact — used to produce both the narrative exemplars and
  informed the rule checklist's content.

---

## Open issues (things that are actually broken or unresolved)

1. **3 pre-existing test failures**, unrelated to any recent work and never diagnosed:
   `test_063_strict_csv_validators_test.py::{test_hallucinated_keystone_value_zero,
   test_partial_coverage_scores_fraction, test_compiled_plan_is_pure_fanout_and_leaks_nothing}`.
   Present at every baseline check this session (734→742→763→766→769→770→785 passed, always the
   same 3 failing) — stable but genuinely unfixed technical debt, not flaky.
2. **RETIRED: the candidate-coverage gate does not improve outcomes**, confirmed at R=3 twice —
   once pre-fix (null) and again after adding a fixed +10-step budget extension (still null, one
   condition's mean regressed below baseline). Root cause of the extension's failure: it correctly
   granted extra steps but its "re-activate root" trigger doesn't cause real re-expansion when the
   root's children are already all `done` — the lever was pulled but connected to nothing. Per the
   plan's stop rule (4 bug-fix rounds + 1 design change without success), no further investment
   planned. Mechanism stays in the codebase, opt-in, `false` by default — see
   `ADAPTIVE_DISTILLATION_HANDOFF.md` Phase 3 for the full writeup and the specific dead-end to
   avoid rediscovering (forced re-expansion needs a different trigger than child-status
   reactivation).
3. ~~The gate's own diagnostic annotation was unvalidated live~~ — confirmed working in the Phase 3
   re-validation run (`candidate_coverage_incomplete` correctly appeared in all 6 runs). No longer
   an open issue, though moot now that the gate itself is retired.
4. **`deepseek/deepseek-v4-flash` has no clean R=3 data** — only tested at R=1/R=2, both under the
   buggy (inert or partially-inert) gate. Its R=2 numbers showed a different failure mode (search
   snippets satisfying coverage without page visits, worse under the original bug) that hasn't
   been re-tested since the fix.
5. **Chain and parallel-merge shapes have no rule-checklist files.** The classifier correctly
   identifies them but auto-injection is a silent no-op (logged, not crashed) — only
   `branch_eliminate.md` exists. Not worth authoring until the branch_eliminate mechanism itself
   is validated to actually help.
6. **`got_improve_enabled` (self-refinement) is dead code** — defined, configurable, but never
   wired into the engine's main loop (confirmed by direct audit this session). Either wire it or
   remove it; currently it's neither.

---

## What isn't ideal (works, but is structurally risky or ugly)

1. **Two parallel loop implementations for the same engine logic.** `idea_engine.py`'s `run()` and
   `testing/execution.py`'s `run_test_execution()` both reimplement the step loop, both call
   `_grounding_replan`, both build the final payload — independently. This is what caused Issue
   #3 (a fix landed in one copy, not the other) and is a standing risk for any future engine
   change: **fixing something in `idea_engine.py` does not mean the benchmark harness sees it.**
   The real fix is consolidating to one loop implementation, not another one-off port.
2. **Settings sprawl with under-tested interactions.** `idea_dag_settings.json` has ~10
   independently-toggleable `got_*` booleans (reexpand, candidate coverage, improve, backtrack,
   dynamic beam, prune, dedup, telemetry routing...) — most are tested individually or in the one
   pairwise combination a given experiment needed, not exhaustively against each other. Historical
   precedent for silent drift: `GoTConfig` dataclass defaults have now disagreed with the shipped
   JSON twice (3 fields fixed this campaign) — there's no automated test asserting the dataclass
   and JSON stay in sync, so a third drift is only a matter of time.
3. **Brute-force R=3 is the only statistical-confidence tool in use.** Expensive (each promising
   R=1 result this campaign needed a full R=3 re-run to check), and noisy on high-variance small
   models. Sequential-testing alternatives (ConSol, E-valuator — see `RESEARCH_NOTES.md`) were
   identified as directly applicable but not adopted.
4. **The "local LLM" framing is simulated, not validated.** All testing uses OpenRouter cheap-tier
   models (nano, deepseek-flash, flash-lite) as stand-ins for genuinely local/self-hosted small
   models. No actual local-inference benchmark data exists in this repo, and no public literature
   was found distinguishing these models' multi-hop tool-use performance from their general
   reasoning/math scores — the gap is real and currently unaddressed by anything but our own harness.
5. **The benchmark suite is still shape-imbalanced.** The original motivating complaint (mostly
   parallel fan-out, few genuine sequential/multi-stage tasks) is still true — only 6 of 38+ tasks
   are chain/mixed shapes, and new task authoring was explicitly deferred this campaign.
6. **`validate_branch_exploration`'s visit fix is coarse.** It caps credited breadth at the raw
   visit *count*, not per-candidate visit identity, because `observability["visit"]` only exposes
   a count, not which URLs were visited. A model could visit 3 unrelated pages and 0 real
   candidates and still get partial credit under the current fix — better than the original bug
   (text-only, no visit requirement at all) but not a precise fix.
7. **Two independently-shipped "help the cheap model" mechanisms coexist with different verdicts**
   (narrative exemplars — shown to backfire — vs. rule checklists — inconclusive) and both remain
   togglable in the codebase. Nothing prevents a future user from reaching for the disproven
   narrative-exemplar mechanism without reading the handoff docs first.

---

See `ADAPTIVE_DISTILLATION_HANDOFF.md` for the full experimental history and `RESEARCH_NOTES.md`
for external literature backing the diagnoses above.
