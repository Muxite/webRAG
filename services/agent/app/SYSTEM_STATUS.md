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
- 97 curated benchmark tasks with function-based validators and live-verified ground truth. Real
  `shape_classifier.py` recount (2026-07-10, not a name-based guess): `chain` 6 (040/051/065/092,
  +096/097 authored Phase 5), `parallel_merge` 4 (055/061/085/086), `branch_eliminate` 5
  (068/076/081/094/095), the remaining 82 unclassified fan-out/breadth (intentional — most tasks
  are meant to be breadth-shaped, not a gap).

**Other shipped capability:**
- OpenAI-compatible `/v1/chat/completions` + `/v1/models` (gateway, queue-backed + auth; and an
  in-process shim) — any OpenAI SDK client can point at the engine as a drop-in.
- An offline, $0 distillation pipeline: Claude Code's own Opus (never an API call) solves a task
  shape live and distills a reasoning artifact — used to produce both the narrative exemplars and
  informed the rule checklist's content.

---

## Open issues (things that are actually broken or unresolved)

1. **RESOLVED (2026-07-10, Phase 5): the 3 pre-existing `test_063_strict_csv_validators_test.py`
   failures were a day-one authoring bug** — the validators test was written against a phantom
   element set (comments referenced "Thulium"/"Praseodymium"/"Terbium"/"Holmium", names absent from
   the source-of-truth `ENTRIES`), not real drift over time (both files landed in the same commit).
   Fixed by grounding all three assertions in the real `ENTRIES` values. Offline suite now green.
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
4. **RESOLVED (2026-07-10, Phase 5): `deepseek/deepseek-v4-flash` now has clean R=3 data.** Ran
   test_095, native `graph` variant, coverage gate off: scores 1.00/0.00/0.27 (mean 0.423), ~$0.0148
   spent. No sign of the old coverage-gate confound — the 0-visit run correctly scored 0 straight
   down the line. Qualitatively higher than nano's Phase 2 R=3 baseline (mean 0.163); n=3,
   data-gathering only, no strong conclusion drawn.
5. **Chain and parallel-merge shapes have no rule-checklist files.** The classifier correctly
   identifies them but auto-injection is a silent no-op (logged, not crashed) — only
   `branch_eliminate.md` exists. Not worth authoring until the branch_eliminate mechanism itself
   is validated to actually help.
6. **`got_improve_enabled` (self-refinement) was dead code — now removed** (2026-07-10). The
   `try_improve_node` mechanism, its `got_improve_*` settings, and the `nodes_improved` stat were
   deleted; it was never wired into the engine's main loop.

---

## What isn't ideal (works, but is structurally risky or ugly)

1. **RESOLVED (2026-07-10, Phase 5): control-loop consolidated.** `idea_engine.py`'s `run()` and
   `testing/execution.py`'s `run_test_execution()` previously reimplemented the step/prune/
   backtrack/finalize loop independently — the root cause of Phase 2's bug #4. Now unified via
   Strangler Fig: shared `IdeaDagEngine._run_loop()`/`.finalize()`, an explicit `fail_soft`
   parameter replacing the silent fail-fast/fail-soft divergence, and a dedicated parity suite
   (`control_loop_parity_test.py`). Committed as `bbea37b`.
2. **RESOLVED (2026-07-10, Phase 5): config-drift guard added.** `config_drift_test.py` asserts
   every `*Config` dataclass group in `idea_policies/config.py` can't silently disagree with
   `idea_dag_settings.json`'s shipped values — the bug class that had bitten this project 3 times.
   No 4th real drift found on first run (only benign `None`/`""` sentinel non-differences).
3. **Brute-force R=3 remains the primary statistical-confidence tool.** ConSol was adopted (opt-in,
   validated including a batched variant that fixes its wall-clock regression — see
   `ADAPTIVE_DISTILLATION_HANDOFF.md` Phase 4/5). E-valuator was piloted (Phase 5) but **not
   adopted** — this repo's best available score-sequence substitute (`validation.grep_validations`)
   has label leakage (it computes the very outcome it's meant to predict), so the pilot couldn't
   demonstrate real value; would need a genuinely decorrelated per-step verifier signal to be worth
   revisiting.
4. **The "local LLM" framing is simulated, not validated** — and, per an explicit 2026-07-10 user
   decision, **this is no longer a planned gap to close**: real local-model validation
   (Ollama/llama.cpp) was dropped from the roadmap entirely, not deferred. All testing continues to
   use OpenRouter cheap-tier models (nano, deepseek-flash, flash-lite) as the permanent proxy.
5. **The benchmark suite's shape mix is real but was previously mis-described.** A proper
   `shape_classifier.py` recount (Phase 5) found `chain`/`parallel_merge` tied at 4/95 each before
   this session's task authoring (now 6/95 `chain` after 2 new tasks), `branch_eliminate` at 5/95,
   and 82/95 intentionally unclassified fan-out/breadth — not the stale "6 of 38+" figure. Most of
   the suite is meant to be breadth-shaped; this is not automatically a gap to close further.
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
