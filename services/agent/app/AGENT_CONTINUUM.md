# The Agent Continuum — Design Doc

Written 2026-08-03, Stage 1 of the multicapability adaptive agent effort (see the Stage 1 plan,
`goal-is-to-continue-fancy-duckling.md`). This is the document `model_tiers.py`'s and
`idea_policies/config.py`'s own comments already cited by name before it existed — the durable
reference for why the capability-tiering work looks the way it does, and where it's headed.

## The central thesis

Every mechanism in this codebase — graph-of-thoughts structure, node contracts, plan-library
retrieval, composers, capability tiering, voting, confidence judging, backtrack — serves one goal:
**improve the performance of cheaper models through structure and memory, making them viable for
complex, long-running, agentic applications where accuracy matters and hallucination can compound
over many steps into a devastating degree of error.** A small per-step error rate is tolerable in a
single-shot answer; across a long agentic run it compounds unless something actively contains it.
That containment — not cleverness for its own sake — is the test for whether a new mechanism
belongs here.

This is also why "long-running, agentic, accuracy-critical" is the target, not just single-shot QA:
the whole value proposition of trading structure for capability only pays off when a task is long
or complex enough that an unstructured cheap model would otherwise drift.

## The continuum, not a hard cutoff

`badmodel-lab` (weak/local model testing, hand-authored mitigations) and the main engine
(`services/agent/app`, built for strong hosted models) sit on one continuum, not two walled-off
systems. Research this session confirmed the execution layer is already substantially shared: the
same task files (`idea_tests/*.py`), the same runner (`idea_test_runner.py`), the same result-JSON
schema, and the same generic OpenAI-compatible provider routing serve both a local Ollama model and
a frontier hosted one today, with zero code changes required to point either harness at either kind
of model.

The real, deliberate axis that stays different across the continuum is **library size and
generality** — badmodel-lab curates a bigger library of hand-authored, near-task-specific
composers/templates because it targets weak models on more constrained, known problem shapes; main
stays on a small, general library because strong models can generalize themselves. Mechanisms
(schema shapes, classifiers, lint, dispatch patterns) are shared freely across the continuum. This
does not mean every mechanism transfers to every domain — see "What doesn't transfer" below.

## Three tiering axes — kept explicitly distinct

Three genuinely different things in this codebase are called "tiers," and none subsumes another:

1. **Model-capability tier** (`model_tiers.py::capability_tier()`) — price-derived,
   `weak | standard | strong`, computed once per run from the model's name/pricing. This is the
   axis that gates *how much mitigation a run gets*.
2. **Task-difficulty curation** (`badmodel-lab/tiers.yaml`: `sanity/micro/reachable/format/hard`) —
   a human researcher's grouping of the fixed `idea_tests` suite by what capability gap it isolates,
   used to pick which test ids to run in a given experiment cell.
3. **Structural-complexity tier** (the `idea_tests/test_NNN_tier[1-5]_*.py` filename convention) —
   how structurally demanding a single task's reasoning shape is (single-fact lookup through
   multi-branch/eliminate-chain/argmax reasoning).

A model-capability-weak run can face a structural-complexity-tier-5 task; a task-difficulty "hard"
cell can be run against a strong model as an anchor. Conflating these axes in code or in
conversation is a recurring source of confusion — keep them named separately.

## The static-classification principle

`capability_tier()` is computed **once, before a run starts, and never re-evaluated mid-run.** This
is deliberate, not an oversight: an agent that reclassifies model capability *during* a run makes
token usage, latency, and behavior non-deterministic across repeated runs of the identical task,
which breaks benchmarking and makes regressions unfalsifiable. Every tier-gated lever in this
codebase (`native_vote_k_tiered_enabled`, `reexpand_max_iterations_tiered_enabled`, and future ones)
must read the tier once and hold it for the run's duration.

**"Adaptive" in this system means something else, and the two must not be conflated.** The graph
engine's own plan is allowed — and expected — to adapt to what it discovers mid-run: re-expansion
(a leaf that turns out under-evidenced grows new children), backtrack (a low-scoring branch is
abandoned), confidence-driven early-exit. These are pre-existing, already-named mechanisms that
change the *shape of the graph*, not a re-assessment of *which model is running it*. A run's
capability-tier classification and its graph's shape are two independent kinds of change; only the
second is allowed to happen live.

## The architecture decision: graph-as-primary, tools as an open registry

The graph-of-thoughts engine (`IdeaDagEngine`, node contracts, subproblem decomposition via
expansion/merge/branch-pair) is the **primary** reasoning structure — not one style among several
peer engines behind a router. Web search/visit are not a special case; they are the first tools
registered against a general, string-keyed tool registry (`LeafActionRegistry`), and other
capabilities — file/shell operations, memory, code-sandbox execution — generalize the same way:
register a `LeafAction` subclass by name, gate it behind `allowed_actions`, done. No enum member is
required per tool (Stage 1 fixed the bug that made this untrue in practice — see below).

The many existing weak-model mitigations — composers (`execution_compiled.py`'s deterministic
answer composition), typed-slot response parsing (an alternative to free JSON, more robust for weak
models per `badmodel-lab/localagent/ir.py`'s design), self-consistency voting, thin-leaf mode,
plan-library retrieval, per-step confidence judging, backtrack — are **helpers**: tier-gated
augmentations layered onto the one engine, applied at full strength for weak/local models and at
near-zero overhead for strong models (gemini flash, sonnet-tier), never forced blanket-wide. A
strong model does not pay a JSON-hygiene tax it doesn't need; a weak model gets the full suite. This
is the concrete expression of "some of our features might help badmodels a lot, but don't help
goodmodels or waste tokens without gain for some models."

**What doesn't transfer, named honestly.** Composers and plan-library retrieval are shaped around a
compiled DAG's leaf-plan structure (a fixed set of independently-resolvable facts to recombine).
They do not transfer to a genuinely open-ended, reactive task (e.g. "find this file, count its
lines, summarize it") without first inventing a DAG-shaped intermediate the task doesn't naturally
have — for tasks that small, the graph mostly degenerates to a single node, and the DAG's real
contribution there is the shared harness/telemetry/tiering plumbing, not branching. That's an
accepted tradeoff, not a hidden one.

## Stage 1: the one foundational fix

`IdeaDagEngine._execute_action` used to coerce a node's action-type string through
`IdeaActionType(str(action_type))` before consulting the registry — this raised for any name that
was never added to the enum, silently defaulting to `THINK`, which meant `LeafActionRegistry`'s own
advertised `register()`/`install_pack()` extension point could never make a new action reachable via
the model's own selection. `idea_policies/extra_actions/pack.py`'s 11 bundled actions
(Wikipedia/arXiv/GitHub/regex/unit-convert/etc.) were dead on arrival for exactly this reason — its
own docstring said as much.

Fixed by resolving through the registry by string name (`self.actions.get(action_name)`, which
already raises for an unknown name) instead of the enum cast, preserving byte-identical behavior for
every pre-existing action and the "not allowed" / "allowed but unregistered" fallback-to-THINK paths.
Verified: `services/agent/tests/action_registry_open_dispatch_test.py` proves a registered non-enum
action is genuinely dispatched end-to-end, including the full `ExtraActionPack` bundle via
`install_pack`. Full offline suite stays at parity (3278 passed vs. the pre-fix 3273, +5 for the new
tests, same one pre-existing unrelated failure — see "Known gaps" below).

## Staged roadmap beyond Stage 1

Named here so later stages aren't invented from scratch each time, not committed to a schedule:

1. **Populate the tool registry** with new domains: file/shell/memory actions (ported from
   `badmodel-lab/localagent/tools/`) and code/sandbox execution (from
   `testing/execution_compiled_code.py`/`SandboxConnector`, the "dockerized home for code work"),
   each as `LeafAction` subclasses registered and tier-gated the same way.
2. **Wire tier-gated response parsing** — JSON action objects (strong tier, current default) vs.
   typed-slot IR (`badmodel-lab/localagent/ir.py`'s router→slot-fill→validate→typed-repair pattern,
   weak tier) — as a helper inside leaf execution, gated by `capability_tier()`. Only after E3
   (below) validates it's worth building, isolated from `localagent`'s other control-flow
   differences.
3. **Retire `badmodel-lab/localagent/loop.py`'s control flow** once the graph engine matches or
   beats it on localagent's own task suite (file/shell/memory/web-read) — not before. Its typed-slot
   IR primitives (`ir.py`, `actions.py`, `state.py`) are small and reusable regardless of whether the
   outer loop itself survives.
4. **Unify benchmark reporting** — `badmodel-lab/analyze.py` bridged onto `scripts/bench_common.py`
   (this session's Stage 1 also does the first slice of this — see the plan's item 4).
5. **Full orthogonal ExecutionStyle × ActionVocabulary × ResponseParsingStrategy protocol system**
   — deliberately NOT built now. Two of three independent architecture proposals this session
   flagged it as premature: today, 2 of those 3 axes have exactly one real implementation each, and
   a protocol built for a cross-product that doesn't yet exist is speculative generality. Revisit
   only if real usage after steps 1-3 shows genuine demand for combinations the registry-based
   approach can't express.

## Experiments — validate assumptions, don't assert them

- **E1 — local-good-model vs. cheap-API-model head-to-head.** No apples-to-apples data exists on
  disk today (confirmed this session: `badmodel-lab/results/cells_long.csv` has only local-subject
  rows, no API-anchor rows). Uses the existing "subjects vs. anchors" split already in
  `badmodel-lab/roster.yaml` (qwen2.5:7b local vs. gpt-4.1-nano API) across the same existing
  `idea_tests` tiers/profiles. Needs live $ authorization; run early since the rest of the tiering
  work leans on assumptions about how local models actually compare.
- **E2 — tool-registry fix parity check.** Done this session: full offline suite green at parity,
  plus the new dispatch tests. A live smoke-cell byte-for-byte reproduction of a previously-recorded
  benchmark result is the remaining live-verification step before anything is layered on top of the
  registry fix in production settings.
- **E3 — typed-slot-IR-as-a-helper validation.** Deferred to the stage that would build it (roadmap
  item 2). Isolate the JSON-vs-typed-slot-parsing variable *inside the graph engine itself*, not
  confounded with `localagent`'s entirely different control flow (anti-repeat suppression,
  finish-gate, greedy budget loop) — does typed-slot parsing measurably help weak models over JSON
  once everything else is held constant? `localagent`'s own results don't answer this in isolation.

## Known gaps (named, not silently carried forward)

- `compiled_plan.py::COMPOSITION_OPS` lists `ratio_argmax` as a supported composition op, but
  `execution_compiled.py::_COMPOSERS` has no handler for it (only `and_filter`, `argmax`,
  `count_threshold`, `subset_sum` are actually implemented) — a pre-existing drift bug, found via
  this session's baseline test run (`compiled_plan_test.py::test_composition_ops_matches_execution_compiled_composers`
  fails on current `master`). Orthogonal to this effort; not fixed here.
- `prompt_hygiene.py` (JSON self-contradiction lint) remains imported only by the compiled test
  harness — genuinely dead code on the live engine path. Wiring it into live prompt construction is
  unscoped, not part of Stage 1.
- `capability_tier()` collapses every unpriced (local) model into a single `weak` bucket regardless
  of actual size (0.5B vs. 70B) — left as-is per an explicit YAGNI decision this session; revisit
  only if E1's data shows this is costing real accuracy on a capable local model.
- `SYSTEM_STATUS.md`'s prior "local-model validation dropped from the roadmap entirely" claim
  (2026-07-10) is superseded by this document — that decision no longer holds.
