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
(`agent/app`, built for strong hosted models) sit on one continuum, not two walled-off
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
Verified: `agent/tests/action_registry_open_dispatch_test.py` proves a registered non-enum
action is genuinely dispatched end-to-end, including the full `ExtraActionPack` bundle via
`install_pack`. Full offline suite stays at parity (3278 passed vs. the pre-fix 3273, +5 for the new
tests, same one pre-existing unrelated failure — see "Known gaps" below).

## Staged roadmap beyond Stage 1

Named here so later stages aren't invented from scratch each time, not committed to a schedule:

1. **Populate the tool registry — DONE 2026-08-06 for file/shell; memory/web deliberately skipped.**
   `localagent/tools/{files,shell}.py` ported as `SandboxToolPack`
   (`idea_policies/extra_actions/sandbox_tools.py`), delegating to the existing
   `SandboxConnector` (codebench's already-security-reviewed confinement) rather than
   reimplementing it — opt-in only, gated by a run's `AgentIO.connector_sandbox` actually being
   set, never installed by default. `move` was NOT ported (`SandboxConnector` has no move
   primitive; adding one was out of scope). `localagent/tools/{web,memory}.py` were confirmed
   redundant, not ported: `WebReadTool` is strictly less than the existing `search`/`visit`
   actions, and `memory.py`'s own docstring says its P1 backend already IS
   `idea_memory.MemoryManager` — porting either would duplicate an already-shared mechanism. One
   real gap flagged, not fixed: `localagent` keys memory to a durable identity, while
   `IdeaDagEngine._memo_namespace()` is per-mandate and ephemeral — a namespace-policy question,
   not a missing tool. **A second, deeper prerequisite gap was found and fixed in the same
   session**: a registered-and-allowed action was still practically unselectable, because the
   expansion prompt only ever listed non-core action NAMES with zero description while its
   `SEARCH + VISIT PIPELINE` section actively steered every task toward search/visit — dispatch
   reachability isn't the same as prompt reachability. Fixed via `LeafAction.menu_line()` /
   `LeafActionRegistry.menu_lines()` + a `{extra_actions_menu}` prompt placeholder (empty and
   therefore byte-identical when no extra action is allowed) and `engine.install_action_pack()`
   (keeps registry + dispatch gate + prompt menu in sync — a settings-snapshot staleness bug
   would otherwise leave the prompt not describing an action the gate had just started allowing).
   Live-verified (3 mandates, gpt-4.1-nano): the model correctly picked `datetime_now` and
   `hacker_news_top` when they were the clear best fit, and reasonably preferred the existing
   `visit` action over the new `wikipedia_summary` for an encyclopedia-summary mandate — the menu
   makes an action selectable, it does not force it, which is the correct behavior.
2. **Wire tier-gated response parsing** — JSON action objects (strong tier, current default) vs.
   typed-slot IR (`badmodel-lab/localagent/ir.py`'s router→slot-fill→validate→typed-repair pattern,
   weak tier) — as a helper inside leaf execution, gated by `capability_tier()`. Only after E3
   (below) validates it's worth building, isolated from `localagent`'s other control-flow
   differences.
3. **Retire `badmodel-lab/localagent/loop.py`'s control flow** once the graph engine matches or
   beats it on localagent's own task suite (file/shell/memory/web-read) — not before. Its typed-slot
   IR primitives (`ir.py`, `actions.py`, `state.py`) are small and reusable regardless of whether the
   outer loop itself survives.
4. **Unify benchmark reporting — in progress, not closed.** `badmodel-lab/analyze.py` now bridges
   `model`/`score`/`usd`/`visits` (Stage 1) plus `grounding_pass` (prefers the authoritative
   `obs.grounding.grounded` flag, same precedence `bench_common.load_row()` uses, before falling
   back to its own grep-based proxy) and `latency_s` (2026-08-06: `canon.get("secs")` — a real
   fix, not just deduplication; `analyze.py`'s own `latency_of()` reads an `observability.timings`
   shape that live result JSONs never actually carry, confirmed empty across a 60-file sample, so
   `latency_s` had been silently `None` in every row until this bridge). `completion_tokens`
   deliberately stays UNbridged — it and `bench_common`'s `total_tokens` measure different things
   (completion-only vs. prompt+completion), the same kind of false-cousin `tier` already warns
   against. Remaining candidates, not yet done: `place`/`profile`/`leaf_mode`/`tier`/`test_id` are
   genuinely badmodel-lab-specific (no `bench_common` equivalent) and may never bridge.
5. **Full orthogonal ExecutionStyle × ActionVocabulary × ResponseParsingStrategy protocol system**
   — deliberately NOT built now. Two of three independent architecture proposals this session
   flagged it as premature: today, 2 of those 3 axes have exactly one real implementation each, and
   a protocol built for a cross-product that doesn't yet exist is speculative generality. Revisit
   only if real usage after steps 1-3 shows genuine demand for combinations the registry-based
   approach can't express.

## Experiments — validate assumptions, don't assert them

- **E1 — local-good-model vs. cheap-API-model head-to-head. DONE 2026-08-03** ($0.1245 of a $2
  budget). `qwen2.5:7b` (local) vs. `gpt-4.1-nano` (API) on the reachable tier (7 tasks), `m1_thin`
  profile, R=3. Mean score 0.85 vs. 0.97; honest keystone pass 18/21 vs. 21/21. **Exact parity on
  5 of 7 tasks** (062, 070, 072, 076, 078 — argmax/subset-sum/count-with-condition shapes); a
  moderate gap on 064 (0.85 vs 1.00, qwen under-reports coverage, doesn't miscompute); a severe gap
  on 069 (0.33 vs 1.00) that is a genuine reasoning failure, not noise — qwen's negation/odd-one-out
  handling is internally self-contradictory across self-consistency samples (one run labels every
  candidate "NOT landlocked"; another emits the literally contradictory
  `"Austria: NOT landlocked -- coastline on none (landlocked)"`), and the vote-extract composer's
  fallback produces garbage when 2 of 3 samples come back UNKNOWN. **Verdict on the coarse
  "every local model = one weak bucket" question**: not obviously wrong (qwen still needs the full
  mitigation stack — without it, 069-style catastrophic failure is real), but genuinely coarse for a
  model this size — a 7B model tying a paid API model on 5 of 7 tasks is a different animal from the
  0.5B-3B subjects `tiers.yaml` documents as flooring at ~0 on this same suite. The one real gap is
  task-shape-specific (negation reasoning), not a uniform capability discount — argues that IF
  capability-tier is ever used to modulate mitigation *strength* rather than just on/off, a
  size-aware or task-shape-aware refinement would out-perform the current flat split. Left as an
  open call for whoever next touches `model_tiers.py`, not forced by this one run.
  **Side finding**: this run surfaced and fixed a real, previously-latent bug in
  `badmodel-lab/run_cell.sh` — its key-sourcing loop never stripped surrounding double-quotes from
  `services/keys.env` values, so `OPENROUTER_API_KEY` was exported literally including the quote
  characters and every OpenRouter call 401'd. This is almost certainly why zero `gpt-4.1-nano` (or
  any OpenRouter-routed) rows existed anywhere in `badmodel-lab/results/` before this session — the
  remote-anchor path had apparently never been successfully exercised through this script. Fixed by
  applying the same dequoting step (`sed -E 's/^"(.*)"$/\1/'`) already used consistently across
  ~15 other key-sourcing scripts in `scripts/`.
- **E2 — tool-registry fix parity check.** Done this session: full offline suite green at parity,
  plus the new dispatch tests. A live smoke-cell byte-for-byte reproduction of a previously-recorded
  benchmark result is the remaining live-verification step before anything is layered on top of the
  registry fix in production settings.
- **E3 — typed-slot-IR-as-a-helper validation.** Deferred to the stage that would build it (roadmap
  item 2). Isolate the JSON-vs-typed-slot-parsing variable *inside the graph engine itself*, not
  confounded with `localagent`'s entirely different control flow (anti-repeat suppression,
  finish-gate, greedy budget loop) — does typed-slot parsing measurably help weak models over JSON
  once everything else is held constant? `localagent`'s own results don't answer this in isolation.
- **E4 — broader model comparison for an approximate tier list. DONE 2026-08-04** ($0.3343 of a $3
  budget, $0.4588 combined with E1). Extended E1's single model pair to `deepseek/deepseek-v4-flash`
  and `google/gemini-2.5-flash-lite` across format/reachable/hard/micro tiers, plus a `qwen2.5:7b`
  hard-tier cell and `gpt-4.1-nano`'s first format-tier run. Full results, per-model
  strengths/weaknesses, and important caveats (deepseek's low scores are likely a token-budget
  measurement artifact, not a clean capability read — see below):
  `badmodel-lab/MODEL_TIER_LIST.md`.
- **E5 — fix the deepseek token-budget bug + expand local roster to 8. DONE 2026-08-04**
  ($0.1496 of a $3 budget; $0.7084 total across E1/E4/E5). Fixed `_is_reasoning_model` (commit
  `d17de329`) and re-tested deepseek on the reachable tier: **0.53 → 0.96, confirmed** — the fix
  resolved the catastrophic negation-task failure (069) into a clean, coherent pass on every rep.
  Root cause validated, not just suspected. Format/hard/micro were not re-tested; still stale.
  Separately, added `qwen2.5:14b` to `badmodel-lab/roster.yaml` (8th local subject) and ran it
  across the full tier set: reachable 0.97, hard 0.95 — both essentially at the paid-API ceiling,
  zero VRAM/stability issues despite leaving only ~1.4GB headroom on the test machine's 12GB card.
  Its most interesting result: `qwen2.5:7b`'s severe negation gap (069, 0.33) resolves cleanly at
  2x scale (0.99), while its k-th-ordinal gap (075) only partially resolves — scale isn't a uniform
  fix across failure modes. Gap-filled reachable-tier data for 4 more local subjects
  (tinyllama 0.25, qwen2.5:0.5b 0.54, llama3.2:1b 0.42, phi3:mini 0.69). Full results:
  `badmodel-lab/MODEL_TIER_LIST.md`.

## Known gaps (named, not silently carried forward)

- ~~`compiled_plan.py::COMPOSITION_OPS` lists `ratio_argmax` as a supported composition op, but
  `execution_compiled.py::_COMPOSERS` has no handler for it~~ **FIXED**, independently, by a parallel
  leaf-extraction/composer session on `compiled-scaffold-dag` (landed in commit `f3d47666`, merged
  into this branch afterward) — `_compose_ratio_argmax` is now registered and
  `compiled_plan_test.py::test_composition_ops_matches_execution_compiled_composers` passes. Built
  as a Tier-A-only, all-or-nothing composer for task 064 after adversarial review found real bugs in
  earlier graceful-degradation designs; see that commit's history for the full story.
- `prompt_hygiene.py` (JSON self-contradiction lint) remains imported only by the compiled test
  harness — genuinely dead code on the live engine path. Wiring it into live prompt construction is
  unscoped, not part of Stage 1.
- **This document didn't count `badmodel-lab/playground/`** (a Docker Compose + SearXNG interactive
  chat demo, built the day before this doc, merged in from a parallel session) as continuum evidence
  — it should have. Per its own `MITIGATION_BRIDGE.md`, it drives `IdeaDagEngine` directly
  (`AGENT_USE_IDEA_DAG=1`), the same engine this document's "graph-as-primary" architecture decision
  describes — i.e. it's a genuinely working, end-to-end, interactive (not just benchmark-harness)
  instance of the continuum thesis, one this document should have cited as evidence rather than
  omitted.
- ~~`capability_tier()` collapses every unpriced (local) model into a single `weak` bucket
  regardless of actual size (0.5B vs. 70B)~~ **REFINED 2026-08-06, live-validated same day.**
  `capability_tier()` deliberately still returns the same 3 strings — the additive
  `model_tiers.local_model_size_band()` splits the weak bucket by the parameter count encoded in an
  Ollama-style `:Nb` tag (`tiny` <2B / `small` 2-6B / `medium` 6-12B / `large` >=12B), returning
  `None` (= "don't refine") for named-size tags (`phi3:mini`, `tinyllama`), MoE tags and any priced
  model. Wired into `native_vote_k_tiered_enabled` behind a second opt-in flag
  (`native_vote_k_size_band_enabled`, k = 4/3/2/1 by band) — a `None` band reproduces today's flat
  weak value exactly. Bands are cut where E1/E5's reachable-tier scores actually step (0.5b 0.54 /
  1b 0.42 vs. 7b 0.85 vs. 14b 0.97). This scales mitigation STRENGTH by size; it does NOT
  reclassify big local models out of `weak`, since scale is not a uniform fix across failure modes.
  **Live validation** (native `graph` engine, tasks 062/072, qwen2.5:0.5b/7b/14b + a phi3:mini
  control, R=2/cell — profiles `a5_native_vote_k_tiered` vs `a6_native_vote_k_sizeband`, see
  `badmodel-lab/profiles/`): the knob is confirmed genuinely wired, not just offline-tested — raw
  `llm.calls` telemetry moves in the predicted direction for every model (0.5b: a6>a5 calls, k=4 vs
  3; 7b/14b: a6<a5 calls, k=2/1 vs 3). Directionally: **7b (medium) holds** — dropping k 3→2 didn't
  measurably hurt. **14b (large) was the sharpest and most surprising result** — k=1 (voting off)
  scored notably *better* than k=3 on both tasks (combined mean 0.324 vs 0.064), consistent with
  (if anything stronger than) "large local models don't need blanket voting," though a partial
  hedged-vs-declarative-answer confound was found in the raw deliverables and n=2/cell is too small
  to fully separate a real effect from sampling noise. **0.5b (tiny) was inconclusive** for a
  different reason: most `a5` reps made **zero page visits** at all, which hard-gates the score to 0
  regardless of vote count — voting cannot rescue a run that never gathers evidence. This surfaces a
  real, more fundamental bottleneck for tiny local models on the native engine: the exploration/
  visit-triggering step, not finalize-time voting. The `phi3:mini` control (band=`None`, k=3 either
  way) confirmed no accidental perturbation. **Not flipped default-on** — n=2/cell is a first-contact
  directional check, not a statistically powered result; full run details in the session that landed
  this (2026-08-06).
- `SYSTEM_STATUS.md`'s prior "local-model validation dropped from the roadmap entirely" claim
  (2026-07-10) is superseded by this document — that decision no longer holds.
- ~~`execution_compiled.py::_is_reasoning_model` (line 366) doesn't cover deepseek~~ **FIXED in E5
  (commit `d17de329`)** — added `"deepseek"` to the `startswith` tuple, mirroring the native
  engine's `model_tiers.py::is_reasoning_model`. Confirmed with a live re-test, not just a code
  review: reachable-tier score recovered 0.53 → 0.96. Format/hard/micro tiers weren't re-tested and
  their old numbers are presumed-but-unconfirmed stale — a residual gap, not the original bug.
- `idea_tests/test_m02_amsterdam_area.py`'s `grounding` check requires an exact URL match against
  a literal string, but Wikipedia's canonical redirect target differs — two E4 models extracted the
  correct fact from the right page but scored 0.0 on grounding anyway, capping overall score at 0.5.
  A scoring artifact, not a model gap; not fixed here.
- `badmodel-lab/tiers.yaml` documents the `hard` tier as "floors even nano" — E4 found nano and
  flash-lite both score 0.99 on it, not floored. The doc comment is stale; not updated here.
