# Design: a unified dependency-DAG with post-evidence scoring and bounded-suffix replanning

Status: design, 2026-08-20. Not built. Large-tier per `docs/DEV_CYCLE.md` — full spec, adversarial
panel, and pre-registration required before any live spend.

## 1. Why this document exists

Tonight's session (26 dev cycles, `docs/handoffs/OVERNIGHT_DEV_CYCLES_2026-08-20.md`) diagnosed and
fixed a long list of independent bugs in the native DAG v2 engine, and confirmed the central,
harder finding at the end of that work: **the chain-task gap against `sequential_react` did not
close**, even with every unconditional fix landed and a promising-but-unproven contract-veto fix
enabled. A fresh capability-spectrum run confirmed the shape holds: `graph:good_adaptive` sweeps
weak models (for reasons unrelated to reasoning quality — LangGraph structurally can't run there) and
loses the majority on OK/good models. Cycle 12's own conclusion, independently reached mid-session,
was: *"the remaining lever is a real architecture change (score outcomes, not plans), not another
prompt/data probe."*

At the user's direction, this document takes that seriously: rather than continuing to find and fix
independent bugs one at a time, it proposes a structural redesign of the engine's core loop, informed
by a literature survey of current agentic-architecture best practice (full synthesis in the Cycle-26
research pass; summarized in §3 below) and by what tonight's own diagnosis already proved about this
specific codebase.

**The reframe that makes this tractable**: several of tonight's narrow, flag-gated fixes are already
*fragments* of the exact patterns the literature converges on. This is not a rewrite from zero — it
is a generalization of mechanisms already built, tested, and partially validated tonight:

| Tonight's narrow fix | The general pattern it's a fragment of |
|---|---|
| Resolved-value channel (`requires_data.slot`, Cycles 1/9/10) — a dependent node's field gets filled from a named source node's structured output at dispatch time | ReWOO's `#E1`/`#E2` placeholder substitution — a minimal single-plan version of dependency-edge data flow |
| F6's narrow fallback-parent re-expansion (Cycle 9) — a degenerate node gets exactly one bounded retry, triggered by a mechanical tag (`fallback_expansion`), corrective-prompted, structurally capped at one attempt | ADaPT's "decompose only the subtask that actually failed" — decomposition triggered by demonstrated failure, not upfront self-assessment |
| F35's contract-veto-requires-datum fix (Cycle 16) — a re-expansion trigger can no longer be silenced by "opened a page matching my own goal's words," only by a verified datum | A step toward LATS's post-rollout scoring — moving the judgment point closer to real evidence, away from subject-match guesswork |
| Compiled v1 (already shipped, 2026-06) — an expensive model authors a full DAG plan offline; a cheap model executes it, scoring 0.725 vs native's 0.198 | Exactly the field's current consensus: concentrate planning complexity on the strong model, keep the weak model's job narrow and schema-bounded |

The proposal below is the generalization of all four rows into one coherent architecture, not a fifth
independent mechanism bolted on top.

## 2. The four structural problems, restated precisely

From `agent/app/DAG_FORMATION_REVIEW.md` and `ENGINE_DESIGN_REVIEW.md`, both independently confirmed
by tonight's empirical work:

1. **The graph is a tree, not a DAG.** No `depends_on` field exists in the expansion schema; the one
   correct dependency-DAG builder (`plan_library.link_dependencies`) is unreachable outside
   template-matched nodes. A chain task is hop *n+1* consuming hop *n*'s value — a fan-in a tree
   cannot express structurally, only via narrow side-channels (`requires_data`, now itself only
   written at two call sites per tonight's Cycle 1 work).
2. **Formation is greedy and context-blind.** Each expansion call sees only its own ancestor path,
   never siblings or the whole graph — there is no global planning step in the native path (Compiled
   v1 is the exception, and it wins by a wide margin specifically because it has one).
3. **Evaluation happens before execution, by construction.** `_expand_or_execute` drops
   DONE/FAILED/SKIPPED children before scoring, so ~every candidate the judge ever sees is
   unexecuted — tonight's Cycles 5/6/11/12 traced this to a structural ceiling (98.2% of flat-scored
   batches sit on a fixed cap/rubric/fallback value, not genuine judge degeneracy) that no prompt
   rewrite or detail-budget fix can lift, because the judge is scoring hypothetical plans, not real
   outcomes.
4. **Formation is mostly one-shot.** Re-expansion exists but is narrowly gated (opt-in, bounded to
   one iteration per lineage, and — before tonight's Cycle 9 — blocked entirely for any node that
   already has children). A badly-planned subtree has no repair path.

## 3. What the literature says (full synthesis: Cycle-26 research pass, cited there)

Condensed to the four recommendations that map directly onto the problems above:

- **Fix #1 (tree→DAG)**: explicit shared-state fields with typed dependency edges (ReWOO
  placeholders; LangGraph `StateGraph`'s fan-in merge nodes) — a downstream node's input is a
  *reference* to a named upstream field, not a re-derivation from ancestor text.
- **Fix #2 (context-blind formation)**: split a global planner pass (sees the whole task, authors the
  whole DAG) from a local executor pass (sees just its assigned node + resolved inputs) — exactly
  Compiled v1's already-proven division, generalized to be the default rather than an alternate mode.
- **Fix #3 (pre-execution scoring)**: move the judgment point to *after* at least a cheap real
  observation exists (LATS's post-rollout scoring; PIVOT's INSPECT stage diagnoses divergence between
  planned and *actually executed* trajectory, never scoring plan text alone).
- **Fix #4 (one-shot formation)**: PIVOT's "freeze the validated prefix, regenerate only the
  unsupported suffix once execution diverges from the plan" — bounded-scope replanning triggered by a
  **mechanical, cheap-to-check divergence signal**, not a discretionary self-assessment.

**The weak-model constraint** (explicitly researched per the user's direction): the safest place for
planning complexity is the strong model; the safest ask of the weak model is a narrow,
schema-bounded action per step. ReWOO's placeholder syntax, LATS's relative candidate scoring, and
any pattern requiring a model to introspect its own capability boundary are all flagged as fragile on
small models — this repo's own SLM literature reference found 0% structured-output accuracy on 2 of 4
small models tested even when task reasoning itself was up to 85% correct. The redesign below keeps
DAG-authoring and dependency-edge creation on the strong/compiled planner and keeps the weak
model's job to filling one field or answering one yes/no divergence check — never authoring plan
structure or comparing candidates against each other.

## 4. Proposed architecture: Plan → Execute (dependency order) → Probe-Score → Patch

### 4.1 Formation: one global DAG-authoring pass, not incremental per-node expansion

Replace the native path's per-node `_handle_expansion_node` calls (each blind to siblings and the
whole graph) with **Compiled v1's authoring mechanism as the default formation step**, not an
alternate mode selected by a separate variant flag. Concretely:

- On mandate receipt, run one strong-model call (or the existing compiled-scaffold authoring policy,
  reused not reinvented) that sees the **whole mandate** and authors the **whole initial DAG**:
  nodes, and for each node a `depends_on: List[str]` field naming prior node IDs whose output this
  node's execution needs.
- Extend the expansion schema (`idea_dag_schemas.py`) with `depends_on` as a first-class field,
  replacing the `requires_data` side-channel's role as the *only* structural mechanism, though
  `requires_data.slot`'s existing dispatch-time resolution machinery (Cycles 1/9/10, already built
  and tested tonight) becomes the **execution-time consumer** of these edges — this is directly
  reusable, not replaced. A node's `depends_on` entries are exactly the `source_node_id`s a
  `requires_data.slot` declaration can point at; the schema addition is what lets the *planner*
  declare these edges up front for arbitrary nodes, not just the two narrow writer sites
  (`plan_library.py`, `post_expansion_hooks.py`) that declare them today.
- `graph.merge_nodes(parent_ids: List[str], ...)` — confirmed in `DAG_FORMATION_REVIEW.md` PART 0 as
  already built, unit-tested, and simply never wired into the native path — becomes reachable: a
  fan-in join is now a real, plannable graph shape, not something only `SimpleMergePolicy`'s
  single-parent `add_child` calls can produce.

### 4.2 Execution: dependency-ordered dispatch, parallel where no edge exists

- A node is eligible to dispatch once every node in its `depends_on` set is DONE (this generalizes
  `_has_required_data`'s existing Condition B, already correctly scoped to ancestor-only — extend it
  to arbitrary named dependencies, not just ancestor-path nodes).
- Independent nodes (no path between them in the dependency graph) execute in parallel, through the
  existing auto-parallel batch machinery (`_run_one`, `execute_all_children`) — **already fixed
  tonight** (Cycle 1's dispatch-coverage fix, Cycle 2's visit-throughput fix) to correctly resolve
  `requires_data.slot` on that path. This section of the redesign is largely "trust and extend
  infrastructure already built and live-verified tonight," not new construction.
- Per the Cognition AI finding cited in the research synthesis: parallelize freely on
  **research/gathering** nodes (search, visit), but keep the **synthesis/merge** node that commits the
  final answer single-threaded — this repo's existing `SimpleMergePolicy`/finalize path already has
  this property by construction (one merge node per join point); the redesign must not introduce
  concurrent writers to a shared answer-composition state.

### 4.3 Scoring: post-evidence, not pre-execution

- Retire the default of scoring every sibling candidate before any of them execute. The redesign's
  default evaluation point moves to **after a node has executed and produced a real observation**
  (tonight's `evaluate_parallel_siblings` flag is exactly this mechanism, already built — the redesign
  makes it the default rather than an unvalidated opt-in, once §5's re-validation gate below is
  satisfied).
- Where a decision must be made *before* any candidate executes (e.g. choosing which of several
  candidate next-actions to try first), use a **cheap probe** rather than full pre-execution text
  scoring: dispatch the top 1-2 candidates by a lightweight heuristic (not a full comparative LLM
  judgment), let real observations arrive, and only then apply the evaluator to genuinely
  differentiable, evidence-backed candidates. This bounds the cost of "LATS-style post-rollout
  scoring" to a small probe width rather than a full tree search, keeping it affordable at every
  model tier.
- This directly resolves the structural ceiling found in Cycles 5/6/11/12: a judge cannot be forced
  onto a fixed cap/rubric/fallback value for "no action_result" if the default path only ever asks it
  to score nodes that *have* an action_result.

### 4.4 Adaptation: bounded-suffix replanning on mechanical divergence

- Generalize Cycle 9's F6 MVP (currently scoped narrowly to `fallback_expansion`-tagged parents)
  into the general PIVOT-style mechanism: when a node's real execution result **diverges** from what
  the plan assumed at authoring time, freeze everything upstream/already-validated and regenerate
  only the invalidated downstream subtree.
- **Keep the trigger mechanical, not discretionary** — this is the single most important constraint
  from the weak-model research (§3): the divergence check must be a cheap, checkable comparison
  (does the resolved `requires_data.slot` value exist / does the contract's declared datum appear in
  the real result / does the executed node's status contradict what a downstream `depends_on` edge
  assumed), never a full self-assessment prompt asking a model "should I replan now?" — that judgment
  call is exactly what weak models are worst at, and is unnecessary when a structural check already
  exists (this repo already has several: F35's datum-verification check, `_has_required_data`'s
  readiness gate, the contract-satisfaction machinery).
- Reuse the existing bounded-retry discipline from Cycle 9 (structural shape-guard capping retries
  independent of any iteration-count knob) rather than inventing a new unbounded loop.
- ADaPT's simpler, cheaper version — decompose only the specific node that just failed, not a whole
  suffix — is the correct default for weak models per §3; PIVOT's fuller "regenerate the whole
  invalidated suffix" is reserved for stronger models or higher-value tasks where the extra
  replanning cost is justified, mirroring this repo's existing capability-tiered instinct.

### 4.5 What this explicitly does NOT change

- **Not a rewrite of the executor/leaf-action layer** (`VisitLeafAction`, `SearchLeafAction`, etc.) —
  those already work and were the target of most of tonight's independent bug fixes (visit-timeout,
  dead-URL recovery, chrome filtering, sibling-URL dedup). This design sits above that layer.
- **Not a replacement for Compiled v1** — it's Compiled v1's authoring mechanism promoted to the
  default formation step, with the DAG's dependency structure now first-class instead of implicit in
  a fixed offline plan, and with the bounded-replanning mechanism (§4.4) added on top so a compiled
  plan is no longer purely fixed-and-unrevisable (the documented tradeoff Compiled v1 accepted, per
  `README.md`'s Versioning section: "a fixed plan can't react to what a step actually reveals").
- **Not a change to the weak-model-facing action vocabulary** — search/visit/think stays the fixed,
  narrow schema weak models fill one field of at a time; no new open-ended authoring burden is placed
  on any model below the strong planner tier.

## 5. Rollout discipline (per `docs/DEV_CYCLE.md`, Large tier)

This is a cross-cutting architecture change, not a narrow bug fix — it needs the full ceremony:

1. **Adversarial panel** (2-3 agents, devil's-advocate framing) on this document before any code —
   specifically stress-testing: does §4.1's schema change break any of tonight's 26 cycles' worth of
   flag-gated mechanisms (resolved-value channel, F6/F9/F35/F36/F37, dedup fixes)? Does the
   dependency-DAG authoring pass risk becoming as expensive/slow as Compiled v1's own documented
   tradeoffs? Is the "mechanical divergence trigger" in §4.4 actually well-defined and cheap for every
   node type, or does it silently degrade to a discretionary judgment call for some action types?
2. **Build as a new, separately selectable execution path**, not a mutation of the existing
   `good_adaptive` arm in place — e.g. a new arm profile (naming TBD by the implementer, consistent
   with existing `_GOT_ARM_PROFILES` conventions) — so it can be A/B'd directly against
   `good_adaptive`, `sequential_react`, and `langgraph_react` without risking regression to what
   ships today. Existing flag-gated mechanisms from tonight (F35, F37, resolved-value channel) should
   be exercised or subsumed by the new path where their mechanism generalizes, and explicitly noted
   as either "absorbed" or "orthogonal, kept as-is" per mechanism.
3. **Full offline test suite gate** before any live spend, as every cycle tonight has done.
4. **Pre-registration** (per `scripts/LADDER_PREREGISTRATION.md`'s precedent) before any live
   validation run — primary metric, task set, arms, and stopping rule stated before results are seen,
   given how many of tonight's own findings (Cycles 15, 21, 26) turned out to hinge on fixture parity
   and avoiding post-hoc metric shopping.
5. **Validation target**: the same 9-task chain set and 8-task mixed set used throughout tonight
   (`046,047,065,093,135,136,137,138,139` / `054,085,055,061,146,147,149,122`), plus the 3-tier
   capability spectrum from Cycle 26 (`llama-3.2-3b` / `gpt-4.1-nano` / `gemini-2.5-flash`) — the
   redesign's success criterion is closing some real fraction of the chain-task gap against
   `sequential_react` *without* regressing the mixed-task performance or the weak-tier
   availability advantage, at k≥2 under fixture parity.

## 6. Open questions for the adversarial panel

- Does authoring a full dependency-DAG upfront (§4.1) reintroduce Compiled v1's documented weakness
  — "a fixed plan can't react to what a step actually reveals" — if §4.4's replanning trigger turns
  out to fire too rarely in practice? What's the cheapest offline check (replaying tonight's stored
  corpus, similar to Cycles 5/12/13's offline-first methodology) that could validate the divergence
  trigger's sensitivity before any live spend?
- `plan_library.link_dependencies` (confirmed built, tested, and correct — `DAG_FORMATION_REVIEW.md`
  F5) already solves §4.1's problem for template-matched nodes. Should the new global authoring pass
  reuse that machinery's wave-depth/topological-ordering logic directly, or does organic (non-
  template) authoring need different logic given the LLM is now authoring `depends_on` edges freely
  rather than a template providing them?
- Cost: Compiled v1's known tradeoff is one expensive upfront call. Does authoring a full DAG (not
  just a linear plan) meaningfully change that cost profile, and does it still clear the bar this
  repo's `feedback_adaptive_cost_framing` memory sets (a cost premium is fine as a deliberate
  strategy, not a defect) — worth an explicit token-cost estimate before implementation, not just
  after.
