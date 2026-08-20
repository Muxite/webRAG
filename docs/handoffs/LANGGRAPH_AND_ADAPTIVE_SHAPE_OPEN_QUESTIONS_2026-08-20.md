# Open questions: LangGraph internals, the capability gap, and shape-adaptive DAG v2

Status: question bank for a future research pass, not started. Written 2026-08-20 at the end of the
26-cycle + architecture-design session (`OVERNIGHT_DEV_CYCLES_2026-08-20.md`,
`docs/superpowers/specs/2026-08-20-unified-dependency-dag-and-bounded-replanning-design.md`).

**Split for dispatch**: section A (LangGraph internals, Q1-7) plus Q17 are answerable by a web-research
agent (LangGraph docs/source, the Cognition AI post already cited in the design spec) at $0 and no
repo access. Sections B, C, D (Q8-16, 18-22) need this repo — code reading, the stored task/run
corpus, or a live A/B — and should go to a local/offline pass, not a web agent. See the two lists at
the end of this document.

**Set 1 returned 2026-08-20 — correction applied**: the web pass (transcribed in full below the
dispatch lists) confirmed LangGraph has no hidden planning system — its useful primitives are a
shared-state reducer model, Pregel/BSP dependency-ordered supersteps, dynamic fan-out (`Send`,
provider-agnostic), and checkpointed replay. `create_react_agent` is a bounded model→tools→model loop
with no separate evaluator step and no plan-level scheduling; any parallelism it has is *intra-turn*,
model-selected. **Important framing correction after this pass**: the design spec's Q20 answer and
its "strong planner decides shape, weak executor never sees a shape decision" framing implicitly
assumes a two-tier setup like Compiled v1 (expensive model authors, cheap model executes). **DAG v2
does not get that lever** — see `project-dag-v2-single-model-constraint` — it is bounded by the
strength of the single model it runs; there is no stronger model available to author for a weaker
one. The goal is to make *that one model* more capable through structure (dependency-ordered
dispatch, resolved-value slots, parallelism), not to route around its weakness with a bigger model.
This **does not invalidate Q15's core finding** (a correct dependency DAG + a ready-set dispatcher
makes sequential/parallel behavior emergent, no explicit mode switch needed) — that mechanism doesn't
require a stronger model, only correct edges, which can come from the same model or a mechanical/
static check. It **does invalidate Q13's framing** ("just default to Compiled v1 on chain tasks") as
a general fix for DAG v2 specifically, since Compiled v1's win depends on the second, stronger
authoring model DAG v2 doesn't have. Q17's node-type-safety table (parallelize evidence-gathering
leaves, serialize merge/commit nodes) still applies — it's a structural rule, not a two-tier one.

Context this builds on: `graph:good_adaptive` loses the majority of chain tasks against
`sequential_react` on OK/good-tier models (unclosed gap, confirmed Cycle 15/26), but sweeps weak
models for an availability reason unrelated to reasoning quality
([[project-langgraph-tool-api-gap]]: LangGraph's `create_react_agent` rejects ~half the cheap-model
market before inference because it requires an OpenAI-style tool-calling endpoint). The unified-DAG
design spec (§3, literature synthesis) proposes plan→execute(dependency-order)→probe-score→patch as
the general architecture. None of the below has been researched yet — it's the question set a future
session (or a Cycle-26-style research pass) should work through before touching code.

## A. How LangGraph actually works, mechanism by mechanism

1. `StateGraph`'s fan-in: when multiple nodes write to the same state key, what's LangGraph's actual
   merge semantics (reducer functions, `Annotated[list, add]` patterns)? Is this a general mechanism
   our `SimpleMergePolicy` could imitate, or does it only work because LangGraph's state is a single
   shared object rather than our per-node tree of results?
2. `create_react_agent`'s loop: is it a fixed-depth ReAct loop with no plan/parallelism at all, or
   does the prebuilt agent do anything resembling batching/parallel tool calls under the hood? If our
   `sequential_react` arm is already structurally equivalent to what `create_react_agent` does, why
   does one beat the other — is the gap prompt/model, or a real mechanism difference?
3. Conditional edges and `Send` API: LangGraph's `Send` primitive lets a node fan out to N dynamically
   created downstream nodes (map-reduce pattern). Is this the actual mechanism behind "parallelized
   graph react," and does it require the same OpenAI-tool-calling assumption that blocks weak models,
   or is it orthogonal (i.e., could a hand-rolled analogue of `Send` work on any text-completion
   model)?
4. Interrupt/checkpoint machinery (`interrupt()`, checkpointers): does LangGraph's durable-execution
   layer offer anything relevant to our bounded-suffix replanning idea (§4.4 of the design spec) —
   e.g., is "freeze validated prefix, regenerate suffix" already a first-class LangGraph pattern we
   could study even without adopting the framework?
5. How does LangGraph decide node execution order when there's no explicit edge dependency — is it
   pure topological/BFS-by-superstep (Pregel-style), and does that model explain why `langgraph_react`
   "just works" at parallel fan-out shapes without any of the auto-parallel-batch machinery our native
   engine needed to build by hand?
6. Tool-binding requirement: is there a LangGraph-supported path that does NOT require OpenAI
   function-calling (e.g., a custom `ToolNode` that parses text-encoded JSON like our engine does)?
   If yes, why does `create_react_agent` default to the API-gated path — packaging convenience, or a
   real correctness reason (e.g., structured retries) we'd be giving up?
7. Where does LangGraph's per-step cost/latency actually go relative to ours — is its "wins on
   OK/good models" result partly explained by fewer wasted LLM calls (no separate judge/evaluator
   step), and if so is that a graph-shape advantage or just "it doesn't do the thing we do that isn't
   paying for itself" (the pre-execution scoring problem §4.3 already names)?

## B. Closing the capability gap specifically (chain tasks, OK/good tier)

8. Is the chain-task loss actually about missing dependency structure (tree-not-DAG, §2 point 1), or
   about *when* evaluation happens (§2 point 3), or genuinely both — is there an offline replay
   (reusing tonight's stored corpus, per the design spec's suggested cheap validation) that could
   attribute the loss to one factor before building anything?
9. `sequential_react` has no judge/evaluator step at all. Does removing pre-execution scoring from
   `good_adaptive` chain tasks alone (without touching formation) already close most of the gap? Is
   there a cheap ablation (disable the evaluator entirely on chain-shaped tasks, keep everything else)
   that would answer this in one offline-corpus pass?
10. For chain tasks specifically, is a full dependency-DAG authoring pass overkill? A chain is a
    degenerate DAG (linear order) — would a much narrower "linear plan with resolved-value slots,
    like ReWOO but for straight chains only" close the gap at a fraction of §4.1's implementation and
    validation cost?
11. What fraction of the chain-task loss is judge-confidence calibration (named as a live open thread:
    a task scored 0.7 confidence on a step whose own reasoning admitted incompleteness) versus
    architecture? Is there a cheap way to separate "the plan shape is wrong" from "the confidence
    number coming out of a correct plan shape is miscalibrated"?
12. Historical pattern was 10W/7T/11L against `sequential_react` on chains — is that loss concentrated
    in a few tasks/models (a fixable outlier cluster) or spread evenly (a genuine structural
    disadvantage)? Worth a breakdown before assuming a full redesign is the only lever.
13. Given Compiled v1 already wins by a wide margin (0.725 vs native 0.198) using a full offline
    authoring pass — is the actual fastest path to closing the chain gap "make Compiled v1 the default
    for chain-shaped tasks" rather than building new native machinery? What would that cost in the
    weak-model-availability tradeoff Compiled v1 doesn't have to pay (it still needs *a* model to
    author, even if the executor is cheap)?

## C. Shape-adaptive DAG v2: being seq_react-like or parallel-graph-react-like on demand

14. What's the actual trigger signal for "this subtask is a chain, be sequential" vs. "this subtask is
    independent, be parallel"? Is it derivable statically at authoring time (a node's `depends_on` set
    is empty vs. non-empty — already the §4.2 proposal), or does it need a runtime signal (e.g., a
    node's early results suggest downstream nodes should now depend on it, even though the planner
    didn't know that upfront)?
15. Is "adapt shape at runtime" fundamentally different from "author the right dependency structure
    upfront and let a uniform dispatcher parallelize whatever has no edges"? I.e., does DAG v2 need a
    *mode switch* between sequential and parallel behavior at all, or does a single dependency-ordered
    dispatcher (§4.2) already produce sequential behavior on chains and parallel behavior on
    independent fan-outs as an emergent property, with no explicit "act like seq_react now" branch
    needed?
16. If a genuine runtime mode switch IS needed (not just emergent from dependency structure), what's
    the mechanical, cheap-to-check trigger (per the design spec's weak-model constraint — never a
    discretionary "should I go sequential now?" self-assessment)? Candidates: a node's contract
    declares a `requires_data` dependency that didn't exist at authoring time; a sibling batch's
    variance in early results exceeds some threshold; an explicit "this looks chain-shaped" static
    check from the planner.
17. Cognition AI's parallelize-gathering/serialize-synthesis finding is cited in the design spec as
    already matching our `SimpleMergePolicy`'s structure — is there a broader version of this rule
    that generalizes to "which node TYPES are safe to parallelize" beyond just merge nodes, that could
    inform a lighter-weight shape-adaptation heuristic than the full dependency-DAG redesign?
18. `SHAPE_ADAPTATION_OPEN_QUESTIONS.md` already has 50 open questions (§A-M) on this exact theme from
    2026-08-16, including the `requires_data` ancestor/sibling scope-mismatch bug and the unmeasured
    branch/parallel shape (9 cells). Which of those 50 questions does the unified-DAG design spec
    already answer or subsume, and which are still live and should be pulled into this list rather
    than duplicated?
19. Is there a minimal experiment — reusing the existing 9-task chain set and 8-task mixed set, at
    k≥2, fixture-parity — that tests "dependency-ordered dispatch alone, no global authoring pass, no
    post-evidence scoring change" in isolation? I.e., can §4.2 be validated completely independently
    of §4.1/§4.3/§4.4, or are they too entangled to test one at a time?
20. For weak models specifically: does shape-adaptation risk asking the weak model to do anything it's
    bad at (self-assessing "should this run in parallel")? Per §3's weak-model constraint, is the
    right design "the strong planner decides the shape, the weak executor never sees a shape decision
    at all" — and if so, does that mean shape-adaptation is purely a Compiled-v1/global-authoring
    concern, not something the *native* per-node-expansion path can ever do safely?

## D. Cross-cutting / prioritization

21. Given the design spec's own Panel B found the full redesign too expensive to validate this
    session, is there a research-only (no live spend) pass that could shrink the space of open
    questions above before the next implementation session — e.g., offline replay of the stored
    corpus to attribute the chain-task loss (question 8), or a LangGraph source read to answer section
    A's mechanism questions, both $0?
22. Do any of section A's answers change the §7.4 "safe slice" recommendation (multi-source fan-in
    resolution at merge nodes)? E.g., if LangGraph's `Send`/reducer pattern suggests a simpler fan-in
    mechanism than what §7.4 proposes extending (`_resolve_slot`), that's worth knowing before
    starting that implementation.

## Dispatch lists

### Set 1 — web-research agent (no repo access needed, $0)

LangGraph documentation/source reading plus the literature already cited in the design spec. No
webRAG-specific facts required to answer these — a general web agent with no memory of this repo can
do this pass standalone, then a follow-up session reconciles findings against Set 2.

- Q1 — `StateGraph` fan-in / reducer semantics
- Q2 — `create_react_agent`'s actual loop structure (parallel/batching or pure fixed-depth ReAct?)
- Q3 — the `Send` API / map-reduce fan-out primitive, and whether it shares the tool-calling-API gate
- Q4 — interrupt/checkpoint machinery vs. our bounded-suffix-replanning idea
- Q5 — LangGraph's execution-order model (Pregel-style superstep BFS?) when no edge is declared
- Q6 — whether a non-tool-calling-API LangGraph path exists, and why `create_react_agent` doesn't default to it
- Q7 — where LangGraph's per-step cost/latency goes relative to ours (does it skip a judge/evaluator step entirely?)
- Q17 — re-read the Cognition AI parallelize-gathering/serialize-synthesis source cited in design spec §3; is there a more general node-type-safety rule in it than what's already extracted?

### Set 2 — local/offline pass (needs this repo: code, stored corpus, or a live A/B)

- Q8 — offline replay of the stored corpus to attribute the chain-task loss (formation vs. scoring-timing vs. both)
- Q9 — ablation: disable the evaluator on chain-shaped tasks only, keep everything else, offline-corpus pass
- Q10 — scope a chain-only linear-plan/resolved-value mechanism as a narrower alternative to the full §4.1 DAG authoring pass
- Q11 — separate "plan shape wrong" from "confidence calibration wrong" in the existing loss breakdown
- Q12 — break down the historical 10W/7T/11L chain result by task/model: outlier cluster or spread evenly?
- Q13 — cost out "default to Compiled v1 on chain-shaped tasks" against the weak-model-availability tradeoff it gives up
- Q14 — is the sequential/parallel trigger derivable statically from `depends_on`, or does it need a runtime signal?
- Q15 — does dependency-ordered dispatch alone make shape-adaptation emergent, with no explicit mode switch needed?
- Q16 — if a runtime mode switch is needed, what's the cheap mechanical (non-discretionary) trigger for it?
- Q18 — reconcile this list against the existing 50 questions in `SHAPE_ADAPTATION_OPEN_QUESTIONS.md` — what's answered/subsumed, what's still live?
- Q19 — design a minimal experiment that validates §4.2 (dependency-ordered dispatch) in isolation from §4.1/§4.3/§4.4
- Q20 — confirm shape-adaptation is a strong-planner/Compiled-v1-only concern, never asked of the native per-node weak-model path
- Q21 — enumerate what else in this list is $0/offline before the next live-spend session
- Q22 — once Set 1 returns, check whether its answers change the §7.4 safe-slice recommendation

## Set 1 results (web-research pass, returned 2026-08-20)

Condensed findings per question — full transcript in session history if the detail below is
insufficient.

- **Q1 (fan-in/reducers)**: `StateGraph` keeps one shared state; each key has an independent reducer
  (`Annotated[T, reducer]`, e.g. `operator.add` for lists, `add_messages` for ID-aware append/replace)
  — replace-on-write is the default with no reducer declared. The transferable idea for
  `SimpleMergePolicy`/§7.4 is the *pattern* (named merge slot + deterministic, ideally associative
  reducer over multiple writes with retained provenance), not the shared-dict data model — our
  per-node result tree needs explicit source-node IDs and scoped resolution that a flat shared state
  doesn't give you for free. Keep `_resolve_slot` as the source-aware layer; add a merge-node contract
  (`inputs: [node_a.output, node_b.output]` + reducer) on top of it, not instead of it.
- **Q2 (`create_react_agent` loop)**: a bounded model→tools→model loop, nothing more — no plan, no
  dependency inference, no autonomous batching. The one real parallelism is intra-turn: if one model
  turn emits multiple tool calls, v1 runs them together in one tool-node step and v2 dispatches each
  via `Send`, but this is still model-selected, same-turn parallelism, not plan-level scheduling. If
  `sequential_react` is architecturally the same shape, the live gap against it is probably prompt/
  parsing/context-construction/retry-policy, not a hidden mechanism — worth checking directly rather
  than assumed.
- **Q3 (`Send`)**: general dynamic fan-out primitive, not tied to tool-calling or any provider API —
  ordinary Python routing code can return `Send` objects; `create_react_agent` just happens to reuse
  it for model-produced tool calls. A hand-rolled analogue (parser emits N work items → dispatcher →
  concurrent workers → explicit merge node) is fully compatible with text-only weak models; the cost
  is that we own schema enforcement, malformed-output repair, and result-to-call correlation ourselves
  instead of getting it from a provider-normalized tool-call format.
- **Q4 (checkpoints)**: relevant as a persistence substrate (checkpoint-then-fork/`update_state(...,
  as_node=...)`-then-resume maps loosely onto "freeze prefix, regenerate suffix"), but it's not a
  first-class suffix-replanning feature — LangGraph doesn't decide the invalid boundary, generate a
  replacement suffix, or prove prefix validity; all of that stays our policy to build. Also: resumed
  nodes replay from their own start, not mid-node, so pre-interrupt side effects need idempotency if
  we ever borrow this pattern.
- **Q5 (execution order)**: Pregel/BSP bulk-synchronous supersteps — plan (find nodes activated by the
  prior step's writes) → execute concurrently (writes invisible to peers mid-step) → commit/reduce.
  No implicit dependency discovery; the executable structure is exactly what the edges/channels
  declare. This is the strongest support for Q15: a correct explicit dependency DAG plus a ready-set
  dispatcher makes chain-vs-parallel behavior *emergent* — LangGraph doesn't need a "be sequential now"
  switch because its scheduler already only activates nodes whose inputs are satisfied.
- **Q6 (non-tool-calling path)**: yes at the framework level — a LangGraph node is just a Python
  function reading/writing state, so a text-completion model + our own parser + a custom tool-exec
  node + conditional routing is ordinary graph composition (LangGraph's own interrupt docs show a
  hand-built agent/tool loop). `create_react_agent` defaults to the tool-calling-API path for
  packaging/interoperability reasons (a normalized call/arg/ID representation), not because the
  framework requires it. This argues for keeping our own textual tool protocol, not for abandoning
  graph-style dispatch as a concept.
- **Q7 (cost/latency source)**: `create_react_agent` has no mandatory separate evaluator/judge step —
  one LLM call per ReAct turn plus tool execution, full stop, unless optional structured-output or a
  guardrail hook is added. This directly motivates the Q9 ablation (evaluator-off on chain tasks,
  formation/model/budget held fixed) as the highest-value first local experiment — if most of the gap
  is the extra scoring call plus whatever confidence/noise it injects, that's a cheap, isolated fix
  distinct from any formation redesign.
- **Q17 (node-type parallel-safety)**: sharper than "gathering parallel, synthesis serial" — parallel-
  safe iff a node's output is independent, read-only-or-append-only evidence, composable under a
  deterministic merge; serialize anything that commits a choice, writes a shared artifact, or consumes
  an unresolved sibling's output. Concretely: search/visit/extraction/classification default-safe;
  merge/rank/plan-revision/final-answer/any stateful write default-unsafe. This is a usable, cheap
  heuristic for a lighter-weight shape-adaptation rule than the full DAG redesign — and it's a
  *contract-metadata* rule (planner tags a node's read/write footprint once), not a per-step model
  self-assessment, so it's compatible with a weak single-tier model per
  `project-dag-v2-single-model-constraint`.

**Correction on delivery** (see note above the dispatch lists): the web pass's own synthesis leaned on
a strong-planner/weak-executor framing for Q14/Q16/Q20 that doesn't fit DAG v2 — DAG v2 has one model
tier, not two. Q1/Q3/Q5/Q7/Q17's findings above are unaffected (they're single-model-compatible
structural mechanisms); only the "give the planner a stronger model" framing needs discarding, not the
underlying scheduling/parallelism findings themselves.

### Revised local/offline priority given the single-model constraint

Order unchanged for Q12 → Q9 → Q8/Q11 → Q19 (none of these assumed a two-tier setup). Two changes:

- **Q13 downgraded**: "default to Compiled v1 on chain tasks" is not a DAG-v2 fix — it's a different,
  already-shipped variant that legitimately uses a second stronger model. Worth keeping as a *routing*
  question (when should the system pick Compiled v1 over DAG v2 at all?) but not as a lever for making
  DAG v2 itself better.
- **Q20 reframed**: not "is shape-adaptation a Compiled-v1-only concern" (no) but "can shape-adaptation
  work with contract metadata + a mechanical dispatcher, supplied by the same single model or a static
  check, with no self-assessment step asked of that model." Q17's node-type-safety table is the
  concrete mechanism to test this with.
