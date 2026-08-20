# DAG Formation Review — how the graph gets built

**Written 2026-08-19. Adversarial review, documentation only — no code changed, no spend.**

Third document of the same cycle:

- [`ASSUMPTION_AUDIT.md`](ASSUMPTION_AUDIT.md) — constants never validated
- [`ENGINE_DESIGN_REVIEW.md`](ENGINE_DESIGN_REVIEW.md) — decisions that cost score at execution/finalize
- **this document** — decisions that produce bad graph *structure*, before execution begins

Evidence labels: **VERIFIED** (read and confirmed here), **REPORTED** (from an existing doc),
**HYPOTHESIS** (untested mechanism).

---

## PART 0 — The headline: `IdeaDag` builds a tree

**VERIFIED.** The class is named `IdeaDag`, the product is a Graph-of-Thoughts engine, and the
structure the engine actually constructs at runtime is a **strict tree**.

Three independent confirmations:

1. **The multi-parent API is unreachable in production.** `IdeaDag.merge_nodes(parent_ids:
   List[str], ...)` (`idea_dag.py:96-127`) is the only method that sets `parent_ids` to more
   than one id. Its callers, repo-wide: `idea_dag_log.py:127-128` (a demo module) and the unit
   tests. **Never `idea_engine.py`, never `idea_policies/merge.py`.** The runtime merge path,
   `SimpleMergePolicy.create_merge_node` (`merge.py:101-107`), calls
   `graph.add_child(parent_id=parent_id, ...)` — single parent.
2. **Traversal assumes a tree anyway.** `path_to_root` (`idea_dag.py:240-251`) follows
   `parents[0]` and ignores the rest, so even a genuinely multi-parent node would be walked as
   if it had one.
3. **The LLM has no vocabulary for a dependency.** `EXPANSION_JSON_SCHEMA`
   (`idea_dag_schemas.py:16-59`) accepts `title` / `action` / `details` per candidate, plus a
   single **graph-level boolean** `meta.execute_all_children`. There is no `depends_on` field.
   The model cannot express "child 3 needs child 1's output" — only "run this whole batch in
   parallel" or "run this whole batch in sequence."

So DAG-ness survives only as a `details.requires_data` side-channel pointer between sibling
node ids — invisible to `iter_depth_first`, `iter_breadth_first`, `leaf_nodes`, `to_dict()`
structure, and every depth/path traversal.

**Why this is the most consequential finding in the cycle.** A chain task is hop *n+1*
consuming hop *n*'s value. That is a **fan-in**, and a tree cannot express it. This gives a
single mechanism for three separately-documented symptoms:

- `SHAPE_ADAPTATION_OPEN_QUESTIONS.md` Q32 — no extraction step exists in finalize (REPORTED).
- Q12 — turning off `auto_parallel_siblings` recovered `chain_coverage` but overall score still
  trailed; the deficit **relocated** to extraction/finalize rather than disappearing (REPORTED).
- `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md:653-665` — `seq_react` beats the graph engine on
  chains, 10W/7T/11L (MEASURED).

If there is no edge to carry a value from hop *n* to hop *n+1*, the value must be re-found by
scanning concatenated page text at finalize. A linear ReAct loop carries it in message history
for free. **That is a coherent explanation for the graph losing on its own designed best
shape**, and it is structural — no amount of scheduling or prompt work fixes a missing edge type.

---

## PART 1 — Formation is greedy, local, and blind

### F1. No global plan step exists — VERIFIED

`LlmExpansionPolicy.expand` (`expansion.py:572`) is the **only** decomposition entry point, and
`_handle_expansion_node` (`idea_engine.py:1045-1130`) calls it identically for the root
(`idea_engine.py:397`) and for every interior node. The model sees only `path_to_root(node)`
(`expansion.py:896-897`) — its own ancestor chain. It never sees sibling subtrees or the graph
as a whole.

**Outcome:** there is no whole-task planning step anywhere. Graph shape is an emergent side
effect of N independent, context-blind LLM calls. Two sibling subtrees can duplicate work or
adopt incompatible decompositions because neither call can see the other.

Contrast worth noting: the compiled-scaffold path (**Compiled v1**) *does* have a global plan —
an expensive model authors the whole DAG offline. It also scores 0.725 vs native's 0.198
(REPORTED, `AGENT_FAILURE_MODES_2026-08-10.md` #2, d=1.77). Global planning is not an untested
idea in this repo; it is the thing that already works, and the native path does not do it.

### F2. Depth is never chosen, tracked, or capped — VERIFIED

Repo-grep of `idea_engine.py` for `depth` returns nothing in the step loop. There is no
`max_depth`, no depth-aware planning signal, no code that reasons about how deep a
decomposition ought to go. The only structural controls are `max_branching` (=5,
`idea_dag_settings.json:2`) and `max_total_nodes` (=500).

**Outcome:** whether a run produces a depth-1 star or a depth-8 chain is pure emergence.
Nothing detects a degenerate shape, because depth is not a variable the system holds.

This compounds `ENGINE_DESIGN_REVIEW.md` PART 3: `auto_parallel_siblings` keeps graphs
depth-1, measured max `path_to_root` is 2, and backtrack's 5-node requirement is therefore
unreachable — 0/261 firings. Depth is simultaneously **unmanaged** and **load-bearing** for
mechanisms that silently depend on it.

### F3. Dependency edges are invented by URL-sniffing, not declared — VERIFIED

`_parse_candidates` (`expansion.py:1407-1450`): for a `visit` candidate with no URL, the code
regexes the title/justification/ancestor text for a URL and, if it can trace the source,
retroactively attaches `details[REQUIRES_DATA]`.

This fires **only** for `visit` actions, and **only** when a URL happens to be recoverable from
prose. Search→search, think→search, and every other hop type never gets a dependency edge.
Sequencing then falls back to the coarse `execute_all_children` flag (PART 0), which forces
either false sequentiality (everything waits) or false parallelism (nothing waits) whenever the
true dependency structure is neither "all" nor "none."

Related, REPORTED: project memory records a `requires_data` **scope mismatch** — an
ancestor-scoped writer against a sibling-scoped reader, failing by construction. Worth
re-checking against current code as part of any fix here.

### F4. `execute_all_children` is declared before the graph knows anything — VERIFIED

`idea_dag_settings.json:60` instructs the model to decide `execute_all_children` *while
proposing the candidate list*, before any child exists, has been scored, or has had
dependencies discovered.

Combined with F3, a graph can hold a `requires_data`-linked pair of children under a parent
that was simultaneously told `execute_all_children: true` — the coarse pre-declared flag and
the specific post-hoc edge in direct contradiction. This is the same **decide-before-the-
information-exists** pattern catalogued in `ENGINE_DESIGN_REVIEW.md` PART 3.

### F5. The one correct dependency-DAG builder is unreachable on the default path — VERIFIED

`plan_library.link_dependencies` (`idea_policies/plan_library.py:195-243`) is the only code in
the repo that turns a real multi-hop `depends_on` chain into `requires_data` edges properly —
fan-in to the deepest wave, topological ordering via `_wave_depths`.

It runs only when `plan_library.enabled` **and** `plan_library.auto_enabled` are both on
(`idea_engine.py:1066`), and only for nodes a template confidently matched
(`idea_engine.py:1030-1038`, otherwise `None, None`). Every organic mandate without a matching
pre-authored template gets F3's degraded wiring instead.

**Same shape as `ASSUMPTION_AUDIT.md`'s T1-1:** the repo solved the problem correctly in one
subsystem and the solution never propagated to the general path.

### F6. Formation is one-shot; a badly-planned node can never be re-planned — VERIFIED

Re-expansion machinery exists (`idea_engine.py:900-927`) but is opt-in/env-gated, bounded to
`max_iterations` (default 1) per lineage, and **explicitly skips any node that already has
children** (`idea_engine.py:907-908`).

**Outcome:** once a node is expanded — even into a single degenerate leaf (F7) — its children
can never be revised, replaced, or supplemented. The only recourse is spawning new siblings
elsewhere, not repairing the malformed subtree. Pair this with `ENGINE_DESIGN_REVIEW.md` D4
(`merge_should_skip` is an irreversible lockout) and a pattern emerges: **the graph is
append-only in practice, with no repair path at any level.**

**PARTIALLY ADDRESSED (narrow MVP), scoped to the F7 degenerate case only.**
`got_reexpand_fallback_nodes_enabled` (opt-in, JSON default false) adds
`_maybe_reexpand_fallback_parent`: when a just-completed leaf carries F7's
`DetailKey.FALLBACK_EXPANSION` tag AND its parent's **whole** child set is that single leaf,
the parent is re-planned through the ordinary `_apply_reexpand` path (via its one carve-out,
`allow_existing_children`) with a corrective hint naming the parse/shape failure. The
superseded guess is marked `SKIPPED` + `DetailKey.FALLBACK_SUPERSEDED`. The final payload
carries `fallback_reexpand_attempted_count` / `fallback_reexpand_recovered_count` when
non-zero. Wired into both the sequential (`_apply_action_result`) and batch dispatch paths.
(`reexpand_fallback_parent_test.py`.)

Bounds and scope, deliberately:
* it repairs a parent **at most once** — the guard is structural (the child set is no longer
  a lone fallback leaf after a retry), so a retry that degenerates again is not retried a
  third time regardless of `reexpand_max_iterations`;
* a repaired parent ends with `max_branching + 1` children (the SKIPPED guess still occupies
  a slot). Accepted soft-cap overshoot, pinned by test;
* **the general F6 finding remains open.** A node with a genuine multi-child plan that turns
  out to be wrong still cannot be re-planned. Only the known-degenerate case is covered.

**Finalize leak: FIXED.** `idea_finalize.py`'s context builders selected on `ACTION_RESULT`
presence and success, so the superseded leaf's already-executed content still reached the
final synthesis alongside the retry's. `_is_superseded` now gates every finalize CONTEXT
selector on the `FALLBACK_SUPERSEDED` marker: `_collect_leaf_results_fallback`,
`_collect_all_visit_content`, `_build_fallback_deliverable`, `_build_node_summary_table`
(whole row dropped — it carries the node's outcome text) and the three chroma query builders
in `_retrieve_final_chroma_context`. Scoped to the marker, so unrelated `SKIPPED` nodes are
selected exactly as before; and deliberately NOT applied to `_visited_sources` /
`_has_grounded_evidence`, which only assert a page was opened — excluding it there could make
the grounding gate refuse an otherwise-grounded run. No-op unless
`got.reexpand_fallback_nodes_enabled` is on, since nothing else stamps the marker.
(`finalize_fallback_superseded_test.py`.)

### F7. Malformed plans collapse to one degenerate node, silently — VERIFIED

`_create_fallback_candidate` (`expansion.py:1479-1536`) fires when `_parse_candidates` returns
zero candidates. In order: a `visit` for the first URL regexed from title/mandate; else a
`search` whose query is `title[:100]` — an arbitrary character truncation, not a query; else a
bare `think` node with no action semantics.

That single candidate becomes the **entire** expansion for the node
(`expansion.py:721-724`). A node that should have fanned into a multi-step subtree collapses to
one leaf, and nothing downstream flags the branch as structurally starved relative to its
siblings.

REPORTED context that makes this concrete: `expansion.py:185-200` documents live telemetry
(qwen2.5:0.5b) where 6 of 8 valid completions returned the wrong JSON shape, so the root's
fallback emitted a search whose query was the mandate's first 100 characters — an instruction
preamble, not an entity — and the run made zero page visits.

**PARTIALLY ADDRESSED (instrumentation only).** Every fallback branch now stamps
``DetailKey.FALLBACK_EXPANSION`` on the candidate it emits, the collapse is logged at WARNING,
and the final payload carries ``degenerate_fallback_count`` when any fired (absent otherwise, so
a healthy run's payload shape is unchanged). The tag is now also the trigger for F6's narrow
re-planning MVP: with ``got_reexpand_fallback_nodes_enabled`` on, a parent whose whole
expansion collapsed to one tagged leaf is re-planned once. See F6 for its bounds and its
residual finalize-context gap. (``expansion_degenerate_fallback_test.py``,
``reexpand_fallback_parent_test.py``.)

### F8. Shape classification happens before any information exists, and changes nothing — VERIFIED

`classify_shape` (`shape_classifier.py:103-120`) keyword-matches the **raw mandate string**
before any search or visit has run. Its only effect is `_auto_reasoning_rules`
(`expansion.py:502-533`) prepending a markdown rules block to the system prompt. It creates no
edges, sets no branching, sets no `execute_all_children`, and does not touch `graph.expand`.

The module's own docstring (`shape_classifier.py:24-31`) concedes `chain`/`parallel_merge` are
"unmeasured beyond the two/three examples each was written against." REPORTED measured recall
(Q1): **19% chain, 4% parallel_merge, 70% return `None`, 1/10 on hand-verified chains.**

**Outcome:** a soft prose nudge the LLM may ignore, derived from a keyword guess about
structure the system has no grounding to make yet.

### F9. Template short-circuit can hijack an arbitrary subtree — VERIFIED

`_plan_library_auto_shortcircuit` (`idea_engine.py:1013-1043`) is invoked from the generic
`_handle_expansion_node` for whatever node is being stepped, and `plan_library_search.py:96`
builds `query_text` with `is_root=(node.node_id == graph.root_id())` — so **non-root nodes can
trigger template substitution** from their own local title/goal text, text that was itself
produced by an earlier context-blind call (F1).

**Outcome:** a subtree's shape can be swapped from "what the LLM would plan given the real
ancestor path" to "what a canned template produces," on the strength of a text snippet never
validated against the whole task's information needs.

**RESOLVED (scope guard, unconditional).** `retrieval.AUTO_APPLY_THRESHOLD` (0.50) is calibrated
over ROOT queries only — the eval set's positives are whole task statements — so a non-root node
on the AUTOMATIC path must now clear `NON_ROOT_AUTO_APPLY_THRESHOLD` (0.54, the weakest correct
positive on that eval set) instead. Between the two bars the match degrades to `suggest`, i.e.
organic expansion. Non-root matching stays available, at a higher bar; the on-demand action keeps
the calibrated bar, since there the model asked for a strategy for that node.
(`plan_library_auto_shortcircuit_test.py`, `plan_library_search_action_test.py`.)

### F35. F33's contract VETO silences the one signal that notices an unfinished chain — VERIFIED

Live case, run `chainfix0820_nano_r1`, task 136 (Clifton Suspension Bridge -> Brunel ->
SS Great Eastern length), `graph`/`good_adaptive` on gpt-4.1-nano, score **0.183** against
`sequential_react`'s 0.75 on the identical mandate. The whole run is **two nodes** and six
decisions:

```
expansion  root  -> "1 sub-problems"           (n_candidates: 1)
evaluation leaf  -> "Identify Victorian engineer of Clifton Suspension Bridge"
selection  leaf  -> same,  action visit
action     leaf  -> visit en.wikipedia.org/wiki/Clifton_Suspension_Bridge  success
grounding  root  -> "grounded"                 (distinct_visits: 1)
finalize   root  -> "finalized"                (grounded: true, goal_achieved: FALSE)
```

Four separate mechanisms had to decline before that could happen, and only one of them is
broken:

1. **Expansion** returned a single candidate covering hop 1 of a mandate that spells out
   three hops (F1: greedy, no global plan). No `FALLBACK_EXPANSION` tag — a genuine
   single candidate, not F7's parse collapse.
2. **The step-confidence judge got it right.** It scored the leaf **0.20** and said so:
   *"it does not specify the name of the enormous iron steamship designed by Brunel or its
   length."* Below the 0.5 threshold, so `_maybe_confidence_reexpand_batch` would have
   re-expanded the leaf into hop 2.
3. **F33's contract veto discarded that score.** `_confidence_triggers_reexpand`
   (`idea_engine.py`) refuses to act on a low judge score when `evaluate_step_contract`
   reports the leaf's contract SATISFIED. Replaying the stored node reproduces it exactly:
   `ContractSatisfaction(applicable=True, satisfied=True, missing=[])`.
4. **The follow-up detector** (`GoTOperations.check_needs_followup`) then answered
   `needs_followup: false`. It DOES receive the mandate (first 1500 chars, so the full
   HOP 1/2/3 text) and a 3000-char page excerpt that does contain "Brunel" — this one is a
   plain nano judgement failure, not a missing-context bug.

**The mechanism.** The contract text is the leaf's explicit `expect` detail when the
expect-contract lever wrote one, **else the leaf's own goal**. That lever is off in every
shipped arm, so in practice every contract is goal-derived. A goal like *"Identify Victorian
engineer of Clifton Suspension Bridge"* names no measurable datum, so the check reduces to
`_subject_present`: does the fetched page mention `suspension`/`victorian`/`clifton`. It does.
"Satisfied" here means **"I opened a page matching the words of my own goal"** — which is true
of every intermediate hop of a chain that is nonetheless nowhere near finished. F33 was
derived from the judge's anti-calibration as a *quality* estimator; the veto it installed also
suppresses the judge as a *progress* signal, and chains are exactly where those diverge.

**Blast radius, measured offline** (replay of the 400 most recent stored `graph` run JSONs;
168 of them carry judge scores): of **251** low-confidence completed visit leaves, F33 vetoes
**171 (68%)**, and **134 of those vetoes (78%) rest on a subject-only contract**. Across all
applicable visit leaves, 484/566 subject-only contracts come back satisfied.

**What was NOT the cause.** The grounding gate (`_grounding_replan` ->
`evaluate_grounding`) is deterministic, LLM-free, and has no chain awareness whatsoever: for a
"do not guess" mandate it asks only `len(successful_visit_urls) >= 1`. It fired
`chosen: 'grounded'` correctly *by its own contract*. It is a floor ("did this run open any
page at all"), never a completion test, and nothing upstream gives it an expected-hop count.
The engine holds no representation of "this mandate implies N more hops" anywhere:
`MandateRequirements` has no hop/waypoint count, `waypoint.py` extracts a value from a
completed hop but predicts none, and F8's shape classifier is advisory-only at 19% chain
recall. Fixing the grounding gate would require that machinery and is **out of scope**.

**FIX (opt-in, default OFF): `got_contract_veto_requires_datum_enabled`.** A satisfied verdict
now carries `datum_verified` — True only when the contract asked for a measurable datum AND a
number was found beside that datum's wording. With the flag on, only a datum-verified contract
may veto the judge; a subject-only one hands the decision back to the judge, i.e. pre-F33
behaviour for that subset. F33's actual finding is preserved: a leaf that DID deliver its
datum is still protected from an anti-calibrated low score, and the contract trigger's positive
half (unsatisfied -> re-expand) is untouched. Flagged rather than unconditional because the
veto currently suppresses 68% of confidence-trigger firings — flipping that on `good_adaptive`
is a live-measurable cost/accuracy change, not a bug fix. Bounded as always by
`reexpand_max_iterations`. Env toggle for A/B:
`IDEA_TEST_GOT_CONTRACT_VETO_REQUIRES_DATUM=1`. (`contract_reexpand_test.py` F35 section —
replays this exact leaf/mandate and pins flag-off byte-identity.)

**Still open after this fix** (task 136 needs both to land): expansion's one-candidate plan for
a three-hop mandate (F1), and the follow-up detector saying "no" with the hop-2 instruction
in its own prompt.

---

## PART 2 — Ranked findings

| # | Finding | Severity | Cost to fix |
|---|---|---|---|
| PART 0 | Graph is a tree; no `depends_on` in the schema; multi-parent API dead | **Structural** | Large |
| F1 | No global plan; greedy per-node, sees only ancestors | **Structural** | Large |
| F5 | Correct dependency builder exists but only for templates | High | Medium |
| F3 | Dependency edges invented by URL-sniffing, visit-only | High | Medium |
| F6 | One-shot formation; no re-plan path for expanded nodes | High | Medium — PARTIAL: degenerate parents only |
| F7 | Malformed plan collapses to one degenerate node, unflagged | High | **Low** — PARTIAL: now flagged |
| F35 | F33's contract veto silences the judge on unfinished chains | High | **Low** — fixed behind a flag |
| F2 | Depth never chosen, tracked, or capped | Medium | Low |
| F9 | Template short-circuit can hijack non-root subtrees | Medium | Low — RESOLVED |
| F4 | `execute_all_children` declared before children exist | Medium | Medium |
| F8 | Shape classifier: pre-information, advisory-only, 4-19% recall | Low (inert) | Low |

---

## PART 3 — What this implies for the score gap

Reading the three reviews together, the native engine's chain deficit has a **structural**
explanation that does not require any of the tuning hypotheses:

1. The graph cannot express a value dependency (PART 0) — no fan-in, no `depends_on`.
2. So hop *n+1* cannot receive hop *n*'s value through an edge.
3. So the value must be recovered from concatenated raw page text at finalize — where **no
   extraction step exists** (Q32, REPORTED).
4. And finalize reads the *truncated* `content` field anyway (`ENGINE_DESIGN_REVIEW.md` D1).

Each step is independently verified; the chain between them is a **HYPOTHESIS**, but a
falsifiable one: if it holds, adding a typed per-hop value channel plus a real dependency edge
should move chain scores substantially, while prompt and scheduling work should not. Q12
already showed scheduling work does not — the deficit relocated rather than closing, which is
what this hypothesis predicts.

**Cheapest discriminating test:** count, offline from stored run JSONs, how often a chain task's
hop *n+1* node's prompt actually contains hop *n*'s extracted value. No live spend; the run
corpus already exists.

**Note on an existing design doc:** `docs/handoffs/RESOLVED_VALUE_CHANNEL_DESIGN_2026-08-16.md`
appears to address exactly this. It should be read before anyone designs a fix — this review
did not evaluate it, and it may already contain the intended solution.

---

## Open questions

- **Was the tree-vs-DAG limitation deliberate?** `merge_nodes` exists, works, and is
  unit-tested. Someone built multi-parent support and it was never wired in. Git history for
  `idea_dag.py` would say whether this was a deferred plan or an abandoned one.
- **Does `requires_data` still have the ancestor-vs-sibling scope mismatch?** Recorded in
  project memory; not re-verified here.
- **Would a global plan step help the native path, or just recreate Compiled v1?** If the
  answer is "recreate Compiled v1," the honest conclusion may be that the native path's value is
  adaptivity, and it should lean into re-planning (F6) rather than up-front planning (F1).
