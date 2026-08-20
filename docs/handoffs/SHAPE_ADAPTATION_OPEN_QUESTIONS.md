# Open questions: recursive shape adaptation in DAG v2 (2026-08-15)

Companion to `SHAPE_ADAPTATION_HANDOFF_2026-08-15.md` (the diagnosis) and
`CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` (the benchmark evidence).

**Target architecture.** A node detects the shape of its own sub-problem and picks an execution
strategy, recursively, at any level. Nodes differ by how high-level they are but most retain the
same capabilities. Overhead must stay bounded, which implies automated detection rather than
per-node LLM deliberation.

**How these are meant to be answered.** With *evidence* — a paper's finding, a real system's
implementation choice, or an experiment run here — not with recommendations. Adversarial evidence
(results showing an approach failed, or was tried and reverted) is as valuable as supporting
evidence. Answers should carry their methodology.

> ⚠️ **Citation discipline.** Below, only literature *areas* and *named systems* are given — never
> specific titles, authors or numbers. This repo's memory records a ~80% rate of
> real-citation-with-fabricated-payload, plus a second mode of real quotes taken from withdrawn
> drafts. Treat every pointer as a search term to verify against a primary source.

---

## §A — Shape detection: calibrating machinery that already exists

You have two detectors and one consumer. Detection is already free and deterministic; the open
questions are accuracy, coverage, and why the loop is open. **Q1 and Q2 were answered this session
— see §A results below.**

**Q1. What is the confusion matrix of `classify_shape` against the suite's real structure?**
*Answered 2026-08-15:* recall 19% on chain, 4% on parallel_merge, 38% on branch_eliminate; returns
`None` for 70% of tasks; **1/10 on the hand-verified live chain set**. Method: ground truth from
docstring headers over 157 tasks, prediction from `classify_shape(get_task_statement())`.

**Q2. Why does `_detect_state_dependencies` miss most chains, and what is it keying on?**
*Answered 2026-08-15:* it fires only on (A) `search` sibling + `visit` sibling with a missing URL —
a *tooling* dependency — or (B) a model-emitted `requires_data` dict naming a sibling. **B fired
0 times in 476 cell logs; the string never appears at all.** Method: source read plus log census.

**Q3. Is dependency better inferred from the emitted plan than predicted from the mandate?**
A plan where candidate B's query text contains a slot only fillable by A's output is a detectable
data dependency — dataflow analysis, not semantic judgement. Condition B's design asks the *model*
to declare it and gets nothing.
*Evidence:* def-use/dataflow analysis from compilers; workflow engines (Airflow, Dagster, Prefect)
resolve DAG edges from declared inputs/outputs rather than prose. Experiment: hand-label
dependencies on 20 chain plans, test whether a textual slot-detector recovers them.

**Q4. Does shape hold for a whole task, or change per subtree?**
Recursion implies a task can be `parallel_merge` at the top and `chain` inside one branch.
`classify_shape` takes only the root mandate and has no per-node notion.
*Evidence:* experiment — run `classify_shape` on each expanded node's sub-goal text and check
whether labels differ within a task and match the subtree's real structure.

**Q5. What do off-the-shelf frameworks do here, and what did it cost them?**
LangGraph, CrewAI, AutoGen, OpenAI's Agents SDK and orchestrator-worker patterns mostly commit to
*developer-declared* topology rather than inferred topology. That choice is itself evidence.
*Evidence:* real-world code — how each expresses conditional edges/handoffs, and whether any infers
structure at runtime. Adversarial: frameworks that tried dynamic topology and reverted.

---

## §B — Recursion and the node contract

The engine today is a **flat graph with depth-first iteration** — no nested sub-solver, no subgraph
abstraction, no per-node budget. Recursion is a real architectural addition.
**Everything here is blocked on §A:** a recursive node has nothing reliable to route on.

**Q6. What is the minimal uniform node contract that permits recursion?**
A node needs a sub-goal, a budget, an evidence handle, a return contract, a termination signal.
`agent/app/solver.py` already exists as a pluggable-strategy seam; the question is whether a *node*
can host a `Solver`.
*Evidence:* codebase audit against the `Solver` protocol.

**Q7. What terminates recursion, and what evidence is there that any criterion converges?**
Candidates: budget, an atomicity test ("is this one tool call?"), confidence, depth cap.
*Evidence:* HTN/hierarchical-planning literature on decomposition termination; adaptive-computation-
time and early-exit-network literature on learned halting. Experiment: how often does an atomicity
test agree with what the engine actually does?

**Q8. Does decomposition depth help, or compound error?**
At 80% per-subgoal success, depth 3 is 51%. The thesis assumes structure fights compounding
hallucination; depth may amplify it.
*Evidence:* least-to-most / plan-and-solve / decomposition-prompting literature — specifically hunt
for papers reporting *negative* depth results. Experiment: score by realized graph depth,
controlling for task difficulty.

**Q9. Should node capability vary by level, and on what axis?**
Axes available: model tier, tool access, budget, prompt, verification strictness.
*Evidence:* model-cascade and router literature; real-world orchestrator-worker patterns where the
planner is strong and workers are cheap. Adversarial: evidence a uniform flat loop matched a tiered
one.

**Q10. How do you memoize subtrees so recursion doesn't re-derive the same sub-goal?**
Working memory is already keyed by mandate hash (`mem_<sha256(mandate)>`).
*Evidence:* experiment — measure near-duplicate sub-goal rate using embeddings Chroma already
computes. If low, memoization isn't worth building.

**Q11. Global scheduler or locally-greedy nodes?**
A local-greedy node can't know a sibling already found the answer; a global scheduler costs
coordination.
*Evidence:* blackboard-architecture literature; how Dagster/Temporal handle dynamic task expansion
centrally versus actor systems locally. Experiment: count runs where early global termination would
have saved spend.

---

## §C — Sequential execution, where the loss actually is

Bare graph loses to a linear loop **6W/10T/13L on chains**, level elsewhere. Re-expansion recovers
+0.123 on chains vs +0.023 on parallel — it reads as a repair for batching, not added capability.

**Q12. Is the chain deficit caused by `auto_parallel_siblings`?**
*Answered 2026-08-15 (run-complete, 54/54 cells):* **partly.** Paired n=36, `chain_coverage`
+0.079 (9W/24T/3L), with **zero losses** in the baseline arm. Paired against `seq_react` on chains
(n=17), turning it off flips adaptive from a tie (0.510 vs 0.493) to a **win** (0.608 vs 0.493,
7W/8T/2L). **But `overall_score` still trails linear** (0.452 vs 0.464): the engine traverses more
of the chain without converting that into a better answer, which relocates the remaining deficit to
**extraction/finalize** rather than scheduling. The bare scaffold still loses to linear with the
flag off (2W/6L). Do not flip the default globally — measured on 9 chain tasks only, and the flag
exists to parallelize fan-out, which was not tested. A shape-conditional setting is what the
evidence supports, and §A is why that is not yet possible.
*Method:* primary metric `chain_coverage` (validator field, waypoints traversed), paired by
(model, task, arm), run-complete.

**Q13. When batching is off, does a later sibling actually *see* the earlier one's result?**
Serializing is necessary but not sufficient — the later node must receive hop 1's output.
*Evidence:* experiment — on a cell where `Forcing sequential` fired, dump node 2's assembled prompt
and check whether hop 1's extracted value literally appears.

**Q14. What is the right granularity of a "step" — one tool call, one hop, one sub-goal?**
ReAct's unit is one tool call; the engine's is a node that may bundle several. Granularity sets how
often the model can course-correct. Measured: `seq_react` 3.1 visits at 21k prompt tokens vs
`good_adaptive` 3.3 at 88k — same evidence, 4× the context.
*Evidence:* ReAct / Reflexion / self-refine line on iteration granularity; real-world code comparing
LangGraph's per-tool-call loop to a node-per-step model.

**Q15. On a chain, how is a wrong hop detected before the rest is wasted?**
134's documented traps are stop-early and over-hop; a wrong hop 1 makes hops 2–3 confidently wrong.
Backtracking is confirmed inert (0/476).
*Evidence:* self-verification / self-consistency literature; tree-search-for-agents (LATS-style
value functions over action trees) on when to prune. Adversarial: results where self-verification
fails on exactly this kind of factual chain.

**Q16. Can a chain be speculatively parallelized — guessing hop N+1 while hop N resolves?**
Trades tokens for wall-clock, and sometimes surfaces more candidates.
*Evidence:* speculative-decoding literature as mechanism analogy; whether any agent framework
speculates on tool calls. Experiment: how often is hop N+1's target guessable from hop N's page?

**Q17. Do chain and parallel shapes need different *verification*, not just different scheduling?**
A chain needs "is this the right hop?"; fan-out needs "did I cover every entity?". The suite already
encodes the second as coverage validators, and the candidate-coverage gate was built for this and
never paid off.
*Evidence:* process- vs outcome-supervision literature on where verification signal should attach.

---

## §D — Overhead and the detection budget

Two independently-written agents both show context-spend buying nothing — that looks like a property
of the regime, not of this engine.

**Q18. What is the per-node cost of the planning machinery, by stage?**
*Answered 2026-08-15:* 8.6 → 22.7 decisions/cell and 23,317 → 68,148 prompt tokens from baseline to
`good_adaptive`. Added: action +4.9, reexpand +3.2, selection +2.5, enforce +2.3, expansion +2.2,
evaluation +1.5; `grounding` and `finalize` flat. **This is the budget a shape-router must fit
inside.**

**Q19. Above what cost can detection not pay for itself?**
`classify_shape` is free, so the question is whether a free detector can be made accurate enough
(Q1 says not yet) or whether accuracy requires a paid one.
*Evidence:* compute break-even directly from Q1 accuracy and Q18 costs.

**Q20. Which recorded-inert mechanisms are genuinely dead, and what do they cost while inert?**
*Partly answered 2026-08-15:* `backtrack`, plan-library retrieval, early-exit and narrative
exemplars are confirmed inert (0/476 logs). The confidence-triggered *re-grounding* is **not**
inert — 580 judgements, mean 0.664, 32.8% below the 0.5 threshold. Whether that signal is
*informative* (the recorded AUC 0.571) needs per-step ground truth and remains open. The
candidate-coverage gate remains untested.

**Q21. Can routing be amortized — learned offline rather than computed per-node?**
Compiled v1 already showed an expensive model can author a plan offline for a cheap executor
(nano at 0.96 of reference, 1/42 cost).
*Evidence:* in-repo Compiled v1 results are the strongest local evidence; router/cascade literature
is the external analogue. Adversarial: Compiled v1's known breadth wrong-grounding gap.

**Q22. Does prompt caching change the economics enough to make per-node planning affordable?**
At 34:1 in:out, a recursive design multiplies prompt re-sending, and nothing currently caches.
*Evidence:* provider docs on cache-hit semantics and minimum prefix lengths (verify — these change).
Experiment: same tasks, caching on/off, $/cell at equal score.

---

## §E — Context and state hand-off

**Q23. Why does additional context buy nothing, in two independently-built agents?**
`langgraph@60` spent 4.9× `langgraph@25`'s prompt tokens for +0.016; `good_adaptive` spends 3×
`seq_react` for parity. The most surprising result of the sweep, and not architecture-specific.
*Evidence:* long-context degradation literature (lost-in-the-middle / position effects / context-rot
style findings — verify current versions, the area moves fast). Experiment: bucket cells by prompt
length and plot score; flat or declining implies retrieval failure, not budget.

**Q24. What representation should cross a node boundary — raw text, extracted values, or a typed record?**
Merges pass large text blobs today. A chain hop arguably needs one extracted value, not a page.
*Evidence:* structured-scratchpad / typed-state work; how DSPy signatures or typed workflow states
constrain what crosses a boundary. Experiment: measure how much of a node's inbound prompt is
actually referenced in its output.

**Q25. Does local context truncation explain the weak-model local results?**
`OLLAMA_CONTEXT_LENGTH=16384` while merge/finalize prompts can exceed it; the compose file documents
a prior silent-truncation incident that "was silently corrupting every result before any
model-capability conclusion was valid."
*Evidence:* experiment — rerun `qwen2.5:7b` at 32768 and compare. Free.

**Q26. Should evidence be re-retrieved per node rather than carried forward?**
Carrying is O(depth × size); retrieving is O(query). Chroma per-run memory already exists.
*Evidence:* RAG-vs-long-context comparison literature (actively contested — good adversarial
material both directions). Experiment: measure how often retrieval currently returns anything useful
(early steps log "Retrieved 0 internal thoughts, 0 observations").

---

## §F — Node heterogeneity and reliability

**Q27. Would constrained decoding at the action-selection point fix weak-model tool use?**
The sharpest irony in the data: LangGraph's schema-enforced function calling drove `qwen2.5:0.5b` to
24 visits, while free-text JSON left weak models emitting unparseable plans and doing **zero**
visits. Separately, a malformed `details` field crashed `_parse_candidates` and destroyed whole
expansion steps (fixed 2026-08-15).
*Evidence:* constrained-decoding / grammar-guided generation (GBNF, outlines-style regex/CFG
constraints, JSON-schema modes); ollama and vLLM both expose structured-output modes. Experiment:
constrain only the action decision, measure visits on weak local models. Adversarial: findings that
constrained decoding degrades reasoning relative to free generation.

**Q28. One tool contract for all nodes, or a different action space by level?**
The alternative is a planner that can only emit sub-goals and a leaf that can only call tools,
shrinking each node's failure surface.
*Evidence:* orchestrator-worker patterns where the planner has no tool access. Experiment: how often
does a high-level node emit a direct tool action rather than a decomposition, and do those cells
score differently?

---

## §G — How you would know

**Q29. What benchmark would discriminate shape adaptation, given the current suite cannot?**
Measured: 56/59 active tasks at difficulty ≥8 (weak models floor), ~44/59 carry fan-out against ~10
pure chains, keystone gates make scores near-binary, and the graph's real best case — parallel
fan-out with a join — has **9 paired cells** behind it.
*Evidence:* experiment — build a shape-balanced, difficulty-laddered set, using Q1's confusion
matrix to guarantee the labels are real rather than assumed.

**Q30. What result would falsify the recursive-shape-adaptation thesis?**
Candidate falsifiers: a correctly-scheduled chain still loses to a linear loop; per-node detection
costs more than it returns at every accuracy level; parallel shapes show no graph advantage even
when properly powered; a flat linear loop with the same adaptive mechanisms matches the recursive
design at lower cost.
*This is a decision to record, not research.* Its absence is what let this session's chain result be
misread for several hours.

---

## §H — Gap pass after the Q12 dissociation, 2026-08-16

A literature triage for §C/§E arrived alongside Q12's answer (traversal ↑, `overall_score` flat —
the deficit relocated to extraction/finalize). Fetched sources landed in `docs/research/web/`
(the six papers already sat in `/mnt/arteta/webrag`). This section is (1) what remained
literature-uncovered in §C/§E after that fetch, and (2) new questions the fetch pass produced by
reading `agent/app/idea_finalize.py` against the sources rather than in the abstract — each new
question below carries the file:line it's grounded in.

**Coverage gap, flagged not answered.** Q15 (wrong-hop detection before the rest of a chain is
wasted — self-verification / self-consistency / tree-search-value-function literature) received no
source in this pass and sits squarely in §C, not recursion. It is now the leading *mechanism*
candidate for the Q12 dissociation: a subtly wrong hop-1 value would let the engine "traverse" more
of the chain (the `chain_coverage` validator counts waypoints visited, not values verified) while
still landing on a wrong final answer. Untouched — still open, still worth a source pass.
Q16 (speculative parallelization) and Q22 (prompt-caching economics) also remain uncovered but are
lower priority: Q16 is a wall-clock/cost question, not a correctness one; Q22 is budget-adjacent
(see Q35 below for where it resurfaces).

**Q31. Does `MERGED_RESULTS` ever get populated on a pure chain, or does finalize always fall onto
the leaf fallback regardless of `auto_parallel_siblings`?**
`agent/app/idea_policies/merge.py:189` is the only writer of `DetailKey.MERGED_RESULTS`, and it
runs from a merge-node handler — the fan-in step a pure chain (no sibling fan-out) likely never
grows. `idea_finalize.py:801` falls back to `_collect_leaf_results_fallback` whenever `merged` is
empty. If chains never produce a merge node, Q12's flag changed *execution* (whether hop 2 sees hop
1's answer while running) without changing *what finalize is handed* — the extraction gap would sit
entirely downstream of scheduling, which narrows where to look next.
*Evidence:* code audit (done, this pass) plus one instrumentation check — log whether `merged` is
non-empty on the `csnopar_g` chain cells from the Q12 run.

**Q32. Does anything in the pipeline reduce a node's result to a single typed "answer to this hop"
value before finalize, or does every downstream consumer only ever see raw lists/text?**
Grepped `extracted_value` / `extracted_answer` / `keystone_value` / `hop_answer` across
`idea_policies/*.py`, `idea_finalize.py`, `idea_engine.py`: **zero hits.** `_build_node_summary_table`
(`idea_finalize.py:352`) reduces a VISIT node to `"visited {url} ({content_len} chars) title=..."` —
never the page's content. The actual text only reaches the model through
`_collect_all_visit_content` (`idea_finalize.py:78`), which concatenates every successful visit's raw
page text (up to 15k chars each, 80k total) with no per-hop label beyond URL/title. There is no
extraction step anywhere; extraction and synthesis happen in one finalize call over concatenated raw
text. This is the sharpest concrete instance of Q24 ("raw text blob, not typed extracted value") —
not a hypothesis, a confirmed absence.
*Evidence:* code audit (done, this pass). Directly actionable: this is what "dump a `graph_OFF`
cell's finalize input" (the handoff's next step) will show once run live.

**Q33. When the correct hop value is present in `visit_content` but the answer is still wrong, is
that a faithfulness/attribution failure rather than a position effect?**
Lost in the Middle (already fetched, `/mnt/arteta/webrag/2024.tacl-1.9.pdf`) is about *where* in the
context correct information sits, not whether the model *uses* correct information it's given
regardless of position. Those are different failure modes with different fixes (reordering context
vs. constraining/verifying generation against evidence).
*Evidence:* RAG faithfulness / attribution / groundedness-evaluation literature — a distinct body
from position-effect work; search terms only, verify against primary sources per this doc's citation
discipline. Experiment: on `graph_OFF` chain cells where the keystone value is confirmed present in
`visit_content`, check whether the finalize answer is entailed by that value (attribution check) vs.
simply wrong regardless of what's in context.

**Q34. Does a distractor-count effect explain more than Lost-in-the-Middle's position framing,
given `_collect_all_visit_content` concatenates every successful visit — chain-relevant or not —
with no relevance ranking?**
A chain task that took a wrong turn and visited extra pages before finding the right one gets all of
those extra pages concatenated into finalize's context alongside the correct one, un-ranked. That's
closer to a multi-document-QA distractor setup than a pure position-effect setup.
*Evidence:* multi-document QA / noisy-retrieval distractor literature (distinct from Lost in the
Middle's positional manipulation methodology — see source README for how to reuse that methodology
directly). Experiment: hold position fixed, vary the count of irrelevant visited pages merged in,
measure keystone extraction. Cheap to run against existing chain-cell logs.

**Q35. Would an explicit per-hop extraction pass beat single-shot long-context finalize reading —
and does it risk losing exactly what Q32 found is currently at least present, even if unused?**
Given Q32's finding (no extraction step exists today), the natural fix is a small "what value does
this page answer for this hop" call inserted before finalize, producing the typed record Q24 already
asked for. But an extra pass is an extra lossy hop.
*Evidence:* retrieve-then-read / extract-then-synthesize architectures and map-reduce/refine-style
summarization chains as the real-world analogue (`claude-code-subagents.md`'s "reduced return
artifact vs. raw working context" distinction is the closest fetched source, though it's framed for
context isolation, not extraction accuracy). Adversarial, explicitly: look for evidence that
extractive pre-passes lose recall relative to single-shot long-context reading — this is not a free
win, and Q22's caching economics cut the other way (a smaller typed payload may drop below a
provider's minimum cacheable-prefix length, while today's blob-heavy prompt is large enough to
benefit from caching as-is). Experiment: same chain cells, extraction-pass on/off, compare keystone
accuracy and $/cell.

---

## §I — Found this pass: `requires_data` Condition B, corrected 2026-08-16 (adversarial review)

Every prior pass (Q2, Q3, item 3 of the handoff's "what to do next") described Condition B's failure
as "the model doesn't emit `requires_data`." A full trace of every writer this pass found a sharper
cause for ONE of the writers: **the URL-backfill writer that's always in the expansion code path
writes a value Condition B can never match, by construction.**

`agent/app/idea_policies/expansion.py:1396-1424` (runs on every VISIT candidate missing a URL) sets
`requires_data.source_node_id` by walking `graph.path_to_root(parent_node_id)` — ancestors of the
node's *parent*. It explicitly excludes the self-reference case (`source_node_id == parent_node_id`)
as a documented deadlock-avoidance fix. `idea_sequencing.py`'s Condition B then checks
`source_node_id in candidate_ids`, where `candidate_ids` is the current sibling batch
(`idea_engine.py:1561-1570`). An ancestor of the parent can never be a member of the parent's own
children — so this writer's output fails Condition B's check on every invocation, independent of task
shape, model quality, or prompt wording. That much held up.

> ⚠️ **Correction, same day.** The paragraph originally here also claimed the two OTHER writers
> (`post_expansion_hooks.py`'s mandate-phrase enforcement, `plan_library.py`'s blueprint-declared
> edges) were "gated behind narrow triggers confirmed rare-to-inert," concluding Condition B was
> "structurally dead, not just unused." **That conclusion does not hold.** A dev-cycle adversarial
> review (`SHAPE_ADAPTATION_HANDOFF_2026-08-15.md` §8) mined the full `agent/idea_test_results/`
> corpus (1620 files, not the 476-cell single-session sample the original "0/476" figure came from)
> and found **183 real historical firings** of Condition B via `plan_library.py`'s sibling-scoped
> `requires_data` — genuine same-parent siblings, `type="urls_from_search"`, 179/183 with the source
> node already `done`. The "0/476" figure was real but scoped to a session that never exercised
> `plan_library`, not a property of the codebase. Condition B works when a task goes through
> `plan_library`'s typed dependency declarations; it is dead only for organic, LLM-authored candidates
> outside that path (and outside the narrow `post_expansion_hooks` enforcement case). This narrows the
> real gap and invalidated a heuristic fix (Q37, a same-batch text-cue detector) that was designed
> against the wrong premise and had a demonstrated false-positive mechanism against `parallel_merge`
> tasks — see the handoff's §8 for what shipped instead (test coverage locking in both the working and
> non-firing cases) and what's still open (re-run Q1's confusion-matrix methodology against the full
> corpus before designing a new detector for what's left).

---

## §J — Fixing detection inside the existing free-detection budget

Both detectors (`classify_shape`, `detect_state_dependencies`) are pure string/structural checks —
no LLM call, no extra graph traversal, effectively \$0 marginal cost today (confirmed by code audit:
neither function performs I/O or model inference). The overhead budget this doc's target architecture
worries about is spent entirely in the *surrounding* adaptive machinery (re-expansion triggers,
confidence judging — see §L). That means detection-accuracy fixes have real headroom before touching
the overhead budget at all.

**Q36. Does widening Condition B to check ancestor membership (`source_node_id` anywhere in
`path_to_root` of any candidate in the batch), not just sibling membership, correctly recover real
dependencies without over-triggering sequential mode on tasks where the ancestor reference is stale or
merely informational?**
Given §I's finding, this is a one-line change to an existing free check, not a new mechanism.
*Evidence:* needs a false-positive-rate check against the existing 476-cell log corpus — does
widening the check flip cells that were correctly batched into unnecessarily-sequential ones, and at
what cost to wall-clock/decisions-per-cell.

**Q37. Can a coreference/slot heuristic over each candidate's proposed query text — ordinals
("the first result"), pronouns ("it", "that company"), or explicit back-references ("from step N") —
recover genuine sibling-to-sibling dependencies at parse time, without an LLM call?**
This gives Q3's dataflow-inference idea a concrete zero-cost implementation, sitting in
`LlmExpansionPolicy._parse_candidates` where candidates are already being walked once per expansion.
*Evidence:* coreference-resolution heuristics from the compiler/NLP dataflow-analysis literature Q3
already named; needs a hand-labeled sample (the same 20 chain plans Q3 proposed) to measure recall
against §I's confirmed-broken alternative.

**Q38. If `classify_shape` were called per-node on each node's local sub-goal text instead of only
the root mandate (all four call sites currently pass root-level text), does its measured recall on
real per-subtree structure improve over the 1/10 root-level chain recall — and does that require
expanding its keyword vocabulary, given real chain tasks rarely use the literal phrases it currently
matches (`"dependency chain"`, etc.)?**
Sharpens Q4 into a two-variable experiment (call site, vocabulary) rather than a redesign question —
the classifier itself doesn't need to change shape, only where it's invoked and what it's tuned on.
*Evidence:* same methodology as Q1 (ground truth from docstring headers), applied per-node instead of
per-task; needs node-level ground truth labels that don't currently exist.

**Q39. Could the step-confidence judge's response schema (already one paid-for LLM call per
completed leaf on `good_adaptive`, confirmed alive at 580 firings) be extended with a second field —
"did this step's content depend on information only available from a specific sibling?" — producing a
retrospective, evidence-based dependency signal at zero marginal LLM-call cost?**
`classify_shape` and Condition B are both *predictive* (guess before execution); the confidence judge
is *retrospective* (observe what actually happened) and is the one mechanism §L of the code audit
confirmed genuinely adapts mid-run. Piggybacking a dependency question onto an already-budgeted call
sidesteps the "detection must stay free" constraint entirely, rather than satisfying it.
*Evidence:* none yet — this is a proposed schema change, not something run. Needs a comparison against
Q36/Q37's alternative on the same held-out chain set: does a retrospective signal actually correct
scheduling decisions before the run ends, given it fires *after* the leaf it's judging has already
completed?

**Q40. Would feeding Q39's retrospective signal (or repeated Condition-A tooling-dependency hits)
back into the `auto_parallel_siblings` gate for a subtree's still-unscheduled siblings — local,
within-run recalibration of the shape verdict — reduce the chain penalty without the global flag flip
Q12 already measured (which recovered `chain_coverage` but not `overall_score`)?**
Today a low-confidence leaf triggers re-expansion of *that* node only; it never changes how the engine
batches siblings it hasn't scheduled yet. This is the gap between "the graph can grow" (confirmed, via
re-expansion) and "the graph can revise its own scheduling policy mid-run" (not attempted anywhere in
the current code).
*Evidence:* none yet — proposed policy change building on Q39.

**Q41. `_RULES_NAMES` and `agent/app/reasoning_rules/` currently hold content for exactly one shape
(`branch_eliminate.md`); a correct `"chain"` or `"parallel_merge"` verdict from `classify_shape` is
computed and then silently discarded (`expansion.py:513-519` logs it and stops). Would authoring
`chain.md` and `parallel_merge.md` — e.g. a chain rule instructing the model to form each candidate as
a single next hop explicitly conditioned on the prior hop's stated result, rather than proposing
several independent-looking next steps — change candidate proposal quality on chain tasks, given this
costs zero extra LLM calls (same prompt-assembly path, marginally more prompt tokens)?**
*Evidence:* none yet. Cheapest experiment in this section to run — content authoring plus an A/B on
existing chain cells, no code change beyond removing the `_RULES_NAMES` restriction.

---

## §K — Is the LangGraph comparison testing the question it's meant to

**Q42. The current LangGraph arm (`agent/app/langgraph_solver.py:262`) calls
`langgraph.prebuilt.create_react_agent` — a flat two-node loop with no `add_conditional_edges`,
`Send`, or custom topology (confirmed: no such call appears anywhere in the file). Every result
comparing DAG v2 to LangGraph so far is therefore "adaptive graph vs. flat ReAct loop," not "shape
inferred automatically vs. shape declared by a developer." Would a second LangGraph arm that
hand-declares branch/chain topology per task via conditional edges — using the same task metadata
`classify_shape` is trying to infer — change what the comparison means?**
Q5 concluded LangGraph "commits to developer-declared topology" as a general framework property; this
pass confirms the specific integration here never exercises that capability at all, so Q5's conclusion
and this repo's LangGraph *results* are about two different things (the framework's design vs. this
harness's unused feature).
*Evidence:* code audit (done, this pass — no conditional-edge usage found). Needs a second arm built
to actually test the declared-vs-inferred question.

**Q43. LangGraph's superstep model batches every node reached by one conditional-edge routing
decision into the same parallel step; sequential dependency is expressed only by placing nodes in
different supersteps — a purely declarative analogue of what DAG v2's `auto_parallel_siblings` gate
tries to infer automatically. Would exposing a per-mandate manual override (a `sequential_hint` field
in task metadata, functionally a routing function written by the task author instead of inferred at
runtime) isolate how much of DAG v2's chain loss is inference error (fixable by §J) versus a cost
inherent to batching-by-default (only fixable by not batching by default)?**
*Evidence:* none yet. This is a diagnostic instrument (ablate away the detection problem entirely by
hand-labeling) rather than a proposed shipped feature — it tells you the ceiling §J's fixes could reach
even if detection were perfect.

---

## §L — Where DAG v2's edge would actually show up, beyond badmodels

The framing "DAG v2 loses everywhere except badmodels" is only as strong as the data behind it. Q29
already found the graph's presumed-strongest case (branch/parallel-merge shapes with real fan-out and
a join) has just 9 paired cells — under any reasonable power threshold. The questions below treat that
as the live strategic gap, not the chain loss (which is already well-characterized by Q1/Q2/§I/Q12).

**Q44. Given Q29's 9-cell count behind DAG v2's presumed-strongest case, is "loses in all fields
except badmodels" supported by adequately-powered data, or is the branch/parallel case simply
unmeasured rather than measured-and-losing?**
This is the strategic question the rest of §L serves. It was already named as "EV-0a" in the handoff's
priority list and, per this pass, is still not run.
*Evidence:* none yet beyond the existing 9-cell count. Needs the properly-powered graph-vs-linear run
on parallel/branch shapes specifically, matching Q12's paired-by-(model,task) methodology.

**Q45. The candidate-coverage gate (`idea_policies/candidate_coverage.py`) is the one piece of
verification infrastructure purpose-built for a specific shape (`branch_eliminate`), and
`branch_eliminate` is also the shape with the classifier's best recall (38%, vs. 19% chain / 4%
parallel_merge) and the only shape with reasoning-rules content today. Would a benchmark run
specifically on `branch_eliminate`-shaped disambiguation tasks — not pooled with chains or
parallel_merge, the way current results are — show a genuine DAG v2 edge, since this is the one shape
where every piece of existing purpose-built infrastructure actually lines up with the task's real
structure?**
*Evidence:* Q17/Q20 already confirmed the coverage gate is built and untested; none of the pooled
results to date isolate this shape.

**Q46. DAG v2's step-confidence re-grounding + re-expansion (§4 of the code audit; confirmed "alive,"
580 firings, 32.8% below threshold) is a recovery-from-a-bad-step mechanism the LangGraph flat-loop arm
structurally lacks (`create_react_agent` has no mid-run restructuring at all). Would a benchmark
category deliberately engineered to include a plausible-but-wrong early step (an ambiguous search
result easily mistaken for the right entity) — testing recovery specifically, orthogonal to task shape
— show a measurable DAG v2 edge that today's shape-pooled scoring dilutes?**
*Evidence:* none yet. `DAG_V2_PREFLIGHT_2026-08-15.md`'s single directional smoke (re-expansion
recovering a 3-hop chain from 0.283 to 0.767) is the only existing hint, at n=1.

**Q47. At matched \$ spend rather than matched architecture, does DAG v2 (any profile) ever beat a
bare frontier model or LangGraph at the same \$/task, on any shape?**
Repeatedly flagged ("score-per-\$", "bare model wins ~2:1 at every API tier on that framing") but never
directly computed as a same-cost comparison segmented by shape — it may differ sharply between chains
(where §I/§J's fixes are the lever) and branch/parallel shapes (where Q44's power gap is the lever).
*Evidence:* none yet. Needs the existing cost telemetry (already collected per Q18) re-sliced by shape
and re-plotted as score-vs-\$ rather than score-vs-arm.

---

## §M — The adaptability ceiling under a genuinely limited overhead budget

**Q48. Given detection itself costs ~\$0 (§J's premise) and the real overhead lives in the
re-expansion/confidence-judge machinery (+3.2 to +4.9 decisions/cell, Q18), is the ceiling on cheap
adaptability set by how much signal can be extracted from calls already happening (Q39's approach)
rather than by how many new checks get added?**
This reframes "high adaptability, limited overhead" as an information-extraction problem on an
existing budget, not a budget-expansion problem — the two are easy to conflate and lead to different
designs (Q39 vs. a hypothetical new per-node classifier call, which this doc does not propose anywhere
precisely because it would break the free-detection premise).
*Evidence:* none yet — a framing question that Q39/Q40's results would settle empirically.

**Q49. Given the coordinator-worker literature's "delegation has a floor cost" finding (splitting work
into narrower briefs can raise total cost, not lower it — `anthropic-cookbook-coordinate-specialist-team.md`)
and this project's own chain tasks averaging ~3 hops, is there a hop-count threshold below which any
additional per-node adaptive machinery is a net loss regardless of how cheap or accurate it becomes —
i.e. a structural argument for routing very-short chains to the flat sequential loop outright, rather
than making the graph engine cheap enough to compete on them?**
*Evidence:* the cited cookbook finding is adversarial evidence for adding *any* machinery to short
chains, independent of this doc's other proposed fixes; no in-repo experiment run yet varying hop-count
as the independent variable.

**Q50. If per-node shape classification (Q38) and local dependency signals (Q37/Q39) land, does the
`auto_parallel_siblings` gate's *scope* also need to change — from "all currently-eligible children in
the ready-set" to "children grouped by nearest branching ancestor" — so that a fan-out subtree and a
chain subtree inside the same task don't get batched together just because the scheduler currently
treats every ready child as one flat pool?**
A task can plausibly be `parallel_merge` at the root with a `chain` inside one branch (Q4's premise);
Q38 would let per-node classification see that, but nothing downstream currently groups siblings by
subtree — `idea_engine.py:1561-1570` builds `candidate_ids` from a node's direct children only, so this
may already be scoped correctly at the *direct-children* level and only breaks if a grandchild's shape
needs to override a parent batching decision made before it was expanded.
*Evidence:* none yet — needs Q38 to land first before this is testable; flagged here so the gate's
scope isn't assumed correct-by-default once per-node classification exists.
