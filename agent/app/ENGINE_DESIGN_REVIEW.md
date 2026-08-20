# Engine Design Review — where the native DAG loses score

**Written 2026-08-19. Adversarial review, documentation only — no code changed, no live spend.**

Companion to [`ASSUMPTION_AUDIT.md`](ASSUMPTION_AUDIT.md) (same cycle). That document covers
constants that were never validated. This one covers **design decisions that plausibly cost
score**, and answers a specific question: what stands between DAG v2 and LangGraph-level
accuracy.

Evidence labels used throughout:

- **VERIFIED** — I read the code and confirmed it during this review; `file:line` given.
- **MEASURED** — a number from a stored benchmark artifact; source doc cited.
- **REPORTED** — stated in an existing doc, not independently re-checked here.
- **HYPOTHESIS** — a mechanism consistent with the evidence that nobody has tested.

---

## PART 0 — The measured position (read this before planning anything)

The LangGraph comparison is **already answered quantitatively**, in
`docs/handoffs/CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` (~400–525 live cells, ~$1.35–1.77,
8 models x 10 tasks). The headline numbers (MEASURED, `:126-128`):

| arm | as deployed | restricted to models LangGraph can run |
|---|---|---|
| `graph:baseline` | 0.386 (n=68) | **0.498** (n=44) |
| `graph:good_adaptive` | 0.428 (n=65) | **0.560** (n=41) |
| `langgraph_react` | 0.323 (n=40) | **0.497** (n=26) |

**Restricted to a fair population, off-the-shelf LangGraph ties our unadapted scaffold to
three decimal places** (0.497 vs 0.498). The adaptive mechanisms add ~0.06 on top, at roughly
2.5x the cost.

The finding that should reframe the goal, though, is not about LangGraph at all
(MEASURED, `:184-189`):

| arm | prompt tok | completion tok | visits | score |
|---|---|---|---|---|
| **`seq_react`** (this repo's own linear ReAct) | 19,255 | 3,374 | 3.1 | **0.516** |
| `graph:good_adaptive` | 58,335 | 2,166 | 3.1 | 0.479 |
| `graph:baseline` | 22,962 | 1,238 | 1.3 | 0.432 |
| `langgraph@60` | 94,935 | 269 | 3.2 | 0.344 |

**Our own linear ReAct loop beats the graph engine at one third the prompt tokens.** And on
chains — the DAG's designed best case — `good_adaptive` 0.356 vs `seq_react` 0.360, 10W/7T/11L
(`:653-665`). The doc's own summary (`:779-781`): *"The mechanisms are the asset; the scaffold
is the liability."*

Two consequences for how to read the rest of this document:

1. **"Get closer to LangGraph" is the wrong target.** LangGraph is not ahead on accuracy where
   both run. The real gap is to `seq_react`, which is in-repo, cheaper, and ahead.
2. **The comparison must stay split.** LangGraph's `create_react_agent` cannot start **4 of 8
   models** (HTTP 400/404, `does not support tools`, reproduced on ollama *and* OpenRouter —
   `:13-20`, `:50-57`). On those models the accuracy comparison is undefined, and that
   population is where the central cheap-model thesis lives. Simplifying the scaffold toward
   ReAct to close gap #1 could silently cost #2. Never report one blended number.

**Fairness is not the explanation.** `langgraph_solver.py` (untracked, 348 lines) gives the
LangGraph arm the same `AgentIO.search`/`visit` tools, the same `search_k`/`page_chars`, the
same `_call_tool_with_retry` from `execution_sequential.py`, the same forced-synthesis
fallback, and the same telemetry shape. The step-budget asymmetry (25 vs 50–90) was tested
directly: raising LangGraph to 60 moved score **+0.016** while burning 4.9x the prompt tokens
(`:551-586`). The comparison is sound.

---

## PART 1 — Top diagnostic lead: the `qwen2.5:7b` anomaly

**HYPOTHESIS — untested, but the cheapest high-value experiment on this page.**

Amid ties everywhere else, one cell is a rout (MEASURED, `:141-143`):

> `qwen2.5:7b` — LangGraph **0.694** vs graph **0.150–0.158**

A 4.4x gap on a single model, against ties elsewhere, is **bug-shaped, not gradient-shaped**.
The results doc logs it as an unresolved open decision (`:421`). Four facts line up:

1. `connector_llm.py` sets **no `num_ctx`** and no ollama options at all — VERIFIED, zero
   hits repo-wide for `num_ctx`/`num_predict` in the engine's LLM connector.
2. The graph arm sends **~58k prompt tokens** vs LangGraph's ~19k — MEASURED, `:184-189`.
3. `qwen2.5:7b` had a truncation problem serious enough to require a **container-level**
   context-length workaround on the isolated `:11435` badmodel-ollama instance — REPORTED
   (project memory, promptbench cycle). That workaround lives in the environment, **not in
   this repo**, so the engine has neither control over it nor visibility into it.
4. ollama truncates silently when a prompt exceeds the served context window, and truncation
   drops the **head** — where the system prompt and task statement live.

**Mechanism:** the arm sending 3x the tokens hits the context ceiling first, and loses its
instructions rather than its evidence. Signature: catastrophic on one small-context local
model, invisible on API models with large windows. That is exactly the observed pattern.

**Why it matters beyond one cell.** If this holds, the 2.5x token premium is not only a cost
story — it is an **accuracy** story on small-context models, i.e. precisely the weak-model
population the project exists to serve. It would also mean the "availability, not accuracy"
headline is computed over a partly-contaminated cell.

**Test (cheap):** re-run that single cell with an explicitly large `num_ctx`, or instrument
the connector to log served context length and prompt length per call and check for a
crossover. One task set, one model, one arm.

**Note either way:** the engine having no `num_ctx` control is a real gap independent of
whether it explains this cell. A silent truncation ceiling that varies by deployment makes
every local-model number non-reproducible across machines.

---

## PART 2 — Score-losing design decisions (all VERIFIED this cycle)

### D1. Finalize reads the truncated page text while the full text sits unread beside it

`VisitLeafAction._visit_single_page` stores **both** `content` (pre-truncated to
`max_observation_chars = 100000`, `config.py:564`) and `content_full=cleaned` (untruncated) —
`actions.py:1311-1327`, `actions.py:1679`.

`_collect_all_visit_content`, which assembles the evidence the final answer is built from,
reads the truncated field (`idea_finalize.py:97`):

```python
content = ar.get("content", "") or ""
```

`content_full` is present in the same dict and is read elsewhere (`idea_finalize.py:244`,
`actions.py:1616` both do `content or content_full`), so this is a **field-choice bug, not an
availability problem**.

Compounding it, `_compact_action_result` (`idea_finalize.py:29-47`) drops `content_full`
outright with the comment *"Full visit content is captured separately in visit_content"* —
which is **factually wrong about the function 50 lines below it**, since `visit_content` is
built from the truncated `content`.

**Failure scenario:** any page longer than `max_observation_chars` is cut once at visit time
and no downstream stage — merge, finalize, or the reconcile passes — can ever recover the
remainder, though it was fetched and held in memory. **Acknowledged by:** nothing; no test.

### D2. `goal_achieved` is near-tautological at the root

`SimpleMergePolicy._validate_goal_achievement` (`merge.py:213-242`), VERIFIED in full:

```python
if isinstance(results, list) and len(results) > 0:
    has_relevant_content = True
    break
```

`results` is populated only by SEARCH actions and its length says nothing about relevance. So
**any search returning any hit marks the goal achieved.** `create_merge_node` runs this on the
**root** (`merge.py:80`) *before* merge-node synthesis executes, and
`idea_finalize.build_final_payload` reads `root.details[GOAL_ACHIEVED]` directly
(`idea_finalize.py:1038`), only consulting the LLM-judged merge-node verdicts if the root's is
False.

**The signal cannot distinguish "the goal was met" from "a search ran."**

### D3. The same function's other branch is near-unsatisfiable — one boolean, two opposite biases

`merge.py:230-236` requires the **entire goal string verbatim** inside page content or query:

```python
if content and original_goal.lower() in content.lower():
```

For a real mandate this essentially never fires. Net effect, VERIFIED: a SEARCH-containing
subtree is judged goal-achieved almost by construction (D2); a VISIT-only subtree is judged
goal-NOT-achieved almost by construction. **One boolean, two contradictory failure modes,
neither tracking task completion.**

Note the self-contradiction with the system's own prompt: `merge.py:101`'s
`merge_planning_addendum` explicitly warns the LLM against judging by "whether keywords
appear" — which is exactly what this validator does.

### D4. `merge_should_skip` is an irreversible lockout

`MergeLeafAction.execute` sets `node.details["merge_should_skip"] = True` when its LLM judges
the goal unmet (`actions.py:2169`; the adjacent `actions.py:2168` sets a separate
`merge_incomplete` flag). `should_create_merge_node` reads it at `merge.py:67` and returns
False for that parent whenever any merge child already exists — both the skip branch and the
already-exists branch return unconditionally (`merge.py:53-72`).
`idea_engine._handle_merge_creation` reads it again at `idea_engine.py:1467` and
`idea_engine.py:1478`, marks the node SKIPPED, and returns to the parent with no signal to
re-plan.

**One bad LLM verdict permanently forecloses re-synthesis for that subtree.** Recovery depends
entirely on unrelated expansion machinery happening to add new children later. Given D2/D3
feed this decision, the gate is unreliable *and* irreversible — the worst pairing.

### D5. Recursive merge compounds lossy summarization

`MergeLeafAction` truncates each child's `content` to 2000 chars before synthesis
(`actions.py:2046-2062`). `SimpleMergePolicy.merge` (`merge.py:123-149`), when a child is
itself a merge node, keeps **only** `result.get("synthesized", {})` (line 136) — raw per-leaf
content is gone from that level up. With `enable_recursive_merge`, each level re-summarizes an
already-truncated, already-summarized view.

Partially masked for finalize, because `_collect_all_visit_content` independently re-scans the
graph for raw VISIT nodes — but **the merge chain's own goal-achievement judgements and
synthesis never benefit from that recovery**, and those are what set `GOAL_ACHIEVED` and mark
subtrees DONE/SKIPPED (D4).

### D6. Early exit can bypass grounding, and the backstop is off by default

`should_exit_early` (`idea_engine.py:370-384`) can jump to finalize from a confidence
statistic alone, with no awareness of `parse_mandate_requirements`/`requires_grounded_answer`.
The hooks that would inject a substantiating visit (`GroundingEvidenceEnforcementHook`,
`MandateNavigationHook`, `post_expansion_hooks.py:341-417`) fire only during normal step
expansion.

The downstream catch, `_apply_grounding_gate` (`idea_finalize.py:258-292`), is gated by
`cfg.final.require_grounding`, **default `False`** (VERIFIED, `config.py:320`). So by default
an early exit on a grounding-required mandate reaches finalize with zero visits and nothing
stops it.

### D7. The grounding gate fires last, after the budget is spent, and can only string-strip

`_apply_grounding_gate` is the final step of `build_final_payload`, running **after** the
finalize call and after `_reconcile_finalize_response`'s variation/recompute/verify passes
(`idea_finalize.py:1008-1029`) have already reasoned against evidence that, in the ungrounded
case, does not exist. When it fires it can only regex-replace URLs with a marker
(`_strip_urls`, `idea_finalize.py:249-255`) and prepend a banner. **The fabricated prose
remains in `final_deliverable`, flagged rather than regenerated.**

### D8. Grounding fails open on parser exception — deliberate, but with a real cost

`grounding.py:79-85` (path: `agent/app/idea_policies/grounding.py`):

```python
except Exception:  # noqa: BLE001 — the gate must never crash finalize
    return False
```

This is an **intentional, commented** tradeoff, not an oversight, and the reasoning is sound:
a crashing gate is worse than a permissive one. The cost is that a mandate shape tripping a
regex bug in `mandate_requirements.py` is silently classified "needs no grounding," with no log
distinguishing it from a legitimate non-research task. For `require_grounding=True` runs this
bypasses the gate exactly when the parser is least trustworthy. **Suggested change is a log
line, not a behavior change.**

### D9. Navigation grounding checks page count, not link provenance

`grounding.py:101-103`:

```python
if not (len(followed) >= 1 or len(visited) >= 2):
```

Two arbitrary visited pages satisfy this — no edge relationship between them is checked — so
"real traversal" is accepted for a wiki-race style mandate on the strength of a page count.

**Correction to how this was first flagged to me:** the docstring is honest about it, saying
"or it visited at least two distinct pages (real traversal, not a single start-page read)"
(`grounding.py:91-95`). This is a disclosed weak proxy, not a doc/code mismatch. The
substantive point stands — count is not provenance — but it is a known approximation.

### D10. Fallback deliverable silently changes the answer's shape

When the finalize LLM call returns empty, `_build_fallback_deliverable`
(`idea_finalize.py:305-349`) concatenates raw section dumps — merge summaries, first 3000
chars of each visited page, first 5 search snippets — with no synthesis, and returns it as
`final_deliverable`. The only markers are a log warning and a buried
`"action_summary": "Fallback: LLM finalize call failed"`. **Nothing in the payload schema tells
a scorer this is a degraded path**, so benchmark rows silently mix two different answer-generation
mechanisms.

### Amplifying context (not a primary finding)

With `allow_partial_success` (default `True`), `success = bool(deliverable.strip())` — success
means "the LLM emitted non-empty text," decoupled from `goal_achieved` and
`has_critical_failures`. This compounds D2: both the completion signal and the success signal
are near-tautological.

---

## PART 3 — One root cause behind many documented dead mechanisms

Four separately-catalogued "inert mechanism" findings appear to share a single ordering
decision. Stated as an invariant the engine violates:

> **A decision that consumes a score must run after that score exists.**

Violations, and what each one kills:

| Site | Violation | Consequence |
|---|---|---|
| `idea_engine.py:1153` | `candidates[:max_branching]` truncates **before** evaluation runs | "Beam" selection is arrival-order (ASSUMPTION_AUDIT T1-3) |
| `expansion.py:193-194` | `evaluation_batch_max_candidates=5` drops candidates before scoring | Same, at batch-scoring |
| `auto_parallel_siblings` (default **on**, `idea_dag_settings.json:153`) | executes all children in one step and **skips evaluation for that batch** | Graphs stay depth-1 → max measured `path_to_root` = 2 → backtrack's 5-node requirement unreachable → **0/261 runs fire** |
| `merge.py:80` | root `goal_achieved` computed before merge synthesis | D2's tautology |

`TECHNIQUE_INVENTORY.md:42-50` already flags the `auto_parallel_siblings` interaction as *"the
root cause behind §4/§5/§6"* — confidence judging, early-exit, and backtrack all being inert or
miscalibrated. Adding the truncation sites suggests the scope is wider still.

**Why this reframing is worth the trouble:** fixing four inert mechanisms is four projects with
four experiments. Fixing an evaluation-ordering invariant is one change with one experiment,
and it is a **precondition** for the others being testable at all. Any A/B on pruning,
backtrack, or confidence gating that runs before this is measuring a mechanism that never
executes, and will return a null result that reads as "this technique doesn't help."

Sequencing consequence: **this must land before E1–E4 of the assumption-audit backlog are
worth running.**

---

## PART 4 — Already documented, still unfixed (briefing, REPORTED)

Do not re-derive these; they have owners in existing docs.

| Item | Status / number | Source |
|---|---|---|
| Native reachable-tier composition wall | best lift 0.068→0.196, "far below the 0.75 bar the compiled path clears" | `TECHNIQUE_INVENTORY.md:52-57` |
| native vs `graph_compiled` | 0.198 vs 0.725 (d=1.77); 0.227 vs 0.813 (d=1.95) | `AGENT_FAILURE_MODES_2026-08-10.md` #2 |
| Confidence judge | run AUC 0.571 (< the free "n judged steps" baseline 0.655); **merge AUC 0.288** | `CONFIDENCE_JUDGE_MISCALIBRATION.md` |
| `node.score` / backtrack | AUC 0.466 ≈ chance; 0/261 firings | `EVALUATION_SCORE_PREDICTIVE_POWER.md` |
| codebench write protocol | **40% SyntaxError** (qwen2.5:14b) vs 0% aider; labelled "most actionable" | `AGENT_FAILURE_MODES_2026-08-10.md` #1 |
| **No extraction step in finalize** | confirmed absence; concatenates raw text (to 80k chars), no per-hop typed value | `SHAPE_ADAPTATION_OPEN_QUESTIONS.md` Q32 |
| Chain deficit survives the scheduling fix | `chain_coverage` +0.079 but overall 0.452 vs 0.464 — deficit **relocated** to extraction/finalize | Q12 |
| Shape classifier recall | 19% chain, 4% parallel_merge, 70% return `None`, 1/10 on hand-verified chains | Q1 |
| `evaluation_user_prompt` | carries the identical input/output framing bug already fixed in expansion; "not yet touched" | `TECHNIQUE_INVENTORY.md:305-307` |

**Q32 + Q12 together are the chain story.** No extraction step exists, so on a chain task hop
*n+1* must locate hop *n*'s value inside concatenated raw page text. A linear ReAct loop carries
that value in its message history for free. That is a concrete mechanism for `seq_react`
beating the graph on chains that has nothing to do with scheduling — matching Q12's finding
that fixing the scheduling merely *relocated* the deficit.

**Doc hygiene warnings:**
- `SYSTEM_STATUS.md:8-12` self-flags stale as of 2026-08-06.
- `EVALUATION_SCORE_PREDICTIVE_POWER.md:24-30` and `CONFIDENCE_JUDGE_MISCALIBRATION.md` use
  **different corpora** (the latter's source data no longer exists on disk); their AUC tables
  are not directly comparable despite identical methodology.
- `SHAPE_ADAPTATION_OPEN_QUESTIONS.md` §I contains a same-day self-correction (the "Condition B
  is structurally dead" claim was walked back after 183 real `plan_library` firings were found).
- `badmodel-lab/analyze.py` and `codebench/analyze_code.py` are **destructive regenerators**
  that have silently rewritten committed CSVs. Check `git status` after running either.

---

## PART 5 — Ranked next actions

Ordered by (evidence strength x score impact) / cost. Everything above E-numbers here is a
*fix*, not an experiment; the assumption-audit backlog (E1–E5) sits downstream of R2.

**R1. Test the `qwen2.5:7b` context-truncation hypothesis.** One cell, one model. Highest
information per dollar on this page: it either explains the single worst measured result and
implicates token bloat as an accuracy problem on weak models, or it rules that out and the
0.158 needs a different explanation. Add `num_ctx` control + a served-context log line to
`connector_llm.py` either way — a deployment-dependent silent truncation ceiling makes every
local number non-reproducible.

**R2. Establish the evaluation-ordering invariant (PART 3).** Blocks meaningful measurement of
pruning, backtrack, confidence gating, and audit items E1–E4. Do this before those experiments,
not after.

**R3. Fix D1 (one-line field choice).** `idea_finalize.py:97` → prefer `content_full`, matching
what `idea_finalize.py:244` already does. Correct the wrong comment at `idea_finalize.py:31`.
Lowest-effort item with a direct path to the final answer; needs a re-measure since it changes
finalize input size.

**R4. Replace `_validate_goal_achievement` (D2/D3).** A signal that fires on "a search ran" and
a signal that requires a verbatim goal string are both unusable, and D4 makes the resulting
verdict irreversible. Options, in order of scope: consult the merge-node LLM verdict instead of
the root's substring check; or make `merge_should_skip` revocable; or both. Note this collides
with `CONFIDENCE_JUDGE_MISCALIBRATION.md`'s merge AUC 0.288 — the LLM verdict is
*anti*-predictive on merges, so "just trust the LLM" is not obviously an improvement and needs
measuring, not assuming.

**R5. Build the per-hop extraction step (Q32/Q35).** The best-evidenced explanation for the
chain deficit, and chains are the shape the DAG is supposed to win. Larger than R1–R4; scope it
as its own cycle.

**R6. Log-only fixes.** D8 (log the fail-open), D10 (flag the fallback path in the payload
schema so benchmark rows stop mixing two mechanisms). Cheap, and D10 improves the fidelity of
every future measurement.

**Deliberately not recommended:** simplifying the scaffold toward `seq_react` to chase the
accuracy gap. It would likely work for the models both can run, and it would abandon the 4/8
models nothing off-the-shelf can run — the population the central thesis is about. The open
product question posed at `CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md:448-452` ("is the goal to
beat LangGraph on quality, or to serve the models it cannot run?") is a **decision for the
owner, not something a code review should settle**.

---

## Open questions this review could not resolve

- **Is the `qwen2.5:7b` cell contaminated?** (R1.) Until tested, the "availability, not
  accuracy" summary rests partly on a cell with a plausible infra explanation.
- **Does the graph scaffold pay for itself anywhere?** On the measured evidence it trails
  `seq_react` on cost *and* score, including on chains. The adaptive mechanisms clearly add
  value (`good_adaptive` > `baseline` everywhere tested) — but they are mechanisms that could
  in principle ride on a simpler execution substrate. Nobody has tested that combination:
  **`seq_react` + adaptive mechanisms is an unmeasured cell**, and on this evidence it is the
  most interesting one in the matrix.
- **How much of D1/D5's information loss actually bites?** Depends on the fraction of pages
  exceeding `max_observation_chars = 100000` and on recursive-merge depth in practice. Both
  are countable offline from stored run JSONs; neither has been counted.
