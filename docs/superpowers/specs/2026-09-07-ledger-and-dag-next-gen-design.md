# Ledger next phase and DAG v3: a critical scoping (2026-09-07)

**Status:** design proposal, not yet approved. Companion to `docs/STATE_OF_EVIDENCE_2026-09-07.md`
and `docs/handoffs/ROADMAP_2026-09-07.md`. New constraint from the user: training a small
model is now allowed (encoders ≤ ~500M params, classical models), but no LLM fine-tuning or LoRA;
completion endpoints stay untouched.

Sources: a code audit of `ledger_tools.py`, `evidence_graph.py`, `quantity_index.py`,
`ledger_risk_coverage.py`, `execution_evidence_loop.py`; a code audit of `idea_engine.py` and
`idea_policies/*`; a corpus inventory of `agent/idea_test_results/` (6,791 r1 cells, 20 GB).
File:line references below come from those audits and were spot-checked.

---

## Part A. The Ledger, critically

### A1. What the certify chain actually measures now

After the quote-capture fix, three of the five clauses are close to tautological on the arms
the campaigns run:

| Clause | What it checks | Reality on the `derive` arm |
|---|---|---|
| 1 | ≥1 DERIVED node | = "the model called `derive`". Sets coverage ≈ adoption rate |
| 2 | `derivation_valid` on every DERIVED | arithmetic only; blind to operand choice (`evidence_graph.py:128-139`) |
| 3 | `operand_supported` recomputed | "does every SOURCE still re-locate in the stored page". Never False on any corpus; near-tautological on a non-drifted artifact |
| 4 | `quote_verified` on every SOURCE | `_quote_for_span` (`ledger_tools.py:170-204`) manufactures a literal substring, so this is **guaranteed True** by construction on `LedgerToolkit` hosts |
| 5 | every deliverable number matches a node | vacuously True on a number-free deliverable (`ledger_risk_coverage.py:190-195`); bare-numeral matching let "120 mph" back a minutes operand |

So the 10.4%-at-6.7%-risk result is real but is mostly "the model derived, with locatable
operands, and its answer number equals what it derived". That is self-consistency plus
adoption, not evidence relevance. The chain has no notion of *which entity a quantity belongs
to* or *whether the operation matches the mandate*. Every certified-but-wrong cell to date is
one of those two failures.

### A2. Mechanical minting exists but is barred from certify, for a good reason

`audit_answer` and `shape_derive_check` mint nodes with zero model participation, but both mint
**from the deliverable**, so `ledger_risk_coverage.py:571` excludes them from certify to avoid
circularity. The zero-derive wall (120/144 cells) is therefore not a missing mechanism; it is
that the only non-circular minting path is the one the model must trigger.

### A3. Scope is numeric-only

Everything mechanical (index, arithmetic, unit refusal, backing) is about quantities. A
non-numeric claim (a name, a date, a category) gets exact-substring quote verification and
nothing else. The "evidence compiler" is, today, an arithmetic auditor.

### A4. The component is a façade nobody runs

`ledger_api.py` has one consumer (`scripts/ledger_run.py`). The harness goes through
`execution_*`. `recheck_*` is called by nothing. Two API defects are open (supplied sources
truncated to 6,000 chars; `clean_operation` reshapes text before offsets are taken).

### A5. Two bugs found in this audit

- `LedgerToolkit.derive` returns a refusal string on `DerivationError` but never calls
  `graph.record_refusal` (`ledger_tools.py:299-300`), unlike the evidence-loop host
  (`execution_evidence_loop.py:1740`). On the `sequential_react` and `langgraph_react` derive
  arms, "correctly refused" is indistinguishable from "never attempted", and
  `ledger_trace.query(kind="derive", status="refused")` can never return anything.
- `unified_verdict` and the unbacked-number check live only in the evidence-loop host; the mint
  campaigns run on the other two hosts, so they have never been exercised where it matters.

### A6. Data reality for anything learned

- Ledger instrumentation exists on ~838 cells (September tail). The other ~5,950 have score,
  observability and graph topology only.
- Reps are not independent: 3–6 reps × 22 tasks × 4 model sizes. Split by task id, always.
  Holdout 213/217/221 is sealed and already consumed for every signal measured so far.
- Model size is a strong confounder with correctness (qwen2.5:7b vs 0.5b). Any learned trust
  signal must be evaluated within model, or it will learn "is this the 7b".
- `derivation_valid` is 723/22 (97% positive): degenerate as a target.
- `operand_supported` has zero stored negatives. Wrong-operand labels as stored: n=12.
- What *is* rich: 4,368 stored pages with full text; task modules 210–221 expose exact operand
  values, entity names, URLs and labeled decoys (`OP_A`, `OP_B`, `_DECOYS`); 4,233
  `answer_audit.numbers[]` rows in three balanced classes; 4,721 `value_verified` extraction
  labels; 6,791 cell-level correct/wrong labels.

---

## Part B. Gaps worth filling in the Ledger

Ordered by value ÷ risk. Each is a separate cycle with its own prereg.

### L1. Pre-answer mechanical derivation ("the host computes")

Run the shape-driven derivation **before** the answer exists, from the mandate and the visited
pages only: demanded operation (from `answer_numbers.mandate_demanded_operation`) + entity roster
(from `candidate_coverage.extract_named_candidates`) + per-entity operand attribution over the
quantity index → compute → compare the model's stated number to the host's. Because operands
come from the mandate and the pages, not from the deliverable, the result is not circular and
can enter certify as a sixth clause ("agrees with host derivation") or replace clauses 1–3.

- Kills the zero-derive wall: coverage no longer depends on the model calling `derive`.
- Kills all four mint02 wrong-entity vindications by construction (one operand per entity).
- Depends on L2 for operand attribution; without it, candidate explosion returns.
- Honest limit: only for mandates with an unambiguous demanded operation (mint02: 64/192 cells
  were `no_unambiguous_shape`). That is a cue-lexicon problem; L2's field matcher helps here too.

### L2. Operand-attribution model (the one classifier worth training first)

The decision every heuristic in `ledger_tools.py` currently fakes with first-match-wins and
fixed preferences (`_locate`, `_find_backed_match`, `_best_explanation`,
`SHAPE_DERIVE_MAX_CANDIDATES=40`): *given an entity and a field phrase from the mandate, which
quantity-index entry on which page is the operand?*

- **Input:** (entity name, field phrase) × (index entry: label line, unit, ±300-char context,
  infobox-vs-prose flag, page title, URL slug, whether the entity name appears within the
  window). Output: P(is the operand). Used as a ranker, replacing caps with top-k.
- **Training labels, without touching the 12 tasks:** weak supervision from the 4,368 stored
  pages themselves. Every infobox label/value pair is a self-labeled (field phrase → entry)
  positive; every other entry on the same page is a negative for that phrase. Page title is the
  entity. This gives tens of thousands of pairs with no task leakage.
- **Evaluation only:** tasks 210–221's `OP_A`/`OP_B` (positives) and `_DECOYS` (hard
  negatives), dev split, holdout sealed. The metric is rank of the true operand and
  precision@1 per (entity, field).
- **Model ladder:** (1) frozen MiniLM embeddings + logistic regression over the pair features;
  (2) if (1) fails on prose-only pages, fine-tune a small cross-encoder (allowed). Start at (1);
  it may be enough because the field phrase ↔ label line match is mostly lexical.
- **Risk:** infobox self-labels teach infobox matching; prose-embedded operands
  (task 216's "2 hours 21 minutes") are the hard case and the one where weak supervision is
  thinnest. Report infobox and prose accuracy separately; do not blend.

### L3. Learned trust channel for risk-coverage

The KPI is risk-coverage. The hand-written certify chain is one point on that curve. A gradient
boosted model over **mechanical evidence features only** (n_rows, n_derived, backed/derived/
unbacked counts and ambiguity from `answer_audit.numbers[]`, hosts visited, distinct pages,
quote-verified rate, unit consistency, `shape_derive` reason, demanded-operation presence,
visit and search counts, tokens) trained on the ~838 instrumented cells with task-grouped CV
gives a full curve and a coverage number at any chosen risk.

- Cheap (sklearn, already in the venv), directly moves the headline KPI.
- Must exclude model identity and size from features and report per-model curves; otherwise
  it learns the confounder.
- Success bar: beats the frozen certify chain's coverage at ≤ its risk, within model, on the
  task-grouped folds; then confirmed once on a fresh prereg'd campaign.
- Honest expectation: modest. n is small and the informative features are the same ones the
  chain already uses. The gain, if any, comes from *soft* combination (ambiguity counts,
  host diversity) that a hard AND-chain cannot express.

### L4. Claims beyond numbers (later)

Extend the row type to (entity, field, span) for non-numeric values and use an off-the-shelf
NLI cross-encoder to score paraphrase support where exact-substring fails. No training until
paraphrase labels exist. First step is measurement: how many `quote_verified=False` rows
(3,045 extraction records) are paraphrases of a supporting span vs genuinely unsupported? A
200-row hand audit decides whether this is worth anything.

### L5. Hygiene, all small

- Record refusals on the `LedgerToolkit` path (A5).
- Fix the two `ledger_api.py` defects; make the harness call the API on at least one host so
  the component contract is exercised by every campaign.
- Second-hostname visit policy (85.6% of cells with ≥2 offered hosts visit one): a mechanical
  rule, prerequisite for any corroboration signal.
- Run `unified_verdict` on the mint hosts once, seeded, to close it out or ship it.

---

## Part C. DAG v2, critically

### C1. Two default-on LLM scalar judges gate real behaviour, both measured at or below chance

- Candidate scoring (`LlmEvaluationPolicy`) → selection, prune (mean − 1σ), dynamic beam.
  Node-level AUC 0.426; "which sibling ran" 0.378 (anti-selective).
- Merge `goal_achieved` → `deliverable_complete` / `finalization_status`. Merge confidence
  AUC 0.288.

Everything else in the judge family is default-off. The mechanical alternatives the external
review listed (new canonical URL, newly supported cell, duplicate hash, repeated query) are
all computable from `IdeaDag` today and none is wired as a score.

### C2. It is a tree that calls itself a DAG

`merge_nodes(parent_ids)` exists and is unwired (`merge.py:453-460` says why: the upward
walk reads `parent_id`, which is None on a multi-parent node). `path_to_root` follows
`parents[0]`. `is_action_ready` ignores predecessors entirely. Edges are never emitted.
The `requires_data` backfill writer at `expansion.py:1758` writes an ancestor id into a field
whose reader checks sibling membership, so that writer's output can never fire.

### C3. Three lossy compressions and no typed record

Page → 2,000 chars/child at merge → 100k blind JSON chop → 1,000-char preview + 80k raw
concat at finalize → prose. The merge prompt says "remove redundancy", which on aggregation
tasks deletes rows that differ only by a number. No `(entity, field, value)` record exists on
the graph path. Validator coverage on aggregation tasks: 0.095 vs 0.732 for the linear arm.

### C4. The fixes are already built and default-off, and unmeasured on the shape that loses

`sibling_evidence_digest_enabled`, `inject_coverage_visits`, `evidence_store_mode` +
`merge_uses_evidence_view`, `resolved_value_channel_enabled`, `parallel_requires_evidence`.
All exist, all off, none measured on the 23 aggregation-shaped tasks. The one A/B that touched
sibling sharing found duplication fell 12.75× → 3.75× with score flat, and the review's
diagnosis of that null is coherent: freed budget was never reinvested in missing cells.

### C5. The reviewed "Cell 4" already exists and was tested on the wrong tasks

`execution_evidence_loop.py` is the deterministic-coverage-plus-typed-records executor the
external review predicted would recover aggregation tasks. It was measured on the numeric
derivation suite (chains), where it lost −0.100 to LangGraph at 3× cost. It has never run on
the 23 aggregation tasks where the −0.461 loss lives.

### C6. Learned pieces: where labels exist and where they do not

| Decision | Labels | Verdict |
|---|---|---|
| Stop/continue | precedent exists (`confidence_early_exit.py`, fitted + held-out certified); coverage features unused | feasible, cheap |
| Candidate scoring | only bag labels (run score on every node); per-node contribution label needs typed-record arms | not yet; build the label channel first |
| Shape classification | 0 stored; ~59 recoverable from a doc, ~231 by re-reading modules; collinear with authoring era | too small, defer |
| Dependency detection | 19 hand-verified slots out of 418 leaves, 0% FP | rule-based is fine; the bug is the writer, not the detector |
| Dedup threshold | batch-level intervention effect only (−0.157, n=47), no pair labels | not worth a model |

---

## Part D. What DAG v3 should be, conditionally

### D1. Configuration ablation on the aggregation-23 (Small, $0, flags exist)

Seeded corpus replay, `graph` arm, five conditions: judges off (mechanical scoring: novelty +
coverage delta, prune disabled), sibling digest on, coverage-visit injection on, evidence view
in merge on, all four together. Paired against the current default and against
`sequential_react`. Decides whether DAG v2 is repairable by configuration.

### D2. The 2×2 on the aggregation-23 (Medium)

{`sequential_react`, `evidence_loop`} × {prose, typed records}, using
`execution_sequential_extract` as the extraction-only control. This is the experiment the
external review asked for and the one the repo never ran. Effect to detect is 0.46, far above
the noise floor.

### D3. Branch on D2

- **If the coverage executor wins on aggregation shape:** DAG v3 = the evidence loop's rows as
  work items, a dependency-aware queue (chain hops through `resolved_value_channel`, which
  exists), parallel dispatch reusing `idea_engine.py:2148-2360`'s semaphore-gather, the Ledger
  as the typed reducer, table-first finalize (exists). The DAG survives only as the dependency
  analyzer LEDGER.md already says it should be. Judges gone; mechanical scores only.
- **If it does not:** stop engine work. The Ledger stays a post-hoc auditor bound to linear
  hosts, and the project's engine story ends at "DAG v2 closed, linear wins, here is why".

Either outcome is a defensible finding. The current state, where the executor exists but was
tested on the wrong shape, is not.

### D4. Learned stop/continue (Small, after D1)

Extend the `confidence_early_exit` fitting precedent from judge-confidence prefixes to
mechanical features: coverage resolved/missing ratio, distinct canonical URLs, repeat-action
rate, grounding verdict, pending count, budget fraction. Same label rule (score ≥ 0.75), same
held-out certification. Replaces the one remaining judge the graph arm depends on for stopping.

---

## Part E. Proposed sequence (replaces roadmap step 2 and sharpens step 5)

> **Superseded (2026-09-08):** this sequence is replaced by `docs/superpowers/plans/2026-09-08-ledger-dag-replan.md`; see the review at `docs/superpowers/reviews/2026-09-08-ledger-dag-adversarial-review.md`. Parts A–D stay the design reference.

| Order | Item | Size | Gate to proceed |
|---|---|---|---|
| 1 | L5 hygiene + refusal bug | Small | tests green; refusals visible in `ledger_trace` |
| 2 | L2 operand-attribution model, weak-supervised, evaluated on 210–221 dev | Medium | precision@1 on infobox operands ≥ 0.9; prose reported separately |
| 3 | L1 pre-answer host derivation, prereg'd mint03 | Medium | coverage ≥ 25% at ≤ 5% risk within model, dev split |
| 4 | L3 learned trust channel | Small | beats certify chain within model on task-grouped CV |
| 5 | D1 ablation + D2 2×2 on aggregation-23 | Medium | decides D3 |
| 6 | D3 (conditional) or component packaging (roadmap step 3) | Large / Medium | per D2 |

Items 2–4 are the Ledger track; item 5 is the DAG track; they do not share files and can run
as parallel lanes under the multi-lane protocol.

## What this does not promise

- L1 covers numeric mandates with an unambiguous demanded operation only.
- L2 may plateau on prose-embedded operands; that limit is reported, not hidden.
- L3 is expected to be a modest gain; its value is the full curve, not a headline number.
- D2 may say the linear arm wins everywhere. That closes the engine question honestly.
- Nothing here changes the LLM: all learned components sit beside frozen completion endpoints.
