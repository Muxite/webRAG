# DAG v3 "Ledger": Master Plan

**Branch:** `dagv2-evidence-ledger` (stays unmerged — this is a falsification project, not a committed rewrite)
**Status:** planning, supersedes the free-form design discussion this doc compresses
**Codename:** **DAG v3 "Ledger"** — the successor to DAG v2, named after the `TaskLedger`/`Evidence`/`Claim` sidecars that already exist in this branch as observe-only telemetry. The plan below is the path to making them authoritative instead of decorative.

## 1. Thesis (falsifiable, lexicographic)

> For cheap local models, an evidence-first executor with deterministic state and typed action selection improves grounded keystone score and requirement-coverage-per-tool-call over `sequential_react`, while reducing repeated no-progress actions — without needing a stronger planner model.

Primary metric order: **(1)** grounded keystone pass rate, **(2)** coverage among grounded runs, **(3)** cost (tokens + tool calls + wall time), **(4)** failure quality (partial/abstain vs fabrication, reported as separate outcome classes, not blended into one score).

Do not accept "the queue feels cleaner" as evidence. Every new abstraction gets a control that isolates its own contribution.

## 2. What tonight's data already tells us

- Native graph (DAG v2) averages 0.462 on core24 vs. 0.643 `sequential_react` / 0.663 `langgraph_react`, but a grounded reduced-scope rerun *reversed* this — graph beat sequential_react when search infra was actually healthy. Treat the core24 gap as **confounded by infra flakiness**, not yet a clean architecture verdict.
- Mean graph size is only 4.6 nodes — it pays coordination/merge/eval overhead without earning real breadth.
- Task 123: 43 visits, 1/4 sub-entities resolved, same subgoals re-issued 5–8 times. Root cause is **branch isolation** (rootward-ancestry-only context, ~1000 chars/page cap, no sibling visibility), not insufficient search cleverness.
- The `ledger_deficit_local` A/B (this branch, tonight) came back **null** on core24 — but the injector never got a chance to fire, because `good_adaptive` already visited every candidate. This is underpowered, not disconfirming.
- Constrained decoding is now live end-to-end for repair paths (commit `ef801010`); the `langgraph_react` tool-calling gap is scoped in a handoff doc but not built (`f1406b32`).

Conclusion: don't jump straight to a full evidence-queue rewrite. Run the cheap ablations first — they'll tell you whether the bottleneck is *context visibility*, *churn*, *action validity*, or genuinely *topology*.

## 3. Phase 0 — Ablations before any rewrite (do this first, on this branch)

Five cheap arms, all built as thin variants of the *existing* DAG v2 engine — not new code paths:

| Arm | Change | Falsifies |
|---|---|---|
| `graph_shared_context` | Expansion/eval sees a compact sibling-evidence digest | Branch isolation is the main failure |
| `graph_no_reexpand` | Novelty cap + deterministic repeated-goal block | Churn is the main failure |
| `graph_constrained_actions` | Codec-only (already built — just gate it on) | Malformed-action instability is the main failure |
| `sequential_react_context_matched` | Cap ReAct's context to the DAG's budget | The graph's loss is mostly a context-budget artifact |
| `evidence_queue_deterministic` | No LLM scheduler, typed state only (Phase B below) | Typed shared state helps even without learned scheduling |

**Kill gate:** if `graph_shared_context` alone closes most of the gap to `sequential_react`, stop — fix DAG v2's context projection and churn guard, do not build a new engine. This is the single most important gate in this plan; it can make everything below unnecessary.

## 4. Target architecture (only if Phase 0 says topology is real)

Replace the model-owned control plane; keep the transport layer.

**Keep unchanged:** `AgentIO`, tool connectors, native Ollama backend, grounding/coverage validators, benchmark harness, telemetry, `sequential_react`.
**Replace (new variant only, existing variants stay byte-identical):** expansion policy, sibling/beam evaluation, tree nodes as runtime state, re-expansion/backtracking, merge nodes.
**Leave alone for now:** `IdeaNode.details` — do not migrate the whole DAG's data model; that confounds the experiment with an unrelated reliability refactor.

Core records (typed, reducer-owned — illegal transitions raise in tests, LangGraph-style):

```python
Requirement.status: OPEN → SUPPORTED | BLOCKED | CONTRADICTED   # not reversible without a contradiction event
Evidence: append-only
Claim.status: UNVERIFIED → VERIFIED | REJECTED | CONTESTED
WorkItem.status: PENDING → RUNNING → DONE | BLOCKED | FAILED
```

The LLM is a **selector among deterministically-minted jobs**, plus one constrained escape hatch (`PROPOSE_QUERY`, rejected if duplicate/unanchored/over length/no new evidence) — never a free-form planner. This is the fix for the current DAG's actual failure (weak model asked to author + track + revisit a plan from 1000-char fragments), not a bet that a bigger action space helps.

Build order, each phase gated by the one before it:

- **A — Correctness first (do this regardless of Phase 0 outcome):** hard final-answer contract (`ANSWER`/`PARTIAL`/`ABSTAIN` — never let `goal_achieved=True` coexist with an unmet critical requirement; never let the finalizer emit a specific fabricated value after saying evidence is insufficient), novelty/churn guard, model-metadata capture (Ollama digest, quantization, context, tool capability — currently unrecorded confounders).
- **B — Deterministic queue, no LLM scheduling.** If this can't match `sequential_react` on coverage-per-visit, stop; the typed state may still be useful but the queue is not the performance lever.
- **C — Constrained selector** (existing codec, wired to job selection + extract/verify transitions).
- **D — Optional: bounded query proposal + deterministic priority score.** Only after C wins.

## 5. Naming and positioning going forward

- **DAG v2** stays the name for the current interleaved plan→act→observe→decide engine (native graph, `good_adaptive`, ledger/evidence sidecars as currently observe-only telemetry).
- **DAG v3 "Ledger"** is reserved for the evidence-queue architecture *if and only if* it clears the Phase 0 gates and the Phase A–D promotion gates in §6. It is not a rename of DAG v2 in place — it is a new execution variant that may or may not become the default.
- Product framing (separate from the engine question): Euglena is "auditable research for constrained decisions" — comparison / eligibility-fit / research-brief shapes, with a structured claim table and an `ANSWER`/`PARTIAL`/`ABSTAIN` output contract as the customer-facing form of the same correctness gate in §4A.

## 6. Promotion gates (merge-worthiness, not "looks nicer")

| Component | Merge condition |
|---|---|
| Final-output contract (§4A) | All adversarial offline fixtures pass — this is a correctness fix, ship independent of everything else |
| Novelty/churn cap | Cuts repeated no-progress actions without lowering core task score |
| Model metadata telemetry | Immediately, once fixture + backward-compat tests pass |
| Constrained codec (repair paths) | Already shipped (`ef801010`) |
| Constrained codec (action/job selection) | Valid-action rate rises, no regression on qwen2.5:7b or weaker |
| Deterministic queue (Phase B) | Beats `sequential_react` on grounded keystone rate or materially cuts cost at equal quality |
| LLM queue selector (Phase C/D) | Beats the deterministic queue, not just old DAG v2 |
| Full DAG v3 as default | Only after wins across ≥2 model sizes and both wide + sequential task shapes |

Likely end state to plan for: `sequential_react` (or the deterministic queue) becomes the default cheap-model path, with DAG v2/v3 tree execution retained only for task shapes with a demonstrated breadth win — not a universal replacement.

## 7. Adversarial risks to design against, not discover later

- **Requirement parser becomes a hidden planner.** Mitigate: constrained schema, immutable original mandate, deterministic entity/count checks, `parser_confidence`, one bounded `REQUEST_REQUIREMENT_REWRITE` retry.
- **"Evidence supports claim" is itself a model judgment.** Mitigate: store exact excerpt offsets; separate `observed`/`extracted`/`verified`; never let extraction alone mark a numeric `SUPPORTED`; require two sources for disputed numerics.
- **Deterministic queue overfits to entity-list tasks.** Mitigate: keep enumerative / exploratory / multi-hop suites disjoint; freeze a holdout suite never used for scheduler/prompt/codec tuning.
- **Hard gating produces "safe but useless."** Mitigate: `PARTIAL` must be mechanically rendered from supported claims — never re-invoke the model to "summarize what's missing" after it was just judged ungrounded.

## 8. Test layers (condensed)

1. **Unit/invariant (offline, scripted):** no-fabrication-after-abstain, bounded churn, snippet-alone can't mark `SUPPORTED`, entity-identity strictness, dedup doesn't reset retry budget, dependency ordering, variant isolation (existing engines byte-identical with feature off).
2. **Differential replay:** same recorded fixtures through graph / deterministic queue / queue+selector; log every step's eligible jobs, selection, novelty key, and requirement delta — if a bad result isn't explainable from this log, don't burn live budget yet.
3. **Purpose-built mechanism suite (new, ~10 tasks):** entity collision, duplicated/syndicated URLs, plausible-but-unsupported numerics, source conflict requiring `VERIFY`, dead-end retry-cap proof, one genuine wide-fanout task, one genuine narrow-sequential task (proves the queue doesn't impose needless breadth).
4. **Live paired eval, in order:** smoke8 → mechanism suite (3–5 reps) → core24 (3 reps, alternating order) → suite59 only if directional → weak-model matrix (qwen 1.5b/7b/14b, llama3.2 1b/3b, phi3:mini, tinyllama where the codec/emulation path supports it).

Fairness floor for every arm: identical model digest/quantization/context/backend/search-provider/tool-budget; fail closed (no silent score) if any of that metadata is missing.

## 9. Explicit non-goals (do not start these before the gates above clear)

- No wider beams / more judge calls on the existing tree — it's under-informed, not under-searched.
- No full `IdeaNode.details` → typed-record migration.
- No value-of-information scheduling — the deficit injector hasn't shown a live win yet because it's never had a chance to fire; build the mechanism suite in §8.3 before trying again.
- No bigger/stronger planner model — DAG v2/v3's research value is same-model local planning *and* execution; that's the whole point of the central thesis (boost cheap models via structure, not via a stronger model upstream).

## 10. Immediate next actions

1. Land Phase 0's five ablation arms as flags on the existing DAG v2 engine (cheapest, most information-dense step — do this before writing any new engine code).
2. Re-run core24 with search infra verified healthy first (tonight's reversal shows infra flakiness previously confounded the graph-vs-sequential comparison — always grep cell logs for "Setup failed" before trusting a result, per the Serper outage incident).
3. Author the mechanism suite (§8.3) — this is the missing piece that makes the ledger/deficit injector's null result interpretable.
4. Only after 1–3: start Phase A (final-answer contract + novelty guard) as a standalone, independently mergeable correctness fix regardless of how the ablations land.
