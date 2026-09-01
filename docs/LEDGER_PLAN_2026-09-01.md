# Ledger: general plan, and how it now differs from the DAG v3 master plan

**Branch:** `dagv2-evidence-ledger` (still unmerged — still a falsification project)
**Supersedes as the operative plan:** `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` §3–§4 build
order and §1 primary metric. The master plan's non-goals (§9) and adversarial risks (§7) stand
unchanged and are re-adopted here verbatim in spirit.

---

## 1. Where the master plan turned out to be right, and where it did not

The master plan's thesis was, in its own words:

> For cheap local models, an evidence-first executor with deterministic state and typed action
> selection improves grounded keystone score and requirement-coverage-per-tool-call over
> `sequential_react` ... without needing a stronger planner model.

**That thesis is not supported by the data we now have.** On the purpose-built numeric suite
(`ledgernum22r3`: 22 tasks × 3 arms × 3 reps = 198 cells, 100% preregistered completion, local
qwen2.5:7b, frozen-corpus replay, $0):

| pair | Δ score | p | p_holm | Δ total tokens | Δ LLM calls |
|---|---|---|---|---|---|
| evidence_loop − langgraph_react | **−0.100** | 0.017 | 0.051 | **+47,259** | +27.9 |
| evidence_loop − sequential_react_extract | +0.001 | 0.985 | 0.985 | +21,558 | +8.4 |
| langgraph_react − sequential_react_extract | +0.093 | 0.046 | 0.091 | −26,118 | −19.6 |

The evidence-first executor **ties** the linear extracting control and **loses** to the
off-the-shelf `langgraph_react` — at roughly three times the wall clock and ~47k more tokens per
cell. Nothing clears Holm correction, so no pair is a settled ranking; but the direction is the
opposite of the thesis, and it is not a near miss on cost.

The master plan anticipated this outcome in §6 and said what to do about it:

> Likely end state to plan for: `sequential_react` (or the deterministic queue) becomes the
> default cheap-model path, with DAG v2/v3 tree execution retained only for task shapes with a
> demonstrated breadth win — not a universal replacement.

We should now treat that hedge as the main line, not the fallback.

## 2. The claim that survives, and why it is worth more than the one that did not

`evidence_loop` produces something the other arms structurally cannot: **a calibrated verdict.**
Risk-coverage over the same 198 cells, at threshold 0.5:

| arm | ALL | ANSWER_OR_PARTIAL | ANSWER_ONLY |
|---|---|---|---|
| **evidence_loop** (native ledger verdict) | 0.439 | 0.453 | **0.600** (n=5) |
| langgraph_react (derived verdict) | 0.591 | 0.629 | **0.273** (n=11) |
| sequential_react_extract (derived verdict) | 0.576 | 0.585 | **0.231** (n=13) |

`evidence_loop` is **monotone**: the more it declines to answer, the more often it is right when it
does. Both arms using the post-hoc *derived* verdict are **inverted at their most confident tier** —
their ANSWER cells score materially *worse* than their own average. That inversion reproduced and
strengthened from the earlier n=4–5 sample to n=11 and n=13.

The operative conclusion: **calibration cannot be retrofitted onto an arm that did not maintain a
ledger while it ran.** The ledger is not a scoring trick; it is the thing that makes "I am
confident here and not there" mean anything. That is a defensible differentiator in a way that
"+0.02 mean score" never was.

A likely mechanism, stated as a hypothesis rather than a finding: the derived rule marks ANSWER
when every checkable claim is grounded, so it selects for **claim-poor** answers — a terse
response with one grounded number passes, a thorough one with eight claims fails on the weakest.
That predicts ANSWER-tier cells are shorter and make fewer assertions. Testable offline against
stored cells; do that before touching the rule.

## 3. Primary metric change

The master plan's metric order was: grounded keystone rate → coverage → cost → failure quality.

**The new order is:**

1. **Calibration** — is the arm's own confidence monotone against ground truth (risk-coverage)?
2. **Verifiability** — fabricated-arithmetic rate, claim-level precision/recall at each pipeline
   edge, share of asserted numbers traceable to a located page span.
3. **Cost** — tokens, tool calls, wall clock. Now a first-class blocker, not a tiebreak: a 3×
   wall-clock premium for a −0.100 score delta is not a rounding error.
4. **Mean score** — reported, never led with, and never claimed at n=22 paired tasks.

This is the Ledger pivot's own KPI statement ("risk-coverage not mean score") made operational.

## 4. Build-order divergence from §4 of the master plan

The master plan's order was **A** correctness contract → **B** deterministic queue → **C**
constrained selector → **D** bounded query proposal.

What actually happened:

- **A shipped** — `Ledger.verdict()` gives the ANSWER/PARTIAL/ABSTAIN contract, and it is the one
  component whose value is now empirically demonstrated (§2).
- **B was never built.** `agent/app/testing/execution_evidence_queue.py` is still the stub its own
  docstring describes: "there is no scheduler at all — the 'queue' is a for-loop over a list."
- **An unplanned layer was built instead:** the evidence/derivation graph
  (`agent/app/testing/evidence_graph.py`), where every asserted number traces to a literal page
  span and every computed number is recomputed in Python. The master plan did not have this in its
  build order — it appears only as a *mitigation* in §7 ("store exact excerpt offsets; separate
  observed/extracted/verified; never let extraction alone mark a numeric SUPPORTED"). That
  mitigation was promoted to a subsystem, and it works: 248 SOURCE nodes and 57 DERIVED nodes over
  66 cells with **zero invalid derivations** and a **fabricated-arithmetic rate of 0**.

**Judgement:** promoting §7's mitigation over §4's Phase B was the right call, and the plan should
say so rather than pretend the order was followed. Phase B's premise was that scheduling was the
lever; the measured failures have consistently been *evidence* failures — silent magnitude loss,
unverifiable quotes, uncalibrated confidence — not scheduling failures.

**Phase B is now formally deferred, not pending.** Do not build the deterministic queue until
something in the data says scheduling is the bottleneck. Nothing does today.

## 5. What the master plan did not anticipate at all

- **Measurement infrastructure was the real bottleneck.** More defects were found in the
  measurement path than in the engine: an arm comparison that loaded identical row-sets into every
  arm and printed a spurious zero delta; preregistration gates that were never evaluated; two of
  three arms never persisting the evidence they read, making a three-arm comparison impossible
  from history; a quantity parser that silently discarded ×10⁷ and marked the result valid.
- **"Green tests, never executed" is the dominant failure mode**, twice over. The derivation layer
  passed every offline test and admitted **zero** nodes on a real cell. A live-cell check under
  corpus replay costs $0 and about a minute, and is now mandatory before calling any layer live.
- **Search quality is a capability axis we had switched off.** `LEDGER_MAX_LIVE_FALLBACKS=0` was
  treated as a purity property; it is a measurement choice. Fallbacks are now recorded per search
  (corpus / live / empty) so a query that misses the frozen corpus is a signal rather than a
  blocked event.

## 6. Plan of record

**Now (correctness and cost, both evidence-backed):**

1. **Visit dedup.** `evidence_loop` re-reads **71 of 264** visits (27%) and
   `sequential_react_extract` **101 of 296** (34%), against `langgraph_react`'s 11%. The loop
   dedups searches and has no visit dedup at all. Each re-visit re-injects a full page into the
   prompt, so this is a direct, unglamorous attack on the +47k-token gap.
2. **Test the claim-poverty hypothesis** for the inverted ANSWER tier (§2) offline, before
   changing the derived verdict.
3. **Prereg gate for live fallbacks.** Provenance now persists; decide whether the gate means
   "total live calls across the run" (sum) or "any cell exceeded N" (max), and keep it UNKNOWN for
   any cell predating the change — absent must never read as zero.

**Next (the deliverable):**

4. **Claim-level metrics on the numeric suite** — precision/recall at retrieved → extracted →
   verified → stated, and the fabricated-arithmetic rate, reported per arm. This is the KPI in §3
   and the evidence for §2.
5. **A second model size.** Every number here is qwen2.5:7b. The central thesis is about cheap
   models generally; one model is not a capability claim.

**Deferred, with reasons:**

- Phase B deterministic queue (§4) — no evidence scheduling is the bottleneck.
- Promoting tasks 210–231 into `ACTIVE_SUITE_IDS` — moves the benchmark denominator.
- Any gate driven by the derived verdict or the query-quality signal — both are report-only until
  shown to track ground truth. The query signal weakened markedly on the larger sample (§7 of the
  handoff), which is exactly why it has not been wired to anything.

## 7. Non-goals, re-adopted from the master plan §9

No stronger planner model. No wider beams or more judge calls. No full `IdeaNode.details`
migration. No value-of-information scheduling. **Added:** no unit or currency conversion, ever —
the derivation layer refuses mismatched dimensions by design, and that refusal is a feature the
incompatible-unit tasks exist to verify.
