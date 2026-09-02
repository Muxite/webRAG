# The Ledger as a module: host vs host + module

## The mismatch this corrects

`docs/LEDGER.md:24` states the product plainly:

> It is a component, not an agent.

Every experiment in this repo has nonetheless measured it as a **rival agent** —
`evidence_loop` against `langgraph_react`. The experiment design has been contradicting the
product definition, and that is not a stylistic complaint. Two consequences, both measured:

1. **Nothing is attributable.** The two arms differ in loop, prompt, budget, step policy and
   output contract simultaneously. A delta cannot be assigned to "the ledger" because the ledger
   is not the only thing that differs.
2. **The comparison does not replicate.** On `ledgerfinal01`, per-arm unsupported-claim rate
   INVERTED between the tuning split and the sealed holdout — `evidence_loop` 0.081 -> 0.320,
   `langgraph_react` 0.327 -> 0.063. See `docs/LEDGER_FINAL01_TUNING_RESULT.md`. Arm orderings on
   this suite are subset effects, not arm properties.

Whether "we beat LangGraph" is also the wrong question commercially is a separate argument. It is
certainly the wrong question *empirically*: at this suite's size it cannot be answered.

## The design

**X' = host + module.** Same system, one change, measured within-host.

**Hosts** (not ours — the thing being augmented):

| host | why it is in the set |
|---|---|
| `langgraph_react` | the off-the-shelf standard; binds tools through a tool-calling API |
| `sequential_react` | plain text/JSON ReAct with a string action dispatch, no tool API at all |

Both is the point. A module that works only in LangGraph might merely suit LangGraph's tool API;
the same module working in a loop with no tool API is what makes "component" a claim rather than
a description.

**Modules** (ours — the Ledger, decomposed into attachable parts):

| module | what it makes true of the host | status |
|---|---|---|
| `derive` (`agent/app/ledger_tools.py`) | every derived number is recomputed in Python; operands must be located on a fetched page; incompatible units refused | built |
| value-verified claim recording | a claim whose value is not on the cited page cannot be recorded | exists as the value gate, not yet attachable |
| confidence / abstain channel | a risk-coverage curve can be drawn at all | shipped to all arms |
| prompted tool transport | the host runs on models it otherwise HTTP 400s on | shipped |
| page freezing + reverify | a third party can re-verify the run offline | shipped to all arms |

## What may be measured, and what may not

This is a design constraint, not a caveat. Attaching a module perturbs the trajectory, and the
trajectory-chaos floor on this stack is large: with decoding deterministic (`LLM_SEED` fixed) and
one mechanism flipped, **12 of 16 tasks were byte-identical and 2 swung by ~0.6**. So:

**Structural outcomes — categorical, and robust at this n:**
- Can the host produce a machine-verified derivation at all? (impossible -> possible)
- What share of asserted numbers carry a mechanically-checked provenance record?
- Can a risk-coverage curve be drawn at all?
- Does the host run on this model at all?
- Does the run replay offline?

**Score — the guard, not the target.** `validation.overall_score` answers only "did attaching the
module cost accuracy?", where a null result is the win: auditability at no accuracy cost is the
honest and sufficient claim. A mean-score *ranking* between configurations is not resolvable here
and must not be reported.

## Why the module refuses rather than converts

`derive` refuses an operand it cannot locate on a fetched page, and refuses incompatible units
rather than converting them (`docs/LEDGER_PLAN_2026-09-01.md` section 7 non-goal). Both refusals
are the product: a derived number that cannot be traced to text the host actually read is exactly
the fabrication this module exists to prevent, and the unit-mismatch tasks (222-224) exist to
verify that a refusal happens rather than a silent conversion.

## Status

`agent/app/ledger_tools.LedgerToolkit` is built and tested (`b156856c`). Its artifact is
`evidence_graph`-shaped and verified to be read unchanged by `evidence_graph.reverify_graph` and
`scripts/claim_metrics.derivation_fabrication_rate` — so every existing offline auditor works on a
host's artifact with no modification.

Host bindings are in progress behind `LEDGER_HOST_MODULES`, default OFF, so an unset run
reproduces today's behaviour exactly and writes no `evidence_graph` key at all (absent, never
zero — a module-off cell's fabricated-arithmetic rate must read UNKNOWN).

---

# Result: `derive` attached to two hosts (2026-09-02)

Seeded (`LLM_SEED=12345`), qwen2.5:7b, frozen corpus, the 12 derived-arithmetic tasks (210-221),
1 rep, $0. Four configurations, one `run_id` each (the cfg hash covers only
`variant_specific_settings`, so env-flag conditions sharing a `run_id` would overwrite each other).
Analysed with `scripts/module_ab.py`.

| | LangGraph off | LangGraph +derive | ReAct off | ReAct +derive |
|---|---|---|---|---|
| cells with a derivation | 0 | **5** | 0 | **4** |
| machine-computed values | 0 | **8** | 0 | **9** |
| of those, invalid | 0 | 0 | 0 | **1** |
| fabricated-arith rate | UNKNOWN | 0.000 | UNKNOWN | **0.111** |
| distinct operands refused | 0 | 20 | 0 | 25 |
| replayable cells | 0/0 | 6/6 | 0/0 | 5/5 |
| overall_score (guard) | 0.938 | 0.863 | 0.646 | 0.671 |

## What this shows

**The categorical change is the result.** Both hosts go from producing no machine-checkable
derivation at all to producing several, each traceable to operands located on a page the host
actually read, and each replayable offline. Their fabricated-arithmetic rate moves from UNKNOWN --
uncomputable by anyone, forever -- to measured. That is not a better score; it is a different kind
of artifact.

**The module caught a real fabrication.** In the ReAct host, 1 of 9 derived values had the model's
proposed answer disagree with the Python recomputation (rate 0.111). Without the module that
number would have entered the answer unflagged and unflaggable.

**The accuracy guard passes, and the score movement is not the module's doing.** In LangGraph,
cells that USED `derive` moved -0.072 and cells that never called it moved -0.077 -- indistinguishable,
and the largest single swing (task 216, -0.80) is in a cell that never used the tool. Under a fixed
seed an ignoring cell would be identical to the off cell except that the tool description is in the
prompt either way, so this is prompt-perturbation trajectory noise, not a cost of the module's
behaviour. The ReAct host shows the same pattern (+0.068 used vs +0.004 unused). Neither delta is
resolvable at n=12 and neither is claimed.

## The honest limitation: adoption

The tool is optional and the model used it in **5 of 12** (LangGraph) and **4 of 12** (ReAct)
derived-arithmetic cells. On the rest the host did the arithmetic in its head and the module
delivered nothing. Auditability that depends on the model choosing to be audited is not a
guarantee.

This is the case for the next module: a finish policy that refuses to accept a computed number in
the final answer unless it came from `derive`. That converts an optional tool into a structural
property, which is the difference between "can be audited" and "is audited".

## Two measurement bugs found and fixed en route

Both were the instrumentation measuring itself, and both would have been reported as findings.

1. **Over-refusal on unit spelling.** The page reads `1,470\nm`; the model wrote `"1,470 metres"`;
   the module refused a value it had genuinely read. `evidence_graph` treats `m` and `metres` as
   one unit only when the unit is supplied separately from the number, and `_locate` was passing
   the whole string as a literal. A large share of the first run's "averted fabrications" were
   this. Fixed in `4c1a45d3`.
2. **Refusal counts inflated by the page scan.** `_locate` tries every registered page, so one
   unlocatable operand emitted one rejection record per page -- reading 176 for a run with about
   five genuinely refused operands. `module_ab.py` now counts distinct refused operands.

The first matrix (`mod_*` run ids) is superseded by the second (`mod2_*`) and should not be cited.
