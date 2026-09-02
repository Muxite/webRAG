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
