# Euglena Ledger

**Codename:** `ledger` · **Status:** pivot declared 2026-08-31 · **Repo:** webRAG (unchanged)

Euglena Ledger is an auditable evidence compiler. Give it a question and a set of
sources; it returns a ledger of atomic claims, each pinned to a verbatim span on a fetched
page, plus any values derived from those claims by deterministic computation, plus a
verdict of ANSWER / PARTIAL / ABSTAIN derived in code from what was actually obtained.

It is a component, not an agent. It has no opinion about how you decompose a task or
plan a strategy. DAG v2 can call it. LangGraph can call it. A human script can call it.

---

## Why the scope changed

DAG v1, Compiled v1, and DAG v2 were general-case agentic systems: plan a task, choose
tools, execute, adapt. That is the niche LangGraph occupies, and LangGraph occupies it as
an industry standard for good reasons — a large team, an enormous integration surface, and
years of iteration. Our own measurements say the same thing. Across two search backends
and 432 executed cells on `core_long24`, our engine and `langgraph_react` were separated
by roughly 0.09 under one backend and were statistically indistinguishable under the
other, with no pairwise test clearing threshold at n=24. Closing that on mean score would
take 61–111 paired observations per this repo's own power table, and would win a contest
whose prize is parity with a framework we cannot out-resource.

Competing on breadth of agentic capability was the wrong bet. Not because the engine is
bad, but because the niche is too wide to be defensible by a project this size.

The niche that *is* defensible is the thing none of the general frameworks do: make a
weak model's answer auditable, replayable, and honest about its own gaps. LangGraph
returns prose. `sequential_react_extract` reports `success=True` on 48 of 48 runs.
Neither can draw a risk-coverage curve, and that is a structural fact about their output
contract rather than a tuning gap.

## The claim being staked

> For weak local models, reliability improves when web evidence is transformed into typed
> atomic records and deterministic derivations *before* answer generation. Raw long-context
> synthesis cannot be trusted as the sole aggregation mechanism.

Stated as an engineering target: at parity accuracy with a baseline doing identical
evidence-gathering, only this system knows when it does not know — and every step that
produced the answer can be replayed, timed, and re-run with one variable changed.

## Design commitments

1. **Parallel, not sequential.** Work is a queue of typed actions, not a chain of turns.
2. **Structured at every level.** Every action has a declared type, inputs, and output
   schema. No stage consumes another stage's prose.
3. **Resilient.** One failed action does not fail the run. The system retries, routes
   around, or tries an alternate approach, and records that it did.
4. **Low-variance, not necessarily deterministic.** LLM calls stay stochastic; everything
   downstream of them — arithmetic, unit algebra, verdict derivation — is deterministic.
   Randomness is confined and measured, not spread.
5. **Able to abstain.** ABSTAIN is a first-class success outcome, derived from evidence
   state in code, never asked of the model.
6. **Auditable by construction.** Every LLM call is individually addressable, with its
   prompt, response, timing, and cost recorded.
7. **Replayable.** Any run can be re-executed from recorded inputs at zero cost, and
   re-executed again with exactly one thing changed.
8. **Self-improving within a run.** As evidence accumulates, the system may revise its
   approach — reformulate a query, choose a different source, widen a roster.

The DAG survives, in a narrower role: analyzing the question to determine which pieces of
evidence depend on which others, so independent work can be dispatched in parallel and
dependent work correctly ordered. It is a dependency analyzer, not a plan executor.

## What the tool is for

The counterfactual harness is not instrumentation around the product. It **is** the
product's second half. When a run produces a wrong answer, the system should let you:

- isolate the single action that introduced the error;
- see whether that action was deterministic or stochastic;
- re-run from that point with a changed prompt, changed structure, or changed evidence;
- and observe what moves the outcome.

That capability is what turns a benchmark score into a finding.

## Subsystems

Each gets its own spec and its own cycle. Several exist in partial form today.

| # | Subsystem | Today |
|---|---|---|
| 1 | Typed action queue + parallel scheduler | partial — engine has parallel visits |
| 2 | DAG evidence-dependency analysis | partial — DAG v2 planning to be re-scoped |
| 3 | Deterministic derivation + unit refusal + abstain | **built, unwired** (`evidence_graph.py`) |
| 4 | Per-call audit log: prompt, response, timing, cost | partial — `trace_recorder.py`, `telemetry.py` |
| 5 | Record/replay + counterfactual re-run | partial — record/replay works for HTTP *and* search, but exact-key fixtures ~never hit for a variable-query agent |
| 6 | Leak-resistant benchmark construction | partial — keystone gates exist; 046/047 asymmetry fixed |

The single largest concrete gap is #5, though not for the reason an earlier draft of this
document claimed. Search **is** already recordable and replayable: `ConnectorSearch`
subclasses `ConnectorHttp` (`agent/app/connector_search.py:104`) and the fixture hook lives
inside `ConnectorHttp.request` (`agent/app/connector_http.py:125-155`), so all three
backends inherit it. Four modes exist — `off`, `record`, `replay` (live-fallback-and-record)
and `replay_strict` (`agent/app/web_fixtures.py:32-39`).

The real blocker is that fixture keys are a sha256 over the *exact* request, query text
included (`web_fixtures.py:74-81`). An adaptive agent re-expands to different pages and
emits different queries on every run, so a cache recorded on one arm misses on another —
measured as a 289 MB record pass producing roughly zero effective hits
(`scripts/BENCHMARK_NATIVE.md:14-19`), which is why `scripts/native_ab_run.sh:50` forces
`IDEA_TEST_FIXTURES=off`. Exact-key replay is structurally incompatible with an agent that
never repeats itself.

So #5 is not "add fixtures to search". It is: freeze a task's evidence universe once, and
let every arm query that frozen universe freely. Until that exists, every arm comparison is
a live, paid, non-reproducible experiment — which is exactly why the backend confound below
was able to invert a headline result.

## KPIs

Mean score is a reporting metric, not the target. The targets are:

- **Risk-coverage.** Selective accuracy as a function of confidence threshold. Requires an
  abstain channel, which is the differentiator.
- **Claim-level precision and recall**, measured at each pipeline edge — retrieved,
  extracted, verified, stated — so a score gap can be localized rather than guessed at.
- **Fabricated-arithmetic rate.** Numbers asserted that no derivation produced.
- **Unsupported-claim rate.** Statements with no evidence span.
- **Replay fidelity.** Fraction of a recorded run reproducible at $0.
- **Cost and wall-clock**, reported alongside every quality number.

## What months of experiments already ruled out

This is the inheritance that makes a scoped project fast. Do not re-litigate these.

**Settings that must stay as they are**
- Memory similarity floor stays at 0.0. Live A/B was score-neutral (+0.017, p=0.70) and
  the 0.40 calibration does not transfer across models.
- Do not flip `got_dedup_enabled`. The original "-0.157, dedup hurts" was measured on
  superseded code; reconfirmation reversed the sign.
- `visit_sibling_url_dedup` stays off. Mechanism correct, 0 false positives, but all real
  collisions were one sidebar link. Fix the URL pool first.
- Roster-gate verdict downgrade stays off. Inverted signal: blocked 46 of 48 eligible
  cells including two scoring 1.00, passed only two, both at 0.40.
- `require_finish_tool` tested negative — costs step budget.
- Reason-first merge ordering: do not flip the default.

**Retired hypotheses**
- "Graph collapses on fan-out" (−0.266) is retired. Breadth is a dead tie across two
  backends and six tasks each (−0.002 / −0.007), including the full N=4→32 sweep.
- "Parallelism compensates for a weak model" was never testable on this suite and is not
  supported where it was tested.
- The quantitative-shape advantage (+0.321) sign-flipped to −0.216 on a second backend.
  Per-shape splits at n=3–6 are noise.
- Capability-tiered DAG Stage 0: live 2x2 validation was NO-GO.
- Task 047 is a genuine capability floor for qwen2.5:7b, not a broken task.

**Judgement calls that cost us**
- LLM judges degrade in ways that are hard to detect. The merge-AUC history is the
  standing warning. Prefer deterministic gates; where a model must decide, record the call
  so the decision can be replayed and second-guessed.
- Reasoning from absence produced seven retracted claims in a single night. No cost line
  did not mean free; no error rows did not mean no errors; a green test suite did not mean
  the code ran in production. Denominators come from the experiment design, never from
  the filesystem.

**Harness traps, each confirmed the hard way**
- `SEARCH_PROVIDER` defaults to **paid** Serper. Per-cell `usd` counts LLM tokens only, so
  a "local, free" run can bill real money and report zero.
- A dead cell writes no result file and is invisible to any analysis that iterates the
  results directory.
- `*_summary.json` and `*.jsonl` inflate a naive glob. Count only `*_r1.json`.
- Background jobs die past roughly one hour without `setsid nohup … disown`.
- 8-way slicing starves fast variants against `OLLAMA_NUM_PARALLEL=1`.
- The benchmark agent is a singleton; launching a second kills the first's run.

**Backend policy**
SearXNG is not required and is measurably worse — 4.00 searches/cell against Serper's
21.85, with visits/cell down 6.67 → 4.82, and the calibration ordering inverts under it.
Do not degrade retrieval to save money. Record once on Serper at full quality and replay
from cache. Never pool results across backends.

## Prior art

Verified against primary sources in
[`docs/research/EVIDENCE_COMPILER_CITATION_LEDGER_2026-08-31.md`](research/EVIDENCE_COMPILER_CITATION_LEDGER_2026-08-31.md).

- **PoT** (2211.12588) — the warrant for a deterministic derivation layer: the model
  formulates, an interpreter computes.
- **RAGChecker** (2408.08067) — claim-level diagnosis split across retrieval, generation,
  and overall metrics. The model for localizing where a score gap enters.
- **FActScore** (2305.14251) — atomic-fact decomposition scored against a reliable source.
  The shape of the output metric.
- **CRAG** (2401.15884) — a lightweight retrieval evaluator returning a confidence degree
  that triggers typed retrieval actions.
- **Self-RAG** (2310.11511) — a four-token state vocabulary: retrieval needed, passage
  relevant, output supported, response useful.
- **Chain-of-Note** (2311.09210) — per-document reading notes for noise robustness and
  rejection. **Trained, not prompted** — its numbers do not transfer to a stock model.
- **RARR** (2210.08726) — post-hoc attribution and repair. Weakest fit; by construction
  there is nothing to retrofit when answers are built from verified claims forward.

None of these does cross-record arithmetic with unit-dimension refusal and an abstain
path. PoT is closest and stops at executing a generated program: no unit algebra, no
provenance on operands, no abstain when an operand is missing. That is open space.

## Not in scope

- General task planning and tool selection. That is LangGraph's niche and DAG v2's.
- Beating LangGraph on mean score. The interesting comparison is risk-coverage and
  claim-level precision, where the baselines cannot compete on their current output
  contract.
- Training or fine-tuning. The constraint is a fixed weak model.

## Next

Scoping per subsystem, then an expanded development cycle. Immediate queue:

1. Search-result fixtures — unblocks deterministic comparison, ends the paid-live regime.
2. Wire the derivation layer against the `Extraction` records that already run live.
3. A purpose-built numeric suite; the effect is unmeasurable at n=24 on `core_long24`.
4. Fresh 046/047 comparison — the archive cannot supply it.
