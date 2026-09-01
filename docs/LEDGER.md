# Euglena Ledger

**Codename:** `ledger` · **Status:** pivot declared 2026-08-31 · **Repo:** webRAG (unchanged)
**Operative plan:** `docs/LEDGER_PLAN_2026-09-01.md` (supersedes the DAG v3 master plan's
thesis, metric order and build order) · **Latest handoff:**
`docs/handoffs/LEDGER_HANDOFF_2026-09-01.md`

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
| 3 | Deterministic derivation + unit refusal + abstain | **WIRED AND LIVE-MEASURED** (`evidence_graph.py` via `execution_evidence_loop.py`); typed `derive` action, verdict gate `LEDGER_DERIVATION_GATE` default OFF; live cell 210-231 run: 21/22 cells carry a graph artifact, 109 source + 16 derived nodes, zero invalid derivations, fabricated-arithmetic rate 0.0 (n=8) |
| 4 | Per-call audit log: prompt, response, timing, cost | **shipped** — `call_id` pairing (was file-order, broke under 32-way concurrency), `stage`/`node_id`, sampling params + `seed`, retry `attempts`, `scripts/trace_read.py`; `seed` is Ollama-only, no-op elsewhere (Anthropic has no such parameter) |
| 5 | Record/replay + counterfactual re-run | **corpus replay LIVE** (`connector_search_corpus.py` + `scripts/build_corpus.py`, 313 docs / $0.11 for the numeric suite) **+ counterfactual replay tooling shipped, provider-limited**: `scripts/replay_call.py` refuses replay without full-capture text rather than reconstructing lossily; `scripts/reverify.py` re-checks quotes/derivations offline, $0. Honest limit: without a seed a re-run is not comparable to its original — true on Anthropic by construction, and even seeded Ollama showed one cold-start exception (first call after model load) |
| 6 | Leak-resistant benchmark construction | keystone gates exist; 046/047 asymmetry fixed; **all 12 verify-leaf `optional_url` leaks now closed** (six fixed 2026-08-31 to match the two fixed earlier); `[LEAK]` lint severity in `scripts/validator_lint.py` makes it un-reintroducible, gated by `validator_lint_test.py` over `ACTIVE_SUITE_IDS` (still 59 — the numeric suite 210-231 is deliberately not promoted) |

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
- "Graph collapses on fan-out" (−0.266) is retired for **fan-out width** specifically —
  the N=4→32 sweep and the literal `breadth`-labeled shape are a dead tie (−0.002 / −0.007,
  two backends, six tasks each). This does **not** extend to **aggregation shape** (tasks
  whose answer requires combining evidence across branches — count/argmax/AND-filter):
  `AGGREGATION_SHAPE_FINDING_2026-08-30.md` measures graph losing those by **−0.461
  (t=−7.73, n=23)** on a blind classification of 59 tasks, with a stated mechanism
  (root-ward-only expansion context, siblings invisible to each other). Width and shape are
  different variables; both results stand. Widening a fan-out is cheap and parity-tied;
  making the branches jointly aware of what's already covered is the open, unmeasured fix.
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

Items 1-4 below are **done** as of 2026-08-31 (see
`docs/handoffs/LEDGER_PROGRAM_HANDOFF_2026-08-31.md`): search-result fixtures (corpus replay),
the derivation layer wired and live-measured, the 22-task numeric suite (210-231, out of
`ACTIVE_SUITE_IDS`), and a fresh three-arm run with evidence persistence (`ledgernum22`,
66/66 cells, superseding the old 046/047 gap). Open queue:

1. `sequential_react`'s evidence-persistence gap (fixed for `langgraph_react`, not this arm).
2. Powered re-measurement — n=22 rep=1 settles nothing against this repo's own n=61-111 power
   table; a repeat under corpus replay is $0 but costs GPU wall clock.
3. Whether the derived verdict is a working selective classifier for non-ledger arms — n=4-5
   ANSWER-tier cells scored *below* their own ALL rate, the wrong direction, too small to call
   an inversion but flagged as the first thing a powered follow-up must test.
4. Subsystems 1 (typed action queue) and 2 (DAG evidence-dependency analysis) — out of scope,
   starting points recorded in the handoff.

---

## Subsystem 5 status — frozen-corpus replay (2026-08-31)

**Shipped.** `SEARCH_PROVIDER=corpus` + `LEDGER_CORPUS_DIR` serve search from a frozen
document set ranked by a pure-Python BM25 index — no service, no GPU, no network, and
byte-identical across processes.

Why the altitude changed: exact-key fixtures hash the literal query, and an adaptive agent
never repeats a query, so a 289 MB record pass produced ~0 effective hits
(`scripts/BENCHMARK_NATIVE.md:14-19`). Ranking a corpus asks a different question — "what
does the frozen evidence hold for this query?" — so any phrasing, recorded or not, returns
results. Every arm can then query one identical evidence universe freely, which is what
makes a controller comparison a controller comparison.

| Piece | Where |
|---|---|
| Backend | `agent/app/connector_search_corpus.py` |
| Factory branch | `agent/app/connector_search.py` (`create_search_backend`) |
| Builder | `scripts/build_corpus.py` |
| Tests | `agent/tests/connector_search_corpus_test.py` (16), `agent/tests/build_corpus_test.py` (7) |

**First corpus cost $0.** Harvested 289 distinct documents (1.44 MB) from 5,973 stored
result cells — evidence already paid for. Live top-up via `scripts/prewarm_fixtures.py` is
a deliberate, budgeted step rather than a prerequisite.

**Spend policy.** A corpus miss falls back to live search and records the result, per the
chosen throughput-first policy. Bounded rather than silent: `LEDGER_MAX_LIVE_FALLBACKS`
(default 25) caps live calls, the resolved provider and document count are logged at
construction, and `live_fallbacks` / `provenance` make per-cell replay fidelity measurable
instead of assumed.

**Design detail worth keeping.** Absorbed live results index the *originating query*
alongside the result text. Indexing content alone means the very query that paid to fetch a
result cannot find it again and bills twice — caching by content is not caching by query.
A test drove this out.

**Known limits.** BM25 ranking is not Serper's ranking, so replay measures controller
behaviour over a fixed evidence universe, not end-to-end production retrieval; a headline
end-to-end number still needs a live run. A frozen corpus also cannot show a regression
caused by the live web changing.
