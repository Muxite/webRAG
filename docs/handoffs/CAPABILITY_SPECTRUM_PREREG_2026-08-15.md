# Pre-registration: capability-spectrum sweep, DAG v2 vs off-the-shelf LangGraph (2026-08-15)

Written **before** any live cell ran. Authorized spend: **$5**, hard-bounded by the driver's
`--real-budget` (true OpenRouter key delta) plus a per-run `IDEA_TEST_USD_CEILING`.

## The question

Not "is DAG v2 better" — that framing has no answer. The question is:

> **At which point on the model-capability curve, if any, does DAG v2's structure earn its cost
> against an agent loop anyone can `pip install`?**

The project thesis says structure should buy the most where the model is weakest. If that is true,
the DAG v2 − LangGraph gap should be **largest at the bottom of the capability ladder and shrink as
the model gets better**. If the gap is flat, or inverts, the thesis is in trouble on this axis.

## Prior probe ($0, run before this doc, result recorded here)

Ollama `/v1/chat/completions` with a `tools` payload, all 10 local models:

| model | LangGraph-able (tool-calling) | DAG-v2-able (JSON as text) |
|---|---|---|
| qwen2.5:0.5b | yes | yes |
| tinyllama:latest | **NO — HTTP 400 `does not support tools`** | yes |
| llama3.2:1b | yes | yes |
| qwen2.5:1.5b | yes | yes |
| gemma2:2b | **NO — HTTP 400** | yes |
| phi3:mini | **NO — HTTP 400** | yes |
| llama3.2:3b | yes | yes |
| qwen2.5:7b | yes | yes |
| llama3.1:8b | yes | yes |
| qwen2.5:14b | yes | yes |

3 of 10 are **categorically excluded** from the off-the-shelf arm: `create_react_agent` binds tools
through the OpenAI function-calling API, and a model with no tool template is rejected at the API
layer before inference happens. The native engine asks for JSON in plain text and parses it itself,
so it can at least *attempt* all 10.

**Honest scope of that claim:** this is a property of the *off-the-shelf path*, not an inherent
model limit — a hand-written text-parsing ReAct loop could drive gemma2. The claim is exactly and
only: *`pip install langgraph` + `create_react_agent` cannot run these models; the native engine
can.* Whether it runs them **usefully** is what the live sweep below is for. A 0.00 score from an
engine that ran is not obviously better than a crash — that has to be argued, not assumed.

## Design

**Arms (3).** Two axes crossed, run as four driver invocations under separate `--run-id`s.

| arm | variant | `IDEA_TEST_ARM` | what it is |
|---|---|---|---|
| `graph:baseline` | `graph` | `baseline` | bare model in the DAG scaffold, adaptive OFF |
| `graph:good_adaptive` | `graph` | `good_adaptive` | DAG v2 proper (the proven winner) |
| `langgraph` | `langgraph_react` | `good_adaptive` (inert) | third-party ReAct loop |

The arm label on the LangGraph cells is inert for every knob that variant reads, **except** that
`good_adaptive` preserves connector-retry parity. `baseline` pins
`connector_retry_on_failure_enabled: False`, which would silently handicap that arm alone and
violate `langgraph_solver`'s own F16 fairness invariant.

Including `graph:baseline` is what separates *"the DAG scaffold"* from *"the adaptive machinery"*.
Without it, a DAG-v2-beats-LangGraph result cannot distinguish the engine from the wrapper.

**Models (8), spanning four tiers.**

| tier | models |
|---|---|
| super-bad local, no tool template | `tinyllama:latest`, `phi3:mini` |
| super-bad local, tool-capable | `qwen2.5:0.5b` |
| bad local | `llama3.2:3b` |
| decent local | `qwen2.5:7b` |
| super-cheap API ($0.027/$0.201 per M) | `meta-llama/llama-3.2-1b-instruct` |
| cheap API ($0.10/$0.40 per M) | `openai/gpt-4.1-nano`, `google/gemini-2.5-flash-lite` |

Two cheap-API models at the same headline price from different vendors, so a result cannot be an
artifact of one family's quirks.

**Tasks.** Shape-stratified, because the prior smoke suggested chains are the only shape where
adaptive pulled away and pooling would average that win into a wash.

- API (5 tasks, R=2): `134` chain, `135` chain, `122` 4-entity fan-out, `140` disambiguation,
  `128` conflicting-source. Two chains deliberately — to test whether 134's win replicates or was
  a fluke.
- Local (2 tasks, R=1): `134` chain, `122` fan-out. Reduced for wall-clock; local cells are free
  in dollars and expensive in hours.

**Power.** This is an exploratory sweep, not a confirmatory run: 10 observations per (model, arm)
on the API side, 2 on the local side. It is powered to detect *categorical* and *large* effects
(a crash, a floor-vs-ceiling gap), not a 0.05 score difference. **No p-values will be quoted.**

## What each outcome would mean — stated before the data

| result | reading |
|---|---|
| DAG v2 gap **largest on weak models, shrinking as capability rises** | thesis supported on this axis |
| gap **flat across the spectrum** | structure is a general wrapper win, not a weak-model rescue — the thesis's specific claim is unsupported even if the engine is fine |
| gap **inverts** (LangGraph better on weak models) | thesis refuted on this axis; the planning machinery costs more than it returns exactly where it was supposed to pay |
| LangGraph **ties on score at ~1/10 cost on every shape but chains** | a **wash at best**. The honest headline becomes "structure buys chain-depth", not "structure buys quality" |
| weak models score ~0 in **both** arms | the tasks are out of range for the tier and this sweep says nothing about either engine — a null result about the benchmark, not the agent |

That last row is the most likely failure mode and the easiest to mistake for a finding.

## Known confounds, recorded up front

1. **Step budgets are not equalized.** graph arms get `max_steps` 50–90, LangGraph gets 25
   (`IDEA_TEST_LANGGRAPH_MAX_STEPS`). This is the unresolved Q1 ("what is the unit of fairness")
   and this sweep does **not** resolve it. Any score gap is confounded with budget.
2. **Cost is the honest co-primary metric.** Score-only tables hide that LangGraph ran 7–13×
   cheaper in the prior smoke.
3. **`llm.calls` is ≈2× the real call count** (`ConnectorLLM` emits two `connector_io` events per
   logical call, and the LangGraph arm was deliberately matched to that doubled scale). Do not
   quote it as a literal count. `search.count` is a **document** count, not a search-call count.
4. **`infra_failed` cells are not quarantined** by `level_ladder`/`gate_report`. Check for them
   manually before reading any aggregate.
5. Local models run at `OLLAMA_CONTEXT_LENGTH=16384`; the graph engine's merge/finalize prompts can
   exceed that and be silently truncated. A weak local model's low score may be a context artifact.
