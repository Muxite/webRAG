# Idea Engine — Feature Catalog

A curated, opinionated backlog of features the standalone `ideaengine` library
could grow. Organized by category, each with effort and value tags so you can
prioritize. Effort: **S** (≤ 1 day), **M** (2–5 days), **L** (1–2 weeks),
**XL** (multi-week). Value: **★** to **★★★★**. Items marked **🎯** are the
ones I'd push to the top of the list if I had to pick.

The library boundary already drafted in
`~/.claude/plans/plan-ways-to-take-golden-storm.md` defines what's in v0.x
(extraction + plugin registry + comparison harness). This document is
*everything that comes after* — the features users would actually pay for
once the core is shipped.

---

## A. Observability & Debugging

### A1. 🎯 Streaming event protocol — `M`, ★★★★
Today the engine returns one big dict after `run()` completes. Add a
`StreamingHook` that fires named events during execution:

```python
async for event in engine.stream(mandate):
    match event.type:
        case "node_expanded": ...      # parent_id, children
        case "action_started": ...     # node_id, action, input
        case "action_completed": ...   # node_id, result_summary
        case "node_scored": ...        # node_id, score, rationale
        case "merge_started": ...      # parent_id, children
        case "final_synthesis": ...    # mandate, evidence_size
```

Drives live UIs, progress bars, real-time cost dashboards, and per-step
debugging without scraping logs. Server-Sent Events (SSE) wrapper for
trivial frontend integration.

### A2. Interactive graph viewer — `M`, ★★★
Standalone webapp (~1 page) that reads a `result.graph` dict and renders
the DAG with status colors, action types as icons, score heatmap, and
click-to-inspect-details. Already 80% present in
`testing/idea_test_visualize.py` and the existing frontend Dashboard;
generalize it into a reusable component shipped with the library.

### A3. Replay & time-travel debugging — `M`, ★★★
Already have `Checkpointer` (`idea_checkpointer.py`). Add a `Replayer`:

```python
runs = checkpointer.list_runs()
replayer = Replayer(runs[-1])
for step in replayer.steps():
    print(step.index, step.graph_snapshot)
    if step.index == 7:
        modified = step.fork()         # branch from step 7
        # ...try alternative path...
```

Lets developers explore "what if" branches without burning fresh LLM
calls. Also unlocks regression fixtures: bug at step 7 → save the
checkpoint → reproduce deterministically in tests.

### A4. Per-node trace timeline — `S`, ★★★
Render `telemetry.py` data as a Gantt-style timeline of every LLM /
HTTP / vector op per node. Surfaces slow stages instantly. Output:
HTML (Plotly) and ASCII (for terminal users).

### A5. 🎯 Cost & token budget dashboard — `S`, ★★★★
Aggregate `model_costs.py` data over a run / a day / a session.
Per-action, per-model breakdown. Plot. Saves real money — every team
running an LLM agent eventually asks "where did the budget go".

### A6. Step-level breakpoints — `S`, ★★
Pass `breakpoint_on={"action": "visit", "url_contains": "github.com"}` to
`engine.run()` and the engine pauses + returns control before that
action fires. Useful for inspecting state in a notebook.

### A7. Structured event log export — `S`, ★★
Today telemetry writes JSONL per run. Add OpenTelemetry exporter so
runs land in Honeycomb / Datadog / Grafana without bespoke scrapers.

---

## B. Performance & Throughput

### B1. 🎯 Multi-layer result cache — `M`, ★★★★
Engine already dedups identical actions within a run
(`graph.has_executed_action`). Promote that to a persistent, content-
addressed cache keyed by `(action, input_hash, model)`. Levels:

- **In-process LRU** — dev iteration.
- **Local disk** (sqlite) — single-user reuse.
- **Redis / Memcached** — team-wide share.

Cuts repeated benchmark runs from $$ to $.

### B2. Search-result cache — `S`, ★★★ (already in plan as
`CachingSearchBackend`)
Wrap `SearchBackend` so identical queries return identical results
across solvers and runs. Crucial for benchmark fairness; also great for
dev iteration.

### B3. Speculative branching — `L`, ★★
While the evaluator is scoring candidates A–E, start executing top-2
predicted winners eagerly. If the score selects one of them, you've
already paid the wall-time cost in parallel. Wasted work on the other
is bounded by `speculative_max_concurrent`.

### B4. Warm LLM connection pool — `S`, ★★
For high-throughput services, pre-warm N async HTTP connections to the
LLM provider so the first request of a run skips TLS handshake cost.

### B5. Prompt compression — `M`, ★★★
Big expansion / final prompts include the full path context. Add a
`PromptCompressor` strategy (summarize old ancestors, drop redundant
event log entries, deduplicate retrieved memories). Plug into the
existing prompt-assembly path. Cuts token cost meaningfully on long
runs without changing semantics.

### B6. Adaptive concurrency — `M`, ★★
Today `asyncio.gather(*leaves)` runs all unblocked siblings in parallel.
Add a token-bucket gate that auto-tunes parallelism to stay within
provider rate limits, with backpressure when 429s appear.

---

## C. Reliability & Resilience

### C1. Circuit breaker per backend — `S`, ★★★
If `SearchBackend.search` 5xxs three times in 60s, open the breaker
for 90s and either fall back to a chained backend or short-circuit
search actions to FAILED with `error_type="search_unavailable"`. Stops
runs from grinding away calling broken services.

### C2. Partial-result deliverable on timeout — `S`, ★★★
If `engine.run()` hits a wall-clock budget, the final-synthesis call
should still fire with whatever the graph has gathered so far. Already
half there via `final_allow_partial_success`; expose as `engine.run(
max_wall_seconds=180)`.

### C3. 🎯 Robust JSON repair — `S`, ★★★★
Several spots (`expansion.py:_parse_candidates`, merge action's
result parsing) handle malformed LLM JSON with ad-hoc try/except.
Centralize via a `parse_llm_json(text, schema=...)` helper that:

1. Tries strict json.
2. Strips ```json fences and trailing commas.
3. Uses Pydantic to coerce + validate against an expected schema.
4. Re-asks the LLM ("you returned invalid JSON — here's the parser
   error, return only valid JSON matching this schema") on failure.

Saves untold hours of debugging weird-character-in-JSON failures.

### C4. Idempotent action keys — `S`, ★★
Pair every action with an idempotency key. If a run crashes mid-action
and we restart, the action sees the same key and short-circuits with
the cached result instead of paying again. Pairs naturally with B1.

### C5. Self-healing prompt drift detection — `M`, ★★
Track the schema of the last 100 expansion responses per model. If
the LLM starts dropping fields ("provider changed model under the
slug"), surface a `prompt_drift_detected` event with a diff.

---

## D. Developer Ergonomics

### D1. 🎯 Synchronous facade for notebooks — `S`, ★★★★
Already in the plan (`engine.run_sync(mandate)`). Add a Jupyter "rich
display" hook so `engine.run_sync(mandate)` in a notebook auto-renders
the graph + deliverable inline.

### D2. Fluent config builder — `S`, ★★★
```python
engine = (
    IdeaEngine.builder()
        .with_llm(OpenAI(api_key=...))
        .with_search(Brave(api_key=...))
        .with_action_pack(WebActionPack())
        .with_max_steps(50)
        .build()
)
```
Reads cleaner than positional kwargs once the library has several
backend types.

### D3. Live prompt reloading — `S`, ★★
In dev mode, watch `prompts/defaults/*.md` and reload templates on
change. Saves the kill-restart loop while iterating on prompts.

### D4. `ideaengine` CLI — `M`, ★★★
```
ideaengine run "What is X?" --model gpt-5-mini --max-steps 30
ideaengine bench --suite ideaengine.suite:full --solvers ideaengine,langgraph
ideaengine inspect <run_id>
ideaengine replay <run_id> --step 7
ideaengine prompts list / show <key> / set <key> <file>
```
Standalone binary via PEP 621 entry point. Big adoption boost.

### D5. Typed action result models — `S`, ★★★
The current `ActionResult` is a fuzzy `Dict[str, Any]` with magic key
strings. Migrate to discriminated-union TypedDicts (`SearchResult`,
`VisitResult`, `ThinkResult`, ...). Catches whole class of bugs at
edit-time.

### D6. Schema-aware extractors — `M`, ★★
Today `NodeDetailsExtractor.get_url(details)` digs through ad-hoc keys.
Replace with Pydantic models for `IdeaNode.details` so consumers get
auto-complete and type checks for free.

---

## E. Advanced Reasoning Capabilities

### E1. 🎯 Reflection / critique pass — `M`, ★★★★
Before final synthesis, run a `Reflector` over the graph:

```python
critique = await reflector.review(graph, mandate)
# {"missing_evidence": [...], "weak_branches": [...], "contradictions": [...]}
```

If critique surfaces gaps, optionally fire a *recovery expansion*
(targeted search/visit nodes to fill the gap) before finalizing.
Massively improves answer quality on multi-source tasks.

### E2. Multi-agent debate — `L`, ★★
Two `IdeaEngine` instances with different prompts ("optimist" vs
"skeptic") generate competing deliverables; an arbiter LLM picks
the stronger one or synthesizes both. Already-popular pattern; cheap
to wire on top of the existing engine.

### E3. Cross-run memory — `M`, ★★★
`MemoryManager` is per-run today (namespace hashed from mandate). Add
a `SharedMemoryStore` that persists across runs with explicit user
scoping. Effective for repeated research on related topics ("I asked
about X last week; today's question about X' should use what I
learned").

### E4. Retrieval reranking — `S`, ★★
Today `MemoryManager` returns top-K by vector distance. Add an optional
reranker (cross-encoder, Cohere Rerank API, or Anthropic-style
score-this-passage) for higher-precision retrieval on long-context
synthesis tasks.

### E5. Tree-of-Thought speculative rollouts — `L`, ★★
Beyond GoT: at each expansion, do shallow LLM-only "imagine the
outcome" simulations for each candidate, score by simulated success,
then execute the most promising. Costs more tokens; better on tasks
where the right path isn't obvious from the candidate description.

### E6. Citation graph — `S`, ★★★
Currently citations live as URLs scattered through the deliverable.
Extract them into a structured `citations: [{claim, source_url,
passage, node_id}]` payload alongside `final_deliverable`. Enables
inline footnote rendering and verifies provenance — every claim
traceable to a node.

---

## F. Safety, Governance & Compliance

### F1. 🎯 PII / secret scrubbing in observability — `S`, ★★★★
`telemetry.py` records every LLM payload. If a mandate contains a
password or API key, it lands in JSONL. Add a `Sanitizer` that scrubs
common PII patterns (credit cards, AWS keys, SSNs, emails) before
writing — opt-out for users who need raw data.

### F2. Content policy hooks — `M`, ★★★
`SafetyHook` Protocol invoked before action execution. Implementations:
block-listed domains, prohibited-topic refusal, jailbreak detection,
output filter. Ships off by default; enterprise users plug in their
own policies.

### F3. Audit log with crypto receipts — `M`, ★★
Append-only audit trail of (mandate, deliverable, model, cost, sources)
per run, hash-chained or signed. Useful for regulated industries
(legal research, medical) where reproducibility + non-repudiation
matter.

### F4. Mandate input validation — `S`, ★★
Pre-flight mandate analyzer flags ambiguities, impossible asks,
contradictions ("find a URL that doesn't exist"). Returns warnings
before the run burns LLM cycles.

---

## G. Cost Optimization

### G1. 🎯 Tiered model routing — `M`, ★★★★
Per-policy model selection. Expansion candidates often look fine from
`gpt-5-nano`; only the final synthesis really needs the flagship.
Today `model_name` is one slot on the engine; add:

```python
EngineConfig(
    expansion_model="gpt-5-nano",
    evaluation_model="gpt-5-nano",
    merge_model="gpt-5-mini",
    final_model="gpt-5.2",
)
```

Reduces typical run cost 3–5× on benchmarks with no quality loss.

### G2. Hard budget caps — `S`, ★★★
`EngineConfig(max_cost_usd=0.50)`. Tracks running cost via
`model_costs.py`; aborts the run with `BudgetExceeded` exception when
the cap fires. Critical for shipped products with per-user limits.

### G3. Cheap-first then upgrade — `M`, ★★
First pass with a cheap model; if `goal_achieved=False`, re-run only
the failed branches with a stronger model. Saves money on the 80% of
mandates the cheap model handles correctly.

### G4. Smart batching across mandates — `L`, ★★
When the same engine instance serves multiple concurrent mandates,
batch their LLM calls into a single provider request where APIs
support it (Anthropic, OpenAI batch endpoint). 50% cost reduction
with 24h latency tradeoff; useful for analytics / overnight jobs.

---

## H. Multimodal & Data Types

### H1. PDF / DOCX visit support — `M`, ★★★
Today `visit` only handles HTML. Add MIME-type sniffing in
`AgentIO.visit` + a `PdfActionAdapter` that calls pdfplumber /
unstructured / mistral OCR. Hugely expands the engine's reach to
research / legal / scientific corpora.

### H2. Image understanding — `M`, ★★★
`visit` that lands on a page with embedded charts/diagrams currently
ignores them. Pass image references to a vision-capable model
(Claude 4.6 Vision, GPT-4o vision) when content includes them; surface
extracted facts as structured nodes.

### H3. Code search action — `M`, ★★
Built-in `CodeActionPack` with `code_search` (GitHub Code Search,
Sourcegraph, or grep over a cloned repo), `code_explain`, `code_diff`.
Targets a different audience (devtools) from the web pack.

### H4. Structured data action — `S`, ★★
`db_query` action backed by a `DatabaseBackend` Protocol (Postgres,
BigQuery, Snowflake). Combine with web research for hybrid
"my data + the web" answers.

### H5. Audio transcription — `S`, ★★
Visit-equivalent for YouTube/podcast URLs: download audio, run
Whisper, treat the transcript as page text. Already feasible as a
custom `VisitAction` subclass; ship as a default.

---

## I. Integrations

### I1. 🎯 LangChain / LangGraph runnable adapter — `S`, ★★★★
Beyond the comparison-harness adapters: ship an `IdeaEngineRunnable`
that satisfies LangChain's `Runnable` interface. Drops into any
existing LangChain pipeline as a single "deep research" step.

### I2. Slack/Discord bot — `S`, ★★
`@ideaengine What's the latest on X?` → streams progress messages
back to the channel. Production-ready with a 200-line wrapper.

### I3. GitHub / Linear / Jira issue ingestion — `S`, ★★
Mandate-from-issue: takes a GitHub issue URL, extracts the problem
description, runs the engine, posts the deliverable as a comment.
Slot for security advisories, customer-support triage.

### I4. MCP server — `M`, ★★★
Expose the engine as a Model Context Protocol server so Claude
Desktop / Cursor / Claude Code can call it as a tool. Aligns with
the ecosystem direction; tiny code surface to maintain.

### I5. Webhooks / Zapier — `S`, ★★
Simple HTTP-trigger interface for non-technical users. Receives a
mandate, returns a webhook with the deliverable when done. Pairs
with G2 (budget caps) for SaaS-style rate-limited access.

---

## J. Deployment & Operations

### J1. 🎯 Stateless engine + work-queue worker — `M`, ★★★★
The current `interface_agent.py` already consumes from RabbitMQ. Ship
that pattern as `ideaengine.workers.QueueWorker` with config for
RabbitMQ / SQS / Redis Streams / NATS. Standard scale-out story:
push a mandate to the queue, N workers process, result published.

### J2. Kubernetes operator — `L`, ★★
Custom resource `IdeaEngineMandate` whose controller spawns a Pod
running the engine, captures the deliverable, and writes it back to
the CR's `status`. Cloud-native packaging for users who already have
K8s.

### J3. Distributed checkpointer — `M`, ★★
Today's Redis checkpointer works for HA. Add a Postgres backend
(per-row, transactional) and a S3 backend (immutable archive). Lets
ops teams pick based on existing infra.

### J4. Autoscaling worker pool — `S`, ★★
Pair J1 with a queue-depth-based autoscaler (existing
`services/lambda_autoscaling/` already does this for webRAG —
generalize it).

### J5. Run lifecycle CRUD API — `S`, ★★
REST endpoints: `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/events`
(SSE), `DELETE /runs/{id}` (cancel). Layered over A1 (streaming events).
Drop-in for products that want their own UI.

---

## K. Testing Infrastructure

### K1. 🎯 MockLLMBackend with recorded transcripts — `S`, ★★★★
Already mentioned in the plan as a zero-dep test default. Record
real LLM calls into a fixture file; replay deterministically in CI.
Cuts test cost to zero. Vital for the comparison harness to ship
deterministic benchmarks.

### K2. Fuzz harness for mandate inputs — `M`, ★★
Generates pathological mandates (empty, only whitespace, gigantic,
prompt-injection attempts, language mismatch) and asserts the engine
exits gracefully on every one. Hardens production deploys.

### K3. Property-based tests for the DAG — `S`, ★★
Hypothesis-driven invariants: "after any sequence of valid step
transitions, every DONE node has an action_result", "no two children
of the same parent share an id", etc. Catches DAG-state bugs that
unit tests miss.

### K4. Regression-by-replay — `S`, ★★★
Take each historical run in `idea_test_results/`, extract the
mandate, re-run with the current code, diff the deliverable
semantically (LLM judge). Surfaces silent quality regressions
between releases.

### K5. Latency / cost budget tests — `S`, ★★
For each test in the suite, assert `wall_time < X` and `cost < Y`.
Catches performance regressions before benchmarks reveal them.

---

## L. Library / API Polish

### L1. 🎯 Hello-world docs + 5-min tutorial — `S`, ★★★★
The single most important "extra feature". A 200-line README and a
5-min Colab notebook lift adoption more than any code feature.

### L2. Versioned, stable error types — `S`, ★★★
Today engine internals raise bare exceptions. Define
`IdeaEngineError` hierarchy: `BackendError`, `ConfigError`,
`BudgetExceeded`, `ActionFailedAll`, `MandateValidationError`. Lets
callers catch specifically.

### L3. Stable observability schema — `S`, ★★
The current `observability` dict is ad-hoc. Pin a `v1` schema and
version it. Future shapes are `v2` so consumers don't break.

### L4. Plugin marketplace — `M`, ★★
Once `ActionPack` and `Backend` plugins exist, a "third-party packs"
registry where the community contributes (Tavily search, DuckDuckGo
search, PostgreSQL backend, etc.). Use the `entry_points` mechanism
in `pyproject.toml`.

---

## My recommended top 10 (if I had a quarter to build them)

1. **A1** — Streaming event protocol. Every other feature is easier with this.
2. **A5** — Cost/token dashboard. People decide whether to keep using the
   engine based on this number.
3. **B1** — Multi-layer result cache. Cuts dev iteration time 10×.
4. **C3** — Robust JSON repair. Eliminates the single largest source of
   flaky engine failures.
5. **D1** — Sync facade with Jupyter renderer. Notebook adoption is huge.
6. **E1** — Reflection / critique pass. Biggest single quality lever.
7. **F1** — PII/secret scrubbing. Required for any production deployment.
8. **G1** — Tiered model routing. 3–5× cost reduction with no quality loss.
9. **I1** — LangChain Runnable adapter. Drops the engine into an existing
   ecosystem with one line.
10. **J1** — QueueWorker package. Production deployment story.

Together: **M-effort each, except E1 which is M+. ≈10 weeks for one engineer
or 4 weeks for a small team.**

Pair these with the comparison harness benchmark numbers (Phase 3 of the
extraction plan) and you have a story that's hard to refuse: *"4× cheaper
than LangChain ReAct, with structural answer quality LangChain can't match,
production-deployed by a 200-line worker, observable end-to-end, and it ships
with a cost dashboard so you know what you're spending."*

---

## What this list deliberately omits

- **A built-in chatbot UI.** Too opinionated for a library; ship the
  streaming protocol (A1) and let users build their own.
- **A "no-code" workflow builder.** Different product entirely.
- **Auto-finetuning of routing policies.** Real value, but premature
  for v0.x — ship more concrete observability first so users can
  see where finetuning would help.
- **Model providers we don't already support.** The `LLMBackend` ABC
  makes adding providers trivial; ship a one-page tutorial instead
  of bloating the package.
- **Anything domain-specific (legal, medical, finance).** Those become
  third-party `ActionPack`s once the plugin system exists.

The goal is a *general-purpose research engine library* whose extension
surface is broad enough that domain-specific applications can be built
on top in days, not months. The features above are the ones that
maximize that extension surface.
