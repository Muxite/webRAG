# Pre-Barrage Audit — features, observability, and what to measure

Session goal (user): *"look for remaining features and things that can be improved or
measured before a final barrage of tests to get statistics; better graph methods to draw;
improve observability; make webRAG a drop-in replacement for a completions endpoint."*

This doc is the audit + a record of what was built. Companion: `COST_BENCHMARK_HANDOFF.md`
(the run recipe and prior rounds). Everything below is offline / no live-$ unless noted.

> **Campaign closed out (2026-07-08)** — see the banner at the top of `COST_BENCHMARK_HANDOFF.md`
> and [`linkedin_package_38tests_2026-07-08/`](../../../linkedin_package_38tests_2026-07-08/README_LINKEDIN.md).
> §5 below (performance follow-ups) is still open/deferred, not done.

---

## 1. Built this session (all offline-verified)

| Area | Files | What | Verify |
|---|---|---|---|
| **DAG visualizer** | `testing/dag_visualizer.py` | Draw the *whole* DAG as a **square, ≥1920px, configurable** PNG. Compiled plans (wave-layered), runtime graphs (longest-path), and topology-only `plan_structure`. matplotlib-only (no graphviz/networkx). CLI: `python -m agent.app.testing.dag_visualizer RESULT.json -o dag.png --size 1920`. | `tests/dag_visualizer_test.py` (9) |
| DAG per-run hook | `idea_test_runner.py` | Opt-in `IDEA_TEST_RENDER_DAG=1` emits `<result>.dag.png` per run (px via `IDEA_TEST_DAG_PX`). Best-effort, never fails a run. | byte-compile |
| DAG demo | `scripts/render_dag_examples.py` | Renders the 5 cached compiled plans + a synthetic runtime graph for eyeballing. | visual |
| **Completions endpoint** | `shared/openai_compat.py` (pure), `gateway/app/openai_router.py` (queue-backed), `agent/app/completions_api.py` (in-process shim) | OpenAI `/v1/chat/completions` + `/v1/models`, **non-streaming**. `messages`→mandate, engine result→`choices`/`usage` + a non-standard `euglena` block (sources, grounded, cost). Gateway path mounts at `GATEWAY_OPENAI_COMPAT=true` (Supabase JWT = API key, consumes 1 credit). Shim runs `engine.run()` in-process, serialized. | `shared/tests/openai_compat_test.py` (9) + `agent/tests/completions_api_test.py` (route, skips w/o fastapi) |
| **Plot sizing** | `scripts/recovery_curve.py` | `_plot` now square ≥1920px (`--size/--dpi`), **CI95 error bars** (already computed, never drawn before), legend below the axes, shortened labels, best-point annotations. | regenerated |
| **Stats hardening** | `scripts/level_ladder.py` | success now `mean±ci95`; new **significance block**: `graph_compiled` vs each baseline per level with Δscore, Cohen's d, and a strict **CI-disjoint** verdict. Turns REPEATS=3 into a defensible win/loss instead of bare means. | ran on `xshape_full_20260615_164736` |

Drop-in usage (shim): `OPENAI_BASE_URL=http://localhost:8088/v1` then any OpenAI SDK with
`model="euglena-graph"`. Needs the same env as a benchmark run (model + search keys, Chroma).

---

## 2. Measurement gaps to close BEFORE the barrage

Ordered by leverage. The first is the one that decides whether the barrage produces a
**defensible** number.

1. **Variance was being thrown away.** Only `recovery_curve.py` used REPEATS≥2 (CI on score);
   `level_ladder.py` and `summarize_bench.py` reported bare means. **FIXED** in `level_ladder`
   (mean±CI + CI-disjoint significance). `summarize_bench.py` still means-only — either route
   the barrage's per-level read through `level_ladder` (recommended) or port the same `_agg`.
   For a *publication* number, raise per-model repeats (n=3 pools to large-n only because
   models are pooled) and use a real paired test (scipy Wilcoxon/Welch) — the CI-disjoint
   verdict is deliberately strict but is a normal-approx proxy, not a p-value.

2. **Captured-but-unplotted metrics.** `summarize_observability` records, but no chart shows:
   `grounding` (verdict + replans), `fixtures` hit/miss (evidence parity — important when the
   barrage replays), per-connector `timings`, `chroma.store/retrieve`, `decisions.trace`.
   At least **groundedness** and **fixture miss-rate** deserve a per-variant bar before the
   barrage — grounding is a headline trust signal and 0 native-graph grounding (0.25) vs
   compiled (0.80) is part of the story.

3. **The thesis economics aren't visualized.** The compiler cost block + `plan_structure` are
   captured per compiled run, but nothing plots **amortized compile cost vs N executions**
   (pay-once-offline is the whole pitch). A small "break-even N" chart (compile $ / (premium$
   − cheap$) per task) would make the headline self-evident.

4. **Fixture parity for 050–054.** Per the handoff, these URL-free tasks record on the
   reference pass and replay-or-record for cheap models — *not* byte-identical evidence. Before
   the barrage, prewarm (`scripts/prewarm_fixtures.py`) so every arm reads identical pages, then
   run strict-replay (`IDEA_TEST_FIXTURES=replay_strict`). Otherwise variance includes
   evidence drift, not just model behavior.

5. **Saturation / floor tests dilute signal.** 026 (≈1.0) and 019 are near-saturated; keep 026
   only as a sanity floor. The discriminating subset is `040,041,042,043,044` + the cross-shape
   `050–054`. Don't average saturated tests into the headline.

---

## 3. Recommended barrage config (when the user authorizes live-$)

Mandatory env from the recipe: `IDEA_TEST_CONCURRENCY=1`, `IDEA_TEST_PARALLEL_ACTION_LIMIT=1`,
keys via `services/keys.env` (CRLF → `tr -d '\r'`), `CHROMA_URL=http://localhost:8001`.

```
RUN_ID=barrage_$(date +%Y%m%d_%H%M%S)   # stamp once, pass to --run-id for clean aggregation
# prewarm fixtures (reference pass) -> strict replay for every arm
IDEA_TEST_FIXTURES=replay_strict
IDEA_TEST_RUNS=5                         # >=5 tightens CI vs the current 3
variants: graph_compiled, sequential_react, graph, naive_rag, parametric
tiers: 0,10,20,40
models: 4 cheap + reference ceiling
IDEA_TEST_RENDER_DAG=1                   # emit a DAG png per run for the gallery
```
Then: `level_ladder.py --run-id $RUN_ID` (now with CI + significance),
`recovery_curve.py --run-id $RUN_ID --size 1920`, `gate_report.py --run-id $RUN_ID`.

Watch: verify-node usage (planner still rarely selects `verify`); compiler over/under-
decomposition (logged via `plan_structure`); 052 needs author `--max-tokens 4096`.

---

## 4. Nice-to-haves not done (deferred, lower leverage)

- Apply the ≥1920 square standard to the full `visualization_*.py` suite (25+ plots,
  hardcoded figsizes). The headline (recovery curve) + DAG are done; the rest is a big,
  low-value sweep — do it only if those plots go in the writeup.
- SSE streaming for the completions endpoint (chose non-streaming; the agent isn't a token
  streamer — would stream coarse node/tick progress then the answer).
- Per-model (not pooled) significance with a real paired test once repeats are higher.

## 5. Performance follow-ups (deferred, diagnosed 2026-07-06 — do NOT apply mid-barrage)

Root causes of barrage wall-clock time, from a read-only investigation of a live `barrage24b`
run. Safe to pick up once the current barrage (and any rescoring off it) is fully done, since
any of these could change measured latency/cost and must not be mixed into an in-flight run.

- **No per-task connector instances.** `ConnectorLLM`/`ConnectorHttp`/`ConnectorSearch` are one
  shared, mutable instance across the whole runner process (`set_model()` mutates
  `self.model_name` in place; `last_usage`/`total_usage` are plain attributes). This is *why*
  `IDEA_TEST_CONCURRENCY=1` is mandatory today — concurrent test-runs would clobber each other's
  model selection and cost/telemetry attribution. Fix: give each test-run its own connector
  instances (or at least per-task model/telemetry scoping) so concurrency can be raised safely.
  This contradicts an older memory note ("per-task connectors unneeded") — that note is stale
  against the current shared-connector code and should be corrected when this is picked up.
- **~~`sequential_react` has no fixture cache for search.~~ RETRACTED 2026-08-31 — this was wrong.**
  `ConnectorSearch` subclasses `ConnectorHttp` (`connector_search.py:104`) and the fixture hook is
  inside `ConnectorHttp.request` (`connector_http.py:125-155`), so search queries and their health
  probes have always recorded and replayed. The `make_key` `json_body` slot exists specifically for
  Serper's POST-body search. The real problem is that keys are a sha256 over the exact request
  including query text (`web_fixtures.py:74-81`), so a variable-query agent almost never hits the
  cache — measured at ~0 effective hits from a 289 MB record pass
  (`scripts/BENCHMARK_NATIVE.md:14-19`). See `docs/LEDGER.md` subsystem 5 for the corpus-replay
  approach that replaces exact-key lookup.
- **`sequential_react`'s per-step loop is serial by construction** (`execution_sequential.py`,
  plain `for step in range(max_steps)`, ~30 LLM calls/test at ~6.9s avg latency each, up to 24.8s
  under retry) — the single biggest wall-clock driver in a batch. Not a bug, just worth knowing
  it's the long pole; any future speed-up here (batching steps, shorter default `max_steps`) needs
  care since it would change the baseline's own behavior, not just harness overhead.

**Do NOT touch as part of any of the above:** `IDEA_TEST_CONCURRENCY=1` /
`IDEA_TEST_PARALLEL_ACTION_LIMIT=1` (correctness-load-bearing until per-task connectors exist),
`_votes_for_model` / price-aware voting k, `IDEA_TEST_COMPILED_CONCURRENCY`, and retry/backoff
settings (`DEFAULT_DELAY`/`JITTER_SECONDS`) — these are either core to the thesis being measured
or rate-limit resilience, not incidental slowness.
