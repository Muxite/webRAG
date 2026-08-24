# Connector instrumentation unification: a regression from yesterday's cycle, fixed, plus two self-inflicted defects caught in review (2026-08-24)

## Context

`docs/handoffs/INFRA_FAILED_CLASSIFICATION_FIX_20260824.md` (yesterday, same date, separate
cycle) shipped `chroma_get_or_create` and `chroma_init` timings so Chroma failures would no
longer be invisible to `infra_failed` classification. This cycle opened by checking that fix
against the severity-rate gate it was meant to feed, and found the two had never been reconciled
against each other. $0 spent — no benchmark or paid API calls this cycle.

## The regression

`_summarize_infra` (`agent/app/testing/utils.py:81`) flags an op as infra-failed when
`rate > 0.5 or e["success"] == 0`. Yesterday's `chroma_get_or_create`/`chroma_init` timings were
emitted **only on failure** — there was no corresponding success timing. An op that never
records a success has `success == 0` permanently, by construction, regardless of how many times
the underlying call actually succeeded. So a single transient `get_or_create` hiccup — one
retryable blip in an otherwise-healthy cell — would flag the entire cell, forever, on every
future run that happened to hit it once. This is exactly the hair-trigger over-flagging the 0.5
threshold was introduced to remove, reintroduced by the very next commit that touched the same
code.

**Root cause worth recording**: the threshold fix and the Chroma fix were correct in isolation
and were reviewed against each other's *code*, not against each other's *invariants*. Nobody
asked "does every op this gate consumes actually have a success channel." The interaction was
nobody's explicit responsibility.

## Fixes shipped (6 commits)

- **`58826f49`** — Chroma instrumentation centralized in `_op` (`connector_chroma.py:207`), the
  funnel every Chroma round-trip already passes through. All 8 ops now emit on **both** success
  and failure, with `infra_failed` stamped via the reused `is_infra_llm_failure`. Folded in the
  duplicated `_on_op_success`/`_on_op_failure` helpers. `_op` takes `op_name`/`op_payload`, not
  `name`/`payload` — Chroma's own methods use `name=` and would collide via `**kwargs`.
  `chroma_init` was deliberately left failure-only and outside `_op`: an exhausted retry cycle is
  a genuine total outage, not a per-op signal, so there is no meaningful "success" timing to pair
  it with. Mirrors the existing `connector_sandbox.py:186` `_done()` precedent.
- **`6f7755f3`** — search-init instrumentation. `init_search_api()` in all three search backends
  emitted no failure timing at all — the same invisibility pattern the old `init_chroma` had
  before yesterday's fix — so a dead search backend was diluted into the generic `http_request`
  bucket instead of being named. Now emits a named `search_init` timing on both success and
  failure via two shared `ConnectorSearch` base-class methods. A missing API key or a
  401/403/422 response is deliberately **not** stamped infra — a bad key is a config problem, not
  a transport outage.
- **`71343deb`** — path sanitization consolidated. `sanitize_path_component` gained
  `replacement="-"` / `preserve_colon=True` defaults; because all 6 trace call sites already call
  it with no options, changing the defaults updated every site with zero edits at the call sites.
  Trace and result filenames now agree on convention. Two scripts that hand-rolled
  `.replace("/", "-")` (`eval_strategy_library_generalization.py:103`,
  `unified_bench_report.py:189`) now import the shared helper instead. `badmodel-lab/localagent/
  run_suite.py:59` fixed in place — it sanitized `:` but not `/`, the actually-dangerous
  character on Linux.
- **`fcf18a35`** — `langgraph_solver.solve()` extraction: two duplicated ~25-line gate blocks
  became a `_SolveState` dataclass plus `async _run_extension(...)`. Behavior-preserving,
  +176/-79 — it did not shrink the file; the win is deduplication, flagged as a readability
  concern in yesterday's handoff's open follow-ups.
- **`9e75f7d3`** — fixes a concurrency defect introduced by `6f7755f3` and caught in review (see
  next section) before it shipped to any live run.
- **`448406cf`** — Chroma init/warmup now skipped in per-cell subprocesses when no requested
  execution variant actually needs Chroma (see Phase 5 below).

## A second self-inflicted defect, caught in review before it could bite

`6f7755f3`'s first version suppressed the search-init probe's own `http_request` timing (so the
probe itself wouldn't double-count against the real query traffic) by temporarily setting
`self._telemetry = None` and restoring it in a `finally` block. This is not reentrancy-safe:
coroutine A saves the real telemetry object and nulls it; B enters the same method, reads
`self._telemetry` as already `None`, and saves `None` as "the original to restore"; A finishes
and restores the real telemetry correctly; B then finishes and restores what it saved — `None`.
**Telemetry is now permanently wiped for the rest of the cell**, silently, because
`ConnectorBase._record_timing` early-returns on `self._telemetry is None` rather than raising.

This was reachable, not theoretical: all three search backends gate real queries behind
`if not await self.init_search_api())`, with `search_api_ready` only set true *after* the probe
returns, and the graph engine runs sibling leaves concurrently via `asyncio.gather`
(`idea_engine.py:1047`) — exactly the breadth fan-out shape under active benchmarking this
month (`BREADTH_PILOT_RESULTS_20260823.md`, `BREADTH_SUITE_WEAKNESS_SWEEP_20260823.md`). The bug
is strictly worse than the double-count it was trying to avoid: a double-counted timing is a
minor accuracy nit, silently losing all telemetry for the remainder of a cell is a total data
loss with no error raised anywhere.

**Fix (`9e75f7d3`)**: a per-call `suppress_timing: bool = False` parameter threaded through
`ConnectorHttp.request()` and checked at its six recording sites. The flag lives in each call's
own stack frame, so it cannot leak between concurrently-running coroutines the way shared
instance state can. `connector_base.py` needed no change. The race test was verified to **fail**
against the committed pre-fix code (`assert None is <_RecordingTelemetry ...>` — i.e., it caught
the wipe) and to pass after the fix, and this was independently re-verified by the reviewer
rather than taken on the author's word.

## Phase 5: measurement overturned the plan's own assumption

The originating plan for this cycle expected the unconditional per-cell `init_chroma()`/
`_warmup_chroma()` call to be cheap enough to just document and skip. Measured instead of
assumed, it wasn't:

- **Healthy, real default** (`CHROMA_MODE=embedded`, `CHROMA_EMBED_DEVICE=auto` — resolves to
  CUDA on this machine): **~5.2–5.6s per cell**, dominated by the SentenceTransformer/CUDA cold
  load. Forcing `CHROMA_EMBED_DEVICE=cpu` instead: ~0.5s. So the cost is the embedding function
  choice, not embedded-vs-http Chroma mode.
- **Degraded** (Chroma unreachable): ~1.6–2.0s per cell, plus a measured **double-charge** —
  `_warmup_chroma`'s `get_or_create_collection` call re-triggers a second full `init_chroma()`
  retry cycle via `_ensure_ready()`, because `chroma_api_ready` is still `False` after the first
  failed attempt.
- The ladder driver spawns one fresh subprocess per cell, so none of this amortizes across a run.
- Waste is conditional on the arm mix in a given matrix: a pure graph-arm ladder run wastes ~0
  (graph genuinely uses Chroma); an arm-comparison study (graph vs `langgraph_react`, graph vs
  baseline — i.e. exactly the breadth pilot and engine-design-review shapes) wastes on roughly
  half its cells. Worked out to ~190s of pure overhead on a representative 72-cell mixed matrix.

Also overturned: the plan asserted the execution arm isn't known yet at Chroma-setup time, so
skipping would need new plumbing. False — `execution_variants` is already parsed at
`idea_test_runner.py:1710`, 124 lines before the setup loop, in the same function. No lazy-init
or new plumbing was needed.

**Fix (`448406cf`)**: a no-Chroma allowlist,
`_NO_CHROMA_VARIANTS = frozenset(BASELINE_VARIANTS) | frozenset(OFFTHESHELF_VARIANTS)`, imported
from the authoritative dispatch tuples already in `testing/runner.py`. It's inverted so an
unknown or future variant automatically defaults to *needing* Chroma — the fail-safe is
structural (unrecognized → assume it needs Chroma → do the init), not a manually-maintained
special case that silently rots. `ConnectorChroma` is still constructed either way; only the
init/warmup calls are skipped.

## Sanitizer unification: the literal version was rejected on evidence

The originating request was to "unify" the two sanitizers (trace paths, result filenames) onto
one convention. Literal unification (collapsing `:` the same way `/` is collapsed) was
investigated and found actively harmful, so it was **deliberately not done**:

- 8 analysis scripts hardcode colon-bearing model tokens directly in globs/regexes
  (`analyze_ladder_*`, `analyze_context_trim_ab_*`, `analyze_coverage_gate_ab_*`,
  `analyze_breadth_pilot_v2_*`, `analyze_stall_recovery_ab_*`). Collapsing `:` → `_` in filenames
  makes these match zero files and silently print an empty analysis instead of erroring.
- Roughly 1,737 colon-bearing and 1,171 provider-hyphen artifacts already on disk would stop
  matching any newly-produced run.
- `_` is the field delimiter in the result-filename template, so collapsing colons into `_` makes
  the model token indistinguishable from a field boundary. `-` is the correct replacement
  character for result names, and colons were never actually the bug — only `/` creates
  directories on Linux, which is the one thing that needed fixing.
- Result-JSON filename output was proven byte-identical across 7 representative model-id shapes
  before and after the change. Three of the 8 dependent analysis scripts were re-run against real
  stored artifacts and confirmed to still produce non-empty output.

## Live validation ($0, local qwen2.5:7b, real Chroma/Serper)

- **Check 1 (the regression itself)**: 3 cells, `graph` arm, concurrency=3, healthy Chroma.
  `chroma_get_or_create` now shows non-zero successes in every cell (1/0, 2/0, 1/0 success/fail
  counts); all three cells report `{"failed": false, "failure_count": 0, "ops": [],
  "rates": {}}`.
- **Check 2**: dead Chroma still flags correctly —
  `{"failed": true, "failure_count": 23, "ops": ["chroma_init"], "rates": {"chroma_init": 1.0}}`;
  the task still completed and scored 0.25, reasoning without memory rather than crashing.
- **Check 3**: dead search backend →
  `{"failed": true, "ops": ["search_init"], "rates": {"search_init": 1.0}}`, and `http_request`
  is **absent** from the timings entirely — confirming the suppression fix works and no longer
  leaks into the generic bucket. (The first attempt at this check used a task that never called
  search at all, making a negative result vacuous rather than informative; it was re-run with a
  task that does call search before being trusted.)
- **Check 4**: trace filename colon preserved, single file, no unintended nested directory —
  verified by exercising the production sanitization code path directly, because traces are
  unlinked on success by design and all runs in this check succeeded. Stated explicitly here
  because the method matters: this was not inferred from run behavior, it was checked directly.

## Test counts

6070 → 6111 passed, 18 skipped, 0 failures throughout.

## Open follow-ups to record (do not drop)

- **`connector_llm.py`'s `self.last_usage` is shared mutable instance state** — a last-write-wins
  attribute set after each LLM call, read and cleared by `pop_last_usage()`. If one `ConnectorLlm`
  instance is shared across concurrent in-flight calls, one call can read or prematurely clear
  another's usage, silently misattributing `llm_calls` and cost. Same trigger condition as this
  cycle's search-probe race: sibling leaves via `asyncio.gather`. **Not investigated this
  cycle** — flagging prominently, because "shared connector reused across concurrent leaves" has
  now produced two real defects in one day, which argues for an audit of that pattern generally
  rather than another one-off fix when this one is eventually hit.
- `test_empty_variant_list_skips` documents an unreachable branch —
  `_parse_execution_variants` always ends `return out or ["graph"]`, so `main()` can never
  actually be handed an empty list. Non-blocking, noted during review.
- Documented-not-changed path conventions, left alone deliberately:
  `badmodel-lab/run_cell.sh:95`'s `tr '/:' '--'`, and `split("/")[-1]` in
  `render_gallery.py`/`recovery_curve.py` (drops the provider prefix from the display name).
- `run_id`/`test_id` interpolated unsanitized at `idea_test_runner.py:1455,1960`,
  `json_telemetry.py:60`, `contract_log.py:61`, `plan_library/retrieval_log.py:67` — investigated,
  no live break found: `IDEA_TEST_RUN_ID` is pre-sanitized upstream by `run_cell.sh`, and the
  fallback value is a UTC timestamp. Recommendation is to leave these alone; sanitizing now would
  change filenames and re-trigger the exact corpus-matching hazard the sanitizer-unification
  section above avoided.
- Still open from earlier cycles, carried forward unchanged: `always_synthesize` needs a
  properly-sized A/B; `require_finish_tool` needs `max_steps` scaling before re-test; task 155's
  over-exploration pattern; task 045's validator strictness; `adaptive_ab_analyze.py` remains
  unverified against the old severity bug from yesterday's cycle (its documented input artifacts
  are empty stub directories).
- ~$1.57 of the OpenRouter authorization remains untouched. This cycle spent $0.

## Lesson worth stating explicitly

Both defects fixed this cycle were self-inflicted by the immediately preceding cycle of work, and
both were caught by review and direct investigation rather than by any test failing on its own.
The pattern in both cases is the same: two independently-correct changes whose *interaction* no
one owned — a severity gate assuming every op has a success channel, and a suppression mechanism
assuming single-flight execution. Worth recording as a process note going forward — when a change
touches a shared gate or shared instance state, check the callers' concurrency and completeness
assumptions explicitly, not just the new code's own correctness — not just logging this as a
one-off incident.

## Commits

- `58826f49` record chroma op timings symmetrically on success and failure via a shared `_op`
  wrapper
- `6f7755f3` record search-init health probe timings symmetrically across all three backends
- `71343deb` consolidate path sanitization onto one shared helper across trace and result
  filenames
- `fcf18a35` extract a shared `solve()` extension helper to de-duplicate the two corrective gate
  blocks
- `9e75f7d3` fix search-probe timing suppression to use a per-call parameter instead of racy
  shared telemetry state
- `448406cf` skip chroma init and warmup in per-cell subprocesses when no requested execution
  variant needs it
