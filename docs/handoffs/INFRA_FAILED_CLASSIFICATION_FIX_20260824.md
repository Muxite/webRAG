# infra_failed classification: a wrong diagnosis corrected, two real bugs fixed (2026-08-24)

## Context

`docs/handoffs/BREADTH_SUITE_WEAKNESS_SWEEP_20260823.md` (open item 6) flagged that an 80-cell
paid sweep had 11/80 cells marked `infra_failed=true` by the test runner, and attributed this to
**ChromaDB init contention at concurrency=4 tainting `langgraph_react` cells that never touch
Chroma**. That attribution was never checked against the actual stored artifacts. This cycle
opened by doing that check, found the diagnosis was wrong, and fixed the real defects instead.
$0 spent — no benchmark or paid API calls this cycle.

## The prior diagnosis was wrong

Reading `agent/idea_test_results/paid_wide_sweep_20260823_rep1_*_r1.json` for all 11 flagged
cells: every one has `infra.ops` of either `["http_request", "visit"]` (9 cells) or
`["http_request"]` (2 cells). **Zero cells have any Chroma op in `infra.ops`.** The trigger was
ordinary outbound web-fetch failures — searches and page visits timing out or erroring at
concurrency, which is normal on a live web-research benchmark — not Chroma. Chroma was never
involved in any of the 11 flagged cells.

## Bug A — the real defect, and its measured impact was nil

`_summarize_infra` in `agent/app/testing/utils.py` set `failed = bool(any_infra_timing)` — an OR
over any single failed op in the whole cell. A cell with 14/16 successful `http_request`s and
10/11 successful `visit`s, which went on to produce a perfectly valid score, was flagged the
same as a cell that failed outright. On a web-research benchmark, occasional failed fetches are
normal operating condition, not a sign the cell's data is bad.

**This was checked, not assumed.** An audit re-ran 11 of the 13 analysis scripts in this repo
that exclude rows on `infra_failed`, both with and without the filter, against their actual
stored result artifacts:

- coverage-gate A/B, context_trim A/B, stall_recovery A/B: 0 of 12 pairs excluded by the filter
  — it was inert for all three, and none of the shipped default-on decisions were affected.
- Breadth pilot NO-GO (`BREADTH_PILOT_RESULTS_20260823.md`): 1 of 18 cells excluded.
  Un-filtering made the finding STRONGER, not weaker: -0.266 → -0.279, t -2.99 → -3.29.
- T1-6 backtrack fix (shipped at marginal p=0.033): 2 of 72 excluded, t identical to two
  decimal places (2.18 → 2.18).
- Graph-beats-seq_react reversal: 2 of 72 excluded, t 3.83 → 3.70.
- `capspec_report`: 8 of ~600 rows excluded, magnitudes shifted 0.005-0.03, no sign flips.
- `adaptive_ab_analyze.py` could **not** be verified — its documented input artifact
  directories are empty stubs. Reporting this honestly rather than estimating an effect: this
  one script's conclusions remain unverified against Bug A.

**Conclusion: the bug was real in mechanism and negligible in effect.** It was a latent hazard,
closed cheaply — not a remediation of corrupted results. The prior handoff's "higher priority
than it looks" framing was a reasonable a priori judgment given what was known at the time, but
it is not what checking the data actually showed.

## Bug B — the inverse defect: Chroma failures were invisible

While Bug A over-flagged web-fetch hiccups, Chroma failures were under-flagged — the opposite
problem, and the one the prior handoff's hypothesis actually needed (but didn't have).
`connector_chroma.py` emitted `chroma_add`/`chroma_query` timings without ever setting
`payload["status"]` or `payload["infra_failed"]`, so a real Chroma failure fell through
`_INFRA_TIMING_NAMES`'s whitelist and got classified as an ordinary task/model failure instead
of an infra one.

The most invisible path was `get_or_create_collection`: on failure it caught the exception,
logged it, and returned `None` — with **no `_record_timing` call at all**. Callers bail on
`coll is None` above their own timing call, so the failure left no telemetry trace whatsoever.
`init_chroma`'s exhausted-retry branch had the same gap.

**Fix**: reused `is_infra_llm_failure` from `connector_llm.py` (already generic, no circular
import needed) to stamp transport/timeout failures on `chroma_add`/`chroma_query`; added new
`chroma_get_or_create` and `chroma_init` timings so those two previously-silent paths now emit
something. Caller/logic errors (a bad `ValueError`, an embedding-dimension mismatch) are
deliberately left unflagged — those are real bugs in the calling code, not infra.

## Bug C — trace paths, and a correction to the initial framing

Raw model ids containing slashes (e.g. `openai/gpt-5-mini`) were interpolated directly into
trace file paths. `mkdir(parents=True)` and `open(..., "a")` both **succeed** against the
resulting unintended nested path — this is a silent success, not a swallowed exception, so
nothing ever surfaced it.

**Correction to the initial framing of this bug**: it was first described as "every paid run
lost its raw traces." That overstates it. Every execution module (`execution.py`,
`execution_compiled.py`, `execution_langgraph.py`, `execution_naive_discretion.py`,
`execution_sequential.py`) deliberately unlinks the trace file after a successful run, relying
on the `observability.timings` rollup instead — that unlink is intentional design, not data
loss. The real loss window is narrower: a run that raises *before* the unlink. Those traces did
survive, but nested one directory deeper than any caller expects, because of the slash. The
lasting artifact of the bug was orphaned empty directories, not lost data: 1,534 `*_openai` dirs
were found on disk (not the 161 originally estimated before actually counting), of which 1,388
were empty and removed; 146 non-empty ones were left in place (May-2026-era result JSONs from a
superseded path, not related to this bug).

**Fix**: added `sanitize_path_component()` to `agent/app/trace_recorder.py`, applied at all 6
call sites across the five execution modules.

## The fix: threshold and its empirical basis

An op is now infra-failed when its per-op infra-classified failure rate **exceeds 0.5**, or it
had zero successes outright. Strict `>` matters here, not `>=`: the worst observed rate among
the 11 wrongly-quarantined cells was exactly 8/16 = 0.500 on `http_request`, and that cell went
on to score fine — `>=` would have kept flagging it. Replayed against all 80 stored cells from
the wide sweep: 0 of 11 previously-flagged cells still flag under the new rule.
`failure_count`/`ops` are unchanged (every classified failure is still counted and named); a new
per-op `rates` field is added so a consumer can apply its own cutoff instead of trusting this
one.

Be honest about what this number is: **0.5 is a judgement call, not a derived constant.** It is
the smallest round cutoff that clears all 11 empirically-known-good cells; 0.3 would re-flag one
of them with no evidence its data was actually bad. Stated revision rule for later sessions: if
valid cells are later found scoring above 0.5, move the threshold up, never down — the failure
mode being fixed here is over-flagging, and a downward move would reintroduce it.

The replay itself used `error_count/count` from the stored rolled-up histograms as an
upper-bound proxy for the per-op rate, because stored result files only keep rolled-up counts,
not a per-record infra/non-infra classification. This is conservative in the safe direction (it
can only over-estimate a failure rate, never under-estimate one), but it is a proxy, not an
exact replay — noted here so a future session doesn't treat the "0 of 11 still flag" number as
more precise than it is.

## Arm-relevance risk: investigated and found imaginary

A specific concern was raised before shipping Bug B's fix: it makes `chroma_init` emit a
timing, and Chroma init/warmup runs unconditionally per cell subprocess
(`agent/app/idea_test_runner.py:1834,1848`) **before the arm is even known** — and
`execution.py:63` documents `connector_chroma` as "unused; kept for signature parity" for
baseline arms. So would a Chroma outage now flag cells belonging to arms that never use
Chroma at all — recreating, for real this time, the exact bug the prior handoff wrongly believed
already existed?

Traced end to end and answered **no**, safe by construction via two independent gates:

1. `ConnectorBase._record_timing` (`agent/app/connector_base.py:123`) is a no-op when
   `self._telemetry is None`. Setup-phase init/warmup runs before any `TelemetrySession`
   exists — telemetry is attached later, per-cell, in `AgentIO.__init__`
   (`agent/app/agent_io.py:70-71`). Any setup-phase failure is dropped on the floor; there is
   nothing yet to contaminate.
2. `chroma_init`/`chroma_get_or_create` timings are only emitted from inside `_ensure_ready()`,
   which is only reachable via a real Chroma method call. `langgraph_solver.py` has zero
   references to store/retrieve-Chroma calls, and baseline executors call no Chroma method at
   all. A stuck `chroma_api_ready=False` just sits inert for those cells — nothing ever calls
   the code path that would turn it into a timing.

## Live validation ($0, local qwen2.5:7b via ollama, real Chroma/Serper)

- **Check 1**: 3 cells at real concurrency=3 (110s wall vs 204s summed serially) — all 3 scored,
  0 flagged. One cell hit a genuine incidental fetch failure and correctly did not flag under
  the new severity gate.
- **Check 2**: induced a search outage. Result:
  `{"failed": true, "ops": ["http_request"], "rates": {"http_request": 0.6}}` — 3/5 failed,
  fires correctly.
- **Check 3**: induced a Chroma outage on the `graph` arm. Result:
  `{"failed": true, "failure_count": 45, "ops": ["chroma_init"], "rates": {"chroma_init": 1.0}}`.
  Note 45 attempts, not 1 — `_ensure_ready()` retries in-run after telemetry is attached, so a
  *persistent* outage is fully visible, not a single missed sample. Task 122 still completed and
  scored (the model reasoned without memory). Previously this would have been recorded as an
  indistinguishable plain task failure with no infra signal at all — that distinguishability is
  the substantive win of Bug B's fix.
- Consequence worth recording: the only Chroma failure mode still invisible after this fix is a
  setup-phase init failure that *recovers* before the run proper starts — which is harmless by
  definition, since the run never actually depended on the failed attempt.

## Commits

- `c7ab60e6` gate infra.failed on per-op failure severity not any single failed op
- `070d05e7` stamp infra_failed on chroma add/query/get_or_create/init failures
- `c09a195c` sanitize model ids before interpolating into trace file paths

(Preceded by four hygiene commits landing the prior cycle's uncommitted breadth-suite work:
`364acaf2`, `ba351857`, `00d1eb86`, `e15c7c63`.)

Offline suite: 6040 → 6070 passed, 18 skipped, 0 failures.

## Open follow-ups (not dropped, not addressed this cycle)

- `delete_collection`, `list_collections`, `delete_from_chroma`, `get_from_chroma` still emit no
  timings on failure — same invisibility pattern as Bug B, out of scope this cycle.
- `agent/app/idea_test_runner.py:1452` has a narrower, separate
  `safe_model = normalized.replace("/", "-")` for result-JSON filenames — a sibling instance of
  Bug C that was never unified with the new `sanitize_path_component()`.
- `langgraph_solver.py`: three sequential opt-in gates each re-run `graph.astream` and reassign
  shared locals — dense, would benefit from a readability pass. Flagged during this cycle's
  review, not a bug.
- Unconditional Chroma init/warmup for arms documented as not using it is confirmed harmless for
  classification purposes (see arm-relevance section above), but is still wasted setup work per
  cell.
- `adaptive_ab_analyze.py` conclusions remain unverified against Bug A — its documented input
  artifacts are empty stub directories.
- Carried over from the prior cycle, still open: `always_synthesize` needs a properly-sized A/B;
  `require_finish_tool` needs `max_steps` scaling before re-test; task 155's over-exploration
  pattern; task 045's validator strictness.
- ~$1.57 remains of the prior $5 OpenRouter authorization. This cycle spent $0.

## CORRECTION / FOLLOW-UP (2026-08-24, later cycle)

Bug B's fix, as shipped here, introduced a regression: `chroma_get_or_create`/`chroma_init`
emitted timings **only on failure**, with no paired success timing. Because the severity gate
this same cycle introduced (`rate > 0.5 or e["success"] == 0`) treats an op with zero recorded
successes as permanently failed, a single transient `get_or_create` blip would flag a cell
forever — reintroducing the exact over-flagging hazard this cycle's Bug A fix was written to
remove, in the very next commit that touched the code. Caught in the immediately following
cycle, not by any test failing here.

Fixed in `58826f49` by moving Chroma instrumentation into the shared `_op` wrapper so all 8 ops
emit on both success and failure. Full write-up, plus two further defects found and fixed in the
same later cycle (a search-init instrumentation gap, and a telemetry-suppression race introduced
while fixing it), in
`docs/handoffs/CONNECTOR_INSTRUMENTATION_UNIFICATION_20260824.md`.
