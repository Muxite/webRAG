# Codebench: unified sandbox tooling, cross-benchmark reporting, and a first live run

## Context

Serper (the live search provider) is down (403 Unauthorized, confirmed live) and Part D Stage 1's
composer validation is blocked on it. Rather than wait, this work pivots to codebench — the
Docker-sandboxed coding-task benchmark (`agent/app/idea_code_tests/test_c01`..`c52`.py` +
`badmodel-lab/codebench/`) — which never touches the live web except through an internal,
network-isolated SearXNG instance, so it's completely unaffected by the outage.

Codebench was built and adversarially reviewed as of `CODEBENCH_HANDOFF.md` (2026-08-03) but has
never been run at scale — only single-cell smoke tests. Three things stand between it and a real
run:

1. **The file-tool interface is split across two independently-drifted implementations.** The
   native GoT engine's `LeafAction` pack (`idea_policies/extra_actions/sandbox_tools.py`) and
   codebench's own ad hoc dispatcher (`testing/execution_compiled_code.py`) both wrap the same
   `SandboxConnector`, but already disagree on which path-argument aliases they accept, and
   codebench is missing a find/grep-style tool entirely — an agent navigating an unfamiliar repo
   inside the real sandbox has no way to search files by pattern. There's also a real, reproducible,
   uncaught crash bug (`UnicodeEncodeError` on malformed content) that can take down an entire
   native-engine run. The explicit design goal here (user's words): "regular webRAG systems should
   be completely capable of sitting at the helm of codebench tasks and working through them — the
   interface should just work." That means one shared dispatcher, not two that drift.
2. **Codebench's results live in a format the QA benchmark's tooling can't read**, and vice versa.
   `badmodel-lab/results/cells.jsonl` (QA) and `badmodel-lab/codebench/results/runs.jsonl` (code)
   are explicitly documented in `score_and_record.py`'s own docstring as "a SIBLING to cells.jsonl's
   schema, not a literal extension of it" — a deliberate, acknowledged fork. There's a third format
   too: the main harness's per-cell `agent/idea_test_results/*.json`.
3. **No one has actually run codebench at real scale yet**, so there's no data on where the
   compiled-scaffold engine actually struggles on coding tasks — which is the actual "improve
   performance" target, and can't be scoped further until real failures exist to look at.

The three phases below are meant to run in that order — each is a prerequisite for the next being
useful (a live run on a broken tool interface produces noise, not signal; a live run whose results
can't be compared against the QA benchmark's existing conventions is harder to act on).

## Phase 1 — One shared sandbox-action dispatcher

**Problem being solved**: `sandbox_tools.py` (native engine) and `execution_compiled_code.py`
(codebench) each independently translate `{action_name, args}` into a `SandboxConnector` method
call. They've already diverged:
- Path-argument aliases: native pack accepts `path/file/relpath/filename/dir/directory`; codebench
  accepts `path/relpath/file/filename/target/file_path` — overlapping but not identical.
- Action vocabulary: native pack has 8 actions (`read_file, write_file, list_dir, patch_file,
  count_lines, word_count, head_file, disk_usage, find_files`); codebench's loop only wires 4 of
  those plus `run_python, run_pytest, search_web` (`write_file, read_file, list_dir, patch_file,
  run_python, run_pytest, search_web`) — missing `count_lines, word_count, head_file, disk_usage,
  find_files` entirely, so codebench's real Docker sandbox has no way to search/inspect files by
  pattern today, only `list_dir` one directory at a time or an indirect `run_python` workaround.

**Design**: a new shared module, `agent/app/sandbox_dispatch.py`, exposing one async
function — `async def dispatch_sandbox_action(sandbox: SandboxConnector, action: str, args: dict) ->
Dict[str, Any]` — that owns:
- The unified path-key alias table (superset of both existing tables: `path, file, relpath,
  filename, dir, directory, target, file_path`).
- All 12 sandbox actions in one place: the 8 native-pack actions + `run_python, run_pytest,
  search_web` (already shared conceptually, just re-homed) + the crash/robustness fixes below,
  fixed once instead of maybe-twice.
- A single normalized result shape (`{ok, output, error, retryable, ...}`-style, matching what
  `LeafAction.execute()` already returns) that both call sites adapt to their own needs.

Both existing call sites become thin wrappers over this:
- `sandbox_tools.py`'s 8 `LeafAction` subclasses keep their class identity (required — the engine's
  `allowed_actions` gating and `LeafActionRegistry` operate on named classes), but each `execute()`
  body shrinks to "extract args, call `dispatch_sandbox_action`, adapt the result to `LeafAction`'s
  return contract." No change to the opt-in `ToolsConfig.sandbox_pack_enabled`/`sandbox_pack_actions`
  gating, or to `LeafActionRegistry`'s core-vs-installed-pack distinction.
- `execution_compiled_code.py`'s `_dispatch_action` if/elif chain is replaced by a direct call to
  `dispatch_sandbox_action`. `_SANDBOX_ACTIONS` and `_CODE_LEAF_SYSTEM` (the system prompt naming
  available tools) are extended to include the 5 previously-missing inspection actions, so codebench
  agents actually see and can use them.

**Concrete bug/robustness fixes, applied once in the shared module**:
1. **Crash fix**: `write_file`/`patch_file` in `connector_sandbox.py` catch only `OSError`, but
   malformed-surrogate content (trivially producible: `json.loads('{"content": "\\ud800"}')`
   succeeds; encoding it strictly does not) raises `UnicodeEncodeError`, a `ValueError` subclass —
   uncaught on the native engine's non-parallel path, this crashes the entire run, not just one
   leaf. Widen the catch to `(OSError, UnicodeError)` at the connector level, returning a clean
   failure ("content contains characters that cannot be encoded") rather than a silent
   best-effort rewrite — matches this codebase's "never fabricate, degrade honestly" convention
   used elsewhere (the compiled-plan composers).
2. **`read_file` size guard**: currently reads the whole file into memory unconditionally before
   truncating the *returned* text — unlike `write_file`/`patch_file`, which check size before
   touching disk. Add the same pre-read size check (reusing `SandboxActionConfig.max_file_bytes`),
   so a file that entered the workdir via the starter-fixture copy or a raw `run_python` write
   (bypassing `write_file`'s own budget) can't be fully materialized in memory regardless of size.
3. **`list_dir` entry cap**: currently returns every entry unconditionally (only `__pycache__` is
   filtered). Cap the returned entry count (first N + an explicit "...and K more" marker), matching
   the bounded-observation philosophy already used elsewhere (`_observation()`'s `obs_chars` clip).

**Explicitly out of scope for this phase** (real gaps found, deliberately not fixed here):
- No delete/remove action — looks like a deliberate "narrow intent" design choice, not an oversight.
- No binary-file support (`errors="replace"` lossy round-trip) — a real feature addition, not a
  robustness fix; would need a base64/raw-bytes mode designed separately if ever wanted.
- Sandbox failures defaulting to `retryable=False` — looks intentional (file ops are local, not
  usually transient); not touched.

**Testing**: new tests in `connector_sandbox_test.py` for the crash fix (a malformed-surrogate write
no longer raises, returns a clean failure), the `read_file` size guard, and the `list_dir` cap.
`sandbox_tools_actions_test.py`'s existing suite (confinement, budgets, gating, timeout/kill
behavior) must stay green unchanged — it's testing the `LeafAction` contract, which this phase
doesn't change, only its internals. New tests in `execution_compiled_code_test.py` confirming the 5
newly-exposed actions dispatch correctly from the code-leaf loop, and that the codebench and native
call sites now share identical path-alias resolution (a single parametrized test run against both
adapters, so future drift is caught immediately rather than silently reintroduced).

## Phase 2 — Additive, non-invasive cross-benchmark reporting

**Problem being solved**: `cells.jsonl` (QA) and `runs.jsonl` (codebench) are deliberately different
schemas serving different pipelines, and migrating either would risk breaking existing tooling that
depends on them (`analyze_code.py`, the QA benchmark's own analysis scripts) — confirmed out of
scope per an earlier decision in this session: codebench's Docker execution/grading pipeline stays
exactly as-is; only the *reporting* layer gets unified.

**Design**: new `scripts/unified_bench_report.py`, following the exact precedent this session
already set with `scripts/mine_failure_taxonomy.py` — a standalone analyzer that reads existing
result files and produces a new view, touching none of them. Three input readers:
- `badmodel-lab/results/cells.jsonl` (QA benchmark)
- `badmodel-lab/codebench/results/runs.jsonl` (codebench)
- `agent/idea_test_results/*.json` (main harness — covers the case where a codebench task
  is ever run through `graph_compiled_code` via the standard harness for a dev/smoke check, not just
  through the Docker pipeline)

Each reader normalizes its rows into one shared common schema: `{run_id, timestamp, benchmark_type:
"qa"|"code", model, task_id, score, passed, cost_usd, duration_s, origin, source_file}`, plus an
`extra: {...}` bag preserving the type-specific fields (`grounding_pass`/`bucket` for QA,
`tests_passed`/`keystone_pass`/`agent_kind` for code) for anyone who needs to drill in past the
common view. Output: a combined CSV + a markdown summary table (mean score / pass rate / cost by
model × benchmark_type), mirroring `recovery_curve.py`'s existing output style/conventions.

**Testing**: unit tests per reader (feed a small fixture of each real file shape, assert the
normalized rows), plus one test confirming the combined-table aggregation math (mean/count) against
a small hand-built mixed fixture.

## Phase 3 — A first live run, scoped small

**Problem being solved**: no data yet on where the compiled-scaffold engine actually struggles on
coding tasks — the real "find work, improve performance" goal — and 52 tasks × N models × 2
agent-kinds (badmodel + aider) at up to 900s/cell is too much to run blind.

**Design**: start with a small representative slice — a handful of tasks spanning the existing
difficulty/category spread (hard vs soft, a couple of "already proven genuinely hard" tasks per
`codebench-task-author.md`'s calibration bar), 1-2 local models, badmodel agent-kind only for the
first pass (skip the `aider` baseline comparison initially — it's a real cost/time multiplier and
not needed to find *this* system's own failure modes). Coordinate the shared `gpu-lock`
(`/home/muk/projects/gpu-lock`) since Ollama is shared across concurrent sessions on this machine,
matching the existing convention `codebench-task-author.md` itself documents for calibration runs.

After the run: read results through Phase 2's `unified_bench_report.py`, and manually inspect any
failing cells' transcripts (the codebench pipeline's own per-cell artifacts) to find concrete,
fixable failure patterns — prompt gaps, tool-usage confusion, budget exhaustion — the same way this
session's Part D Stage 1 traced a scoring gap back to a specific, fixable root cause rather than
accepting a raw score at face value. Scale up the run (more tasks/models, add the aider comparison)
only after this first pass's findings are triaged — concrete follow-up fixes are a separate,
later cycle scoped to whatever this run actually finds, not designed speculatively now.

**Budget/authorization**: local models are near-$0, but this still needs an explicit go-ahead at
execution time (GPU-lock coordination, wall-clock cost) — this phase's actual run is not
pre-authorized by this spec, matching this session's established convention for live work.

## Verification (all phases)

- `PYTHONPATH=.:services:agent ./.venv/bin/pytest -q agent/tests` green after each
  phase's code changes.
- Phase 1: the existing `sandbox_tools_actions_test.py`/`execution_compiled_code_test.py` suites
  pass unchanged (contract preserved), plus the new crash/guard/parity tests described above.
- Phase 2: `unified_bench_report.py` run against the real existing `cells.jsonl`/`runs.jsonl` files
  on disk as a live smoke check (read-only, $0) — confirm it produces a sane combined table without
  errors, before trusting it for Phase 3's analysis.
- Phase 3: a real live run (small scope, local models) with explicit go-ahead at execution time, its
  own results read back through Phase 2's report, and at least one concrete traced failure pattern
  as the deliverable — not just an aggregate score.
