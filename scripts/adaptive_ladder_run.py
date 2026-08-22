#!/usr/bin/env python3
"""Parallel compute-ladder A/B driver for the NATIVE adaptive engine (LIVE $).

Question: can we get materially better reasoning out of ONE cheap "bad" model by burning more of
its (cheap) tokens/searches (i.e. does an adaptive agent that re-expands, re-grounds on low
confidence, and self-consistency-votes beat the same model run bare? And how close does it get to a
premium model at a fraction of the cost?

Design (all decided with the caller):
  * ONE cheap agent model (default openai/gpt-5-mini), execution_variant=graph, in three arms that
    spend progressively more compute:
        baseline: adaptive OFF (bare cheap model)
        good_adaptive: re-expansion + confidence-driven re-grounding + corrective context
        full: good_adaptive + k-vote x3 + backtrack + expect-contract (max searches/tokens)
  * A premium REFERENCE bar (default google/gemini-3.1-pro-preview + sequential_react). The agent
    never uses it; it is only the quality ceiling we compare the cheap ladder against.
  * 8 tasks spanning all 4 adaptive archetypes incl. the D re-expansion flagship; R=5 for the cheap
    ladder, R=3 for the reference.

Why a bespoke driver rather than native_ab_run.sh: that driver is serial (one cell at a time,
~14h for this matrix). Here every cell is an ISOLATED process at IDEA_TEST_CONCURRENCY=1 (clean
per-run cost/score attribution (no shared-connector cross-talk), but up to JOBS run at once for
wall-clock. Fairness rules are preserved: connector-retry ON in every arm, fixtures OFF, and arms
of the same (task,rep) are enqueued adjacently so they share the same network window; the analyzer
pairs on (task,rep). A hard global BUDGET stops enqueueing, and a per-run USD ceiling guards runaways.

Usage (from repo root):
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/adaptive_ladder_run.py \
      --run-id ladder --jobs 6 --budget 22 --reps 5 --ref-reps 3
Resume-safe: cells with a COMPLETE, parseable result JSON are skipped; a cell that has failed
`--max-attempts` times with no complete result is marked dead in a durable ledger and skipped too
(instead of being retried forever). Only ONE driver may run against a given `--run-id` at a time
(cross-process PID lock in the run's output dir).
"""
import argparse
import atexit
import glob
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

try:
    # Recover real spend from the same cost table the runner uses.
    from agent.app.model_costs import estimate_cost as _estimate_cost
except Exception:  # pragma: no cover - keep the driver runnable even if the import path shifts
    _estimate_cost = None

# Repo root and the interpreter used for cell subprocesses. Both are env-overridable so the
# driver runs unchanged inside the `ladder-benchmark` compose service, where the repo lives at
# /app and there is no .venv (the image's system python already has the deps installed).
REPO = os.environ.get("WEBRAG_REPO") or "/home/muk/projects/webRAG"
CELL_PYTHON = os.environ.get("WEBRAG_PYTHON") or f"{REPO}/.venv/bin/python"
RESULTS_DIR = f"{REPO}/agent/idea_test_results"
# Default ladder: baseline (control), good_adaptive (established), max_burn (productive).
# The full arm (k-vote + backtrack) was net-negative and dropped; max_burn uses deeper re-expansion.
LADDER_ARMS = ["baseline", "good_adaptive", "max_burn"]

# Chroma isolation/embedding config. Embedded mode: each cell gets its own SQLite file
# (no cross-subprocess lock contention) and embedding runs on GPU (embed_device auto/cuda).
# log_dir: per-cell stdout/stderr capture for debugging failed cells (run_cell()).
RUN_CFG = {"chroma_mode": "embedded", "embed_device": "auto", "embedded_root": None, "log_dir": None}


def cell_db_path(cell):
    """Unique per-cell embedded-chroma dir (fresh memory per run_id+task)."""
    return os.path.join(RUN_CFG["embedded_root"], f"{cell['run_id']}_{cell['task']}")

# Named task sets (see agent/app/BENCHMARK_SUITE_50.md). All ids are validated + deduped.
TASK_SETS = {
    "smoke8": ["122", "125", "128", "130", "134", "138", "140", "144"],
    "core24": [f"{n:03d}" for n in range(122, 146)],
    "suite50": (
        [f"{n:03d}" for n in range(122, 146)]
        + ["049", "055", "059", "060", "067", "072", "075", "041", "062",
           "024", "044", "093", "046", "047", "073"]
        + ["068", "095", "108", "040", "054", "065", "042", "056", "061",
           "070", "090"]
    ),
    # Diagnostic barrage: exemplar of each shape in the active 59-suite, plus 2 reasoning
    # tasks (200/206) and 1 stacked-axis task (146) for broader coverage.
    "barrage20": [
        "122", "108", "134", "040", "132", "140", "143", "070", "072", "079",
        "094", "044", "047", "052", "071", "073", "200", "206", "146",
    ],
    # LangGraph comparison tasks: 10 Tier A single-topic sequential shapes (survivor,
    # conflicting-source, chain, re-expansion). LangGraph can't run wide parallel fan-out,
    # so only sequential shapes are included. Built from smoke8 with extra chain/conflicting tasks.
    "langgraph10": ["122", "125", "128", "129", "130", "134", "135", "138", "140", "144"],
}
# ACTIVE barrage suite (59): suite50 minus task 024 (un-gated, LLM-judge-scored, had
# hallucination issues) plus 10 new parallel shapes (breadth/argmax/count/AND-filter/nearest).
# Derived from suite50 so the two never silently diverge. Pinned against CI lint gate's manifest
# (validator_lint_test.ACTIVE_SUITE_IDS) so the barrage never runs unchecked tasks.
TASK_SETS["suite59"] = [t for t in TASK_SETS["suite50"] if t != "024"] + [
    "052", "071", "078", "079", "081", "082", "084", "085", "091", "094",
]
TASKS = TASK_SETS["smoke8"]  # default; overridden by --task-set / --tasks


def keyval(name):
    # An already-exported value wins, and a missing keys.env is not fatal: a $0 local run
    # (compose `ladder-benchmark`, keyless SearXNG + Ollama) has no paid keys to read.
    if os.environ.get(name):
        return os.environ[name]
    try:
        fh = open(f"{REPO}/services/keys.env")
    except OSError:
        return ""
    with fh:
        for line in fh:
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('\r\n').strip('"')
    return ""


def base_env():
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": "services:agent",
        "OPENROUTER_API_KEY": keyval("OPENROUTER_API_KEY"),
        "SEARCH_API_KEY": keyval("SEARCH_API_KEY"),
        "SERPER_KEY": keyval("SERPER_KEY"),
        # serper by default; an inherited SEARCH_PROVIDER wins so a $0 local run can point at the
        # keyless self-hosted SearXNG (SEARCH_PROVIDER=searxng SEARXNG_URL=...) when the paid keys
        # are exhausted. Both arms of a paired A/B always share whichever backend is chosen.
        "SEARCH_PROVIDER": os.environ.get("SEARCH_PROVIDER") or "serper",
        "LLM_PROVIDER": "openrouter",
        "MODEL_API_URL": "https://openrouter.ai/api/v1",
        # Only consulted in --chroma-mode http; overridable so an in-container run can name the
        # `chroma` compose service instead of a host port publication.
        "CHROMA_URL": os.environ.get("CHROMA_URL") or "http://localhost:8001",
        # 45s was a dead knob: the outer per-action budget (20s in idea_dag_settings.json) always binds
        # first, making the inner aiohttp timeout ineffective. Using 20s makes the declared and effective
        # caps match; retries still get 2 attempts within the window.
        "DEFAULT_TIMEOUT": "20", "DEFAULT_DELAY": "2", "JITTER_SECONDS": "0.5",
        "IDEA_TEST_CONCURRENCY": "1",
        "IDEA_TEST_PARALLEL_ACTION_LIMIT": "1",
        "IDEA_TEST_FIXTURES": "off",
        "IDEA_TEST_CONNECTOR_RETRY": "1",
        "IDEA_TEST_EFFORT_TIERS": "0",
        "IDEA_TEST_PREFLIGHT_JSON_TOKENS": "4096",
        "IDEA_TEST_RUNS": "1",
        "IDEA_TEST_USD_CEILING": "0.60",
    })
    return env


def cell_env(cell):
    env = base_env()
    env["IDEA_TEST_IDS"] = cell["task"]
    env["IDEA_TEST_MODELS"] = cell["model"]
    env["IDEA_TEST_EXECUTION_VARIANTS"] = cell["variant"]
    env["IDEA_TEST_RUN_ID"] = cell["run_id"]
    # Grading is deterministic grep, so validation only needs preflight. Using the cell's own
    # model avoids spawning an expensive separate model. Mismatched validation models can cause 402
    # errors that abort the run.
    env["IDEA_TEST_VALIDATION_MODEL"] = cell["model"]
    if cell["arm"]:
        env["IDEA_TEST_ARM"] = cell["arm"]
    else:
        env.pop("IDEA_TEST_ARM", None)
    # Local-model routing to OpenAI-compatible endpoint (Ollama) instead of OpenRouter.
    # Sets openai_compatible provider, a dummy API key (SDK requires truthy), and disables JSON
    # preflight so genuinely weak models that can't emit clean JSON are observed, not excluded.
    if cell.get("provider"):
        env["LLM_PROVIDER"] = cell["provider"]
        # LOCAL_API_URL overrides every axis's hard-coded host endpoint in one place. The axes
        # below all name http://localhost:11435/v1 (the host publication of badmodel-ollama);
        # from inside a container that address is the container itself, so the compose service
        # sets LOCAL_API_URL=http://badmodel-ollama:11434/v1 instead of forking the axis table.
        env["MODEL_API_URL"] = (os.environ.get("LOCAL_API_URL")
                                or cell.get("api_url") or env["MODEL_API_URL"])
        env["OPENAI_API_KEY"] = "ollama"
        env["IDEA_TEST_PREFLIGHT_JSON"] = "0"
        # Smoke-test finding: default 90s LLM_READ_TIMEOUT (tuned for cloud API) is insufficient for
        # local generation. qwen2.5:14b on a consumer GPU needs more time. Two consecutive 90s timeouts
        # with retry killed all cells before completion. 480s provides room while staying within
        # the driver's 1800s per-cell subprocess cap.
        env["LLM_READ_TIMEOUT"] = "480"
        # Preflight has independent 20s-per-candidate timeout that killed some cells. Root cause:
        # local-model lock only serializes concurrent cells, not different models in sequence. A cold-load
        # of a different model after eviction from the single-loaded-model slot takes longer than 100s.
        # 120s per candidate covers cold load time.
        env["IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS"] = "120"
        # Engine-level decision/expansion timeout (defaults to 180s) hits before LLM_READ_TIMEOUT.
        # This ceiling killed cells regardless of prompt size. New env overrides
        # IDEA_TEST_EXPANSION_TIMEOUT/IDEA_TEST_FINAL_TIMEOUT allow local cells sufficient room.
        env["IDEA_TEST_EXPANSION_TIMEOUT"] = "480"
        env["IDEA_TEST_FINAL_TIMEOUT"] = "480"
    if cell.get("telemetry"):
        env["IDEA_TEST_JSON_TELEMETRY"] = "1"
    env["CHROMA_MODE"] = RUN_CFG["chroma_mode"]
    env["CHROMA_EMBED_DEVICE"] = RUN_CFG["embed_device"]
    if RUN_CFG["chroma_mode"] == "embedded":
        env["CHROMA_EMBEDDED_PATH"] = cell_db_path(cell)
    for k, v in (cell.get("burn") or {}).items():
        env[k] = str(v)
    return env


def cell_key(cell):
    """Durable per-cell identity for the attempt ledger. `run_id` already encodes the model
    tag/arm/rep, and `task` is the test id, so the pair uniquely identifies one matrix cell."""
    return f"{cell['run_id']}_{cell['task']}"


def load_ledger(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_ledger(path, ledger):
    """Atomic write (temp + os.replace): a crash mid-write must never leave a corrupt ledger that
    then mis-reports attempt counts: either wrongly resurrecting a dead cell or, worse, wrongly
    declaring a live one dead and abandoning it."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ledger, fh, indent=0, sort_keys=True)
    os.replace(tmp, path)  # atomic on POSIX


def openrouter_key_usage(api_key):
    """Cumulative USD used on this OpenRouter key, all-time, or None on any error (never raises).

    Verified against OpenRouter's current API docs (2026-07): ``GET /api/v1/key`` ->
    ``{"data": {"usage": <cumulative USD, all-time>, "limit": ..., "limit_remaining": ..., ...}}``.
    NOT ``/api/v1/credits`` (that path 404s today) and the field is ``usage``, not ``total_usage``.
    Because this is a live, ever-growing counter (not scoped to this campaign), callers baseline it
    once and compare the delta. See --real-budget wiring in main().
    """
    if not api_key:
        return None
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r).get("data", {}) or {}
        usage = data.get("usage")
        return float(usage) if usage is not None else None
    except Exception:
        return None


def is_dead(cell, ledger, max_attempts):
    """True if this cell has been attempted `max_attempts`+ times with no complete result yet.
    i.e. it is hopeless and should be given up on rather than resubmitted forever. A cell that
    eventually DID produce a complete result is never dead, no matter its attempt count (checked
    first in `run_cell`, but re-checked here so a stale ledger entry can't override a real result)."""
    rec = ledger.get(cell_key(cell))
    if not rec or rec.get("attempts", 0) < max_attempts:
        return False
    return not has_complete_result(cell["run_id"], cell["task"])


def acquire_pid_lock(lock_path):
    """Cross-process guard: refuse to start a second driver against the same run-id/out_dir.

    Two drivers racing the same run-id would both see "no result yet" for a cell and submit it
    twice (running the SAME task concurrently, which cross-contaminates the chroma working memory)
    the per-task serialization exists to protect (mem_<sha256(mandate)> is shared across all
    arms/reps of one task), and wastes money on a duplicate run. Raises SystemExit if the lock is
    held by a live process; otherwise claims it and registers an atexit release.
    """
    if os.path.exists(lock_path):
        old_pid = None
        try:
            old_pid = int(open(lock_path).read().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None:
            try:
                os.kill(old_pid, 0)  # raises if not alive; no-op signal otherwise
            except ProcessLookupError:
                pass  # stale lock left by a dead process. Safe to reclaim
            except PermissionError:
                raise SystemExit(
                    f"lock held by a live pid we can't signal in {lock_path}; refusing to start")
            else:
                raise SystemExit(
                    f"another driver (pid {old_pid}) holds {lock_path}; refusing to start. "
                    f"If that process is really gone, remove the lock file and retry.")
    with open(lock_path, "w") as fh:
        fh.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(lock_path) and os.remove(lock_path))


def check_run_meta(meta_path, this_meta):
    """Refuse to silently mix results from different run configs under one run-id/out_dir.

    Confirmed live on a prior barrage: one run-id accumulated a single-model run AND two different
    axis configs in one shared log, making per-model tallies meaningless. Raises SystemExit on an
    axis/model mismatch against a previously recorded invocation; otherwise (re)writes meta_path.
    """
    if os.path.exists(meta_path):
        try:
            prev_meta = json.load(open(meta_path))
        except Exception:
            prev_meta = {}
        if prev_meta and (prev_meta.get("axis"), prev_meta.get("model")) != \
                (this_meta.get("axis"), this_meta.get("model")):
            raise SystemExit(
                f"run-id already used for axis={prev_meta.get('axis')!r} "
                f"model={prev_meta.get('model')!r}; refusing to mix with "
                f"axis={this_meta.get('axis')!r} model={this_meta.get('model')!r} under the same "
                f"run-id. Use a fresh --run-id.")
    with open(meta_path, "w") as fh:
        json.dump(this_meta, fh, indent=2)


def result_files(run_id, task):
    fs = [f for f in glob.glob(f"{RESULTS_DIR}/{run_id}_{task}_*.json")
          if not (f.endswith("_summary.json") or "_report_" in os.path.basename(f))]
    return fs


def has_complete_result(run_id, task):
    """A cell is 'done' only if a result JSON exists AND parses AND carries a real outcome.

    Guards against a truncated/partial JSON (e.g. killed mid-write by the 1800s timeout kill)
    being mistaken for a finished cell and skipped forever with no score or cost ever recorded.
    The runner now writes results atomically (temp + os.replace), so a partial file should no
    longer occur going forward. This also protects any result written before that fix landed.
    """
    for f in result_files(run_id, task):
        try:
            d = json.load(open(f))
        except Exception:
            continue  # truncated / corrupt -> not done, allow re-run
        if ("validation" in d and d.get("validation", {}).get("overall_score") is not None) \
                or ("error" in d) or ("execution" in d):
            return True
    return False


def recover_cell_usd(run_id, task):
    """Best-effort USD for a cell with NO result JSON (timeout/crash/kill), from its trace jsonl.

    The runner streams `llm_usage` events to `{run_id}_{task}_{raw_model}[_variant].jsonl`; because
    raw model slugs contain '/' (e.g. openai/gpt-4.1-nano), the trace can land in a spurious subdir
    (`{run_id}_{task}_openai/gpt-4.1-nano.jsonl`), a runner quirk, not this driver's to fix, so
    this globs both forms. This is a LOWER BOUND: an LLM call in flight at kill-time emits no usage
    event, and non-LLM spend (search) is not in this trace. Still far better than booking $0 for a
    cell that really spent money.
    """
    if _estimate_cost is None:
        return None
    traces = (glob.glob(f"{RESULTS_DIR}/{run_id}_{task}_*.jsonl")
              + glob.glob(f"{RESULTS_DIR}/{run_id}_{task}_*/*.jsonl"))
    total = None
    for t in traces:
        inp = out = 0
        model = None
        try:
            with open(t) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("event") != "llm_usage":
                        continue
                    pay = d.get("payload") or {}
                    u = pay.get("usage") or {}
                    model = pay.get("model") or model
                    inp += int(u.get("prompt_tokens") or 0)
                    out += int(u.get("completion_tokens") or 0)
        except Exception:
            continue
        if model and (inp or out):
            c = _estimate_cost(model, inp, out)
            if c is not None:
                total = (total or 0.0) + c
    return total


def cell_usd_and_score(run_id, task):
    """Best-effort (usd, score) for a completed cell. Mirrors adaptive_ab_analyze._obs."""
    usd = score = None
    for f in result_files(run_id, task):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ex = d.get("execution", {})
        ob = ex.get("output", {}).get("observability") or ex.get("observability") or {}
        usd = (ob.get("cost", {}) or {}).get("usd")
        score = d.get("validation", {}).get("overall_score")
    return usd, score


# Multi-model AXIS (decided 2026-07-22): two cheap agent LADDERS + one strong REACT-only reference,
# all in ONE driver process so the per-task lock prevents same-task chroma-memory collisions ACROSS
# models. run_id encodes a model tag so the analyzer pairs within a model:
#   {RUN_ID}_{tag}_{arm}_rep{r}  (ladder)   /   {RUN_ID}_{tag}_ref_rep{r}  (reference)
AXES = {
    "final3": {
        "ladders": [
            {"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 5, "burn": None},
            # The productive burn now lives in the `max_burn` ARM profile (uniform, deeper
            # re-expansion), not a per-model k-vote override. K-vote was net-negative.
            {"model": "deepseek/deepseek-v4-flash", "tag": "ds", "reps": 5, "burn": None},
            {"model": "openai/gpt-5.6-luna", "tag": "luna", "reps": 5, "burn": None},
            {"model": "openai/gpt-5.6-terra", "tag": "terra", "reps": 5, "burn": None},
        ],
        "reference": {"model": "anthropic/claude-sonnet-5", "tag": "sonnet",
                      "variant": "sequential_react", "reps": 3},
    },
    # 2026-08-07 diagnostic barrage: local (Ollama) + API ladders side by side, so the SAME
    # task-major interleaved/resumable/budget-tracked driver that already serves final3 also
    # covers strong-vs-weak and local-vs-non-local in one pass. `provider`/`api_url` route a
    # ladder entry at a local OpenAI-compatible endpoint instead of OpenRouter (see cell_env()).
    # qwen2.5:1.5b/qwen2.5:14b are both "subjects" in badmodel-lab/roster.yaml, served from the
    # same badmodel-ollama container (port 11435). See badmodel-lab/run_cell.sh for precedent.
    "barrage20": {
        "json_telemetry": True,  # capture parse-failure classes; no-op unless read, additive only
        "ladders": [
            {"model": "qwen2.5:1.5b", "tag": "q15", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            {"model": "qwen2.5:14b", "tag": "q14", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            {"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1, "burn": None},
            {"model": "deepseek/deepseek-v4-flash", "tag": "ds", "reps": 1, "burn": None},
            {"model": "openai/gpt-5.6-luna", "tag": "luna", "reps": 1, "burn": None},
            {"model": "openai/gpt-5.6-terra", "tag": "terra", "reps": 1, "burn": None},
        ],
        "reference": {"model": "anthropic/claude-sonnet-5", "tag": "sonnet",
                      "variant": "sequential_react", "reps": 1},
    },
    # 2026-08-15 capability-spectrum sweep (DAG v2 vs off-the-shelf LangGraph). The question is
    # not "is DAG v2 better" in the abstract but "at which point on the model-capability curve, if
    # any, does its structure earn its cost". Two axes, run under SEPARATE --run-ids so the free
    # GPU-serial local sweep and the paid API sweep proceed concurrently without contending
    # (embedded chroma gives every cell its own SQLite file, so cross-driver contamination is not
    # possible). Neither axis carries a `reference`. A premium ceiling bar is not what this asks.
    #
    # Run each axis TWICE, once per execution variant:
    #   --variant graph           --arms baseline,good_adaptive   (the native ladder)
    #   --variant langgraph_react --arms good_adaptive            (the external baseline)
    # The arm label is inert for langgraph_react EXCEPT that `good_adaptive` preserves the
    # connector-retry parity langgraph_solver's F16 invariant depends on; `baseline` would
    # silently disable it (connector_retry_on_failure_enabled=False) and handicap that arm only.
    "capspec_api": {
        "json_telemetry": True,
        "ladders": [
            # super-cheap AND genuinely weak: the cheapest real "bad model on an API" available
            {"model": "meta-llama/llama-3.2-1b-instruct", "tag": "l1b", "reps": 2, "burn": None},
            # the established cheap-tier reference point for every prior result in this repo
            {"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 2, "burn": None},
            # same headline price as nano, different vendor/family. Guards against the whole
            # comparison being an artifact of one model's quirks
            {"model": "google/gemini-2.5-flash-lite", "tag": "flash", "reps": 2, "burn": None},
        ],
    },
    # 2026-08-15, follow-up: chain-shape-only axis. Wave 3 found `sequential_react` matches
    # `graph:good_adaptive` on score at 1/3 the input tokens with equal evidence volume, which
    # raises the internal question "what is the graph structure buying?" LangGraph not involved.
    # If the graph engine's advantage is real it should appear on CHAINS (where a later step must
    # condition on an earlier one) and nowhere else. Deliberately R=1 over MORE tasks rather than
    # R=2 over fewer: this session measured that task count, not rep count, is what resolves arm
    # ranking (keystone gates make per-task scores near-binary).
    "capspec_chain": {
        "json_telemetry": True,
        "ladders": [
            {"model": "meta-llama/llama-3.2-1b-instruct", "tag": "l1b", "reps": 1, "burn": None},
            {"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1, "burn": None},
            {"model": "google/gemini-2.5-flash-lite", "tag": "flash", "reps": 1, "burn": None},
        ],
    },
    "capspec_local": {
        "json_telemetry": True,
        "ladders": [
            # tinyllama and phi3:mini have NO tool-calling template. Ollama hard-rejects a
            # /v1/chat/completions call carrying `tools` with HTTP 400. They are in the roster
            # precisely so that rejection is RECORDED as the langgraph arm's result rather than
            # assumed: an off-the-shelf ReAct agent cannot run them at all, while the native
            # engine only needs the model to emit JSON as text (probed 2026-08-15: both do).
            {"model": "tinyllama:latest", "tag": "tiny", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            {"model": "phi3:mini", "tag": "phi3", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            # tool-capable, but tiny. The bottom of the ladder where both arms CAN run
            {"model": "qwen2.5:0.5b", "tag": "q05", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            {"model": "llama3.2:3b", "tag": "l32", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            # the strongest local subject that still fits the 12GB card comfortably alongside a
            # 16k context. Where a "bad model" stops being bad
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 1, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
    # 2026-08-22, E1 confirmation (ASSUMPTION_AUDIT.md PART 5): the dedup A/B repeated on a
    # SECOND model tier. The original finding (-0.157, p=0.0007) is one model (gpt-4.1-nano)
    # on one task set, and it was measured on pre-Cycle-19 code, before the all-flagged batch
    # stopped collapsing to ``candidates[:1]``. A local model costs $0 in LLM spend, which is
    # what lets this run at all. Drive with
    #   --axis e1_dedup_local --arms good_adaptive,good_adaptive_nodedup --tasks <core24 subset>
    # so the two arms differ in ``got_dedup_enabled`` and nothing else.
    "e1_dedup_local": {
        "json_telemetry": True,
        "ladders": [
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 3, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
    # 2026-08-22, E3 (ASSUMPTION_AUDIT.md PART 5): the memory-retrieval similarity floor's live
    # A/B, at the value the offline histogram calibrated (0.40). Local model, so $0. Drive with
    #   --axis e3_memfloor_local --arms good_adaptive,good_adaptive_memfloor --tasks <core24 subset>
    # so the two arms differ in ``memory_retrieval_similarity_floor`` and nothing else.
    "e3_memfloor_local": {
        "json_telemetry": True,
        "ladders": [
            # R=4 rather than E1's 3: the floor cuts ~6% of retrieved rows, so the expected
            # effect is smaller than dedup's and the paired test needs the extra rep more than
            # it needs a wider task set (the mechanism only bites where retrieval happens).
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 4, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
    # 2026-08-22, T1-4 (ASSUMPTION_AUDIT.md PART 2): the sibling visit-URL dedup's live A/B.
    # The collision RATE was measured offline; what the claim map does to a run was not. Local
    # model, so $0. Drive with
    #   --axis t14_urldedup_local --arms good_adaptive,good_adaptive_urldedup --tasks <fan-out set>
    # so the two arms differ in ``visit_sibling_url_dedup`` and nothing else. Task selection
    # matters more here than in E1/E3: the mechanism can only fire in a run that actually
    # produces a sibling visit fan-out whose leaves resolve their URLs from the shared pool.
    "t14_urldedup_local": {
        "json_telemetry": True,
        "ladders": [
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 3, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
    # 2026-08-22, the strong-agent-trace bundle's end-to-end live A/B (project_strong_agent_
    # trace_guidance): numeric_provenance + race_value_agreement + chain_closure cleared
    # capture-replay + isolated live-probe + a combined end-to-end run already, but that run
    # checked composition (no double-penalty), not whether the bundle actually improves score.
    # Local model, $0. Drive with
    #   --axis tracemech_local --arms good_adaptive,good_adaptive_tracemech --tasks <core24>
    "tracemech_local": {
        "json_telemetry": True,
        "ladders": [
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 3, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
    # 2026-08-22, the safety-net bundle's live A/B: D4 merge-retry-after-skip + D6 early-exit-
    # respects-grounding. Both are "decline a premature termination" gates, neither previously
    # live-validated. Local model, $0. Drive with
    #   --axis safetynet_local --arms good_adaptive,good_adaptive_safetynet --tasks <core24>
    "safetynet_local": {
        "json_telemetry": True,
        "ladders": [
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 3, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
    # 2026-08-22, T1-5's live A/B: does substituting raw_score for the capped score in the
    # dynamic beam's spread computation actually help, or just widen the beam for no score gain
    # (the concern noted in ASSUMPTION_AUDIT.md's T1-5 entry, given E1/T1-4 both found wider
    # fan-out buys little at this tier). Local model, $0. Drive with
    #   --axis beamraw_local --arms good_adaptive,good_adaptive_beamraw --tasks <core24>
    "beamraw_local": {
        "json_telemetry": True,
        "ladders": [
            {"model": "qwen2.5:7b", "tag": "q7", "reps": 3, "burn": None,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
        ],
    },
}


def build_cells(run_id, reps, ref_reps, model, ref_model, ref_variant, include_ref, tasks, arms=None):
    """Single-model ladder + optional reference (legacy/simple mode)."""
    arms = arms or LADDER_ARMS
    cells = []
    for task in tasks:
        for rep in range(1, reps + 1):
            for arm in arms:
                cells.append({"task": task, "rep": rep, "arm": arm, "model": model,
                              "variant": "graph", "burn": None, "run_id": f"{run_id}_{arm}_rep{rep}"})
    if include_ref:
        for task in tasks:
            for rep in range(1, ref_reps + 1):
                cells.append({"task": task, "rep": rep, "arm": None, "model": ref_model,
                              "variant": ref_variant, "burn": None,
                              "run_id": f"{run_id}_reference_rep{rep}"})
    return cells


def build_axis_cells(run_id, axis, tasks, arms=None, variant="graph"):
    """Full multi-model axis. Ordered task-major so same-task cells are adjacent (per-task lock +
    shared window). Any per-model `burn` override applies only to the top (most-burn) arm.
    `variant` lets --variant point the SAME task/model grid at graph_compiled or naive_discretion
    under a different --run-id (Phase 2 technique-coverage reuse), without touching this axis's
    reference cell, which always keeps its own axis-defined variant (e.g. sequential_react)."""
    arms = arms or LADDER_ARMS
    top_arm = arms[-1]
    telemetry = bool(axis.get("json_telemetry"))
    cells = []
    for task in tasks:
        for lad in axis["ladders"]:
            for rep in range(1, lad["reps"] + 1):
                for arm in arms:
                    cells.append({"task": task, "rep": rep, "arm": arm, "model": lad["model"],
                                  "variant": variant,
                                  "burn": lad.get("burn") if arm == top_arm else None,
                                  "provider": lad.get("provider"), "api_url": lad.get("api_url"),
                                  "local": bool(lad.get("provider")), "telemetry": telemetry,
                                  "run_id": f"{run_id}_{lad['tag']}_{arm}_rep{rep}"})
        ref = axis.get("reference")
        if ref:
            for rep in range(1, ref["reps"] + 1):
                cells.append({"task": task, "rep": rep, "arm": None, "model": ref["model"],
                              "variant": ref["variant"], "burn": None,
                              "provider": ref.get("provider"), "api_url": ref.get("api_url"),
                              "local": bool(ref.get("provider")), "telemetry": telemetry,
                              "run_id": f"{run_id}_{ref['tag']}_ref_rep{rep}"})
    return cells


_CHILDREN_LOCK = threading.Lock()
_CHILDREN = {}  # pid -> subprocess.Popen; lets a repeated SIGINT/SIGTERM hard-kill in-flight cells


def _kill_process_group(proc):
    """Best-effort SIGKILL of a cell subprocess's whole process group. Each cell is spawned with
    start_new_session=True (its own group), so this reaches any grandchildren too (e.g. a stray
    browser process) instead of only the direct child."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def run_cell(cell):
    if has_complete_result(cell["run_id"], cell["task"]):
        return cell, "skip", 0.0, None, 0.0  # already have a real result → resume-safe
    t0 = time.time()
    # 2026-08-07 finding: 24/72 cells in a live checkpoint-2 run exited rc=0 in normal time with
    # NO result JSON at all. Silently, un-diagnosable, because stdout/stderr previously went
    # straight to DEVNULL. Capture to a per-cell log file (append mode: a resumed retry's output
    # lands after the prior attempt's, so the full attempt history is visible in one place)
    # instead, so the NEXT time this (or anything else) fails silently there's something to read.
    log_path = (os.path.join(RUN_CFG["log_dir"], f"{cell_key(cell)}.log")
                if RUN_CFG.get("log_dir") else None)
    log_fh = open(log_path, "a") if log_path else subprocess.DEVNULL
    if log_path:
        log_fh.write(f"\n=== attempt started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_fh.flush()
    # 1800s (was 1200): the 'full'/'max_burn' arm's longest runs are its highest-scorers; a too-tight
    # cap dropped them and biased the heavy arms downward (survivorship). Give heavy runs room to
    # finish. start_new_session=True puts each cell in its OWN process group so a timeout or a
    # hard-abort (repeated SIGINT/SIGTERM) can kill it and any children as a unit.
    proc = subprocess.Popen(
        [CELL_PYTHON, "-m", "agent.app.idea_test_runner"],
        cwd=REPO, env=cell_env(cell), stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    with _CHILDREN_LOCK:
        _CHILDREN[proc.pid] = proc
    try:
        rc = proc.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        rc = "timeout"
    finally:
        with _CHILDREN_LOCK:
            _CHILDREN.pop(proc.pid, None)
        if log_path:
            log_fh.close()
    dt = time.time() - t0
    usd, score = cell_usd_and_score(cell["run_id"], cell["task"])
    if usd is None:
        # No result JSON (timeout/crash/kill). Recover real spend from the live trace so --budget
        # and --real-budget stay honest instead of booking $0 for a cell that really spent money.
        usd = recover_cell_usd(cell["run_id"], cell["task"])
    status = "ok" if rc == 0 else f"{rc}"
    # Resume experiment (2026-08-07): reclaim the per-cell embedded DB ONLY on a genuine complete
    # result. A killed/timed-out attempt's partial chroma memory (already-fetched, already-embedded
    # pages) is deliberately LEFT IN PLACE instead. A retry of the SAME cell (same run_id+task ->
    # same cell_db_path, see cell_db_path()) starts a fresh subprocess but against that same chroma
    # dir, so the engine's mandate-hash-keyed memory lookup can reuse whatever it already indexed
    # instead of re-fetching/re-embedding from zero on every timeout. Orphaned dirs for cells that
    # are finally given up on (never complete within --max-attempts) are reclaimed in fill()'s DEAD
    # branch instead, once no further retry can use them.
    if RUN_CFG["chroma_mode"] == "embedded" and RUN_CFG["embedded_root"] and \
            has_complete_result(cell["run_id"], cell["task"]):
        shutil.rmtree(cell_db_path(cell), ignore_errors=True)
    return cell, status, (usd or 0.0), score, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="ladder")
    ap.add_argument("--jobs", type=int, default=8, help="max parallel cells; effective cap = #distinct tasks (per-task serialized)")
    ap.add_argument("--budget", type=float, default=22.0,
                    help="hard global USD stop (stops enqueueing); now includes recovered "
                         "timeout/abort spend, not just booked result-JSON cost")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ref-reps", type=int, default=3)
    ap.add_argument("--model", default="openai/gpt-5-mini")
    ap.add_argument("--ref-model", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--ref-variant", default="sequential_react")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--task-set", default="smoke8", choices=list(TASK_SETS),
                    help="named validated task set (see BENCHMARK_SUITE_50.md)")
    ap.add_argument("--tasks", default="", help="explicit space/comma task ids; overrides --task-set")
    ap.add_argument("--axis", default="", choices=[""] + list(AXES),
                    help="multi-model axis (e.g. 'final3'); overrides --model/--ref-* single-model mode")
    ap.add_argument("--chroma-mode", default="embedded", choices=["http", "embedded"],
                    help="embedded = per-cell isolated SQLite (no shared-server contention); http = shared server")
    ap.add_argument("--embed-device", default="auto", choices=["cpu", "cuda", "auto"],
                    help="chroma embedding device; auto = GPU if available else CPU")
    ap.add_argument("--arms", default="", help="space/comma ladder arms (IDEA_TEST_ARM profiles), "
                    "top arm gets any per-model burn; default: " + " ".join(LADDER_ARMS))
    ap.add_argument("--variant", default="graph",
                    help="execution_variant for subject (non-reference) cells in --axis mode, "
                         "e.g. graph, graph_compiled, naive_discretion. Lets a second invocation "
                         "point the SAME task/model grid at a different technique under a fresh "
                         "--run-id, reusing all resume/lock/budget machinery. Reference cells keep "
                         "their own axis-defined variant regardless (e.g. sequential_react).")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="give up on a cell (mark it dead in the ledger) after this many non-skip "
                         "attempts with no complete result, instead of retrying it forever on "
                         "every resume")
    ap.add_argument("--real-budget", type=float, default=0.0,
                    help="hard stop on TRUE cumulative OpenRouter key spend (USD; delta of the "
                         "key's all-time usage since this campaign's baseline; 0=off). The "
                         "baseline persists in the ledger across resumes, so restarting the "
                         "driver never resets the clock, and it can't be fooled by timeouts or "
                         "in-flight calls the way the local --budget estimate can.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RUN_CFG["chroma_mode"] = args.chroma_mode
    RUN_CFG["embed_device"] = args.embed_device
    arms = [a for a in args.arms.replace(",", " ").split()] or LADDER_ARMS

    tasks = ([t for t in args.tasks.replace(",", " ").split()] if args.tasks
             else TASK_SETS[args.task_set])
    if args.axis:
        cells = build_axis_cells(args.run_id, AXES[args.axis], tasks, arms=arms, variant=args.variant)
    else:
        cells = build_cells(args.run_id, args.reps, args.ref_reps, args.model,
                            args.ref_model, args.ref_variant, not args.no_ref, tasks, arms=arms)
    out_dir = f"{RESULTS_DIR}/_{args.run_id}"
    os.makedirs(out_dir, exist_ok=True)
    RUN_CFG["embedded_root"] = f"{out_dir}/_chroma"
    if RUN_CFG["chroma_mode"] == "embedded":
        os.makedirs(RUN_CFG["embedded_root"], exist_ok=True)
    # 2026-08-07: per-cell stdout/stderr capture (see run_cell()). Was DEVNULL, making any
    # rc=0-but-no-result cell (a real occurrence, not hypothetical) completely undiagnosable.
    RUN_CFG["log_dir"] = f"{out_dir}/cell_logs"
    os.makedirs(RUN_CFG["log_dir"], exist_ok=True)

    # Per-invocation timestamped log: a long barrage is restarted/resumed many times, and a single
    # shared driver.log across invocations (worse, across DIFFERENT axis/model configs reusing a
    # run-id) made tallies meaningless. Confirmed live on a prior run (3 invocations, 2 different
    # axis configs, one shared log, no way to tell which lines belonged to which). One file per
    # invocation makes each run's log unambiguous.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logpath = f"{out_dir}/driver_{stamp}.log"
    log = open(logpath, "a")

    def emit(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    n_done = sum(1 for c in cells if has_complete_result(c["run_id"], c["task"]))
    emit(f"run_id={args.run_id} cells={len(cells)} (already-done={n_done}) jobs={args.jobs} "
         f"budget=${args.budget:.2f} max_attempts={args.max_attempts} tasks={len(tasks)} "
         f"chroma={RUN_CFG['chroma_mode']}/{RUN_CFG['embed_device']} "
         f"{'axis='+args.axis if args.axis else 'model='+args.model}")
    from collections import Counter
    by_mc = Counter((c["model"].split("/")[-1], c["arm"] or "ref") for c in cells)
    for (m, arm), n in sorted(by_mc.items()):
        emit(f"  {m:<24} {arm:<14} {n} cells")
    if args.dry_run:
        emit("dry-run: not executing"); return

    # --- Cross-process run lock (F9/S6): refuse a second driver instance against this run-id.
    acquire_pid_lock(f"{out_dir}/driver.lock")

    # --- run-id/axis guard (F9/S4): refuse to silently mix results from different run configs
    # under one run-id.
    check_run_meta(f"{out_dir}/run_meta.json",
                   {"axis": args.axis or "", "model": "" if args.axis else args.model,
                    "task_set": args.task_set, "arms": arms})

    # --- durable attempt ledger (F6/S1): survives process restart so resume can give up on a
    # hopeless cell instead of re-burning its full timeout cost forever.
    ledger_path = f"{out_dir}/ledger.json"
    ledger = load_ledger(ledger_path)

    # --- true cumulative ceiling (F8/S2): baseline the OpenRouter key's real all-time usage once
    # per campaign (persisted in the ledger so a restart never resets the clock), then hard-stop
    # enqueueing on the REAL delta. The only spend number timeouts/kills/in-flight calls can't fool.
    or_key = keyval("OPENROUTER_API_KEY")
    real_baseline = (ledger.get("_meta") or {}).get("real_baseline")
    if args.real_budget > 0 and real_baseline is None:
        real_baseline = openrouter_key_usage(or_key)
        if real_baseline is not None:
            ledger.setdefault("_meta", {})["real_baseline"] = real_baseline
            save_ledger(ledger_path, ledger)
            emit(f"real-budget armed: baseline key usage=${real_baseline:.4f} "
                 f"ceiling=+${args.real_budget:.2f}")
        else:
            emit("!! --real-budget requested but GET /api/v1/key was unreachable; real-budget "
                 "stays disarmed until a later poll succeeds (local --budget still applies)")

    # Per-task serialization: at most ONE in-flight cell per task, because the engine's working
    # memory lives in a chroma collection keyed by mandate hash (idea_engine._memo_namespace →
    # mem_<sha256(mandate)>), which is shared across arms/reps of the SAME task. Running two same-task
    # cells at once would cross-contaminate that store. Different tasks are fully independent, so we
    # still parallelize across the 8 tasks. Arms of a given (task,rep) run back-to-back, which
    # preserves the shared-network-window pairing the analyzer relies on.
    spent = 0.0
    done = 0
    stopped = False
    pending = list(cells)
    busy_tasks = set()
    local_busy = False  # 0a: at most one local-provider cell in flight process-wide (see fill())
    futs = {}  # future -> cell

    def _drain(signum, _frame):
        # F9/S8: graceful drain on the first signal (stop enqueueing, let in-flight cells finish
        # naturally); a repeated signal hard-aborts in-flight subprocesses instead of waiting up to
        # 1800s each. Either way the ledger keeps flushing after every finished cell (see below), so
        # no partial/unaccounted state is left behind.
        nonlocal stopped
        if not stopped:
            stopped = True
            emit(f"!! signal {signum}. Draining: no new cells enqueued, in-flight cells finish "
                 f"naturally (send the signal again to hard-abort in-flight subprocesses)")
        else:
            with _CHILDREN_LOCK:
                procs = list(_CHILDREN.values())
            emit(f"!! signal {signum} again. Hard-aborting {len(procs)} in-flight subprocess(es)")
            for p in procs:
                _kill_process_group(p)

    signal.signal(signal.SIGINT, _drain)
    signal.signal(signal.SIGTERM, _drain)

    def fill(ex):
        nonlocal local_busy
        for c in list(pending):
            if len(futs) >= args.jobs:
                break
            if c["task"] in busy_tasks:
                continue
            # 0a: local models share ONE GPU-resident Ollama slot (OLLAMA_MAX_LOADED_MODELS=1).
            # two concurrent local cells on different tasks would fight over it and thrash
            # (constant reload), the same failure class as the chroma hang that burned the prior
            # barrage. Serialize local cells globally; API cells are unaffected and keep
            # parallelizing up to --jobs (this `continue` just lets the loop try the next pending
            # cell, which may well be a non-local one for a different task).
            if c.get("local") and local_busy:
                continue
            if is_dead(c, ledger, args.max_attempts):
                pending.remove(c)
                rec = ledger.get(cell_key(c), {})
                emit(f"  DEAD   task={c['task']} {cell_key(c)} attempts={rec.get('attempts')} "
                     f"last_status={rec.get('last_status')}. Giving up "
                     f"(max-attempts={args.max_attempts})")
                # Resume experiment (2026-08-07): a live cell's partial embedded-chroma memory is
                # now preserved across a killed/timed-out attempt so a RETRY can reuse it (see
                # run_cell()). But a cell that's finally given up on will never retry again, so
                # reclaim its orphaned dir here instead of leaking it forever.
                if RUN_CFG["chroma_mode"] == "embedded" and RUN_CFG["embedded_root"]:
                    shutil.rmtree(cell_db_path(c), ignore_errors=True)
                continue
            pending.remove(c)
            busy_tasks.add(c["task"])
            if c.get("local"):
                local_busy = True
            futs[ex.submit(run_cell, c)] = c

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        fill(ex)
        while futs:
            finished, _ = wait(set(futs.keys()), return_when=FIRST_COMPLETED)
            for fut in finished:
                cell = futs.pop(fut)
                busy_tasks.discard(cell["task"])
                if cell.get("local"):
                    local_busy = False
                try:
                    _, status, usd, score, dt = fut.result()
                except Exception as exc:
                    # F9/S7: one cell's exception (subprocess spawn failure, a glob/JSON error, ...)
                    # must never crash the whole driver and orphan every other in-flight cell.
                    status, usd, score, dt = f"error:{type(exc).__name__}", 0.0, None, 0.0
                    emit(f"  !! task={cell['task']} {cell_key(cell)} raised "
                         f"{type(exc).__name__}: {exc}")
                if status != "skip":
                    spent += usd
                    done += 1
                    # F6: record the attempt durably so a future resume converges instead of
                    # retrying a hopeless cell forever.
                    k = cell_key(cell)
                    rec = ledger.get(k, {"attempts": 0})
                    rec["attempts"] = rec.get("attempts", 0) + 1
                    rec["last_status"] = status
                    rec["est_usd"] = round(rec.get("est_usd", 0.0) + (usd or 0.0), 6)
                    rec["ts"] = time.time()
                    ledger[k] = rec
                    save_ledger(ledger_path, ledger)
                sc = f"{score:.2f}" if isinstance(score, (int, float)) else "?"
                tag = cell["arm"] or "reference"
                mdl = cell["model"].split("/")[-1]      # disambiguate nano vs ds vs sonnet
                emit(f"  task={cell['task']} rep={cell['rep']} {mdl:<18} {tag:<13} "
                     f"{status:<8} score={sc:<5} ${usd:.3f} {dt:>5.0f}s  | done={done}/{len(cells)} "
                     f"inflight={len(futs)} spent≈${spent:.2f}")
                if spent >= args.budget and not stopped:
                    stopped = True
                    emit(f"!! BUDGET ${args.budget:.2f} reached (spent≈${spent:.2f}). No new cells enqueued")
                if args.real_budget > 0 and not stopped and real_baseline is not None:
                    cur = openrouter_key_usage(or_key)  # polled once per finished cell (cheap)
                    if cur is not None and (cur - real_baseline) >= args.real_budget:
                        stopped = True
                        emit(f"!! REAL-BUDGET ${args.real_budget:.2f} reached (true key spend "
                             f"+${cur - real_baseline:.2f}). No new cells enqueued")
            if not stopped:
                fill(ex)
    emit(f"DONE. Executed {done} cells, spent≈${spent:.2f}, {len(pending)} unstarted. "
         f"Analyze with adaptive_ab_analyze.py")


if __name__ == "__main__":
    main()
