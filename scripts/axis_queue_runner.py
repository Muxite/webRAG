#!/usr/bin/env python3
"""Unattended sequential driver over MULTIPLE adaptive_ladder_run.py axes.

Why this exists: adaptive_ladder_run.py already resumes/locks/budgets a SINGLE --axis
invocation, but every local (badmodel-ollama) axis today gets launched by hand, one at a
time. Local cells are also confirmed fully serialized regardless of --jobs (one GPU-resident
Ollama model at a time, OLLAMA_MAX_LOADED_MODELS=1 -- see adaptive_ladder_run.py's `fill()`
local_busy flag and docs/handoffs/LADDER_FINAL_20260822_RESULTS.md). So "keep the GPU busy
overnight" cannot mean parallel axes -- it means chaining axes back-to-back with zero idle
gap between one finishing and the next starting, unattended, resumable, and safe to leave
running across an axis that individually fails.

Usage (from repo root):
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/axis_queue_runner.py \
      --queue scripts/axis_queues/phase0_qwen7b.json [--print-only]

Queue file: a JSON list of cell specs, each a dict of adaptive_ladder_run.py CLI flags:
  [{"run_id": "dagv3p0", "axis": "capspec_local", "task_set": "core24",
    "arms": "good_adaptive,good_adaptive_constrained", "jobs": 8}, ...]
Optional per-entry keys: "tasks" (explicit task ids, overrides task_set), "variant",
"reps"/"ref_reps" (only meaningful in non-axis mode), "extra_args" (list of raw extra CLI
tokens for anything not covered above).

Resume-safe by construction: each queued entry just re-invokes adaptive_ladder_run.py with
its own --run-id, which already skips cells with a complete result (has_complete_result) and
refuses to mix configs under one run-id (check_run_meta). Re-running this queue runner after
a crash/interrupt simply re-issues every entry; already-done cells cost nothing but a resume
scan.

--slices N: expand each queue entry into N concurrent sub-processes covering disjoint task
subsets, dealt round-robin (not contiguous -- task cost is highly skewed, so a contiguous
split would leave one slice idle for minutes while another is still deep in the expensive
tasks). This is safe because each adaptive_ladder_run.py invocation is already a fully
isolated subprocess: distinct --run-id values get disjoint Chroma dirs (cell_db_path keys on
run_id), disjoint result filenames (has_complete_result globs on run_id), and a disjoint
driver.lock/run_meta.json/ledger.json (acquire_pid_lock is scoped to
RESULTS_DIR/_{run_id}/). Today's GPU serialization comes only from the `local_busy` flag
inside ONE driver process; N slices under N distinct run-ids never share that flag, so total
local GPU concurrency is exactly N by construction (jobs stays 1 per slice, bounding each
slice's own local_busy to one cell at a time).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = os.environ.get("WEBRAG_REPO") or "/home/muk/projects/webRAG"
CELL_PYTHON = os.environ.get("WEBRAG_PYTHON") or f"{REPO}/.venv/bin/python"
DRIVER_DIR = f"{REPO}/agent/idea_test_results/_axis_queue"
LOCK_PATH = f"{DRIVER_DIR}/queue_runner.lock"

sys.path.insert(0, f"{REPO}/scripts")
from adaptive_ladder_run import keyval, TASK_SETS  # noqa: E402  (reuse the same reader + task tables)


def acquire_pid_lock(lock_path):
    """Refuse a second queue runner instance; stale locks (dead PID) are reclaimed."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    if os.path.exists(lock_path):
        try:
            with open(lock_path) as fh:
                old_pid = int(fh.read().strip())
            os.kill(old_pid, 0)  # raises if the process is gone
            print(f"!! another axis_queue_runner is already running (pid={old_pid}); refusing to start")
            sys.exit(1)
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale or unreadable lock; safe to reclaim
    with open(lock_path, "w") as fh:
        fh.write(str(os.getpid()))


def release_pid_lock(lock_path):
    try:
        os.remove(lock_path)
    except OSError:
        pass


def search_infra_healthy():
    """Live preflight: a real Serper query must succeed before the queue burns GPU hours on
    what might otherwise turn into infra-confounded cells (the exact failure mode that
    silently corrupted the 2026-08-23 run -- see project memory 'Serper key outage')."""
    key = keyval("SERPER_KEY")
    provider = os.environ.get("SEARCH_PROVIDER") or "serper"
    if provider != "serper":
        print(f"search preflight: SEARCH_PROVIDER={provider!r}, skipping Serper-specific check")
        return True
    if not key:
        print("!! search preflight FAILED: no SERPER_KEY found in env or services/keys.env")
        return False
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=json.dumps({"q": "webrag axis queue runner preflight"}).encode(),
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(f"!! search preflight FAILED: HTTP {resp.status}")
                return False
            body = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001 - report and fail closed, don't guess
        print(f"!! search preflight FAILED: {exc!r}")
        return False
    if "organic" not in body:
        print(f"!! search preflight FAILED: unexpected response shape, keys={list(body)}")
        return False
    print("search preflight OK: Serper responded with organic results")
    return True


def build_cell_command(entry):
    cmd = [CELL_PYTHON, f"{REPO}/scripts/adaptive_ladder_run.py",
           "--run-id", entry["run_id"], "--jobs", str(entry.get("jobs", 8))]
    if entry.get("tasks"):
        cmd += ["--tasks", entry["tasks"]]
    else:
        cmd += ["--task-set", entry.get("task_set", "core24")]
    if entry.get("axis"):
        cmd += ["--axis", entry["axis"]]
    if entry.get("arms"):
        cmd += ["--arms", entry["arms"]]
    if entry.get("variant"):
        cmd += ["--variant", entry["variant"]]
    for k in ("reps", "ref_reps"):
        if entry.get(k) is not None:
            cmd += [f"--{k.replace('_', '-')}", str(entry[k])]
    cmd += entry.get("extra_args", [])
    return cmd


def resolve_entry_tasks(entry):
    """Resolve the full, ordered task-id list a queue entry covers, pre-slicing.

    Params:
        entry: one queue entry dict. Uses "tasks" (an explicit comma/space
            separated id string) if present, else "task_set" (a name in
            TASK_SETS, defaulting to "core24" -- matching build_cell_command's
            own default).

    Returns:
        list[str] of task ids in queue-file/TASK_SETS order.

    Raises:
        KeyError: entry names a task_set not present in TASK_SETS.
    """
    if entry.get("tasks"):
        return entry["tasks"].replace(",", " ").split()
    task_set = entry.get("task_set", "core24")
    if task_set not in TASK_SETS:
        raise KeyError(f"unknown task_set {task_set!r} (known: {sorted(TASK_SETS)})")
    return list(TASK_SETS[task_set])


def deal_round_robin(tasks, n):
    """Deal `tasks` round-robin into `n` slices so cost-skewed tasks spread evenly.

    Params:
        tasks: ordered list of task ids.
        n: number of slices to deal into.

    Returns:
        list of n lists (each possibly empty, when n > len(tasks)); every
        input task appears in exactly one output slice.

    Raises:
        ValueError: n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    slices = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        slices[i % n].append(task)
    return slices


def expand_entry(entry, n):
    """Expand one queue entry into concurrent slice entries.

    Params:
        entry: one queue entry dict (never mutated).
        n: requested slice count.

    Returns:
        When n <= 1: [entry] unchanged (same object, no run_id suffix, no
        "tasks" rewrite) -- so the n<=1 path is behaviourally identical to
        pre-slicing runs. When n > 1: up to n new dicts (shallow copies of
        entry), each with "run_id" suffixed "_s{i}", "tasks" set to its
        dealt comma-joined subset, and "task_set" dropped (tasks now takes
        precedence in build_cell_command). Slices that would be empty
        (n exceeds the task count) are omitted, so fewer than n entries may
        come back.

    Raises:
        ValueError: n < 1.
        KeyError: entry names an unknown task_set (via resolve_entry_tasks).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n <= 1:
        return [entry]
    dealt = deal_round_robin(resolve_entry_tasks(entry), n)
    out = []
    for i, subset in enumerate(dealt):
        if not subset:
            continue
        sliced = dict(entry)
        sliced["run_id"] = f"{entry['run_id']}_s{i}"
        sliced["tasks"] = ",".join(subset)
        sliced.pop("task_set", None)
        out.append(sliced)
    return out


def run_slices_concurrently(group, i, total, emit):
    """Launch every slice in `group` as a concurrent subprocess and wait for all.

    Params:
        group: list of >=2 already-expanded queue entries sharing one
            original queue slot, to run at the same time.
        i: zero-based index of the original queue entry (for log prefixes).
        total: total number of original queue entries (for log prefixes).
        emit: callable(str) for timestamped logging, as used by main().

    Returns:
        list of (entry, returncode, elapsed_seconds) tuples, one per slice
        in `group` order. A slice's non-zero returncode is recorded, not
        raised: one slice failing must not abort its siblings or the rest
        of the queue. A slice whose subprocess could not even be launched
        (e.g. bad interpreter path) is recorded with returncode -1.

    Raises:
        Nothing slice-specific; launch failures are caught per-slice.
    """
    launched = []
    for entry in group:
        cmd = build_cell_command(entry)
        emit(f"[{i+1}/{total}] START (slice) run_id={entry['run_id']} "
             f"axis={entry.get('axis','')} cmd={' '.join(cmd)}")
        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, cwd=REPO)
        except OSError as exc:
            emit(f"[{i+1}/{total}] FAILED-TO-LAUNCH run_id={entry['run_id']}: {exc!r}")
            launched.append((entry, None, t0))
        else:
            launched.append((entry, proc, t0))

    results = []
    for entry, proc, t0 in launched:
        if proc is None:
            results.append((entry, -1, 0.0))
            continue
        rc = proc.wait()
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAILED rc={rc}"
        emit(f"[{i+1}/{total}] {status} (slice) run_id={entry['run_id']} "
             f"axis={entry.get('axis','')} elapsed={dt:.0f}s")
        results.append((entry, rc, dt))
    return results


def process_entry(i, total, entry, slices, emit):
    """Run one queue entry to completion, sliced across `slices` processes if requested.

    Params:
        i: zero-based index of this entry in the queue.
        total: total number of queue entries.
        entry: the queue entry dict.
        slices: requested slice count (>=1); 1 keeps the original
            single-subprocess, subprocess.run-based path byte-identical to
            pre-slicing behaviour.
        emit: callable(str) for timestamped logging.

    Returns:
        list of (entry, returncode, elapsed_seconds) tuples: length 1 for
        the unsliced path, length len(group) for the sliced path.

    Raises:
        ValueError: slices < 1.
        KeyError: entry names an unknown task_set when slicing needs to
            resolve it (via expand_entry).
    """
    group = expand_entry(entry, slices)
    if len(group) == 1:
        e = group[0]
        cmd = build_cell_command(e)
        emit(f"[{i+1}/{total}] START run_id={e['run_id']} axis={e.get('axis','')} "
             f"cmd={' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO)
        dt = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        emit(f"[{i+1}/{total}] {status} run_id={e['run_id']} axis={e.get('axis','')} "
             f"elapsed={dt:.0f}s")
        return [(e, proc.returncode, dt)]

    emit(f"[{i+1}/{total}] launching {len(group)} slices concurrently for "
         f"run_id={entry['run_id']}")
    return run_slices_concurrently(group, i, total, emit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="path to a JSON list of cell specs")
    ap.add_argument("--print-only", action="store_true",
                     help="print the full queue plan and exit; no GPU time spent")
    ap.add_argument("--skip-preflight", action="store_true",
                     help="skip the live search-infra health check (debugging only)")
    ap.add_argument("--slices", type=int, default=1,
                     help="expand each entry into N concurrent processes over disjoint, "
                          "round-robin-dealt task subsets (default 1: no slicing, unchanged "
                          "sequential behaviour)")
    args = ap.parse_args()

    if args.slices < 1:
        print(f"!! --slices must be >= 1, got {args.slices}"); sys.exit(1)

    with open(args.queue) as fh:
        queue = json.load(fh)
    if not isinstance(queue, list) or not queue:
        print("!! queue file must be a non-empty JSON list"); sys.exit(1)

    if args.print_only:
        print(f"axis_queue_runner: {len(queue)} entries queued (--print-only, not executing)")
        for i, entry in enumerate(queue):
            group = expand_entry(entry, args.slices)
            if len(group) == 1:
                print(f"  [{i}] {' '.join(build_cell_command(group[0]))}")
            else:
                print(f"  [{i}] {len(group)} concurrent slices:")
                for e in group:
                    print(f"      {' '.join(build_cell_command(e))}")
        return

    if not args.skip_preflight and not search_infra_healthy():
        print("!! aborting entire queue: search infra preflight failed. Fix SERPER_KEY / "
              "SEARCH_PROVIDER before burning GPU hours on cells that would just be infra-"
              "confounded results.")
        sys.exit(1)

    os.makedirs(DRIVER_DIR, exist_ok=True)
    acquire_pid_lock(LOCK_PATH)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"{DRIVER_DIR}/driver_{stamp}.log"
    log = open(log_path, "a")

    def emit(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    emit(f"axis_queue_runner starting: {len(queue)} entries, log={log_path}")
    try:
        for i, entry in enumerate(queue):
            # Re-check search infra before EVERY entry, not just once at queue start. An
            # 11+ hour unattended queue can outlive a search key (this repo already lost a
            # full 144-cell run once to a dead Serper key that went undetected for the
            # entire run -- docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md).
            # A one-time preflight only catches a key that was ALREADY dead at launch.
            if not args.skip_preflight and not search_infra_healthy():
                emit(f"[{i+1}/{len(queue)}] SKIPPED run_id={entry['run_id']} axis={entry.get('axis','')} "
                     f"-- search infra unhealthy at this point in the queue; not burning GPU "
                     f"hours on what would be infra-confounded cells. Will re-check before the "
                     f"next entry in case it recovers.")
                continue
            process_entry(i, len(queue), entry, args.slices, emit)
            # Deliberately no early-exit on a cell-command failure: a single bad axis/arm must
            # not stall the rest of an unattended overnight queue. Every entry's own driver log
            # + result JSONs remain the source of truth for what actually happened; this log is
            # only the top-level sequencing record. Infra failures (above) are handled
            # separately by skipping rather than running a doomed entry at all.
        emit("axis_queue_runner: all entries processed")
    finally:
        release_pid_lock(LOCK_PATH)
        log.close()


if __name__ == "__main__":
    main()
