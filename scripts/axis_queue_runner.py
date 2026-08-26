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
from adaptive_ladder_run import keyval  # noqa: E402  (reuse the same keys.env reader)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="path to a JSON list of cell specs")
    ap.add_argument("--print-only", action="store_true",
                     help="print the full queue plan and exit; no GPU time spent")
    ap.add_argument("--skip-preflight", action="store_true",
                     help="skip the live search-infra health check (debugging only)")
    args = ap.parse_args()

    with open(args.queue) as fh:
        queue = json.load(fh)
    if not isinstance(queue, list) or not queue:
        print("!! queue file must be a non-empty JSON list"); sys.exit(1)

    if args.print_only:
        print(f"axis_queue_runner: {len(queue)} entries queued (--print-only, not executing)")
        for i, entry in enumerate(queue):
            print(f"  [{i}] {' '.join(build_cell_command(entry))}")
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
            cmd = build_cell_command(entry)
            emit(f"[{i+1}/{len(queue)}] START run_id={entry['run_id']} axis={entry.get('axis','')} "
                 f"cmd={' '.join(cmd)}")
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=REPO)
            dt = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            emit(f"[{i+1}/{len(queue)}] {status} run_id={entry['run_id']} axis={entry.get('axis','')} "
                 f"elapsed={dt:.0f}s")
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
