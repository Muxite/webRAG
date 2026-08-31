#!/usr/bin/env python3
"""bench_status.py -- is a benchmark run still going, and did it finish.

Why this exists: adaptive_ladder_run.py is launched with nohup and left running unattended
(often chained by axis_queue_runner.py across many queue entries, each its own subprocess). A
shell "exit 0" only tells you the *launcher* returned control, not that the driver is alive or
that it made progress -- a nohup'd driver killed by OOM, a lost SSH session, or a crash inside
the ThreadPoolExecutor all leave no shell-visible trace, and "exit 0" has been trusted as
"finished" three nights running when the driver actually died holding its lock. This script
answers the question for real: it reads the PID out of `driver.lock` and checks the process is
still alive (and is actually the driver, not a PID recycled by an unrelated process since);
cross-references `ledger.json` for real per-cell status; and surfaces the last line the driver
itself wrote to its log. A lock file whose PID is dead is reported STALE -- exactly the
false-"exit 0" case this exists to catch.

Run-ids launched via axis_queue_runner.py's ``--slices`` carry a ``_s{i}`` infix (e.g.
``night_a1_g_s3``); those are one logical run split across N concurrent subprocesses, each with
its own output dir/lock/ledger. This script groups slices back into their logical run for the
summary view and shows the per-slice breakdown in `--run-id` detail.

Never trusts ``*_summary.json`` (reflects only the last cell of a multi-invocation run -- see
adaptive_ladder_run.py's ``has_complete_result``, which this script imports rather than
reimplementing).

Usage:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/bench_status.py
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/bench_status.py --limit 5
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/bench_status.py --run-id night_a1
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/bench_status.py --since 20260828
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_ladder_run as ladder  # noqa: E402  (reuse has_complete_result/load_ledger/RESULTS_DIR)

RESULTS_DIR = ladder.RESULTS_DIR
DEFAULT_MAX_ATTEMPTS = 3  # mirrors adaptive_ladder_run.py's --max-attempts default
LOCK_NAME = "driver.lock"
LEDGER_NAME = "ledger.json"

SLICE_RE = re.compile(r"^(.*)_s(\d+)$")
CELLS_RE = re.compile(r"\bcells=(\d+)\b")

RUNNING, DONE, STALE, DEAD, UNKNOWN = "RUNNING", "DONE", "STALE", "DEAD", "UNKNOWN"
_STATUS_RANK = {STALE: 4, RUNNING: 3, DEAD: 2, UNKNOWN: 1, DONE: 0}


def parse_slice(run_id: str):
    """Split a possibly slice-qualified run-id into (logical_id, slice_index).

    Params:
        run_id: a run dir's basename with the leading ``_`` stripped, e.g. ``night_a1_g_s3``.
    Returns:
        ``(logical_id, slice_index)``; slice_index is ``None`` when run_id has no ``_sN`` suffix.
    """
    m = SLICE_RE.match(run_id)
    if not m:
        return run_id, None
    return m.group(1), int(m.group(2))


def is_run_dir(path: str) -> bool:
    """True if `path` looks like an adaptive_ladder_run.py output dir, not queue-runner scratch
    (e.g. ``_axis_queue``, which also writes ``driver_*.log`` but never a ledger/run_meta)."""
    if not os.path.isdir(path):
        return False
    return (os.path.exists(os.path.join(path, LEDGER_NAME))
            or os.path.exists(os.path.join(path, "run_meta.json"))
            or os.path.exists(os.path.join(path, LOCK_NAME)))


def discover_run_dirs(results_dir: str) -> List[str]:
    """All benchmark-run output dirs under `results_dir`, i.e. ``_*`` dirs that pass
    :func:`is_run_dir`. Sorted for deterministic grouping/output order."""
    return sorted(p for p in glob.glob(os.path.join(results_dir, "_*")) if is_run_dir(p))


def read_lock_pid(run_dir: str) -> Optional[int]:
    """The PID recorded in `run_dir`'s driver.lock, or None if absent/unreadable/malformed."""
    path = os.path.join(run_dir, LOCK_NAME)
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def _proc_cmdline(pid: int) -> Optional[str]:
    """Best-effort space-joined argv of a live process. None if unreadable (process gone,
    permission denied, or no /proc) -- never raises."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def pid_is_live_driver(pid: int, run_id: str) -> bool:
    """Whether `pid` is really a live adaptive_ladder_run.py driver for this exact run-id.

    ``os.kill(pid, 0)`` succeeding only proves *some* process currently holds that pid, not
    that it is the driver that wrote the lock file -- pids get recycled. Cross-checks argv for
    the script name and an exact ``--run-id <run_id>`` token pair when /proc is readable.

    Params:
        pid: process id read from driver.lock.
        run_id: the (slice-qualified) run-id the lock file belongs to.
    Returns:
        True if the pid is alive and (when its cmdline could be read) that cmdline names
        adaptive_ladder_run.py with a matching ``--run-id``. If the pid is alive but its
        cmdline can't be read (e.g. sandboxed /proc), liveness alone is trusted.
    Raises:
        Never raises -- a gone process or permission error is treated as "not live".
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = _proc_cmdline(pid)
    if cmdline is None:
        return True
    tokens = cmdline.split(" ")
    has_script = any("adaptive_ladder_run.py" in t for t in tokens)
    has_run_id = any(a == "--run-id" and b == run_id for a, b in zip(tokens, tokens[1:]))
    return has_script and has_run_id


def latest_driver_log(run_dir: str) -> Optional[str]:
    """Path to the most recent ``driver_*.log`` in `run_dir` (timestamped names sort
    chronologically), or None if the driver never got far enough to open one."""
    logs = sorted(glob.glob(os.path.join(run_dir, "driver_*.log")))
    return logs[-1] if logs else None


def parse_expected_cells(run_dir: str) -> Optional[int]:
    """The cell count the driver announced at startup (``cells=N`` in its first log line), or
    None if no log exists / the line was never written (e.g. it died before logging)."""
    path = latest_driver_log(run_dir)
    if not path:
        return None
    try:
        with open(path) as fh:
            for line in fh:
                m = CELLS_RE.search(line)
                if m:
                    return int(m.group(1))
    except OSError:
        return None
    return None


def last_log_line(run_dir: str) -> Optional[str]:
    """The last non-blank line written to `run_dir`'s most recent driver log, or None."""
    path = latest_driver_log(run_dir)
    if not path:
        return None
    try:
        lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
    except OSError:
        return None
    return lines[-1] if lines else None


def ledger_cell_counts(ledger: dict, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Dict[str, int]:
    """Bucket ledger entries into complete / dead / pending.

    Params:
        ledger: the parsed ledger.json (as returned by ``ladder.load_ledger``); the ``_meta``
            key (real-budget bookkeeping) is ignored.
        max_attempts: attempts at/above which a non-"ok" cell is given up on (mirrors
            adaptive_ladder_run.py's is_dead(); the actual value used by a given run isn't
            recorded anywhere durable, so this defaults to the driver's own CLI default).
    Returns:
        ``{"complete": n, "dead": n, "pending": n, "total": n}``.
    """
    complete = dead = 0
    for key, rec in ledger.items():
        if key == "_meta":
            continue
        if rec.get("last_status") == "ok":
            complete += 1
        elif rec.get("attempts", 0) >= max_attempts:
            dead += 1
    total = sum(1 for k in ledger if k != "_meta")
    return {"complete": complete, "dead": dead, "pending": total - complete - dead, "total": total}


@dataclass
class SliceReport:
    dir_name: str
    run_id: str  # slice-qualified, e.g. night_a1_g_s3
    slice_index: Optional[int]
    status: str
    pid: Optional[int]
    expected_cells: Optional[int]
    counts: Dict[str, int]
    last_log_line: Optional[str]
    start_time: Optional[float]
    end_time: Optional[float]  # None while RUNNING (still open-ended)


def _slice_start_time(run_dir: str) -> Optional[float]:
    """Best-effort run-start timestamp: the lock file's creation time (written once, right at
    startup) if present, else the earliest driver log's creation time."""
    lock = os.path.join(run_dir, LOCK_NAME)
    if os.path.exists(lock):
        return os.path.getctime(lock)
    logs = sorted(glob.glob(os.path.join(run_dir, "driver_*.log")))
    return os.path.getctime(logs[0]) if logs else None


def build_slice_report(run_dir: str) -> SliceReport:
    """Classify and summarize one adaptive_ladder_run.py output dir (one slice, or a whole
    unsliced run if it was never split)."""
    run_id = os.path.basename(run_dir).lstrip("_")
    _, slice_index = parse_slice(run_id)
    pid = read_lock_pid(run_dir)
    ledger = ladder.load_ledger(os.path.join(run_dir, LEDGER_NAME))
    counts = ledger_cell_counts(ledger)
    expected = parse_expected_cells(run_dir)
    start_time = _slice_start_time(run_dir)

    if pid is not None:
        status = RUNNING if pid_is_live_driver(pid, run_id) else STALE
        end_time = None
    else:
        log = latest_driver_log(run_dir)
        end_time = os.path.getmtime(log) if log else None
        if expected is None:
            status = UNKNOWN
        elif counts["complete"] + counts["dead"] >= expected:
            status = DONE
        else:
            status = DEAD

    return SliceReport(
        dir_name=run_dir, run_id=run_id, slice_index=slice_index, status=status, pid=pid,
        expected_cells=expected, counts=counts, last_log_line=last_log_line(run_dir),
        start_time=start_time, end_time=end_time,
    )


@dataclass
class RunReport:
    logical_id: str
    status: str
    slices: List[SliceReport] = field(default_factory=list)

    @property
    def expected_total(self) -> Optional[int]:
        vals = [s.expected_cells for s in self.slices]
        return None if any(v is None for v in vals) else sum(vals)

    @property
    def counts(self) -> Dict[str, int]:
        total = {"complete": 0, "dead": 0, "pending": 0, "total": 0}
        for s in self.slices:
            for k in total:
                total[k] += s.counts[k]
        return total

    @property
    def start_time(self) -> Optional[float]:
        vals = [s.start_time for s in self.slices if s.start_time is not None]
        return min(vals) if vals else None

    @property
    def last_activity(self) -> float:
        """Timestamp used to rank runs newest-first: `now` if any slice is still RUNNING,
        else the latest known end_time, else the start_time, else 0.0 (sorts last)."""
        if any(s.status == RUNNING for s in self.slices):
            return time.time()
        ends = [s.end_time for s in self.slices if s.end_time is not None]
        if ends:
            return max(ends)
        starts = [s.start_time for s in self.slices if s.start_time is not None]
        return max(starts) if starts else 0.0

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return self.last_activity - self.start_time

    @property
    def last_log_line(self) -> Optional[str]:
        active = max(self.slices, key=lambda s: (s.end_time or time.time()), default=None)
        return active.last_log_line if active else None


def _aggregate_status(slices: List[SliceReport]) -> str:
    return max((s.status for s in slices), key=lambda st: _STATUS_RANK[st])


def group_slices(run_dirs: List[str]) -> Dict[str, List[str]]:
    """Group run dirs by logical run-id (slice suffix stripped)."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for d in run_dirs:
        run_id = os.path.basename(d).lstrip("_")
        logical, _ = parse_slice(run_id)
        groups[logical].append(d)
    return groups


def build_run_report(logical_id: str, slice_dirs: List[str]) -> RunReport:
    slices = sorted((build_slice_report(d) for d in slice_dirs),
                     key=lambda s: (s.slice_index if s.slice_index is not None else -1))
    return RunReport(logical_id=logical_id, status=_aggregate_status(slices), slices=slices)


def collect_run_reports(results_dir: str = RESULTS_DIR) -> List[RunReport]:
    groups = group_slices(discover_run_dirs(results_dir))
    return [build_run_report(logical, dirs) for logical, dirs in groups.items()]


def _date_of(ts: Optional[float]) -> Optional[str]:
    return time.strftime("%Y%m%d", time.localtime(ts)) if ts else None


def filter_since(runs: List[RunReport], since: str) -> List[RunReport]:
    """Keep only runs that started on/after `since` (``YYYYMMDD``, same lexicographic-prefix
    convention as summarize_bench.py/level_ladder.py/recovery_curve.py's ``--since``, applied
    here to each run's start time since run-ids in this repo aren't themselves date-stamped)."""
    if not since:
        return runs
    return [r for r in runs if (_date_of(r.start_time) or "") >= since]


def _fmt_elapsed(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def format_summary_line(run: RunReport) -> str:
    c = run.counts
    expected = run.expected_total if run.expected_total is not None else "?"
    slice_note = f" ({len(run.slices)} slices)" if len(run.slices) > 1 else ""
    return (f"{run.status:<7} {run.logical_id}{slice_note}  "
            f"cells {c['complete']}/{expected} (dead={c['dead']})  "
            f"elapsed={_fmt_elapsed(run.elapsed_seconds)}")


def format_detail(run: RunReport) -> str:
    lines = [format_summary_line(run), f"  last: {run.last_log_line or '(no log yet)'}"]
    for s in run.slices:
        exp = s.expected_cells if s.expected_cells is not None else "?"
        lines.append(
            f"    slice {s.slice_index if s.slice_index is not None else '-':<3} {s.status:<7} "
            f"pid={s.pid} cells {s.counts['complete']}/{exp} (dead={s.counts['dead']})  "
            f"elapsed={_fmt_elapsed((s.end_time or time.time()) - s.start_time if s.start_time else None)}"
        )
        lines.append(f"      last: {s.last_log_line or '(no log yet)'}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default="", help="Show detail for one logical run (prefix match)")
    ap.add_argument("--since", default="", help="Only runs that started on/after this YYYYMMDD")
    ap.add_argument("--limit", type=int, default=15, help="Max runs to list (default: 15)")
    args = ap.parse_args(argv)

    runs = collect_run_reports()
    runs = filter_since(runs, args.since)

    if args.run_id:
        matches = sorted((r for r in runs if r.logical_id.startswith(args.run_id)),
                          key=lambda r: r.last_activity, reverse=True)
        if not matches:
            print(f"no run matching --run-id {args.run_id!r}")
            return 1
        for r in matches:
            print(format_detail(r))
        return 0

    runs.sort(key=lambda r: r.last_activity, reverse=True)
    for r in runs[: args.limit]:
        print(format_summary_line(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
