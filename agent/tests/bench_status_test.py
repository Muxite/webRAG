"""Regression tests for scripts/bench_status.py -- the "is my run still going" checker.

Fixture-driven against a tmp results dir (never touches the real agent/idea_test_results/).
Covers the documented contracts: RUNNING/DONE/STALE/DEAD classification (a lock file with a
dead pid must report STALE -- the false-"exit 0" case this script exists to catch), slice
grouping under a logical run-id, --since filtering, a missing or malformed ledger.json, a run
dir with no lock file at all, and that *_summary.json files are never mistaken for a run dir.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import bench_status as bs  # noqa: E402
import adaptive_ladder_run as ladder  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(ladder, "RESULTS_DIR", str(tmp_path))
    return tmp_path


def _make_run_dir(tmp_path, run_id, *, ledger=None, cells=None, lock_pid=None,
                   log_lines=None):
    """Build a minimal adaptive_ladder_run.py output dir."""
    d = tmp_path / f"_{run_id}"
    d.mkdir()
    (d / "run_meta.json").write_text(json.dumps({"axis": "x", "model": ""}))
    if ledger is not None:
        (d / "ledger.json").write_text(json.dumps(ledger))
    if lock_pid is not None:
        (d / "driver.lock").write_text(str(lock_pid))
    lines = list(log_lines or [])
    if cells is not None and not any("cells=" in ln for ln in lines):
        lines.insert(0, f"[00:00:00] run_id={run_id} cells={cells} (already-done=0) jobs=1")
    if lines:
        (d / "driver_20260829_000000.log").write_text("\n".join(lines) + "\n")
    return d


def _dead_pid():
    """A PID essentially guaranteed not to be alive (same convention as
    adaptive_ladder_run_test.py's stale-lock test)."""
    return 999999999


# ----------------------------------------------------------------------------- classification --
def test_running_when_lock_pid_alive_and_matches_driver_cmdline(tmp_path, monkeypatch):
    d = _make_run_dir(tmp_path, "runA", cells=2, lock_pid=os.getpid())
    monkeypatch.setattr(bs, "_proc_cmdline",
                         lambda pid: f"{sys.executable} scripts/adaptive_ladder_run.py --run-id runA")
    report = bs.build_slice_report(str(d))
    assert report.status == bs.RUNNING


def test_running_pid_but_cmdline_is_a_different_run_id_is_not_trusted(tmp_path, monkeypatch):
    d = _make_run_dir(tmp_path, "runA", cells=2, lock_pid=os.getpid())
    monkeypatch.setattr(bs, "_proc_cmdline",
                         lambda pid: f"{sys.executable} scripts/adaptive_ladder_run.py --run-id runB")
    assert bs.pid_is_live_driver(os.getpid(), "runA") is False


def test_stale_when_lock_pid_is_dead():
    dead = _dead_pid()
    assert bs.pid_is_live_driver(dead, "anything") is False


def test_stale_status_reported_for_dead_lock_pid(tmp_path):
    dead = _dead_pid()
    d = _make_run_dir(tmp_path, "runA", cells=2, lock_pid=dead)
    report = bs.build_slice_report(str(d))
    assert report.status == bs.STALE


def test_done_when_no_lock_and_ledger_covers_expected_cells(tmp_path):
    ledger = {"runA_001": {"last_status": "ok", "attempts": 1},
              "runA_002": {"last_status": "ok", "attempts": 1}}
    d = _make_run_dir(tmp_path, "runA", ledger=ledger, cells=2)
    report = bs.build_slice_report(str(d))
    assert report.status == bs.DONE
    assert report.counts == {"complete": 2, "dead": 0, "pending": 0, "total": 2}


def test_dead_when_no_lock_and_ledger_falls_short_of_expected(tmp_path):
    ledger = {"runA_001": {"last_status": "ok", "attempts": 1}}
    d = _make_run_dir(tmp_path, "runA", ledger=ledger, cells=2)
    report = bs.build_slice_report(str(d))
    assert report.status == bs.DEAD


def test_ledger_dead_bucket_counts_cells_that_hit_max_attempts_without_completing():
    ledger = {"a": {"last_status": "timeout", "attempts": 3},
              "b": {"last_status": "ok", "attempts": 1},
              "c": {"last_status": "1", "attempts": 1}}  # not yet given up -> pending
    counts = bs.ledger_cell_counts(ledger)
    assert counts == {"complete": 1, "dead": 1, "pending": 1, "total": 3}


# ------------------------------------------------------------------------- missing/malformed ----
def test_missing_ledger_counts_as_all_zero(tmp_path):
    d = _make_run_dir(tmp_path, "runA", cells=2)  # no ledger.json at all
    report = bs.build_slice_report(str(d))
    assert report.counts == {"complete": 0, "dead": 0, "pending": 0, "total": 0}


def test_malformed_ledger_json_does_not_raise(tmp_path):
    d = _make_run_dir(tmp_path, "runA", cells=2)
    (d / "ledger.json").write_text("{not valid json")
    report = bs.build_slice_report(str(d))  # must not raise
    assert report.counts["total"] == 0


def test_run_dir_with_no_lock_file_and_no_log_is_unknown(tmp_path):
    d = tmp_path / "_runA"
    d.mkdir()
    (d / "ledger.json").write_text("{}")
    report = bs.build_slice_report(str(d))
    assert report.status == bs.UNKNOWN


# ---------------------------------------------------------------------------- summary exclusion -
def test_summary_json_file_is_never_treated_as_a_run_dir(tmp_path):
    _make_run_dir(tmp_path, "runA", cells=1, ledger={"runA_001": {"last_status": "ok", "attempts": 1}})
    (tmp_path / "runA_001_summary.json").write_text("{}")  # a FILE, not a "_"-prefixed dir
    dirs = bs.discover_run_dirs(str(tmp_path))
    assert all(os.path.isdir(p) for p in dirs)
    assert not any(p.endswith("summary.json") for p in dirs)


def test_axis_queue_scratch_dir_excluded_from_run_discovery(tmp_path):
    # axis_queue_runner.py's own scratch dir: writes driver_*.log but no ledger/run_meta/lock.
    d = tmp_path / "_axis_queue"
    d.mkdir()
    (d / "driver_20260829_000000.log").write_text("[00:00:00] queue runner started\n")
    dirs = bs.discover_run_dirs(str(tmp_path))
    assert str(d) not in dirs


# --------------------------------------------------------------------------------- grouping -----
def test_slice_dirs_group_under_logical_run_id(tmp_path):
    for i in range(3):
        _make_run_dir(tmp_path, f"night_a1_g_s{i}", cells=2,
                       ledger={f"night_a1_g_s{i}_001": {"last_status": "ok", "attempts": 1}})
    groups = bs.group_slices(bs.discover_run_dirs(str(tmp_path)))
    assert set(groups.keys()) == {"night_a1_g"}
    assert len(groups["night_a1_g"]) == 3


def test_parse_slice_splits_infix_and_leaves_unsliced_ids_alone():
    assert bs.parse_slice("night_a1_g_s3") == ("night_a1_g", 3)
    assert bs.parse_slice("phaseP_smoke") == ("phaseP_smoke", None)


def test_run_report_aggregates_counts_across_slices(tmp_path):
    for i in range(2):
        _make_run_dir(tmp_path, f"runB_s{i}", cells=1,
                       ledger={f"runB_s{i}_001": {"last_status": "ok", "attempts": 1}})
    reports = bs.collect_run_reports(str(tmp_path))
    assert len(reports) == 1
    run = reports[0]
    assert run.logical_id == "runB"
    assert run.counts["complete"] == 2
    assert run.expected_total == 2
    assert run.status == bs.DONE


def test_run_report_status_is_running_if_any_slice_is_still_running(tmp_path, monkeypatch):
    _make_run_dir(tmp_path, "runC_s0", cells=1,
                   ledger={"runC_s0_001": {"last_status": "ok", "attempts": 1}})
    _make_run_dir(tmp_path, "runC_s1", cells=1, lock_pid=os.getpid())
    monkeypatch.setattr(bs, "_proc_cmdline",
                         lambda pid: f"{sys.executable} scripts/adaptive_ladder_run.py --run-id runC_s1")
    reports = bs.collect_run_reports(str(tmp_path))
    assert reports[0].status == bs.RUNNING


# ------------------------------------------------------------------------------------- --since --
def test_since_filters_out_runs_that_started_before_the_cutoff(tmp_path):
    d_old = _make_run_dir(tmp_path, "oldrun", cells=1)
    d_new = _make_run_dir(tmp_path, "newrun", cells=1)
    old_ts = time.time() - 3 * 86400
    os.utime(d_old, (old_ts, old_ts))
    (d_old / "run_meta.json").write_text(json.dumps({"axis": "x", "model": ""}))
    os.utime(d_old / "run_meta.json", (old_ts, old_ts))

    monkeypatch_ctime = {}

    def fake_start_time(run_dir):
        return old_ts if "oldrun" in run_dir else time.time()

    orig = bs._slice_start_time
    bs._slice_start_time = fake_start_time
    try:
        runs = bs.collect_run_reports(str(tmp_path))
        cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - 86400))
        kept = bs.filter_since(runs, cutoff)
    finally:
        bs._slice_start_time = orig
    kept_ids = {r.logical_id for r in kept}
    assert "newrun" in kept_ids
    assert "oldrun" not in kept_ids


def test_since_empty_string_keeps_every_run(tmp_path):
    _make_run_dir(tmp_path, "runA", cells=1)
    runs = bs.collect_run_reports(str(tmp_path))
    assert bs.filter_since(runs, "") == runs


# ----------------------------------------------------------------------------- log line reading -
def test_last_log_line_returns_most_recent_nonblank_line(tmp_path):
    d = _make_run_dir(tmp_path, "runA", cells=1,
                       log_lines=["[00:00:01] first", "", "[00:00:02] second"])
    assert bs.last_log_line(str(d)) == "[00:00:02] second"


def test_last_log_line_none_when_no_log_file(tmp_path):
    d = tmp_path / "_runA"
    d.mkdir()
    (d / "ledger.json").write_text("{}")
    assert bs.last_log_line(str(d)) is None
