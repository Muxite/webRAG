"""Regression tests for the pre-barrage driver hardening in scripts/adaptive_ladder_run.py
(F5 validate-on-resume, F6 durable attempt-ledger, F7 timeout-spend recovery, F9 PID lock/run-meta
guard). These are narrow unit tests of pure/isolable helpers — anything requiring a live subprocess
run or a real OpenRouter call (the actual --real-budget poll) is exercised by a live driver run, not
here.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import adaptive_ladder_run as ladder  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_results_dir(tmp_path, monkeypatch):
    """Point the module's RESULTS_DIR at a scratch dir so tests never touch real results."""
    monkeypatch.setattr(ladder, "RESULTS_DIR", str(tmp_path))
    return tmp_path


def _write_result(tmp_path, run_id, task, body):
    path = tmp_path / f"{run_id}_{task}_gpt-5-mini_graph_r1.json"
    path.write_text(json.dumps(body))
    return path


# --------------------------------------------------------------------------------- F5 --------
def test_has_complete_result_true_for_real_result(tmp_path):
    _write_result(tmp_path, "run1", "005",
                  {"execution": {}, "validation": {"overall_score": 0.8}})
    assert ladder.has_complete_result("run1", "005") is True


def test_has_complete_result_false_for_truncated_json(tmp_path):
    # Simulates a crash mid-write (Ctrl-C, timeout kill) before the atomic-rename fix existed
    # elsewhere, or any other partial writer: the file exists but does not parse.
    path = tmp_path / "run1_006_gpt-5-mini_graph_r1.json"
    path.write_text('{"execution": {"incomplete"')
    assert ladder.has_complete_result("run1", "006") is False


def test_has_complete_result_false_when_no_file():
    assert ladder.has_complete_result("run1", "999") is False


def test_result_files_excludes_summary_and_report_files(tmp_path):
    _write_result(tmp_path, "run1", "007", {"execution": {}, "validation": {"overall_score": 1.0}})
    (tmp_path / "run1_007_summary.json").write_text("{}")
    (tmp_path / "run1_007_gpt-5-mini_graph_report_v2.json").write_text("{}")
    files = ladder.result_files("run1", "007")
    assert len(files) == 1
    assert "summary" not in files[0]
    assert "_report_" not in files[0]


# --------------------------------------------------------------------------------- F6 --------
def _cell(run_id="run1", task="006", arm="baseline"):
    return {"run_id": run_id, "task": task, "rep": 1, "arm": arm, "model": "openai/gpt-5-mini"}


def test_is_dead_false_with_no_ledger_record():
    assert ladder.is_dead(_cell(), {}, max_attempts=3) is False


def test_is_dead_false_under_attempt_threshold():
    cell = _cell()
    ledger = {ladder.cell_key(cell): {"attempts": 2}}
    assert ladder.is_dead(cell, ledger, max_attempts=3) is False


def test_is_dead_true_at_threshold_with_no_result(tmp_path):
    # task 006 never wrote a result -> attempts exhausted -> hopeless -> dead.
    cell = _cell()
    ledger = {ladder.cell_key(cell): {"attempts": 3}}
    assert ladder.is_dead(cell, ledger, max_attempts=3) is True


def test_is_dead_false_if_a_complete_result_exists_despite_high_attempts(tmp_path):
    # A cell that eventually succeeded is NEVER dead, no matter how many attempts it took —
    # a stale ledger entry must not override a real result.
    cell = _cell(task="005")
    _write_result(tmp_path, "run1", "005", {"execution": {}, "validation": {"overall_score": 0.9}})
    ledger = {ladder.cell_key(cell): {"attempts": 10}}
    assert ladder.is_dead(cell, ledger, max_attempts=3) is False


def test_ledger_save_load_roundtrip_is_atomic(tmp_path):
    path = str(tmp_path / "ledger.json")
    ladder.save_ledger(path, {"a": {"attempts": 1}})
    assert ladder.load_ledger(path) == {"a": {"attempts": 1}}
    # no leftover temp file after a successful atomic replace
    assert not os.path.exists(path + ".tmp")


def test_load_ledger_missing_or_corrupt_returns_empty(tmp_path):
    assert ladder.load_ledger(str(tmp_path / "nope.json")) == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert ladder.load_ledger(str(corrupt)) == {}


# --------------------------------------------------------------------------------- F7 --------
def test_recover_cell_usd_from_flat_trace(tmp_path):
    trace = tmp_path / "run2_010_gpt-5-mini.jsonl"
    trace.write_text(
        json.dumps({"event": "llm_usage",
                    "payload": {"model": "gpt-5-mini",
                                "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}}) + "\n"
        # an error event (no usage) must be safely ignored, not counted as $0 tokens
        + json.dumps({"event": "llm_usage", "payload": {"model": "gpt-5-mini", "error": "boom"}}) + "\n"
    )
    usd = ladder.recover_cell_usd("run2", "010")
    assert usd is not None and usd > 0


def test_recover_cell_usd_from_spurious_subdir_trace(tmp_path):
    # Runner quirk (not this driver's to fix): a model slug containing '/' makes TraceRecorder
    # create a spurious subdirectory. recover_cell_usd must still find it.
    subdir = tmp_path / "run2_011_openai"
    subdir.mkdir()
    (subdir / "gpt-5-mini.jsonl").write_text(
        json.dumps({"event": "llm_usage",
                    "payload": {"model": "openai/gpt-5-mini",
                                "usage": {"prompt_tokens": 2000, "completion_tokens": 300}}}) + "\n"
    )
    usd = ladder.recover_cell_usd("run2", "011")
    assert usd is not None and usd > 0


def test_recover_cell_usd_none_when_no_trace():
    assert ladder.recover_cell_usd("run2", "999") is None


# --------------------------------------------------------------------------------- F8 --------
def test_openrouter_key_usage_never_raises_on_bad_key():
    assert ladder.openrouter_key_usage("sk-definitely-not-a-real-key") is None
    assert ladder.openrouter_key_usage("") is None


# --------------------------------------------------------------------------------- F9 --------
def test_pid_lock_reclaims_stale_lock_from_dead_pid(tmp_path):
    lock_path = str(tmp_path / "driver.lock")
    # A pid essentially guaranteed not to be alive.
    Path(lock_path).write_text("999999999")
    ladder.acquire_pid_lock(lock_path)
    assert Path(lock_path).read_text() == str(os.getpid())


def test_pid_lock_refuses_second_acquire_against_live_pid(tmp_path):
    lock_path = str(tmp_path / "driver.lock")
    ladder.acquire_pid_lock(lock_path)  # claims it with our own (live) pid
    with pytest.raises(SystemExit):
        ladder.acquire_pid_lock(lock_path)


def test_run_meta_guard_allows_matching_config(tmp_path):
    meta_path = str(tmp_path / "run_meta.json")
    ladder.check_run_meta(meta_path, {"axis": "final3", "model": ""})
    ladder.check_run_meta(meta_path, {"axis": "final3", "model": ""})  # same config -> no raise


def test_run_meta_guard_refuses_mismatched_config(tmp_path):
    meta_path = str(tmp_path / "run_meta.json")
    ladder.check_run_meta(meta_path, {"axis": "final3", "model": ""})
    with pytest.raises(SystemExit):
        ladder.check_run_meta(meta_path, {"axis": "", "model": "openai/gpt-5-mini"})
