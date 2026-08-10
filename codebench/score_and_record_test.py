"""Tests for score_and_record.py's hard-task scoring logic — plain pytest /
sys.path-injection convention, matching badmodel-lab/analyze_calibration_test.py for other
standalone scripts that live outside the services.* package tree."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_and_record as sar  # noqa: E402


KEYSTONE = ["tests/test_fib_mod.py::test_n3", "tests/test_fib_mod.py::test_n6"]


def _report(outcomes: dict) -> dict:
    tests = [{"nodeid": k, "outcome": v} for k, v in outcomes.items()]
    passed = sum(1 for v in outcomes.values() if v == "passed")
    return {"summary": {"total": len(tests), "passed": passed}, "tests": tests}


def test_score_hard_all_pass():
    report = _report({k: "passed" for k in KEYSTONE} | {"tests/test_fib_mod.py::test_n0": "passed"})
    out = sar._score_hard(report, KEYSTONE)
    assert out == {"tests_passed": 3, "tests_total": 3, "score": 1.0, "keystone_pass": 1}


def test_score_hard_partial_credit_keystone_fails():
    report = _report({
        "tests/test_fib_mod.py::test_n3": "passed",
        "tests/test_fib_mod.py::test_n6": "failed",
        "tests/test_fib_mod.py::test_n0": "passed",
    })
    out = sar._score_hard(report, KEYSTONE)
    assert out["tests_passed"] == 2
    assert out["tests_total"] == 3
    assert out["score"] == round(2 / 3, 4)  # _score_hard rounds to 4dp
    assert out["keystone_pass"] == 0, "one keystone test failed -> keystone gate must fail"


def test_score_hard_missing_keystone_node_counts_as_failed():
    # e.g. the agent's file crashed at collection and the keystone test never ran at all.
    report = _report({"tests/test_fib_mod.py::test_n0": "passed"})
    out = sar._score_hard(report, KEYSTONE)
    assert out["keystone_pass"] == 0


def test_score_hard_crash_report_is_zero():
    out = sar._score_hard({"error": "grading suite produced no report"}, KEYSTONE)
    assert out == {"tests_passed": 0, "tests_total": 0, "score": 0.0, "keystone_pass": 0}


def test_score_hard_empty_keystone_list_never_passes():
    # a task with no keystone_test_ids configured shouldn't silently grant a free pass.
    report = _report({"tests/test_fib_mod.py::test_n0": "passed"})
    out = sar._score_hard(report, [])
    assert out["keystone_pass"] == 0


def test_sandbox_actions_count_reads_raw_not_submission(tmp_path):
    cell_dir = tmp_path / "cell"
    (cell_dir / "raw").mkdir(parents=True)
    (cell_dir / "raw" / ".codebench_run_summary.json").write_text(
        json.dumps({"sandbox_actions_count": 7})
    )
    assert sar._sandbox_actions_count(cell_dir) == 7


def test_sandbox_actions_count_absent_is_none(tmp_path):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    assert sar._sandbox_actions_count(cell_dir) is None


def test_main_writes_one_jsonl_row_for_hard_task(tmp_path):
    tasks_dir = tmp_path / "tasks"
    (tasks_dir / "c01").mkdir(parents=True)
    (tasks_dir / "c01" / "meta.json").write_text(json.dumps({
        "category": "hard", "visibility": "visible", "keystone_test_ids": KEYSTONE,
    }))

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "exit_code").write_text("0")
    (cell_dir / "duration_s").write_text("12.5")
    (cell_dir / "grade_report.json").write_text(json.dumps(
        _report({k: "passed" for k in KEYSTONE})
    ))

    results_file = tmp_path / "results" / "runs.jsonl"

    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "score_and_record.py"),
         "--task-id", "c01", "--agent-kind", "badmodel", "--model", "gemma2:2b",
         "--cell-dir", str(cell_dir), "--run-id", "test_run",
         "--results-file", str(results_file), "--tasks-dir", str(tasks_dir)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in results_file.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == "c01"
    assert row["agent_kind"] == "badmodel"
    assert row["model"] == "gemma2:2b"
    assert row["task_category"] == "hard"
    assert row["score"] == 1.0
    assert row["keystone_pass"] == 1
    assert row["sandbox_exit_code"] == "0"
    assert row["duration_s"] == 12.5
