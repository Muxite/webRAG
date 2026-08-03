"""Tests for analyze_code.py — plain pytest / sys.path-injection convention, matching
badmodel-lab/analyze_calibration_test.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_code as ac  # noqa: E402


def _row(model, agent_kind, task_category, score, keystone_pass=1, tests_passed=3, tests_total=3):
    return {
        "model": model, "agent_kind": agent_kind, "task_category": task_category,
        "score": score, "keystone_pass": keystone_pass,
        "tests_passed": tests_passed, "tests_total": tests_total,
    }


def test_load_runs_parses_jsonl(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    assert ac.load_runs(path) == [{"a": 1}, {"a": 2}]


def test_load_runs_missing_file_returns_empty(tmp_path):
    assert ac.load_runs(tmp_path / "missing.jsonl") == []


def test_load_runs_ignores_blank_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert ac.load_runs(path) == [{"a": 1}, {"a": 2}]


def test_mean_ignores_none():
    assert ac._mean([1.0, None, 3.0]) == 2.0


def test_mean_empty_is_none():
    assert ac._mean([None, None]) is None


def test_comparison_runs_without_crashing_on_mixed_categories(capsys):
    rows = [
        _row("gemma2:2b", "badmodel", "hard", 1.0),
        _row("gemma2:2b", "aider", "hard", 0.5, keystone_pass=0),
        _row("gemma2:2b", "badmodel", "soft", None, keystone_pass=None, tests_passed=None, tests_total=None),
    ]
    ac.print_agent_kind_comparison(rows)
    ac.print_unscored_warning(rows)
    out = capsys.readouterr().out
    assert "gemma2:2b" in out
    assert "badmodel beats aider" in out
    assert "soft-task judging isn't wired yet" in out


def test_main_handles_empty_results_file(tmp_path):
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "analyze_code.py"),
         "--results-file", str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "no rows" in result.stdout
