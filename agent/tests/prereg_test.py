"""Unit tests for scripts/prereg.py -- preregistration and denominator auditing.

The trap this closes: a cell that dies before writing output leaves no file, so any analysis that
iterates the results directory cannot see it. langgraph silently lost 6-7 of 48 cells and its mean
was computed over survivors, which made it look like the best arm. The denominator must come from
the experiment design, never from the filesystem.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import prereg  # noqa: E402


SPEC = {
    "run_id": "ledger001",
    "hypothesis": "corpus replay produces identical evidence across arms",
    "tasks": ["122", "130"],
    "arms": ["evidence_loop", "langgraph_react"],
    "reps": 2,
    "model": "qwen2.5:7b",
    "primary_endpoint": "validation.overall_score",
    "budget_usd": 0.0,
    "provider": "corpus",
    "abort_conditions": {"max_infra_failed_rate": 0.2},
}


def _result(tmp_path, run_id, task, model, variant, rep):
    name = f"{run_id}_{task}_{model}_{variant}_cfgdeadbeef_r{rep}.json"
    (tmp_path / name).write_text(json.dumps({"execution": {"output": {}}}), encoding="utf-8")


def test_expected_cells_is_the_full_cartesian_product():
    """2 tasks x 2 arms x 2 reps = 8. This number is the denominator, not the file count."""
    cells = prereg.expected_cells(SPEC)
    assert len(cells) == 8
    assert {c["task"] for c in cells} == {"122", "130"}
    assert {c["rep"] for c in cells} == {1, 2}


def test_validate_accepts_a_complete_spec():
    assert prereg.validate(SPEC) == []


@pytest.mark.parametrize("missing", ["run_id", "hypothesis", "tasks", "arms", "reps",
                                     "primary_endpoint", "abort_conditions"])
def test_validate_names_every_missing_required_field(missing):
    """A prereg without a stated hypothesis or endpoint is a run, not an experiment."""
    spec = {k: v for k, v in SPEC.items() if k != missing}
    errors = prereg.validate(spec)
    assert any(missing in error for error in errors)


def test_validate_rejects_a_non_positive_rep_count():
    assert prereg.validate({**SPEC, "reps": 0}) != []


def test_audit_reports_a_dead_cell_as_missing_not_absent(tmp_path):
    """The whole point: 6 files out of 8 expected means 2 FAILURES, not a sample of 6."""
    for task in ("122", "130"):
        for variant in ("evidence_loop", "langgraph_react"):
            for rep in (1, 2):
                if task == "130" and variant == "langgraph_react":
                    continue  # these two cells died before writing anything
                _result(tmp_path, "ledger001", task, "qwen2.5:7b", variant, rep)
    report = prereg.audit(SPEC, str(tmp_path))
    assert report["expected"] == 8
    assert report["found"] == 6
    assert len(report["missing"]) == 2
    assert all(cell["task"] == "130" for cell in report["missing"])


def test_audit_ignores_summary_and_trace_files(tmp_path):
    """A naive glob counts *_summary.json and *.jsonl; only canonical cells are results."""
    _result(tmp_path, "ledger001", "122", "qwen2.5:7b", "evidence_loop", 1)
    (tmp_path / "ledger001_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ledger001_122_qwen2.5:7b_evidence_loop_cfgdeadbeef_r1.jsonl").write_text(
        "{}\n", encoding="utf-8")
    assert prereg.audit(SPEC, str(tmp_path))["found"] == 1


def test_audit_is_complete_only_when_every_expected_cell_landed(tmp_path):
    for task in ("122", "130"):
        for variant in ("evidence_loop", "langgraph_react"):
            for rep in (1, 2):
                _result(tmp_path, "ledger001", task, "qwen2.5:7b", variant, rep)
    report = prereg.audit(SPEC, str(tmp_path))
    assert report["complete"] is True
    assert report["missing"] == []


def test_audit_of_an_empty_results_dir_is_zero_percent_not_an_error(tmp_path):
    """An unattended run must be able to poll its own progress before any cell has landed."""
    report = prereg.audit(SPEC, str(tmp_path))
    assert report["found"] == 0
    assert report["complete"] is False


def test_write_and_load_roundtrip(tmp_path):
    path = prereg.write(str(tmp_path), SPEC)
    assert prereg.load(str(path))["run_id"] == "ledger001"


def test_write_refuses_an_invalid_spec(tmp_path):
    """A malformed prereg must fail at write time, not silently produce a bad denominator."""
    with pytest.raises(ValueError):
        prereg.write(str(tmp_path), {"run_id": "x"})


def test_audit_does_not_confuse_an_arm_with_another_arms_prefix(tmp_path):
    """ADVERSARIAL: `sequential_react` is a prefix of `sequential_react_extract`.

    A substring match would count every extract cell as a plain sequential_react cell, so a run
    where one arm died entirely would report itself complete.
    """
    spec = {**SPEC, "arms": ["sequential_react", "sequential_react_extract"],
            "tasks": ["122"], "reps": 1}
    _result(tmp_path, "ledger001", "122", "qwen2.5:7b", "sequential_react_extract", 1)
    report = prereg.audit(spec, str(tmp_path))
    assert report["found"] == 1
    assert [c["arm"] for c in report["missing"]] == ["sequential_react"]


def test_audit_does_not_confuse_a_task_with_a_longer_task_id(tmp_path):
    """ADVERSARIAL: task 122 must not be satisfied by a file for task 1220."""
    spec = {**SPEC, "tasks": ["122"], "arms": ["evidence_loop"], "reps": 1}
    _result(tmp_path, "ledger001", "1220", "qwen2.5:7b", "evidence_loop", 1)
    assert prereg.audit(spec, str(tmp_path))["found"] == 0


def test_audit_does_not_confuse_rep_1_with_rep_10(tmp_path):
    spec = {**SPEC, "tasks": ["122"], "arms": ["evidence_loop"], "reps": 1}
    _result(tmp_path, "ledger001", "122", "qwen2.5:7b", "evidence_loop", 10)
    assert prereg.audit(spec, str(tmp_path))["found"] == 0
