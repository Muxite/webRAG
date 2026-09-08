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


def _result(tmp_path, run_id, task, model, variant, rep, infra_failed=False):
    name = f"{run_id}_{task}_{model}_{variant}_cfgdeadbeef_r{rep}.json"
    body = {"execution": {"output": {}}, "infra_failed": infra_failed}
    (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")


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


# -- abort_conditions gating -------------------------------------------------------------------
#
# validate() previously only checked that ``abort_conditions`` was present; audit() never read
# its contents, so the three conditions already in use in real specs (min_completion_rate,
# max_infra_failed_rate, max_live_fallbacks) were documentation, not gates. These tests pin down
# that audit() actually evaluates them against the landed cells.

def _full_spec(tmp_path, abort_conditions):
    spec = {**SPEC, "tasks": ["122", "130"], "arms": ["evidence_loop", "langgraph_react"],
            "reps": 2, "abort_conditions": abort_conditions}
    for task in ("122", "130"):
        for variant in ("evidence_loop", "langgraph_react"):
            for rep in (1, 2):
                _result(tmp_path, "ledger001", task, "qwen2.5:7b", variant, rep)
    return spec


def test_validate_rejects_an_unrecognised_abort_condition_key():
    """A typo'd key (e.g. max_infra_falied_rate) must not silently disable a gate."""
    spec = {**SPEC, "abort_conditions": {"max_infra_falied_rate": 0.2}}
    errors = prereg.validate(spec)
    assert any("max_infra_falied_rate" in error for error in errors)


def test_write_refuses_a_spec_with_an_unknown_abort_condition_key(tmp_path):
    spec = {**SPEC, "abort_conditions": {"min_completion_rate": 0.9, "totally_made_up": 1}}
    with pytest.raises(ValueError):
        prereg.write(str(tmp_path), spec)


def test_audit_min_completion_rate_passes_when_met(tmp_path):
    spec = _full_spec(tmp_path, {"min_completion_rate": 0.5})
    report = prereg.audit(spec, str(tmp_path))
    assert report["gates"]["min_completion_rate"]["status"] == "pass"
    assert report["gates_passed"] is True


def test_audit_min_completion_rate_fails_when_missed(tmp_path):
    spec = {**SPEC, "tasks": ["122", "130"], "arms": ["evidence_loop", "langgraph_react"],
            "reps": 2, "abort_conditions": {"min_completion_rate": 0.9}}
    # only land 6 of 8 cells
    for task in ("122", "130"):
        for variant in ("evidence_loop", "langgraph_react"):
            for rep in (1, 2):
                if task == "130" and variant == "langgraph_react":
                    continue
                _result(tmp_path, "ledger001", task, "qwen2.5:7b", variant, rep)
    report = prereg.audit(spec, str(tmp_path))
    assert report["gates"]["min_completion_rate"]["status"] == "fail"
    assert report["gates_passed"] is False


def test_audit_max_infra_failed_rate_passes_when_zero_failures(tmp_path):
    spec = _full_spec(tmp_path, {"max_infra_failed_rate": 0.2})
    report = prereg.audit(spec, str(tmp_path))
    assert report["gates"]["max_infra_failed_rate"]["status"] == "pass"
    assert report["gates"]["max_infra_failed_rate"]["value"] == 0.0


def test_audit_max_infra_failed_rate_fails_when_exceeded(tmp_path):
    spec = {**SPEC, "tasks": ["122", "130"], "arms": ["evidence_loop", "langgraph_react"],
            "reps": 2, "abort_conditions": {"max_infra_failed_rate": 0.2}}
    for task in ("122", "130"):
        for variant in ("evidence_loop", "langgraph_react"):
            for rep in (1, 2):
                failed = task == "130" and rep == 1
                _result(tmp_path, "ledger001", task, "qwen2.5:7b", variant, rep,
                        infra_failed=failed)
    report = prereg.audit(spec, str(tmp_path))
    gate = report["gates"]["max_infra_failed_rate"]
    assert gate["status"] == "fail"
    assert gate["value"] == pytest.approx(2 / 8)


def test_audit_max_live_fallbacks_is_always_unknown_not_a_silent_pass(tmp_path):
    """Live-fallback counts live only on the in-memory connector; no stored cell records them.

    A gate that always reports "pass" for data it never actually checked is worse than no gate,
    so this must come back UNKNOWN, never "pass".
    """
    spec = _full_spec(tmp_path, {"max_live_fallbacks": 0})
    report = prereg.audit(spec, str(tmp_path))
    gate = report["gates"]["max_live_fallbacks"]
    assert gate["status"] == "unknown"
    assert report["gates_passed"] is False


def test_audit_unknown_gate_never_reads_as_overall_pass(tmp_path):
    spec = _full_spec(tmp_path, {"min_completion_rate": 0.5, "max_live_fallbacks": 0})
    report = prereg.audit(spec, str(tmp_path))
    assert report["gates"]["min_completion_rate"]["status"] == "pass"
    assert report["gates"]["max_live_fallbacks"]["status"] == "unknown"
    assert report["gates_passed"] is False


def test_audit_preserves_the_existing_return_contract(tmp_path):
    """audit() is already consumed elsewhere; the original keys must remain, only extended."""
    spec = _full_spec(tmp_path, {"min_completion_rate": 0.5})
    report = prereg.audit(spec, str(tmp_path))
    for key in ("run_id", "expected", "found", "missing", "complete", "completion_rate"):
        assert key in report


# --------------------------------------------------------------------- max_live_fallbacks gate


def _cell_with_search(tmp_path, name, *, live=None):
    """A landed cell whose observability carries (or omits) the provenance block."""
    import json
    search = {"count": 3}
    if live is not None:
        search.update({"live_fallbacks": live, "corpus_hits": 3 - live, "empty_results": 0})
    path = tmp_path / name
    path.write_text(json.dumps({
        "test_metadata": {"test_id": "210"},
        "execution": {"observability": {"search": search}},
        "infra_failed": False,
    }))
    return path


def _spec_of_cells_without_provenance(tmp_path, *, timings):
    """The full 8-cell grid, every cell lacking the provenance block. ``timings=None`` omits the
    telemetry block entirely; otherwise it is written verbatim."""
    execution = {"observability": {"search": {"count": 0}}}
    if timings is not None:
        execution["telemetry_raw"] = {"timings": timings}
    for task in ("122", "130"):
        for variant in ("evidence_loop", "langgraph_react"):
            for rep in (1, 2):
                name = f"ledger001_{task}_qwen2.5:7b_{variant}_cfgdeadbeef_r{rep}.json"
                (tmp_path / name).write_text(json.dumps({
                    "test_metadata": {"test_id": task},
                    "execution": execution,
                    "infra_failed": False,
                }), encoding="utf-8")
    return {**SPEC, "tasks": ["122", "130"], "arms": ["evidence_loop", "langgraph_react"],
            "reps": 2, "abort_conditions": {"max_live_fallbacks": 0}}


def test_live_fallback_gate_counts_a_cell_that_never_searched_as_a_real_zero(tmp_path):
    """The provenance block is only written when a search backend is touched, so a model that
    never searches (qwen2.5:0.5b invents URLs instead) leaves the key absent while its true
    live-fallback count is provably 0 — you cannot fall back on a search you never made.
    Without this the gate reports UNKNOWN for exactly the weakest models on the ladder."""
    spec = _spec_of_cells_without_provenance(tmp_path, timings=[{"name": "visit"}])
    gate = prereg.audit(spec, str(tmp_path))["gates"]["max_live_fallbacks"]
    assert gate["status"] == "pass", gate


def test_live_fallback_gate_is_unknown_when_telemetry_is_absent_entirely(tmp_path):
    """Absent is never zero. A missing timings block proves nothing about whether a search ran —
    telemetry may simply not have been captured — so it must NOT be folded in as a genuine 0."""
    spec = _spec_of_cells_without_provenance(tmp_path, timings=None)
    gate = prereg.audit(spec, str(tmp_path))["gates"]["max_live_fallbacks"]
    assert gate["status"] == "unknown", gate


def test_live_fallback_gate_is_unknown_when_a_search_ran_but_left_no_provenance(tmp_path):
    """The original stale-code case must still read UNKNOWN: a search DID run, so its
    live-fallback count is a real unknown rather than a provable zero."""
    spec = _spec_of_cells_without_provenance(tmp_path, timings=[{"name": "search"}])
    gate = prereg.audit(spec, str(tmp_path))["gates"]["max_live_fallbacks"]
    assert gate["status"] == "unknown", gate


def test_live_fallback_gate_passes_when_every_cell_recorded_zero(tmp_path):
    """The gate became checkable when 88a57429 persisted per-search provenance.

    It was hardcoded UNKNOWN because the count used to live only on the in-memory
    ConnectorSearchCorpus instance. It is now written to observability.search.live_fallbacks, and
    a gate that reports UNKNOWN when the data exists is a gate that silently stopped working.
    """
    from scripts.prereg import _evaluate_abort_conditions

    paths = [_cell_with_search(tmp_path, f"c{i}.json", live=0) for i in range(3)]
    gates = _evaluate_abort_conditions({"abort_conditions": {"max_live_fallbacks": 0}}, 1.0, paths)

    assert gates["max_live_fallbacks"]["status"] == "pass"
    assert gates["max_live_fallbacks"]["value"] == 0


def test_live_fallback_gate_sums_across_the_run_and_can_fail(tmp_path):
    """Run-level budget: the question is whether this run touched live search at all, so the
    counts SUM. A max-per-cell rule would let many small leaks through unnoticed."""
    from scripts.prereg import _evaluate_abort_conditions

    paths = [_cell_with_search(tmp_path, "a.json", live=1),
             _cell_with_search(tmp_path, "b.json", live=2)]
    gates = _evaluate_abort_conditions({"abort_conditions": {"max_live_fallbacks": 0}}, 1.0, paths)

    assert gates["max_live_fallbacks"]["status"] == "fail"
    assert gates["max_live_fallbacks"]["value"] == 3


def test_live_fallback_gate_is_unknown_when_any_cell_predates_the_field(tmp_path):
    """Absent must never read as zero. A cell written before 88a57429 carries no provenance at
    all, and folding it in as 0 would manufacture a pass the data cannot support."""
    from scripts.prereg import _evaluate_abort_conditions

    paths = [_cell_with_search(tmp_path, "new.json", live=0),
             _cell_with_search(tmp_path, "old.json", live=None)]
    gates = _evaluate_abort_conditions({"abort_conditions": {"max_live_fallbacks": 0}}, 1.0, paths)

    assert gates["max_live_fallbacks"]["status"] == "unknown"
    assert "1" in gates["max_live_fallbacks"]["detail"]


# ---------------------------------------------------------------------------
# per-arm completion + min_usable_paired_n
# ---------------------------------------------------------------------------

def test_audit_reports_completion_per_arm(tmp_path):
    """A run-wide rate hides a dead arm: 8 of 8 minus one whole arm still reads 50%."""
    for task in ("122", "130"):
        for rep in (1, 2):
            _result(tmp_path, "ledger001", task, "qwen2.5:7b", "evidence_loop", rep)
    _result(tmp_path, "ledger001", "122", "qwen2.5:7b", "langgraph_react", 1)
    report = prereg.audit(SPEC, str(tmp_path))
    per_arm = report["per_arm"]
    assert per_arm["evidence_loop"] == {"expected": 4, "found": 4, "missing": 0,
                                        "completion_rate": 1.0}
    assert per_arm["langgraph_react"]["found"] == 1
    assert per_arm["langgraph_react"]["completion_rate"] == 0.25


def test_min_completion_rate_gate_fails_when_one_arm_is_short(tmp_path):
    """Run-wide completion 0.75 clears a 0.7 threshold; the dead arm's 0.5 must not."""
    spec = dict(SPEC, abort_conditions={"min_completion_rate": 0.7})
    for task in ("122", "130"):
        for rep in (1, 2):
            _result(tmp_path, "ledger001", task, "qwen2.5:7b", "evidence_loop", rep)
    _result(tmp_path, "ledger001", "122", "qwen2.5:7b", "langgraph_react", 1)
    _result(tmp_path, "ledger001", "130", "qwen2.5:7b", "langgraph_react", 1)
    report = prereg.audit(spec, str(tmp_path))
    assert report["completion_rate"] == 0.75
    gate = report["gates"]["min_completion_rate"]
    assert gate["status"] == "fail"
    assert "langgraph_react" in gate["detail"]
    assert report["gates_passed"] is False


def test_min_completion_rate_gate_passes_when_every_arm_clears(tmp_path):
    spec = dict(SPEC, abort_conditions={"min_completion_rate": 0.7})
    for arm in ("evidence_loop", "langgraph_react"):
        for task in ("122", "130"):
            for rep in (1, 2):
                _result(tmp_path, "ledger001", task, "qwen2.5:7b", arm, rep)
    report = prereg.audit(spec, str(tmp_path))
    assert report["gates"]["min_completion_rate"]["status"] == "pass"
    assert report["gates_passed"] is True


def test_min_usable_paired_n_is_a_known_abort_condition():
    spec = dict(SPEC, abort_conditions={"min_usable_paired_n": 2})
    assert prereg.validate(spec) == []


def test_validate_still_rejects_unknown_abort_conditions():
    spec = dict(SPEC, abort_conditions={"min_usable_pared_n": 2})
    errors = prereg.validate(spec)
    assert any("unknown abort_conditions key" in e for e in errors)


def test_min_usable_paired_n_counts_tasks_present_for_every_arm(tmp_path):
    # task 122 landed for both arms; task 130 only for evidence_loop -> 1 usable paired task.
    spec = dict(SPEC, abort_conditions={"min_usable_paired_n": 2})
    for task in ("122", "130"):
        for rep in (1, 2):
            _result(tmp_path, "ledger001", task, "qwen2.5:7b", "evidence_loop", rep)
    _result(tmp_path, "ledger001", "122", "qwen2.5:7b", "langgraph_react", 1)
    report = prereg.audit(spec, str(tmp_path))
    assert report["usable_paired_n"] == 1
    gate = report["gates"]["min_usable_paired_n"]
    assert gate["status"] == "fail"
    assert gate["value"] == 1


def test_min_usable_paired_n_passes_when_every_task_is_paired(tmp_path):
    spec = dict(SPEC, abort_conditions={"min_usable_paired_n": 2})
    for arm in ("evidence_loop", "langgraph_react"):
        for task in ("122", "130"):
            _result(tmp_path, "ledger001", task, "qwen2.5:7b", arm, 1)
    report = prereg.audit(spec, str(tmp_path))
    assert report["usable_paired_n"] == 2
    assert report["gates"]["min_usable_paired_n"]["status"] == "pass"


def test_audit_cli_prints_the_per_arm_table(tmp_path, capsys, monkeypatch):
    prereg_dir = tmp_path / "prereg"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    prereg.write(str(prereg_dir), SPEC)
    for arm in ("evidence_loop", "langgraph_react"):
        for task in ("122", "130"):
            for rep in (1, 2):
                _result(results_dir, "ledger001", task, "qwen2.5:7b", arm, rep)
    monkeypatch.setattr(sys, "argv", ["prereg.py", "audit", "--run-id", "ledger001",
                                      "--prereg-dir", str(prereg_dir),
                                      "--results-dir", str(results_dir)])
    assert prereg.main() == 0
    out = capsys.readouterr().out
    assert "per-arm completion" in out
    assert "evidence_loop" in out and "langgraph_react" in out
