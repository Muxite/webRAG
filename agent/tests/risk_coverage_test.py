"""Unit tests for scripts/risk_coverage.py.

Risk-coverage over arm_verdict.derive_verdict is the differentiator described in the module
docstring: before ``arm_verdict`` gave every arm a verdict, only ``evidence_loop`` could ever
plot a curve. These tests exercise the pure computation (``coverage_point`` /
``risk_coverage_curve`` / ``curves_by_variant``) directly against synthetic verdict/score pairs,
plus one integration test of ``load_cells`` against real files written to a temp dir in the
on-disk shape ``bench_common.load_row`` / stored cells actually use.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import risk_coverage  # noqa: E402

from agent.app.testing.arm_verdict import VERDICT_ANSWER, VERDICT_PARTIAL, VERDICT_ABSTAIN


def _cell(verdict, score, variant="evidence_loop", test_id="t1"):
    return {"verdict": verdict, "score": score, "variant": variant, "test_id": test_id}


# ---------------------------------------------------------------------------------------------
# coverage_point / risk_coverage_curve
# ---------------------------------------------------------------------------------------------

def test_all_answer_all_correct_zero_risk_full_coverage():
    cells = [_cell(VERDICT_ANSWER, 1.0) for _ in range(5)]
    point = risk_coverage.coverage_point(cells, min_rank=2)
    assert point["coverage"] == 1.0
    assert point["selective_accuracy"] == 1.0
    assert point["risk"] == 0.0
    assert point["n"] == 5 and point["total"] == 5


def test_abstain_only_at_answer_rank_gives_zero_coverage_and_nan_risk():
    cells = [_cell(VERDICT_ABSTAIN, 0.0) for _ in range(4)]
    point = risk_coverage.coverage_point(cells, min_rank=2)
    assert point["coverage"] == 0.0
    assert point["n"] == 0
    assert math.isnan(point["selective_accuracy"])
    assert math.isnan(point["risk"])


def test_all_rank_zero_always_covers_everything_regardless_of_verdict():
    cells = [_cell(VERDICT_ABSTAIN, 0.0), _cell(VERDICT_ANSWER, 1.0), _cell(VERDICT_PARTIAL, 0.5)]
    point = risk_coverage.coverage_point(cells, min_rank=0)
    assert point["coverage"] == 1.0
    assert point["n"] == 3


def test_selective_accuracy_correct_uses_threshold_not_equality():
    cells = [_cell(VERDICT_ANSWER, 0.6), _cell(VERDICT_ANSWER, 0.4)]
    point = risk_coverage.coverage_point(cells, min_rank=2, threshold=0.5)
    # one of two ANSWER cells clears the 0.5 threshold
    assert point["n"] == 2
    assert point["selective_accuracy"] == 0.5
    assert point["risk"] == 0.5


def test_curve_is_monotonically_non_decreasing_coverage_as_tau_relaxes():
    cells = ([_cell(VERDICT_ANSWER, 1.0)] * 2 + [_cell(VERDICT_PARTIAL, 0.5)] * 3
            + [_cell(VERDICT_ABSTAIN, 0.0)] * 5)
    curve = risk_coverage.risk_coverage_curve(cells)
    coverages = [p["coverage"] for p in curve]  # ANSWER_ONLY, ANSWER_OR_PARTIAL, ALL
    assert coverages == sorted(coverages)
    assert coverages[-1] == 1.0


def test_curve_labels_in_order_most_confident_first():
    cells = [_cell(VERDICT_ANSWER, 1.0)]
    curve = risk_coverage.risk_coverage_curve(cells)
    assert [p["tau"] for p in curve] == ["ANSWER_ONLY", "ANSWER_OR_PARTIAL", "ALL"]


def test_curves_by_variant_keeps_arms_separate():
    cells = [_cell(VERDICT_ANSWER, 1.0, variant="graph"),
             _cell(VERDICT_ABSTAIN, 0.0, variant="langgraph_react")]
    curves = risk_coverage.curves_by_variant(cells)
    assert set(curves.keys()) == {"graph", "langgraph_react"}
    # graph's ANSWER_ONLY point covers its one cell; langgraph_react's covers none.
    graph_answer_only = curves["graph"][0]
    lg_answer_only = curves["langgraph_react"][0]
    assert graph_answer_only["n"] == 1
    assert lg_answer_only["n"] == 0


def test_is_correct_none_score_is_never_correct():
    assert risk_coverage.is_correct(None) is False


# ---------------------------------------------------------------------------------------------
# load_cells integration: real on-disk shape, via bench_common's own path scoping
# ---------------------------------------------------------------------------------------------

def _write_cell(path, *, variant, score, final_deliverable, pages=None):
    payload = {
        "test_metadata": {"test_id": "999"},
        "model": "test-model",
        "execution_variant": variant,
        "execution": {
            "output": {"final_deliverable": final_deliverable, "pages": pages or []},
            "graph": {},
            "observability": {},
        },
        "validation": {"overall_score": score},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_cells_reads_real_files_and_derives_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(risk_coverage.bench_common, "results_dir", lambda: tmp_path)
    _write_cell(
        tmp_path / "myrun_999_test-model_evidence_loop_cfgabc_r1.json",
        variant="evidence_loop", score=1.0,
        final_deliverable="The value is 20310 feet.",
        pages=[{"url": "https://example.com/x", "text": "Recorded value: 20310 feet."}],
    )
    _write_cell(
        tmp_path / "myrun_999_test-model_langgraph_react_cfgabc_r1.json",
        variant="langgraph_react", score=1.0,
        final_deliverable="The value is 20310 feet.",
    )
    cells = risk_coverage.load_cells(run_ids=["myrun"])
    by_variant = {c["variant"]: c for c in cells}
    assert by_variant["evidence_loop"]["verdict"] == VERDICT_ANSWER
    # langgraph_react persists no page-level evidence in this shape -> honestly ABSTAIN, not a
    # fabricated ANSWER just because the text happens to match.
    assert by_variant["langgraph_react"]["verdict"] == VERDICT_ABSTAIN


def test_load_cells_skips_files_with_no_score(tmp_path, monkeypatch):
    monkeypatch.setattr(risk_coverage.bench_common, "results_dir", lambda: tmp_path)
    path = tmp_path / "myrun_999_test-model_graph_cfgabc_r1.json"
    payload = {
        "test_metadata": {"test_id": "999"}, "model": "test-model",
        "execution_variant": "graph",
        "execution": {"output": {"final_deliverable": "x"}, "graph": {}, "observability": {}},
        "validation": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    cells = risk_coverage.load_cells(run_ids=["myrun"])
    assert cells == []


def test_load_cells_variant_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(risk_coverage.bench_common, "results_dir", lambda: tmp_path)
    _write_cell(tmp_path / "myrun_999_m_graph_cfgabc_r1.json", variant="graph", score=1.0,
               final_deliverable="x")
    _write_cell(tmp_path / "myrun_999_m_evidence_loop_cfgabc_r1.json", variant="evidence_loop",
               score=1.0, final_deliverable="x")
    cells = risk_coverage.load_cells(run_ids=["myrun"], variants=["graph"])
    assert {c["variant"] for c in cells} == {"graph"}
