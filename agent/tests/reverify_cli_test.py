"""Tests for scripts/reverify.py -- fully offline (no model, no network, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import reverify as rv  # noqa: E402

from agent.app.testing import execution_evidence_loop as el
from agent.app.testing.evidence_graph import EvidenceGraph


def _cell_with_pages_only(extractions, pages):
    return {"execution": {"output": {"extractions": extractions, "pages": pages}}}


def _cell_with_graph(extractions, pages, graph_artifact):
    return {
        "execution": {
            "output": {"extractions": extractions, "pages": pages, "evidence_graph": graph_artifact}
        }
    }


PAGE_TEXT = "Toni Morrison wrote Beloved."


# --------------------------------------------------------------------------------------
# resolve_output
# --------------------------------------------------------------------------------------

def test_resolve_output_unwraps_execution_then_output():
    output = {"extractions": [], "pages": []}
    assert rv.resolve_output({"execution": {"output": output}}) is output
    assert rv.resolve_output({"output": output}) is output
    assert rv.resolve_output(output) is output


def test_resolve_output_returns_empty_dict_for_junk():
    assert rv.resolve_output({"nope": 1}) == {"nope": 1}
    assert rv.resolve_output("not a dict") == {}


# --------------------------------------------------------------------------------------
# load_cell
# --------------------------------------------------------------------------------------

def test_load_cell_reads_json_object(tmp_path):
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"a": 1}))
    assert rv.load_cell(path) == {"a": 1}


def test_load_cell_rejects_a_non_object_top_level(tmp_path):
    path = tmp_path / "cell.json"
    path.write_text(json.dumps([1, 2, 3]))
    try:
        rv.load_cell(path)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------------------
# run_reverify -- with and without an evidence_graph artifact
# --------------------------------------------------------------------------------------

def test_run_reverify_reports_missing_graph_status_when_no_artifact():
    pages = [el.store_page("p1", "https://a.example", PAGE_TEXT, 1000)]
    extractions = [{"page_id": "p1", "quote": "Toni Morrison wrote Beloved"}]
    cell = _cell_with_pages_only(extractions, pages)
    report = rv.run_reverify(cell)
    assert report["graph_status"] == rv.GRAPH_STATUS_MISSING
    assert report["graph"] is None
    assert report["cell"]["counts"]["verified"] == 1


def test_run_reverify_runs_the_graph_audit_when_an_artifact_is_present():
    graph = EvidenceGraph()
    graph.add_page("p1", "https://a.example", PAGE_TEXT)
    graph.add_source("p1", "Toni Morrison", quote="Toni Morrison")
    artifact = graph.to_dict()

    pages = [el.store_page("p1", "https://a.example", PAGE_TEXT, 1000)]
    extractions = [{"page_id": "p1", "quote": "Toni Morrison wrote Beloved"}]
    cell = _cell_with_graph(extractions, pages, artifact)

    report = rv.run_reverify(cell)
    assert report["graph_status"] == rv.GRAPH_STATUS_OK
    assert report["graph"]["counts"]["verified"] == 1
    assert report["graph"]["counts"]["failed"] == 0


def test_run_reverify_graph_audit_catches_page_drift():
    graph = EvidenceGraph()
    graph.add_page("p1", "https://a.example", PAGE_TEXT)
    graph.add_source("p1", "Toni Morrison", quote="Toni Morrison")
    artifact = graph.to_dict()
    artifact["pages"][0]["text"] = "an entirely different page now"

    cell = _cell_with_graph([], [], artifact)
    report = rv.run_reverify(cell)
    assert report["graph"]["counts"]["failed"] == 1
    assert any(row["drifted"] or row["verified"] is False for row in report["graph"]["nodes"])


# --------------------------------------------------------------------------------------
# has_genuine_failure
# --------------------------------------------------------------------------------------

def test_has_genuine_failure_true_on_cell_failure():
    pages = [el.store_page("p1", "https://a.example", PAGE_TEXT, 1000)]
    extractions = [{"page_id": "p1", "quote": "totally absent text"}]
    report = rv.run_reverify(_cell_with_pages_only(extractions, pages))
    assert report["cell"]["counts"]["failed"] == 1
    assert rv.has_genuine_failure(report) is True


def test_has_genuine_failure_false_when_only_unverifiable():
    report = rv.run_reverify(_cell_with_pages_only([{"page_id": "p9", "quote": "orphan"}], []))
    assert report["cell"]["counts"]["failed"] == 0
    assert rv.has_genuine_failure(report) is False


# --------------------------------------------------------------------------------------
# main() CLI
# --------------------------------------------------------------------------------------

def test_main_text_report_exits_zero_by_default(tmp_path, capsys):
    pages = [el.store_page("p1", "https://a.example", PAGE_TEXT, 1000)]
    extractions = [{"page_id": "p1", "quote": "totally absent text"}]
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(_cell_with_pages_only(extractions, pages)))

    exit_code = rv.main([str(cell_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "reverify_cell" in out
    assert "SKIPPED" in out


def test_main_strict_exits_nonzero_on_genuine_failure(tmp_path, capsys):
    pages = [el.store_page("p1", "https://a.example", PAGE_TEXT, 1000)]
    extractions = [{"page_id": "p1", "quote": "totally absent text"}]
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(_cell_with_pages_only(extractions, pages)))

    exit_code = rv.main([str(cell_path), "--strict"])
    assert exit_code == 1


def test_main_json_output_is_valid_and_matches_run_reverify(tmp_path, capsys):
    pages = [el.store_page("p1", "https://a.example", PAGE_TEXT, 1000)]
    extractions = [{"page_id": "p1", "quote": "Toni Morrison wrote Beloved"}]
    cell = _cell_with_pages_only(extractions, pages)
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cell))

    exit_code = rv.main([str(cell_path), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["graph_status"] == rv.GRAPH_STATUS_MISSING
    assert payload["cell"]["counts"]["verified"] == 1


def test_main_reports_a_clear_error_for_a_missing_file(tmp_path, capsys):
    exit_code = rv.main([str(tmp_path / "nope.json")])
    assert exit_code == 1
    assert "reverify:" in capsys.readouterr().err
