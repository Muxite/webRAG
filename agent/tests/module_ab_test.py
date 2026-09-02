"""`scripts/module_ab.py` — the host-vs-host+module analysis.

The behaviour worth pinning is what this script REFUSES to do: report a missing evidence graph as
a 0% fabrication rate, and lead with a score ranking the suite cannot resolve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from module_ab import load_run, structural, summarize  # noqa: E402


def _write(dirpath: Path, run: str, task: int, score: float, graph=None):
    payload = {
        "test_metadata": {"test_id": str(task)},
        "validation": {"overall_score": score},
        "infra_failed": False,
        "execution": {"output": ({"evidence_graph": graph} if graph is not None else {})},
    }
    (dirpath / f"{run}_{task}_m_v_cfg0_r1.json").write_text(json.dumps(payload))


def test_a_cell_with_no_evidence_graph_reports_unknown_never_zero(tmp_path):
    """Absent is not zero. A host without the module cannot have a fabrication rate computed at
    all -- reporting 0.0 would credit it for an audit it never underwent, and would make the
    module look like it changed nothing."""
    _write(tmp_path, "off", 210, 0.5)

    facts = structural(json.loads((tmp_path / "off_210_m_v_cfg0_r1.json").read_text()))

    assert facts["has_graph"] is False
    assert facts["fabrication_rate"] is None
    assert summarize(load_run("off", tmp_path))["fabrication_rate"] is None


def test_derived_nodes_and_invalid_ones_are_counted_separately(tmp_path):
    graph = {"pages": [], "nodes": [
        {"kind": "source", "value": "1"},
        {"kind": "derived", "value": "2", "derivation_valid": True},
        {"kind": "derived", "value": "3", "derivation_valid": False},
    ]}
    _write(tmp_path, "on", 210, 0.5, graph)

    got = summarize(load_run("on", tmp_path))

    assert got["derived_nodes"] == 2
    assert got["invalid_nodes"] == 1
    assert got["fabrication_rate"] == 0.5
    assert got["derivation_available"] == 1


def test_infra_failed_cells_are_dropped_from_both_sides(tmp_path):
    _write(tmp_path, "on", 210, 0.5)
    bad = json.loads((tmp_path / "on_210_m_v_cfg0_r1.json").read_text())
    bad["infra_failed"] = True
    (tmp_path / "on_211_m_v_cfg0_r1.json").write_text(json.dumps(bad))

    assert set(load_run("on", tmp_path)) == {210}


def test_the_report_never_presents_the_score_delta_as_a_ranking(tmp_path):
    """The suite's measured trajectory-chaos floor makes a score ordering between two
    configurations unresolvable at this n, so the output must say so wherever it prints one."""
    from module_ab import report

    text = report("off", "on", summarize({}), summarize({}), [0.0, 0.6, -0.6])

    assert "GUARD" in text
    assert "not a ranking" in text
    assert "tasks that moved at all" in text
