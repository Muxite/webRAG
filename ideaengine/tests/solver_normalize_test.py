"""Unit tests for `solver._normalize_engine_result` — pure helpers, no deps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_solver():
    here = Path(__file__).resolve().parent
    target = here.parent / "ideaengine" / "solver.py"
    spec = importlib.util.spec_from_file_location("_solver_under_test", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_solver_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_solver = _load_solver()


def test_normalize_minimum_shape():
    raw = {"final_deliverable": "the answer", "success": True}
    out = _solver._normalize_engine_result(raw)
    assert out["final_deliverable"] == "the answer"
    assert out["success"] is True
    assert "observability" in out
    assert out["observability"]["visit"]["count"] == 0


def test_normalize_synthesizes_observability_from_graph():
    raw = {
        "final_deliverable": "ans",
        "success": True,
        "graph": {
            "nodes": {
                "n1": {"details": {"action": "search"}},
                "n2": {"details": {"action": "visit"}},
                "n3": {"details": {"action": "visit"}},
                "n4": {"details": {"action": "think"}},
            }
        },
    }
    out = _solver._normalize_engine_result(raw)
    obs = out["observability"]
    assert obs["search"]["count"] == 1
    assert obs["visit"]["count"] == 2
    assert obs["think"]["count"] == 1
    assert obs["save"]["count"] == 0


def test_normalize_carries_optional_fields():
    raw = {
        "final_deliverable": "x",
        "success": False,
        "goal_achieved": False,
        "has_failures": True,
        "warning": "pending nodes remain",
        "graph": {"nodes": {}},
        "got_stats": {"nodes_pruned": 3},
    }
    out = _solver._normalize_engine_result(raw)
    assert out["goal_achieved"] is False
    assert out["has_failures"] is True
    assert out["warning"] == "pending nodes remain"
    assert "graph" in out
    assert out["observability"].get("got_stats") == {"nodes_pruned": 3}


def test_normalize_coerces_missing_keys():
    raw = {}
    out = _solver._normalize_engine_result(raw)
    assert out["final_deliverable"] == ""
    assert out["success"] is False
