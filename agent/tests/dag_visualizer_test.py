"""Offline tests for the DAG visualizer.

Headless (Agg backend is forced inside the module). Each render must produce a real PNG
that is square and at least the policy floor (1920 px) on a side — the product requirement.
We assert the image dimensions via the PNG IHDR header so the test needs no image library.
"""
import struct

import pytest

from agent.app.testing.dag_visualizer import (
    MIN_SIDE_PX,
    render_plan_dag,
    render_plan_structure,
    render_result_file,
    render_runtime_dag,
)


def _png_size(path):
    """Return (width, height) read straight from the PNG IHDR chunk (no deps)."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width, height = struct.unpack(">II", head[16:24])
    return width, height


FANOUT_PLAN = {
    "leaves": [
        {"id": "a", "instruction": "find A", "expect": "A"},
        {"id": "b", "instruction": "find B", "expect": "B"},
        {"id": "c", "instruction": "find C", "expect": "C"},
    ],
    "aggregation": "combine A, B, C",
}

CHAIN_PLAN = {
    "leaves": [
        {"id": "a", "instruction": "find A", "expect": "A"},
        {"id": "b", "instruction": "find B from {a}", "expect": "B", "depends_on": ["a"]},
        {"id": "c", "instruction": "find C from {b}", "expect": "C", "depends_on": ["b"]},
    ],
    "aggregation": "report C",
}

MIXED_PLAN = {
    "leaves": [
        {"id": "x", "instruction": "find X", "expect": "X"},
        {"id": "y", "instruction": "find Y", "expect": "Y"},
        {"id": "z", "instruction": "find Z from {x}", "expect": "Z", "depends_on": ["x"]},
    ],
    "aggregation": "combine",
}


@pytest.mark.parametrize("plan", [FANOUT_PLAN, CHAIN_PLAN, MIXED_PLAN])
def test_render_plan_dag_is_square_and_hi_res(tmp_path, plan):
    out = render_plan_dag(plan, mandate="test", out_path=str(tmp_path / "p.png"))
    w, h = _png_size(out)
    assert w == h, "export must be square (1:1)"
    assert w >= MIN_SIDE_PX, f"export must be >= {MIN_SIDE_PX}px"


def test_side_px_below_floor_is_clamped(tmp_path):
    out = render_plan_dag(FANOUT_PLAN, side_px=640, out_path=str(tmp_path / "small.png"))
    w, h = _png_size(out)
    assert w == h == MIN_SIDE_PX, "a too-small request must clamp up to the floor"


def test_configurable_larger_size(tmp_path):
    out = render_plan_dag(FANOUT_PLAN, side_px=3000, out_path=str(tmp_path / "big.png"))
    w, h = _png_size(out)
    assert w == h == 3000, "a larger configured size must be honored exactly"


def test_render_plan_structure_topology_only(tmp_path):
    structure = {
        "leaf_count": 3, "edge_count": 2,
        "waves": [["a"], ["b"], ["c"]], "wave_widths": [1, 1, 1],
        "edges": ["a->b", "b->c"], "is_dag_chain": True, "is_pure_fanout": False,
    }
    out = render_plan_structure(structure, out_path=str(tmp_path / "s.png"))
    w, h = _png_size(out)
    assert w == h >= MIN_SIDE_PX


def test_render_runtime_dag(tmp_path):
    graph = {"nodes": {
        "root": {"title": "root", "children": ["v1", "v2"], "status": "done"},
        "v1": {"title": "visit 1", "children": ["m"], "status": "done",
               "details": {"action": "visit"}, "score": 0.9},
        "v2": {"title": "visit 2", "children": ["m"], "status": "done",
               "details": {"action": "visit"}},
        "m": {"title": "merge", "children": [], "status": "done",
              "details": {"action": "merge"}},
    }}
    out = render_runtime_dag(graph, out_path=str(tmp_path / "r.png"))
    w, h = _png_size(out)
    assert w == h >= MIN_SIDE_PX


def test_render_result_file_dispatch_compiled(tmp_path):
    import json
    result = {
        "model": "openai/gpt-4.1-nano", "execution_variant": "graph_compiled",
        "test_metadata": {"id": "051", "statement": "demo mandate"},
        "execution": {"output": {"plan_structure": {
            "leaf_count": 2, "waves": [["a"], ["b"]], "wave_widths": [1, 1],
            "edges": ["a->b"],
        }}},
    }
    p = tmp_path / "res.json"
    p.write_text(json.dumps(result))
    out = render_result_file(str(p), out_path=str(tmp_path / "res.dag.png"))
    w, h = _png_size(out)
    assert w == h >= MIN_SIDE_PX


def test_render_result_file_raises_when_empty(tmp_path):
    import json
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"execution": {"output": {}}}))
    with pytest.raises(ValueError):
        render_result_file(str(p), out_path=str(tmp_path / "x.png"))
