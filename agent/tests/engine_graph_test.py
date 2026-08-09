"""
Unit tests for the EngineStateGraph framework.
"""
from __future__ import annotations

import pytest

from agent.app.engine_graph import EngineStateGraph, GraphPhase


@pytest.mark.asyncio
async def test_linear_flow():
    async def a(s):
        s["seen"] = s.get("seen", []) + ["a"]
        return s

    async def b(s):
        s["seen"].append("b")
        return s

    g = EngineStateGraph()
    g.add_node("a", a).add_node("b", b)
    g.add_edge("a", "b").add_edge("b", EngineStateGraph.END)
    g.set_entry("a")
    runnable = g.compile()
    final = await runnable.invoke({})
    assert final["seen"] == ["a", "b"]
    assert final["_trace"] == ["a", "b"]


@pytest.mark.asyncio
async def test_conditional_routing():
    async def root(s):
        return s

    async def left(s):
        s["went"] = "left"
        return s

    async def right(s):
        s["went"] = "right"
        return s

    def router(s):
        return "L" if s.get("choose_left") else "R"

    g = EngineStateGraph()
    g.add_node("root", root).add_node("left", left).add_node("right", right)
    g.add_conditional_edge("root", router, {"L": "left", "R": "right"})
    g.add_edge("left", EngineStateGraph.END)
    g.add_edge("right", EngineStateGraph.END)
    g.set_entry("root")
    runnable = g.compile()

    final_left = await runnable.invoke({"choose_left": True})
    final_right = await runnable.invoke({"choose_left": False})
    assert final_left["went"] == "left"
    assert final_right["went"] == "right"


def test_compile_rejects_missing_entry():
    g = EngineStateGraph()
    with pytest.raises(ValueError, match="Entry"):
        g.compile()


def test_compile_rejects_dangling_node():
    async def a(s):
        return s

    g = EngineStateGraph()
    g.add_node("a", a)
    g.set_entry("a")
    with pytest.raises(ValueError, match="no outgoing edge"):
        g.compile()


def test_compile_rejects_edge_to_unknown():
    async def a(s):
        return s

    g = EngineStateGraph()
    g.add_node("a", a)
    g.add_edge("a", "missing")
    g.set_entry("a")
    with pytest.raises(ValueError, match="unknown node"):
        g.compile()


def test_duplicate_node_rejected():
    async def a(s):
        return s

    g = EngineStateGraph()
    g.add_node("a", a)
    with pytest.raises(ValueError, match="Duplicate"):
        g.add_node("a", a)


def test_cannot_combine_edges_on_same_node():
    async def a(s):
        return s

    g = EngineStateGraph()
    g.add_node("a", a)
    g.add_edge("a", EngineStateGraph.END)
    with pytest.raises(ValueError, match="already has"):
        g.add_conditional_edge("a", lambda s: "x", {"x": EngineStateGraph.END})


@pytest.mark.asyncio
async def test_max_steps_loop_protection():
    async def loop(s):
        return s

    def go(s):
        return "again"

    g = EngineStateGraph()
    g.add_node("loop", loop)
    g.add_conditional_edge("loop", go, {"again": "loop"})
    g.set_entry("loop")
    runnable = g.compile(max_steps=10)
    with pytest.raises(RuntimeError, match="max_steps"):
        await runnable.invoke({})


@pytest.mark.asyncio
async def test_async_router_supported():
    async def root(s):
        return s

    async def end_node(s):
        s["hit"] = True
        return s

    async def async_router(s):
        return "ok"

    g = EngineStateGraph()
    g.add_node("root", root).add_node("end_node", end_node)
    g.add_conditional_edge("root", async_router, {"ok": "end_node"})
    g.add_edge("end_node", EngineStateGraph.END)
    g.set_entry("root")
    runnable = g.compile()
    final = await runnable.invoke({})
    assert final.get("hit") is True


def test_graph_phase_enum_values():
    assert GraphPhase.EXPAND.value == "expand"
    assert GraphPhase.DONE.value == "done"
