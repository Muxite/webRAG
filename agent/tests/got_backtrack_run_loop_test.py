"""End-to-end check that `got_backtrack_enabled` actually fires inside the real
run loop (`IdeaDagEngine._run_loop`), not just in the isolated
`should_backtrack`/`find_backtrack_target` unit tests (`got_backtrack_test.py`).

Drives the real `GoTOperations.should_backtrack` / `find_backtrack_target`
(no stubbing) against a scripted `step()` that always hands control back to a
deep, low-scoring dead-end node. With the flag ON the loop must redirect
`current_id` away from that dead end each iteration; with it OFF (default)
the loop must stay byte-identical (no redirect).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import agent.app.idea_engine as engine_mod
from agent.app.idea_engine import IdeaDagEngine


def _build_dead_end_graph():
    """root -> a(0.1) -> b(0.1) -> c(0.1): three consecutive low-score nodes."""
    from agent.app.idea_dag import IdeaDag

    graph = IdeaDag(root_title="root")
    a = graph.add_child(graph.root_id(), "a", details={})
    graph.evaluate(a.node_id, 0.1)
    b = graph.add_child(a.node_id, "b", details={})
    graph.evaluate(b.node_id, 0.1)
    c = graph.add_child(b.node_id, "c", details={})
    graph.evaluate(c.node_id, 0.1)
    return graph, c.node_id


async def _fixed_payload(*args, **kwargs):
    return {"final_deliverable": "answer", "success": True}


def _install(monkeypatch, deep_node_id):
    monkeypatch.setattr(engine_mod, "build_final_payload", _fixed_payload)
    monkeypatch.setattr(engine_mod, "default_post_expansion_hooks", lambda: [])

    async def _stay_at_dead_end(self, graph, current_id, step_index):
        # Whatever current_id we were redirected to, step() always drives back
        # down to the same dead-end leaf (simulates a model stuck re-selecting
        # the same low-scoring branch).
        return deep_node_id

    monkeypatch.setattr(IdeaDagEngine, "step", _stay_at_dead_end)


def _make_engine(settings):
    from agent.app.got_operations import GoTOperations

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings=dict(settings), model_name="m")
    # `_run_loop` is normally reached via `run()` -> `prepare()`, which wires
    # `self._got`. Calling `_run_loop` directly (as the parity tests do) needs
    # it wired manually so the real (unstubbed) `should_backtrack` /
    # `find_backtrack_target` are actually consulted.
    engine._got = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    return engine


@pytest.mark.asyncio
async def test_backtrack_redirects_current_id_in_real_loop(monkeypatch):
    graph, deep_id = _build_dead_end_graph()
    _install(monkeypatch, deep_id)
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 2,
        "got_backtrack_low_score_threshold": 0.3,
    }
    engine = _make_engine(settings)

    _, current_id, steps = await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=1, steps=0,
    )

    # should_backtrack fires (2 consecutive low-score ancestors of "c": b, then a);
    # find_backtrack_target walks up looking for a decent-scoring ancestor (none
    # exist besides the unscored root) and falls back to the root.
    assert current_id == graph.root_id(), (
        "backtrack must redirect current_id away from the dead-end leaf"
    )
    assert engine._got.dead_end_count == 1


@pytest.mark.asyncio
async def test_backtrack_disabled_leaves_current_id_untouched(monkeypatch):
    """Regression: flag OFF (default) -> no redirect, byte-identical to pre-backtrack behavior."""
    graph, deep_id = _build_dead_end_graph()
    _install(monkeypatch, deep_id)
    settings = {}  # got_backtrack_enabled defaults False
    engine = _make_engine(settings)

    _, current_id, steps = await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=1, steps=0,
    )

    assert current_id == deep_id, "flag off -> step()'s return value is never redirected"
    assert engine._got.dead_end_count == 0


@pytest.mark.asyncio
async def test_backtrack_keeps_redirecting_every_step_while_stuck(monkeypatch):
    """Over several steps, a model that keeps re-selecting the same dead end is
    redirected every time (bounded by max_steps; never gets stuck oscillating
    forever because the outer loop is itself step-budget-bounded)."""
    graph, deep_id = _build_dead_end_graph()
    _install(monkeypatch, deep_id)
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 2,
        "got_backtrack_low_score_threshold": 0.3,
    }
    engine = _make_engine(settings)

    _, current_id, steps = await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=5, steps=0,
    )

    assert steps == 5, "the loop terminates on the step budget, never hangs"
    assert current_id == graph.root_id()
    assert engine._got.dead_end_count == 5
