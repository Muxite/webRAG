"""The agent-debug stepper applies the same post-step control-loop passes a real run does.

`DebugSession` drives `engine.step()` itself instead of running `IdeaDagEngine._run_loop`, so
the loop's prune / backtrack / early-exit passes used to be unreachable while stepping. The
prune pass is on by DEFAULT, so a stepped graph could diverge from the graph the benchmark
produces for the same mandate, which makes the debugger unreliable for reproducing a run.

These drive the real (unstubbed) `GoTOperations` gates through the extracted
`IdeaDagEngine.maybe_prune` / `maybe_backtrack` / `maybe_early_exit` helpers, from inside a
`DebugSession`. Flag-off control cases pin the unchanged behavior.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.interactive.controller import Controller
from agent.app.interactive.session import DebugSession


def _make_engine(settings):
    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings=dict(settings), model_name="m")
    # Normally wired by `prepare()`; the stepper is handed an already-prepared engine.
    engine._got = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    return engine


def _session(engine, graph, max_steps=10):
    sess = DebugSession(engine=engine, graph=graph, ctrl=Controller(), max_steps=max_steps)
    sess._out = lambda *_a, **_k: None
    return sess


def _scored_graph():
    """root -> weak(0.1, pending), strong(0.9, pending): one prune candidate."""
    graph = IdeaDag(root_title="root")
    weak = graph.add_child(graph.root_id(), "weak", details={})
    graph.evaluate(weak.node_id, 0.1)
    strong = graph.add_child(graph.root_id(), "strong", details={})
    graph.evaluate(strong.node_id, 0.9)
    return graph, weak.node_id, strong.node_id


_PRUNE_SETTINGS = {
    "got_prune_interval_steps": 1,
    "got_prune_min_nodes_before_prune": 2,
    "got_prune_score_threshold": 0.5,
    "got_adaptive_policies": False,  # fixed threshold, so the test is deterministic
}


def _install_step(monkeypatch, calls: List[str], returns=None):
    """Replace `step()` with a recorder; `returns` is a fixed next-id (None to stop)."""

    async def _step(self, graph, current_id, step_index):
        calls.append(current_id)
        return returns

    monkeypatch.setattr(IdeaDagEngine, "step", _step)


# ------------------------------------------------------------------------------- prune


@pytest.mark.asyncio
async def test_debug_session_prunes_low_score_nodes(monkeypatch):
    graph, weak_id, strong_id = _scored_graph()
    _install_step(monkeypatch, [])
    sess = _session(_make_engine(_PRUNE_SETTINGS), graph)

    await sess._engine_step(graph.root_id())

    assert graph.get_node(weak_id).status is IdeaNodeStatus.SKIPPED
    assert graph.get_node(weak_id).details.get("_got_pruned") is True
    assert graph.get_node(strong_id).status is not IdeaNodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_debug_session_prune_respects_the_step_interval(monkeypatch):
    """Off-interval steps must not prune — the stepper uses the loop's own cadence."""
    graph, weak_id, _ = _scored_graph()
    _install_step(monkeypatch, [])
    settings = {**_PRUNE_SETTINGS, "got_prune_interval_steps": 5}
    sess = _session(_make_engine(settings), graph)

    await sess._engine_step(graph.root_id())
    assert graph.get_node(weak_id).status is not IdeaNodeStatus.SKIPPED

    for _ in range(4):
        await sess._engine_step(graph.root_id())
    assert sess._step == 5
    assert graph.get_node(weak_id).status is IdeaNodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_debug_session_prune_disabled_leaves_graph_untouched(monkeypatch):
    graph, weak_id, _ = _scored_graph()
    _install_step(monkeypatch, [])
    settings = {**_PRUNE_SETTINGS, "got_prune_enabled": False}
    sess = _session(_make_engine(settings), graph)

    await sess._engine_step(graph.root_id())
    assert graph.get_node(weak_id).status is not IdeaNodeStatus.SKIPPED
    assert "_got_pruned" not in graph.get_node(weak_id).details


@pytest.mark.asyncio
async def test_autorun_prunes_too(monkeypatch):
    """The auto-run path drives `step()` on its own; it gets the same passes."""
    graph, weak_id, _ = _scored_graph()
    _install_step(monkeypatch, [])
    sess = _session(_make_engine(_PRUNE_SETTINGS), graph)

    await sess._autorun(graph.root_id(), depth=0)

    assert graph.get_node(weak_id).status is IdeaNodeStatus.SKIPPED


# --------------------------------------------------------------------------- backtrack


def _dead_end_graph():
    """root -> a(0.1) -> b(0.1) -> c(0.1): three consecutive low-score nodes."""
    graph = IdeaDag(root_title="root")
    a = graph.add_child(graph.root_id(), "a", details={})
    graph.evaluate(a.node_id, 0.1)
    b = graph.add_child(a.node_id, "b", details={})
    graph.evaluate(b.node_id, 0.1)
    c = graph.add_child(b.node_id, "c", details={})
    graph.evaluate(c.node_id, 0.1)
    return graph, c.node_id


_BACKTRACK_SETTINGS = {
    "got_backtrack_enabled": True,
    "got_backtrack_dead_end_threshold": 2,
    "got_backtrack_low_score_threshold": 0.3,
    "got_prune_enabled": False,  # isolate the backtrack pass
}


@pytest.mark.asyncio
async def test_autorun_follows_the_backtrack_redirect(monkeypatch):
    graph, dead_id = _dead_end_graph()
    calls: List[str] = []
    _install_step(monkeypatch, calls, returns=dead_id)
    sess = _session(_make_engine(_BACKTRACK_SETTINGS), graph, max_steps=2)

    await sess._autorun(dead_id, depth=0)

    # step() keeps re-selecting the dead end; the second step must run from the
    # backtrack target (the root) rather than from the dead end again.
    assert calls == [dead_id, graph.root_id()]


@pytest.mark.asyncio
async def test_autorun_without_backtrack_stays_on_the_dead_end(monkeypatch):
    """Flag off (the default) -> step()'s return value is never redirected."""
    graph, dead_id = _dead_end_graph()
    calls: List[str] = []
    _install_step(monkeypatch, calls, returns=dead_id)
    settings = {"got_prune_enabled": False}
    sess = _session(_make_engine(settings), graph, max_steps=2)

    await sess._autorun(dead_id, depth=0)

    # `_autorun` stops as soon as step() hands back the node it started on (childless,
    # so trivially "all children terminal"). No redirect ever happens.
    assert calls == [dead_id]


# -------------------------------------------------------------------------- early exit


@pytest.mark.asyncio
async def test_early_exit_halts_the_session_without_marking_it_quit(monkeypatch):
    graph, _weak, _strong = _scored_graph()
    _install_step(monkeypatch, [])
    engine = _make_engine({"native_confidence_early_exit_enabled": True, "got_prune_enabled": False})
    engine._got.should_exit_early = lambda *_a, **_k: True
    sess = _session(engine, graph)

    await sess._engine_step(graph.root_id())

    assert sess._early_exit is True
    assert sess._halt() is True
    result = sess._finish()
    # `quit_early` gates debug_runner's final-answer print: an early exit means
    # "finalize with what we have", so it must stay False.
    assert result["early_exit"] is True
    assert result["quit_early"] is False


@pytest.mark.asyncio
async def test_early_exit_disabled_does_not_halt(monkeypatch):
    graph, _weak, _strong = _scored_graph()
    _install_step(monkeypatch, [])
    engine = _make_engine({"got_prune_enabled": False})  # early exit defaults off
    engine._got.should_exit_early = lambda *_a, **_k: True
    sess = _session(engine, graph)

    await sess._engine_step(graph.root_id())

    assert sess._early_exit is False
    assert sess._halt() is False


# ----------------------------------------------------------------------------- fakes


@pytest.mark.asyncio
async def test_stub_engine_without_the_helpers_is_tolerated():
    """Test/stub engines expose only `step()`; the stepper must not require the passes."""

    class _StubEngine:
        async def step(self, graph, node_id, step_index):
            return "next-id"

    graph, _weak, _strong = _scored_graph()
    sess = _session(_StubEngine(), graph)

    assert await sess._engine_step(graph.root_id()) == "next-id"


@pytest.mark.asyncio
async def test_a_failing_control_loop_pass_does_not_kill_the_session():
    class _BadEngine:
        async def step(self, graph, node_id, step_index):
            return "next-id"

        def maybe_prune(self, graph, steps):
            raise RuntimeError("boom")

    graph, _weak, _strong = _scored_graph()
    sess = _session(_BadEngine(), graph)

    assert await sess._engine_step(graph.root_id()) == "next-id"


# ------------------------------------------------------------- extracted helpers (unit)


def test_maybe_backtrack_returns_current_id_unchanged_when_disabled():
    graph, dead_id = _dead_end_graph()
    engine = _make_engine({})
    assert engine.maybe_backtrack(graph, dead_id, steps=1) == dead_id


def test_maybe_backtrack_returns_current_id_when_there_is_no_got():
    graph, dead_id = _dead_end_graph()
    engine = _make_engine(_BACKTRACK_SETTINGS)
    engine._got = None
    assert engine.maybe_backtrack(graph, dead_id, steps=1) == dead_id


def test_maybe_prune_is_a_no_op_without_got():
    graph, weak_id, _ = _scored_graph()
    engine = _make_engine(_PRUNE_SETTINGS)
    engine._got = None
    engine.maybe_prune(graph, steps=1)
    assert graph.get_node(weak_id).status is not IdeaNodeStatus.SKIPPED


def test_maybe_early_exit_is_false_when_disabled():
    graph, _weak, _strong = _scored_graph()
    engine = _make_engine({})
    engine._got.should_exit_early = lambda *_a, **_k: True
    assert engine.maybe_early_exit(graph, graph.root_id(), steps=1) is False
