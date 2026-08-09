"""Tests for IdeaDagEngine.prepare() — the shared engine/graph setup helper.

`prepare()` is the single source of truth used by both `run()` and the
interactive debugger (debug_runner). It wires the memo namespace, memory manager
and GoT operations, and returns the starting graph either fresh or restored from
a checkpoint.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey


def _engine():
    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    return IdeaDagEngine(io=io, settings={}, model_name="m")


@pytest.mark.asyncio
async def test_prepare_fresh_graph():
    engine = _engine()
    mandate = "Do the thing"
    graph, current_id, steps = await engine.prepare(mandate)

    assert isinstance(graph, IdeaDag)
    assert current_id == graph.root_id()
    assert steps == 0
    # Per-run wiring happened.
    assert engine._memory_manager is not None
    assert engine._got is not None
    assert engine._current_mandate == mandate
    # Namespace stamped into settings + root details.
    ns = IdeaDagEngine._memo_namespace(mandate)
    assert engine.settings[DetailKey.MEMO_NAMESPACE.value] == ns
    root = graph.get_node(current_id)
    assert root.details.get("mandate") == mandate
    assert root.details.get("memo_namespace") == ns


@pytest.mark.asyncio
async def test_prepare_resets_per_run_state():
    engine = _engine()
    engine._candidate_coverage_extension_applied = True
    engine._step_confidences = [{"x": 1}]
    engine._leaf_completion_count = 5
    await engine.prepare("fresh mandate")
    assert engine._candidate_coverage_extension_applied is False
    assert engine._step_confidences == []
    assert engine._leaf_completion_count == 0


@pytest.mark.asyncio
async def test_prepare_restores_from_checkpoint():
    engine = _engine()
    mandate = "Resume this"
    saved = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
    saved_graph = saved.to_dict()

    class _FakeCheckpointer:
        async def load(self, run_id):
            return {
                "run_id": run_id,
                "step_index": 3,
                "snapshot": {
                    "graph": saved_graph,
                    "current_id": saved.root_id(),
                    "parallel_leaves_total": 2,
                    "got_dead_end_count": 6,
                },
            }

        async def save(self, *a, **k):
            return None

    engine._checkpointer = _FakeCheckpointer()
    graph, current_id, steps = await engine.prepare(mandate, run_id="r1")

    assert current_id == saved.root_id()
    assert steps == 4  # step_index + 1
    assert engine._step_index == 4
    assert engine._parallel_leaves_total == 2
    assert engine._got.dead_end_count == 6


@pytest.mark.asyncio
async def test_prepare_falls_back_to_fresh_on_checkpoint_load_error():
    engine = _engine()

    class _BoomCheckpointer:
        async def load(self, run_id):
            raise RuntimeError("corrupt")

        async def save(self, *a, **k):
            return None

    engine._checkpointer = _BoomCheckpointer()
    graph, current_id, steps = await engine.prepare("m", run_id="r1")
    assert steps == 0
    assert current_id == graph.root_id()


@pytest.mark.asyncio
async def test_prepare_fresh_when_no_run_id_even_with_checkpointer():
    engine = _engine()

    class _WouldLoad:
        async def load(self, run_id):  # should never be called without run_id
            raise AssertionError("load must not run without run_id")

        async def save(self, *a, **k):
            return None

    engine._checkpointer = _WouldLoad()
    graph, current_id, steps = await engine.prepare("m")
    assert steps == 0
