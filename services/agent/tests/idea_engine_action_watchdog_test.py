"""Regression tests for the per-action timeout watchdog (`_execute_action_guarded`).

The per-type action cap (`_action_timeout_for`, default 20s) used to be enforced
ONLY on the auto-parallel gather path; leaf / merge / single-best-child selection
called `_execute_action` bare, so on the default engine variant a slow action could
wedge the whole cell with no run-level watchdog to catch it. `_execute_action_guarded`
now wraps every non-auto-parallel path. These pin that a hung action fails open
(node FAILED, returns None) and a fast action passes through untouched.
"""
import asyncio

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


class _StubIO:
    telemetry = None
    connector_chroma = None

    def set_telemetry(self, telemetry):
        return None


def _make_engine():
    return IdeaDagEngine(
        io=_StubIO(),
        settings={"allow_unscored_selection": True, "min_score_threshold": 0.0},
    )


@pytest.mark.asyncio
async def test_guarded_action_timeout_marks_failed_and_returns_none():
    """A slow action is guillotined at the per-action timeout, not left to hang."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(), "Slow visit",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    engine._action_timeout_for = lambda name: 0.05

    async def _slow(*_a, **_k):
        await asyncio.sleep(5)
        return {"success": True}

    engine._execute_action = _slow

    # Outer wait_for asserts the watchdog itself returns promptly (< the 5s sleep).
    result = await asyncio.wait_for(
        engine._execute_action_guarded(graph, graph.root_id(), node.node_id),
        timeout=2.0,
    )
    assert result is None
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.FAILED
    assert "timeout" in (refreshed.details.get("action_error") or "")


@pytest.mark.asyncio
async def test_guarded_action_passes_through_fast_result():
    """A fast action's result is returned unchanged; no status is forced."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(), "Fast think",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
    )
    sentinel = {"success": True, "action": "think"}

    async def _fast(*_a, **_k):
        return sentinel

    engine._execute_action = _fast
    engine._action_timeout_for = lambda name: 5.0

    result = await engine._execute_action_guarded(graph, graph.root_id(), node.node_id)
    assert result is sentinel
    assert graph.get_node(node.node_id).status != IdeaNodeStatus.FAILED
