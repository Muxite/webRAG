"""Regression tests for engine checkpoint resume.

The engine saves `got_dead_end_count` in every checkpoint snapshot and, on
resume, restores it via `self._got.dead_end_count = ...`. That write is only
valid if `GoTOperations.dead_end_count` exposes a setter. Before the fix the
property was read-only, so the write raised AttributeError inside the broad
`except` that guards checkpoint restore — silently discarding the ENTIRE
restore (graph/current_id/steps reset) and starting the run fresh on every
resume.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine


def test_dead_end_count_is_settable():
    got = GoTOperations(settings={}, io=MagicMock(), memory_manager=None)
    got.dead_end_count = 7
    assert got.dead_end_count == 7


@pytest.mark.asyncio
async def test_run_resumes_from_checkpoint_with_dead_end_count(monkeypatch):
    """A checkpoint carrying got_dead_end_count must be fully restored.

    Drives the real `run()` restore branch: the loaded snapshot must survive
    (steps advanced past max_steps so the step loop is skipped) and the dead-end
    counter must be re-applied onto the fresh GoTOperations instance.
    """
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "x", "success": True}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings={}, model_name="m")

    mandate = "Resume me"
    graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
    # Give the graph a distinctive extra node so we can prove it was restored
    # rather than rebuilt from scratch.
    snapshot_graph = graph.to_dict()

    class _FakeCheckpointer:
        async def load(self, run_id):
            return {
                "run_id": run_id,
                "step_index": 4,
                "snapshot": {
                    "graph": snapshot_graph,
                    "current_id": graph.root_id(),
                    "parallel_leaves_total": 3,
                    "got_dead_end_count": 9,
                },
            }

        async def save(self, *args, **kwargs):
            return None

    engine._checkpointer = _FakeCheckpointer()

    # max_steps <= restored steps (4 + 1 = 5) so the step loop is skipped and we
    # exercise only the restore + finalize path.
    result = await engine.run(mandate, max_steps=5, run_id="run-xyz")

    # Restore succeeded: dead-end counter re-applied (would be 0 if restore aborted).
    assert engine._got.dead_end_count == 9
    assert getattr(engine, "_parallel_leaves_total", None) == 3
    assert result.get("got_stats", {}).get("dead_ends_detected") == 9
