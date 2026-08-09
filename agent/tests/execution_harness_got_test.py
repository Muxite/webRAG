"""The benchmark harness must exercise the same post-step GoT loop as engine.run().

`engine.run()` performs pruning (default ON) and dead-end backtracking OUTSIDE
`step()`, in its own per-step loop. The benchmark driver `run_test_execution`
drives `step()` manually, so unless it mirrors that post-step loop, pruning and
backtracking stay inert for every benchmark graph run — even though `_got` is
assigned and `prune_enabled` defaults True.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.app.got_operations import GoTOperations
import agent.app.idea_engine as engine_mod
from agent.app.idea_engine import IdeaDagEngine
import agent.app.testing.execution as execution


class _FakeTestModule:
    metadata = {"test_id": "harness_got"}

    def get_task_statement(self):
        return "Trivial mandate with no substantiation requirement."


@pytest.mark.asyncio
async def test_harness_invokes_prune_each_interval(monkeypatch):
    prune_calls = []

    def _fake_identify(self, graph):
        prune_calls.append(True)
        return []

    async def _fake_step(self, graph, current_id, step_index):
        # Keep the loop alive so it reaches the prune interval; stay on root.
        return graph.root_id()

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "", "success": False}

    monkeypatch.setattr(GoTOperations, "identify_prune_candidates", _fake_identify)
    monkeypatch.setattr(IdeaDagEngine, "step", _fake_step)
    # The harness now delegates finalize to IdeaDagEngine.finalize, which calls the
    # build_final_payload bound in idea_engine (not the one re-exported by execution).
    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    def _fake_summarize(result, telemetry, model_name):
        return {}

    # prune_interval defaults to 5; run 5 steps so exactly one prune fires.
    result = await execution.run_test_execution(
        test_module=_FakeTestModule(),
        model_name="test-model",
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
        idea_settings={"max_steps": 5},
        run_stamp="teststamp",
        summarize_observability_func=_fake_summarize,
    )

    assert prune_calls, "harness never invoked identify_prune_candidates — prune loop is dead-wired"
