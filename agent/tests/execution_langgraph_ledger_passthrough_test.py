"""The `run_offtheshelf_execution` passthrough for ledger finish-hook keys.

`LangGraphSolver.solve` writes `evidence_graph` / `answer_audit` / `shape_derive` /
`host_derive` at its exit, and `execution_langgraph.run_offtheshelf_execution` copies each into
`output` by an explicit allowlist. That allowlist has now been missed three times (mint01 smoke:
`answer_audit`; mint03 smoke: `host_derive`), each time with the solver's own tests green, because
they stop at the solver. This test drives the wrapper with a stub solver so the seam itself is
pinned: every key the solver emits must reach `output`, and none may appear when the solver
emitted nothing.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from app.testing import execution_langgraph


LEDGER_KEYS = ("evidence_graph", "answer_audit", "shape_derive", "host_derive", "host_prefetch")


class _StubModule:
    metadata = {"test_id": "999"}

    def get_task_statement(self) -> str:
        return "Compute the difference between A and B."


def _run_with_solver_result(monkeypatch, solver_result: Dict[str, Any]) -> Dict[str, Any]:
    class _StubSolver:
        def __init__(self, *args, **kwargs):
            pass

        async def solve(self, mandate, *, max_steps, settings, telemetry):
            return dict(solver_result)

    monkeypatch.setattr(execution_langgraph, "LangGraphSolver", _StubSolver)
    monkeypatch.setattr(execution_langgraph, "traces_retained", lambda: False)
    result = asyncio.run(
        execution_langgraph.run_offtheshelf_execution(
            _StubModule(),
            "stub-model",
            connector_llm=None,
            connector_search=None,
            connector_http=None,
            connector_chroma=None,
            run_stamp="passthrough_test",
            summarize_observability_func=lambda *_a, **_k: {},
        )
    )
    return result["output"]


def test_every_ledger_key_the_solver_emits_reaches_output(monkeypatch):
    payload = {key: {"marker": key} for key in LEDGER_KEYS}
    output = _run_with_solver_result(
        monkeypatch, {"final_deliverable": "x", "success": True, **payload})
    for key in LEDGER_KEYS:
        assert output.get(key) == {"marker": key}, key


@pytest.mark.parametrize("key", LEDGER_KEYS)
def test_a_single_ledger_key_passes_through_alone(monkeypatch, key):
    output = _run_with_solver_result(
        monkeypatch, {"final_deliverable": "x", "success": True, key: {"reason": "computed"}})
    assert output[key] == {"reason": "computed"}
    for other in LEDGER_KEYS:
        if other != key:
            assert other not in output, other


def test_no_ledger_keys_when_the_solver_emitted_none(monkeypatch):
    output = _run_with_solver_result(monkeypatch, {"final_deliverable": "x", "success": True})
    for key in LEDGER_KEYS:
        assert key not in output, key
