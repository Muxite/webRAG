"""Offline smoke tests for the deterministic evidence-queue STUB (Phase 0 harness plumbing).

``testing/execution_evidence_queue.py`` is not the DAG v3 architecture -- it is one
search/visit/extract cycle per enumerated requirement, with no scheduler and no typed state (see
that module's docstring). These tests exist to prove exactly what the stub is for: it dispatches,
it runs end to end against a fixture, and it produces a well-formed result object with the same
shape every other variant returns. Answer quality is explicitly NOT asserted.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_evidence_queue as eq


def _agent_io(page_text: str = "PAGE CONTENT", answer: str = "FINAL ANSWER"):
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={"messages": []})
    io.query_llm = AsyncMock(return_value=answer)
    io.search = AsyncMock(return_value=[{"title": "t", "url": "https://example.com/a"}])
    io.visit = AsyncMock(return_value=page_text)
    return io


# --------------------------------------------------------------------------------------
# requirement enumeration
# --------------------------------------------------------------------------------------


def test_requirements_come_from_the_shared_enumerator():
    mandate = "Compare the following:\n1. Erie Canal\n2. Suez Canal\n3. Panama Canal"
    reqs = eq._requirements(mandate)
    assert len(reqs) == 3
    assert any("Erie" in r for r in reqs)


def test_an_unenumerated_mandate_still_yields_one_requirement():
    reqs = eq._requirements("Who wrote Beloved?")
    assert len(reqs) == 1


def test_an_empty_mandate_yields_no_requirements():
    assert eq._requirements("") == []
    assert eq._requirements("   ") == []


def test_the_roster_is_capped():
    mandate = "Compare:\n" + "\n".join(f"{i}. Candidate {i}" for i in range(1, 20))
    assert len(eq._requirements(mandate)) <= eq._MAX_REQUIREMENTS


# --------------------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------------------


def test_one_search_and_one_visit_per_requirement():
    io = _agent_io()
    mandate = "Compare the following:\n1. Erie Canal\n2. Suez Canal"
    out = asyncio.run(eq._run_evidence_queue(io, mandate, "m", max_tokens=512))
    assert out == "FINAL ANSWER"
    assert io.search.await_count == 2
    assert io.visit.await_count == 2
    # two EXTRACT calls plus exactly one synthesis
    assert io.query_llm.await_count == 3


def test_a_failing_requirement_does_not_end_the_run():
    io = _agent_io()
    io.search = AsyncMock(side_effect=[RuntimeError("search down"),
                                       [{"url": "https://example.com/b"}]])
    mandate = "Compare the following:\n1. Erie Canal\n2. Suez Canal"
    out = asyncio.run(eq._run_evidence_queue(io, mandate, "m", max_tokens=512))
    assert out == "FINAL ANSWER"
    assert io.visit.await_count == 1        # only the surviving requirement got visited


def test_a_resultless_search_is_skipped_without_a_visit():
    io = _agent_io()
    io.search = AsyncMock(return_value=[])
    out = asyncio.run(eq._run_evidence_queue(io, "Who wrote Beloved?", "m", max_tokens=512))
    assert out == "FINAL ANSWER"            # still synthesizes, honestly, from nothing
    assert io.visit.await_count == 0
    assert io.query_llm.await_count == 1


def test_an_empty_page_contributes_no_extract():
    io = _agent_io(page_text="   ")
    asyncio.run(eq._run_evidence_queue(io, "Who wrote Beloved?", "m", max_tokens=512))
    assert io.query_llm.await_count == 1    # synthesis only


# --------------------------------------------------------------------------------------
# end-to-end result shape
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_variant_runs_end_to_end_and_returns_the_standard_shape(monkeypatch):
    async def _fake_loop(agent_io, mandate, model_name, max_tokens):
        return "STUB ANSWER"

    monkeypatch.setattr(eq, "_run_evidence_queue", _fake_loop)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."

    result = await eq.run_evidence_queue_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )

    assert set(result) >= {"output", "graph", "observability", "duration_seconds", "telemetry"}
    assert result["output"]["final_deliverable"] == "STUB ANSWER"
    assert result["output"]["success"] is True
    assert result["output"]["action_summary"] == "evidence_queue_deterministic"
    assert result["telemetry"]["correlation_id"].endswith("evidence_queue_deterministic_r1")


@pytest.mark.asyncio
async def test_a_crashing_loop_still_returns_a_well_formed_failed_result(monkeypatch):
    """Same failure contract as the sibling variants: report, never raise into the harness."""
    async def _boom(agent_io, mandate, model_name, max_tokens):
        raise RuntimeError("nope")

    monkeypatch.setattr(eq, "_run_evidence_queue", _boom)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."

    result = await eq.run_evidence_queue_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )
    assert result["output"]["final_deliverable"] == ""
    assert result["output"]["success"] is False
