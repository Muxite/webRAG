"""Offline tests for the ``sequential_react_extract`` arm (testing/execution_sequential_extract.py).

The arm exists for FAIRNESS: ``evidence_loop`` spends an extra LLM call per visit on typed
extraction, so comparing it against ``sequential_react`` measures that extra compute rather than
the ledger. This arm is the sequential control's loop PLUS exactly that extraction call, so
``sequential_react`` vs this isolates the cost of extraction and this vs ``evidence_loop``
isolates the ledger and its gating.

These tests pin both halves of that contract: the loop still behaves like the control when no
visit succeeds (no extraction fires), a successful visit produces ``Extraction`` records and a
stored page, and NO ledger/verdict field ever reaches the output.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_sequential as seq
from agent.app.testing import execution_sequential_extract as sxe


def _io(replies, page_text="PAGE CONTENT with Toni Morrison in it", synth="SYNTH ANSWER"):
    """A mocked ``AgentIO``; ``replies`` are the LLM replies IN CALL ORDER (step decisions and
    per-hop extraction replies interleaved exactly as the loop consumes them)."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=lambda **kw: {"messages": kw.get("messages", [])})
    encoded = [r if isinstance(r, str) else json.dumps(r) for r in replies]
    io.query_llm = AsyncMock(side_effect=[*encoded, synth, synth, synth])
    io.search = AsyncMock(return_value=[
        {"title": "Toni Morrison", "url": "https://en.wikipedia.org/wiki/Toni_Morrison",
         "description": "novelist"}])
    io.visit = AsyncMock(return_value=page_text)
    return io


def _extraction(entity, field, value, quote, verdict="SUPPORTED"):
    return json.dumps({"extractions": [
        {"entity": entity, "field": field, "value": value, "verdict": verdict, "quote": quote}]})


TASK = "Who wrote Beloved?"


# --------------------------------------------------------------------------------------
# registration — an unknown name silently falls back to ["graph"], so pin the real one
# --------------------------------------------------------------------------------------


def test_the_variant_name_parses_to_itself():
    from agent.app.idea_test_runner import _parse_execution_variants

    assert _parse_execution_variants("sequential_react_extract") == ["sequential_react_extract"]


def test_a_misspelled_variant_still_falls_back_to_graph():
    # The trap this arm has to avoid: the parser is silent about unknown names.
    from agent.app.idea_test_runner import _parse_execution_variants

    assert _parse_execution_variants("sequential_react_extractt") == ["graph"]


def test_the_variant_is_registered_in_the_dispatch():
    from agent.app.testing import runner as harness_runner

    assert harness_runner.EXTRACTING_LINEAR_VARIANTS == ("sequential_react_extract",)
    assert "sequential_react_extract" in harness_runner.KNOWN_EXECUTION_VARIANTS


# --------------------------------------------------------------------------------------
# the loop is the control's loop
# --------------------------------------------------------------------------------------


def test_no_successful_visit_means_no_extraction_call_and_control_behavior():
    decisions = [
        {"thought": "find author", "action": "search", "args": {"query": "Beloved author"}},
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison"}},
    ]
    control = _io(list(decisions))
    arm = _io(list(decisions))
    expected = asyncio.run(seq._run_react(control, TASK, "m", max_steps=6, max_tokens=512))
    result = asyncio.run(sxe._run_react_extract(arm, TASK, "m", max_steps=6, max_tokens=512))

    assert result.deliverable == expected
    assert arm.query_llm.await_count == control.query_llm.await_count  # no extra call
    assert arm.search.await_count == control.search.await_count
    assert result.extractions == [] and result.pages == []


def test_a_failed_visit_does_not_extract():
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com"}},
        {"thought": "answer", "action": "finish", "args": {"answer": "done"}},
    ]
    control = _io(list(decisions), page_text="")
    arm = _io(list(decisions), page_text="")
    expected = asyncio.run(seq._run_react(control, TASK, "m", max_steps=6, max_tokens=512))
    result = asyncio.run(sxe._run_react_extract(arm, TASK, "m", max_steps=6, max_tokens=512))

    assert result.deliverable == expected
    assert arm.query_llm.await_count == control.query_llm.await_count
    assert result.extractions == [] and result.pages == []


def test_fenced_decision_is_recovered():
    # A model that wraps its decision in a ```json fence must still be parsed via
    # prompted_tools.extract_decision, not dropped as invalid JSON (matches the control's fix).
    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=lambda **kw: {"messages": kw.get("messages", [])})
    fenced = '```json\n{"thought": "fenced", "action": "finish", "args": {"answer": "FENCED ANSWER"}}\n```'
    io.query_llm = AsyncMock(side_effect=[fenced])
    io.search = AsyncMock(return_value=[])
    io.visit = AsyncMock(return_value="")
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=6, max_tokens=512))
    assert result.deliverable == "FENCED ANSWER"
    io.search.assert_not_awaited()
    io.visit.assert_not_awaited()


def test_forced_synthesis_when_the_model_never_finishes():
    decisions = [{"thought": "search", "action": "search", "args": {"query": "q"}}]
    io = _io(decisions, synth="FORCED SYNTHESIS")
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=1, max_tokens=512))
    assert result.deliverable == "FORCED SYNTHESIS"


def test_a_malformed_decision_does_not_crash():
    io = _io(["not json at all"], synth="FINAL")
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=1, max_tokens=256))
    assert result.deliverable == "FINAL"


# --------------------------------------------------------------------------------------
# the one added thing: per-hop typed extraction
# --------------------------------------------------------------------------------------


def test_a_successful_visit_extracts_records_and_stores_the_page():
    decisions = [
        {"thought": "read", "action": "visit",
         "args": {"url": "https://en.wikipedia.org/wiki/Toni_Morrison"}},
        _extraction("Beloved", "author", "Toni Morrison", "Toni Morrison"),
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison"}},
    ]
    io = _io(decisions)
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=6, max_tokens=512))

    assert result.deliverable == "Toni Morrison"
    assert len(result.extractions) == 1
    record = result.extractions[0]
    assert record.value == "Toni Morrison"
    assert record.source_url == "https://en.wikipedia.org/wiki/Toni_Morrison"
    assert record.quote_verified is True          # the quote is literally on the page
    assert record.page_id == "p1"
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page["page_id"] == "p1" and page["url"] == record.source_url
    assert page["content_hash"] and page["text"].startswith("PAGE CONTENT")


def test_extraction_costs_exactly_one_extra_llm_call_per_successful_visit():
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://a.example/x"}},
        _extraction("Beloved", "author", "Toni Morrison", "Toni Morrison"),
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison"}},
    ]
    control = _io([d for d in decisions if not isinstance(d, str)])
    arm = _io(list(decisions))
    asyncio.run(seq._run_react(control, TASK, "m", max_steps=6, max_tokens=512))
    asyncio.run(sxe._run_react_extract(arm, TASK, "m", max_steps=6, max_tokens=512))
    assert arm.query_llm.await_count == control.query_llm.await_count + 1


def test_malformed_extraction_json_does_not_crash_the_cell():
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://a.example/x"}},
        "}{ not json",
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison"}},
    ]
    io = _io(decisions)
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=6, max_tokens=512))
    assert result.deliverable == "Toni Morrison"
    assert result.extractions == []
    assert len(result.pages) == 1     # the page is still frozen for offline re-audit


def test_a_failing_extraction_call_does_not_end_the_run():
    io = _io([{"thought": "read", "action": "visit", "args": {"url": "https://a.example/x"}}])
    io.query_llm = AsyncMock(side_effect=[
        json.dumps({"action": "visit", "args": {"url": "https://a.example/x"}}),
        RuntimeError("extraction backend down"),
        json.dumps({"action": "finish", "args": {"answer": "Toni Morrison"}}),
    ])
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=6, max_tokens=512))
    assert result.deliverable == "Toni Morrison"
    assert result.extractions == []


# --------------------------------------------------------------------------------------
# the deliverable is produced the way the control produces it — no ledger, no gating
# --------------------------------------------------------------------------------------


def test_the_deliverable_is_undecorated():
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://a.example/x"}},
        _extraction("Beloved", "author", "Toni Morrison", "Toni Morrison"),
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison"}},
    ]
    io = _io(decisions)
    result = asyncio.run(sxe._run_react_extract(io, TASK, "m", max_steps=6, max_tokens=512))
    assert result.deliverable == "Toni Morrison"
    assert "EVIDENCE TABLE" not in result.deliverable
    assert "VERDICT" not in result.deliverable


@pytest.mark.asyncio
async def test_the_variant_runs_end_to_end_and_returns_the_standard_shape(monkeypatch):
    async def _fake_loop(agent_io, mandate, model_name, max_steps, max_tokens, **kwargs):
        return sxe.SequentialExtractResult(deliverable="STUB ANSWER", extractions=[], pages=[])

    monkeypatch.setattr(sxe, "_run_react_extract", _fake_loop)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = TASK

    result = await sxe.run_sequential_extract_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {"visit": {"count": 0}},
    )

    assert set(result) >= {"output", "graph", "observability", "duration_seconds", "telemetry"}
    output = result["output"]
    assert output["final_deliverable"] == "STUB ANSWER"
    assert output["success"] is True
    assert output["action_summary"] == "sequential_react_extract"
    assert result["telemetry"]["correlation_id"].endswith("sequential_react_extract_r1")


@pytest.mark.asyncio
async def test_no_ledger_or_verdict_field_reaches_the_output(monkeypatch):
    record = sxe.Extraction(entity="Beloved", field="author", value="Toni Morrison",
                            verdict="SUPPORTED", source_url="https://a.example/x",
                            quote="Toni Morrison", quote_verified=True, page_id="p1")

    async def _fake_loop(agent_io, mandate, model_name, max_steps, max_tokens, **kwargs):
        return sxe.SequentialExtractResult(
            deliverable="ANSWER", extractions=[record],
            pages=[sxe.store_page("p1", "https://a.example/x", "text", 100)])

    monkeypatch.setattr(sxe, "_run_react_extract", _fake_loop)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = TASK

    result = await sxe.run_sequential_extract_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    forbidden = ("ledger", "ledger_verdict", "ledger_status_counts", "ledger_resolution_counts",
                 "rows_resolved", "rows_quote_backed", "rows_unresolved",
                 "unverified_provenance", "quote_verified_count", "quote_unverified_count",
                 "quote_unchecked_count", "quote_fail_reasons")
    assert not [key for key in forbidden if key in output]
    # The gathered evidence IS reported (that is the point of the arm) — just not as a ledger.
    assert output["extractions"] == [record.as_dict()]
    assert output["pages"][0]["page_id"] == "p1"


@pytest.mark.asyncio
async def test_a_crashing_loop_still_returns_a_well_formed_failed_result(monkeypatch):
    async def _boom(agent_io, mandate, model_name, max_steps, max_tokens, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(sxe, "_run_react_extract", _boom)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = TASK

    result = await sxe.run_sequential_extract_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    assert result["output"]["final_deliverable"] == ""
    assert result["output"]["success"] is False


# --------------------------------------------------------------------------------------
# budget: "better" must never be silently "spent more"
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cell_reports_its_token_and_wall_clock_budget(monkeypatch):
    async def _fake_loop(agent_io, mandate, model_name, max_steps, max_tokens, **kwargs):
        return sxe.SequentialExtractResult(deliverable="ANSWER", extractions=[], pages=[])

    monkeypatch.setattr(sxe, "_run_react_extract", _fake_loop)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = TASK

    def _observability(result, telemetry, model_name):
        from agent.app.testing.utils import summarize_observability

        return summarize_observability(result, telemetry, model_name)

    result = await sxe.run_sequential_extract_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=_observability,
    )
    assert isinstance(result["observability"]["llm"]["total_tokens"], int)
    assert isinstance(result["duration_seconds"], float)
    assert result["output"]["total_tokens"] == result["observability"]["llm"]["total_tokens"]
    assert result["output"]["duration_seconds"] == result["duration_seconds"]
