"""Tests for the sequential_react structural finish gate (W2 §1-2, opt-in, default OFF).

Gate: refuse a `finish(answer)` whose committed answer carries a number the evidence ledger
cannot back, up to 2 corrective retries, then abstain with a banner. Validated offline against
docs/analysis/GATE_PRECISION_PRECHECK_2026-09-04.md's four refinements:
  1. check only the committed finish-answer argument, never the whole transcript;
  2. zero extractable numbers in the answer is itself an automatic refuse;
  3. never float()-cast node.value directly (compound strings like "1,642 metres (5,387 feet)");
  4. emit per-cell telemetry so live behavior can be audited.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.ledger_tools import LedgerToolkit
from agent.app.testing import execution_sequential as seq

_GATE_ON = seq.FinishGate(enabled=True, max_retries=2)


def _agent_io(decisions, search_results=None, page_text="PAGE CONTENT", synth="SYNTH ANSWER"):
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={"messages": []})
    io.query_llm = AsyncMock(side_effect=[*(json.dumps(d) for d in decisions), synth])
    io.search = AsyncMock(return_value=search_results or [
        {"title": "T", "url": "https://example.com/page", "description": "d"}])
    io.visit = AsyncMock(return_value=page_text)
    return io


def _telemetry():
    return {
        "evaluations": 0, "refusals": 0, "retries_used": 0,
        "final_state": "not_evaluated", "unbacked_numbers": [],
    }


# --------------------------------------------------------------------------- unit-level: predicate


def test_extract_number_tokens_handles_thousands_separators_and_decimals():
    out = seq._extract_number_tokens("1,642 metres and 5387.5 feet, -3 also 12")
    assert out == [
        ("1,642", 1642.0), ("5387.5", 5387.5), ("-3", -3.0), ("12", 12.0),
    ]


def test_numbers_within_tolerance_relative_half_percent():
    assert seq._numbers_within_tolerance(1000.0, 1004.0) is True    # 0.4% delta
    assert seq._numbers_within_tolerance(1000.0, 1010.0) is False   # 1.0% delta
    assert seq._numbers_within_tolerance(0.0, 0.0) is True


def test_predicate_zero_extractable_numbers_auto_refuses_even_with_no_ledger():
    # Refinement 2: undefined-predicate-passes is exactly the gap the pre-check found and closed.
    refuse, unbacked = seq._finish_gate_predicate("I cannot determine this.", None, 0)
    assert refuse is True
    assert unbacked == []


def test_predicate_no_ledger_context_does_not_fire_on_backed_number_grounds():
    # ledger_kit is None -> nothing to check backing against; a numeric answer still passes.
    refuse, unbacked = seq._finish_gate_predicate("The answer is 42.", None, 0)
    assert refuse is False
    assert unbacked == []


def test_predicate_ledger_bound_but_zero_nodes_and_zero_derive_attempts_does_not_fire():
    kit = LedgerToolkit()
    refuse, unbacked = seq._finish_gate_predicate("The answer is 42.", kit, 0)
    assert refuse is False


def test_predicate_ledger_bound_with_a_derive_attempt_but_no_nodes_still_checks():
    # A WRONG_ARITY/UNKNOWN_OPERATION refusal never mints a node, but it IS an attempt -- the
    # ledger has been engaged, so an unrelated number should no longer sail through for free.
    kit = LedgerToolkit()
    kit.derive("unknown_op", ["1", "2"])  # refused before any node is minted
    refuse, unbacked = seq._finish_gate_predicate("The answer is 42.", kit, 1)
    assert refuse is True
    assert unbacked == ["42"]


def test_predicate_compound_node_value_backs_the_answer():
    # Refinement 3: a stored node.value like "1,642 metres (5,387 feet; 898 fathoms)" must not be
    # float()-cast whole -- the embedded tokens are what a real answer would cite.
    kit = MagicMock()
    kit.artifact = MagicMock(return_value={
        "nodes": [{"value": "1,642 metres (5,387 feet; 898 fathoms)"}],
    })
    refuse, unbacked = seq._finish_gate_predicate("The depth is 5,387 feet.", kit, 1)
    assert refuse is False
    assert unbacked == []


def test_predicate_url_in_answer_does_not_count_url_digits_as_unbacked():
    # Refinement 1: a citation URL's digits (e.g. a year in the slug) must never be treated as a
    # committed answer number.
    kit = MagicMock()
    kit.artifact = MagicMock(return_value={"nodes": [{"value": "1642"}]})
    answer = "The answer is 1642 metres. https://en.wikipedia.org/wiki/Article_2024"
    refuse, unbacked = seq._finish_gate_predicate(answer, kit, 1)
    assert refuse is False
    assert unbacked == []


def test_predicate_unbacked_number_is_named_and_refused():
    kit = MagicMock()
    kit.artifact = MagicMock(return_value={"nodes": [{"value": "1642"}]})
    refuse, unbacked = seq._finish_gate_predicate("The answer is 9999.", kit, 1)
    assert refuse is True
    assert unbacked == ["9999"]


# --------------------------------------------------------------------------- integration: _run_react


def test_gate_inert_when_flag_off_default():
    # Default FinishGate() is disabled -> unconditional finish, byte-identical to before the gate.
    decisions = [{"thought": "done", "action": "finish", "args": {"answer": "9999 unbacked"}}]
    io = _agent_io(decisions)
    out = asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512))
    assert out == "9999 unbacked"
    io.query_llm.assert_awaited_once()  # no retry LLM call at all


def test_backed_answer_passes_without_retry():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/x"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "sum", "operands": ["1000 metres", "642 metres"]}},
        {"thought": "answer", "action": "finish", "args": {"answer": "The total is 1642 metres."}},
    ]
    io = _agent_io(decisions, page_text="Height A is 1000 metres. Height B is 642 metres.")
    telemetry = _telemetry()
    out = asyncio.run(seq._run_react(
        io, "task", "m", max_steps=6, max_tokens=512,
        ledger_kit=kit, finish_gate=_GATE_ON, gate_telemetry=telemetry,
    ))
    assert out == "The total is 1642 metres."
    assert telemetry["final_state"] == "passed"
    assert telemetry["refusals"] == 0
    assert telemetry["retries_used"] == 0
    assert telemetry["evaluations"] == 1


def test_unbacked_number_refused_then_passes_after_derive_on_retry():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/x"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "sum", "operands": ["1000 metres", "642 metres"]}},
        # First finish commits an unrelated, unbacked number -> refused.
        {"thought": "guess", "action": "finish", "args": {"answer": "The total is 7777 metres."}},
        # Model corrects to the derived value on retry -> passes.
        {"thought": "fix", "action": "finish", "args": {"answer": "The total is 1642 metres."}},
    ]
    io = _agent_io(decisions, page_text="Height A is 1000 metres. Height B is 642 metres.")
    telemetry = _telemetry()
    out = asyncio.run(seq._run_react(
        io, "task", "m", max_steps=6, max_tokens=512,
        ledger_kit=kit, finish_gate=_GATE_ON, gate_telemetry=telemetry,
    ))
    assert out == "The total is 1642 metres."
    assert telemetry["final_state"] == "passed"
    assert telemetry["refusals"] == 1
    assert telemetry["retries_used"] == 1
    assert telemetry["evaluations"] == 2
    assert telemetry["unbacked_numbers"] == ["7777"]
    # The corrective observation landed in the scratchpad the model saw on its next turn.
    second_finish_prompt = io.build_llm_payload.call_args_list[-1].kwargs["messages"][1]["content"]
    assert "not backed by the evidence ledger" in second_finish_prompt


def test_two_failed_retries_yields_abstention_banner_and_reason():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/x"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "sum", "operands": ["1000 metres", "642 metres"]}},
        {"thought": "guess1", "action": "finish", "args": {"answer": "The total is 7777 metres."}},
        {"thought": "guess2", "action": "finish", "args": {"answer": "The total is 8888 metres."}},
        {"thought": "guess3", "action": "finish", "args": {"answer": "The total is 9999 metres."}},
    ]
    io = _agent_io(decisions, page_text="Height A is 1000 metres. Height B is 642 metres.")
    telemetry = _telemetry()
    out = asyncio.run(seq._run_react(
        io, "task", "m", max_steps=10, max_tokens=512,
        ledger_kit=kit, finish_gate=_GATE_ON, gate_telemetry=telemetry,
    ))
    assert out.startswith(seq._GATE_ABSTENTION_BANNER)
    assert "9999 metres" in out
    assert telemetry["final_state"] == "abstained"
    assert telemetry["refusals"] == 3
    assert telemetry["retries_used"] == 2
    assert telemetry["evaluations"] == 3
    assert telemetry["unbacked_numbers"] == ["9999"]


def test_zero_number_answer_is_auto_refused_via_the_react_loop():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/x"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "sum", "operands": ["1000 metres", "642 metres"]}},
        {"thought": "hedge", "action": "finish", "args": {"answer": "I cannot determine this."}},
        {"thought": "answer", "action": "finish", "args": {"answer": "The total is 1642 metres."}},
    ]
    io = _agent_io(decisions, page_text="Height A is 1000 metres. Height B is 642 metres.")
    telemetry = _telemetry()
    out = asyncio.run(seq._run_react(
        io, "task", "m", max_steps=6, max_tokens=512,
        ledger_kit=kit, finish_gate=_GATE_ON, gate_telemetry=telemetry,
    ))
    assert out == "The total is 1642 metres."
    assert telemetry["refusals"] == 1
    assert telemetry["unbacked_numbers"] == []  # nothing to name -- the whole answer was the issue


def test_gate_fails_open_on_internal_exception(monkeypatch):
    monkeypatch.setattr(seq, "_finish_gate_predicate",
                         MagicMock(side_effect=RuntimeError("boom")))
    decisions = [{"thought": "done", "action": "finish", "args": {"answer": "9999 unbacked"}}]
    io = _agent_io(decisions)
    telemetry = _telemetry()
    out = asyncio.run(seq._run_react(
        io, "task", "m", max_steps=6, max_tokens=512,
        ledger_kit=LedgerToolkit(), finish_gate=_GATE_ON, gate_telemetry=telemetry,
    ))
    assert out == "9999 unbacked"  # fails open: answer ships despite the internal error
    assert telemetry["final_state"] == "passed"


def test_budget_forced_termination_does_not_retry():
    # Ledger already has context (pre-seeded, as if an earlier step derived something) so the
    # backed-number check is live; the FIRST decision the loop is given already IS the last
    # allowed step (max_steps=1), so there is no room left for a corrective retry.
    kit = LedgerToolkit()
    kit.register_page("https://example.com/x", "Height A is 1000 metres. Height B is 642 metres.")
    kit.derive("sum", ["1000 metres", "642 metres"])
    decisions = [{"thought": "guess", "action": "finish", "args": {"answer": "9999 unbacked"}}]
    io = _agent_io(decisions)
    telemetry = _telemetry()
    out = asyncio.run(seq._run_react(
        io, "task", "m", max_steps=1, max_tokens=512,
        ledger_kit=kit, finish_gate=_GATE_ON, gate_telemetry=telemetry,
    ))
    assert out.startswith(seq._GATE_ABSTENTION_BANNER)
    assert telemetry["final_state"] == "forced_by_budget"
    assert telemetry["retries_used"] == 0
    io.query_llm.assert_awaited_once()  # exactly one decision call, no retry round-trip


# --------------------------------------------------------------------------- wiring: run_sequential_execution


@pytest.mark.asyncio
async def test_run_sequential_execution_threads_finish_gate_flag_from_settings(monkeypatch):
    captured = {}

    async def _fake_run_react(agent_io, mandate, model_name, max_steps, max_tokens, **kwargs):
        captured["finish_gate"] = kwargs.get("finish_gate")
        captured["gate_telemetry"] = kwargs.get("gate_telemetry")
        return "answer"

    monkeypatch.setattr(seq, "_run_react", _fake_run_react)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."

    result_off = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
        idea_settings={},
    )
    assert captured["finish_gate"].enabled is False
    assert "gate_evaluations" not in result_off["output"]  # inert -> no telemetry keys at all

    result_on = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
        idea_settings={"final_require_derivation_for_numeric": True},
    )
    assert captured["finish_gate"].enabled is True
    assert result_on["output"]["gate_evaluations"] == 0
    assert result_on["output"]["gate_final_state"] == "not_evaluated"
    assert result_on["output"]["gate_unbacked_numbers"] == []


@pytest.mark.asyncio
async def test_run_sequential_execution_reports_abstention_reason_when_gate_fires(monkeypatch):
    """Wiring test: whatever `_run_react` reports through `gate_telemetry` must reach the cell's
    `output` dict verbatim -- this is what a later live audit reads, so the mapping itself
    (rather than the react loop's own retry/abstain logic, already covered above) is the thing
    under test here."""

    async def _fake_run_react(agent_io, mandate, model_name, max_steps, max_tokens, **kwargs):
        telemetry = kwargs["gate_telemetry"]
        telemetry.update({
            "evaluations": 1, "refusals": 1, "retries_used": 0,
            "final_state": "forced_by_budget", "unbacked_numbers": ["9999"],
        })
        return seq._GATE_ABSTENTION_BANNER + "The total is 9999 metres."

    monkeypatch.setattr(seq, "_run_react", _fake_run_react)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "What is the height?"

    result = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
        idea_settings={"final_require_derivation_for_numeric": True},
    )
    output = result["output"]
    assert output["gate_final_state"] == "forced_by_budget"
    assert output["gate_refusals"] == 1
    assert output["gate_unbacked_numbers"] == ["9999"]
    assert output["finish_gate"] == "refused-unbacked"
    assert output["finish_gate_reason"] == "unbacked_numeric_answer"
    assert output["final_deliverable"].startswith(seq._GATE_ABSTENTION_BANNER)
