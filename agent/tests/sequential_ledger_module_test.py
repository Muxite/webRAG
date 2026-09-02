"""Binding the Ledger module (agent/app/ledger_tools.py) onto the sequential ReAct host.

`docs/LEDGER.md` says the product "is a component, not an agent", but the only proof that would
demonstrate host-agnosticism is the SAME module working over two hosts with genuinely different
tool-calling shapes. `execution_sequential.py` is a plain text/JSON string-dispatch loop -- no
tool-calling API at all -- so a `derive` action working here (alongside a LangGraph binding built
separately) is the actual evidence, not a restatement of what the module already tests on its own
(see `agent/tests/ledger_tools_test.py`, which is frozen and untouched by this file).

Gated by `LEDGER_HOST_MODULES` (comma-separated, default OFF -- unset must reproduce today's
behavior EXACTLY, including no `evidence_graph` key at all).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_sequential as seq
from agent.app.ledger_tools import LedgerToolkit
from agent.app.testing.evidence_graph import reverify_graph

PAGE = ("Ekibastuz GRES-2 has a chimney 419.7 metres tall. "
        "The Inco Superstack is 380.0 metres tall.")


def _agent_io(decisions, page_text=PAGE, synth="SYNTH"):
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={"messages": []})
    io.query_llm = AsyncMock(side_effect=[*(json.dumps(d) for d in decisions), synth])
    io.search = AsyncMock(return_value=[])
    io.visit = AsyncMock(return_value=page_text)
    return io


def _observation(io, step):
    return io.build_llm_payload.call_args_list[step].kwargs["messages"][1]["content"]


# -- flag off: default path is byte-identical -------------------------------------------------

def test_prompt_omits_derive_when_flag_off():
    assert "derive" not in seq._system_prompt(has_sandbox=False).lower()


def test_derive_action_is_rejected_as_invalid_without_a_bound_ledger_kit():
    decisions = [
        {"thought": "t", "action": "derive",
         "args": {"operation": "difference", "operands": ["419.7 metres", "380.0 metres"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512))  # no ledger_kit
    obs = _observation(io, 1)
    assert "INVALID ACTION" in obs
    assert "DERIVED" not in obs


def test_run_sequential_execution_has_no_evidence_graph_key_when_flag_off(monkeypatch):
    monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    monkeypatch.setattr(seq, "_run_react", AsyncMock(return_value="answer"))
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."
    result = asyncio.run(seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    ))
    assert "evidence_graph" not in result["output"]


# -- flag on: derive works over operands from a visited page -----------------------------------

def _prompt_when_on():
    return seq._system_prompt(has_sandbox=False, has_derive=True)


def test_prompt_describes_derive_for_a_weak_model_when_flag_on():
    prompt = _prompt_when_on()
    assert "derive(" in prompt
    assert "operation" in prompt and "operands" in prompt
    assert "already read" in prompt.lower() or "already visited" in prompt.lower()
    assert "computes the result" in prompt.lower() or "does not trust" in prompt.lower() \
        or "not your own arithmetic" in prompt.lower() or "do not trust your own arithmetic" in prompt.lower()


def test_derive_over_an_operand_from_a_visited_page_is_computed():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "difference", "operands": ["419.7 metres", "380.0 metres"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "39.7 metres"}},
    ]
    io = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512, ledger_kit=kit))
    assert "DERIVED" in _observation(io, 2)
    assert "39.7" in _observation(io, 2)


def test_derive_action_visible_in_invalid_action_list_when_flag_on():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "t", "action": "bogus", "args": {}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, ledger_kit=kit))
    assert "derive" in _observation(io, 1).lower()


def test_an_operand_never_read_is_refused_not_computed():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "difference", "operands": ["419.7 metres", "999.9 metres"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512, ledger_kit=kit))
    assert "REFUSED" in _observation(io, 2).upper()


def test_an_operand_past_the_models_visible_window_is_refused(monkeypatch):
    """The registered page text must be the SAME truncated window the model was shown, not the
    fuller fetched text -- else a value the model never actually saw (past its visible window,
    recalled from parametric memory) could be credited as read-off-the-page. This also keeps the
    host comparison valid: `execution_evidence_loop` truncates at fetch before grounding
    (`content = (await agent_io.visit(...))[:page_chars]`), so this host must match that, not
    ground more permissively than the reference implementation."""
    monkeypatch.setenv("IDEA_TEST_SEQ_PAGE_CHARS", "20")
    kit = LedgerToolkit()
    page_text = ("x" * 30) + " the chimney is 419.7 metres tall and the tower is 380.0 metres tall"
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "difference", "operands": ["419.7 metres", "380.0 metres"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text=page_text)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512, ledger_kit=kit))
    obs = _observation(io, 2)
    assert "REFUSED" in obs.upper()
    assert "419.7" in obs  # the refused operand is named, not silently computed
    # sanity: the value really was truncated out of what the model was shown
    assert "419.7" not in _observation(io, 1)


def test_a_unit_mismatch_is_refused():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "sum", "operands": ["200.0 USD", "50.0 km"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text="The bridge cost 200.0 USD and spans 50.0 km.")
    asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512, ledger_kit=kit))
    assert "REFUSED" in _observation(io, 2).upper()


def test_visit_return_value_to_the_model_is_unchanged_when_ledger_kit_bound():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text="hello world")
    io_baseline = _agent_io(decisions, page_text="hello world")
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, ledger_kit=kit))
    asyncio.run(seq._run_react(io_baseline, "task", "m", max_steps=4, max_tokens=512))
    assert _observation(io, 1) == _observation(io_baseline, 1)


# -- artifact persistence + round-trip -----------------------------------------------------------

@pytest.mark.asyncio
async def test_run_sequential_execution_persists_evidence_graph_when_flag_on(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive")
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "difference", "operands": ["419.7 metres", "380.0 metres"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "39.7 metres"}},
    ]

    class _FakeAgentIO:
        def __init__(self, *a, **kw):
            pass
        build_llm_payload = MagicMock(return_value={"messages": []})
        query_llm = AsyncMock(side_effect=[*(json.dumps(d) for d in decisions), "SYNTH"])
        search = AsyncMock(return_value=[])
        visit = AsyncMock(return_value=PAGE)

    monkeypatch.setattr(seq, "AgentIO", _FakeAgentIO)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."
    result = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )
    graph = result["output"]["evidence_graph"]
    assert graph is not None
    assert any(node["kind"] == "derived" for node in graph["nodes"])

    reverified = reverify_graph(graph)
    assert reverified is not None
