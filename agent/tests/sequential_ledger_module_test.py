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


# -- quantity index rendered into the visit observation ----------------------------------------

def test_flag_off_visit_observation_has_no_index_text():
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text="Chimney\n419.7\nmetres")
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512))  # no ledger_kit

    assert "QUANTITIES" not in _observation(io, 1).upper()


def test_flag_on_visit_observation_appends_the_rendered_index_after_the_page_text():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text="Chimney\n419.7\nmetres")
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, ledger_kit=kit))

    obs = _observation(io, 1)
    assert "419.7\nmetres" in obs
    assert "q1" in obs
    assert obs.index("q1") > obs.index("419.7\nmetres"), "index must come AFTER the page text"


def test_a_page_with_no_extractable_quantity_adds_nothing_to_the_observation():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text="Nothing quantitative here.")
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, ledger_kit=kit))

    assert "QUANTITIES" not in _observation(io, 1).upper()


def test_derive_prompt_line_mentions_q_ids_as_optional():
    prompt = _prompt_when_on()
    lowered = prompt.lower()
    assert "q2" in lowered or "q<n>" in lowered
    assert "optional" in lowered or "never required" in lowered or "not required" in lowered


def test_a_q_id_operand_resolves_over_the_sequential_host():
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "compute", "action": "derive",
         "args": {"operation": "difference", "operands": ["q1", "q2"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions, page_text="Chimney\n419.7\nmetres\nAnnex\n380.0\nmetres")
    asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512, ledger_kit=kit))

    obs = _observation(io, 2)
    assert "DERIVED" in obs
    assert "39.7" in obs


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


# -- W1: `answer_audit` token -- host wiring only (LedgerToolkit.audit_answer itself is Lane A's) ---

def _system_message(io, step):
    return io.build_llm_payload.call_args_list[step].kwargs["messages"][0]["content"]


def test_token_parsing_derive_alone(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive")
    assert seq._ledger_derive_enabled() is True
    assert seq._ledger_answer_audit_enabled() is False


def test_token_parsing_answer_audit_alone(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "answer_audit")
    assert seq._ledger_derive_enabled() is False
    assert seq._ledger_answer_audit_enabled() is True


def test_token_parsing_both(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive,answer_audit")
    assert seq._ledger_derive_enabled() is True
    assert seq._ledger_answer_audit_enabled() is True


def test_token_parsing_neither(monkeypatch):
    monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    assert seq._ledger_derive_enabled() is False
    assert seq._ledger_answer_audit_enabled() is False


def test_derive_tool_stays_model_invisible_when_only_answer_audit_bound():
    """A `ledger_kit` bound with `derive_enabled=False` (what `run_sequential_execution` passes
    when only `answer_audit` is in `LEDGER_HOST_MODULES`) must reproduce the flag-off model
    surface EXACTLY -- no `derive` mention in the system prompt, `derive` still rejected as an
    invalid action -- even though a real `LedgerToolkit` is bound (so pages still register for
    `audit_answer` to have an index to match against)."""
    kit = LedgerToolkit()
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "t", "action": "derive",
         "args": {"operation": "difference", "operands": ["419.7 metres", "380.0 metres"]}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions)
    io_baseline = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=6, max_tokens=512,
                               ledger_kit=kit, derive_enabled=False))
    asyncio.run(seq._run_react(io_baseline, "task", "m", max_steps=6, max_tokens=512))  # no kit at all

    assert _system_message(io, 0) == _system_message(io_baseline, 0)
    assert "derive" not in _system_message(io, 0).lower()
    obs = _observation(io, 2)
    assert "INVALID ACTION" in obs
    assert "DERIVED" not in obs
    # the "Use <verb>/<verb>/..." available-actions list must not name derive -- unlike the
    # scratchpad above it (which legitimately echoes the model's OWN rejected "derive" attempt)
    assert "derive" not in obs.split("INVALID ACTION")[-1].lower()
    obs_baseline = _observation(io_baseline, 2)
    assert obs.split("INVALID ACTION")[-1] == obs_baseline.split("INVALID ACTION")[-1]
    # but the kit DID register the page, unlike a truly unbound one -- proven via the audit path
    assert kit.audit_answer("the chimney is 419.7 metres tall")["numbers"][0]["status"] == "backed"


@pytest.mark.asyncio
async def test_run_sequential_execution_stores_answer_audit_and_includes_minted_nodes(monkeypatch):
    """W1 outer hook: `output["answer_audit"]` is the dict `audit_answer` returns, and its minted
    SOURCE node is present in the SAME `output["evidence_graph"]` artifact (audit_answer called
    BEFORE `artifact()`, per the ordering requirement) -- even though the model never called
    `derive` at all here (zero-derive cell, the case W1 exists to cover)."""
    monkeypatch.setenv("LEDGER_HOST_MODULES", "answer_audit")
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "The chimney is 419.7 metres tall."}},
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
    tm.get_task_statement.return_value = "How tall is the chimney?"
    result = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    assert "answer_audit" in output
    audit = output["answer_audit"]
    assert audit["numbers_total"] >= 1
    assert any(n["status"] == "backed" for n in audit["numbers"])

    graph = output["evidence_graph"]
    minted_ids = {n["node_id"] for n in audit["numbers"] if n["node_id"]}
    graph_ids = {n["id"] for n in graph["nodes"]}
    assert minted_ids and minted_ids <= graph_ids
    assert any(n.get("minted_by") == "answer_audit" for n in graph["nodes"])


@pytest.mark.asyncio
async def test_run_sequential_execution_omits_answer_audit_when_token_absent(monkeypatch):
    monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "The chimney is 419.7 metres tall."}},
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
    tm.get_task_statement.return_value = "How tall is the chimney?"
    result = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    assert "answer_audit" not in output
    assert "evidence_graph" not in output
