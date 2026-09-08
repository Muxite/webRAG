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


# -- `shape_derive` token -- host wiring only (LedgerToolkit.shape_derive_check itself is Lane A's,
# see agent/tests/ledger_tools_test.py) ------------------------------------------------------------

def test_token_parsing_shape_derive_alone(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "shape_derive")
    assert seq._ledger_derive_enabled() is False
    assert seq._ledger_answer_audit_enabled() is False
    assert seq._ledger_shape_derive_enabled() is True


def test_token_parsing_all_three(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive,answer_audit,shape_derive")
    assert seq._ledger_derive_enabled() is True
    assert seq._ledger_answer_audit_enabled() is True
    assert seq._ledger_shape_derive_enabled() is True


def test_token_parsing_neither_includes_shape_derive(monkeypatch):
    monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    assert seq._ledger_shape_derive_enabled() is False


def _fake_agent_io_class(decisions, page_text=PAGE):
    class _FakeAgentIO:
        def __init__(self, *a, **kw):
            pass
        build_llm_payload = MagicMock(return_value={"messages": []})
        query_llm = AsyncMock(side_effect=[*(json.dumps(d) for d in decisions), "SYNTH"])
        search = AsyncMock(return_value=[])
        visit = AsyncMock(return_value=page_text)
    return _FakeAgentIO


async def _run_with_modules(monkeypatch, modules, mandate="What is the absolute difference "
                            "between the chimneys, in m?", page_text=PAGE):
    monkeypatch.setenv("LEDGER_HOST_MODULES", modules)
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish",
         "args": {"answer": "The absolute difference is 39.7 metres."}},
    ]
    monkeypatch.setattr(seq, "AgentIO", _fake_agent_io_class(decisions, page_text=page_text))
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = mandate
    result = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )
    return result["output"]


@pytest.mark.asyncio
async def test_run_sequential_execution_stores_shape_derive_and_includes_minted_nodes(monkeypatch):
    """`output["shape_derive"]` is the dict `shape_derive_check` returns, and a matched node is
    present in the SAME `output["evidence_graph"]` artifact (called BEFORE `artifact()`, same
    ordering requirement W1's `answer_audit` hook already follows)."""
    output = await _run_with_modules(monkeypatch, "shape_derive")
    assert "shape_derive" in output
    shape = output["shape_derive"]
    assert shape["demanded_operation"] == "difference"
    assert shape["verdict"] is True
    assert shape["matched"] is not None

    graph = output["evidence_graph"]
    graph_ids = {n["id"] for n in graph["nodes"]}
    assert shape["matched"]["derived_node_id"] in graph_ids
    assert any(n.get("minted_by") == "shape_derive" for n in graph["nodes"])


@pytest.mark.asyncio
async def test_run_sequential_execution_omits_shape_derive_when_token_absent(monkeypatch):
    output = await _run_with_modules(monkeypatch, "answer_audit")
    assert "shape_derive" not in output
    assert "answer_audit" in output


@pytest.mark.asyncio
async def test_run_sequential_execution_omits_shape_derive_and_evidence_graph_when_no_module(
        monkeypatch):
    output = await _run_with_modules(monkeypatch, "")
    assert "shape_derive" not in output
    assert "evidence_graph" not in output


@pytest.mark.asyncio
async def test_shape_derive_model_invisibility_prompt_and_tools_are_byte_identical(monkeypatch):
    """Adding `shape_derive` to `LEDGER_HOST_MODULES` alongside `derive,answer_audit` must not
    change one byte of what the model sees: same system prompt, same valid-action list, same
    per-visit observation text. `shape_derive_check` runs finish-time, host-side only."""
    decisions_a = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish",
         "args": {"answer": "The absolute difference is 38.7 metres."}},
    ]
    decisions_b = [dict(d) for d in decisions_a]
    io_without = _agent_io(decisions_a)
    io_with = _agent_io(decisions_b)

    kit_without = LedgerToolkit()
    kit_with = LedgerToolkit()

    await seq._run_react(io_without, "task", "m", max_steps=6, max_tokens=512,
                         ledger_kit=kit_without, derive_enabled=True)
    await seq._run_react(io_with, "task", "m", max_steps=6, max_tokens=512,
                         ledger_kit=kit_with, derive_enabled=True)

    assert _system_message(io_without, 0) == _system_message(io_with, 0)
    assert _observation(io_without, 1) == _observation(io_with, 1)


# -- `host_derive` token -- host wiring only (LedgerToolkit.host_derive itself is tested in
# agent/tests/host_derive_test.py) ---------------------------------------------------------------
#
# The fourth token differs from the three above in one way that matters to the wiring: it never
# reads the answer. It takes the MANDATE and the pages the run registered, so it produces a number
# the model had no hand in -- which is only true if the host calls it with the task statement.

#: A two-slot mandate of the 216 shape (two FIELDS of one entity), with an infobox-shaped page, so
#: the wiring test can assert a real `computed` result rather than a refusal.
_HOST_DERIVE_MANDATE = (
    "You are given NO raw figures -- search to find the page(s) you need. You need TWO values:\n"
    "  A. Open the Wikipedia page for Tokaido Shinkansen and read its route (line) length, "
    "in km.\n"
    "  B. Open the Wikipedia page for Tokaido Shinkansen and read its journey time, in hours.\n"
    "\nThen COMPUTE the RATIO of the first value to the second (first value divided by the "
    "second).\n")

_HOST_DERIVE_PAGE = ("Tokaido Shinkansen\nThe Tokaido Shinkansen is a Japanese high-speed rail "
                     "line.\nLine length\n515.4\nkm\nJourney time\n2.35\nh\nOpened\n1964\n")

#: Every key `host_derive`'s contract promises, on every reason. Other lanes read these.
_HOST_DERIVE_KEYS = {"reason", "operation", "absolute", "mode", "value", "value_text", "unit",
                     "node_id", "winner_entity", "slots", "ranker", "n_pages", "n_entries",
                     "min_score"}


def test_token_parsing_host_derive_alone(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "host_derive")
    assert seq._ledger_derive_enabled() is False
    assert seq._ledger_answer_audit_enabled() is False
    assert seq._ledger_shape_derive_enabled() is False
    assert seq._ledger_host_derive_enabled() is True


def test_token_parsing_all_four(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive,answer_audit,shape_derive,host_derive")
    assert seq._ledger_derive_enabled() is True
    assert seq._ledger_answer_audit_enabled() is True
    assert seq._ledger_shape_derive_enabled() is True
    assert seq._ledger_host_derive_enabled() is True


def test_token_parsing_neither_includes_host_derive(monkeypatch):
    monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    assert seq._ledger_host_derive_enabled() is False


@pytest.mark.asyncio
async def test_run_sequential_execution_stores_host_derive_and_includes_minted_nodes(monkeypatch):
    """`output["host_derive"]` is the dict `host_derive` returns, and the nodes it minted are in
    the SAME `output["evidence_graph"]` artifact -- i.e. it ran BEFORE `artifact()`."""
    output = await _run_with_modules(monkeypatch, "host_derive",
                                     mandate=_HOST_DERIVE_MANDATE, page_text=_HOST_DERIVE_PAGE)

    assert set(output["host_derive"]) == _HOST_DERIVE_KEYS
    host = output["host_derive"]
    assert host["reason"] == "computed"
    assert host["operation"] == "quotient"
    assert host["unit"] == "km/h"
    assert host["value"] == pytest.approx(515.4 / 2.35, abs=1e-6)

    graph_ids = {n["id"] for n in output["evidence_graph"]["nodes"]}
    assert host["node_id"] in graph_ids
    assert any(n.get("minted_by") == "host_derive" for n in output["evidence_graph"]["nodes"])


@pytest.mark.asyncio
async def test_run_sequential_execution_omits_host_derive_when_token_absent(monkeypatch):
    output = await _run_with_modules(monkeypatch, "shape_derive",
                                     mandate=_HOST_DERIVE_MANDATE, page_text=_HOST_DERIVE_PAGE)
    assert "host_derive" not in output
    assert "shape_derive" in output


@pytest.mark.asyncio
async def test_host_derive_changes_nothing_the_model_sees(monkeypatch):
    """The token adds one output key and nothing else: same prompt, same actions, same answer.
    `host_derive` registers no tool and writes no prompt text -- it runs finish-time, host-side."""
    without = await _run_with_modules(monkeypatch, "derive,answer_audit,shape_derive",
                                      mandate=_HOST_DERIVE_MANDATE, page_text=_HOST_DERIVE_PAGE)
    with_token = await _run_with_modules(
        monkeypatch, "derive,answer_audit,shape_derive,host_derive",
        mandate=_HOST_DERIVE_MANDATE, page_text=_HOST_DERIVE_PAGE)

    assert set(with_token) - set(without) == {"host_derive"}
    assert with_token["final_deliverable"] == without["final_deliverable"]
    assert with_token["shape_derive"] == without["shape_derive"]
    assert with_token["answer_audit"] == without["answer_audit"]


# -- `host_prefetch` token -- host wiring only (the prefetcher itself is tested in
# agent/tests/host_prefetch_test.py) ----------------------------------------------------------

_PREFETCH_MANDATE = ("For EACH of the following two chimneys, read its HEIGHT in metres from the "
                     "infobox:\n  1. GRES-2 Power Station chimney\n  2. Inco Superstack\n"
                     "Then compute the absolute difference between the two heights, in metres.")


def _prefetch_html(label, value, lead):
    return (f"<html><body><table class='infobox'><tr><th>{label}</th><td>{value}</td></tr>"
            f"</table><p>{lead}</p></body></html>")


class _PrefetchHttp:
    """Serves the Wikipedia search API and two articles; anything else is a 404."""

    def __init__(self):
        self.calls = []

    def set_telemetry(self, *_a, **_k):
        pass

    async def request(self, method, url, retries=2, **kwargs):
        self.calls.append(url)
        if "api.php" in url:
            query = url.split("srsearch=")[1].split("&")[0]
            title = "Ekibastuz_GRES-2_Power_Station" if "GRES" in query else "Inco_Superstack"
            data = {"query": {"search": [{"title": title.replace("_", " "), "snippet": ""}]}}
            return type("R", (), {"status": 200, "error": False, "data": data})()
        if url.endswith("Ekibastuz_GRES-2_Power_Station"):
            return type("R", (), {"status": 200, "error": False, "data": _prefetch_html(
                "Height", "419.7 m (1,377 ft)", "The Ekibastuz GRES-2 Power Station chimney.")})()
        if url.endswith("Inco_Superstack"):
            return type("R", (), {"status": 200, "error": False, "data": _prefetch_html(
                "Height", "380 m (1,250 ft)", "The Inco Superstack is a smokestack.")})()
        return type("R", (), {"status": 404, "error": True, "data": "nf"})()


def _prefetch_agent_io_class(decisions, http, search):
    class _FakeAgentIO:
        def __init__(self, *a, **kw):
            self.connector_http = http
            self.connector_search = search
        build_llm_payload = MagicMock(return_value={"messages": []})
        query_llm = AsyncMock(side_effect=[*(json.dumps(d) for d in decisions), "SYNTH"])
        search = AsyncMock(return_value=[])
        visit = AsyncMock(return_value=PAGE)
    return _FakeAgentIO


async def _run_prefetch_cell(monkeypatch, modules):
    if modules:
        monkeypatch.setenv("LEDGER_HOST_MODULES", modules)
    else:
        monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    http = _PrefetchHttp()
    search = MagicMock()
    search.query_search = AsyncMock(return_value=[])
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://example.com/a"}},
        {"thought": "done", "action": "finish", "args": {"answer": "39.7 metres."}},
    ]
    monkeypatch.setattr(seq, "AgentIO", _prefetch_agent_io_class(decisions, http, search))
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = _PREFETCH_MANDATE
    result = await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
    )
    return result, http, search


def test_token_parsing_host_prefetch_alone(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "host_prefetch")
    assert seq._ledger_host_prefetch_enabled() is True
    assert seq._ledger_host_derive_enabled() is False
    assert seq._ledger_derive_enabled() is False


def test_token_parsing_all_five(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive,answer_audit,shape_derive,host_derive,host_prefetch")
    assert seq._ledger_host_prefetch_enabled() is True
    assert seq._ledger_host_derive_enabled() is True


def test_token_parsing_neither_includes_host_prefetch(monkeypatch):
    monkeypatch.delenv("LEDGER_HOST_MODULES", raising=False)
    assert seq._ledger_host_prefetch_enabled() is False


@pytest.mark.asyncio
async def test_host_prefetch_registers_tagged_pages_without_touching_what_the_model_saw(monkeypatch):
    """With the token, the host registers one full-text page per slot entity, tagged
    `source == "host_prefetch"`, via `connector_http` directly -- and `output.pages` / the
    visit count are exactly what they are without the token."""
    with_token, http, search = await _run_prefetch_cell(monkeypatch, "host_prefetch,host_derive")
    without, http_off, _ = await _run_prefetch_cell(monkeypatch, "host_derive")

    prefetch = with_token["output"]["host_prefetch"]
    assert prefetch["error"] is None
    assert prefetch["registered"] == 2 and prefetch["searches"] == 2 and prefetch["fetches"] == 2
    assert [row["status"] for row in prefetch["entities"]] == ["prefetched", "prefetched"]
    pages = with_token["output"]["evidence_graph"]["pages"]
    tagged = [p for p in pages if p.get("source") == "host_prefetch"]
    assert len(tagged) == 2 and all(p["truncated"] is False for p in tagged)
    assert search.query_search.await_count == 0          # API sufficed; no web search
    assert http_off.calls == []                          # token absent: connector untouched
    # Nothing the validator grounds against moved: same `output.pages`, same visit telemetry.
    assert with_token["output"].get("pages") == without["output"].get("pages")
    assert with_token.get("visits") == without.get("visits")
    # And the roster is now complete enough for `host_derive` to compute from the host copies.
    assert with_token["output"]["host_derive"]["reason"] == "computed"
    assert with_token["output"]["host_derive"]["value"] == pytest.approx(39.7)


@pytest.mark.asyncio
async def test_host_prefetch_absent_leaves_the_artifact_byte_identical(monkeypatch):
    result, http, _ = await _run_prefetch_cell(monkeypatch, "host_derive")
    assert "host_prefetch" not in result["output"]
    assert http.calls == []
    assert all("source" not in p for p in result["output"]["evidence_graph"]["pages"])


@pytest.mark.asyncio
async def test_host_prefetch_alone_still_constructs_the_kit_and_persists_the_graph(monkeypatch):
    result, _http, _ = await _run_prefetch_cell(monkeypatch, "host_prefetch")
    assert "evidence_graph" in result["output"]
    assert result["output"]["host_prefetch"]["registered"] == 2
    assert "host_derive" not in result["output"]
