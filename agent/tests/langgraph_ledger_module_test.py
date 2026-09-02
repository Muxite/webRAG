"""Offline unit tests for binding the Euglena Ledger onto `LangGraphSolver` as an attachable
module (`agent/app/ledger_tools.LedgerToolkit`), gated by `LEDGER_HOST_MODULES`.

Every prior experiment measured the ledger as a RIVAL agent (`evidence_loop` vs `langgraph_react`)
-- two arms differing in loop, prompt, budget, step policy and output contract simultaneously, so
a delta could never be attributed to the ledger itself. These tests pin the honest alternative:
host vs host + module, one change, same system. `LEDGER_HOST_MODULES` unset must reproduce prior
behavior EXACTLY -- a bit-identical tool list and no `evidence_graph` key at all (absent, not
empty).
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agent.app.langgraph_solver import LangGraphSolver, _DERIVE_TOOL_DOC, _make_tools
from agent.app.ledger_tools import LedgerToolkit
from agent.app.testing import evidence_graph


class _FakeAgentIO:
    """Duck-typed stand-in for `AgentIO` -- mirrors `langgraph_solver_test.py`'s fixture, kept
    local here since that file is out of scope for this change (frozen, unmodified tests)."""

    def __init__(self, pages=None):
        self.search_calls = []
        self.visit_calls = []
        self._pages = pages or {}

    async def search(self, query, count=10, timeout_seconds=None):
        self.search_calls.append((query, count))
        return [{"title": "Result A", "url": "https://example.com/a", "description": "desc A"}]

    async def visit(self, url, timeout_seconds=None):
        self.visit_calls.append(url)
        return self._pages.get(url, "page content here")


class _StubLLM:
    """Stand-in for `ChatOpenAI` -- only the synthesis pass's `ainvoke` is exercised."""

    def __init__(self, answer="synthesized answer"):
        self.answer = answer
        self.calls = []

    def bind_tools(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.answer)


def _run_solve_with_messages(monkeypatch, messages, llm=None, **solver_kwargs):
    """Drive `LangGraphSolver.solve` against a stubbed graph that just replays `messages`.
    Copied from `langgraph_solver_test.py`'s helper of the same name (that file is frozen)."""
    from agent.app import langgraph_solver

    class _StubGraph:
        async def astream(self, _inputs, config=None, stream_mode=None):
            yield {"messages": messages}

    llm = llm or _StubLLM()
    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: _StubGraph())
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: llm)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", **solver_kwargs,
    )
    return asyncio.run(solver.solve("the task", max_steps=4)), llm


def _natural_termination_messages():
    return [HumanMessage(content="the task"), AIMessage(content="the final answer text")]


# -- flag off: bit-identical to prior behavior -------------------------------------------------


def test_make_tools_with_no_ledger_kit_is_unchanged():
    """The default -- no `ledger_kit` argument at all -- must be exactly `["search", "visit"]`,
    matching `langgraph_solver_test.py::test_make_tools_includes_finish_only_when_required`."""
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000)
    assert [t.name for t in tools] == ["search", "visit"]


def test_flag_off_produces_no_evidence_graph_key(monkeypatch):
    result, _llm = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    assert "evidence_graph" not in result


def test_flag_off_is_the_constructor_default(monkeypatch):
    """No `ledger_host_modules` kwarg at all -- the library/direct-construction default -- must
    behave identically to an explicit empty list."""
    result_default, _ = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    result_empty, _ = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=[])
    assert "evidence_graph" not in result_default
    assert "evidence_graph" not in result_empty


# -- flag on: tool present on both transports ----------------------------------------------------


def test_make_tools_appends_derive_tool_when_ledger_kit_is_bound():
    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit)
    assert [t.name for t in tools] == ["search", "visit", "derive"]
    assert tools[-1].description == _DERIVE_TOOL_DOC


def test_derive_tool_docstring_tells_a_weak_model_operands_must_be_read_first():
    """The model-facing contract: operands must come from a visited page, and the tool computes
    the answer rather than trusting the model's own arithmetic."""
    lowered = _DERIVE_TOOL_DOC.lower()
    assert "already read" in lowered or "already visited" in lowered
    assert "computes the answer" in lowered or "recompute" in lowered or "does not trust" in lowered.replace("n't", "not")


def test_derive_tool_bound_on_the_emulated_transport(monkeypatch):
    from agent.app import langgraph_solver

    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    monkeypatch.setattr(langgraph_solver, "resolve_tool_transport_mode", lambda: "prompted")
    transport, tool_transport = asyncio.run(solver._build_transport(_StubLLM(), tools, "sys"))
    assert tool_transport == "emulated"
    assert "derive" in transport._by_name


def test_derive_tool_bound_on_the_native_transport(monkeypatch):
    from agent.app import langgraph_solver

    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    captured = {}

    def _capturing_create_react_agent(llm, bound_tools, **kwargs):
        captured["tools"] = bound_tools
        return object()

    monkeypatch.setattr(langgraph_solver, "resolve_tool_transport_mode", lambda: "native")
    monkeypatch.setattr(langgraph_solver, "create_react_agent", _capturing_create_react_agent)
    _transport, tool_transport = asyncio.run(solver._build_transport(_StubLLM(), tools, "sys"))
    assert tool_transport == "native"
    assert [t.name for t in captured["tools"]] == ["search", "visit", "derive"]


# -- the tool actually working --------------------------------------------------------------------


def test_derive_over_an_operand_read_on_a_visited_page_returns_the_recomputed_value():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={
        "https://example.com/tower": "The tower is 419.7 metres tall and the annex is 380.0 metres tall.",
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/tower"}))
    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "380.0 metres"],
    }))

    assert "DERIVED" in out
    assert "39.7" in out


def test_derive_refuses_an_operand_never_read():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={"https://example.com/tower": "The tower is 419.7 metres tall."})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/tower"}))
    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "999.9 metres"],
    }))

    assert "REFUSED" in out.upper()
    assert "999.9" in out


def test_visiting_a_page_never_shown_to_derive_still_registers_it_via_the_kit():
    """The registration site is `visit`, not `derive` -- the model does not need to name a page,
    only the value; a value that was fetched (even if truncated for display) is a valid operand."""
    kit = LedgerToolkit(max_page_chars=6000)
    fake_io = _FakeAgentIO(pages={"https://example.com/x": "Revenue was 500.0 million dollars."})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, _derive_tool = tools

    assert kit.artifact()["pages"] == []
    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/x"}))
    assert len(kit.artifact()["pages"]) == 1


# -- the persisted artifact ------------------------------------------------------------------------


def test_flag_on_result_carries_the_evidence_graph_key(monkeypatch):
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["derive"])
    assert "evidence_graph" in result
    assert result["evidence_graph"]["pages"] == []
    assert result["evidence_graph"]["nodes"] == []


def test_unrecognized_ledger_module_name_is_ignored_not_an_error(monkeypatch):
    """The list is meant to grow -- an entry this build doesn't recognize must not raise or turn
    the module on."""
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["some_future_module"])
    assert "evidence_graph" not in result


def test_persisted_artifact_round_trips_through_reverify_graph():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={
        "https://example.com/tower": "The tower is 419.7 metres tall and the annex is 380.0 metres tall.",
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools
    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/tower"}))
    asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "380.0 metres"],
    }))

    artifact = kit.artifact()
    report = evidence_graph.reverify_graph(artifact)

    assert report["counts"]["source"] == 2
    assert report["counts"]["derived"] == 1
    assert report["counts"]["failed"] == 0


def test_a_value_past_the_models_visible_window_is_not_admissible_as_an_operand():
    """The module must ground against what the model READ, not merely what was fetched.

    Registering the full fetched text while showing the model a truncated window opens a
    false-grounding path: a model that produces a number from parametric memory, where that number
    happens to sit past its visible window, would be credited as having read it off the page. That
    is precisely the fabrication this module exists to catch.

    It also breaks the host comparison the module was built to enable. `execution_evidence_loop`
    truncates AT FETCH (`content = visit(...)[:page_chars]`), so its ledger can only ever ground
    against the model's own view; a host binding that grounded against more would be applying
    weaker admission rules than the reference implementation, in the looser direction.
    """
    kit = LedgerToolkit()
    hidden = "The secret figure is 987.6 metres."
    fake_io = _FakeAgentIO(pages={
        "https://example.com/long": "The tower is 419.7 metres tall. " + ("padding. " * 40) + hidden,
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=60, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    shown = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/long"}))
    assert "987.6" not in shown, "fixture must actually hide the value from the model"

    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "987.6 metres"],
    }))

    assert "REFUSED" in out.upper()
    assert "987.6" in out
