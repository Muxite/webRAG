"""Offline unit tests for `LangGraphSolver`'s pure helpers (`agent/app/langgraph_solver.py`).
Free, no LLM, no network — mirrors `agent/tests/solver_normalize_test.py`'s pattern of testing
result-mapping logic in isolation.
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.app.langgraph_solver import (
    LangGraphSolver, _STEP_EXHAUSTED_TEXT, _extract_usage, _make_tools, _record_io_parity,
)


def test_extract_usage_pulls_tokens_from_ai_messages_only():
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="thinking", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}],
                   usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}),
        AIMessage(content="final answer"),  # no usage_metadata -> skipped
    ]
    usages = _extract_usage(messages)
    assert usages == [
        {"prompt_tokens": 100, "completion_tokens": 20},
        {"prompt_tokens": 50, "completion_tokens": 10},
    ]


def test_extract_usage_empty_list_returns_empty():
    assert _extract_usage([]) == []


def test_extract_usage_ignores_non_ai_messages():
    messages = [HumanMessage(content="hi"), SystemMessage(content="sys")]
    assert _extract_usage(messages) == []


class _FakeAgentIO:
    """Duck-typed stand-in for `AgentIO` — records calls, returns canned data, no real I/O."""

    def __init__(self):
        self.search_calls = []
        self.visit_calls = []

    async def search(self, query, count=10, timeout_seconds=None):
        self.search_calls.append((query, count))
        return [{"title": "Result A", "url": "https://example.com/a", "description": "desc A"}]

    async def visit(self, url, timeout_seconds=None):
        self.visit_calls.append(url)
        return "page content here"


def test_search_tool_delegates_to_agent_io_and_formats_results():
    fake_io = _FakeAgentIO()
    search_tool, _visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    out = asyncio.run(search_tool.ainvoke({"query": "test query"}))
    assert fake_io.search_calls == [("test query", 6)]
    assert "Result A" in out
    assert "https://example.com/a" in out


def test_search_tool_handles_empty_results():
    class _EmptyIO(_FakeAgentIO):
        async def search(self, query, count=10, timeout_seconds=None):
            self.search_calls.append((query, count))
            return []

    fake_io = _EmptyIO()
    search_tool, _visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    out = asyncio.run(search_tool.ainvoke({"query": "nothing"}))
    assert out == "No results."


def test_search_tool_marks_a_dead_backend_distinguishably_from_no_results():
    """`query_search` returns None (not raising) when its health probe fails, so a 403'd backend
    would otherwise read as "No results." and the model rephrases a dead query for turns on end."""
    class _DeadBackendIO(_FakeAgentIO):
        async def search(self, query, count=10, timeout_seconds=None):
            self.search_calls.append((query, count))
            return None

    search_tool, _visit_tool = _make_tools(_DeadBackendIO(), search_k=6, page_chars=6000)
    out = asyncio.run(search_tool.ainvoke({"query": "anything"}))
    assert out.startswith("SEARCH BACKEND UNAVAILABLE")
    assert out != "No results."


def test_visit_tool_delegates_to_agent_io_and_truncates():
    fake_io = _FakeAgentIO()
    _search_tool, visit_tool = _make_tools(fake_io, search_k=6, page_chars=5)
    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/page"}))
    assert fake_io.visit_calls == ["https://example.com/page"]
    assert out == "page "  # truncated to page_chars=5


def test_visit_tool_reports_no_content():
    class _EmptyVisitIO(_FakeAgentIO):
        async def visit(self, url, timeout_seconds=None):
            self.visit_calls.append(url)
            return ""

    fake_io = _EmptyVisitIO()
    _search_tool, visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/dead"}))
    assert out == "[No main content found]"


def test_langgraph_solver_name_and_construction():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    assert solver.name == "langgraph_react"
    assert LangGraphSolver.name == "langgraph_react"
    assert solver._full_capture is False


class _FakeTelemetry:
    """Records `record_event` calls; the only telemetry surface `_record_io_parity` touches."""

    def __init__(self):
        self.events = []

    def record_event(self, name, payload):
        self.events.append((name, payload))


def _capture_messages():
    return [
        SystemMessage(content="sys prompt"),
        HumanMessage(content="the task"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}]),
        ToolMessage(content="search results", tool_call_id="1"),
        AIMessage(content="final answer"),
    ]


def test_record_io_parity_full_capture_includes_role_content_pairs():
    telemetry = _FakeTelemetry()
    _record_io_parity(telemetry, _capture_messages(), full_capture=True)

    payloads = [payload["payload"] for _name, payload in telemetry.events]
    assert len(payloads) == 4  # one in/out pair per assistant turn

    first_in = payloads[0]
    assert first_in["prompt_text"] == "sys prompt\nthe task"
    assert first_in["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "the task"},
    ]
    assert payloads[1]["completion_text"] == ""

    second_in = payloads[2]
    assert second_in["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "search", "args": {"query": "x"}}]},
        {"role": "tool", "content": "search results"},
    ]
    assert payloads[3]["completion_text"] == "final answer"


def test_record_io_parity_full_capture_records_tool_calls():
    """A tool-only AIMessage has empty content, so without this the capture shows `[assistant] ''`
    and records neither the queries searched nor the URLs visited."""
    telemetry = _FakeTelemetry()
    _record_io_parity(telemetry, _capture_messages(), full_capture=True)
    payloads = [payload["payload"] for _name, payload in telemetry.events]

    assert payloads[1]["completion_tool_calls"] == [{"name": "search", "args": {"query": "x"}}]
    # Additive only: the prose turn keeps its content and gains no tool-call key.
    assert payloads[3]["completion_text"] == "final answer"
    assert "completion_tool_calls" not in payloads[3]


def test_record_io_parity_default_off_adds_no_new_keys():
    off, on_default = _FakeTelemetry(), _FakeTelemetry()
    _record_io_parity(off, _capture_messages(), full_capture=False)
    _record_io_parity(on_default, _capture_messages())

    assert off.events == on_default.events
    for _name, payload in off.events:
        keys = set(payload["payload"])
        assert keys in ({"prompt_chars", "prompt_words"}, {"completion_chars", "completion_words"})


class _StubLLM:
    """Stand-in for `ChatOpenAI` — only the synthesis pass's `ainvoke` is exercised."""

    def __init__(self, answer="synthesized answer"):
        self.answer = answer
        self.calls = []

    def bind_tools(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.answer)


def _run_solve_with_messages(monkeypatch, messages, llm=None, **solver_kwargs):
    """Drive `LangGraphSolver.solve` against a stubbed graph that just replays `messages`."""
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


def _exhausted_messages():
    return [
        HumanMessage(content="the task"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "x"}, "id": "1"}]),
        ToolMessage(content="Evidence: the founding year was 1861.", tool_call_id="1"),
        # What create_react_agent substitutes for the model's turn on step exhaustion.
        AIMessage(content=_STEP_EXHAUSTED_TEXT),
    ]


def test_step_exhaustion_canned_message_triggers_forced_synthesis(monkeypatch):
    result, llm = _run_solve_with_messages(monkeypatch, _exhausted_messages())
    assert result["final_deliverable"] == "synthesized answer"
    assert result["success"] is True
    assert "step budget" in result.get("warning", "")
    assert len(llm.calls) == 1  # the synthesis pass ran on the gathered evidence


def test_step_exhaustion_canned_message_is_never_the_deliverable(monkeypatch):
    """Even with no evidence to synthesize from, the canned apology must not be returned as an
    answer (it would score as a genuine attempt and hide the exhausted-budget condition)."""
    messages = [HumanMessage(content="the task"), AIMessage(content=_STEP_EXHAUSTED_TEXT)]
    result, _llm = _run_solve_with_messages(monkeypatch, messages)
    assert result["final_deliverable"] == ""
    assert result["success"] is False
    assert "step budget" in result.get("warning", "")


def test_canned_step_exhaustion_string_matches_installed_langgraph():
    """Tripwire: the constant is copied from langgraph's source, so a library-side reword would
    silently make the fix above inert."""
    from pathlib import Path

    import langgraph.prebuilt.chat_agent_executor as executor

    assert _STEP_EXHAUSTED_TEXT in Path(executor.__file__).read_text(encoding="utf-8")


def test_natural_termination_is_returned_verbatim_by_default(monkeypatch):
    messages = [
        HumanMessage(content="the task"),
        ToolMessage(content="Evidence: 1861.", tool_call_id="1"),
        AIMessage(content="The answer is 1861."),
    ]
    result, llm = _run_solve_with_messages(monkeypatch, messages)
    assert result["final_deliverable"] == "The answer is 1861."
    assert llm.calls == []  # no extra synthesis call, no cost change


def test_always_synthesize_reworks_a_natural_termination(monkeypatch):
    messages = [
        HumanMessage(content="the task"),
        ToolMessage(content="Evidence: 1861.", tool_call_id="1"),
        AIMessage(content="Now, I will compute the difference."),
    ]
    result, llm = _run_solve_with_messages(monkeypatch, messages, always_synthesize=True)
    assert result["final_deliverable"] == "synthesized answer"
    assert len(llm.calls) == 1
    assert "DRAFT ANSWER" in llm.calls[0][-1].content


def test_always_synthesize_keeps_the_original_answer_when_synthesis_is_empty(monkeypatch):
    messages = [
        HumanMessage(content="the task"),
        ToolMessage(content="Evidence: 1861.", tool_call_id="1"),
        AIMessage(content="The answer is 1861."),
    ]
    result, _llm = _run_solve_with_messages(
        monkeypatch, messages, llm=_StubLLM(answer=""), always_synthesize=True,
    )
    assert result["final_deliverable"] == "The answer is 1861."


class _StubTestModule:
    metadata = {"test_id": "999"}

    def get_task_statement(self):
        return "do the thing"


def _run_offtheshelf_capturing_kwargs(monkeypatch, verbosity):
    from agent.app.testing import execution_langgraph

    captured = {}

    class _StubSolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def solve(self, mandate, **kwargs):
            return {"final_deliverable": "answer", "success": True, "observability": {}}

    monkeypatch.setattr(execution_langgraph, "LangGraphSolver", _StubSolver)
    if verbosity is None:
        monkeypatch.delenv("IDEA_TEST_REPORT_VERBOSITY", raising=False)
    else:
        monkeypatch.setenv("IDEA_TEST_REPORT_VERBOSITY", verbosity)
    asyncio.run(execution_langgraph.run_offtheshelf_execution(
        test_module=_StubTestModule(), model_name="openai/gpt-5-mini",
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        run_stamp="teststamp",
    ))
    return captured


def test_offtheshelf_execution_enables_full_capture_at_verbosity_3(monkeypatch):
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, "3")["full_capture"] is True


def test_offtheshelf_execution_leaves_full_capture_off_by_default(monkeypatch):
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["full_capture"] is False
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, "2")["full_capture"] is False
