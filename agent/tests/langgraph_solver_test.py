"""Offline unit tests for `LangGraphSolver`'s pure helpers (`agent/app/langgraph_solver.py`).
Free, no LLM, no network — mirrors `agent/tests/solver_normalize_test.py`'s pattern of testing
result-mapping logic in isolation.
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.app.langgraph_solver import LangGraphSolver, _extract_usage, _make_tools


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
