"""Offline unit tests for `LangGraphSolver`'s pure helpers (`agent/app/langgraph_solver.py`).
Free, no LLM, no network — mirrors `agent/tests/solver_normalize_test.py`'s pattern of testing
result-mapping logic in isolation.
"""
import asyncio
import inspect

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.app.langgraph_solver import (
    LangGraphSolver, _CANDIDATE_COVERAGE_EXTENSION_STEPS, _CANDIDATE_COVERAGE_MAX_EXTENSION_STEPS,
    _STALL_MAX_EPISODES, _STALL_WINDOW, _STEP_EXHAUSTED_TEXT, _TRIM_RECENT_TOOL_MESSAGES,
    _TRIM_TOOL_CHARS, _TRIM_TOTAL_TOOL_CHARS, _candidate_coverage_extension_steps,
    _extract_usage, _finish_answer, _make_tools, _record_io_parity, _trailing_stall_run,
    _context_fit_budget, _has_visitable_url, _make_trim_for_model,
    _trim_for_model, _visit_haystacks,
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


def test_dead_backend_marker_forbids_parametric_answers_and_uncited_urls():
    """Freeing the rephrase-loop turns let the model answer from memory instead of refusing, with
    fabricated citation URLs; the marker has to spell out both refusals."""
    class _DeadBackendIO(_FakeAgentIO):
        async def search(self, query, count=10, timeout_seconds=None):
            self.search_calls.append((query, count))
            return None

    search_tool, _visit_tool = _make_tools(_DeadBackendIO(), search_k=6, page_chars=6000)
    out = asyncio.run(search_tool.ainvoke({"query": "anything"}))
    assert "do NOT answer from prior knowledge" in out
    assert "cannot be determined from available sources" in out
    assert "Cite only URLs you actually visited with the visit tool" in out
    assert out != "No results."


def test_search_marks_only_the_already_visited_result():
    """A LATER, DISTINCT search that happens to surface an already-visited URL must name it
    (never filter it), while the OTHER results stay unmarked. Uses two different queries (not a
    repeat of the same one) so this is isolated from the search-dedup guard tested separately."""
    class _TwoResultIO(_FakeAgentIO):
        async def search(self, query, count=10, timeout_seconds=None):
            self.search_calls.append((query, count))
            return [
                {"title": "Seen", "url": "https://example.com/a", "description": "desc A"},
                {"title": "Fresh", "url": "https://example.com/b", "description": "desc B"},
            ]

    search_tool, visit_tool = _make_tools(_TwoResultIO(), search_k=6, page_chars=6000)
    before = asyncio.run(search_tool.ainvoke({"query": "q1"}))
    assert "ALREADY VISITED" not in before

    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))
    after = asyncio.run(search_tool.ainvoke({"query": "q2"}))
    seen_line = [l for l in after.splitlines() if "example.com/a" in l][0]
    fresh_line = [l for l in after.splitlines() if "example.com/b" in l][0]
    assert seen_line.endswith("[ALREADY VISITED]")
    assert "ALREADY VISITED" not in fresh_line
    # Annotated, not dropped: revisiting can be deliberate.
    assert "https://example.com/a" in after
    assert "desc A" in after


def test_search_dedup_returns_already_searched_message_on_repeat():
    """The mirror-image gap to visit-dedup: nothing previously stopped a repeated identical
    search query from re-running indefinitely."""
    fake_io = _FakeAgentIO()
    search_tool, _visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    first = asyncio.run(search_tool.ainvoke({"query": "Mont Blanc first ascent"}))
    second = asyncio.run(search_tool.ainvoke({"query": "Mont Blanc first ascent"}))
    assert "Result A" in first
    assert second.startswith("ALREADY SEARCHED")
    assert "Mont Blanc first ascent" in second
    assert fake_io.search_calls == [("Mont Blanc first ascent", 6)]  # second call never reached agent_io


def test_search_dedup_normalizes_whitespace_and_case():
    fake_io = _FakeAgentIO()
    search_tool, _visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    asyncio.run(search_tool.ainvoke({"query": "Mont   Blanc"}))
    second = asyncio.run(search_tool.ainvoke({"query": "  mont blanc  "}))
    assert second.startswith("ALREADY SEARCHED")
    assert len(fake_io.search_calls) == 1


def test_search_dedup_does_not_block_distinct_queries():
    """Legitimate fan-out exploration (one query per candidate) must be unaffected."""
    fake_io = _FakeAgentIO()
    search_tool, _visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    first = asyncio.run(search_tool.ainvoke({"query": "Mont Blanc first ascent"}))
    second = asyncio.run(search_tool.ainvoke({"query": "Matterhorn first ascent"}))
    assert "Result A" in first
    assert "Result A" in second
    assert not second.startswith("ALREADY SEARCHED")
    assert len(fake_io.search_calls) == 2


def test_search_dedup_does_not_leak_between_tool_builds():
    fake_io = _FakeAgentIO()
    _s1, _v1 = _make_tools(fake_io, search_k=6, page_chars=6000)
    asyncio.run(_s1.ainvoke({"query": "same query"}))
    s2, _v2 = _make_tools(fake_io, search_k=6, page_chars=6000)
    second = asyncio.run(s2.ainvoke({"query": "same query"}))
    assert not second.startswith("ALREADY SEARCHED")


def test_build_llm_sends_the_configured_seed(monkeypatch):
    """LLM_SEED reaches every other arm through llm_backends; this one builds its own
    ChatOpenAI, so it sampled unseeded — which voids ladder02's own "determinism verified
    byte-identical, so reps=1 suffices" justification for this arm."""
    from agent.app.langgraph_solver import LangGraphSolver

    monkeypatch.setenv("LLM_SEED", "12345")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    llm = LangGraphSolver.__new__(LangGraphSolver)
    llm._model_name = "qwen2.5:1.5b"
    assert LangGraphSolver._build_llm(llm).seed == 12345


def test_build_llm_leaves_sampling_alone_when_no_seed_is_set(monkeypatch):
    """Unset must stay unset: an unseeded run has to behave exactly as it did before."""
    from agent.app.langgraph_solver import LangGraphSolver

    monkeypatch.delenv("LLM_SEED", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    llm = LangGraphSolver.__new__(LangGraphSolver)
    llm._model_name = "qwen2.5:1.5b"
    assert LangGraphSolver._build_llm(llm).seed is None


def test_build_llm_ignores_an_unparsable_seed(monkeypatch):
    from agent.app.langgraph_solver import LangGraphSolver

    monkeypatch.setenv("LLM_SEED", "not-a-number")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    llm = LangGraphSolver.__new__(LangGraphSolver)
    llm._model_name = "qwen2.5:1.5b"
    assert LangGraphSolver._build_llm(llm).seed is None


def test_observation_executed_reports_a_deduped_search_as_not_executed():
    """probe1 (2026-09-03): phi3:mini spent 10 of 12 dispatches on deduped repeats against 2 real
    calls. The refusal carries no TOOL ERROR prefix, so run_tool_loop's sniffing counted it as a
    successful execution and no give-up could bound the loop."""
    from agent.app.langgraph_solver import _observation_executed, _already_searched_message

    assert _observation_executed(_already_searched_message("Mont Blanc")) is False


def test_observation_executed_still_reports_a_tool_error_as_not_executed():
    """Regression guard: returning the explicit (text, executed) tuple DISABLES
    _resolve_dispatch's TOOL_ERROR_PREFIX sniffing, so dropping this half would report a genuine
    dispatch failure as executed and silently undo 0fa6e733."""
    from agent.app.langgraph_solver import _observation_executed
    from agent.app.prompted_tools import TOOL_ERROR_PREFIX

    assert _observation_executed(f"{TOOL_ERROR_PREFIX} ValidationError: bad args") is False


def test_observation_executed_reports_a_real_result_as_executed():
    from agent.app.langgraph_solver import _observation_executed

    assert _observation_executed("1. Mont Blanc - https://example.com/a\n   desc A") is True


def test_repeated_deduped_search_gives_up_instead_of_burning_the_step_budget():
    """Three CONSECUTIVE refusals trip the existing max_tool_errors give-up. Without the
    executed=False signal the loop ran until the step budget was gone."""
    import agent.app.prompted_tools as pt
    from agent.app.langgraph_solver import _observation_executed, _already_searched_message

    dispatched = []

    async def dispatch_tool(action, args):
        dispatched.append(args.get("query"))
        obs = _already_searched_message(args.get("query", ""))
        return obs, _observation_executed(obs)

    async def call_model(_view):
        return '{"thought": "again", "action": "search", "args": {"query": "same"}}', None

    steps = []
    asyncio.run(pt.run_tool_loop(
        tools=[pt.ToolSpec(name="search", description="d", arg_names=("query",))],
        render_view=lambda: "view", call_model=call_model,
        dispatch_tool=dispatch_tool, on_step=steps.append, turns=25,
    ))

    # The loop returns on the give-up rather than raising; what matters is that it is bounded
    # by max_tool_errors (3) and not by the 25-turn step budget.
    assert steps[-1].kind == "tool_error_give_up", [s.kind for s in steps]
    assert len(dispatched) == pt.MAX_TOOL_ERRORS_DEFAULT, dispatched


def test_interleaving_a_real_search_resets_the_give_up_streak():
    """A model that alternates a fresh query with a repeat is exploring, not looping, and must
    not be cut off."""
    import agent.app.prompted_tools as pt
    from agent.app.langgraph_solver import _observation_executed, _already_searched_message

    seen, queries = set(), iter(["a", "a", "b", "b", "c", "c", "d", "d"])

    async def dispatch_tool(action, args):
        q = args.get("query", "")
        if q in seen:
            obs = _already_searched_message(q)
        else:
            seen.add(q)
            obs = f"1. result for {q}"
        return obs, _observation_executed(obs)

    async def call_model(_view):
        try:
            q = next(queries)
        except StopIteration:
            return '{"thought": "done", "action": "finish", "args": {"answer": "x"}}', None
        return ('{"thought": "t", "action": "search", "args": {"query": "%s"}}' % q), None

    steps = []
    asyncio.run(pt.run_tool_loop(
        tools=[pt.ToolSpec(name="search", description="d", arg_names=("query",))],
        render_view=lambda: "view", call_model=call_model,
        dispatch_tool=dispatch_tool, on_step=steps.append, turns=25,
    ))
    assert not any(s.kind == "tool_error_give_up" for s in steps), [s.kind for s in steps]


def test_visit_tool_delegates_to_agent_io_and_truncates():
    fake_io = _FakeAgentIO()
    _search_tool, visit_tool = _make_tools(fake_io, search_k=6, page_chars=5)
    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/page"}))
    assert fake_io.visit_calls == ["https://example.com/page"]
    # The header is never truncated; page_chars still bounds the CONTENT.
    assert out == "SOURCE: https://example.com/page\npage "


def test_visit_tool_reports_no_content():
    class _EmptyVisitIO(_FakeAgentIO):
        async def visit(self, url, timeout_seconds=None):
            self.visit_calls.append(url)
            return ""

    fake_io = _EmptyVisitIO()
    _search_tool, visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/dead"}))
    assert out == "SOURCE: https://example.com/dead\n[No main content found]"


def test_visit_tool_labels_the_source_url():
    """An unattributed blob was live-observed being mistaken for a search result; the URL is the
    only identity `AgentIO.visit` (text-only, no title) can supply."""
    _search_tool, visit_tool = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000)
    out = asyncio.run(visit_tool.ainvoke({"url": "https://en.wikipedia.org/wiki/Parral"}))
    assert out.startswith("SOURCE: https://en.wikipedia.org/wiki/Parral\n")
    assert "page content here" in out
    assert "ALREADY VISITED" not in out


def test_visit_tool_marks_a_repeat_visit_but_still_returns_the_content():
    fake_io = _FakeAgentIO()
    _search_tool, visit_tool = _make_tools(fake_io, search_k=6, page_chars=6000)
    url = "https://en.wikipedia.org/wiki/Parral"
    first = asyncio.run(visit_tool.ainvoke({"url": url}))
    second = asyncio.run(visit_tool.ainvoke({"url": url}))

    assert "ALREADY VISITED THIS URL IN THIS CONVERSATION" not in first
    assert second.startswith("ALREADY VISITED THIS URL IN THIS CONVERSATION")
    assert f"SOURCE: {url}" in second
    assert "page content here" in second  # re-reading can be deliberate; content is not withheld
    assert fake_io.visit_calls == [url, url]


def test_visit_tool_does_not_mark_distinct_urls_as_repeats():
    _search_tool, visit_tool = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000)
    first = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))
    second = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/b"}))
    assert "ALREADY VISITED" not in first
    assert "ALREADY VISITED" not in second
    assert second.startswith("SOURCE: https://example.com/b\n")


def test_visited_urls_do_not_leak_between_tool_builds():
    """`_make_tools` runs once per `solve()`, so the tracking set must be per-run — a shared set
    would mark a fresh benchmark cell's FIRST visit as a repeat."""
    fake_io = _FakeAgentIO()
    url = "https://example.com/a"
    _s1, visit_one = _make_tools(fake_io, search_k=6, page_chars=6000)
    asyncio.run(visit_one.ainvoke({"url": url}))
    _s2, visit_two = _make_tools(fake_io, search_k=6, page_chars=6000)
    assert "ALREADY VISITED" not in asyncio.run(visit_two.ainvoke({"url": url}))


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


def test_offtheshelf_execution_passes_context_budget_env_vars(monkeypatch):
    """Without this wiring the solver's search_k/page_chars were unreachable from a run's env, so
    this arm could not join a context-budget sweep the other arms take part in."""
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_SEARCH_K", "3")
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_PAGE_CHARS", "1234")
    captured = _run_offtheshelf_capturing_kwargs(monkeypatch, None)
    assert captured["search_k"] == 3
    assert captured["page_chars"] == 1234


def test_offtheshelf_execution_falls_back_to_solver_defaults(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_LANGGRAPH_SEARCH_K", raising=False)
    monkeypatch.delenv("IDEA_TEST_LANGGRAPH_PAGE_CHARS", raising=False)
    captured = _run_offtheshelf_capturing_kwargs(monkeypatch, None)
    defaults = inspect.signature(LangGraphSolver.__init__).parameters
    assert captured["search_k"] == defaults["search_k"].default
    assert captured["page_chars"] == defaults["page_chars"].default


def test_offtheshelf_execution_defaults_candidate_coverage_gate_on(monkeypatch):
    """DEFAULT ON as of 2026-08-23 (live-confirmed, n=12, +0.227 mean score, t=2.56, never lost
    a paired cell — see docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md)."""
    monkeypatch.delenv("IDEA_TEST_LANGGRAPH_CANDIDATE_COVERAGE_GATE", raising=False)
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["candidate_coverage_gate"] is True


def test_offtheshelf_execution_candidate_coverage_gate_can_be_opted_out(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_CANDIDATE_COVERAGE_GATE", "0")
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["candidate_coverage_gate"] is False


# --- candidate-coverage gate (built from the 2026-08-23 breadth-pilot fabrication case:
# task 152 rep1 — 42 searches, 0 visits, a fully-cited answer with a wrong keystone fact) ---

def test_visit_haystacks_excludes_search_only_tool_messages():
    """A search-result ToolMessage names a candidate without ever reading its page — the exact
    short-circuit the gate exists to prevent must not count as coverage."""
    messages = [
        ToolMessage(content="1. Mont Blanc — https://en.wikipedia.org/wiki/Mont_Blanc\n   desc",
                    tool_call_id="1"),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nFirst ascent 1786.",
                    tool_call_id="2"),
    ]
    haystacks = _visit_haystacks(messages)
    assert len(haystacks) == 1
    assert haystacks[0].identity == "https://en.wikipedia.org/wiki/Mont_Blanc"
    assert "First ascent 1786" in haystacks[0].body


def test_visit_haystacks_empty_for_no_visits():
    messages = [ToolMessage(content="1. Mont Blanc — https://x/y\n   desc", tool_call_id="1")]
    assert _visit_haystacks(messages) == []


class _SequencedStubGraph:
    """Stand-in for `create_react_agent`'s graph — yields a different fixed message list on each
    successive `astream` call, so the coverage gate's corrective extension pass can be observed
    separately from the run's initial pass."""

    def __init__(self, message_sequence):
        self._sequence = list(message_sequence)
        self.calls = []  # each entry: the `messages` list passed as input
        self.configs = []  # each entry: the `config` dict passed alongside it

    async def astream(self, inputs, config=None, stream_mode=None):
        self.calls.append(list(inputs.get("messages", [])))
        self.configs.append(config)
        idx = min(len(self.calls) - 1, len(self._sequence) - 1)
        yield {"messages": self._sequence[idx]}


def _run_solve_with_sequenced_graph(monkeypatch, message_sequence, mandate, llm=None, **solver_kwargs):
    from agent.app import langgraph_solver

    stub_graph = _SequencedStubGraph(message_sequence)
    llm = llm or _StubLLM()
    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: stub_graph)
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: llm)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", **solver_kwargs,
    )
    result = asyncio.run(solver.solve(mandate, max_steps=4))
    return result, llm, stub_graph


_TWO_CANDIDATE_MANDATE = (
    "For EACH of the following:\n1. Mont Blanc\n2. Matterhorn\n"
    "report the first-ascent year, citing the page you read it from."
)


def _search_only_fabrication_messages():
    """Reproduces task 152 rep1's shape: search results naming both candidates, zero visits, a
    prose final turn that answers anyway."""
    return [
        HumanMessage(content=_TWO_CANDIDATE_MANDATE),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "Mont Blanc"}, "id": "1"}]),
        ToolMessage(content="1. Mont Blanc — https://en.wikipedia.org/wiki/Mont_Blanc\n   desc",
                    tool_call_id="1"),
        AIMessage(content="Mont Blanc: 1786. Matterhorn: 1865."),
    ]


def test_coverage_gate_off_by_default_does_not_alter_a_fabricated_answer(monkeypatch):
    """Default behavior (gate disabled) must be byte-identical to before this change — no second
    `astream` call, no cost/latency change."""
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_search_only_fabrication_messages()], _TWO_CANDIDATE_MANDATE,
    )
    assert result["final_deliverable"] == "Mont Blanc: 1786. Matterhorn: 1865."
    assert len(stub_graph.calls) == 1


def test_coverage_gate_triggers_extension_when_named_candidates_never_visited(monkeypatch):
    extension_messages = _search_only_fabrication_messages() + [
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Mont_Blanc"}, "id": "2"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nMont Blanc first ascent 1786.",
                    tool_call_id="2"),
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Matterhorn"}, "id": "3"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Matterhorn\nMatterhorn first ascent 1865.",
                    tool_call_id="3"),
        AIMessage(content="Mont Blanc: 1786 (visited). Matterhorn: 1865 (visited)."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch,
        [_search_only_fabrication_messages(), extension_messages],
        _TWO_CANDIDATE_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2  # initial pass + one corrective extension, never more
    corrective_turn = stub_graph.calls[1][-1]
    assert isinstance(corrective_turn, HumanMessage)
    assert "Mont Blanc" in corrective_turn.content and "Matterhorn" in corrective_turn.content
    assert result["final_deliverable"] == "Mont Blanc: 1786 (visited). Matterhorn: 1865 (visited)."


# --- scaled extension budget (2026-08-23): two independent full-capture confirmation runs on
# task 152 (7-way fan-out) found the fixed +10-step budget too small — rep1 completed only 6/7
# visits before running out; rep2 (task 156, also 7-way) hit GraphRecursionError INSIDE the
# extension, producing a completely empty final answer despite 4 real visits happening. ---

def test_candidate_coverage_extension_steps_floors_at_the_original_fixed_value():
    assert _candidate_coverage_extension_steps(0) == _CANDIDATE_COVERAGE_EXTENSION_STEPS
    assert _candidate_coverage_extension_steps(1) == _CANDIDATE_COVERAGE_EXTENSION_STEPS
    assert _candidate_coverage_extension_steps(2) == _CANDIDATE_COVERAGE_EXTENSION_STEPS


def test_candidate_coverage_extension_steps_scales_for_wide_fan_outs():
    # 7 missing (the exact live-observed failure case) must exceed the old fixed value.
    seven = _candidate_coverage_extension_steps(7)
    assert seven > _CANDIDATE_COVERAGE_EXTENSION_STEPS
    five = _candidate_coverage_extension_steps(5)
    assert _CANDIDATE_COVERAGE_EXTENSION_STEPS <= five <= seven  # monotonic in missing count


def test_candidate_coverage_extension_steps_is_capped():
    assert _candidate_coverage_extension_steps(1000) == _CANDIDATE_COVERAGE_MAX_EXTENSION_STEPS


def _seven_candidate_mandate():
    listing = "\n".join(f"{i}. Mountain{i}" for i in range(1, 8))
    return f"For EACH of the following seven mountains:\n{listing}\nreport the first-ascent year."


def test_coverage_gate_scales_recursion_limit_for_a_wide_fan_out(monkeypatch):
    """The recursion_limit passed to the extension's astream call must reflect the SCALED
    budget for a 7-way gap, not the old fixed +10 (recursion_limit=20) that live-crashed."""
    mandate = _seven_candidate_mandate()
    no_visits = [HumanMessage(content=mandate), AIMessage(content="I don't have enough info.")]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [no_visits, no_visits], mandate, candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2
    extension_config = stub_graph.configs[1]
    expected_limit = max(4, _candidate_coverage_extension_steps(7) * 2)
    assert extension_config["recursion_limit"] == expected_limit
    assert expected_limit > 20  # strictly more headroom than the old fixed budget gave


def test_coverage_gate_does_not_fire_when_coverage_is_already_satisfied(monkeypatch):
    satisfied_messages = [
        HumanMessage(content=_TWO_CANDIDATE_MANDATE),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nMont Blanc first ascent 1786.",
                    tool_call_id="1"),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Matterhorn\nMatterhorn first ascent 1865.",
                    tool_call_id="2"),
        AIMessage(content="Mont Blanc: 1786. Matterhorn: 1865."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [satisfied_messages], _TWO_CANDIDATE_MANDATE, candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 1  # already satisfied -> no extension call spent
    assert result["final_deliverable"] == "Mont Blanc: 1786. Matterhorn: 1865."


def test_coverage_gate_is_a_noop_when_mandate_names_no_candidates(monkeypatch):
    """Fails OPEN, matching `extract_named_candidates`'s own contract — a prose mandate must
    never gain an extension call it has no roster to check."""
    prose_mandate = "What is the capital of France?"
    messages = [HumanMessage(content=prose_mandate), AIMessage(content="Paris.")]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [messages], prose_mandate, candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 1
    assert result["final_deliverable"] == "Paris."


def test_offtheshelf_execution_defaults_context_trim_on(monkeypatch):
    """DEFAULT ON as of 2026-08-23 (live-confirmed, n=12, both conditions with
    candidate_coverage_gate=1: +0.216 mean score, t=2.23, W/T/L 6/3/3 — see
    docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md)."""
    monkeypatch.delenv("IDEA_TEST_LANGGRAPH_CONTEXT_TRIM", raising=False)
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["context_trim"] is True


def test_offtheshelf_execution_context_trim_can_be_opted_out(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_CONTEXT_TRIM", "0")
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["context_trim"] is False


# --- context-trim pre_model_hook (built from the 2026-08-23 evidence: task 152, all 7 pages
# visited via the coverage gate, still only surfaced 3/7 facts — 316,990 chars of raw visited-page
# text with zero compaction, vs sequential_react's bounded ~18,000-char working context) ---

def _tool_msgs(*contents):
    return [ToolMessage(content=c, tool_call_id=str(i)) for i, c in enumerate(contents)]


def test_trim_for_model_leaves_recent_tool_messages_uncapped():
    big = "x" * 6000
    messages = [HumanMessage(content="task")] + _tool_msgs(big, big, big, big)
    out = _trim_for_model({"messages": messages})["llm_input_messages"]
    tool_out = [m for m in out if isinstance(m, ToolMessage)]
    assert len(tool_out) == 4
    # last _TRIM_RECENT_TOOL_MESSAGES pass through unclipped; the rest get clipped
    for m in tool_out[-_TRIM_RECENT_TOOL_MESSAGES:]:
        assert len(m.content) == 6000
    for m in tool_out[:-_TRIM_RECENT_TOOL_MESSAGES]:
        assert len(m.content) == _TRIM_TOOL_CHARS


def test_trim_for_model_enforces_total_tool_char_budget():
    # Many OLDER tool messages (excludes the protected recent window), each under the
    # per-message cap but summing well over budget among themselves.
    contents = ["y" * 1000] * 30
    messages = [HumanMessage(content="task")] + _tool_msgs(*contents)
    out = _trim_for_model({"messages": messages})["llm_input_messages"]
    tool_out = [m for m in out if isinstance(m, ToolMessage)]
    older_total = sum(len(m.content) for m in tool_out[:-_TRIM_RECENT_TOOL_MESSAGES])
    assert older_total <= _TRIM_TOTAL_TOOL_CHARS
    assert len(tool_out) < 30  # oldest ones were dropped entirely


def test_trim_for_model_drops_oldest_first_when_over_budget():
    contents = [f"msg{i}-" + "z" * 1000 for i in range(25)]
    messages = [HumanMessage(content="task")] + _tool_msgs(*contents)
    out = _trim_for_model({"messages": messages})["llm_input_messages"]
    surviving = [m.content for m in out if isinstance(m, ToolMessage)]
    assert not any(c.startswith("msg0-") for c in surviving)   # oldest dropped
    assert any(c.startswith("msg24-") for c in surviving)      # newest (protected) kept


def _valid_tool_call_ids(messages):
    """Every ToolMessage's tool_call_id must appear in SOME AIMessage's tool_calls, and every
    AIMessage tool_call id must have a matching ToolMessage — the LangGraph/provider invariant
    this fix exists to preserve."""
    ai_call_ids = {
        tc.get("id") for m in messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])
    }
    tool_result_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return ai_call_ids == tool_result_ids


def test_trim_for_model_dropping_a_tool_message_also_strips_its_ai_tool_call():
    """Regression test for a real live failure: dropping a ToolMessage without also removing the
    matching tool_call from its requesting AIMessage produced an invalid chat history
    (`ValueError: Found AIMessages with tool_calls that do not have a corresponding ToolMessage`)
    — an infra failure on a real benchmark cell (task 157, trim_ab_20260823_trimon_rep2)."""
    old_ai = AIMessage(content="", tool_calls=[
        {"name": "search", "args": {"query": "a"}, "id": "call_a"},
        {"name": "search", "args": {"query": "b"}, "id": "call_b"},
    ])
    old_tool_a = ToolMessage(content="z" * 1000, tool_call_id="call_a")
    old_tool_b = ToolMessage(content="z" * 1000, tool_call_id="call_b")
    # Enough additional older tool messages to force the total-budget drop path.
    filler_ai = [AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": f"fill{i}"}])
                 for i in range(20)]
    filler_tool = [ToolMessage(content="z" * 1000, tool_call_id=f"fill{i}") for i in range(20)]
    interleaved_filler = [m for pair in zip(filler_ai, filler_tool) for m in pair]
    messages = [HumanMessage(content="task"), old_ai, old_tool_a, old_tool_b] + interleaved_filler

    out = _trim_for_model({"messages": messages})["llm_input_messages"]
    assert _valid_tool_call_ids(out)  # the invariant that broke live


def test_trim_for_model_drops_an_ai_message_left_with_no_tool_calls_and_no_content():
    old_ai = AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "call_a"}])
    old_tool = ToolMessage(content="z" * 1000, tool_call_id="call_a")
    filler_ai = [AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": f"fill{i}"}])
                 for i in range(25)]
    filler_tool = [ToolMessage(content="z" * 1000, tool_call_id=f"fill{i}") for i in range(25)]
    interleaved_filler = [m for pair in zip(filler_ai, filler_tool) for m in pair]
    messages = [HumanMessage(content="task"), old_ai, old_tool] + interleaved_filler

    out = _trim_for_model({"messages": messages})["llm_input_messages"]
    assert _valid_tool_call_ids(out)
    # The now-empty old_ai (its only tool_call dropped, no prose content) must be gone entirely.
    assert old_ai not in out


def test_trim_for_model_never_mutates_state_messages():
    original_content = "x" * 6000
    messages = [HumanMessage(content="task")] + _tool_msgs(original_content, original_content,
                                                             original_content, original_content)
    state = {"messages": messages}
    _trim_for_model(state)
    # The ORIGINAL objects/content passed in must be untouched.
    for m in state["messages"]:
        if isinstance(m, ToolMessage):
            assert m.content == original_content
    assert state["messages"] is messages  # the list itself was never reassigned


def test_trim_for_model_preserves_system_and_human_messages():
    big = "x" * 50000
    messages = [SystemMessage(content=big), HumanMessage(content=big)] + _tool_msgs("y" * 1000)
    out = _trim_for_model({"messages": messages})["llm_input_messages"]
    assert out[0].content == big
    assert out[1].content == big


def _run_solve_capturing_create_react_agent_kwargs(monkeypatch, **solver_kwargs):
    from agent.app import langgraph_solver

    captured = {}

    class _EmptyStubGraph:
        async def astream(self, _inputs, config=None, stream_mode=None):
            yield {"messages": [HumanMessage(content="task"), AIMessage(content="answer")]}

    def _capturing_create_react_agent(*args, **kwargs):
        captured.update(kwargs)
        return _EmptyStubGraph()

    monkeypatch.setattr(langgraph_solver, "create_react_agent", _capturing_create_react_agent)
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: _StubLLM())
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", **solver_kwargs,
    )
    asyncio.run(solver.solve("the task", max_steps=4))
    return captured


def test_context_trim_off_by_default_passes_no_pre_model_hook(monkeypatch):
    captured = _run_solve_capturing_create_react_agent_kwargs(monkeypatch)
    assert captured["pre_model_hook"] is None


def test_context_trim_on_wires_pre_model_hook(monkeypatch):
    captured = _run_solve_capturing_create_react_agent_kwargs(monkeypatch, context_trim=True)
    assert captured["pre_model_hook"] is _trim_for_model


def test_context_trim_does_not_interfere_with_coverage_gate(monkeypatch):
    """`solve()` itself must never trim `messages` before scanning it — `_trim_for_model` only
    ever reaches the model via `pre_model_hook` (untested by these message-replaying stub graphs,
    which is exactly the point: the coverage gate's own bookkeeping must be unaffected by
    context_trim regardless of what the model was shown)."""
    extension_messages = _search_only_fabrication_messages() + [
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Mont_Blanc"}, "id": "2"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nMont Blanc first ascent 1786.",
                    tool_call_id="2"),
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Matterhorn"}, "id": "3"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Matterhorn\nMatterhorn first ascent 1865.",
                    tool_call_id="3"),
        AIMessage(content="Mont Blanc: 1786 (visited). Matterhorn: 1865 (visited)."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch,
        [_search_only_fabrication_messages(), extension_messages],
        _TWO_CANDIDATE_MANDATE,
        candidate_coverage_gate=True,
        context_trim=True,
    )
    assert len(stub_graph.calls) == 2
    assert result["final_deliverable"] == "Mont Blanc: 1786 (visited). Matterhorn: 1865 (visited)."


def test_coverage_gate_extension_still_missing_falls_through_to_forced_synthesis(monkeypatch):
    """If the corrective extension ALSO ends without visiting everything, the run must not just
    accept the (still under-grounded) answer silently — forced synthesis (using whatever evidence
    now exists, including anything gained during the extension) is the existing safety net."""
    still_missing_messages = _search_only_fabrication_messages() + [
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Mont_Blanc"}, "id": "2"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nMont Blanc first ascent 1786.",
                    tool_call_id="2"),
        AIMessage(content=_STEP_EXHAUSTED_TEXT),
    ]
    result, llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch,
        [_search_only_fabrication_messages(), still_missing_messages],
        _TWO_CANDIDATE_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2
    assert result["final_deliverable"] == "synthesized answer"
    assert len(llm.calls) == 1  # the forced-synthesis pass, using the extension's partial evidence
    assert "Mont Blanc first ascent 1786" in llm.calls[0][-1].content


# --- stall-recovery gate (neither arm has ANY dead-end detection today — a 2026-08-23
# capability survey found sequential_react and langgraph_react both rely entirely on the model's
# own judgment plus the raw step budget; the native engine's real backtrack machinery is tied to
# its scored DAG structure and not portable to this arm's linear message list) ---

def test_offtheshelf_execution_passes_stall_recovery_gate_env_var(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_STALL_RECOVERY_GATE", "1")
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["stall_recovery_gate"] is True


def test_offtheshelf_execution_leaves_stall_recovery_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_LANGGRAPH_STALL_RECOVERY_GATE", raising=False)
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["stall_recovery_gate"] is False


def test_trailing_stall_run_counts_consecutive_non_progress_tool_messages():
    messages = [
        ToolMessage(content="1. Mont Blanc — https://x/y\n   desc", tool_call_id="1"),  # good
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "q2"}, "id": "2"}]),
        ToolMessage(content="No results.", tool_call_id="2"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "q3"}, "id": "3"}]),
        ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="3"),
    ]
    assert _trailing_stall_run(messages) == 2


def test_trailing_stall_run_stops_at_the_first_good_result_from_the_end():
    messages = [
        ToolMessage(content="No results.", tool_call_id="1"),
        ToolMessage(content="SOURCE: https://x/y\ngood content", tool_call_id="2"),  # breaks the run
        ToolMessage(content="No results.", tool_call_id="3"),
    ]
    assert _trailing_stall_run(messages) == 1


def test_trailing_stall_run_zero_when_no_tool_messages():
    assert _trailing_stall_run([HumanMessage(content="task"), AIMessage(content="thinking")]) == 0


def test_stall_recovery_triggers_after_threshold_non_progress_results(monkeypatch):
    stalled_messages = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="No results.", tool_call_id="1"),
        ToolMessage(content="No results.", tool_call_id="2"),
        ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="3"),
    ]
    assert _trailing_stall_run(stalled_messages) == _STALL_WINDOW  # sanity: fixture matches threshold
    recovered_messages = stalled_messages + [
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/X\nthe answer is 42", tool_call_id="4"),
        AIMessage(content="The answer is 42."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [stalled_messages, recovered_messages], "find the fact",
        stall_recovery_gate=True,
    )
    assert len(stub_graph.calls) == 2
    corrective_turn = stub_graph.calls[1][-1]
    assert isinstance(corrective_turn, HumanMessage)
    assert "no progress" in corrective_turn.content.lower()
    assert result["final_deliverable"] == "The answer is 42."


def test_stall_recovery_off_by_default_does_not_trigger(monkeypatch):
    stalled_messages = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="No results.", tool_call_id="1"),
        ToolMessage(content="No results.", tool_call_id="2"),
        ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="3"),
        AIMessage(content="I could not find it."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [stalled_messages], "find the fact",
    )
    assert len(stub_graph.calls) == 1
    assert result["final_deliverable"] == "I could not find it."


def test_stall_recovery_does_not_trigger_below_threshold(monkeypatch):
    almost_stalled = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="No results.", tool_call_id="1"),
        ToolMessage(content="SOURCE: https://x/y\nsome content", tool_call_id="2"),  # breaks the run
        AIMessage(content="Based on the content, the answer is 42."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [almost_stalled], "find the fact", stall_recovery_gate=True,
    )
    assert len(stub_graph.calls) == 1


def test_stall_recovery_bounded_by_max_episodes(monkeypatch):
    """A persistently-stuck run must fall through after `_STALL_MAX_EPISODES`, never loop
    indefinitely."""
    still_stalled = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="No results.", tool_call_id="1"),
        ToolMessage(content="No results.", tool_call_id="2"),
        ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="3"),
    ]
    result, llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch,
        [still_stalled] * (_STALL_MAX_EPISODES + 3),  # keeps returning the same stalled tail
        "find the fact",
        stall_recovery_gate=True,
    )
    assert len(stub_graph.calls) == 1 + _STALL_MAX_EPISODES  # initial + capped corrective passes, never more
    # No clean prose answer ever appeared -> falls through to the existing forced-synthesis
    # safety net (using whatever "evidence" — here, just the error strings — is on hand).
    assert result["final_deliverable"] == "synthesized answer"
    assert len(llm.calls) == 1


def test_stall_recovery_runs_before_coverage_gate(monkeypatch):
    """A run that's both stalled AND missing a named candidate should recover from the stall
    first, then (if still missing coverage) get the coverage-gate's own corrective pass."""
    stalled_messages = [
        HumanMessage(content=_TWO_CANDIDATE_MANDATE),
        ToolMessage(content="No results.", tool_call_id="1"),
        ToolMessage(content="No results.", tool_call_id="2"),
        ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="3"),
    ]
    after_stall_recovery = stalled_messages + [
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Mont_Blanc"}, "id": "4"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nMont Blanc first ascent 1786.",
                    tool_call_id="4"),
        AIMessage(content="Mont Blanc: 1786."),  # still missing Matterhorn
    ]
    after_coverage_gate = after_stall_recovery + [
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Matterhorn"}, "id": "5"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Matterhorn\nMatterhorn first ascent 1865.",
                    tool_call_id="5"),
        AIMessage(content="Mont Blanc: 1786. Matterhorn: 1865."),
    ]
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch,
        [stalled_messages, after_stall_recovery, after_coverage_gate],
        _TWO_CANDIDATE_MANDATE,
        stall_recovery_gate=True,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 3
    assert result["final_deliverable"] == "Mont Blanc: 1786. Matterhorn: 1865."


# --- characterization tests (pre-refactor, 2026-08-24): pin down two behaviors that are only
# monotonic today by accident of the flat function's shared local variables, and that a
# state-threading extraction (`_run_extension`/`_SolveState`) could silently break with zero
# other test failures. Written and confirmed green BEFORE the extraction. ---

class _RecursionThenCleanGraph:
    """First `astream` call yields a partial state and then raises `GraphRecursionError` —
    exactly like the primary run hitting its step budget mid-stream. Every subsequent call
    (the gate extension) behaves like a normal, error-free pass with no exception at all."""

    def __init__(self, partial_messages, clean_messages):
        self._partial = partial_messages
        self._clean = clean_messages
        self.calls = []
        self.configs = []

    async def astream(self, inputs, config=None, stream_mode=None):
        self.calls.append(list(inputs.get("messages", [])))
        self.configs.append(config)
        if len(self.calls) == 1:
            yield {"messages": self._partial}
            raise GraphRecursionError("primary run hit its step budget")
        yield {"messages": self._clean}


def _run_solve_with_recursion_then_clean(monkeypatch, partial_messages, clean_messages, mandate, **solver_kwargs):
    from agent.app import langgraph_solver

    stub_graph = _RecursionThenCleanGraph(partial_messages, clean_messages)
    llm = _StubLLM()
    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: stub_graph)
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: llm)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", **solver_kwargs,
    )
    result = asyncio.run(solver.solve(mandate, max_steps=4))
    return result, llm, stub_graph


def test_recursion_hit_and_run_error_stay_sticky_through_a_clean_extension(monkeypatch):
    """CHARACTERIZATION (pre-refactor): the primary run hits `GraphRecursionError`, and the
    candidate-coverage extension that follows completes CLEANLY (no exception of its own, a real
    prose answer). `recursion_hit`/`run_error` must still be reported — they are set once by the
    primary run's except-block and NEVER cleared by anything downstream. `solve` only ever sets
    these flags, never resets them; a refactor that threaded a fresh per-extension state object
    (instead of mutating the same locals throughout) could silently drop this with no other test
    catching it, since every other coverage-gate test uses a graph that never raises at all."""
    partial_messages = [
        HumanMessage(content=_TWO_CANDIDATE_MANDATE),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "Mont Blanc"}, "id": "1"}],
                  usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        ToolMessage(content="1. Mont Blanc — https://en.wikipedia.org/wiki/Mont_Blanc\n   desc",
                    tool_call_id="1"),
    ]
    clean_extension_messages = partial_messages + [
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Mont_Blanc"}, "id": "2"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Mont_Blanc\nMont Blanc first ascent 1786.",
                    tool_call_id="2"),
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Matterhorn"}, "id": "3"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/Matterhorn\nMatterhorn first ascent 1865.",
                    tool_call_id="3"),
        AIMessage(content="Mont Blanc: 1786 (visited). Matterhorn: 1865 (visited)."),
    ]
    result, llm, stub_graph = _run_solve_with_recursion_then_clean(
        monkeypatch, partial_messages, clean_extension_messages, _TWO_CANDIDATE_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2  # primary (crashed) + one clean coverage extension
    assert result["final_deliverable"] == "Mont Blanc: 1786 (visited). Matterhorn: 1865 (visited)."
    # Sticky: the warning still names the step-budget outcome despite the clean extension.
    assert "step budget" in result.get("warning", "")
    assert llm.calls == []  # extension produced a real prose answer -> forced synthesis never ran


def test_usages_are_recomputed_not_accumulated_across_stall_episodes(monkeypatch):
    """CHARACTERIZATION (pre-refactor): `usages = _extract_usage(messages)` REPLACES the prior
    value after every extension pass; it is never `.extend()`-ed episode-over-episode (only the
    final forced-synthesis usage is ever `.extend()`-ed, once, at the very end). Because each
    stall-recovery episode's returned message list already contains every earlier turn (the stub,
    like the real graph, threads full history), a naive accumulate-refactor
    (`usages.extend(_extract_usage(messages))` inside the loop) would double- and triple-count
    the same AIMessage turns across episodes. This drives two full stall-recovery episodes (the
    max, `_STALL_MAX_EPISODES == 2`) with a distinct `usage_metadata` on every AI turn and pins
    the exact final count, which is only correct under recompute-not-accumulate semantics."""
    mandate = "find the fact"

    def _ai(call_id, usage_i):
        return AIMessage(
            content="", tool_calls=[{"name": "search", "args": {"query": call_id}, "id": call_id}],
            usage_metadata={"input_tokens": usage_i, "output_tokens": usage_i, "total_tokens": usage_i * 2},
        )

    primary_messages = [
        HumanMessage(content=mandate),
        _ai("1", 1), ToolMessage(content="No results.", tool_call_id="1"),
        _ai("2", 2), ToolMessage(content="No results.", tool_call_id="2"),
        _ai("3", 3), ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="3"),
    ]
    assert _trailing_stall_run(primary_messages) == _STALL_WINDOW  # sanity: triggers episode 1

    episode_1_messages = primary_messages + [
        _ai("4", 4), ToolMessage(content="No results.", tool_call_id="4"),
        _ai("5", 5), ToolMessage(content="No results.", tool_call_id="5"),
        _ai("6", 6), ToolMessage(content="SEARCH ERROR: timeout", tool_call_id="6"),
    ]
    assert _trailing_stall_run(episode_1_messages) >= _STALL_WINDOW  # sanity: triggers episode 2 (the max)

    episode_2_messages = episode_1_messages + [
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/X\nthe answer is 42", tool_call_id="7"),
        AIMessage(content="The answer is 42.",
                  usage_metadata={"input_tokens": 7, "output_tokens": 7, "total_tokens": 14}),
    ]

    result, llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [primary_messages, episode_1_messages, episode_2_messages], mandate,
        stall_recovery_gate=True,
    )
    assert len(stub_graph.calls) == 3  # primary + 2 stall-recovery episodes (the configured max)
    assert result["final_deliverable"] == "The answer is 42."
    # 7 AI turns carry usage_metadata across the FINAL full message list (1..6 from the stall
    # episodes, +1 from the recovered final turn) -> recompute-from-scratch must land on exactly
    # 7, never double-counted (e.g. 3+6+7=16) by an accumulate-refactor across the 3 astream calls.
    assert result["llm_calls"] == 7
    assert llm.calls == []  # a real prose answer landed -> forced synthesis never ran, no +1


# --- require_finish_tool (2026-08-23): imitates sequential_react's explicit finish(answer)
# action. sequential_react never suffers the "narration accepted as final answer" bug because
# the model must deliberately choose to submit; create_react_agent has no native equivalent
# (any tool-call-free turn ends the run). Built from task 156 rep1: 5 real dam visits sitting
# unused in the message history while an unfinished "Let's start with the Vajont Dam..."
# narration scored 0.0 on every check. Distinct from always_synthesize (which sometimes
# rewrites an already-good answer WORSE, live-observed on task 157) — a real finish() call's
# text is used verbatim, never rewritten. The mechanism below is unit-tested and sound in
# isolation; a live A/B (2026-08-23) found adding the extra tool measurably hurt step-
# constrained tasks by reducing step efficiency (mean -0.44 on a 7-way task) — net negative
# as currently scoped, stays opt-in. See docs/handoffs/BREADTH_SUITE_WEAKNESS_SWEEP_20260823.md. ---

def test_finish_answer_returns_none_when_never_called():
    messages = [HumanMessage(content="task"), AIMessage(content="Here is my answer: 42.")]
    assert _finish_answer(messages) is None


def test_finish_answer_extracts_the_argument():
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "finish", "args": {"answer": "The answer is 42."}, "id": "1"}]),
    ]
    assert _finish_answer(messages) == "The answer is 42."


def test_finish_answer_ignores_other_tool_calls():
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "q"}, "id": "1"}]),
        ToolMessage(content="results", tool_call_id="1"),
    ]
    assert _finish_answer(messages) is None


def test_finish_answer_uses_the_last_call_when_called_more_than_once():
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "finish", "args": {"answer": "draft answer"}, "id": "1"}]),
        ToolMessage(content="Answer submitted.", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "finish", "args": {"answer": "final corrected answer"}, "id": "2"}]),
    ]
    assert _finish_answer(messages) == "final corrected answer"


def test_finish_answer_ignores_an_empty_answer_argument():
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "finish", "args": {"answer": "   "}, "id": "1"}]),
    ]
    assert _finish_answer(messages) is None


def test_make_tools_includes_finish_only_when_required():
    fake_io = _FakeAgentIO()
    default_tools = _make_tools(fake_io, search_k=6, page_chars=6000)
    assert [t.name for t in default_tools] == ["search", "visit"]

    with_finish = _make_tools(fake_io, search_k=6, page_chars=6000, require_finish_tool=True)
    assert [t.name for t in with_finish] == ["search", "visit", "finish"]


def test_offtheshelf_execution_passes_require_finish_tool_env_var(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_REQUIRE_FINISH_TOOL", "1")
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["require_finish_tool"] is True


def test_offtheshelf_execution_leaves_require_finish_tool_off_by_default(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_LANGGRAPH_REQUIRE_FINISH_TOOL", raising=False)
    assert _run_offtheshelf_capturing_kwargs(monkeypatch, None)["require_finish_tool"] is False


def test_require_finish_tool_uses_the_finish_call_verbatim(monkeypatch):
    messages = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/X\nthe answer is 42", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "finish", "args": {"answer": "The answer is 42, cited from X."}, "id": "2"}]),
    ]
    result, llm, _stub = _run_solve_with_sequenced_graph(
        monkeypatch, [messages], "find the fact", require_finish_tool=True,
    )
    assert result["final_deliverable"] == "The answer is 42, cited from X."
    assert llm.calls == []  # verbatim — no synthesis rewrite, unlike always_synthesize


def test_require_finish_tool_discards_an_unfinished_narration(monkeypatch):
    """The exact task 156 rep1 shape: real evidence gathered, but the run ends on a bare
    tool-call-free narration turn that never called finish — must NOT be trusted as the answer."""
    messages = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/X\nthe answer is 42", tool_call_id="1"),
        AIMessage(content="Let's continue with the next item."),  # no finish call -> not trusted
    ]
    result, llm, _stub = _run_solve_with_sequenced_graph(
        monkeypatch, [messages], "find the fact", require_finish_tool=True,
    )
    assert result["final_deliverable"] == "synthesized answer"  # forced-synthesis safety net
    assert len(llm.calls) == 1
    assert "the answer is 42" in llm.calls[0][-1].content  # synthesis saw the real evidence


def test_require_finish_tool_off_by_default_trusts_natural_termination(monkeypatch):
    """Confirms the flag actually gates the behavior change — default (off) is unaffected."""
    messages = [
        HumanMessage(content="find the fact"),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/X\nthe answer is 42", tool_call_id="1"),
        AIMessage(content="The answer is 42."),
    ]
    result, llm, _stub = _run_solve_with_sequenced_graph(monkeypatch, [messages], "find the fact")
    assert result["final_deliverable"] == "The answer is 42."
    assert llm.calls == []


# --- FIX A: context-fit trim budget (docs/TINY_MODEL_INVESTIGATION.md §3.1) ---------------
# gemma2:2b's served window is 8192 tokens; the fixed global trim budget (3 unclipped recent
# tool messages at up to page_chars=6000, plus 18,000 chars of clipped older ones) can put
# ~36,000 chars of tool payload alone in front of a model whose ENTIRE window is ~32,000 chars.
# Ollama truncates at the HEAD, so what gets dropped is the system message carrying the whole
# tool protocol — measured: keystone 0/11 in cells that hit the cap, 8/13 in cells that did not.

def test_context_fit_budget_is_a_no_op_for_a_32k_model():
    """qwen2.5:7b (32768) must see EXACTLY today's constants — the no-harm requirement."""
    budget = _context_fit_budget(32768, page_chars=6000)
    assert budget.recent_tool_messages == _TRIM_RECENT_TOOL_MESSAGES
    assert budget.tool_chars == _TRIM_TOOL_CHARS
    assert budget.total_tool_chars == _TRIM_TOTAL_TOOL_CHARS
    assert budget.page_chars == 6000


def test_context_fit_budget_never_exceeds_the_configured_page_chars():
    assert _context_fit_budget(131072, page_chars=6000).page_chars == 6000
    assert _context_fit_budget(131072, page_chars=2000).page_chars == 2000


def test_context_fit_budget_shrinks_every_dimension_for_an_8k_model():
    budget = _context_fit_budget(8192, page_chars=6000)
    assert budget.recent_tool_messages < _TRIM_RECENT_TOOL_MESSAGES
    assert budget.tool_chars < _TRIM_TOOL_CHARS
    assert budget.total_tool_chars < _TRIM_TOTAL_TOOL_CHARS
    assert budget.page_chars < 6000


def test_context_fit_budget_worst_case_tool_payload_fits_the_window():
    """The property that actually matters: the WORST-CASE tool payload the trim can emit
    (unclipped recent window + the older-message budget) must leave room for the system
    message inside the model's real window, at ~4 chars/token."""
    for ctx in (2048, 4096, 8192, 16384, 32768):
        budget = _context_fit_budget(ctx, page_chars=6000)
        worst_case = budget.recent_tool_messages * budget.page_chars + budget.total_tool_chars
        assert worst_case <= ctx * 2, (ctx, worst_case)


def test_make_trim_for_model_applies_the_scaled_budget():
    budget = _context_fit_budget(8192, page_chars=2500)
    trim = _make_trim_for_model(budget)
    big = "x" * 5000
    messages = [HumanMessage(content="task")] + _tool_msgs(big, big, big, big, big)
    out = trim({"messages": messages})["llm_input_messages"]
    tool_out = [m for m in out if isinstance(m, ToolMessage)]
    older = tool_out[:-budget.recent_tool_messages]
    assert all(len(m.content) <= budget.tool_chars for m in older)
    assert sum(len(m.content) for m in older) <= budget.total_tool_chars
    # the protected recent window is still unclipped
    assert all(len(m.content) == 5000 for m in tool_out[-budget.recent_tool_messages:])


def _run_solve_with_served_window(monkeypatch, window, **solver_kwargs):
    """Run `solve()` against the sequenced stub graph with the model's served context window
    forced to `window` (None = unknown, i.e. a hosted model), capturing what `_make_tools` and
    `create_react_agent` were given."""
    from agent.app import langgraph_solver

    captured = {}
    real_make_tools = langgraph_solver._make_tools

    def _capturing_make_tools(agent_io, search_k, page_chars, *a, **k):
        captured["page_chars"] = page_chars
        return real_make_tools(agent_io, search_k, page_chars, *a, **k)

    async def _fake_window(self):
        return window

    monkeypatch.setattr(langgraph_solver, "_make_tools", _capturing_make_tools)
    monkeypatch.setattr(LangGraphSolver, "_served_context_tokens", _fake_window)

    stub_graph = _SequencedStubGraph([[HumanMessage(content="task"), AIMessage(content="answer")]])

    def _capturing_create_react_agent(*a, **k):
        captured.update(k)
        return stub_graph

    monkeypatch.setattr(langgraph_solver, "create_react_agent", _capturing_create_react_agent)
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: _StubLLM())
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="gemma2:2b", **solver_kwargs,
    )
    asyncio.run(solver.solve("the task", max_steps=4))
    return captured


def test_context_fit_scales_the_hook_and_page_chars_for_a_small_window(monkeypatch):
    captured = _run_solve_with_served_window(monkeypatch, 8192, context_trim=True)
    assert captured["page_chars"] < 6000
    hook = captured["pre_model_hook"]
    assert hook is not _trim_for_model  # a budget-scaled closure, not the fixed global one
    big = "x" * 6000
    out = hook({"messages": _tool_msgs(big, big, big, big)})["llm_input_messages"]
    assert len(out[0].content) < _TRIM_TOOL_CHARS


def test_context_fit_is_inert_when_the_served_window_is_unknown(monkeypatch):
    """A hosted model (no /api/show) must keep today's exact hook and page_chars."""
    captured = _run_solve_with_served_window(monkeypatch, None, context_trim=True)
    assert captured["page_chars"] == 6000
    assert captured["pre_model_hook"] is _trim_for_model


def test_context_fit_is_inert_for_a_large_window(monkeypatch):
    captured = _run_solve_with_served_window(monkeypatch, 32768, context_trim=True)
    assert captured["page_chars"] == 6000
    assert captured["pre_model_hook"] is _trim_for_model


def test_context_fit_env_flag_off_keeps_the_fixed_global_budget(monkeypatch):
    monkeypatch.setenv("LEDGER_CONTEXT_FIT", "0")
    captured = _run_solve_with_served_window(monkeypatch, 8192, context_trim=True)
    assert captured["page_chars"] == 6000
    assert captured["pre_model_hook"] is _trim_for_model


def test_context_fit_does_nothing_when_context_trim_is_off(monkeypatch):
    captured = _run_solve_with_served_window(monkeypatch, 8192)
    assert captured["page_chars"] == 6000
    assert captured["pre_model_hook"] is None


def test_context_fit_never_probes_when_context_trim_is_off(monkeypatch):
    """No extra /api/show round trip on a run that isn't trimming."""
    from agent.app import langgraph_solver

    calls = []

    async def _counting_metadata(connector_llm, model_name):
        calls.append(model_name)
        return {}

    monkeypatch.setattr(langgraph_solver, "collect_model_metadata", _counting_metadata)
    monkeypatch.setattr(langgraph_solver, "create_react_agent",
                        lambda *a, **k: _SequencedStubGraph([[AIMessage(content="a")]]))
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: _StubLLM())
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="gemma2:2b",
    )
    asyncio.run(solver.solve("the task", max_steps=4))
    assert calls == []


def test_served_context_tokens_survives_a_failed_probe(monkeypatch):
    from agent.app import langgraph_solver

    async def _boom(connector_llm, model_name):
        raise RuntimeError("probe down")

    monkeypatch.setattr(langgraph_solver, "collect_model_metadata", _boom)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="gemma2:2b",
    )
    assert asyncio.run(solver._served_context_tokens()) is None


# --- FIX B: zero-visit finish gate (docs/TINY_MODEL_INVESTIGATION.md §3.2) ----------------
# phi3:mini emits 98% valid JSON, searches correctly, gets the right page at rank 1 — and then
# answers straight from the search snippet, never calling `visit`. Three of the four validators
# on this task family require a real fetch, so such a run scores exactly 0.000 by construction.
# The existing coverage gate cannot catch it: `extract_named_candidates` returns [] here.

_UNNAMED_MANDATE = "Report the height of the tallest chimney, citing the page you read it from."


def _zero_visit_messages(tool_content="1. GRES-2 — https://en.wikipedia.org/wiki/GRES-2\n   215 m"):
    return [
        HumanMessage(content=_UNNAMED_MANDATE),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": "GRES-2"}, "id": "1"}]),
        ToolMessage(content=tool_content, tool_call_id="1"),
        AIMessage(content="It is 215 metres."),
    ]


def _visited_messages():
    return [
        HumanMessage(content=_UNNAMED_MANDATE),
        AIMessage(content="", tool_calls=[{"name": "visit", "args": {"url": "https://en.wikipedia.org/wiki/GRES-2"}, "id": "2"}]),
        ToolMessage(content="SOURCE: https://en.wikipedia.org/wiki/GRES-2\nThe chimney is 419.7 m.", tool_call_id="2"),
        AIMessage(content="419.7 m (https://en.wikipedia.org/wiki/GRES-2)."),
    ]


def test_zero_visit_gate_nudges_a_run_that_read_no_page(monkeypatch):
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_zero_visit_messages(), _visited_messages()], _UNNAMED_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2
    nudge = stub_graph.calls[1][-1].content
    assert "visit" in nudge.lower()
    assert result["final_deliverable"] == "419.7 m (https://en.wikipedia.org/wiki/GRES-2)."


def test_zero_visit_gate_is_bounded_to_one_nudge(monkeypatch):
    """A model that ignores the nudge must not loop — exactly one extra pass, then finalize."""
    result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_zero_visit_messages(), _zero_visit_messages()], _UNNAMED_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2
    assert result["final_deliverable"] == "It is 215 metres."  # nudged, not hard-failed


def test_zero_visit_gate_is_inert_when_nothing_could_have_been_visited(monkeypatch):
    """A search that returned nothing (or a task with no retrieval step at all) must never be
    punished for a retrieval failure that is not the model's fault."""
    _result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_zero_visit_messages(tool_content="No results found.")], _UNNAMED_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 1


def test_zero_visit_gate_is_inert_with_no_tool_calls_at_all(monkeypatch):
    messages = [HumanMessage(content=_UNNAMED_MANDATE), AIMessage(content="It is 215 metres.")]
    _result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [messages], _UNNAMED_MANDATE, candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 1


def test_zero_visit_gate_is_inert_when_a_page_was_read(monkeypatch):
    _result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_visited_messages()], _UNNAMED_MANDATE, candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 1


def test_zero_visit_gate_does_not_double_extend_after_the_coverage_gate(monkeypatch):
    """When the named-roster coverage gate already spent its one-time extension, the zero-visit
    gate must not spend a second one on the same run."""
    _result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch,
        [_search_only_fabrication_messages(), _search_only_fabrication_messages()],
        _TWO_CANDIDATE_MANDATE,
        candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 2


def test_zero_visit_gate_env_flag_off_leaves_the_run_untouched(monkeypatch):
    monkeypatch.setenv("LEDGER_ZERO_VISIT_GATE", "0")
    _result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_zero_visit_messages()], _UNNAMED_MANDATE, candidate_coverage_gate=True,
    )
    assert len(stub_graph.calls) == 1


def test_zero_visit_gate_requires_the_coverage_gate(monkeypatch):
    """It rides on the same harness flag as its sibling gate, so a caller that opted out of
    coverage enforcement entirely gets today's behavior unchanged."""
    _result, _llm, stub_graph = _run_solve_with_sequenced_graph(
        monkeypatch, [_zero_visit_messages()], _UNNAMED_MANDATE,
    )
    assert len(stub_graph.calls) == 1


def test_has_visitable_url_ignores_prose_without_links():
    assert _has_visitable_url(_zero_visit_messages()) is True
    assert _has_visitable_url(_zero_visit_messages(tool_content="No results found.")) is False
    assert _has_visitable_url([HumanMessage(content="https://example.com")]) is False
