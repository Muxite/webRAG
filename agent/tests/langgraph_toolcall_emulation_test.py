"""Tool-call emulation for models that cannot take an HTTP ``tools`` parameter.

`create_react_agent` binds tools unconditionally, so four models on this project's local roster
(tinyllama, phi3:mini, gemma2:2b via Ollama's 400; llama-3.2-1b via OpenRouter's 404) fail the
`langgraph_react` arm in ~0.1s and score a genuine 0 — while the native engine runs them fine
over text/JSON. Any comparison drawn from that is measuring an implementation gap we imposed.
These tests pin the fix: capability detection, the text/JSON transport it selects, and the
`tool_transport` field that lets analysis tell a fair comparison from a rigged one.

Offline: no LLM, no network — the Ollama probe's httpx client and the LLM are both stubbed.
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.app import model_capabilities as mc
from agent.app.langgraph_solver import (
    LangGraphSolver, _EmulatedToolCallTransport, _SolveState, _extract_json_object,
    _final_answer, _finish_answer, _make_tools,
)
from agent.app.testing import model_metadata as mm


# --------------------------------------------------------------------------- capability probe


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _client_factory(capabilities, *, fail=False):
    """An httpx.AsyncClient stand-in whose ``/api/show`` reports ``capabilities``."""
    calls = []

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None):
            calls.append(url)
            if fail:
                raise RuntimeError("connection refused")
            return _Resp({"details": {}, "model_info": {}, "capabilities": capabilities})

        async def get(self, url):
            calls.append(url)
            return _Resp({"models": []})

    return _FakeClient, calls


@pytest.fixture(autouse=True)
def _clear_caches():
    mc.reset_capability_cache()
    mm._CACHE.clear()
    yield
    mc.reset_capability_cache()
    mm._CACHE.clear()


def _supports(model, provider="ollama", api_url="http://localhost:11434/v1"):
    return asyncio.run(mc.supports_native_tool_calling(model, provider, api_url))


def test_local_model_declaring_tools_capability_supports_native_tool_calling(monkeypatch):
    client, _calls = _client_factory(["completion", "tools"])
    monkeypatch.setattr(mm.httpx, "AsyncClient", client)
    assert _supports("qwen2.5:7b") is True


def test_local_model_without_tools_capability_does_not_support_native_tool_calling(monkeypatch):
    client, _calls = _client_factory(["completion"])
    monkeypatch.setattr(mm.httpx, "AsyncClient", client)
    assert _supports("tinyllama") is False


def test_unknown_capability_falls_back_to_emulation(monkeypatch):
    """Fail SAFE: a probe that cannot answer must route to the transport that always works,
    never to the one that 400s and scores a hard 0."""
    client, _calls = _client_factory(None, fail=True)
    monkeypatch.setattr(mm.httpx, "AsyncClient", client)
    assert _supports("mystery:latest") is False


def test_missing_capabilities_key_falls_back_to_emulation(monkeypatch):
    """An Ollama build that omits ``capabilities`` entirely is unknown, not capable."""
    client, _calls = _client_factory(None)
    monkeypatch.setattr(mm.httpx, "AsyncClient", client)
    assert _supports("mystery:latest") is False


def test_capability_is_cached_per_process(monkeypatch):
    client, calls = _client_factory(["completion", "tools"])
    monkeypatch.setattr(mm.httpx, "AsyncClient", client)
    assert _supports("qwen2.5:7b") is True
    before = len(calls)
    assert _supports("qwen2.5:7b") is True
    assert len(calls) == before


def test_hosted_model_on_the_deny_list_takes_emulation(monkeypatch):
    monkeypatch.setattr(mm.httpx, "AsyncClient", _client_factory(["tools"])[0])
    assert asyncio.run(mc.supports_native_tool_calling(
        "meta-llama/llama-3.2-1b-instruct", "openrouter", "https://openrouter.ai/api/v1")) is False


def test_hosted_model_not_on_the_deny_list_keeps_the_native_path(monkeypatch):
    monkeypatch.setattr(mm.httpx, "AsyncClient", _client_factory(["tools"])[0])
    assert asyncio.run(mc.supports_native_tool_calling(
        "openai/gpt-5-mini", "openrouter", "https://openrouter.ai/api/v1")) is True


def test_capability_probe_never_raises(monkeypatch):
    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(mm.httpx, "AsyncClient", _Boom)
    assert _supports("anything") is False


# --------------------------------------------------------------------------- JSON extraction


def test_extract_json_object_reads_a_bare_object():
    assert _extract_json_object('{"action": "search"}') == {"action": "search"}


def test_extract_json_object_reads_a_fenced_object():
    raw = 'Sure!\n```json\n{"action": "visit", "args": {"url": "https://x/y"}}\n```\n'
    assert _extract_json_object(raw) == {"action": "visit", "args": {"url": "https://x/y"}}


def test_extract_json_object_takes_the_first_dict_of_a_list():
    assert _extract_json_object('[{"action": "search"}]') == {"action": "search"}


def test_extract_json_object_returns_none_for_prose():
    assert _extract_json_object("I think the answer is 1786.") is None


def test_extract_json_object_returns_none_for_empty_input():
    assert _extract_json_object("") is None
    assert _extract_json_object(None) is None


# --------------------------------------------------------------------------- emulated loop


class _FakeAgentIO:
    """Duck-typed `AgentIO` stand-in — records calls, returns canned data, no real I/O."""

    connector_sandbox = None

    def __init__(self):
        self.search_calls = []
        self.visit_calls = []

    async def search(self, query, count=10, timeout_seconds=None):
        self.search_calls.append((query, count))
        return [{"title": "Result A", "url": "https://example.com/a", "description": "desc A"}]

    async def visit(self, url, timeout_seconds=None):
        self.visit_calls.append(url)
        return "Mont Blanc was first climbed in 1786."


class _ScriptedLLM:
    """Replays canned completions in order; records every prompt it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def ainvoke(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        text = self.responses.pop(0) if self.responses else ""
        return AIMessage(content=text, usage_metadata={
            "input_tokens": 7, "output_tokens": 3, "total_tokens": 10})


def _run_emulated(responses, recursion_limit=20, tools=None, agent_io=None):
    agent_io = agent_io or _FakeAgentIO()
    tools = tools if tools is not None else _make_tools(agent_io, search_k=6, page_chars=6000)
    llm = _ScriptedLLM(responses)
    transport = _EmulatedToolCallTransport(llm, tools, "SYSTEM PROMPT", pre_model_hook=None)
    state = _SolveState(messages=[])
    error = None
    try:
        asyncio.run(transport.run(state, [HumanMessage(content="the task")], recursion_limit))
    except BaseException as exc:  # noqa: BLE001 — asserted on by the caller
        error = exc
    return state, llm, agent_io, error


def test_emulated_loop_dispatches_the_same_tools_with_the_same_args():
    state, _llm, io, error = _run_emulated([
        '{"thought": "look it up", "action": "search", "args": {"query": "mont blanc"}}',
        '{"thought": "read it", "action": "visit", "args": {"url": "https://example.com/a"}}',
        '{"thought": "done", "action": "finish", "args": {"answer": "1786"}}',
    ])
    assert error is None
    assert io.search_calls == [("mont blanc", 6)]
    assert io.visit_calls == ["https://example.com/a"]
    assert _final_answer(state.messages) == "1786"


def test_emulated_loop_never_sends_a_tools_parameter():
    """The whole point: the transport must not reintroduce the parameter that 400s."""
    _state, llm, _io, _error = _run_emulated([
        '{"thought": "done", "action": "finish", "args": {"answer": "x"}}',
    ])
    assert llm.prompts, "the model was never called"


def test_emulated_loop_builds_transcript_messages_the_post_processing_understands():
    state, _llm, _io, _error = _run_emulated([
        '{"thought": "look", "action": "search", "args": {"query": "q"}}',
        '{"thought": "done", "action": "finish", "args": {"answer": "the answer"}}',
    ])
    tool_calls = [m for m in state.messages
                  if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
    assert [c["name"] for m in tool_calls for c in m.tool_calls] == ["search"]
    observations = [m for m in state.messages if isinstance(m, ToolMessage)]
    assert len(observations) == 1
    assert observations[0].tool_call_id == tool_calls[0].tool_calls[0]["id"]
    assert "Result A" in observations[0].content


def test_emulated_loop_records_usage_on_every_model_turn():
    from agent.app.langgraph_solver import _extract_usage

    state, _llm, _io, _error = _run_emulated([
        '{"thought": "look", "action": "search", "args": {"query": "q"}}',
        '{"thought": "done", "action": "finish", "args": {"answer": "a"}}',
    ])
    assert _extract_usage(state.messages) == [
        {"prompt_tokens": 7, "completion_tokens": 3},
        {"prompt_tokens": 7, "completion_tokens": 3},
    ]


def test_malformed_output_does_not_crash_and_produces_a_usable_nudge():
    state, llm, _io, error = _run_emulated([
        "I am just going to think out loud for a moment.",
        '{"thought": "ok", "action": "finish", "args": {"answer": "recovered"}}',
    ])
    assert error is None
    assert _final_answer(state.messages) == "recovered"
    nudged = "\n".join(m.content for m in llm.prompts[-1])
    assert "JSON" in nudged


def test_persistent_malformed_output_degrades_to_the_models_prose():
    """After the nudge budget is spent, a model that simply cannot emit JSON still yields its
    last prose turn rather than an empty deliverable."""
    state, _llm, _io, error = _run_emulated(["not json"] * 6, recursion_limit=20)
    assert error is None
    assert _final_answer(state.messages) == "not json"


def test_unknown_action_produces_an_invalid_action_observation():
    state, _llm, io, error = _run_emulated([
        '{"thought": "hmm", "action": "teleport", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert io.search_calls == []
    assert any("INVALID ACTION" in getattr(m, "content", "") for m in state.messages)


def test_bad_tool_arguments_become_an_observation_not_a_crash():
    state, _llm, io, error = _run_emulated([
        '{"thought": "oops", "action": "search", "args": {"wrong_slot": "q"}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert io.search_calls == []
    assert any("TOOL ERROR" in getattr(m, "content", "") for m in state.messages)


def test_non_dict_args_are_coerced_instead_of_crashing():
    state, _llm, _io, error = _run_emulated([
        '{"thought": "oops", "action": "search", "args": "mont blanc"}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert _final_answer(state.messages) == "done"


def test_step_exhaustion_raises_the_same_error_the_native_transport_raises():
    """`solve()` already turns `GraphRecursionError` into the forced-synthesis path; raising the
    same exception means the emulated transport needs no separate handling downstream."""
    state, _llm, io, error = _run_emulated(
        ['{"thought": "again", "action": "search", "args": {"query": "q%d"}}' % i
         for i in range(10)],
        recursion_limit=6,
    )
    assert isinstance(error, GraphRecursionError)
    assert len(io.search_calls) == 3  # recursion_limit // 2 model turns
    assert state.messages, "partial evidence must survive the raise"


def test_finish_emits_a_finish_tool_call_when_the_finish_tool_exists():
    io = _FakeAgentIO()
    tools = _make_tools(io, search_k=6, page_chars=6000, require_finish_tool=True)
    state, _llm, _io, error = _run_emulated(
        ['{"thought": "done", "action": "finish", "args": {"answer": "submitted answer"}}'],
        tools=tools, agent_io=io,
    )
    assert error is None
    assert _finish_answer(state.messages) == "submitted answer"


def test_transcript_is_fed_back_to_the_model_as_text():
    _state, llm, _io, _error = _run_emulated([
        '{"thought": "look", "action": "search", "args": {"query": "mont blanc"}}',
        '{"thought": "done", "action": "finish", "args": {"answer": "a"}}',
    ])
    second_prompt = "\n".join(m.content for m in llm.prompts[1])
    assert "mont blanc" in second_prompt
    assert "Result A" in second_prompt


# --------------------------------------------------------------------------- transport routing


class _StubLLM:
    def __init__(self, answer="synthesized answer"):
        self.answer = answer
        self.calls = []

    def bind_tools(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.answer)


def _solve_with_capability(monkeypatch, capable, **solver_kwargs):
    from agent.app import langgraph_solver

    native_calls = []

    class _StubGraph:
        async def astream(self, _inputs, config=None, stream_mode=None):
            yield {"messages": [HumanMessage(content="the task"),
                                AIMessage(content="native answer")]}

    def _create(*args, **kwargs):
        native_calls.append(kwargs)
        return _StubGraph()

    async def _capability(model_name, provider=None, api_url=None):
        return capable

    monkeypatch.setattr(langgraph_solver, "create_react_agent", _create)
    monkeypatch.setattr(langgraph_solver, "supports_native_tool_calling", _capability)
    monkeypatch.setattr(LangGraphSolver, "_build_llm",
                        lambda self: _ScriptedLLM([
                            '{"thought": "done", "action": "finish", '
                            '"args": {"answer": "emulated answer"}}']))
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="tinyllama", **solver_kwargs,
    )
    return asyncio.run(solver.solve("the task", max_steps=4)), native_calls


def test_tool_capable_model_still_takes_the_native_path(monkeypatch):
    result, native_calls = _solve_with_capability(monkeypatch, True)
    assert len(native_calls) == 1
    assert result["final_deliverable"] == "native answer"
    assert result["tool_transport"] == "native"


def test_non_tool_capable_model_takes_the_emulated_path(monkeypatch):
    result, native_calls = _solve_with_capability(monkeypatch, False)
    assert native_calls == []  # create_react_agent never constructed -> no `tools` parameter
    assert result["final_deliverable"] == "emulated answer"
    assert result["tool_transport"] == "emulated"


def test_emulation_can_be_disabled_leaving_the_native_path_forced(monkeypatch):
    result, native_calls = _solve_with_capability(monkeypatch, False, tool_call_emulation=False)
    assert len(native_calls) == 1
    assert result["tool_transport"] == "native"


def test_capability_detection_is_skipped_when_emulation_is_off(monkeypatch):
    """No probe, no latency, no behavior change for a run that opted out."""
    from agent.app import langgraph_solver

    async def _boom(*a, **k):
        raise AssertionError("capability probe must not run when emulation is disabled")

    monkeypatch.setattr(langgraph_solver, "supports_native_tool_calling", _boom)
    result, native_calls = _solve_with_capability(monkeypatch, True, tool_call_emulation=False)
    assert len(native_calls) == 1
    assert result["tool_transport"] == "native"


# --------------------------------------------------------------------------- result payload


class _StubTestModule:
    metadata = {"test_id": "999"}

    def get_task_statement(self):
        return "do the thing"


def _run_offtheshelf(monkeypatch, solver_result):
    from agent.app.testing import execution_langgraph

    captured = {}

    class _StubSolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def solve(self, mandate, **kwargs):
            return dict(solver_result)

    monkeypatch.setattr(execution_langgraph, "LangGraphSolver", _StubSolver)
    result = asyncio.run(execution_langgraph.run_offtheshelf_execution(
        test_module=_StubTestModule(), model_name="tinyllama",
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        run_stamp="teststamp",
    ))
    return result, captured


def test_tool_transport_is_recorded_on_the_emulated_path(monkeypatch):
    result, _captured = _run_offtheshelf(monkeypatch, {
        "final_deliverable": "a", "success": True, "observability": {},
        "tool_transport": "emulated",
    })
    assert result["output"]["tool_transport"] == "emulated"


def test_tool_transport_is_recorded_on_the_native_path(monkeypatch):
    result, _captured = _run_offtheshelf(monkeypatch, {
        "final_deliverable": "a", "success": True, "observability": {},
        "tool_transport": "native",
    })
    assert result["output"]["tool_transport"] == "native"


def test_tool_transport_is_unknown_when_the_solver_never_reported_one(monkeypatch):
    """A construction-time failure (missing API key) never reaches the solver's own routing —
    the field must still be present and must not claim a path that was never taken."""
    result, _captured = _run_offtheshelf(monkeypatch, {
        "final_deliverable": "", "success": False, "observability": {},
    })
    assert result["output"]["tool_transport"] == "unknown"


def test_offtheshelf_execution_enables_emulation_by_default(monkeypatch):
    _result, captured = _run_offtheshelf(monkeypatch, {
        "final_deliverable": "a", "success": True, "observability": {}})
    assert captured["tool_call_emulation"] is True


def test_offtheshelf_execution_emulation_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LANGGRAPH_TOOL_EMULATION", "0")
    _result, captured = _run_offtheshelf(monkeypatch, {
        "final_deliverable": "a", "success": True, "observability": {}})
    assert captured["tool_call_emulation"] is False
