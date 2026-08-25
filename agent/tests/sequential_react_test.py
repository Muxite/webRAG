"""
Unit tests for the sequential ReAct agent loop (testing/execution_sequential.py) — free.

Drive the loop with a mocked AgentIO and assert it issues the right tool calls
(search -> visit -> finish), passes args through, falls back to a forced synthesis
when the model never calls finish within the step budget, and (F16) retries a TRANSIENT
tool failure exactly like the graph arms do when the shared
``connector_retry_on_failure_enabled`` flag is on.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_sequential as seq


def _agent_io(decisions, search_results=None, page_text="PAGE CONTENT", synth="SYNTH ANSWER"):
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={"messages": []})
    # query_llm returns each decision JSON in order, then the synthesis text.
    io.query_llm = AsyncMock(side_effect=[*(json.dumps(d) for d in decisions), synth])
    io.search = AsyncMock(return_value=search_results or [
        {"title": "Toni Morrison", "url": "https://en.wikipedia.org/wiki/Toni_Morrison", "description": "novelist"}])
    io.visit = AsyncMock(return_value=page_text)
    return io


def test_react_search_visit_finish():
    decisions = [
        {"thought": "find author", "action": "search", "args": {"query": "Beloved author"}},
        {"thought": "read page", "action": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Toni_Morrison"}},
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison; MA Cornell. https://en.wikipedia.org/wiki/Toni_Morrison"}},
    ]
    io = _agent_io(decisions)
    out = asyncio.run(seq._run_react(io, "Who wrote Beloved and where did she get her MA?", "m", max_steps=6, max_tokens=512))
    assert "Toni Morrison" in out and "Cornell" in out
    io.search.assert_awaited_once()
    io.visit.assert_awaited_once()
    # visit got the exact URL the model chose
    assert io.visit.await_args.args[0] == "https://en.wikipedia.org/wiki/Toni_Morrison"


def test_react_forced_synthesis_when_no_finish():
    # Model keeps searching and never finishes; at max_steps the loop forces a synthesis.
    decisions = [{"thought": "search", "action": "search", "args": {"query": "q"}}]
    io = _agent_io(decisions, synth="FORCED SYNTHESIS")
    out = asyncio.run(seq._run_react(io, "task", "m", max_steps=1, max_tokens=512))
    assert out == "FORCED SYNTHESIS"


def test_react_dedup_repeated_search_nudges_without_researching():
    # Breadth-loop guard: a repeated query (modulo case/whitespace) must NOT trigger a second
    # web search — the loop nudges toward visit/finish instead. Distinct queries are unaffected.
    decisions = [
        {"thought": "search", "action": "search", "args": {"query": "deepest lake"}},
        {"thought": "search again", "action": "search", "args": {"query": "Deepest Lake "}},  # dup
        {"thought": "done", "action": "finish",
         "args": {"answer": "Lake Baikal https://en.wikipedia.org/wiki/Lake_Baikal"}},
    ]
    io = _agent_io(decisions)
    out = asyncio.run(seq._run_react(io, "Which lake is deepest?", "m", max_steps=6, max_tokens=512))
    assert "Baikal" in out
    io.search.assert_awaited_once()  # the duplicate did NOT re-run search


def test_react_dedup_distinct_searches_both_run():
    # Two DIFFERENT queries (e.g. two authors in a fan-out) must both search — guard is repeat-only.
    decisions = [
        {"thought": "a", "action": "search", "args": {"query": "author of Beloved"}},
        {"thought": "b", "action": "search", "args": {"query": "author of Pride and Prejudice"}},
        {"thought": "done", "action": "finish", "args": {"answer": "done"}},
    ]
    io = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "two authors", "m", max_steps=6, max_tokens=512))
    assert io.search.await_count == 2


def test_react_invalid_json_does_not_crash():
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["not json", "FINAL"])
    io.search = AsyncMock(return_value=[])
    io.visit = AsyncMock(return_value="")
    out = asyncio.run(seq._run_react(io, "task", "m", max_steps=1, max_tokens=256))
    assert out == "FINAL"  # invalid decision on the last step -> forced synthesis


def test_react_list_shaped_decision_does_not_crash():
    # Reasoning models sometimes return the step as a LIST (e.g. [{...}]) instead of a
    # dict; the loop must take the first dict and act on it, not crash on .get().
    decisions = [[{"thought": "wrapped in a list", "action": "finish", "args": {"answer": "LIST ANSWER"}}]]
    io = _agent_io(decisions)
    out = asyncio.run(seq._run_react(io, "task", "m", max_steps=3, max_tokens=512))
    assert out == "LIST ANSWER"
    io.search.assert_not_awaited()
    io.visit.assert_not_awaited()


def test_react_non_dict_decision_falls_through_to_invalid_action():
    # A bare scalar (or a list with no dict) must not crash; it becomes an empty
    # decision -> INVALID ACTION observation, then forced synthesis at max_steps.
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["[1, 2, 3]", "FORCED"])
    io.search = AsyncMock(return_value=[])
    io.visit = AsyncMock(return_value="")
    out = asyncio.run(seq._run_react(io, "task", "m", max_steps=1, max_tokens=256))
    assert out == "FORCED"


def test_react_step_token_budget_default_is_reasoning_adequate(monkeypatch):
    # Per-step decision budget must default to >=4096 so a reasoning model's action
    # JSON is not truncated, and must honor the env override.
    monkeypatch.delenv("IDEA_TEST_SEQ_STEP_MAX_TOKENS", raising=False)
    decisions = [{"thought": "done", "action": "finish", "args": {"answer": "A"}}]
    io = _agent_io(decisions)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=2, max_tokens=512))
    step_call = io.build_llm_payload.call_args_list[0]
    assert step_call.kwargs["max_tokens"] == 4096

    monkeypatch.setenv("IDEA_TEST_SEQ_STEP_MAX_TOKENS", "9000")
    io2 = _agent_io(decisions)
    asyncio.run(seq._run_react(io2, "task", "m", max_steps=2, max_tokens=512))
    assert io2.build_llm_payload.call_args_list[0].kwargs["max_tokens"] == 9000


# ------------------------------------------- F16: arm-symmetric tool retry (2026-08-09) ---------
# The graph arms retry a transient search/visit failure in place (idea_engine._maybe_retry_tool_
# failure) whenever `connector_retry_on_failure_enabled` is set — which every benchmark driver
# sets for EVERY arm (IDEA_TEST_CONNECTOR_RETRY=1). This arm had no retry at all, so the same
# flaky network hit only the reference model: an infra confound, not a model difference.
_RETRY_ON = seq.ToolRetry(enabled=True, max_attempts=2, backoff_seconds=0.0)

_SEARCH_THEN_FINISH = [
    {"thought": "look", "action": "search", "args": {"query": "q"}},
    {"thought": "done", "action": "finish", "args": {"answer": "A"}},
]
_VISIT_THEN_FINISH = [
    {"thought": "read", "action": "visit", "args": {"url": "https://en.wikipedia.org/wiki/X"}},
    {"thought": "done", "action": "finish", "args": {"answer": "A"}},
]


def _observation(io, step=1):
    """The scratchpad the loop fed back into the NEXT step's prompt."""
    return io.build_llm_payload.call_args_list[step].kwargs["messages"][1]["content"]


def test_tool_retry_defaults_to_disabled():
    # No settings dict (any non-benchmark caller) -> unchanged behavior.
    assert seq.ToolRetry.from_settings(None).enabled is False
    assert seq.ToolRetry.from_settings({}).enabled is False


def test_tool_retry_reads_the_same_settings_keys_as_the_graph_arm():
    retry = seq.ToolRetry.from_settings({
        "connector_retry_on_failure_enabled": True,
        "connector_retry_max_attempts": 3,
        "connector_retry_backoff_seconds": 0.25,
    })
    assert retry == seq.ToolRetry(enabled=True, max_attempts=3, backoff_seconds=0.25)


def test_transient_search_failure_is_not_retried_when_flag_off():
    io = _agent_io(_SEARCH_THEN_FINISH)
    io.search = AsyncMock(side_effect=[asyncio.TimeoutError("search timed out")])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512))
    assert io.search.await_count == 1                    # default (flag off) is unchanged
    assert "SEARCH ERROR" in _observation(io)


def test_transient_search_failure_is_retried_when_flag_on():
    io = _agent_io(_SEARCH_THEN_FINISH)
    io.search = AsyncMock(side_effect=[
        asyncio.TimeoutError("search timed out"),
        [{"title": "T", "url": "https://en.wikipedia.org/wiki/X", "description": "d"}],
    ])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.search.await_count == 2
    obs = _observation(io)
    assert "SEARCH RESULTS" in obs and "SEARCH ERROR" not in obs


def test_empty_search_results_are_retried():
    # Success-with-EMPTY payload is a tool failure in the graph arm too (`is_tool_failure`).
    io = _agent_io(_SEARCH_THEN_FINISH)
    io.search = AsyncMock(side_effect=[[], [{"title": "T", "url": "u", "description": "d"}]])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.search.await_count == 2


def test_retry_is_bounded_by_max_attempts_and_keeps_the_error_observation():
    io = _agent_io(_SEARCH_THEN_FINISH)
    io.search = AsyncMock(side_effect=[asyncio.TimeoutError("boom")] * 5)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.search.await_count == 3                    # 1 initial + max_attempts(2) retries
    assert "SEARCH ERROR" in _observation(io)


def test_transient_visit_failure_is_retried_and_recovers_evidence():
    io = _agent_io(_VISIT_THEN_FINISH)
    io.visit = AsyncMock(side_effect=[
        RuntimeError("HTTP visit failed: https://en.wikipedia.org/wiki/X status=503"),
        "REAL PAGE TEXT",
    ])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.visit.await_count == 2
    assert "REAL PAGE TEXT" in _observation(io)


# --- query reformulation on retry (ported from idea_engine.py's
# _reformulate_search_query_if_multi_entity: an AND-shaped multi-quoted-entity query that
# returned nothing just fails again if resent unchanged) ---

def test_reformulate_multi_entity_query_or_joins_quoted_phrases():
    out = seq._reformulate_multi_entity_query('"Erie Canal" "Suez Canal" opening year')
    assert out == '"Erie Canal" OR "Suez Canal" opening year'


def test_reformulate_multi_entity_query_none_for_single_phrase():
    assert seq._reformulate_multi_entity_query('"Erie Canal" opening year') is None


def test_reformulate_multi_entity_query_none_for_no_quotes():
    assert seq._reformulate_multi_entity_query("Erie Canal opening year") is None


def test_reformulate_multi_entity_query_none_when_already_or_joined():
    assert seq._reformulate_multi_entity_query('"Erie Canal" OR "Suez Canal"') is None


def test_reformulate_multi_entity_query_none_for_empty():
    assert seq._reformulate_multi_entity_query("") is None
    assert seq._reformulate_multi_entity_query(None) is None


def test_search_retry_reformulates_a_multi_entity_query_that_returned_nothing():
    decisions = [
        {"thought": "look", "action": "search", "args": {"query": '"Erie Canal" "Suez Canal"'}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions)
    io.search = AsyncMock(side_effect=[
        [],  # first attempt: the AND-shaped query returns nothing
        [{"title": "T", "url": "https://en.wikipedia.org/wiki/X", "description": "d"}],
    ])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.search.await_count == 2
    second_call_query = io.search.await_args_list[1].args[0]
    assert second_call_query == '"Erie Canal" OR "Suez Canal"'


def test_search_retry_does_not_reformulate_a_single_entity_query():
    """A normal (non-multi-entity) query is retried UNCHANGED, exactly as before this fix."""
    io = _agent_io(_SEARCH_THEN_FINISH)
    io.search = AsyncMock(side_effect=[[], [{"title": "T", "url": "u", "description": "d"}]])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.search.await_args_list[0].args[0] == "q"
    assert io.search.await_args_list[1].args[0] == "q"


def test_search_retry_reformulation_is_inert_when_retry_flag_off():
    decisions = [
        {"thought": "look", "action": "search", "args": {"query": '"Erie Canal" "Suez Canal"'}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    io = _agent_io(decisions)
    io.search = AsyncMock(return_value=[])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512))  # retry defaults off
    assert io.search.await_count == 1  # no retry at all -> no reformulation attempted either


def test_permanent_visit_failure_is_not_retried():
    # 403 bot-block: re-running it only burns budget — the graph arm doesn't retry it either.
    io = _agent_io(_VISIT_THEN_FINISH)
    io.visit = AsyncMock(side_effect=[
        RuntimeError("HTTP visit failed: https://en.wikipedia.org/wiki/X status=403")] * 3)
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.visit.await_count == 1
    assert "VISIT ERROR" in _observation(io)


def test_empty_page_is_retried():
    io = _agent_io(_VISIT_THEN_FINISH)
    io.visit = AsyncMock(side_effect=[seq._EMPTY_PAGE, "REAL PAGE TEXT"])
    asyncio.run(seq._run_react(io, "task", "m", max_steps=4, max_tokens=512, retry=_RETRY_ON))
    assert io.visit.await_count == 2
    assert "REAL PAGE TEXT" in _observation(io)


@pytest.mark.asyncio
async def test_run_sequential_execution_threads_the_retry_flag_from_settings(monkeypatch):
    """End-to-end wiring: the arm's settings dict (the same one the runner hands every variant)
    must reach the react loop as a ToolRetry, or the flag would be silently inert."""
    captured = {}

    async def _fake_run_react(agent_io, mandate, model_name, max_steps, max_tokens, retry=None,
                              context_cap=None):
        captured["retry"] = retry
        captured["context_cap"] = context_cap
        return "answer"

    monkeypatch.setattr(seq, "_run_react", _fake_run_react)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."
    await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
        idea_settings={"connector_retry_on_failure_enabled": True},
    )
    assert captured["retry"].enabled is True
    # The Phase 0 context cap threads the same way, and is OFF unless its own flag is set.
    assert captured["context_cap"].enabled is False


@pytest.mark.asyncio
async def test_run_sequential_execution_threads_the_context_cap_from_settings(monkeypatch):
    """Phase 0's `sequential_react_context_matched` axis must reach the react loop as a resolved
    SequentialContextCap carrying the DAG's own budget, or the arm would be silently inert."""
    captured = {}

    async def _fake_run_react(agent_io, mandate, model_name, max_steps, max_tokens, retry=None,
                              context_cap=None):
        captured["context_cap"] = context_cap
        return "answer"

    monkeypatch.setattr(seq, "_run_react", _fake_run_react)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."
    await seq.run_sequential_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
        idea_settings={
            "run_policy_sequential_context_cap_enabled": True,
            "expansion_ancestor_content_chars": 1000,
            "expansion_max_context_nodes": 5,
        },
    )
    cap = captured["context_cap"]
    assert cap.enabled is True
    assert (cap.per_step_chars, cap.total_chars) == (1000, 5000)


@pytest.mark.asyncio
async def test_run_complete_test_passes_settings_to_the_sequential_runner(monkeypatch):
    """The harness must hand THIS arm the run's settings like it does every other variant —
    otherwise the shared retry flag never reaches it and the arms stay asymmetric."""
    from agent.app.testing import runner as harness_runner

    dispatched = AsyncMock(return_value={"output": {}, "graph": {}, "observability": {}})
    monkeypatch.setattr(harness_runner, "run_sequential_execution", dispatched)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.validation_runner.run = AsyncMock(return_value={"overall_score": 0.0})
    await harness_runner.run_complete_test(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        idea_settings={"connector_retry_on_failure_enabled": True},
        run_stamp="r1", summarize_observability_func=lambda *a, **k: {},
        execution_variant="sequential_react",
    )
    assert dispatched.await_args.kwargs["idea_settings"] == {
        "connector_retry_on_failure_enabled": True}
