"""
Unit tests for the sequential ReAct agent loop (testing/execution_sequential.py) — free.

Drive the loop with a mocked AgentIO and assert it issues the right tool calls
(search -> visit -> finish), passes args through, and falls back to a forced synthesis
when the model never calls finish within the step budget.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

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


def test_react_invalid_json_does_not_crash():
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["not json", "FINAL"])
    io.search = AsyncMock(return_value=[])
    io.visit = AsyncMock(return_value="")
    out = asyncio.run(seq._run_react(io, "task", "m", max_steps=1, max_tokens=256))
    assert out == "FINAL"  # invalid decision on the last step -> forced synthesis
