import pytest
import asyncio
import logging
from agent.app.agent import Agent
import os
from unittest.mock import AsyncMock, MagicMock

from agent.app.tick_output import TickOutput, ActionType


@pytest.mark.skipif(os.environ.get("LITE") == "true", reason="Expensive test.")
@pytest.mark.asyncio
async def test_agent_find_panda_diet(caplog):
    """
    Make the agent perform a simple task, but also cite a source.
    Utilizes searching, visiting, thinking.
    """
    caplog.set_level("INFO")
    mandate = "Find out what pandas eat, but give a source."
    async with Agent(mandate=mandate, max_ticks=5) as agent:
        output = await agent.run()
        result = output['deliverables']
        logging.info(f"Agent deliverables: {result}")
        logging.info(f"Agent history: {output['history']}")

        string = "".join(result)
        assert "panda" in string
        assert "http" in string
        assert "eat" in string


@pytest.mark.asyncio
async def test_apply_tick_output_updates_history_notes_and_deliverables():
    agent = Agent(mandate="m", max_ticks=1)
    agent.current_tick = 1
    to = TickOutput({
        "history_update": "learned X",
        "note_update": "note Y",
        "deliverable": "final Z",
        "next_action": "think"
    })
    agent._apply_tick_output(to)
    assert agent.history[-1].endswith("learned X")
    assert agent.notes[-1] == "note Y"
    assert agent.deliverables[-1] == "final Z"


@pytest.mark.asyncio
async def test_do_action_think_and_exit_branches(caplog):
    caplog.set_level("INFO")
    agent = Agent(mandate="m", max_ticks=1)

    to_think = TickOutput({"next_action": "think"})
    await agent._do_action(to_think)
    assert any("thinking" in rec.message.lower() for rec in caplog.records)

    to_exit = TickOutput({"next_action": "exit"})
    await agent._do_action(to_exit)
    assert any("exit action" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_perform_web_search_success_and_failure(monkeypatch):
    agent = Agent(mandate="m", max_ticks=1)

    agent.connector_search.query_search = AsyncMock(return_value=[
        {"url": "https://a", "description": "A"},
        {"url": "https://b", "description": "B"},
    ])
    await agent._perform_web_search("q", count=2)
    assert "Web search for" in agent.observations
    assert "https://a" in agent.observations and "https://b" in agent.observations

    agent.observations = ""
    agent.connector_search.query_search = AsyncMock(return_value=None)
    await agent._perform_web_search("q2", count=1)
    assert "[Search API unavailable or failed]" in agent.observations


@pytest.mark.asyncio
async def test_perform_visit_invalid_url_logs_error(caplog):
    caplog.set_level("ERROR")
    agent = Agent(mandate="m", max_ticks=1)
    await agent._perform_visit("notaurl")
    assert any("invalid url" in rec.message.lower() for rec in caplog.records)
    assert "Could not fetch URL" in agent.observations


@pytest.mark.asyncio
async def test_perform_visit_success_and_http_error(monkeypatch):
    agent = Agent(mandate="m", max_ticks=1)

    agent.connector_http.request = AsyncMock(return_value=MagicMock(error=False, data="Hello World", status=200))
    await agent._perform_visit("https://example.com")
    assert "Visited https://example.com:" in agent.observations
    assert "Hello World" in agent.observations

    agent.observations = ""
    agent.connector_http.request = AsyncMock(return_value=MagicMock(error=True, data=None, status=500))
    await agent._perform_visit("https://example.com")
    assert "[Could not fetch URL: 500]" in agent.observations


@pytest.mark.asyncio
async def test_store_chroma_skips_when_unavailable_and_calls_when_available(monkeypatch):
    agent = Agent(mandate="m", max_ticks=1)

    agent.connector_chroma.chroma_api_ready = False
    agent.connector_chroma.add_to_chroma = AsyncMock()
    await agent._store_chroma(["id1"], [{"a": 1}], ["doc"])
    agent.connector_chroma.add_to_chroma.assert_not_awaited()

    agent.connector_chroma.chroma_api_ready = True
    await agent._store_chroma(["id1"], [{"a": 1}], ["doc"])
    agent.connector_chroma.add_to_chroma.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_chroma_returns_documents_when_available(monkeypatch):
    agent = Agent(mandate="m", max_ticks=1)

    agent.connector_chroma.chroma_api_ready = False
    agent.connector_chroma.query_chroma = AsyncMock()
    docs = await agent._retrieve_chroma(["topic"])
    assert docs == []
    agent.connector_chroma.query_chroma.assert_not_awaited()

    agent.connector_chroma.chroma_api_ready = True
    agent.connector_chroma.query_chroma = AsyncMock(return_value={
        "documents": [["D1", "D2"], ["D3"]]
    })
    docs = await agent._retrieve_chroma(["topic"])
    assert docs == ["D1", "D2", "D3"]


@pytest.mark.asyncio
async def test_final_output_uses_llm_and_returns_success(monkeypatch):
    agent = Agent(mandate="Summarize", max_ticks=1)
    agent.deliverables = ["A", "B"]
    agent.notes = ["N1", "N2"]

    agent.connector_llm.query_llm = AsyncMock(return_value=__import__("json").dumps({
        "deliverable": "A B",
        "summary": "s"
    }))

    result = await agent._final_output()
    d = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    assert d["success"] is True
    assert "A" in d.get("final_deliverable", "")
    assert "B" in d.get("final_deliverable", "")


@pytest.mark.asyncio
async def test_run_happy_path_one_tick(monkeypatch):
    """Simulate a single tick that exits quickly and produces a final output."""
    agent = Agent(mandate="say hi and exit", max_ticks=2)

    monkeypatch.setattr(agent, "initialize", AsyncMock(return_value=True))

    import json as _json
    tick_content = {
        "history_update": "started",
        "note_update": "ok",
        "cache_update": [],
        "cache_retrieve": [],
        "deliverable": "hi",
        "next_action": "exit"
    }
    final_content = {"deliverable": "hi", "summary": "done"}
    agent.connector_llm.query_llm = AsyncMock(side_effect=[
        _json.dumps(tick_content),
        _json.dumps(final_content),
    ])

    agent.connector_chroma.init_chroma = AsyncMock(return_value=True)
    agent.connector_chroma.add_to_chroma = AsyncMock()
    agent.connector_chroma.query_chroma = AsyncMock(return_value={"documents": []})

    result = await agent.run()
    assert result["success"] is True
    assert "hi" in (result.get("final_deliverable") or "")


@pytest.mark.asyncio
async def test_run_initialization_failure(monkeypatch):
    agent = Agent(mandate="m", max_ticks=1)
    monkeypatch.setattr(agent, "initialize", AsyncMock(return_value=False))
    result = await agent.run()
    assert result["success"] is False
    assert result.get("deliverables") == []
