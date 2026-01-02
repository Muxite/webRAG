import pytest
import asyncio
import logging
import json
from agent.app.agent import Agent
import os
from unittest.mock import AsyncMock, MagicMock

from agent.app.tick_output import TickOutput
from shared.connector_config import ConnectorConfig
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma


@pytest.fixture
def connectors():
    config = ConnectorConfig()
    return {
        "connector_llm": ConnectorLLM(config),
        "connector_search": ConnectorSearch(config),
        "connector_http": ConnectorHttp(config),
        "connector_chroma": ConnectorChroma(config),
    }


@pytest.fixture
def ready_agent(connectors):
    agent = Agent(mandate="test", max_ticks=1, **connectors)
    agent.connector_llm.llm_api_ready = True
    agent.connector_search.search_api_ready = True
    agent.connector_chroma.chroma_api_ready = True
    return agent


@pytest.mark.skipif(os.environ.get("LITE") == "true", reason="Expensive test.")
@pytest.mark.asyncio
async def test_agent_find_panda_diet(caplog, connectors):
    caplog.set_level("INFO")
    async with Agent(mandate="Find out what pandas eat, but give a source.", max_ticks=5, **connectors) as agent:
        output = await agent.run()
        string = "".join(output['deliverables'])
        assert "panda" in string
        assert "http" in string
        assert "eat" in string


@pytest.mark.asyncio
async def test_apply_tick_output_updates_history_notes_and_deliverables(ready_agent):
    ready_agent.current_tick = 1
    to = TickOutput({
        "history_update": "learned X",
        "note_update": "note Y",
        "deliverable": "final Z",
        "next_action": "think"
    })
    ready_agent._apply_tick_output(to)
    assert ready_agent.history[-1].endswith("learned X")
    assert ready_agent.notes[-1] == "note Y"
    assert ready_agent.deliverables[-1] == "final Z"


@pytest.mark.asyncio
async def test_do_action_think(caplog, ready_agent):
    caplog.set_level("INFO")
    to = TickOutput({"next_action": "think"})
    await ready_agent._do_action(to)
    assert any("thinking" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_do_action_exit(caplog, ready_agent):
    caplog.set_level("INFO")
    to = TickOutput({"next_action": "exit"})
    await ready_agent._do_action(to)
    assert any("exit action" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_perform_web_search_success(ready_agent):
    ready_agent.connector_search.query_search = AsyncMock(return_value=[
        {"url": "https://a", "description": "A"},
        {"url": "https://b", "description": "B"},
    ])
    await ready_agent._perform_web_search("q", count=2)
    assert "Web search for" in ready_agent.observations
    assert "https://a" in ready_agent.observations and "https://b" in ready_agent.observations


@pytest.mark.asyncio
async def test_perform_web_search_failure(ready_agent):
    ready_agent.connector_search.query_search = AsyncMock(return_value=None)
    await ready_agent._perform_web_search("q")
    assert "[Search API unavailable or failed]" in ready_agent.observations


@pytest.mark.asyncio
async def test_perform_visit_invalid_url(caplog, connectors):
    caplog.set_level("ERROR")
    agent = Agent(mandate="m", max_ticks=1, **connectors)
    async with agent:
        await agent._perform_visit("notaurl")
    assert any("invalid url" in rec.message.lower() for rec in caplog.records)
    assert "Could not fetch URL" in agent.observations


@pytest.mark.asyncio
async def test_perform_visit_success(connectors):
    agent = Agent(mandate="m", max_ticks=1, **connectors)
    agent.connector_http.request = AsyncMock(return_value=MagicMock(error=False, data="Hello World", status=200))
    async with agent:
        await agent._perform_visit("https://example.com")
    assert "Visited https://example.com:" in agent.observations
    assert "Hello World" in agent.observations


@pytest.mark.asyncio
async def test_perform_visit_http_error(connectors):
    agent = Agent(mandate="m", max_ticks=1, **connectors)
    agent.connector_http.request = AsyncMock(return_value=MagicMock(error=True, data=None, status=500))
    async with agent:
        await agent._perform_visit("https://example.com")
    assert "[Could not fetch URL: 500]" in agent.observations


@pytest.mark.asyncio
async def test_store_chroma_unavailable(ready_agent):
    ready_agent.connector_chroma.chroma_api_ready = False
    ready_agent.connector_chroma.add_to_chroma = AsyncMock()
    await ready_agent._store_chroma(["id1"], [{"a": 1}], ["doc"])
    ready_agent.connector_chroma.add_to_chroma.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_chroma_available(ready_agent):
    ready_agent.connector_chroma.add_to_chroma = AsyncMock()
    await ready_agent._store_chroma(["id1"], [{"a": 1}], ["doc"])
    ready_agent.connector_chroma.add_to_chroma.assert_awaited_once()


@pytest.mark.asyncio
async def test_retrieve_chroma_unavailable(ready_agent):
    ready_agent.connector_chroma.chroma_api_ready = False
    ready_agent.connector_chroma.query_chroma = AsyncMock()
    docs = await ready_agent._retrieve_chroma(["topic"])
    assert docs == []
    ready_agent.connector_chroma.query_chroma.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_chroma_available(ready_agent):
    ready_agent.connector_chroma.query_chroma = AsyncMock(return_value={
        "documents": [["D1", "D2"], ["D3"]]
    })
    docs = await ready_agent._retrieve_chroma(["topic"])
    assert docs == ["D1", "D2", "D3"]


@pytest.mark.asyncio
async def test_final_output(ready_agent):
    ready_agent.deliverables = ["A", "B"]
    ready_agent.notes = ["N1", "N2"]
    ready_agent.connector_llm.query_llm = AsyncMock(return_value=json.dumps({
        "deliverable": "A B",
        "summary": "s"
    }))
    result = await ready_agent._final_output()
    d = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    assert d["success"] is True
    assert "A" in d.get("final_deliverable", "")
    assert "B" in d.get("final_deliverable", "")


@pytest.mark.asyncio
async def test_run_happy_path(ready_agent):
    tick_content = {
        "history_update": "started",
        "note_update": "ok",
        "cache_update": [],
        "cache_retrieve": [],
        "deliverable": "hi",
        "next_action": "exit"
    }
    final_content = {"deliverable": "hi", "summary": "done"}
    ready_agent.connector_llm.query_llm = AsyncMock(side_effect=[
        json.dumps(tick_content),
        json.dumps(final_content),
    ])
    ready_agent.connector_chroma.add_to_chroma = AsyncMock()
    ready_agent.connector_chroma.query_chroma = AsyncMock(return_value={"documents": []})
    async with ready_agent:
        result = await ready_agent.run()
    assert result["success"] is True
    assert "hi" in (result.get("final_deliverable") or "")


@pytest.mark.asyncio
async def test_run_initialization_failure(connectors):
    agent = Agent(mandate="m", max_ticks=1, **connectors)
    agent.connector_chroma.chroma_api_ready = False
    async with agent:
        result = await agent.run()
    assert result["success"] is False
    assert result.get("deliverables") == []


@pytest.mark.asyncio
async def test_initialize_success(ready_agent):
    result = await ready_agent.initialize()
    assert result is True


@pytest.mark.asyncio
async def test_initialize_failure_missing_connector(connectors):
    agent = Agent(mandate="m", max_ticks=1, **connectors)
    agent.connector_llm = None
    result = await agent.initialize()
    assert result is False


@pytest.mark.asyncio
async def test_initialize_failure_not_ready(connectors):
    agent = Agent(mandate="m", max_ticks=1, **connectors)
    agent.connector_chroma.chroma_api_ready = False
    result = await agent.initialize()
    assert result is False
