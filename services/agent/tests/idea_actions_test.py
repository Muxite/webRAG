import os
import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.actions import (
    SearchLeafAction,
    VisitLeafAction,
    SaveLeafAction,
    ThinkLeafAction,
)
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.agent_io import AgentIO
from shared.connector_config import ConnectorConfig


class FakeIO:
    def __init__(self):
        self.last_search = None
        self.last_visit = None
        self.last_save = None

    async def search(self, query: str, count: int = 10, timeout_seconds=None):
        self.last_search = {"query": query, "count": count}
        return [
            {"title": "A", "url": "https://a.example", "description": "a"},
            {"title": "B", "url": "https://b.example", "description": "b"},
        ]

    async def fetch_url(self, url: str, retries: int = 3, timeout_seconds=None) -> str:
        self.last_visit = {"url": url, "retries": retries}
        return "<html><body><a href='https://x.example'>X</a><p>Alpha</p></body></html>"

    async def visit(self, url: str) -> str:
        self.last_visit = {"url": url}
        return "Alpha"

    async def store_chroma(self, documents, metadatas, ids, timeout_seconds=None):
        self.last_save = {"documents": documents, "metadatas": metadatas, "ids": ids}
        return True

    async def retrieve_chroma(self, topics, n_results: int = 3, timeout_seconds=None):
        return ["Doc A", "Doc B"]


class DummyLLM:
    def set_telemetry(self, telemetry):
        return None

    def build_payload(self, messages, json_mode, model_name=None, temperature=0.5, max_tokens=4096):
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if model_name:
            payload["model"] = model_name
        return payload

    async def query_llm(self, payload, model_name=None):
        return None

    def pop_last_usage(self):
        return None


async def _build_real_io(require_search: bool = False, require_chroma: bool = False):
    config = ConnectorConfig()
    if require_search and not config.search_api_key:
        pytest.skip("SEARCH_API_KEY not configured for real IO")
    if require_chroma and not config.chroma_url:
        pytest.skip("CHROMA_URL not configured for real IO")

    search = ConnectorSearch(config)
    http = ConnectorHttp(config)
    chroma = ConnectorChroma(config)
    llm = DummyLLM()

    await search.__aenter__()
    await http.__aenter__()
    if require_search:
        await search.init_search_api()
    if require_chroma:
        await chroma.init_chroma()

    io = AgentIO(
        connector_llm=llm,
        connector_search=search,
        connector_http=http,
        connector_chroma=chroma,
    )

    async def _cleanup():
        await search.__aexit__(None, None, None)
        await http.__aexit__(None, None, None)

    return io, _cleanup


async def _get_io(require_search: bool = False, require_chroma: bool = False):
    use_real = os.environ.get("IDEA_ACTIONS_USE_REAL_IO", "").lower() in ("1", "true", "yes", "on")
    if not use_real:
        return FakeIO(), None
    return await _build_real_io(require_search=require_search, require_chroma=require_chroma)


@pytest.mark.asyncio
async def test_search_action():
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "search", details={"query": "fish", "count": 2})
    io, cleanup = await _get_io(require_search=True)
    action = SearchLeafAction()
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload["action"] == "search"
    assert payload["query"] == "fish"
    assert payload["count"] == 2
    assert len(payload["results"]) == 2


@pytest.mark.asyncio
async def test_visit_action():
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "visit", details={"url": "https://example.com"})
    io, cleanup = await _get_io()
    action = VisitLeafAction(settings={"max_links_per_visit": 3, "max_observation_chars": 10})
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload["action"] == "visit"
    assert payload["url"] == "https://example.com"
    assert "links" in payload


@pytest.mark.asyncio
async def test_save_action():
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "save",
        details={"documents": ["Doc A"], "metadatas": [{"title": "A"}]},
    )
    io, cleanup = await _get_io(require_chroma=True)
    action = SaveLeafAction()
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload["action"] == "save"
    assert payload["count"] == 1
    assert payload["success"] is True
    if hasattr(io, "last_save"):
        assert io.last_save is not None


@pytest.mark.asyncio
async def test_think_action():
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "think", details={"note": "x"})
    io, cleanup = await _get_io()
    action = ThinkLeafAction()
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload["action"] == "think"
    assert payload["node_id"] == node.node_id
