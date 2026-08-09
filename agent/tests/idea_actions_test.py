import os
import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.actions import (
    SearchLeafAction,
    VisitLeafAction,
    SaveLeafAction,
    ThinkLeafAction,
)
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.agent_io import AgentIO
from shared.connector_config import ConnectorConfig


class FakeIO:
    """
    Minimal IO stub for unit tests.
    Provides search, fetch_url, visit, store_chroma, retrieve_chroma.
    """

    def __init__(self):
        self.last_search = None
        self.last_visit = None
        self.last_save = None
        self.telemetry = None
        self.connector_chroma = None

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

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return None


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
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.SEARCH.value
    assert payload[ActionResultKey.QUERY.value] == "fish"
    assert payload[ActionResultKey.COUNT.value] == 2
    assert len(payload[ActionResultKey.RESULTS.value]) == 2
    assert payload[ActionResultKey.SUCCESS.value] is True


@pytest.mark.asyncio
async def test_search_action_uses_title_as_fallback_query():
    """When no query is provided, the node title is used as the search query."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "search for cats", details={})
    io, cleanup = await _get_io(require_search=True)
    action = SearchLeafAction()
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert payload[ActionResultKey.QUERY.value] == "search for cats"


@pytest.mark.asyncio
async def test_visit_action_with_url():
    """Visit action with explicit url in details."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "visit", details={"url": "https://example.com"})
    io, cleanup = await _get_io()
    action = VisitLeafAction(settings={"max_links_per_visit": 3, "max_observation_chars": 10})
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.URL.value] == "https://example.com"
    assert "links" in payload


@pytest.mark.asyncio
async def test_visit_action_with_optional_url():
    """Visit action using optional_url (the preferred field name)."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "visit",
        details={"optional_url": "https://example.com", "link_count": 1},
    )
    io, cleanup = await _get_io()
    action = VisitLeafAction(settings={"max_links_per_visit": 3, "max_observation_chars": 10})
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
    assert payload[ActionResultKey.URL.value] == "https://example.com"
    assert payload[ActionResultKey.SUCCESS.value] is True


@pytest.mark.asyncio
async def test_visit_action_returns_content_fields():
    """Visit result contains content, content_full, links, page metadata."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "visit", details={"url": "https://example.com"})
    io, cleanup = await _get_io()
    action = VisitLeafAction(settings={"max_links_per_visit": 3, "max_observation_chars": 100000})
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload[ActionResultKey.SUCCESS.value] is True
    assert ActionResultKey.CONTENT.value in payload
    assert "content_full" in payload
    assert "links" in payload
    assert "links_count" in payload


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
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.SAVE.value
    assert payload[ActionResultKey.COUNT.value] == 1
    assert payload[ActionResultKey.SUCCESS.value] is True
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
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.THINK.value
    assert payload[ActionResultKey.NODE_ID.value] == node.node_id
    assert payload[ActionResultKey.SUCCESS.value] is True


@pytest.mark.asyncio
async def test_search_action_failure_returns_retryable():
    """A search action that fails should return a failure result with retryable info."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "search", details={"query": "fish"})

    class FailingIO(FakeIO):
        async def search(self, query, count=10, timeout_seconds=None):
            raise RuntimeError("Network error")

    io = FailingIO()
    action = SearchLeafAction()
    payload = await action.execute(graph, node.node_id, io)
    assert payload[ActionResultKey.SUCCESS.value] is False
    assert ActionResultKey.ERROR.value in payload
    assert payload[ActionResultKey.QUERY.value] == "fish"


@pytest.mark.asyncio
async def test_visit_action_clears_placeholder_urls():
    """Placeholder URLs like '<chosen_next_url>' should be cleared, not visited."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "visit",
        details={"optional_url": "<chosen_next_url from search>"},
    )
    io, cleanup = await _get_io()
    action = VisitLeafAction(settings={"max_links_per_visit": 1, "max_observation_chars": 10})
    try:
        payload = await action.execute(graph, node.node_id, io)
    finally:
        if cleanup:
            await cleanup()
    assert payload[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
