"""Offline unit tests for the Serper-backed search connector (app/connector_search_serper.py).

Brave's SEARCH_API_KEY ran out of quota (confirmed live, HTTP 402, 2026-08-08) and Serper
(google.serper.dev) became the default provider. Serper's request shape differs from Brave's
(POST + JSON body vs. GET + querystring, ``X-API-KEY`` header vs. ``X-Subscription-Token``) but
its response shape (``organic`` array of ``{title, link, snippet, ...}``) is compatible with the
shared ``_collect()`` parser unchanged. The result-shape test below uses a fixture shaped exactly
like a real Serper response (organic + knowledgeGraph + peopleAlsoAsk + relatedSearches), matching
the live example this connector was built from, to make sure ``_collect`` only pulls from
``organic`` and drops the other top-level sections it doesn't understand.
"""
import types

import pytest

from agent.app.connector_search import ConnectorSearch, create_search_backend
from agent.app.connector_search_serper import ConnectorSearchSerper
from shared.connector_config import ConnectorConfig

# Shaped like a real google.serper.dev/search response for {"q": "apple inc"} — organic entries
# plus the extra top-level sections Serper always includes (knowledgeGraph/peopleAlsoAsk/
# relatedSearches), which _collect() must ignore rather than choke on.
_EXAMPLE_SERPER_RESPONSE = {
    "searchParameters": {"q": "apple inc", "type": "search"},
    "knowledgeGraph": {
        "title": "Apple",
        "type": "Technology company",
        "website": "http://www.apple.com/",
    },
    "organic": [
        {
            "title": "Apple",
            "link": "https://www.apple.com/",
            "snippet": "Discover the innovative world of Apple...",
            "position": 1,
        },
        {
            "title": "Apple Inc. - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "snippet": "Apple Inc. is an American multinational technology company...",
            "position": 2,
        },
    ],
    "peopleAlsoAsk": [
        {"question": "What does Apple Inc do?", "snippet": "...", "link": "https://example.test"},
    ],
    "relatedSearches": [{"query": "apple inc stock"}],
}


def _connector():
    cs = ConnectorSearchSerper(ConnectorConfig())
    cs.search_api_key = "k"
    cs.search_api_ready = True  # skip the live health probe
    return cs


def test_it_is_a_connector_search_so_existing_wiring_accepts_it():
    assert isinstance(_connector(), ConnectorSearch)


def test_url_is_serper_not_brave():
    assert _connector().url == "https://google.serper.dev/search"


@pytest.mark.asyncio
async def test_health_probe_posts_json_with_api_key_header(monkeypatch):
    cs = ConnectorSearchSerper(ConnectorConfig())
    cs.search_api_key = "k"
    captured = {}

    async def fake_request(method, url, retries=2, headers=None, json=None, **kwargs):
        captured.update({"method": method, "url": url, "headers": headers, "json": json})
        return types.SimpleNamespace(error=False, status=200, data={"organic": []})

    monkeypatch.setattr(cs, "request", fake_request)
    assert await cs.init_search_api() is True
    assert captured["method"] == "POST"
    assert captured["url"] == "https://google.serper.dev/search"
    assert captured["headers"]["X-API-KEY"] == "k"
    assert captured["json"] == {"q": "health check"}


@pytest.mark.asyncio
async def test_query_search_maps_real_shaped_serper_response(monkeypatch):
    cs = _connector()
    captured = {}

    async def fake_request(method, url, retries=3, headers=None, json=None):
        captured.update({"method": method, "url": url, "headers": headers, "json": json})
        return types.SimpleNamespace(error=False, status=200, data=_EXAMPLE_SERPER_RESPONSE)

    monkeypatch.setattr(cs, "request", fake_request)
    results = await cs.query_search("apple inc", count=10)

    assert captured["method"] == "POST"
    assert captured["json"] == {"q": "apple inc"}
    assert captured["headers"]["X-API-KEY"] == "k"
    # Only organic[] is mapped — knowledgeGraph/peopleAlsoAsk/relatedSearches are ignored.
    assert results == [
        {
            "title": "Apple",
            "url": "https://www.apple.com/",
            "description": "Discover the innovative world of Apple...",
        },
        {
            "title": "Apple Inc. - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "description": "Apple Inc. is an American multinational technology company...",
        },
    ]


@pytest.mark.asyncio
async def test_query_search_honours_count_client_side(monkeypatch):
    cs = _connector()

    async def fake_request(method, url, retries=3, headers=None, json=None):
        return types.SimpleNamespace(error=False, status=200, data={
            "organic": [
                {"title": str(i), "link": f"https://a.test/{i}", "snippet": ""} for i in range(10)
            ]
        })

    monkeypatch.setattr(cs, "request", fake_request)
    assert len(await cs.query_search("q", count=3)) == 3


@pytest.mark.asyncio
async def test_query_search_raises_on_backend_error(monkeypatch):
    cs = _connector()

    async def fake_request(method, url, retries=3, headers=None, json=None):
        return types.SimpleNamespace(error=True, status=402, data="payment required")

    monkeypatch.setattr(cs, "request", fake_request)
    with pytest.raises(RuntimeError):
        await cs.query_search("q")


def test_factory_defaults_to_serper(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    assert isinstance(create_search_backend(ConnectorConfig()), ConnectorSearchSerper)


def test_factory_picks_serper_explicitly(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "serper")
    assert isinstance(create_search_backend(ConnectorConfig()), ConnectorSearchSerper)


def test_factory_picks_brave_explicitly(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    backend = create_search_backend(ConnectorConfig())
    assert isinstance(backend, ConnectorSearch)
    assert not isinstance(backend, ConnectorSearchSerper)


def test_factory_picks_searxng_explicitly(monkeypatch):
    """The keyless backend is reachable through the factory, not only by hand-instantiation:
    a $0 local benchmark has no other search surface once the paid keys are exhausted."""
    from agent.app.connector_search_searxng import ConnectorSearchXNG

    monkeypatch.setenv("SEARCH_PROVIDER", "searxng")
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test:8080")
    backend = create_search_backend(ConnectorConfig())
    assert isinstance(backend, ConnectorSearchXNG)
    assert backend.url == "http://searxng.test:8080/search"


def test_factory_falls_back_to_serper_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "bing")
    assert isinstance(create_search_backend(ConnectorConfig()), ConnectorSearchSerper)


def test_config_resolves_serper_key_when_provider_is_serper(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("SERPER_KEY", "serper-secret")
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    assert ConnectorConfig().search_api_key == "serper-secret"


def test_config_resolves_brave_key_when_provider_is_brave(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("SEARCH_API_KEY", "brave-secret")
    assert ConnectorConfig().search_api_key == "brave-secret"
