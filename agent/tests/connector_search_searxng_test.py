"""Offline unit tests for the SearXNG-backed search connector (app/connector_search_searxng.py).

The codebench sandbox runs on an ``--internal`` Docker network whose only web surface is the
bundled SearXNG instance, so the connector must (a) point at ``$SEARXNG_URL``, (b) need no API
key, and (c) map SearXNG's ``results[].content`` onto the same ``title/url/description`` shape
every existing consumer of ``ConnectorSearch`` already expects.
"""
import types

import pytest

from agent.app.connector_search import ConnectorSearch
from agent.app.connector_search_searxng import DEFAULT_SEARXNG_URL, ConnectorSearchXNG
from shared.connector_config import ConnectorConfig


def _connector(monkeypatch, base_url=None, env=None):
    if env is None:
        monkeypatch.delenv("SEARXNG_URL", raising=False)
    else:
        monkeypatch.setenv("SEARXNG_URL", env)
    return ConnectorSearchXNG(ConnectorConfig(), base_url=base_url)


def test_base_url_precedence_explicit_then_env_then_default(monkeypatch):
    assert _connector(monkeypatch).url == f"{DEFAULT_SEARXNG_URL}/search"
    assert _connector(monkeypatch, env="http://sx:8080/").url == "http://sx:8080/search"
    assert _connector(monkeypatch, base_url="http://a:1", env="http://b:2").url == "http://a:1/search"


def test_it_is_a_connector_search_so_existing_wiring_accepts_it(monkeypatch):
    assert isinstance(_connector(monkeypatch), ConnectorSearch)


@pytest.mark.asyncio
async def test_health_probe_needs_no_api_key(monkeypatch):
    cs = _connector(monkeypatch, base_url="http://sx:8080")
    cs.search_api_key = None
    captured = {}

    async def fake_request(method, url, retries=2, params=None, **kwargs):
        captured.update({"url": url, "params": params})
        return types.SimpleNamespace(error=False, status=200, data={"results": []})

    monkeypatch.setattr(cs, "request", fake_request)
    assert await cs.init_search_api() is True
    assert captured["url"] == "http://sx:8080/search"
    assert captured["params"]["format"] == "json"


@pytest.mark.asyncio
async def test_query_search_maps_searxng_results(monkeypatch):
    cs = _connector(monkeypatch, base_url="http://sx:8080")
    cs.search_api_ready = True

    async def fake_request(method, url, retries=3, params=None, **kwargs):
        return types.SimpleNamespace(error=False, status=200, data={"results": [
            {"title": "One", "url": "https://a.test/1", "content": "first"},
            {"title": "Two", "url": "https://a.test/2", "content": "second"},
            {"title": "No URL", "content": "dropped"},
        ]})

    monkeypatch.setattr(cs, "request", fake_request)
    results = await cs.query_search("itertools docs", count=5)
    assert results == [
        {"title": "One", "url": "https://a.test/1", "description": "first"},
        {"title": "Two", "url": "https://a.test/2", "description": "second"},
    ]


@pytest.mark.asyncio
async def test_query_search_honours_count(monkeypatch):
    cs = _connector(monkeypatch, base_url="http://sx:8080")
    cs.search_api_ready = True

    async def fake_request(method, url, retries=3, params=None, **kwargs):
        return types.SimpleNamespace(error=False, status=200, data={"results": [
            {"title": str(i), "url": f"https://a.test/{i}", "content": ""} for i in range(10)
        ]})

    monkeypatch.setattr(cs, "request", fake_request)
    assert len(await cs.query_search("q", count=3)) == 3


@pytest.mark.asyncio
async def test_query_search_raises_on_backend_error(monkeypatch):
    cs = _connector(monkeypatch, base_url="http://sx:8080")
    cs.search_api_ready = True

    async def fake_request(method, url, retries=3, params=None, **kwargs):
        return types.SimpleNamespace(error=True, status=502, data="bad gateway")

    monkeypatch.setattr(cs, "request", fake_request)
    with pytest.raises(RuntimeError):
        await cs.query_search("q")
