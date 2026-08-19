"""Coverage for the connector reachability API endpoints."""
from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from shared.request_result import RequestResult

from app.main import create_app
from tests.conftest import FakeBrowser, FakeHttp, FakeSearch


# --------------------------------------------------------------------------- #
# /search
# --------------------------------------------------------------------------- #
def test_search_success(make_client):
    results = [
        {"title": "Euglena", "url": "https://en.wikipedia.org/wiki/Euglena", "description": "A genus."},
        {"title": "Other", "url": "https://example.com", "description": ""},
    ]
    client = make_client(search=FakeSearch(results=results))
    resp = client.post("/search", json={"query": "euglena", "count": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "euglena"
    assert body["count"] == 5
    assert body["result_count"] == 2
    assert body["results"][0]["url"] == "https://en.wikipedia.org/wiki/Euglena"
    assert body["results"][1]["description"] == ""


def test_search_empty_results(make_client):
    client = make_client(search=FakeSearch(results=[]))
    resp = client.post("/search", json={"query": "no-such-thing"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_count"] == 0
    assert body["results"] == []


def test_search_connector_not_ready_returns_503(make_client):
    # query_search returns None when the API key is missing/invalid.
    client = make_client(search=FakeSearch(results=None))
    resp = client.post("/search", json={"query": "x"})
    assert resp.status_code == 503
    assert "not ready" in resp.json()["detail"].lower()


def test_search_provider_error_returns_502(make_client):
    client = make_client(search=FakeSearch(error=RuntimeError("Search API query failed: status=429")))
    resp = client.post("/search", json={"query": "x"})
    assert resp.status_code == 502
    assert "429" in resp.json()["detail"]


def test_search_connector_uninitialized_returns_503(make_client):
    # No lifespan run, no injected search -> state stays None.
    client = make_client(search=None, http=FakeHttp(), browser=FakeBrowser())
    resp = client.post("/search", json={"query": "x"})
    assert resp.status_code == 503
    assert "not initialized" in resp.json()["detail"].lower()


def test_search_malformed_missing_query_returns_422(make_client):
    client = make_client(search=FakeSearch(results=[]))
    resp = client.post("/search", json={"count": 3})
    assert resp.status_code == 422


def test_search_malformed_wrong_type_returns_422(make_client):
    client = make_client(search=FakeSearch(results=[]))
    resp = client.post("/search", json={"query": "x", "count": "many"})
    assert resp.status_code == 422


def test_search_count_out_of_range_returns_422(make_client):
    client = make_client(search=FakeSearch(results=[]))
    resp = client.post("/search", json={"query": "x", "count": 999})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /visit
# --------------------------------------------------------------------------- #
def test_visit_success_via_http(make_client):
    html = "<html><body>hello world</body></html>"
    client = make_client(http=FakeHttp(result=RequestResult(status=200, data=html, error=False)))
    resp = client.post("/visit", json={"url": "https://example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["status"] == 200
    assert body["via"] == "http"
    assert body["browser_fallback_used"] is False
    assert body["content_chars"] == len(html)
    assert body["content"] == html
    assert body["content_truncated"] is False
    assert body["reason"] is None


def test_visit_content_truncation(make_client):
    html = "x" * 100
    client = make_client(http=FakeHttp(result=RequestResult(status=200, data=html, error=False)))
    resp = client.post("/visit", json={"url": "https://example.com", "max_content_chars": 10})
    body = resp.json()
    assert body["content_chars"] == 100
    assert body["content"] == "x" * 10
    assert body["content_truncated"] is True


def test_visit_content_omitted_when_cap_zero(make_client):
    client = make_client(http=FakeHttp(result=RequestResult(status=200, data="abc", error=False)))
    resp = client.post("/visit", json={"url": "https://example.com", "max_content_chars": 0})
    body = resp.json()
    assert body["reachable"] is True
    assert body["content"] is None
    assert body["content_chars"] == 3


def test_visit_404_is_reachable_false_with_reason(make_client):
    client = make_client(
        http=FakeHttp(result=RequestResult(status=404, data="Not Found", error=True)),
        browser=FakeBrowser(),
    )
    resp = client.post("/visit", json={"url": "https://example.com/missing"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False
    assert body["status"] == 404
    assert "404" in body["reason"]
    assert body["browser_fallback_used"] is False  # 404 does not trigger fallback


def test_visit_connection_error_is_reachable_false(make_client):
    # A status=None *error result* (not a raised exception) mirrors the engine:
    # no browser fallback is triggered, and the transport reason is surfaced.
    client = make_client(
        http=FakeHttp(result=RequestResult(status=None, data="Request failed: Cannot connect to host", error=True)),
        browser=FakeBrowser(result=RequestResult(status=200, data="unused", error=False)),
    )
    resp = client.post("/visit", json={"url": "https://nonexistent.invalid"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False
    assert body["status"] is None
    assert body["browser_fallback_used"] is False
    assert "cannot connect" in body["reason"].lower()


def test_visit_request_raises_connection_error(make_client):
    # If request() itself raises (not just returns an error result).
    client = make_client(
        http=FakeHttp(exc=ConnectionError("connection refused")),
        browser=FakeBrowser(result=RequestResult(status=None, data="Browser not available", error=True)),
    )
    resp = client.post("/visit", json={"url": "https://refused.invalid"})
    body = resp.json()
    assert body["reachable"] is False
    assert body["browser_fallback_used"] is True


def test_visit_timeout(make_client):
    async def _slow_request(*args, **kwargs):
        await asyncio.sleep(1.0)
        return RequestResult(status=200, data="late", error=False)

    http = FakeHttp()
    http.request = _slow_request  # type: ignore[assignment]
    client = make_client(http=http, browser=None)
    resp = client.post("/visit", json={"url": "https://slow.example", "timeout": 0.01})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False
    assert "timeout" in body["reason"].lower()


def test_visit_browser_fallback_success_on_403(make_client):
    html = "<html>unblocked by browser</html>"
    client = make_client(
        http=FakeHttp(result=RequestResult(status=403, data="Forbidden", error=True)),
        browser=FakeBrowser(result=RequestResult(status=200, data=html, error=False)),
    )
    resp = client.post("/visit", json={"url": "https://bot-blocked.example"})
    body = resp.json()
    assert body["reachable"] is True
    assert body["via"] == "browser"
    assert body["browser_fallback_used"] is True
    assert body["content"] == html


def test_visit_browser_fallback_timeout(make_client):
    async def _slow_fetch(*args, **kwargs):
        await asyncio.sleep(1.0)
        return RequestResult(status=200, data="late", error=False)

    browser = FakeBrowser()
    browser.fetch_page = _slow_fetch  # type: ignore[assignment]
    client = make_client(
        http=FakeHttp(result=RequestResult(status=403, data="Forbidden", error=True)),
        browser=browser,
    )
    resp = client.post("/visit", json={"url": "https://blocked.example", "timeout": 0.01})
    body = resp.json()
    assert body["reachable"] is False
    assert body["browser_fallback_used"] is True
    assert "timeout" in body["reason"].lower()


def test_visit_browser_fallback_raises(make_client):
    client = make_client(
        http=FakeHttp(result=RequestResult(status=401, data="Unauthorized", error=True)),
        browser=FakeBrowser(exc=RuntimeError("playwright boom")),
    )
    resp = client.post("/visit", json={"url": "https://blocked.example"})
    body = resp.json()
    assert body["reachable"] is False
    assert body["browser_fallback_used"] is True
    assert "browser error" in body["reason"].lower()


def test_visit_http_connector_uninitialized_returns_503(make_client):
    client = make_client(http=None)  # no lifespan, stays None
    resp = client.post("/visit", json={"url": "https://example.com"})
    assert resp.status_code == 503


def test_visit_malformed_missing_url_returns_422(make_client):
    client = make_client(http=FakeHttp())
    resp = client.post("/visit", json={"timeout": 1.0})
    assert resp.status_code == 422


def test_visit_malformed_bad_timeout_returns_422(make_client):
    client = make_client(http=FakeHttp())
    resp = client.post("/visit", json={"url": "https://example.com", "timeout": -5})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_all_initialized(make_client):
    client = make_client(search=FakeSearch(), http=FakeHttp(), browser=FakeBrowser())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    comps = body["components"]
    assert comps["process"] is True
    assert comps["search_connector"] is True
    assert comps["http_connector"] is True
    assert comps["browser_connector"] is True


def test_health_reports_degraded_search_connector(make_client):
    # Search connector never initialized -> component flag is False, but the
    # process-level status stays healthy (informational component health).
    client = make_client(search=None, http=FakeHttp(), browser=FakeBrowser())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["components"]["search_connector"] is False


# --------------------------------------------------------------------------- #
# lifespan (real connector construction + cleanup paths)
# --------------------------------------------------------------------------- #
def test_lifespan_constructs_and_cleans_up(monkeypatch):
    # With TestClient as a context manager, startup builds real connectors and
    # shutdown tears them down. Injecting fakes exercises the "already set" and
    # cleanup branches without touching the network.
    fake_search = FakeSearch(results=[])
    fake_http = FakeHttp(result=RequestResult(status=200, data="ok", error=False))
    fake_browser = FakeBrowser()
    app = create_app(search=fake_search, http=fake_http, browser=fake_browser)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
    assert fake_search.reset_called is True
    assert fake_http.reset_called is True
    assert fake_browser.closed is True


def test_lifespan_builds_real_connectors_when_none():
    # No injected connectors: lifespan constructs the real ones. They are built
    # lazily (no network) so /health can report them as initialized.
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        comps = resp.json()["components"]
        assert comps["search_connector"] is True
        assert comps["http_connector"] is True
        assert comps["browser_connector"] is True


def test_openapi_docs_available():
    app = create_app(search=FakeSearch(), http=FakeHttp(), browser=FakeBrowser())
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
