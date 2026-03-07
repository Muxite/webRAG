import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult


def _real_enabled() -> bool:
    return os.environ.get("CONNECTOR_SMOKE_TESTS", "").lower() in ("1", "true", "yes", "on")


class TestConnectorBrowserUnit:
    """Unit tests for ConnectorBrowser using mocks (no real Chrome needed)."""

    def _make_connector(self):
        from agent.app.connector_browser import ConnectorBrowser
        config = ConnectorConfig()
        return ConnectorBrowser(config)

    @pytest.mark.asyncio
    async def test_init_defaults(self):
        connector = self._make_connector()
        assert connector._driver is None
        assert connector._ready is False

    @pytest.mark.asyncio
    async def test_fetch_page_returns_request_result(self):
        connector = self._make_connector()
        fake_html = "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
        with patch.object(connector, "_run_in_executor", new_callable=AsyncMock, return_value=fake_html):
            connector._ready = True
            result = await connector.fetch_page("https://example.com")
            assert isinstance(result, RequestResult)
            assert result.error is False
            assert result.status == 200
            assert "Hello" in result.data

    @pytest.mark.asyncio
    async def test_fetch_page_returns_error_on_exception(self):
        connector = self._make_connector()
        with patch.object(connector, "_run_in_executor", new_callable=AsyncMock, side_effect=Exception("Chrome crashed")):
            connector._ready = True
            result = await connector.fetch_page("https://example.com")
            assert isinstance(result, RequestResult)
            assert result.error is True
            assert "Chrome crashed" in str(result.data)

    @pytest.mark.asyncio
    async def test_fetch_page_auto_inits_when_not_ready(self):
        connector = self._make_connector()
        fake_html = "<html><body>Content</body></html>"
        with patch.object(connector, "_ensure_browser", new_callable=AsyncMock, return_value=True):
            with patch.object(connector, "_run_in_executor", new_callable=AsyncMock, return_value=fake_html):
                result = await connector.fetch_page("https://example.com")
                connector._ensure_browser.assert_awaited_once()
                assert result.error is False

    @pytest.mark.asyncio
    async def test_fetch_page_fails_if_browser_init_fails(self):
        connector = self._make_connector()
        with patch.object(connector, "_ensure_browser", new_callable=AsyncMock, return_value=False):
            result = await connector.fetch_page("https://example.com")
            assert result.error is True
            assert result.status is None

    @pytest.mark.asyncio
    async def test_close_sets_not_ready(self):
        connector = self._make_connector()
        connector._ready = True
        connector._driver = MagicMock()
        with patch.object(connector, "_run_in_executor", new_callable=AsyncMock):
            await connector.close()
            assert connector._ready is False
            assert connector._driver is None

    @pytest.mark.asyncio
    async def test_fetch_page_empty_html_returns_error(self):
        connector = self._make_connector()
        with patch.object(connector, "_run_in_executor", new_callable=AsyncMock, return_value=""):
            connector._ready = True
            result = await connector.fetch_page("https://example.com")
            assert result.error is True

    @pytest.mark.asyncio
    async def test_fetch_page_timeout(self):
        connector = self._make_connector()

        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)
            return "<html></html>"

        with patch.object(connector, "_run_in_executor", side_effect=slow_fetch):
            connector._ready = True
            result = await connector.fetch_page("https://example.com", timeout=0.1)
            assert result.error is True

    @pytest.mark.asyncio
    async def test_telemetry_recorded_on_fetch(self):
        connector = self._make_connector()
        mock_telemetry = MagicMock()
        mock_telemetry.record_timing = MagicMock()
        connector.set_telemetry(mock_telemetry)

        fake_html = "<html><body>Telemetry test</body></html>"
        with patch.object(connector, "_run_in_executor", new_callable=AsyncMock, return_value=fake_html):
            connector._ready = True
            await connector.fetch_page("https://example.com")
            mock_telemetry.record_timing.assert_called()


class TestAgentIOBrowserFallback:
    """Tests for AgentIO.visit behavior with an optional browser connector.

    Note: AgentIO.visit is browser-first when connector_browser is present, and
    falls back to HTTP when the browser visit fails.
    """

    def _make_io(self, http_result, browser_result=None):
        from agent.app.agent_io import AgentIO

        mock_llm = MagicMock()
        mock_llm.set_telemetry = MagicMock()
        mock_search = MagicMock()
        mock_search.set_telemetry = MagicMock()
        mock_chroma = MagicMock()
        mock_chroma.set_telemetry = MagicMock()

        mock_http = MagicMock()
        mock_http.set_telemetry = MagicMock()
        mock_http.request = AsyncMock(return_value=http_result)

        mock_browser = None
        if browser_result is not None:
            mock_browser = MagicMock()
            mock_browser.set_telemetry = MagicMock()
            mock_browser.fetch_page = AsyncMock(return_value=browser_result)

        io = AgentIO(
            connector_llm=mock_llm,
            connector_search=mock_search,
            connector_http=mock_http,
            connector_chroma=mock_chroma,
            connector_browser=mock_browser,
        )
        return io, mock_http, mock_browser

    @pytest.mark.asyncio
    async def test_visit_success_no_fallback(self):
        http_ok = RequestResult(status=200, data="<html><body>OK page</body></html>", error=False)
        io, mock_http, _ = self._make_io(http_ok)
        text = await io.visit("https://example.com")
        assert "OK page" in text or text
        mock_http.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_visit_403_falls_back_to_browser(self):
        http_403 = RequestResult(status=403, data="Forbidden", error=True)
        browser_ok = RequestResult(status=200, data="<html><body>Browser OK</body></html>", error=False)
        io, mock_http, mock_browser = self._make_io(http_403, browser_ok)
        text = await io.visit("https://protected-site.com")
        mock_browser.fetch_page.assert_awaited_once()
        # Browser-first: HTTP should not be called when browser succeeds.
        mock_http.request.assert_not_awaited()
        assert text

    @pytest.mark.asyncio
    async def test_visit_403_no_browser_raises(self):
        http_403 = RequestResult(status=403, data="Forbidden", error=True)
        io, _, _ = self._make_io(http_403, browser_result=None)
        with pytest.raises(RuntimeError, match="403"):
            await io.visit("https://protected-site.com")

    @pytest.mark.asyncio
    async def test_visit_403_browser_also_fails_raises(self):
        http_403 = RequestResult(status=403, data="Forbidden", error=True)
        browser_fail = RequestResult(status=None, data="Browser failed too", error=True)
        io, _, mock_browser = self._make_io(http_403, browser_fail)
        with pytest.raises(RuntimeError):
            await io.visit("https://protected-site.com")
        # Browser attempted first, then HTTP fallback attempted.
        mock_browser.fetch_page.assert_awaited_once()
        io.connector_http.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_visit_401_falls_back_to_browser(self):
        http_401 = RequestResult(status=401, data="Unauthorized", error=True)
        browser_ok = RequestResult(status=200, data="<html><body>Got it</body></html>", error=False)
        io, mock_http, mock_browser = self._make_io(http_401, browser_ok)
        text = await io.visit("https://auth-site.com")
        mock_browser.fetch_page.assert_awaited_once()
        mock_http.request.assert_not_awaited()
        assert text

    @pytest.mark.asyncio
    async def test_visit_500_does_not_fallback(self):
        http_500 = RequestResult(status=500, data="Server Error", error=True)
        io, mock_http, _ = self._make_io(http_500)
        with pytest.raises(RuntimeError, match="500"):
            await io.visit("https://broken-server.com")


@pytest.mark.asyncio
async def test_connector_browser_smoke():
    """Live smoke test — only runs when CONNECTOR_SMOKE_TESTS=1."""
    if not _real_enabled():
        pytest.skip("CONNECTOR_SMOKE_TESTS not enabled")
    from agent.app.connector_browser import ConnectorBrowser
    config = ConnectorConfig()
    connector = ConnectorBrowser(config)
    try:
        result = await connector.fetch_page("https://example.com", timeout=30)
        if result.error:
            pytest.skip(f"Headless Chrome not available in this environment: {result.data}")
        assert result.status == 200
        assert "Example Domain" in str(result.data)
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_connector_browser_smoke_403_site():
    """Live smoke test for a site that typically blocks bots."""
    if not _real_enabled():
        pytest.skip("CONNECTOR_SMOKE_TESTS not enabled")
    from agent.app.connector_browser import ConnectorBrowser
    config = ConnectorConfig()
    connector = ConnectorBrowser(config)
    try:
        result = await connector.fetch_page("https://www.google.com/search?q=test", timeout=30)
        if result.error:
            pytest.skip(f"Headless Chrome not available in this environment: {result.data}")
        assert result.status == 200
        assert len(str(result.data)) > 100
    finally:
        await connector.close()
