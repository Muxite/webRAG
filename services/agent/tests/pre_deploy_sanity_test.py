"""
Pre-deployment sanity tests. Run on every agent-test to catch import errors,
indentation bugs, and core fetch/visit behavior before deployment.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult


def test_imports_all_critical_modules():
    """
    Import chain from main entrypoint. Catches IndentationError, SyntaxError,
    and missing dependencies before deployment.
    """
    from agent.app.main import health_handler
    from agent.app.interface_agent import InterfaceAgent
    from agent.app.idea_engine import IdeaDagEngine
    from agent.app.agent_io import AgentIO
    from agent.app.connector_browser import ConnectorBrowser, BROWSER_FALLBACK_STATUSES
    from agent.app.connector_http import ConnectorHttp
    assert health_handler is not None
    assert InterfaceAgent is not None
    assert IdeaDagEngine is not None
    assert AgentIO is not None
    assert ConnectorBrowser is not None
    assert BROWSER_FALLBACK_STATUSES == {401, 403}
    assert ConnectorHttp is not None


@pytest.mark.asyncio
async def test_agent_io_http_first_success_no_browser():
    """
    HTTP succeeds first; browser never called when HTTP returns 200.
    """
    from agent.app.agent_io import AgentIO

    http_ok = RequestResult(status=200, data="<html><body>OK</body></html>", error=False)
    mock_llm = MagicMock()
    mock_llm.set_telemetry = MagicMock()
    mock_search = MagicMock()
    mock_search.set_telemetry = MagicMock()
    mock_chroma = MagicMock()
    mock_chroma.set_telemetry = MagicMock()
    mock_http = MagicMock()
    mock_http.set_telemetry = MagicMock()
    mock_http.request = AsyncMock(return_value=http_ok)
    mock_browser = MagicMock()
    mock_browser.set_telemetry = MagicMock()
    mock_browser.fetch_page = AsyncMock()

    io = AgentIO(
        connector_llm=mock_llm,
        connector_search=mock_search,
        connector_http=mock_http,
        connector_chroma=mock_chroma,
        connector_browser=mock_browser,
    )
    text = await io.visit("https://example.com")
    mock_http.request.assert_awaited_once()
    mock_browser.fetch_page.assert_not_awaited()
    assert "OK" in text or text


@pytest.mark.asyncio
async def test_agent_io_http_403_falls_back_to_browser():
    """
    HTTP 403 triggers browser fallback; browser success returns content.
    """
    from agent.app.agent_io import AgentIO

    http_403 = RequestResult(status=403, data="Forbidden", error=True)
    browser_ok = RequestResult(status=200, data="<html><body>Browser OK</body></html>", error=False)
    mock_llm = MagicMock()
    mock_llm.set_telemetry = MagicMock()
    mock_search = MagicMock()
    mock_search.set_telemetry = MagicMock()
    mock_chroma = MagicMock()
    mock_chroma.set_telemetry = MagicMock()
    mock_http = MagicMock()
    mock_http.set_telemetry = MagicMock()
    mock_http.request = AsyncMock(return_value=http_403)
    mock_browser = MagicMock()
    mock_browser.set_telemetry = MagicMock()
    mock_browser.fetch_page = AsyncMock(return_value=browser_ok)

    io = AgentIO(
        connector_llm=mock_llm,
        connector_search=mock_search,
        connector_http=mock_http,
        connector_chroma=mock_chroma,
        connector_browser=mock_browser,
    )
    text = await io.visit("https://protected.example.com")
    mock_http.request.assert_awaited_once()
    mock_browser.fetch_page.assert_awaited_once()
    assert "Browser OK" in text or text


@pytest.mark.asyncio
async def test_connector_browser_permanently_unavailable_skips_retry():
    """
    After first _ensure_browser failure, fetch_page returns immediately without
    retrying driver creation.
    """
    from agent.app.connector_browser import ConnectorBrowser

    config = ConnectorConfig()
    connector = ConnectorBrowser(config)
    connector._permanently_unavailable = True
    result = await connector.fetch_page("https://example.com")
    assert result.error is True
    assert "Browser not available" in str(result.data)


@pytest.mark.asyncio
async def test_connector_browser_permanently_unavailable_set_on_init_failure():
    """
    First _ensure_browser failure sets _permanently_unavailable; second call
    skips _ensure_browser.
    """
    from agent.app.connector_browser import ConnectorBrowser

    config = ConnectorConfig()
    connector = ConnectorBrowser(config)

    async def mock_ensure_fail():
        connector._permanently_unavailable = True
        return False

    with patch.object(connector, "_ensure_browser", side_effect=mock_ensure_fail):
        r1 = await connector.fetch_page("https://example.com")
        r2 = await connector.fetch_page("https://example.com")
    assert r1.error is True
    assert r2.error is True
    assert connector._permanently_unavailable is True
