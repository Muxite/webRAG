"""
Live startup-preflight tests (no mocks).

These are gated behind env flags because they require outbound internet access
and may be blocked in some CI/VPC environments.
"""

import os

import pytest

from agent.app.connector_browser import ConnectorBrowser
from agent.app.connector_http import ConnectorHttp
from agent.app.startup_preflight import run_startup_preflight
from shared.connector_config import ConnectorConfig


def _live_enabled() -> bool:
    return os.environ.get("LIVE_PREFLIGHT_TESTS", "").lower() in ("1", "true", "yes", "on")


def _browser_live_enabled() -> bool:
    return os.environ.get("LIVE_PREFLIGHT_BROWSER_TESTS", "").lower() in ("1", "true", "yes", "on")


@pytest.mark.asyncio
async def test_startup_preflight_hits_wikipedia_live_http():
    """
    Verifies real outbound HTTP can retrieve a large Wikipedia page.
    """
    if not _live_enabled():
        pytest.skip("LIVE_PREFLIGHT_TESTS not enabled")

    config = ConnectorConfig()
    http = ConnectorHttp(config)
    await http.__aenter__()
    try:
        result = await run_startup_preflight(
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            connector_http=http,
            connector_browser=None,
            timeout_seconds=15.0,
            retries=1,
            enable_browser=False,
            # "shitload of chars" threshold; keep comfortably above the default.
            min_content_chars=20000,
        )
        assert result.http_ok is True
        assert result.http_content_chars >= 20000
    finally:
        await http.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_startup_preflight_hits_wikipedia_live_browser():
    """
    Optional: Verifies browser connector can start and fetch a large Wikipedia page.
    """
    if not (_live_enabled() and _browser_live_enabled()):
        pytest.skip("LIVE_PREFLIGHT_TESTS and LIVE_PREFLIGHT_BROWSER_TESTS required")

    config = ConnectorConfig()
    http = ConnectorHttp(config)
    browser = ConnectorBrowser(config)
    await http.__aenter__()
    try:
        result = await run_startup_preflight(
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            connector_http=http,
            connector_browser=browser,
            timeout_seconds=30.0,
            retries=1,
            enable_browser=True,
            min_content_chars=20000,
        )
        assert result.http_ok is True
        assert result.http_content_chars >= 20000
        assert result.browser_attempted is True
        assert result.browser_ok is True
        assert (result.browser_content_chars or 0) >= 20000
    finally:
        await http.__aexit__(None, None, None)
        try:
            await browser.close()
        except Exception:
            pass

