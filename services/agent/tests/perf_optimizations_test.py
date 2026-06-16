"""
Tests locking in the performance-optimization behaviors:
- CPU-bound HTML parsing offloaded to a pure/synchronous helper
- Browser connector defaults to speed-over-stealth
- Browser routing blocks heavy resource types
- Separate retry backoff config (retry_base_delay)
- Visit-page concurrency setting present
"""
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from shared.connector_config import ConnectorConfig


SAMPLE_HTML = (
    "<html><head><title>My Title</title></head>"
    "<body><h1>Heading</h1>"
    "<p>This is the main body content about pythons and snakes.</p>"
    "<a href='https://example.com/page2'>Next page</a>"
    "</body></html>"
)


class TestParseVisitHtml:
    """The CPU-bound parse helper must be a pure, synchronous function."""

    def _make_action(self):
        from agent.app.idea_policies.actions import VisitLeafAction
        return VisitLeafAction({"max_links_per_visit": 20})

    def test_parse_returns_expected_fields(self):
        action = self._make_action()
        result = action._parse_visit_html(SAMPLE_HTML, "https://example.com")
        assert result["page_title"] == "My Title"
        assert result["h1_text"] == "Heading"
        assert isinstance(result["cleaned_links"], list)
        assert isinstance(result["content_text"], str)
        assert result["content_total_chars"] >= 0
        # full key set the async caller depends on
        for key in (
            "cleaned", "cleaned_links", "cleaned_link_contexts", "page_title",
            "h1_text", "content_text", "content_payload", "links_for_llm",
            "content_with_links", "final_content", "content_total_chars",
        ):
            assert key in result

    def test_parse_is_synchronous(self):
        import inspect
        action = self._make_action()
        assert not inspect.iscoroutinefunction(action._parse_visit_html)


class TestBrowserDefaults:
    """ConnectorBrowser must default to speed-over-stealth."""

    def test_speed_over_stealth_by_default(self, monkeypatch):
        monkeypatch.delenv("BROWSER_STEALTH_MODE", raising=False)
        from agent.app.connector_browser import ConnectorBrowser
        conn = ConnectorBrowser(ConnectorConfig())
        assert conn._stealth_mode is False
        assert conn._page_load_timeout == 12

    def test_stealth_mode_opt_in(self, monkeypatch):
        monkeypatch.setenv("BROWSER_STEALTH_MODE", "true")
        from agent.app.connector_browser import ConnectorBrowser
        conn = ConnectorBrowser(ConnectorConfig())
        assert conn._stealth_mode is True

    @pytest.mark.asyncio
    async def test_route_handler_blocks_heavy_resources(self):
        from agent.app.connector_browser import ConnectorBrowser
        conn = ConnectorBrowser(ConnectorConfig())

        # image -> aborted
        img_route = MagicMock()
        img_route.request.resource_type = "image"
        img_route.abort = AsyncMock()
        img_route.continue_ = AsyncMock()
        await conn._route_handler(img_route)
        img_route.abort.assert_awaited_once()
        img_route.continue_.assert_not_awaited()

        # document -> allowed through
        doc_route = MagicMock()
        doc_route.request.resource_type = "document"
        doc_route.abort = AsyncMock()
        doc_route.continue_ = AsyncMock()
        await conn._route_handler(doc_route)
        doc_route.continue_.assert_awaited_once()
        doc_route.abort.assert_not_awaited()


class TestConfigAndSettings:
    def test_retry_base_delay_default(self, monkeypatch):
        monkeypatch.delenv("RETRY_BASE_DELAY", raising=False)
        assert ConnectorConfig().retry_base_delay == 0.5

    def test_retry_base_delay_env_override(self, monkeypatch):
        monkeypatch.setenv("RETRY_BASE_DELAY", "0")
        assert ConnectorConfig().retry_base_delay == 0.0

    def test_settings_have_concurrency_and_auto_parallel(self):
        here = os.path.dirname(__file__)
        settings_path = os.path.join(here, "..", "app", "idea_dag_settings.json")
        with open(settings_path) as f:
            settings = json.load(f)
        assert settings.get("visit_page_concurrency", 0) >= 1
        assert settings.get("auto_parallel_siblings") is True
