"""
Unit tests for llm_backends factory and OpenRouter header injection.
"""
from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in [
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODEL_API_URL",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_X_TITLE",
    ]:
        monkeypatch.delenv(k, raising=False)
    yield


def test_factory_returns_openai_compatible_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import create_llm_backend, OpenAICompatibleBackend

    c = ConnectorConfig()
    b = create_llm_backend(c, logging.getLogger("t"))
    assert isinstance(b, OpenAICompatibleBackend)


def test_factory_returns_openrouter_when_selected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import create_llm_backend, OpenRouterBackend

    c = ConnectorConfig()
    assert c.llm_provider == "openrouter"
    assert c.llm_api_url == "https://openrouter.ai/api/v1"
    b = create_llm_backend(c, logging.getLogger("t"))
    assert isinstance(b, OpenRouterBackend)


def test_openrouter_resolves_key_priority(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa-fallback")
    from shared.connector_config import ConnectorConfig

    c = ConnectorConfig()
    assert c.llm_api_key == "sk-or-1"


def test_openrouter_backend_attaches_attribution_headers(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_X_TITLE", "TestApp")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import OpenRouterBackend

    captured = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("agent.app.llm_backends.AsyncOpenAI", _FakeClient):
        OpenRouterBackend(ConnectorConfig(), logging.getLogger("t"))

    assert captured["api_key"] == "sk-or-x"
    assert captured["base_url"].endswith("/api/v1")
    headers = captured.get("default_headers") or {}
    assert headers.get("HTTP-Referer") == "https://example.com"
    assert headers.get("X-Title") == "TestApp"


def test_max_completion_tokens_recognizes_slug(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import OpenAICompatibleBackend

    b = OpenAICompatibleBackend(ConnectorConfig(), logging.getLogger("t"))
    assert b._get_max_completion_tokens_limit("openai/gpt-5-mini") == 128000
    assert b._get_max_completion_tokens_limit("gpt-5-mini") == 128000
    assert b._get_max_completion_tokens_limit("anthropic/claude-opus-4.7") == 64000
