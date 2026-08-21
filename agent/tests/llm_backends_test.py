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
        "LLM_NUM_CTX",
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
    # deepseek was uncapped (None) and put the raw 100k-120k finalize/merge budget on the wire,
    # which OpenRouter reserves against the daily credit -> a 402 cliff mid-run.
    assert b._get_max_completion_tokens_limit("deepseek/deepseek-v4-flash") == 32768


# --- Bug #1: reasoning_effort must survive simplify_payload for accepting models ----------
def test_accepts_reasoning_effort_predicate():
    from agent.app.llm_backends import accepts_reasoning_effort

    for accepting in ("gpt-5-mini", "openai/gpt-5-mini", "gpt-5"):
        assert accepts_reasoning_effort(accepting) is True, accepting
    # gpt-4.1 was in the allowlist but is NOT a reasoning model: the param is ignored at best and
    # a provider 400 at worst, so it must not reach the wire.
    for rejecting in ("gpt-4.1", "gpt-4.1-nano", "openai/gpt-4.1-nano", "gpt-4o",
                      "deepseek/deepseek-v4-flash", "anthropic/claude-opus-4.7",
                      "gemini-3.1-pro-preview", "", None):
        assert accepts_reasoning_effort(rejecting) is False, rejecting


def test_simplify_preserves_reasoning_effort_for_gpt5(monkeypatch):
    # The starvation-bug fix: a reasoning-effort hint on gpt-5-mini reaches the wire.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import OpenAICompatibleBackend

    b = OpenAICompatibleBackend(ConnectorConfig(), logging.getLogger("t"))
    payload = {"model": "openai/gpt-5-mini", "reasoning_effort": "minimal", "text": {"verbosity": "low"}}
    out = b.simplify_payload(payload)
    assert out["reasoning_effort"] == "minimal", "gpt-5-mini must keep reasoning_effort"
    # `text` remains stripped (unchanged behavior — the OpenAI chat endpoint rejects it).
    assert "text" not in out


def test_simplify_strips_reasoning_effort_for_non_accepting_model(monkeypatch):
    # A model whose endpoint 400s on reasoning_effort still has it stripped.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import OpenAICompatibleBackend

    b = OpenAICompatibleBackend(ConnectorConfig(), logging.getLogger("t"))
    payload = {"model": "gemini-3.1-pro-preview", "reasoning_effort": "minimal"}
    out = b.simplify_payload(payload)
    assert "reasoning_effort" not in out, "non-accepting model must strip reasoning_effort"


# --- ollama num_ctx truncation: the shim ignores every context override, /api/chat honors it ---
class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _FakeHTTP:
    """Stands in for the backend's httpx.AsyncClient (no network in unit tests)."""

    def __init__(self, chat_payload=None, version_payload=None, version_status=200):
        self.posts = []
        self.gets = []
        self._chat = chat_payload or {
            "message": {"role": "assistant", "content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 18279,
            "eval_count": 12,
        }
        self._version = version_payload
        self._version_status = version_status

    async def post(self, url, json=None):
        self.posts.append((url, json))
        return _FakeResponse(self._chat)

    async def get(self, url):
        self.gets.append(url)
        if self._version is None:
            raise __import__("httpx").ConnectError("refused")
        return _FakeResponse(self._version, status_code=self._version_status)

    async def aclose(self):
        return None


def _ollama_backend(monkeypatch, provider="ollama", url="http://localhost:11435/v1", **fake_kwargs):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("MODEL_API_URL", url)
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import OllamaNativeBackend

    b = OllamaNativeBackend(ConnectorConfig(), logging.getLogger("t"))
    fake = _FakeHTTP(**fake_kwargs)
    b._http = fake
    return b, fake


def test_factory_routes_ollama_provider_to_native_backend(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_API_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import create_llm_backend, OllamaNativeBackend

    b = create_llm_backend(ConnectorConfig(), logging.getLogger("t"))
    assert isinstance(b, OllamaNativeBackend)
    assert b.num_ctx == 32768


def test_factory_routes_self_hosted_openai_compatible_to_native_backend(monkeypatch):
    # The benchmark scripts run ollama as LLM_PROVIDER=openai_compatible + a localhost shim URL,
    # so provider alone would miss exactly the runs that truncated.
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MODEL_API_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import create_llm_backend, OllamaNativeBackend

    assert isinstance(create_llm_backend(ConnectorConfig(), logging.getLogger("t")), OllamaNativeBackend)


def test_factory_leaves_hosted_providers_on_the_openai_path(monkeypatch):
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import (
        create_llm_backend,
        OllamaNativeBackend,
        OpenRouterBackend,
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    hosted = create_llm_backend(ConnectorConfig(), logging.getLogger("t"))
    assert not isinstance(hosted, OllamaNativeBackend)

    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    router = create_llm_backend(ConnectorConfig(), logging.getLogger("t"))
    assert isinstance(router, OpenRouterBackend) and not isinstance(router, OllamaNativeBackend)


def test_factory_num_ctx_zero_disables_the_native_path(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MODEL_API_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("LLM_NUM_CTX", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import create_llm_backend, OllamaNativeBackend

    assert not isinstance(create_llm_backend(ConnectorConfig(), logging.getLogger("t")), OllamaNativeBackend)


def test_is_self_hosted_url_classification():
    from agent.app.llm_backends import is_self_hosted_url

    for local in ("http://localhost:11435/v1", "http://127.0.0.1:11434", "http://192.168.1.9:11434/v1",
                  "http://ollama:11434/v1", "http://host.docker.internal:11434/v1"):
        assert is_self_hosted_url(local) is True, local
    for hosted in ("https://api.openai.com/v1", "https://openrouter.ai/api/v1",
                   "https://api.anthropic.com", "", None):
        assert is_self_hosted_url(hosted) is False, hosted


@pytest.mark.asyncio
async def test_ollama_native_completion_sends_num_ctx(monkeypatch):
    b, fake = _ollama_backend(monkeypatch)
    payload = {
        "model": "qwen2.5:7b",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    text, usage = await b.complete(payload, "qwen2.5:7b")

    url, body = fake.posts[0]
    assert url.endswith("/api/chat"), "must use the NATIVE endpoint; the shim ignores num_ctx"
    assert body["options"]["num_ctx"] == 32768
    assert body["options"]["num_predict"] == 512
    assert body["format"] == "json"
    assert body["messages"] == payload["messages"]
    assert body["stream"] is False
    assert text == "ok"
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (18279, 12, 18291)


@pytest.mark.asyncio
async def test_ollama_native_forwards_json_schema_as_format(monkeypatch):
    b, fake = _ollama_backend(monkeypatch)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    await b.complete(
        {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
        },
        "qwen2.5:7b",
    )
    assert fake.posts[0][1]["format"] == schema


@pytest.mark.asyncio
async def test_ollama_native_logs_served_context(monkeypatch, caplog):
    b, _ = _ollama_backend(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="t")
    await b.complete({"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "hi"}]}, "qwen2.5:7b")
    assert any("18279" in r.getMessage() for r in caplog.records), "served context must be logged"


@pytest.mark.asyncio
async def test_ollama_native_warns_when_served_context_hits_the_ceiling(monkeypatch, caplog):
    # The silent failure this fix exists for: served == the window means the head may be gone.
    b, _ = _ollama_backend(
        monkeypatch,
        chat_payload={
            "message": {"content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 32767,
            "eval_count": 3,
        },
    )
    caplog.set_level(logging.WARNING, logger="t")
    await b.complete({"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "hi"}]}, "qwen2.5:7b")
    assert any("truncated" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_self_hosted_non_ollama_falls_back_to_the_shim(monkeypatch):
    # A private-network vLLM / llama.cpp server fails the /api/version probe and keeps the old path.
    b, fake = _ollama_backend(monkeypatch, provider="openai_compatible")
    calls = {}

    async def _super_complete(payload, model_name):
        calls["payload"] = payload
        return "shimmed", None

    from agent.app.llm_backends import OpenAICompatibleBackend

    monkeypatch.setattr(OpenAICompatibleBackend, "complete", staticmethod(_super_complete), raising=True)
    text, _ = await b.complete({"model": "m", "messages": []}, "m")
    assert text == "shimmed"
    assert fake.posts == [] and fake.gets, "probe runs once, then the shim handles the call"
    assert "num_ctx" not in calls["payload"]


@pytest.mark.asyncio
async def test_openrouter_completion_is_unchanged_and_has_no_num_ctx(monkeypatch):
    # Regression guard: hosted providers must put the payload on the wire byte-identically.
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import create_llm_backend

    b = create_llm_backend(ConnectorConfig(), logging.getLogger("t"))
    captured = {}

    class _Msg:
        content = "hello"

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = object()

    async def _create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    b.client.chat.completions.create = _create
    payload = {"model": "openai/gpt-5-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}
    text, _ = await b.complete(dict(payload), "openai/gpt-5-mini")
    assert text == "hello"
    assert captured == payload, "hosted payload must reach the wire unchanged"
    assert "num_ctx" not in captured and "options" not in captured
