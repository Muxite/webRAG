"""Per-cell model-identity capture (DAG v3 plan §4A/§8 fairness floor).

A benchmark cell recorded only the model TAG, so two arms served different quantizations or
different digests of "the same" model were indistinguishable in the result JSON. These pin
the capture: what a local Ollama cell records, that a hosted cell claims nothing it cannot
know, and that a probe failure degrades to a recorded ``error`` instead of failing the cell.

No network: the httpx client is stubbed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.app.testing import model_metadata as mm


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        _FakeClient.calls.append(("post", url, json))
        return _Resp({
            "details": {
                "quantization_level": "Q4_K_M", "parameter_size": "7.6B", "family": "qwen2",
            },
            "model_info": {"qwen2.context_length": 32768, "qwen2.block_count": 28},
            "capabilities": ["completion", "tools"],
        })

    async def get(self, url):
        _FakeClient.calls.append(("get", url, None))
        return _Resp({"models": [
            {"name": "llama3.2:3b", "digest": "aaa", "size": 1},
            {"name": "qwen2.5:7b", "digest": "sha256:beef", "size": 4683087332},
        ]})


class _BoomClient(_FakeClient):
    async def post(self, url, json=None):
        raise RuntimeError("connection refused")


def _connector(url, provider="openai_compatible", num_ctx=16384):
    return SimpleNamespace(
        config=SimpleNamespace(llm_api_url=url, llm_provider=provider, llm_num_ctx=num_ctx),
        _backend=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    mm._CACHE.clear()
    _FakeClient.calls = []
    yield
    mm._CACHE.clear()


def test_local_ollama_cell_records_digest_quantization_context_and_tool_support(monkeypatch):
    monkeypatch.setattr(mm.httpx, "AsyncClient", _FakeClient)
    rec = asyncio.run(
        mm.collect_model_metadata(_connector("http://localhost:11435/v1"), "qwen2.5:7b")
    )
    assert rec["local"] is True
    assert rec["digest"] == "sha256:beef"
    assert rec["quantization"] == "Q4_K_M"
    assert rec["parameter_size"] == "7.6B"
    assert rec["family"] == "qwen2"
    # Both windows: what the run asked for, and what the artifact actually supports.
    assert rec["num_ctx"] == 16384
    assert rec["model_context_length"] == 32768
    assert rec["tool_calling"] is True
    assert rec["capabilities"] == ["completion", "tools"]
    # The /v1 suffix belongs to the OpenAI shim, not to the native API.
    assert all("/v1/" not in url for _, url, _ in _FakeClient.calls)


def test_a_model_without_tool_capability_is_recorded_as_such(monkeypatch):
    class _NoTools(_FakeClient):
        async def post(self, url, json=None):
            return _Resp({"details": {}, "model_info": {}, "capabilities": ["completion"]})

    monkeypatch.setattr(mm.httpx, "AsyncClient", _NoTools)
    rec = asyncio.run(mm.collect_model_metadata(_connector("http://ollama:11434"), "llama3.2:3b"))
    assert rec["tool_calling"] is False


def test_hosted_provider_claims_only_what_it_knows(monkeypatch):
    monkeypatch.setattr(mm.httpx, "AsyncClient", _FakeClient)
    rec = asyncio.run(mm.collect_model_metadata(
        _connector("https://openrouter.ai/api/v1", provider="openrouter", num_ctx=0),
        "openai/gpt-4.1-nano",
    ))
    assert rec["local"] is False
    assert rec["num_ctx"] is None
    assert "digest" not in rec and "quantization" not in rec
    assert _FakeClient.calls == []  # no probe against a hosted endpoint


def test_probe_failure_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(mm.httpx, "AsyncClient", _BoomClient)
    rec = asyncio.run(mm.collect_model_metadata(_connector("http://127.0.0.1:11434"), "qwen2.5:7b"))
    assert "connection refused" in rec["error"]
    assert rec["local"] is True
    assert "digest" not in rec


def test_a_malformed_connector_degrades_instead_of_failing_the_cell():
    rec = asyncio.run(mm.collect_model_metadata(object(), "qwen2.5:7b"))
    assert rec["model"] == "qwen2.5:7b"
    assert rec["local"] is False


def test_the_probe_runs_once_per_model_and_endpoint(monkeypatch):
    monkeypatch.setattr(mm.httpx, "AsyncClient", _FakeClient)
    conn = _connector("http://localhost:11435/v1")
    asyncio.run(mm.collect_model_metadata(conn, "qwen2.5:7b"))
    first = len(_FakeClient.calls)
    asyncio.run(mm.collect_model_metadata(conn, "qwen2.5:7b"))
    assert len(_FakeClient.calls) == first  # served from cache, no second probe


@pytest.mark.asyncio
async def test_the_cell_result_carries_the_metadata_block(monkeypatch):
    """The record has to reach the serialized cell result, where an after-the-fact audit of a
    confounded A/B pair can read it (``idea_test_runner`` dumps this dict verbatim)."""
    from unittest.mock import AsyncMock, MagicMock

    from agent.app.testing import runner as harness_runner

    monkeypatch.setattr(harness_runner, "run_test_execution", AsyncMock(return_value={
        "output": {}, "graph": {}, "observability": {},
    }))
    monkeypatch.setattr(
        harness_runner, "collect_model_metadata",
        AsyncMock(return_value={"model": "qwen2.5:7b", "digest": "sha256:beef"}),
    )
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.validation_runner.run = AsyncMock(return_value={"score": 0.0})

    result = await harness_runner.run_complete_test(
        test_module=tm, model_name="qwen2.5:7b",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        idea_settings={}, run_stamp="r1", summarize_observability_func=lambda *a, **k: {},
    )
    assert result["model_metadata"]["digest"] == "sha256:beef"


def test_local_detection_matches_the_backend_factory_minus_num_ctx():
    """A local run that forgot LLM_NUM_CTX is exactly the confounded cell to expose, so
    locality does not depend on num_ctx the way the backend factory's choice does."""
    assert mm.is_local_ollama(
        SimpleNamespace(llm_api_url="http://localhost:11435/v1", llm_provider="openai_compatible",
                        llm_num_ctx=0)
    ) is True
    assert mm.is_local_ollama(
        SimpleNamespace(llm_api_url="https://openrouter.ai/api/v1", llm_provider="openrouter",
                        llm_num_ctx=0)
    ) is False
