"""
Unit tests for ConnectorLLM with a mocked LLMBackend. No network.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.app.connector_llm import ConnectorLLM
from agent.app.llm_backends import LLMBackend


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODEL_API_URL",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "MODEL_NAME",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-5-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")


def _make_connector_with_mock_backend():
    from shared.connector_config import ConnectorConfig

    cfg = ConnectorConfig()
    mock_backend = AsyncMock(spec=LLMBackend)
    mock_backend.normalize_payload.side_effect = lambda p, *_, **__: p
    mock_backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        connector = ConnectorLLM(cfg)
    return connector, mock_backend


def test_build_payload_basic():
    connector, _ = _make_connector_with_mock_backend()
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "hi"}],
        json_mode=False,
        temperature=0.4,
    )
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_build_payload_json_schema():
    connector, _ = _make_connector_with_mock_backend()
    schema = {"name": "x", "schema": {"type": "object"}}
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "go"}],
        json_mode=True,
        json_schema=schema,
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"] == schema


def test_build_payload_json_object_no_schema():
    connector, _ = _make_connector_with_mock_backend()
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "go"}],
        json_mode=True,
    )
    assert payload["response_format"]["type"] == "json_object"


def test_reasoning_effort_attached_for_gpt5_slug():
    connector, _ = _make_connector_with_mock_backend()
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "go"}],
        json_mode=False,
        model_name="openai/gpt-5-mini",
        reasoning_effort="high",
    )
    assert payload.get("reasoning_effort") == "high"


def test_reasoning_effort_not_attached_for_non_gpt5():
    connector, _ = _make_connector_with_mock_backend()
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "go"}],
        json_mode=False,
        model_name="anthropic/claude-opus-4.7",
        reasoning_effort="high",
    )
    assert "reasoning_effort" not in payload


def test_record_usage_accumulates():
    connector, _ = _make_connector_with_mock_backend()
    usage_a = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    usage_b = SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60)
    connector._record_usage(usage_a)
    connector._record_usage(usage_b)
    assert connector.total_usage["prompt_tokens"] == 150
    assert connector.total_usage["completion_tokens"] == 30
    assert connector.total_usage["total_tokens"] == 180
    assert connector.last_usage["prompt_tokens"] == 50


def test_record_usage_handles_anthropic_field_names():
    connector, _ = _make_connector_with_mock_backend()
    # Anthropic SDK exposes input_tokens / output_tokens, not prompt_/completion_tokens.
    usage = SimpleNamespace(input_tokens=30, output_tokens=5, total_tokens=None)
    connector._record_usage(usage)
    assert connector.last_usage["prompt_tokens"] == 30
    assert connector.last_usage["completion_tokens"] == 5
    assert connector.last_usage["total_tokens"] == 35


def test_record_usage_ignores_none():
    connector, _ = _make_connector_with_mock_backend()
    connector._record_usage(None)
    assert connector.last_usage is None
    assert connector.total_usage["total_tokens"] == 0


def test_set_model_updates_default():
    connector, _ = _make_connector_with_mock_backend()
    connector.set_model("anthropic/claude-opus-4.7")
    assert connector.model_name == "anthropic/claude-opus-4.7"
    connector.set_model("   ")  # whitespace ignored
    assert connector.model_name == "anthropic/claude-opus-4.7"


def test_set_model_profile_round_trip():
    connector, _ = _make_connector_with_mock_backend()
    connector.set_model_profile("openai/gpt-5-mini", {"temperature": 0.2})
    assert connector.model_profiles["openai/gpt-5-mini"] == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_query_llm_records_usage_and_returns_content():
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.return_value = (
        "hello world",
        SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    out = await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})
    assert out == "hello world"
    assert connector.last_usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_query_llm_returns_none_on_unrecoverable_error():
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.side_effect = RuntimeError("boom")
    out = await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})
    assert out is None


# --- F17: infra-failure quarantine (402/408/429/5xx retried, not silently scored as a real 0) ---

class _FakeStatusError(Exception):
    """Stand-in for openai.APIStatusError — should_retry only inspects .status_code."""

    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_is_infra_llm_failure_classifies_payment_and_rate_limit_status_codes():
    from agent.app.connector_llm import is_infra_llm_failure

    assert is_infra_llm_failure(_FakeStatusError(402)) is True   # OpenRouter payment-required
    assert is_infra_llm_failure(_FakeStatusError(408)) is True   # request timeout
    assert is_infra_llm_failure(_FakeStatusError(429)) is True   # rate limit
    assert is_infra_llm_failure(_FakeStatusError(503)) is True   # provider unavailable


def test_is_infra_llm_failure_false_for_content_and_client_errors():
    from agent.app.connector_llm import is_infra_llm_failure
    from agent.app.llm_backends import LLMContentError

    # A 400 (bad request / malformed payload) is a genuine client-side defect, not infra.
    assert is_infra_llm_failure(_FakeStatusError(400)) is False
    assert is_infra_llm_failure(LLMContentError("truncated")) is False
    assert is_infra_llm_failure(RuntimeError("some other bug")) is False


def test_is_infra_llm_failure_classifies_timeouts_and_transport_errors():
    import asyncio as _asyncio
    from agent.app.connector_llm import is_infra_llm_failure

    assert is_infra_llm_failure(_asyncio.TimeoutError()) is True

    class FakeAPIConnectionError(Exception):
        pass

    assert is_infra_llm_failure(FakeAPIConnectionError("connection reset")) is True


@pytest.mark.asyncio
async def test_query_llm_retries_402_then_succeeds():
    """A 402 (OpenRouter daily-cap style) must be retried, not treated as an immediate fatal —
    the pre-fix behavior returned None on the very first 402, which a downstream grader scores
    as a genuine 0 indistinguishable from a bad model answer."""
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.side_effect = [
        _FakeStatusError(402),
        ("recovered after retry", SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
    ]
    out = await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})
    assert out == "recovered after retry"
    assert mock_backend.complete.call_count == 2


@pytest.mark.asyncio
async def test_query_llm_tags_infra_failed_in_telemetry_on_terminal_402():
    """When every retry attempt still 402s, query_llm returns None (unchanged) but must tag the
    failure as infra so the result-JSON layer can quarantine the cell instead of scoring a real 0."""
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.side_effect = _FakeStatusError(402)
    telemetry = MagicMock()
    telemetry.record_llm_usage = MagicMock()
    connector.set_telemetry(telemetry)

    out = await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})

    assert out is None
    # _record_timing ultimately calls telemetry.record_timing; assert the infra flag reached it.
    assert telemetry.record_timing.called
    _, kwargs = telemetry.record_timing.call_args
    assert kwargs["payload"]["infra_failed"] is True
    # record_llm_usage (the per-call usage/error trace) is also tagged.
    usage_call_kwargs = telemetry.record_llm_usage.call_args.args[0]
    assert usage_call_kwargs["infra_failed"] is True


@pytest.mark.asyncio
async def test_query_llm_does_not_tag_infra_failed_for_genuine_content_error():
    """A non-infra terminal failure (e.g. a plain bug) must NOT be tagged infra_failed — only
    real payment/rate-limit/timeout/transport signals should be quarantined."""
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.side_effect = RuntimeError("some other bug")
    telemetry = MagicMock()
    telemetry.record_llm_usage = MagicMock()
    connector.set_telemetry(telemetry)

    out = await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})

    assert out is None
    _, kwargs = telemetry.record_timing.call_args
    assert kwargs["payload"]["infra_failed"] is False
