"""
Unit tests for ConnectorLLM full-capture text recording. No network.

Covers the fix where _record_io only ever recorded counts (prompt_chars/
completion_chars) into telemetry, never the real text, so full capture had
nothing to recover.
"""
from __future__ import annotations

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


def _record_io_calls(telemetry):
    """Return the (event, payload_dict) pairs from record_event('connector_io', ...) calls."""
    calls = []
    for call in telemetry.record_event.call_args_list:
        args, kwargs = call
        event = args[0] if args else kwargs.get("event")
        entry = args[1] if len(args) > 1 else kwargs.get("payload")
        if event == "connector_io":
            calls.append(entry)
    return calls


@pytest.mark.asyncio
async def test_full_capture_off_does_not_include_text():
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.return_value = (
        "hello world",
        SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    telemetry = MagicMock()
    connector.set_telemetry(telemetry)
    # full capture left at default (off)

    prompt = "what is the capital of France"
    await connector.query_llm({"messages": [{"role": "user", "content": prompt}], "model": "openai/gpt-5-mini"})

    io_calls = _record_io_calls(telemetry)
    in_call = next(c for c in io_calls if c["direction"] == "in")
    out_call = next(c for c in io_calls if c["direction"] == "out")

    # Summarized payload: strings are compressed to {"chars": N}, so counts remain
    # visible as the "model" plain value and prompt/completion string fields are summarized.
    assert "prompt_text" not in in_call["payload"]
    assert "completion_text" not in out_call["payload"]


@pytest.mark.asyncio
async def test_full_capture_on_includes_exact_text_and_keeps_counts():
    connector, mock_backend = _make_connector_with_mock_backend()
    completion = "Paris is the capital of France."
    mock_backend.complete.return_value = (
        completion,
        SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    telemetry = MagicMock()
    connector.set_telemetry(telemetry)
    connector.set_full_capture(True)

    prompt = "what is the capital of France"
    await connector.query_llm({"messages": [{"role": "user", "content": prompt}], "model": "openai/gpt-5-mini"})

    io_calls = _record_io_calls(telemetry)
    in_call = next(c for c in io_calls if c["direction"] == "in")
    out_call = next(c for c in io_calls if c["direction"] == "out")

    in_payload = in_call["payload"]
    out_payload = out_call["payload"]

    assert in_payload["prompt_text"] == prompt
    assert in_payload["prompt_chars"] == len(prompt)
    assert in_payload["prompt_words"] == len(prompt.split())
    assert in_payload["model"] == "openai/gpt-5-mini"
    assert "max_tokens" in in_payload

    assert out_payload["completion_text"] == completion
    assert out_payload["completion_chars"] == len(completion)
    assert out_payload["completion_words"] == len(completion.split())
    assert out_payload["model"] == "openai/gpt-5-mini"

    assert in_payload["messages"] == [{"role": "user", "content": prompt}]


@pytest.mark.asyncio
async def test_full_capture_on_preserves_roles_across_messages():
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.return_value = (
        "the answer",
        SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    telemetry = MagicMock()
    connector.set_telemetry(telemetry)
    connector.set_full_capture(True)

    system_msg = {"role": "system", "content": "You are terse."}
    user_msg = {"role": "user", "content": "What is 2+2?"}
    await connector.query_llm({"messages": [system_msg, user_msg], "model": "openai/gpt-5-mini"})

    io_calls = _record_io_calls(telemetry)
    in_call = next(c for c in io_calls if c["direction"] == "in")
    in_payload = in_call["payload"]

    assert in_payload["messages"] == [system_msg, user_msg]
    # prompt_text stays exactly as today (backward compatibility), a flattened blob.
    assert "You are terse." in in_payload["prompt_text"]
    assert "What is 2+2?" in in_payload["prompt_text"]


@pytest.mark.asyncio
async def test_full_capture_off_does_not_include_messages():
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.return_value = (
        "hello world",
        SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    telemetry = MagicMock()
    connector.set_telemetry(telemetry)
    # full capture left at default (off)

    await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})

    io_calls = _record_io_calls(telemetry)
    in_call = next(c for c in io_calls if c["direction"] == "in")
    assert "messages" not in in_call["payload"]


@pytest.mark.asyncio
async def test_full_capture_missing_attribute_does_not_raise():
    """query_llm's own getattr(self, "_full_capture", False) guard must not blow up even if
    a subclass/mock never set _full_capture. We stub _record_io itself (ConnectorBase's
    unrelated direct self._full_capture access is out of scope for this fix) and only assert
    that the gate in connector_llm.py behaves safely and defaults to no text capture."""
    connector, mock_backend = _make_connector_with_mock_backend()
    mock_backend.complete.return_value = (
        "ok",
        SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    recorded = []
    connector._record_io = lambda **kwargs: recorded.append(kwargs)
    # Simulate an object that never went through ConnectorBase.__init__ and so never got
    # _full_capture set at all.
    delattr(connector, "_full_capture")

    out = await connector.query_llm({"messages": [{"role": "user", "content": "hi"}], "model": "openai/gpt-5-mini"})

    assert out == "ok"
    in_call = next(c for c in recorded if c["direction"] == "in")
    out_call = next(c for c in recorded if c["direction"] == "out")
    assert "prompt_text" not in in_call["payload"]
    assert "completion_text" not in out_call["payload"]
