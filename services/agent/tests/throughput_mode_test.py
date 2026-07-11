"""Offline tests for benchmark THROUGHPUT MODE (opt-in / additive; concurrency=1 unchanged).

Covers three additive speed mechanisms:
  1. Parallel preflight fan-out across pooled connector slots (idea_test_runner).
  2. Process-global in-flight LLM limiter (connector_llm) — the concurrency>1 429 safeguard.
  3. Faster-fail: a deterministic ``finish_reason=length`` truncation is NON-retryable
     (LLMContentError), so it fails fast instead of burning the retry budget.
"""
from __future__ import annotations

import asyncio

import pytest


# ---------------------------------------------------------------------------
# 1. Parallel preflight fan-out
# ---------------------------------------------------------------------------
class _FakeLLM:
    def __init__(self, tag):
        self.tag = tag


def _fake_pool(n):
    return [{"llm": _FakeLLM(i)} for i in range(n)]


@pytest.mark.asyncio
async def test_preflight_fans_out_across_pool(monkeypatch):
    import agent.app.idea_test_runner as runner

    inflight = 0
    max_inflight = 0
    checked = []

    async def fake_check(connector_llm, model_name):
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        checked.append((connector_llm.tag, model_name))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        inflight -= 1
        return True

    monkeypatch.setattr(runner, "preflight_check_llm", fake_check)

    pool = _fake_pool(3)
    models = ["m1", "m2", "m3"]
    results = await runner._run_preflight_parallel(pool, models)

    assert results == {"m1": True, "m2": True, "m3": True}
    assert max_inflight == 3, f"a 3-slot pool must check 3 models concurrently, got {max_inflight}"
    # Each concurrent check ran on a distinct pooled connector (no set_model race).
    assert {tag for tag, _ in checked} == {0, 1, 2}


@pytest.mark.asyncio
async def test_preflight_pool_size_one_is_serial(monkeypatch):
    import agent.app.idea_test_runner as runner

    inflight = 0
    max_inflight = 0

    async def fake_check(connector_llm, model_name):
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        await asyncio.sleep(0)
        inflight -= 1
        return model_name != "bad"

    monkeypatch.setattr(runner, "preflight_check_llm", fake_check)

    results = await runner._run_preflight_parallel(_fake_pool(1), ["m1", "bad", "m3"])
    assert results == {"m1": True, "bad": False, "m3": True}
    assert max_inflight == 1, "pool size 1 must stay serial (byte-identical attribution mode)"


@pytest.mark.asyncio
async def test_preflight_more_models_than_slots_processed_in_waves(monkeypatch):
    import agent.app.idea_test_runner as runner

    inflight = 0
    max_inflight = 0

    async def fake_check(connector_llm, model_name):
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        inflight -= 1
        return True

    monkeypatch.setattr(runner, "preflight_check_llm", fake_check)

    results = await runner._run_preflight_parallel(_fake_pool(2), ["a", "b", "c", "d", "e"])
    assert set(results) == {"a", "b", "c", "d", "e"}
    assert all(results.values())
    # Never more than pool_size (2) in flight at once.
    assert max_inflight == 2


# ---------------------------------------------------------------------------
# 2. In-flight LLM limiter
# ---------------------------------------------------------------------------
def _make_connector(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from unittest.mock import patch
    from agent.app.llm_backends import LLMBackend
    from shared.connector_config import ConnectorConfig

    with patch("agent.app.connector_llm.create_llm_backend"):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(ConnectorConfig())


def test_llm_max_inflight_default_and_override(monkeypatch):
    from agent.app.connector_llm import llm_max_inflight

    monkeypatch.delenv("IDEA_TEST_LLM_MAX_INFLIGHT", raising=False)
    assert llm_max_inflight() == 32
    monkeypatch.setenv("IDEA_TEST_LLM_MAX_INFLIGHT", "3")
    assert llm_max_inflight() == 3
    monkeypatch.setenv("IDEA_TEST_LLM_MAX_INFLIGHT", "garbage")
    assert llm_max_inflight() == 32


@pytest.mark.asyncio
async def test_inflight_semaphore_bounds_concurrency(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_LLM_MAX_INFLIGHT", "2")
    import agent.app.connector_llm as cl
    # Force a fresh semaphore bound to this test's loop + limit.
    cl._INFLIGHT_STATE.update({"loop": None, "limit": None, "sem": None})

    inflight = 0
    max_inflight = 0

    async def worker():
        nonlocal inflight, max_inflight
        async with cl._inflight_semaphore():
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            await asyncio.sleep(0.01)
            inflight -= 1

    await asyncio.gather(*[worker() for _ in range(8)])
    assert max_inflight == 2, f"limiter must bound in-flight to 2, saw {max_inflight}"


# ---------------------------------------------------------------------------
# 3. Faster-fail: finish_reason=length is non-retryable
# ---------------------------------------------------------------------------
def test_llm_content_error_is_runtimeerror():
    from agent.app.llm_backends import LLMContentError

    assert issubclass(LLMContentError, RuntimeError)


def test_extract_content_raises_content_error_on_length(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    import logging
    from shared.connector_config import ConnectorConfig
    from agent.app.llm_backends import OpenAICompatibleBackend, LLMContentError

    b = OpenAICompatibleBackend(ConnectorConfig(), logging.getLogger("t"))

    class _Msg:
        content = None

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Resp:
        choices = [_Choice()]

    with pytest.raises(LLMContentError):
        b._extract_content(_Resp(), "gpt-5-mini")


@pytest.mark.asyncio
async def test_starved_call_is_not_retried(monkeypatch):
    # A finish_reason=length truncation must fail after exactly ONE wire attempt (no retry
    # burn), unlike a transient error which the loop retries.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    from unittest.mock import AsyncMock, patch
    from agent.app.llm_backends import LLMContentError
    from shared.connector_config import ConnectorConfig

    with patch("agent.app.connector_llm.create_llm_backend") as mk:
        backend = mk.return_value
        backend.simplify_payload.side_effect = lambda p: dict(p)
        backend.complete = AsyncMock(side_effect=LLMContentError("finish_reason=length"))
        from agent.app.connector_llm import ConnectorLLM

        conn = ConnectorLLM(ConnectorConfig())
        payload = {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]}
        result = await conn.query_llm(payload, model_name="gpt-5-mini")

    assert result is None, "a starved call surfaces as a failed (None) result"
    assert backend.complete.await_count == 1, (
        f"finish_reason=length must NOT be retried, saw {backend.complete.await_count} attempts"
    )
