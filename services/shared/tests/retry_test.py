"""
Unit tests for shared.retry.Retry.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.retry import Retry


@pytest.mark.asyncio
async def test_retry_returns_first_successful_result():
    func = AsyncMock(return_value="ok")
    out = await Retry(func=func, max_attempts=3, base_delay=0.001).run()
    assert out == "ok"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retry_retries_on_exception_until_success():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "done"

    out = await Retry(
        func=flaky,
        max_attempts=5,
        base_delay=0.001,
        retry_exceptions=(RuntimeError,),
    ).run()
    assert out == "done"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_raises_on_fail_when_configured():
    async def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await Retry(
            func=boom,
            max_attempts=2,
            base_delay=0.001,
            retry_exceptions=(RuntimeError,),
            raise_on_fail=True,
        ).run()


@pytest.mark.asyncio
async def test_retry_returns_falsy_result_when_no_raise():
    async def returns_empty():
        return None

    out = await Retry(
        func=returns_empty,
        max_attempts=2,
        base_delay=0.001,
    ).run()
    # All attempts return None and there's no exception → raise_on_fail=False returns last_result.
    assert out is None


@pytest.mark.asyncio
async def test_retry_should_retry_predicate_overrides_defaults():
    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        return calls["n"]

    def stop_when_three(result, exc, attempt):
        return result is not None and result < 3

    out = await Retry(
        func=func,
        max_attempts=10,
        base_delay=0.001,
        should_retry=stop_when_three,
    ).run()
    assert out == 3
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_works_with_sync_callable():
    state = {"n": 0}

    def sync_flaky():
        state["n"] += 1
        if state["n"] < 2:
            raise ValueError("once")
        return "ok"

    out = await Retry(
        func=sync_flaky,
        max_attempts=3,
        base_delay=0.001,
        retry_exceptions=(ValueError,),
    ).run()
    assert out == "ok"


@pytest.mark.asyncio
async def test_retry_on_retry_callback_invoked():
    callback = MagicMock()
    calls = {"n": 0}

    async def fail_once():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("x")
        return "ok"

    await Retry(
        func=fail_once,
        max_attempts=3,
        base_delay=0.001,
        retry_exceptions=(RuntimeError,),
        on_retry=callback,
    ).run()
    assert callback.call_count >= 1
