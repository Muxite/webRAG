import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from shared.retry import Retry


@pytest.mark.asyncio
async def test_retry_succeeds_immediately():
    async def func():
        return "success"
    retry = Retry(func, max_attempts=3)
    
    result = await retry.run()
    
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_retries_on_exception():
    call_count = 0
    async def func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("fail")
        return "success"
    retry = Retry(func, max_attempts=3, base_delay=0.01)
    
    result = await retry.run()
    
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_max_attempts():
    call_count = 0
    async def func():
        nonlocal call_count
        call_count += 1
        raise Exception("always fails")
    retry = Retry(func, max_attempts=2, base_delay=0.01)
    
    result = await retry.run()
    
    assert result is None
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_raises_on_fail():
    async def func():
        raise ValueError("error")
    retry = Retry(func, max_attempts=2, base_delay=0.01, raise_on_fail=True)
    
    with pytest.raises(ValueError, match="error"):
        await retry.run()


@pytest.mark.asyncio
async def test_retry_exponential_backoff():
    delays = []
    call_count = 0
    
    def on_retry(attempt, delay, exc):
        delays.append(delay)
    
    async def func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("fail")
        return "success"
    
    retry = Retry(
        func,
        max_attempts=3,
        base_delay=0.01,
        multiplier=2.0,
        on_retry=on_retry
    )
    
    await retry.run()
    
    assert len(delays) == 2
    assert delays[0] < delays[1]


@pytest.mark.asyncio
async def test_retry_should_retry_callback():
    results = []
    call_count = 0
    
    def should_retry(result, exc, attempt):
        results.append((result, exc, attempt))
        return attempt < 3
    
    async def func():
        nonlocal call_count
        call_count += 1
        return None
    
    retry = Retry(func, max_attempts=5, base_delay=0.01, should_retry=should_retry)
    
    await retry.run()
    
    assert len(results) == 3
    assert call_count == 3
