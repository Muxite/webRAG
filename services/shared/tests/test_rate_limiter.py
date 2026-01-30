import pytest
import asyncio
import time
from shared.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_waits_on_first_call():
    limiter = RateLimiter(period=0.1)
    
    start = time.time()
    await limiter.acquire()
    elapsed = time.time() - start
    
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_rate_limiter_enforces_period():
    limiter = RateLimiter(period=0.2)
    
    await limiter.acquire()
    
    start = time.time()
    await limiter.acquire()
    elapsed = time.time() - start
    
    assert elapsed >= 0.15
    assert elapsed < 0.3


@pytest.mark.asyncio
async def test_rate_limiter_allows_after_period():
    limiter = RateLimiter(period=0.1)
    
    await limiter.acquire()
    await asyncio.sleep(0.15)
    
    start = time.time()
    await limiter.acquire()
    elapsed = time.time() - start
    
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_calls():
    limiter = RateLimiter(period=0.1)
    
    async def acquire():
        await limiter.acquire()
        return time.time()
    
    times = await asyncio.gather(*[acquire() for _ in range(3)])
    
    for i in range(1, len(times)):
        assert times[i] - times[i-1] >= 0.09
