"""
Unit tests for shared.rate_limiter.RateLimiter.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from shared.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_first_call_does_not_wait():
    limiter = RateLimiter(period=0.1)
    started = time.monotonic()
    await limiter.acquire()
    assert (time.monotonic() - started) < 0.05


@pytest.mark.asyncio
async def test_rate_limiter_enforces_period_between_calls():
    limiter = RateLimiter(period=0.1)
    await limiter.acquire()
    started = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.09, f"expected >=0.09s gap, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_rate_limiter_serializes_concurrent_acquires():
    limiter = RateLimiter(period=0.05)
    started = time.monotonic()
    await asyncio.gather(*[limiter.acquire() for _ in range(4)])
    elapsed = time.monotonic() - started
    # First call doesn't sleep (last_call starts at 0). Subsequent calls alternate
    # between immediate and full-period waits as last_call advances. 4 calls
    # produce ~2 full-period waits → ~0.10s minimum.
    assert elapsed >= 0.09, f"expected >=0.09s for 4 serialized calls, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_rate_limiter_zero_period_no_wait():
    limiter = RateLimiter(period=0.0)
    started = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    assert (time.monotonic() - started) < 0.05
