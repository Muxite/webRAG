"""Tests for the per-action-type timeout helper.

`IdeaDagEngine._action_timeout_for(action_name)` returns the timeout to use
for a given action, preferring `{action}_timeout_seconds` and falling back
to `action_timeout_seconds`. We replicate the lookup inline rather than
importing the engine (which pulls in bs4/chromadb).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def action_timeout_for(settings: Dict[str, Any], action_name: Optional[str]) -> float:
    """Mirror of `IdeaDagEngine._action_timeout_for` for isolated testing."""
    fallback = float(settings.get("action_timeout_seconds", 120))
    if not action_name:
        return fallback
    per_type = settings.get(f"{action_name}_timeout_seconds")
    if per_type is None:
        return fallback
    try:
        return float(per_type)
    except (TypeError, ValueError):
        return fallback


def test_per_type_visit_timeout_preferred():
    settings = {"action_timeout_seconds": 120, "visit_timeout_seconds": 20}
    assert action_timeout_for(settings, "visit") == 20.0


def test_per_type_search_timeout_preferred():
    settings = {"action_timeout_seconds": 120, "search_timeout_seconds": 15}
    assert action_timeout_for(settings, "search") == 15.0


def test_falls_back_when_no_per_type_override():
    settings = {"action_timeout_seconds": 60}
    assert action_timeout_for(settings, "think") == 60.0


def test_falls_back_on_none_action_name():
    settings = {"action_timeout_seconds": 99}
    assert action_timeout_for(settings, None) == 99.0


def test_falls_back_on_unparseable_override():
    settings = {"action_timeout_seconds": 90, "visit_timeout_seconds": "not-a-number"}
    assert action_timeout_for(settings, "visit") == 90.0


def test_uses_default_when_action_timeout_missing():
    # No action_timeout_seconds at all → built-in default 120.
    assert action_timeout_for({}, "visit") == 120.0


def test_zero_per_type_override_honored():
    # Zero is a valid number; should NOT fall back to default 120.
    settings = {"action_timeout_seconds": 120, "visit_timeout_seconds": 0}
    assert action_timeout_for(settings, "visit") == 0.0


def test_int_override_returns_float():
    settings = {"visit_timeout_seconds": 7}  # int
    assert action_timeout_for(settings, "visit") == 7.0
    assert isinstance(action_timeout_for(settings, "visit"), float)


# ---- Semaphore-based parallel limit: smoke-test the asyncio pattern ----

import asyncio


async def _do_work(semaphore: asyncio.Semaphore, in_flight: list, max_seen: list, idx: int) -> int:
    async with semaphore:
        in_flight[0] += 1
        max_seen[0] = max(max_seen[0], in_flight[0])
        await asyncio.sleep(0.005)  # small yield
        in_flight[0] -= 1
        return idx


def test_semaphore_caps_concurrency():
    """When 10 tasks share a Semaphore(3), at most 3 are ever in flight."""

    async def run():
        sem = asyncio.Semaphore(3)
        in_flight = [0]
        max_seen = [0]
        results = await asyncio.gather(
            *[_do_work(sem, in_flight, max_seen, i) for i in range(10)]
        )
        return results, max_seen[0]

    results, max_concurrent = asyncio.run(run())
    assert sorted(results) == list(range(10))
    assert max_concurrent <= 3


def test_semaphore_limit_one_serializes():
    """Semaphore(1) is effectively a sequential gate."""

    async def run():
        sem = asyncio.Semaphore(1)
        in_flight = [0]
        max_seen = [0]
        await asyncio.gather(
            *[_do_work(sem, in_flight, max_seen, i) for i in range(5)]
        )
        return max_seen[0]

    assert asyncio.run(run()) == 1
