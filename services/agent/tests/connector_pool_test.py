"""Tests for the per-slot connector pool in the benchmark matrix runner.

Historically the matrix built exactly ONE shared ConnectorLLM/Search/Http/Chroma
set and passed it into every test-run, forcing IDEA_TEST_CONCURRENCY=1 (a shared
ConnectorLLM.set_model() mutates instance state and races when two different-model
runs overlap). The runner now builds a POOL sized to the resolved concurrency, so
each concurrent slot gets its own isolated set. These tests exercise the pure pool
factory without touching the network-bound matrix orchestration.
"""
from __future__ import annotations

from unittest.mock import patch

from agent.app import idea_test_runner as runner


class _FakeConnector:
    """Stand-in that records the config it was built with (identity-tracked)."""

    _counter = 0

    def __init__(self, config):
        type(self)._counter += 1
        self.config = config
        self.instance_id = type(self)._counter


def _patched_pool(pool_size: int):
    with patch.object(runner, "ConnectorLLM", _FakeConnector), \
         patch.object(runner, "ConnectorSearch", _FakeConnector), \
         patch.object(runner, "ConnectorHttp", _FakeConnector), \
         patch.object(runner, "ConnectorChroma", _FakeConnector), \
         patch.object(runner, "ConnectorBrowser", _FakeConnector):
        config = object()
        return runner._build_connector_pool(config, pool_size), config


def test_pool_size_matches_concurrency():
    pool, _ = _patched_pool(3)
    assert len(pool) == 3
    for cset in pool:
        # F18: every slot also gets its own browser-fallback connector now (cheap —
        # Playwright launches lazily on first use, gated at the call site instead).
        assert set(cset.keys()) == {"llm", "search", "http", "chroma", "browser"}


def test_concurrency_one_degenerates_to_single_set():
    """The critical regression guarantee: concurrency=1 -> exactly one connector set."""
    pool, _ = _patched_pool(1)
    assert len(pool) == 1


def test_pool_never_smaller_than_one():
    for bad in (0, -1):
        pool, _ = _patched_pool(bad)
        assert len(pool) == 1, f"pool_size={bad} must clamp up to 1"


def test_each_set_has_independent_connector_instances():
    """Every connector across every slot must be a distinct object (no sharing)."""
    pool, config = _patched_pool(3)
    seen_ids = set()
    for cset in pool:
        for key in ("llm", "search", "http", "chroma", "browser"):
            conn = cset[key]
            assert conn.config is config, "all connectors share the read-only config"
            assert conn.instance_id not in seen_ids, "connector instances must not be reused"
            seen_ids.add(conn.instance_id)
    # 3 slots * 5 connectors == 15 distinct instances.
    assert len(seen_ids) == 15


def test_slots_do_not_share_llm_instance():
    """The core race fix: no two slots may share a ConnectorLLM (set_model mutates it)."""
    pool, _ = _patched_pool(2)
    assert pool[0]["llm"] is not pool[1]["llm"]
