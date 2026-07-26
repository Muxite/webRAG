"""Regression tests for the Brave-422 fairness fix (connector_search).

Brave 422s (non-retryable) on queries over 400 chars / 50 words or count>20. That hard-failed
the react reference's long queries (40/40) while the graph arm's concise ones survived (0/40) —
an unfair handicap. Every outgoing query is now clipped to Brave's limits, symmetrically.
"""
import types

import pytest

from agent.app.connector_search import (
    ConnectorSearch,
    _sanitize_brave_query,
    BRAVE_MAX_QUERY_CHARS,
    BRAVE_MAX_QUERY_WORDS,
    BRAVE_MAX_COUNT,
)
from shared.connector_config import ConnectorConfig


def test_short_query_unchanged():
    q = "what is the capital of France"
    assert _sanitize_brave_query(q) == q


def test_long_query_clipped_to_word_and_char_caps():
    out = _sanitize_brave_query("word " * 100)  # 100 words, ~500 chars
    assert len(out.split()) <= BRAVE_MAX_QUERY_WORDS
    assert len(out) <= BRAVE_MAX_QUERY_CHARS


def test_whitespace_and_control_chars_collapsed():
    assert _sanitize_brave_query("a\n\nb\tc") == "a b c"
    assert _sanitize_brave_query("") == ""


def test_single_long_token_clips_to_char_cap_not_empty():
    out = _sanitize_brave_query("x" * 500)   # no spaces to break on
    assert len(out) == BRAVE_MAX_QUERY_CHARS


@pytest.mark.asyncio
async def test_query_search_clamps_count_and_sanitizes(monkeypatch):
    cs = ConnectorSearch(ConnectorConfig())
    cs.search_api_key = "k"
    cs.search_api_ready = True  # skip the live health probe
    captured = {}

    async def fake_request(method, url, retries=3, headers=None, params=None):
        captured.update(params or {})
        return types.SimpleNamespace(error=False, status=200, data={"web": {"results": []}})

    monkeypatch.setattr(cs, "request", fake_request)
    await cs.query_search("q " * 100, count=99)
    assert captured["count"] == BRAVE_MAX_COUNT           # clamped from 99
    assert len(captured["q"]) <= BRAVE_MAX_QUERY_CHARS     # never sends a 422-shaped query
    assert len(captured["q"].split()) <= BRAVE_MAX_QUERY_WORDS
