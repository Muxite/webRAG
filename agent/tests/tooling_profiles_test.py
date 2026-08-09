"""
Unit tests for the tooling-ablation axis (testing/tooling.py + runner wiring).

These are free (no LLM/network): they assert the profile registry, the
variant<->profile mapping, the runner's variant parsing of profile names, and that
the search-only ``minimal`` executor calls search but never visits a page.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing.tooling import (
    TOOLING_PROFILES,
    profile_for_variant,
    resolve_profile,
)
from agent.app.testing.runner import BASELINE_VARIANTS
from agent.app.idea_test_runner import _parse_execution_variants
from agent.app.testing import execution as execution_mod


def test_three_rungs_present_with_expected_variants():
    assert set(TOOLING_PROFILES) == {"minimal", "partial", "full"}
    assert TOOLING_PROFILES["minimal"].variant == "minimal"
    assert TOOLING_PROFILES["partial"].variant == "naive_rag"
    assert TOOLING_PROFILES["full"].variant == "graph"


def test_minimal_has_no_crawl_action():
    # The whole point of the minimal rung: search but never visit.
    assert "visit" not in TOOLING_PROFILES["minimal"].allowed_actions
    assert "search" in TOOLING_PROFILES["minimal"].allowed_actions
    assert "visit" in TOOLING_PROFILES["partial"].allowed_actions
    assert "verify" in TOOLING_PROFILES["full"].allowed_actions


def test_profile_for_variant_roundtrips_and_falls_back():
    assert profile_for_variant("graph") == "full"
    assert profile_for_variant("naive_rag") == "partial"
    assert profile_for_variant("minimal") == "minimal"
    # Non-laddered variants pass through unchanged so the field is always set.
    assert profile_for_variant("parametric") == "parametric"
    assert profile_for_variant("sequential") == "sequential"


def test_resolve_profile_is_case_insensitive():
    assert resolve_profile("FULL").variant == "graph"
    assert resolve_profile("  minimal ").variant == "minimal"
    assert resolve_profile("nope") is None


def test_runner_parses_profile_names_to_variants():
    assert _parse_execution_variants("minimal,partial,full") == ["minimal", "naive_rag", "graph"]
    # Dedupes when a profile and its underlying variant are both named.
    assert _parse_execution_variants("partial,naive_rag") == ["naive_rag"]


def test_minimal_is_a_baseline_variant():
    assert "minimal" in BASELINE_VARIANTS


def test_search_only_executor_searches_but_never_visits():
    agent_io = MagicMock()
    agent_io.search = AsyncMock(return_value=[
        {"url": "https://example.com/a", "title": "A", "description": "snippet a"},
        {"url": "https://example.com/b", "title": "B", "description": "snippet b"},
    ])
    agent_io.visit = AsyncMock(return_value="SHOULD NOT BE CALLED")
    agent_io.build_llm_payload = MagicMock(return_value={"messages": []})
    agent_io.query_llm = AsyncMock(return_value="answer from snippets")

    out = asyncio.run(execution_mod._run_search_only(agent_io, "task?", "cheap-model", 1024))

    assert out == "answer from snippets"
    agent_io.search.assert_awaited_once()
    agent_io.visit.assert_not_called()
    # The synthesis prompt must carry the snippet text so the model can use it.
    user_msg = agent_io.build_llm_payload.call_args.kwargs["messages"][-1]["content"]
    assert "snippet a" in user_msg and "https://example.com/a" in user_msg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
