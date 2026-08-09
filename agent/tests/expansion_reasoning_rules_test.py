"""IDEA_TEST_REASONING_RULES injects a flat imperative rule checklist into the
expansion system prompt — offline, no LLM.

Fully independent of IDEA_TEST_REASONING_EXEMPLAR: either may be set alone. Unset/
none/invalid keeps the prompt byte-identical to today; a valid name (branch_eliminate)
prepends the reasoning_rules/*.md content ahead of the system template, and AFTER the
exemplar block when both are set.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.expansion import LlmExpansionPolicy
from agent.app.llm_backends import LLMBackend

_RULES_DIR = Path(__file__).resolve().parents[1] / "app" / "reasoning_rules"
_RULES_HEADER = "## Mandatory reasoning rules"
_EXEMPLAR_HEADER = "## Reference reasoning pattern"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_API_KEY", "OPENAI_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-nano")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.delenv("IDEA_TEST_REASONING_RULES", raising=False)
    monkeypatch.delenv("IDEA_TEST_REASONING_EXEMPLAR", raising=False)
    from agent.app.idea_policies import expansion as exp

    exp._RULES_CACHE.clear()
    exp._RULES_WARNED.clear()
    exp._EXEMPLAR_CACHE.clear()
    exp._EXEMPLAR_WARNED.clear()


def _make_io():
    mock_backend = AsyncMock(spec=LLMBackend)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        pass
    return AsyncMock()


def _build_system_prompt(policy) -> str:
    graph = IdeaDag(root_title="Find facts", root_details={"mandate": "Find facts"})
    messages = policy._build_messages(graph, graph.get_node(graph.root_id()))
    return "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )


def test_unset_env_leaves_prompt_free_of_rules_block():
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    assert _RULES_HEADER not in system


def test_unset_matches_none_byte_for_byte(monkeypatch):
    policy = LlmExpansionPolicy(io=_make_io())
    unset_system = _build_system_prompt(policy)
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "none")
    none_system = _build_system_prompt(policy)
    assert unset_system == none_system
    assert _RULES_HEADER not in none_system


def test_branch_eliminate_content_injected(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "branch_eliminate")
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    rules_text = (_RULES_DIR / "branch_eliminate.md").read_text(encoding="utf-8").strip()
    assert _RULES_HEADER in system
    assert rules_text in system


def test_invalid_value_treated_as_unset_and_warns(monkeypatch, caplog):
    policy = LlmExpansionPolicy(io=_make_io())
    baseline = _build_system_prompt(policy)
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "bogus")
    with caplog.at_level("WARNING"):
        bogus_system = _build_system_prompt(policy)
    assert bogus_system == baseline
    assert _RULES_HEADER not in bogus_system
    assert any("bogus" in r.getMessage() for r in caplog.records)


def test_case_insensitive_and_trimmed_name(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "  Branch_Eliminate  ")
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    assert _RULES_HEADER in system


def test_independent_of_exemplar_env(monkeypatch):
    # Rules set, exemplar unset: rules present, exemplar absent.
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "branch_eliminate")
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    assert _RULES_HEADER in system
    assert _EXEMPLAR_HEADER not in system


def test_both_set_exemplar_precedes_rules(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_REASONING_EXEMPLAR", "chain")
    monkeypatch.setenv("IDEA_TEST_REASONING_RULES", "branch_eliminate")
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    assert _EXEMPLAR_HEADER in system
    assert _RULES_HEADER in system
    # Ordering: exemplar block -> rules block -> system template.
    assert system.index(_EXEMPLAR_HEADER) < system.index(_RULES_HEADER)
