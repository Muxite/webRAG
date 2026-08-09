"""IDEA_TEST_REASONING_EXEMPLAR injects a general reasoning few-shot into the
expansion system prompt — offline, no LLM.

Unset/none/invalid keeps the prompt byte-identical to today's behavior; a valid
name (chain/mixed/parallel) prepends the corresponding reasoning_exemplars/*.md
content as a labeled reference block ahead of the addendum/instructions.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.expansion import LlmExpansionPolicy
from agent.app.llm_backends import LLMBackend

_EXEMPLAR_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "reasoning_exemplars"
)
_BLOCK_HEADER = "## Reference reasoning pattern"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_API_KEY", "OPENAI_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-nano")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.delenv("IDEA_TEST_REASONING_EXEMPLAR", raising=False)
    # Reset the module-level caches so warning/read tests are deterministic.
    from agent.app.idea_policies import expansion as exp

    exp._EXEMPLAR_CACHE.clear()
    exp._EXEMPLAR_WARNED.clear()


def _make_io():
    mock_backend = AsyncMock(spec=LLMBackend)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        pass
    io = AsyncMock()
    return io


def _build_system_prompt(policy) -> str:
    graph = IdeaDag(root_title="Find facts", root_details={"mandate": "Find facts"})
    messages = policy._build_messages(graph, graph.get_node(graph.root_id()))
    return "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )


def test_unset_env_leaves_prompt_free_of_exemplar_block():
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    assert _BLOCK_HEADER not in system


def test_unset_matches_none_byte_for_byte(monkeypatch):
    policy = LlmExpansionPolicy(io=_make_io())
    unset_system = _build_system_prompt(policy)
    monkeypatch.setenv("IDEA_TEST_REASONING_EXEMPLAR", "none")
    none_system = _build_system_prompt(policy)
    assert unset_system == none_system
    assert _BLOCK_HEADER not in none_system


def test_chain_exemplar_content_injected(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_REASONING_EXEMPLAR", "chain")
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    chain_text = (_EXEMPLAR_DIR / "chain.md").read_text(encoding="utf-8").strip()
    assert _BLOCK_HEADER in system
    assert chain_text in system
    assert "Now apply that reasoning pattern" in system
    # The exemplar block precedes the instructions/addendum in the system prompt.
    assert system.index(_BLOCK_HEADER) < system.index("Now apply that reasoning pattern")


@pytest.mark.parametrize("name", ["chain", "mixed", "parallel"])
def test_all_valid_names_inject_their_own_file(monkeypatch, name):
    monkeypatch.setenv("IDEA_TEST_REASONING_EXEMPLAR", name)
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    text = (_EXEMPLAR_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    assert text in system


def test_invalid_value_treated_as_unset_and_warns(monkeypatch, caplog):
    policy = LlmExpansionPolicy(io=_make_io())
    baseline = _build_system_prompt(policy)
    monkeypatch.setenv("IDEA_TEST_REASONING_EXEMPLAR", "bogus")
    with caplog.at_level("WARNING"):
        bogus_system = _build_system_prompt(policy)
    assert bogus_system == baseline
    assert _BLOCK_HEADER not in bogus_system
    assert any("bogus" in r.getMessage() for r in caplog.records)


def test_case_insensitive_name(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_REASONING_EXEMPLAR", "  Chain  ")
    policy = LlmExpansionPolicy(io=_make_io())
    system = _build_system_prompt(policy)
    assert _BLOCK_HEADER in system
