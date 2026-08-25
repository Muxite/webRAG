"""Offline tests for the sequential_react per-turn context cap — no LLM, no network.

Phase 0's ``sequential_react_context_matched`` ablation
(docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md section 3) exists so a DAG-vs-``sequential_react``
comparison is not confounded by the linear arm seeing far more evidence at decision time. The
load-bearing claims:

* flag off (the shipped default) assembles the scratchpad BYTE-IDENTICALLY to the pre-feature
  code, including the 1500-character per-step observation cap;
* on, the caps are the DAG's OWN budget, read from ``expansion_ancestor_content_chars`` and
  ``expansion_max_context_nodes`` rather than restated, so the "matched" arm cannot drift;
* trimming drops WHOLE oldest steps — never a step cut mid-observation, which would show the
  model a truncated URL or value that reads as a different one.
"""
from __future__ import annotations

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.config import IdeaConfig
from agent.app.testing.execution_sequential import (
    _SCRATCHPAD_WINDOW,
    _UNCAPPED_OBSERVATION_CHARS,
    SequentialContextCap,
    _build_history,
)

ON = {"run_policy_sequential_context_cap_enabled": True}


def _steps(n: int, chars: int = 100) -> list:
    return [f"STEP {i}: " + ("x" * chars) for i in range(n)]


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def test_the_flag_ships_absent_and_therefore_off():
    settings = load_idea_dag_settings()
    assert "run_policy_sequential_context_cap_enabled" not in settings
    assert IdeaConfig.from_settings(settings).run_policy.sequential_context_cap_enabled is False


def test_no_settings_means_uncapped():
    for settings in (None, {}, {"run_policy_sequential_context_cap_enabled": False}):
        cap = SequentialContextCap.from_settings(settings)
        assert cap.enabled is False
        assert cap.observation_chars() == _UNCAPPED_OBSERVATION_CHARS


def test_the_cap_is_read_from_the_dags_own_budget_keys():
    cap = SequentialContextCap.from_settings(
        dict(ON, expansion_ancestor_content_chars=1000, expansion_max_context_nodes=5)
    )
    assert cap.enabled is True
    assert cap.per_step_chars == 1000        # the per-source budget
    assert cap.total_chars == 5000           # x max_context_nodes -> the per-turn budget
    assert cap.observation_chars() == 1000

    # Raising the DAG's own budget raises the matched arm's, with no second literal to update.
    wider = SequentialContextCap.from_settings(
        dict(ON, expansion_ancestor_content_chars=3000, expansion_max_context_nodes=4)
    )
    assert (wider.per_step_chars, wider.total_chars) == (3000, 12000)


def test_the_shipped_settings_resolve_to_the_documented_budget():
    settings = dict(load_idea_dag_settings())
    settings.update(ON)
    cap = SequentialContextCap.from_settings(settings)
    expansion = IdeaConfig.from_settings(settings).expansion
    assert cap.per_step_chars == expansion.ancestor_content_chars
    assert cap.total_chars == expansion.ancestor_content_chars * expansion.max_context_nodes


# --------------------------------------------------------------------------------------
# history assembly
# --------------------------------------------------------------------------------------


def test_uncapped_history_is_the_historical_window_join():
    scratchpad = _steps(_SCRATCHPAD_WINDOW + 5)
    assert _build_history(scratchpad, SequentialContextCap()) == "\n\n".join(
        scratchpad[-_SCRATCHPAD_WINDOW:]
    )


def test_an_empty_scratchpad_reads_the_same_either_way():
    assert _build_history([], SequentialContextCap()) == "(no actions yet)"
    assert _build_history([], SequentialContextCap(enabled=True)) == "(no actions yet)"


def test_capped_history_drops_whole_oldest_steps():
    scratchpad = _steps(6, chars=100)
    cap = SequentialContextCap(enabled=True, per_step_chars=100, total_chars=250)
    history = _build_history(scratchpad, cap)
    parts = history.split("\n\n")
    assert parts == scratchpad[-len(parts):]      # a suffix of the scratchpad, in order
    assert len(parts) < len(scratchpad)           # something really was dropped
    for part in parts:                            # and every survivor is a WHOLE step
        assert part in scratchpad


def test_the_most_recent_step_always_survives_even_over_budget():
    """A budget smaller than one step must not blank the model's view of what it just did."""
    scratchpad = _steps(4, chars=500)
    cap = SequentialContextCap(enabled=True, per_step_chars=10, total_chars=10)
    assert _build_history(scratchpad, cap) == scratchpad[-1]


def test_capping_strictly_reduces_context():
    scratchpad = _steps(_SCRATCHPAD_WINDOW, chars=1000)
    uncapped = _build_history(scratchpad, SequentialContextCap())
    capped = _build_history(
        scratchpad, SequentialContextCap(enabled=True, per_step_chars=1000, total_chars=5000)
    )
    assert len(capped) < len(uncapped)
    assert uncapped.endswith(capped)
