"""Offline tests for the sibling-context ledger delta in the expansion prompt — no LLM.

Expansion context is root-ward only (``IdeaDag.path_to_root``), so a node being expanded cannot
see what a SIBLING branch already resolved. ``run_policy_sibling_context_delta`` appends one
bounded line rendering the run's task-ledger snapshot to close that gap.

The load-bearing claims:

* flag off (the shipped default) is BYTE-IDENTICAL even when a fully populated ledger is sitting
  on the root node — a half-configured profile must not leak it;
* the flag DEPENDS on ``run_policy_ledger_mode == "observe"``; on its own it is inert, and a
  missing/garbage snapshot degrades to "no block" rather than an error;
* the entity lists are capped at whole names with the ``"... [truncated]"`` suffix the
  ancestor-content compaction already uses.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.expansion import (
    _LEDGER_DELTA_MAX_ENTITIES,
    LlmExpansionPolicy,
)
from agent.app.task_ledger import TaskLedger

HEADER = "[Ledger]"
ON = {"run_policy_sibling_context_delta": True, "run_policy_ledger_mode": "observe"}

LEDGER = {
    "entities": ["candidate_a", "candidate_b", "candidate_c", "candidate_d"],
    "requirements_total": 4,
    "requirements_supported": 2,
    "unresolved_entities": ["candidate_b", "candidate_d"],
}


class FakeIO:
    telemetry = None


def _system(ledger=None, **settings) -> str:
    policy = LlmExpansionPolicy(io=FakeIO(), model_name="m", settings=settings or None)
    root_details = {"mandate": "Research"}
    if ledger is not None:
        root_details[DetailKey.TASK_LEDGER.value] = ledger
    graph = IdeaDag(root_title="Research", root_details=root_details)
    messages = policy._build_messages(graph, graph.get_node(graph.root_id()))
    return next(m["content"] for m in messages if m["role"] == "system")


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def test_the_flag_ships_absent_and_therefore_off():
    settings = load_idea_dag_settings()
    assert "run_policy_sibling_context_delta" not in settings
    assert IdeaConfig.from_settings(settings).run_policy.sibling_context_delta is False


# --------------------------------------------------------------------------------------
# flag off: byte-identical
# --------------------------------------------------------------------------------------


def test_a_populated_ledger_does_not_leak_into_the_prompt_when_the_flag_is_off():
    """Default path, with the data present: the prompt must be the pre-feature bytes."""
    assert _system(ledger=LEDGER) == _system()
    assert HEADER not in _system(ledger=LEDGER)


def test_the_flag_alone_is_inert_without_observe_mode():
    """The block renders the ledger, and ``ledger_mode: off`` means no ledger exists."""
    on_but_unledgered = _system(ledger=LEDGER, run_policy_sibling_context_delta=True)
    assert on_but_unledgered == _system()
    assert HEADER not in on_but_unledgered


def test_observe_mode_alone_is_inert_without_the_flag():
    assert _system(ledger=LEDGER, run_policy_ledger_mode="observe") == _system()


# --------------------------------------------------------------------------------------
# flag on
# --------------------------------------------------------------------------------------


def test_the_block_reports_supported_and_open_requirements():
    system = _system(ledger=LEDGER, **ON)
    assert "[Ledger] 2/4 requirements resolved by other branches: candidate_a, candidate_c." in system
    assert "Still open: candidate_b, candidate_d." in system


def test_a_fully_resolved_roster_omits_the_still_open_sentence():
    ledger = dict(LEDGER, unresolved_entities=[], requirements_supported=4)
    system = _system(ledger=ledger, **ON)
    assert "4/4 requirements resolved" in system
    assert "Still open" not in system


def test_a_fully_unresolved_roster_omits_the_resolved_list():
    ledger = dict(LEDGER, unresolved_entities=list(LEDGER["entities"]), requirements_supported=0)
    system = _system(ledger=ledger, **ON)
    assert "[Ledger] 0/4 requirements resolved by other branches." in system
    assert "Still open: candidate_a, candidate_b, candidate_c, candidate_d." in system


def test_long_entity_lists_truncate_at_whole_names():
    names = [f"candidate_{i:02d}" for i in range(_LEDGER_DELTA_MAX_ENTITIES + 5)]
    ledger = {
        "entities": names,
        "requirements_total": len(names),
        "requirements_supported": 0,
        "unresolved_entities": list(names),
    }
    system = _system(ledger=ledger, **ON)
    assert "... [truncated]" in system
    for name in names[:_LEDGER_DELTA_MAX_ENTITIES]:
        assert name in system
    # A capped name is omitted whole, never cut mid-name into a different-looking entity.
    for name in names[_LEDGER_DELTA_MAX_ENTITIES:]:
        assert name not in system


def test_the_block_renders_a_real_compiled_ledger():
    """End-to-end shape check against ``TaskLedger.to_dict()`` rather than a hand-built dict."""
    mandate = "Compare candidate_a, candidate_b and candidate_c."
    graph = IdeaDag(root_title="Research", root_details={"mandate": mandate})
    snapshot = TaskLedger.compile(mandate, None, graph).to_dict()
    system = _system(ledger=snapshot, **ON)
    if snapshot["entities"]:
        assert HEADER in system
    else:
        assert HEADER not in system


# --------------------------------------------------------------------------------------
# graceful degradation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ledger",
    [
        None,
        {},
        {"entities": []},
        {"entities": "not-a-list"},
        "not-a-dict",
        [1, 2, 3],
    ],
)
def test_an_absent_or_malformed_snapshot_degrades_to_no_block(ledger):
    """No crash and no partial block: the prompt falls back to the flag-off bytes."""
    system = _system(ledger=ledger, **ON)
    assert HEADER not in system
    assert system == _system()


def test_a_missing_unresolved_list_reads_as_everything_resolved():
    """Fails OPEN, like the ledger itself: an unusable half is treated as nothing to flag."""
    system = _system(ledger={"entities": ["a"], "unresolved_entities": None}, **ON)
    assert "[Ledger] 1/1 requirements resolved by other branches: a." in system
    assert "Still open" not in system
