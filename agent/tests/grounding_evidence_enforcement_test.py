"""Regression tests for GroundingEvidenceEnforcementHook (C1d).

Root cause reproduced here: a mandate phrased as plain grounding language
("do not guess from memory") — with no literal "must visit" phrase and no
navigation target — could be planned as search-only children, run every
search to completion, and finalize with ZERO visited pages. `evaluate_grounding`
already flags this ("at least one visited source page" missing) but nothing
upstream injected the missing visit, so `_grounding_replan` gave up
(`injected <= 0`) and the run finalized ungrounded with 0 visits ever executed
(observed on benchmark task 122: 4 search leaves -> merge -> finalize, grounded
False, grounding_replans 0).

These tests lock in the fix: once a search has completed, the hook injects a
single visit node wired to that search's results — breaking the zero-visit
path — while staying a no-op whenever a visit is already planned/succeeded,
the mandate doesn't require grounding, or another hook (must-visit /
navigation) already owns the enforcement.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.candidate_coverage import CandidateCoverageResult
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.post_expansion_hooks import GroundingEvidenceEnforcementHook

GROUNDING_MANDATE = (
    "You are given NO URLs — navigate Wikipedia yourself and READ the pages "
    "(do not guess from memory). Determine which of these telescopes is largest."
)


def _graph_with_done_search(mandate=GROUNDING_MANDATE):
    graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
    root = graph.root_id()
    search = graph.add_child(
        parent_id=root,
        title="Search Wikipedia for FAST telescope",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.SEARCH.value,
                "success": True,
                "results": [{"url": "https://en.wikipedia.org/wiki/FAST", "title": "FAST"}],
            },
        },
    )
    search.status = IdeaNodeStatus.DONE
    return graph, root, search.node_id


def _visit_children(graph, node_id):
    return [
        c for cid in graph.get_node(node_id).children
        if (c := graph.get_node(cid)) and NodeDetailsExtractor.get_action(c.details) == IdeaActionType.VISIT.value
    ]


# --- direct hook unit tests -----------------------------------------------------

def test_injects_visit_once_search_completed():
    graph, root, search_id = _graph_with_done_search()
    GroundingEvidenceEnforcementHook().apply(graph, root, 0, GROUNDING_MANDATE, logging.getLogger("test"))

    visits = _visit_children(graph, root)
    assert len(visits) == 1
    injected = visits[0]
    assert injected.details.get(DetailKey.REQUIRES_DATA.value, {}).get("source_node_id") == search_id


def test_noop_when_no_search_completed_yet():
    mandate = GROUNDING_MANDATE
    graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
    root = graph.root_id()
    graph.add_child(
        parent_id=root,
        title="Search Wikipedia for FAST telescope",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.IS_LEAF.value: True},
    )
    GroundingEvidenceEnforcementHook().apply(graph, root, 0, mandate, logging.getLogger("test"))
    assert _visit_children(graph, root) == []


def test_noop_when_mandate_has_no_grounding_language():
    graph, root, _ = _graph_with_done_search(mandate="Summarize the search results.")
    GroundingEvidenceEnforcementHook().apply(graph, root, 0, "Summarize the search results.", logging.getLogger("test"))
    assert _visit_children(graph, root) == []


def test_noop_when_visit_already_planned():
    graph, root, search_id = _graph_with_done_search()
    graph.add_child(
        parent_id=root,
        title="Visit FAST",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.IS_LEAF.value: True},
    )
    GroundingEvidenceEnforcementHook().apply(graph, root, 0, GROUNDING_MANDATE, logging.getLogger("test"))
    assert len(_visit_children(graph, root)) == 1  # unchanged, not doubled


def test_noop_when_already_grounded_by_successful_visit():
    graph, root, search_id = _graph_with_done_search()
    visited = graph.add_child(
        parent_id=root,
        title="Visit FAST",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "url": "https://en.wikipedia.org/wiki/FAST",
            },
        },
    )
    visited.status = IdeaNodeStatus.DONE
    GroundingEvidenceEnforcementHook().apply(graph, root, 0, GROUNDING_MANDATE, logging.getLogger("test"))
    # Still just the one (already-successful) visit child; no duplicate injected.
    assert len(_visit_children(graph, root)) == 1


def test_noop_when_must_visit_phrase_present():
    """MandatePhraseEnforcementHook already owns this trigger; don't double-handle."""
    mandate = "Do not guess from memory. You must visit the page to confirm the figure."
    graph, root, _ = _graph_with_done_search(mandate=mandate)
    GroundingEvidenceEnforcementHook().apply(graph, root, 0, mandate, logging.getLogger("test"))
    assert _visit_children(graph, root) == []


# --- engine-level: the soft grounding gate now actually forces a visit ----------

def _engine():
    io = AsyncMock()
    io.telemetry = None
    return IdeaDagEngine(io=io, settings={})


def test_grounding_replan_now_injects_a_visit_for_plain_grounding_mandate():
    """Before the fix: `_grounding_replan` detected `grounded=False` (0 visits) for a
    plain 'do not guess' mandate but no hook injected anything (`injected <= 0`), so it
    gave up (`ungrounded-no-followup`) and the run finalized with zero visits ever
    executed. After the fix: the same call now injects a visit node and asks for
    another pass.
    """
    engine = _engine()
    graph, root, _ = _graph_with_done_search()
    with patch(
        "agent.app.idea_engine.evaluate_candidate_coverage",
        return_value=CandidateCoverageResult(satisfied=True, named=[], resolved=[], missing=[]),
    ):
        result = engine._grounding_replan(graph, GROUNDING_MANDATE, steps=0, max_steps=50)

    assert result is True
    assert getattr(engine, "_grounding_replans", 0) == 1
    assert len(_visit_children(graph, root)) == 1


def test_grounding_replan_still_gives_up_when_nothing_to_visit():
    """No search has completed yet -> nothing for the hook to seed a visit from. The
    gate must still fail closed (no infinite loop) exactly as before the fix.
    """
    engine = _engine()
    mandate = GROUNDING_MANDATE
    graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
    with patch(
        "agent.app.idea_engine.evaluate_candidate_coverage",
        return_value=CandidateCoverageResult(satisfied=True, named=[], resolved=[], missing=[]),
    ):
        result = engine._grounding_replan(graph, mandate, steps=0, max_steps=50)
    assert result is False
    assert getattr(engine, "_grounding_replans", 0) == 0
