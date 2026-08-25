"""An ACHIEVED verdict evidenced only by unvisited search snippets.

The 2026-08-21 reason-first A/B watched qwen2.5:14b elect the right telescope from SEARCH
RESULT SNIPPETS alone -- it never opened a page stating the illuminated aperture, and returned
``goal_achieved: true`` regardless, in BOTH prompt-ordering arms. ``goal_evidence_provenance``
reads the distinction straight out of the merged payloads (a snippet lives inside a SEARCH
result's ``results`` list; fetched content lives in the result's own fields), so no extra LLM
call is involved.

Detection is unconditional (marker + warning); the downgrade is gated behind
``merge_require_visited_evidence_enabled``, default OFF.

No network: every LLM response is scripted.
"""
from __future__ import annotations

import asyncio
import json
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.merge import (
    GOAL_EVIDENCE_DIRECT,
    GOAL_EVIDENCE_NONE,
    GOAL_EVIDENCE_SNIPPET,
    goal_evidence_provenance,
)

_GOAL = "Which telescope has the largest dish diameter"
_MARKER = "goal_achieved_snippet_only"

_SNIPPET_CHILD = {
    "node_id": "c0",
    "title": "search for the largest dish",
    "status": "done",
    "result": {
        "success": True,
        "action": "search",
        "query": "largest radio telescope dish",
        "results": [
            {"title": "FAST telescope has the largest dish",
             "url": "https://example.org/fast",
             "description": "Five-hundred-metre Aperture Spherical Telescope."},
        ],
    },
    "is_merge": False,
}

_VISITED_CHILD = {
    "node_id": "c1",
    "title": "visit the FAST page",
    "status": "done",
    "result": {
        "success": True,
        "action": "visit",
        "url": "https://example.org/fast",
        "content": "The FAST telescope has the largest dish, 500 m diameter, 300 m illuminated.",
    },
    "is_merge": False,
}

_OFF_TOPIC_VISIT = {
    "node_id": "c2",
    "title": "visit an unrelated page",
    "status": "done",
    "result": {"success": True, "action": "visit", "url": "https://example.org/eiffel",
               "content": "The Eiffel Tower was completed in 1889."},
    "is_merge": False,
}


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _run(children, achieved=True, **overrides) -> tuple[IdeaDag, str, str, dict]:
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "compare dishes", status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = _GOAL
    merge = graph.add_child(parent.node_id, "Merge: compare dishes", status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = children
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({
        "summary": "FAST wins", "goal_achieved": achieved,
        "goal_evaluation": "answered", "missing_requirements": [],
    })
    result = asyncio.run(
        MergeLeafAction(settings=settings).execute(graph, merge.node_id, _ScriptedIO(response))
    )
    return graph, parent.node_id, merge.node_id, result


# --- the classifier ---------------------------------------------------------------------

def test_snippet_only_evidence_is_classified_snippet():
    assert goal_evidence_provenance(_GOAL, [_SNIPPET_CHILD]) == GOAL_EVIDENCE_SNIPPET


def test_an_off_topic_visit_does_not_launder_snippet_evidence():
    """Something was fetched, but nothing fetched addresses the GOAL."""
    assert goal_evidence_provenance(_GOAL, [_SNIPPET_CHILD, _OFF_TOPIC_VISIT]) == GOAL_EVIDENCE_SNIPPET


def test_fetched_content_addressing_the_goal_is_direct():
    assert goal_evidence_provenance(_GOAL, [_SNIPPET_CHILD, _VISITED_CHILD]) == GOAL_EVIDENCE_DIRECT


def test_nothing_relevant_anywhere_is_none_not_snippet():
    """``none`` is not a negative verdict -- a paraphrase-only answer lands here too."""
    assert goal_evidence_provenance(_GOAL, [_OFF_TOPIC_VISIT]) == GOAL_EVIDENCE_NONE


def test_a_nested_merge_fails_open_to_direct():
    """A merge hands up a synthesis, not the pages behind it; its provenance is unknowable."""
    nested = {"node_id": "m0", "title": "Merge", "status": "done", "is_merge": True,
              "result": {"summary": "The FAST telescope has the largest dish diameter, 500 m.",
                         "goal_achieved": True}}
    assert goal_evidence_provenance(_GOAL, [nested]) == GOAL_EVIDENCE_DIRECT


def test_a_query_restating_the_goal_is_not_direct_evidence():
    """``query`` is the node's own input read back -- it would otherwise self-certify."""
    echo = {"node_id": "c9", "title": "search", "status": "done", "is_merge": False,
            "result": {"success": True, "action": "search",
                       "query": "which telescope has the largest dish diameter", "results": []}}
    assert goal_evidence_provenance(_GOAL, [echo]) == GOAL_EVIDENCE_NONE


def test_an_empty_goal_never_classifies():
    assert goal_evidence_provenance("", [_SNIPPET_CHILD]) == GOAL_EVIDENCE_NONE


# --- the merge action -------------------------------------------------------------------

def test_detection_is_unconditional_and_the_default_verdict_is_untouched(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _run([_SNIPPET_CHILD])
    details = graph.get_node(merge_id).details
    assert details[_MARKER] is True
    assert result["goal_achieved"] is True
    assert details[DetailKey.GOAL_ACHIEVED.value] is True
    assert graph.get_node(merge_id).status == IdeaNodeStatus.DONE
    assert graph.get_node(parent_id).status == IdeaNodeStatus.DONE
    assert any("unvisited search-result snippet" in r.message for r in caplog.records)
    assert not any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_flag_downgrades_a_snippet_only_verdict(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _run(
            [_SNIPPET_CHILD], merge_require_visited_evidence_enabled=True
        )
    details = graph.get_node(merge_id).details
    assert result["goal_achieved"] is False
    assert details[DetailKey.GOAL_ACHIEVED.value] is False
    assert details["merge_incomplete"] is True
    assert details["merge_should_skip"] is True
    assert graph.get_node(parent_id).status == IdeaNodeStatus.ACTIVE
    # The downgrade now propagates explicitly rather than leaving the parent unwritten, so a
    # stale optimistic stamp cannot survive it (see merge_negative_verdict_propagation_test).
    assert graph.get_node(parent_id).details[DetailKey.GOAL_ACHIEVED.value] is False
    assert any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_raw_model_output_survives_the_downgrade():
    _, _, _, result = _run([_SNIPPET_CHILD], merge_require_visited_evidence_enabled=True)
    assert result["synthesized"]["goal_achieved"] is True
    assert result["goal_achieved"] is False


def test_a_visited_verdict_is_untouched_with_the_flag_on(caplog):
    with caplog.at_level(logging.WARNING):
        graph, _, merge_id, result = _run(
            [_SNIPPET_CHILD, _VISITED_CHILD], merge_require_visited_evidence_enabled=True
        )
    assert result["goal_achieved"] is True
    assert _MARKER not in graph.get_node(merge_id).details
    assert not any("snippet" in r.message for r in caplog.records)


def test_evidence_the_overlap_bar_missed_is_not_downgraded():
    """``none`` must not be read as a negative -- only a positive snippet match acts."""
    graph, _, merge_id, result = _run(
        [_OFF_TOPIC_VISIT], merge_require_visited_evidence_enabled=True
    )
    assert result["goal_achieved"] is True
    assert _MARKER not in graph.get_node(merge_id).details


def test_a_not_achieved_verdict_is_never_examined():
    graph, _, merge_id, result = _run([_SNIPPET_CHILD], achieved=False,
                                      merge_require_visited_evidence_enabled=True)
    assert result["goal_achieved"] is False
    assert _MARKER not in graph.get_node(merge_id).details
