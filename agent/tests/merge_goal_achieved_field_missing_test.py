"""A missing ``goal_achieved`` field is a schema failure, not a negative verdict.

``MergeLeafAction.execute`` reads ``synthesized_data.get("goal_achieved", False)``, so a
completion that never emitted the field at all was indistinguishable from one that deliberately
answered ``false`` -- the 2026-08-21 A/B measured this on 7 of 160 real completions
(llama3.2:3b echoing the input blob back under the schema's field names, one 14b
``goaled_achieved`` typo whose ``goal_evaluation`` plainly said ACHIEVED).

The default is unchanged (still not-achieved, the safe direction); only the diagnosis is new.
The regression half of this file guards the inverse mistake -- a real ``false`` must NOT be
relabelled a schema failure.

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

_MARKER = "goal_achieved_field_missing"


def _merge_node() -> tuple[IdeaDag, str]:
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "compare dish diameters", status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = "Which telescope has the largest dish diameter"
    merge = graph.add_child(parent.node_id, "Merge: compare dish diameters",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = [{
        "node_id": "c0",
        "title": "FAST",
        "status": "done",
        "result": {"success": True, "action": "visit", "url": "https://example.org/fast",
                   "content": "FAST is the telescope with the largest dish diameter, 500 m."},
        "is_merge": False,
    }]
    return graph, merge.node_id


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _run(response: str) -> tuple[IdeaDag, str, dict]:
    graph, merge_id = _merge_node()
    result = asyncio.run(
        MergeLeafAction(settings=load_idea_dag_settings()).execute(
            graph, merge_id, _ScriptedIO(response)
        )
    )
    return graph, merge_id, result


def test_an_absent_goal_achieved_field_is_flagged_and_logged(caplog):
    """The llama3.2:3b input-echo shape: schema field names, no verdict."""
    with caplog.at_level(logging.WARNING):
        graph, merge_id, result = _run(json.dumps({
            "summary": "FAST, Arecibo, RATAN-600",
            "key_findings": ["FAST is 500 m"],
        }))
    details = graph.get_node(merge_id).details
    assert details[_MARKER] is True
    assert details[DetailKey.GOAL_ACHIEVED.value] is False
    assert result["goal_achieved"] is False
    assert any("did not include a usable goal_achieved field" in r.message for r in caplog.records)


def test_a_typoed_field_name_counts_as_absent(caplog):
    """The 14b case: ``goaled_achieved: true`` beside an evaluation that says achieved."""
    with caplog.at_level(logging.WARNING):
        graph, merge_id, result = _run(json.dumps({
            "summary": "FAST wins",
            "goaled_achieved": True,
            "goal_evaluation": "The goal was achieved: FAST has the largest dish.",
        }))
    assert graph.get_node(merge_id).details[_MARKER] is True
    assert result["goal_achieved"] is False
    assert any("schema-adherence failure" in r.message for r in caplog.records)


def test_a_deliberate_false_verdict_is_not_flagged(caplog):
    """Regression: every genuine negative verdict must NOT be relabelled a schema failure."""
    with caplog.at_level(logging.WARNING):
        graph, merge_id, result = _run(json.dumps({
            "summary": "inconclusive",
            "goal_achieved": False,
            "goal_evaluation": "RATAN-600 was never checked",
            "missing_requirements": ["RATAN-600"],
        }))
    assert _MARKER not in graph.get_node(merge_id).details
    assert result["goal_achieved"] is False
    assert not any("goal_achieved field" in r.message for r in caplog.records)


def test_a_true_verdict_is_not_flagged():
    graph, merge_id, result = _run(json.dumps({
        "summary": "FAST wins", "goal_achieved": True,
        "goal_evaluation": "answered", "missing_requirements": [],
    }))
    assert _MARKER not in graph.get_node(merge_id).details
    assert result["goal_achieved"] is True


def test_a_falsy_non_bool_present_field_is_not_flagged():
    """``goal_achieved: ""`` is a (badly typed) answer -- the field IS there."""
    graph, merge_id, result = _run(json.dumps({
        "summary": "s", "goal_achieved": "", "goal_evaluation": "e",
    }))
    assert _MARKER not in graph.get_node(merge_id).details
    assert result["goal_achieved"] is False


def test_an_unparseable_response_keeps_its_own_diagnosis(caplog):
    """The JSON-decode fallback supplies the key itself, so it is not a schema-adherence miss."""
    with caplog.at_level(logging.WARNING):
        graph, merge_id, result = _run("not json at all")
    assert _MARKER not in graph.get_node(merge_id).details
    assert result["synthesized"]["goal_evaluation"] == "Failed to parse LLM response"
    assert not any("schema-adherence failure" in r.message for r in caplog.records)
