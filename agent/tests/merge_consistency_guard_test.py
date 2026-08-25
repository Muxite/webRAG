"""A merge completion cannot both declare the goal met and list what is still missing.

``MergeLeafAction.execute`` read ``goal_achieved`` and ``missing_requirements`` out of the same
JSON object and never compared them. Honouring a contradictory ``goal_achieved: true`` marks the
merge node DONE *and* force-marks its parent DONE, permanently terminating a branch the model's
own other field says is incomplete -- and nothing downstream reconsiders a DONE parent.

The guard resolves the contradiction toward the more specific claim: a non-empty
``missing_requirements`` downgrades the verdict, which then falls through the existing
not-achieved branch (``merge_incomplete`` / ``merge_should_skip``). The raw model output is left
untouched inside ``synthesized`` as an audit record.

No network: every LLM response is scripted.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _merge_node() -> tuple[IdeaDag, str, str]:
    """Parent (ACTIVE) with a merge child holding one merged result."""
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
                   "content": "FAST is 500 m across."},
        "is_merge": False,
    }]
    return graph, parent.node_id, merge.node_id


def _run(response: dict) -> tuple[IdeaDag, str, str, dict]:
    graph, parent_id, merge_id = _merge_node()
    result = asyncio.run(
        MergeLeafAction(settings=load_idea_dag_settings()).execute(
            graph, merge_id, _ScriptedIO(json.dumps(response))
        )
    )
    return graph, parent_id, merge_id, result


def test_a_consistent_achieved_verdict_is_honoured():
    """Regression: the ordinary success path must not move."""
    graph, parent_id, merge_id, result = _run({
        "summary": "FAST wins", "goal_achieved": True,
        "goal_evaluation": "answered", "missing_requirements": [],
    })
    merge = graph.get_node(merge_id)
    assert merge.details[DetailKey.GOAL_ACHIEVED.value] is True
    assert merge.status == IdeaNodeStatus.DONE
    assert graph.get_node(parent_id).status == IdeaNodeStatus.DONE
    assert graph.get_node(parent_id).details[DetailKey.GOAL_ACHIEVED.value] is True
    assert result["goal_achieved"] is True


def test_achieved_with_missing_requirements_is_downgraded():
    graph, parent_id, merge_id, result = _run({
        "summary": "FAST is 500 m", "goal_achieved": True,
        "goal_evaluation": "mostly answered",
        "missing_requirements": ["RATAN-600's ring diameter was never checked"],
    })
    merge = graph.get_node(merge_id)
    assert merge.details[DetailKey.GOAL_ACHIEVED.value] is False
    assert result["goal_achieved"] is False


def test_the_downgrade_leaves_neither_node_force_marked_done():
    graph, parent_id, merge_id, _ = _run({
        "summary": "s", "goal_achieved": True, "goal_evaluation": "e",
        "missing_requirements": ["RATAN-600"],
    })
    parent = graph.get_node(parent_id)
    assert graph.get_node(merge_id).status != IdeaNodeStatus.DONE
    assert parent.status == IdeaNodeStatus.ACTIVE
    # The downgrade now propagates explicitly rather than leaving the parent unwritten, so a
    # stale optimistic stamp cannot survive it (see merge_negative_verdict_propagation_test).
    assert parent.details[DetailKey.GOAL_ACHIEVED.value] is False


def test_the_downgrade_routes_through_the_existing_not_achieved_branch():
    graph, _, merge_id, _ = _run({
        "summary": "s", "goal_achieved": True, "goal_evaluation": "e",
        "missing_requirements": ["RATAN-600"],
    })
    details = graph.get_node(merge_id).details
    assert details["merge_incomplete"] is True
    assert details["merge_should_skip"] is True
    assert details["missing_requirements"] == ["RATAN-600"]


def test_an_already_consistent_not_achieved_verdict_is_unchanged():
    graph, parent_id, merge_id, result = _run({
        "summary": "s", "goal_achieved": False, "goal_evaluation": "e",
        "missing_requirements": ["RATAN-600"],
    })
    assert result["goal_achieved"] is False
    assert graph.get_node(merge_id).details["merge_should_skip"] is True
    assert graph.get_node(parent_id).status == IdeaNodeStatus.ACTIVE


def test_the_raw_model_output_is_preserved_for_audit():
    """The downgrade is the engine's verdict; ``synthesized`` still shows what the model said."""
    _, _, _, result = _run({
        "summary": "s", "goal_achieved": True, "goal_evaluation": "e",
        "missing_requirements": ["RATAN-600"],
    })
    assert result["synthesized"]["goal_achieved"] is True
    assert result["goal_achieved"] is False


def test_a_truthy_non_bool_achieved_with_missing_requirements_is_also_downgraded():
    """Normalization runs first, so the guard sees a real bool whatever the model emitted."""
    _, _, merge_id, result = _run({
        "summary": "s", "goal_achieved": "yes", "goal_evaluation": "e",
        "missing_requirements": ["RATAN-600"],
    })
    assert result["goal_achieved"] is False
