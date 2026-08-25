"""``MergeLeafAction``'s verdict propagation was one-directional.

On success it wrote ``goal_achieved=True`` onto the PARENT and marked it DONE. On failure it
wrote ``merge_incomplete``/``merge_should_skip`` onto itself and told the parent nothing --
so a parent (often the root) carrying an earlier optimistic stamp kept it. Paired with
finalize's old root-first read, that is how a run covering 3 of 7 candidates finalized as a
success.

The asymmetry is closed here: a not-achieved verdict propagates too. Finalize no longer
depends on this (``resolve_goal_achieved`` prefers the root-most merge node's own verdict),
so this is defence in depth plus an honest graph dump for offline forensics -- the analysis
scripts read ``execution.graph`` directly.

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


def _run(response: dict, parent_seed: dict | None = None):
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "enumerate first ascents",
                             status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = "List the first-ascent year of all seven summits"
    parent.details.update(parent_seed or {})
    merge = graph.add_child(parent.node_id, "Merge: enumerate first ascents",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = [{
        "node_id": "c0",
        "title": "Mont Blanc",
        "status": "done",
        "result": {"success": True, "action": "visit", "url": "https://example.org/mb",
                   "content": "Mont Blanc was first climbed in 1786."},
        "is_merge": False,
    }]
    asyncio.run(
        MergeLeafAction(settings=load_idea_dag_settings()).execute(
            graph, merge.node_id, _ScriptedIO(json.dumps(response))
        )
    )
    return graph, parent, graph.get_node(merge.node_id)


_INCOMPLETE = {
    "summary": "Only one of the seven summits was researched.",
    "key_findings": ["Mont Blanc 1786"],
    "goal_evaluation": "Six of seven candidates were never visited.",
    "goal_achieved": False,
    "missing_requirements": ["Matterhorn", "Kilimanjaro", "Aconcagua", "Erebus", "Denali",
                             "Vinson Massif"],
}

_COMPLETE = {
    "summary": "All seven first-ascent years were gathered.",
    "key_findings": ["Mont Blanc 1786"],
    "goal_evaluation": "Every candidate was visited and dated.",
    "goal_achieved": True,
    "missing_requirements": [],
}


def test_not_achieved_propagates_to_the_parent():
    _, parent, merge = _run(_INCOMPLETE)
    assert merge.details["merge_should_skip"] is True
    assert parent.details[DetailKey.GOAL_ACHIEVED.value] is False


def test_not_achieved_clears_a_stale_optimistic_parent_stamp():
    """The live shape: something wrote True on the parent before the merge ran."""
    _, parent, _ = _run(_INCOMPLETE,
                        parent_seed={DetailKey.GOAL_ACHIEVED.value: True})
    assert parent.details[DetailKey.GOAL_ACHIEVED.value] is False


def test_not_achieved_does_not_mark_the_parent_done():
    """Only the success branch may terminate the parent branch."""
    _, parent, _ = _run(_INCOMPLETE)
    assert parent.status == IdeaNodeStatus.ACTIVE


def test_achieved_still_propagates_true_and_marks_the_parent_done():
    _, parent, merge = _run(_COMPLETE)
    assert merge.status == IdeaNodeStatus.DONE
    assert parent.details[DetailKey.GOAL_ACHIEVED.value] is True
    assert parent.status == IdeaNodeStatus.DONE
