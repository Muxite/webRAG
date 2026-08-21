"""Which goal text a merge node synthesizes (and is validated) against.

``MergeLeafAction.execute`` used to resolve ``{original_goal}`` as
``node GOAL -> node ORIGINAL_GOAL -> node.title``. A merge node's title is a structural label
(``Merge: {parent.title}``, or whatever the planner named its synthesis step), and every node
the engine touches gets a GOAL stamped from its own title when nothing better exists -- so both
the merge PROMPT and the goal-relevance checks were handed a label instead of the research
question, and the parent fallback underneath was unreachable because the title is never empty.

Resolution now walks UP to the nearest ancestor carrying real goal text, bottoming out at the
root's goal/mandate.

No network: every LLM response is scripted.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus

_MANDATE = "Find the illuminated aperture of the telescope with the largest dish"
_SUB_GOAL = "Find the current status of each candidate telescope"

_CHILD = {
    "node_id": "c0",
    "title": "search the candidates",
    "status": "done",
    "is_merge": False,
    "result": {
        "success": True,
        "action": "search",
        "query": "largest radio telescope dish",
        "results": [{"title": "FAST", "url": "https://example.org/fast",
                     "description": "Five-hundred-metre Aperture Spherical Telescope."}],
    },
}


class _CapturingIO:
    """Records the merge prompt actually sent."""

    def __init__(self):
        self.messages = None

    def build_llm_payload(self, messages=None, **kw):
        self.messages = messages
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return json.dumps({"summary": "FAST", "goal_achieved": False,
                           "goal_evaluation": "partial", "missing_requirements": []})


def _tree(intermediate_goal: str | None = None, merge_goal: str | None = None,
          merge_title: str = "Merge: gather candidate telescopes"):
    """root (real mandate) -> intermediate decompose step -> merge node."""
    graph = IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})
    graph.get_node(graph.root_id()).details[DetailKey.GOAL.value] = _MANDATE
    parent = graph.add_child(graph.root_id(), "gather candidate telescopes",
                             status=IdeaNodeStatus.ACTIVE)
    if intermediate_goal is not None:
        parent.details[DetailKey.GOAL.value] = intermediate_goal
    merge = graph.add_child(parent.node_id, merge_title, status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.ACTION.value] = IdeaActionType.MERGE.value
    merge.details[DetailKey.MERGED_RESULTS.value] = [_CHILD]
    if merge_goal is not None:
        merge.details[DetailKey.GOAL.value] = merge_goal
        merge.details[DetailKey.ORIGINAL_GOAL.value] = merge_goal
    return graph, parent, merge


def _prompt_goal(graph, merge_id) -> str:
    io = _CapturingIO()
    asyncio.run(
        MergeLeafAction(settings=load_idea_dag_settings()).execute(graph, merge_id, io)
    )
    user = [m for m in io.messages if m.get("role") == "user"][-1]["content"]
    return json.loads(user)["original_goal"]


def test_a_merge_with_no_real_goal_anywhere_resolves_to_the_root_mandate():
    """The intermediate step carries neither a GOAL nor a usable title, so the walk reaches
    root. The old chain stopped at the merge node's own ``Merge: ...`` title."""
    graph, parent, merge = _tree()
    parent.title = ""
    assert _prompt_goal(graph, merge.node_id) == _MANDATE


def test_the_merge_nodes_own_title_is_never_the_goal():
    graph, _, merge = _tree(merge_title="Merge: gather candidate telescopes")
    assert "Merge:" not in _prompt_goal(graph, merge.node_id)


def test_a_planner_authored_merge_goal_echoing_its_own_title_is_ignored():
    """``idea_engine`` stamps GOAL from the child's title when the plan authored none --
    on a merge child that is the structural label again, not a research question."""
    graph, _, merge = _tree(intermediate_goal=_SUB_GOAL,
                            merge_goal="Synthesize the findings",
                            merge_title="Synthesize the findings")
    assert _prompt_goal(graph, merge.node_id) == _SUB_GOAL


def test_a_merge_goal_stamped_from_the_parent_is_kept():
    """``_handle_merge_creation`` stamps the PARENT's goal onto the merge node; that is real
    text, distinct from the node's own title, and must survive."""
    graph, _, merge = _tree(merge_goal=_SUB_GOAL)
    assert _prompt_goal(graph, merge.node_id) == _SUB_GOAL


def test_a_real_intermediate_sub_goal_beats_walking_all_the_way_to_root():
    graph, _, merge = _tree(intermediate_goal=_SUB_GOAL)
    assert _prompt_goal(graph, merge.node_id) == _SUB_GOAL


def test_an_intermediate_step_with_no_goal_contributes_its_title():
    """A decompose step's title IS its sub-goal -- unlike a merge node's."""
    graph, parent, merge = _tree()
    parent.title = "Compare the dish diameters"
    assert _prompt_goal(graph, merge.node_id) == "Compare the dish diameters"


def test_a_merge_directly_under_root_gets_the_root_goal():
    graph = IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})
    graph.get_node(graph.root_id()).details[DetailKey.GOAL.value] = _MANDATE
    merge = graph.add_child(graph.root_id(), f"Merge: {_MANDATE}",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.ACTION.value] = IdeaActionType.MERGE.value
    merge.details[DetailKey.MERGED_RESULTS.value] = [_CHILD]
    assert _prompt_goal(graph, merge.node_id) == _MANDATE


def test_a_root_that_never_got_a_goal_stamped_falls_back_to_its_mandate():
    graph, parent, merge = _tree()
    graph.get_node(graph.root_id()).details.pop(DetailKey.GOAL.value, None)
    parent.details.pop(DetailKey.GOAL.value, None)
    parent.title = ""
    assert _prompt_goal(graph, merge.node_id) == _MANDATE


def test_the_provenance_check_sees_the_same_goal_as_the_prompt():
    """Snippet-only detection measures overlap against the goal; with the merge label it
    could never match, so the check was inert on exactly the nodes it exists for."""
    graph = IdeaDag(root_title="Which telescope has the largest dish diameter",
                    root_details={"mandate": "Which telescope has the largest dish diameter"})
    graph.get_node(graph.root_id()).details[DetailKey.GOAL.value] = (
        "Which telescope has the largest dish diameter"
    )
    merge = graph.add_child(graph.root_id(), "Merge: compare dishes",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.ACTION.value] = IdeaActionType.MERGE.value
    merge.details[DetailKey.MERGED_RESULTS.value] = [{
        "node_id": "c0", "title": "search", "status": "done", "is_merge": False,
        "result": {"success": True, "action": "search", "query": "largest dish",
                   "results": [{"title": "FAST telescope has the largest dish diameter",
                                "url": "https://example.org/fast",
                                "description": "500 m dish."}]},
    }]

    class _AchievedIO(_CapturingIO):
        async def query_llm_with_fallback(self, payload, model_name=None,
                                          fallback_model=None, timeout_seconds=None):
            return json.dumps({"summary": "FAST", "goal_achieved": True,
                               "goal_evaluation": "answered", "missing_requirements": []})

    asyncio.run(
        MergeLeafAction(settings=load_idea_dag_settings()).execute(
            graph, merge.node_id, _AchievedIO()
        )
    )
    assert graph.get_node(merge.node_id).details.get("goal_achieved_snippet_only") is True
