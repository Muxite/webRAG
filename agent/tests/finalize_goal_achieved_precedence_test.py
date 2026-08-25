"""The ROOT's ``goal_achieved`` used to outrank every LLM-verified merge verdict below it.

``SimpleMergePolicy.merge`` recurses to root, so its cheap keyword-overlap pre-check
(``_validate_goal_achievement``) wrote ``goal_achieved`` onto the ROOT at merge-*creation*
time -- before any merge node's own LLM call had run. ``build_final_payload`` then read the
root FIRST and only fell back to a *disjunction* over merge nodes, which can raise a False to
True but can never lower a True to False.

The observable consequence on a 7-candidate breadth task: ``MergeLeafAction`` correctly found
3/7 coverage, set ``goal_achieved=False`` + ``merge_should_skip=True``, and the engine marked
the merge node SKIPPED -- and the run still reported ``success=True`` / ``goal_achieved=True``
off the root's stale optimistic stamp. Every measurement taken on that scoreboard was grading
against a gate that could not fail.

Two changes are pinned here:

* the cheap pre-check now writes ``goal_achieved_provisional``, so ``goal_achieved`` is only
  ever written by the authoritative :class:`MergeLeafAction` path;
* :func:`resolve_goal_achieved` prefers the merge node closest to the root -- the run's actual
  final aggregation -- over the root's own stamp, and treats ``merge_should_skip`` as a veto.

No network: every case is a hand-built graph.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import resolve_goal_achieved
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.merge import SimpleMergePolicy


GOAL = "Which telescope has the largest dish diameter"


def _merge_details(**extra) -> dict:
    details = {DetailKey.ACTION.value: IdeaActionType.MERGE.value}
    details.update(extra)
    return details


def _graph() -> IdeaDag:
    return IdeaDag(root_title="root", root_details={DetailKey.GOAL.value: GOAL})


def test_skipped_merge_vetoes_the_roots_optimistic_stamp():
    """The exact 2-of-7-coverage shape: root says True, the merge that ran says otherwise."""
    graph = _graph()
    root = graph.get_node(graph.root_id())
    root.details[DetailKey.GOAL_ACHIEVED.value] = True

    merge = graph.add_child(
        graph.root_id(),
        "merge",
        details=_merge_details(**{
            DetailKey.GOAL_ACHIEVED.value: False,
            "merge_should_skip": True,
        }),
        status=IdeaNodeStatus.SKIPPED,
    )
    assert merge.status == IdeaNodeStatus.SKIPPED

    assert resolve_goal_achieved(graph, root) is False


def test_merge_should_skip_vetoes_even_without_an_explicit_false():
    graph = _graph()
    root = graph.get_node(graph.root_id())
    root.details[DetailKey.GOAL_ACHIEVED.value] = True
    graph.add_child(
        graph.root_id(),
        "merge",
        details=_merge_details(merge_should_skip=True),
        status=IdeaNodeStatus.SKIPPED,
    )

    assert resolve_goal_achieved(graph, root) is False


def test_successful_merge_still_reports_achieved():
    """The fix must not turn every run False -- a genuine success has to survive it."""
    graph = _graph()
    root = graph.get_node(graph.root_id())
    graph.add_child(
        graph.root_id(),
        "merge",
        details=_merge_details(**{DetailKey.GOAL_ACHIEVED.value: True}),
        status=IdeaNodeStatus.DONE,
    )

    assert resolve_goal_achieved(graph, root) is True


def test_root_most_merge_outranks_a_deeper_one():
    """A deep sub-merge succeeding does not make the run's top-level aggregation succeed."""
    graph = _graph()
    root = graph.get_node(graph.root_id())

    top = graph.add_child(
        graph.root_id(),
        "top merge",
        details=_merge_details(**{
            DetailKey.GOAL_ACHIEVED.value: False,
            "merge_should_skip": True,
        }),
        status=IdeaNodeStatus.SKIPPED,
    )
    branch = graph.add_child(top.node_id, "branch", details={})
    graph.add_child(
        branch.node_id,
        "sub merge",
        details=_merge_details(**{DetailKey.GOAL_ACHIEVED.value: True}),
        status=IdeaNodeStatus.DONE,
    )

    assert resolve_goal_achieved(graph, root) is False


def test_sibling_merges_at_the_same_depth_must_all_agree():
    graph = _graph()
    root = graph.get_node(graph.root_id())
    graph.add_child(
        graph.root_id(), "merge a",
        details=_merge_details(**{DetailKey.GOAL_ACHIEVED.value: True}),
        status=IdeaNodeStatus.DONE,
    )
    graph.add_child(
        graph.root_id(), "merge b",
        details=_merge_details(**{DetailKey.GOAL_ACHIEVED.value: False}),
        status=IdeaNodeStatus.SKIPPED,
    )

    assert resolve_goal_achieved(graph, root) is False


def test_falls_back_to_root_when_no_merge_expressed_a_verdict():
    """Runs that never built a merge node keep the previous behaviour."""
    graph = _graph()
    root = graph.get_node(graph.root_id())
    root.details[DetailKey.GOAL_ACHIEVED.value] = True
    graph.add_child(graph.root_id(), "leaf", details={})

    assert resolve_goal_achieved(graph, root) is True

    root.details[DetailKey.GOAL_ACHIEVED.value] = False
    assert resolve_goal_achieved(graph, root) is False


def test_merge_node_without_any_verdict_key_does_not_veto():
    """A merge node that has not run yet carries no opinion and must not be read as False."""
    graph = _graph()
    root = graph.get_node(graph.root_id())
    root.details[DetailKey.GOAL_ACHIEVED.value] = True
    graph.add_child(
        graph.root_id(), "pending merge",
        details=_merge_details(),
        status=IdeaNodeStatus.PENDING,
    )

    assert resolve_goal_achieved(graph, root) is True


def test_no_root_returns_false():
    graph = _graph()
    assert resolve_goal_achieved(graph, None) is False


def test_cheap_precheck_no_longer_writes_the_authoritative_key():
    """``SimpleMergePolicy.merge`` must leave ``goal_achieved`` for ``MergeLeafAction``."""
    graph = _graph()
    child = graph.add_child(
        graph.root_id(),
        "visit",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "url": "https://example.org/fast",
                "content": (
                    "FAST, in Guizhou, is the world's largest filled-aperture radio dish at "
                    "500 m diameter, ahead of the retired 305 m Arecibo reflector."
                ),
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    assert child.status == IdeaNodeStatus.DONE

    policy = SimpleMergePolicy(settings=load_idea_dag_settings())
    outcome = policy.merge(graph, graph.root_id(), recursive=True)

    root = graph.get_node(graph.root_id())
    # The pre-check still runs and still reports its verdict...
    assert outcome["goal_achieved"] is True
    assert root.details[DetailKey.GOAL_ACHIEVED_PROVISIONAL.value] is True
    # ...but it no longer occupies the key finalize trusts.
    assert DetailKey.GOAL_ACHIEVED.value not in root.details
