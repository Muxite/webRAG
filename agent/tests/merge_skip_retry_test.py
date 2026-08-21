"""A skipped merge should not lock its parent out of ever merging again.

``should_create_merge_node``'s dedup check returned ``False`` from BOTH arms of its
``merge_should_skip`` test, so the first merge node a parent ever got ended synthesis for that
parent permanently -- including a merge skipped precisely because the goal was judged unmet,
which is the one case where later sibling findings should get a second hearing
(``ENGINE_DESIGN_REVIEW`` D4).

``merge_retry_after_skip_enabled`` (default OFF) permits exactly one narrow reopening: every
merge under the parent was skipped, and the substantive-child count has grown past the count
stamped when the skipped merge was minted. The skipped node itself is never touched -- the retry
mints a fresh sibling and leaves the old one as the run's audit record.

No network: nothing here needs an LLM except the consistency-guard case, whose response is
scripted.
"""
from __future__ import annotations

import asyncio
import copy
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.merge import (
    SUBSTANTIVE_CHILD_COUNT_AT_CREATION,
    SimpleMergePolicy,
)


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _policy(**overrides) -> SimpleMergePolicy:
    settings = load_idea_dag_settings()
    settings.update(overrides)
    return SimpleMergePolicy(settings=settings)


def _visit_result(content: str) -> dict:
    return {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://example.org/telescopes",
        "title": "telescopes",
        "content": content,
        "content_full": content,
    }


def _empty_search_result() -> dict:
    """The real Serper-403 shape: success-flagged, parameters echoed, nothing found."""
    return {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "query": "largest radio telescope dish",
        "intent": "find the dish diameter",
        "count": 5,
        "results": [],
    }


def _parent_with_one_finding() -> tuple[IdeaDag, str]:
    """Parent whose two children carry ONE substantive result between them."""
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "compare dish diameters",
                             status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = "Which telescope has the largest dish diameter"
    for title, result in (("visit FAST", _visit_result("FAST is 500 m across.")),
                          ("search dishes", _empty_search_result())):
        child = graph.add_child(parent.node_id, title,
                                details={DetailKey.ACTION.value: result["action"]},
                                status=IdeaNodeStatus.DONE)
        child.details[DetailKey.ACTION_RESULT.value] = result
    return graph, parent.node_id


def _add_finding(graph: IdeaDag, parent_id: str, content: str) -> str:
    child = graph.add_child(parent_id, "visit RATAN-600",
                            details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
                            status=IdeaNodeStatus.DONE)
    child.details[DetailKey.ACTION_RESULT.value] = _visit_result(content)
    return child.node_id


def _skipped_merge(policy: SimpleMergePolicy, graph: IdeaDag, parent_id: str) -> str:
    """Mint a merge node and put it in the state the engine leaves a skipped merge in."""
    merge_id = policy.create_merge_node(graph, parent_id)
    merge = graph.get_node(merge_id)
    merge.details["merge_incomplete"] = True
    merge.details["merge_should_skip"] = True
    merge.details["merge_skipped_reason"] = "Goal not achieved according to evaluation"
    merge.status = IdeaNodeStatus.SKIPPED
    return merge_id


def test_flag_off_keeps_the_lockout_even_after_new_evidence_arrives():
    """The regression pin: today's behaviour must be exactly what a default run still does."""
    policy = _policy()
    graph, parent_id = _parent_with_one_finding()
    _skipped_merge(policy, graph, parent_id)
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")
    assert policy.should_create_merge_node(graph, parent_id) is False


def test_flag_off_does_not_stamp_the_baseline_onto_merge_details():
    """Default-off runs produce byte-identical merge-node details."""
    graph, parent_id = _parent_with_one_finding()
    merge_id = _policy().create_merge_node(graph, parent_id)
    assert SUBSTANTIVE_CHILD_COUNT_AT_CREATION not in graph.get_node(merge_id).details


def test_flag_on_stamps_the_substantive_count_at_creation():
    graph, parent_id = _parent_with_one_finding()
    policy = _policy(merge_retry_after_skip_enabled=True)
    merge_id = policy.create_merge_node(graph, parent_id)
    assert graph.get_node(merge_id).details[SUBSTANTIVE_CHILD_COUNT_AT_CREATION] == 1


def test_flag_on_permits_a_retry_once_a_new_substantive_child_lands():
    policy = _policy(merge_retry_after_skip_enabled=True)
    graph, parent_id = _parent_with_one_finding()
    merge_id = _skipped_merge(policy, graph, parent_id)
    assert graph.get_node(merge_id).details[SUBSTANTIVE_CHILD_COUNT_AT_CREATION] == 1

    assert policy.should_create_merge_node(graph, parent_id) is False
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")
    assert policy.should_create_merge_node(graph, parent_id) is True


def test_the_retry_creates_a_second_distinct_merge_node():
    policy = _policy(merge_retry_after_skip_enabled=True)
    graph, parent_id = _parent_with_one_finding()
    first_id = _skipped_merge(policy, graph, parent_id)
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")

    second_id = policy.create_merge_node(graph, parent_id)
    assert second_id and second_id != first_id
    second = graph.get_node(second_id)
    assert second.status == IdeaNodeStatus.PENDING
    assert "merge_should_skip" not in second.details
    # The fresh node's own baseline is the grown count, so it cannot itself be retried
    # without yet more evidence.
    assert second.details[SUBSTANTIVE_CHILD_COUNT_AT_CREATION] == 2
    assert policy.should_create_merge_node(graph, parent_id) is False


def test_the_retry_leaves_the_skipped_merge_node_untouched():
    """The load-bearing safety property: statuses stay append-only, the audit record stands."""
    policy = _policy(merge_retry_after_skip_enabled=True)
    graph, parent_id = _parent_with_one_finding()
    first_id = _skipped_merge(policy, graph, parent_id)
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")
    before = copy.deepcopy(graph.get_node(first_id).details)

    policy.create_merge_node(graph, parent_id)

    first = graph.get_node(first_id)
    assert first.status == IdeaNodeStatus.SKIPPED
    assert first.details["merge_should_skip"] is True
    assert first.details == before


def test_flag_on_still_blocks_a_retry_when_no_new_evidence_arrived():
    """Re-running the same synthesis over the same children can only re-derive the same verdict."""
    policy = _policy(merge_retry_after_skip_enabled=True)
    graph, parent_id = _parent_with_one_finding()
    _skipped_merge(policy, graph, parent_id)
    # A child that found nothing does not move the count.
    barren = graph.add_child(parent_id, "search again",
                             details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
                             status=IdeaNodeStatus.DONE)
    barren.details[DetailKey.ACTION_RESULT.value] = _empty_search_result()
    assert policy.should_create_merge_node(graph, parent_id) is False


def test_a_merge_node_without_a_stamp_fails_closed():
    """Turning the flag on mid-run must not blanket-retry merges minted while it was off."""
    graph, parent_id = _parent_with_one_finding()
    off = _policy()
    merge_id = _skipped_merge(off, graph, parent_id)
    assert SUBSTANTIVE_CHILD_COUNT_AT_CREATION not in graph.get_node(merge_id).details
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")
    assert _policy(merge_retry_after_skip_enabled=True).should_create_merge_node(
        graph, parent_id
    ) is False


def test_a_merge_that_was_not_skipped_never_retries():
    policy = _policy(merge_retry_after_skip_enabled=True)
    graph, parent_id = _parent_with_one_finding()
    merge_id = policy.create_merge_node(graph, parent_id)
    graph.get_node(merge_id).status = IdeaNodeStatus.DONE
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")
    assert policy.should_create_merge_node(graph, parent_id) is False


def test_a_consistency_guard_downgrade_is_a_retryable_skip():
    """Closes the loop with the guard shipped in ``edc3f328``.

    A completion claiming ``goal_achieved: true`` while listing what is still missing is
    downgraded and routed through the not-achieved branch, which sets ``merge_should_skip``.
    That skip is the exact case D4 should reopen once the missing piece is actually found.
    """
    settings = load_idea_dag_settings()
    settings["merge_retry_after_skip_enabled"] = True
    policy = SimpleMergePolicy(settings=settings)
    graph, parent_id = _parent_with_one_finding()
    merge_id = policy.create_merge_node(graph, parent_id)

    result = asyncio.run(MergeLeafAction(settings=settings).execute(
        graph, merge_id, _ScriptedIO(json.dumps({
            "summary": "FAST is 500 m",
            "goal_achieved": True,
            "goal_evaluation": "mostly answered",
            "missing_requirements": ["RATAN-600's ring diameter was never checked"],
        })),
    ))
    assert result["goal_achieved"] is False
    merge = graph.get_node(merge_id)
    assert merge.details["merge_should_skip"] is True
    merge.status = IdeaNodeStatus.SKIPPED  # what ``_handle_merge_creation`` does next

    assert policy.should_create_merge_node(graph, parent_id) is False
    _add_finding(graph, parent_id, "RATAN-600 is a 576 m ring.")
    assert policy.should_create_merge_node(graph, parent_id) is True


def test_the_flag_has_a_typed_view_and_a_json_default():
    settings = load_idea_dag_settings()
    assert settings["merge_retry_after_skip_enabled"] is False
    cfg = IdeaConfig.from_settings(settings)
    assert cfg.merge.retry_after_skip_enabled is False
    settings["merge_retry_after_skip_enabled"] = True
    assert IdeaConfig.from_settings(settings).merge.retry_after_skip_enabled is True
