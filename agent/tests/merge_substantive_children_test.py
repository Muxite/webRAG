"""A merge is only worth an LLM call when 2+ children actually found something.

``should_create_merge_node`` used to gate on ``len(node.children) >= 2`` alone. The
2026-08-21 diagnostic found 4 of 7 real merge calls passing that gate while synthesizing a
SINGLE content-bearing child: the other "child" was a Serper-403 search the connector had
recorded as ``success: true, content: "", url: None`` -- success-shaped and empty, so no
status check could see it. Combining one source with nothing can only restate that source,
and the merge's ``goal_achieved`` verdict then terminates the branch on it.

The gate below counts children carrying a non-echoed, non-empty result field. Genuine
multi-source merges (the diagnostic's 4-telescope case) must still fire.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.merge import SimpleMergePolicy


def _policy(**overrides) -> SimpleMergePolicy:
    settings = load_idea_dag_settings()
    settings.update(overrides)
    return SimpleMergePolicy(settings=settings)


def _visit_result(content: str = "The Arecibo dish measured 305 m.") -> dict:
    return {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://example.org/arecibo",
        "title": "Arecibo",
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
        "vector_context": [],
        "count": 5,
        "results": [],
    }


def _parent_with(results, statuses=None) -> tuple[IdeaDag, str]:
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent", status=IdeaNodeStatus.ACTIVE)
    for i, result in enumerate(results):
        child = graph.add_child(
            parent.node_id,
            f"child {i}",
            details={DetailKey.ACTION.value: (result or {}).get("action", "visit")},
            status=(statuses or {}).get(i, IdeaNodeStatus.DONE),
        )
        if result is not None:
            child.details[DetailKey.ACTION_RESULT.value] = result
    return graph, parent.node_id


def test_one_substantive_child_plus_an_empty_search_does_not_merge():
    graph, parent_id = _parent_with([_visit_result(), _empty_search_result()])
    assert _policy().should_create_merge_node(graph, parent_id) is False


def test_two_substantive_children_still_merge():
    graph, parent_id = _parent_with([_visit_result("Arecibo: 305 m"), _visit_result("FAST: 500 m")])
    assert _policy().should_create_merge_node(graph, parent_id) is True


def test_the_four_telescope_case_is_unaffected():
    """The genuine multi-content merge the compaction fix was built for."""
    graph, parent_id = _parent_with([
        _visit_result("Arecibo: 305 m"),
        _visit_result("RATAN-600: 576 m ring"),
        _visit_result("Green Bank: 100 m"),
        _visit_result("FAST: 500 m"),
    ])
    assert _policy().should_create_merge_node(graph, parent_id) is True


def test_a_search_that_returned_hits_counts_as_substantive():
    hits = dict(_empty_search_result(), results=[{"url": "https://x", "title": "FAST"}])
    graph, parent_id = _parent_with([_visit_result(), hits])
    assert _policy().should_create_merge_node(graph, parent_id) is True


def test_two_empty_searches_do_not_merge():
    graph, parent_id = _parent_with([_empty_search_result(), _empty_search_result()])
    assert _policy().should_create_merge_node(graph, parent_id) is False


def test_a_child_with_no_result_at_all_is_not_substantive():
    graph, parent_id = _parent_with([_visit_result(), None],
                                    statuses={1: IdeaNodeStatus.FAILED})
    assert _policy().should_create_merge_node(graph, parent_id) is False


def test_think_and_merge_style_payloads_count():
    think = {"action": IdeaActionType.THINK.value, "success": True,
             "thinking_content": "FAST is the largest filled-aperture dish."}
    synth = {"action": IdeaActionType.MERGE.value, "success": True,
             "synthesized": {"summary": "two candidates"}}
    assert SimpleMergePolicy._payload_is_substantive(think) is True
    assert SimpleMergePolicy._payload_is_substantive(synth) is True


def test_an_unknown_action_shape_is_treated_as_substantive():
    """Fails open: a result shape this gate has never seen merges exactly as before."""
    novel = {"action": "some_future_action", "success": True, "verdict": "supported"}
    assert SimpleMergePolicy._payload_is_substantive(novel) is True


def test_a_list_valued_result_is_substantive_if_any_element_is():
    assert SimpleMergePolicy._payload_is_substantive(
        [_empty_search_result(), _visit_result()]
    ) is True
    assert SimpleMergePolicy._payload_is_substantive(
        [_empty_search_result(), _empty_search_result()]
    ) is False


def test_the_kill_switch_restores_the_count_only_gate():
    graph, parent_id = _parent_with([_visit_result(), _empty_search_result()])
    policy = _policy(merge_require_substantive_children_enabled=False)
    assert policy.should_create_merge_node(graph, parent_id) is True


def test_the_flag_has_a_typed_view_and_a_json_default():
    settings = load_idea_dag_settings()
    assert settings["merge_require_substantive_children_enabled"] is True
    cfg = IdeaConfig.from_settings(settings)
    assert cfg.merge.require_substantive_children_enabled is True
    settings["merge_require_substantive_children_enabled"] = False
    assert IdeaConfig.from_settings(settings).merge.require_substantive_children_enabled is False
