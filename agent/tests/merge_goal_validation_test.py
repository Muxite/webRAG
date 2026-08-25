"""``_validate_goal_achievement`` used to be tautological on one side and unsatisfiable on the other.

Three checks decided whether children had gathered anything addressing the goal:

* ``goal.lower() in content.lower()`` -- demanded the page repeat the goal's exact phrasing
  verbatim. Real pages essentially never do, so a correct answer written in the page's own words
  scored no better than an empty page.
* the same test against ``query``.
* ``len(results) > 0`` -- ANY non-empty search-result list passed, a list of entirely off-topic
  hits included.

The second one is the dangerous half. :meth:`SimpleMergePolicy.merge` recurses to root, so this
verdict is written to the ROOT's ``goal_achieved`` before any merge node's own LLM call runs, and
``idea_finalize`` reads the root's flag FIRST -- ahead of the LLM-verified merge verdicts. One
off-topic search hit anywhere could therefore short-circuit finalization for the whole run.

Both checks are now the same containment-overlap test on the goal's content words. No network:
every case is a fixture.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.merge import (
    SimpleMergePolicy,
    _GOAL_RELEVANCE_MIN_OVERLAP,
    _goal_tokens,
    _relevance_overlap,
)


GOAL = "Which telescope has the largest dish diameter"

_PARAPHRASE = (
    "FAST, in Guizhou, is the world's largest filled-aperture radio dish at 500 m diameter, "
    "ahead of the retired 305 m Arecibo reflector."
)
_OFF_TOPIC = (
    "The Eiffel Tower was completed in 1889 and stands 330 metres tall, a wrought-iron lattice "
    "designed by Gustave Eiffel for the World's Fair."
)


def _policy() -> SimpleMergePolicy:
    return SimpleMergePolicy(settings=load_idea_dag_settings())


def _visit(content: str) -> dict:
    return {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://example.org/page",
        "content": content,
    }


def _search(results: list) -> dict:
    return {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "query": "notable structures",
        "count": 5,
        "results": results,
    }


_ON_TOPIC_HIT = {
    "title": "List of largest radio telescopes",
    "url": "https://en.wikipedia.org/wiki/List_of_radio_telescopes",
    "description": "FAST has the largest single-dish diameter of any telescope, at 500 m.",
}
_OFF_TOPIC_HIT = {
    "title": "Eiffel Tower - Wikipedia",
    "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
    "description": "The tower was completed in 1889 and stands 330 metres tall.",
}


def _validate(result: dict | None, goal: str = GOAL) -> bool:
    """Run the check exactly as ``merge`` calls it, on a single merged child entry."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "parent", status=IdeaNodeStatus.ACTIVE)
    node.details[DetailKey.GOAL.value] = goal
    entry = {"node_id": "c0", "title": "child", "status": "done", "result": result}
    return _policy()._validate_goal_achievement(graph, node, [entry])


# --- the tokenizer/overlap primitives -------------------------------------------------

def test_tokens_are_lowercased_split_on_punctuation_and_drop_shorts():
    assert _goal_tokens("Largest dish, in m?  A B") == {"largest", "dish"}
    assert _goal_tokens(None) == set()


def test_overlap_is_containment_not_jaccard():
    """A long page covering the whole goal scores 1.0; Jaccard would drown it in page size."""
    goal = _goal_tokens("dish diameter")
    page = _goal_tokens("dish diameter " + "unrelated filler words here " * 50)
    assert _relevance_overlap(goal, page) == 1.0
    assert _relevance_overlap(set(), page) == 0.0


# --- content / query ------------------------------------------------------------------

def test_content_quoting_the_goal_verbatim_still_passes():
    """Regression: the one case the old substring check could actually satisfy."""
    assert _validate(_visit(f"Article answering '{GOAL}': it is FAST.")) is True


def test_content_paraphrasing_the_goal_now_passes():
    """Was FALSE before: no verbatim substring, despite being the actual answer."""
    assert _relevance_overlap(_goal_tokens(GOAL), _goal_tokens(_PARAPHRASE)) >= _GOAL_RELEVANCE_MIN_OVERLAP
    assert _validate(_visit(_PARAPHRASE)) is True


def test_content_unrelated_to_the_goal_fails():
    assert _validate(_visit(_OFF_TOPIC)) is False


def test_a_query_restating_the_goal_passes():
    result = _search([])
    result["query"] = "largest telescope dish diameter comparison"
    assert _validate(result) is True


def test_an_off_topic_query_with_no_results_fails():
    assert _validate(_search([])) is False


# --- search results -------------------------------------------------------------------

def test_an_empty_results_list_fails():
    assert _validate(_search([])) is False


def test_a_non_empty_but_off_topic_results_list_now_fails():
    """The tautology: this passed before purely because the list was non-empty."""
    assert _validate(_search([_OFF_TOPIC_HIT, _OFF_TOPIC_HIT])) is False


def test_an_on_topic_hit_passes():
    assert _validate(_search([_ON_TOPIC_HIT])) is True


def test_one_on_topic_hit_among_off_topic_ones_is_enough():
    assert _validate(_search([_OFF_TOPIC_HIT, _ON_TOPIC_HIT, _OFF_TOPIC_HIT])) is True


def test_hits_carrying_only_urls_fail():
    """Bare URLs are not evidence the goal was addressed."""
    assert _validate(_search([{"url": "https://example.org/a"}, {"url": "https://example.org/b"}])) is False


def test_a_non_dict_hit_is_scored_on_its_text():
    assert _validate(_search(["the largest telescope dish diameter is FAST's"])) is True
    assert _validate(_search(["1889 wrought iron"])) is False


# --- degenerate inputs ----------------------------------------------------------------

def test_a_node_with_no_goal_is_permissive_as_before():
    assert _validate(_visit(_OFF_TOPIC), goal="") is True


def test_a_goal_with_no_content_words_is_permissive():
    """Nothing to measure overlap against -- keep the old answer rather than fail everything."""
    assert _validate(_visit(_OFF_TOPIC), goal="?? 5 a b") is True


def test_a_non_dict_result_entry_is_skipped():
    assert _validate(None) is False


# --- the root cascade (why this check matters beyond its own node) --------------------

def _root_with_action_children(results: list) -> IdeaDag:
    """The shallow DAG shape: root's own children are leaf actions."""
    graph = IdeaDag(root_title="root")
    root = graph.get_node(graph.root_id())
    root.details[DetailKey.GOAL.value] = GOAL
    for i, result in enumerate(results):
        child = graph.add_child(
            graph.root_id(),
            f"child {i}",
            details={DetailKey.ACTION.value: result.get("action", "visit")},
            status=IdeaNodeStatus.DONE,
        )
        child.details[DetailKey.ACTION_RESULT.value] = result
    return graph


def test_an_off_topic_result_no_longer_marks_the_root_goal_achieved():
    """The amplification case. The pre-check now writes the PROVISIONAL key -- finalize
    reads the authoritative ``goal_achieved``, which only ``MergeLeafAction`` writes -- but
    the overlap verdict itself still has to come out False here."""
    graph = _root_with_action_children([_search([_OFF_TOPIC_HIT]), _visit(_OFF_TOPIC)])
    _policy().merge(graph, graph.root_id(), recursive=True)
    assert graph.get_node(graph.root_id()).details[DetailKey.GOAL_ACHIEVED_PROVISIONAL.value] is False


def test_a_genuinely_on_topic_result_still_marks_the_root_goal_achieved():
    graph = _root_with_action_children([_search([_ON_TOPIC_HIT]), _visit(_PARAPHRASE)])
    _policy().merge(graph, graph.root_id(), recursive=True)
    assert graph.get_node(graph.root_id()).details[DetailKey.GOAL_ACHIEVED_PROVISIONAL.value] is True


def test_the_recursion_carries_the_verdict_up_from_a_child_level_merge():
    """``merge`` walks to root, so merging mid-tree writes the root's flag too."""
    graph = _root_with_action_children([_search([_OFF_TOPIC_HIT])])
    root_id = graph.root_id()
    sub = graph.add_child(root_id, "sub-goal", status=IdeaNodeStatus.ACTIVE)
    sub.details[DetailKey.GOAL.value] = GOAL
    leaf = graph.add_child(sub.node_id, "leaf", status=IdeaNodeStatus.DONE)
    leaf.details[DetailKey.ACTION_RESULT.value] = _search([_OFF_TOPIC_HIT])
    _policy().merge(graph, sub.node_id, recursive=True)
    assert sub.details[DetailKey.GOAL_ACHIEVED_PROVISIONAL.value] is False
    assert graph.get_node(root_id).details[DetailKey.GOAL_ACHIEVED_PROVISIONAL.value] is False
