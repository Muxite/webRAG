"""Regression tests for F37: the opt-in page-identity relevance signal layered onto the
grounding gate (`agent.app.idea_policies.grounding.evaluate_grounding`).

Root cause: `evaluate_grounding` is pure set arithmetic over visited URLs — any 2 visited
pages (or 1 followed link) satisfy a navigation mandate, with zero title/URL/content
relevance check. Two completely off-topic pages trivially "ground" the answer. These tests
lock in the fix (`require_page_identity=True`) and its byte-identical-off default.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.grounding import evaluate_grounding
from agent.app.idea_policies.mandate_requirements import MandateRequirements

MANDATE = (
    "Navigate Wikipedia by following links (do not use search) to find who designed the "
    "Golden Gate Bridge."
)


def _add_visit(graph, root, *, goal, url, title):
    node = graph.add_child(
        parent_id=root,
        title=goal,
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.GOAL.value: goal,
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "url": url,
                "page_title": title,
            },
        },
    )
    node.status = IdeaNodeStatus.DONE
    return node


def _off_topic_graph():
    """Two completely off-topic visits (as in the diagnostic's failure mode) — neither page
    has anything to do with the leaf's own subject (Golden Gate Bridge)."""
    graph = IdeaDag(root_title=MANDATE, root_details={"mandate": MANDATE})
    root = graph.root_id()
    _add_visit(
        graph, root,
        goal="Visit the page about the Golden Gate Bridge designer.",
        url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        title="Eiffel Tower - Wikipedia",
    )
    _add_visit(
        graph, root,
        goal="Visit the page about the Golden Gate Bridge designer.",
        url="https://en.wikipedia.org/wiki/Statue_of_Liberty",
        title="Statue of Liberty - Wikipedia",
    )
    return graph


def _on_topic_graph():
    graph = IdeaDag(root_title=MANDATE, root_details={"mandate": MANDATE})
    root = graph.root_id()
    _add_visit(
        graph, root,
        goal="Visit the page about the Golden Gate Bridge designer.",
        url="https://en.wikipedia.org/wiki/Golden_Gate_Bridge",
        title="Golden Gate Bridge - Wikipedia",
    )
    _add_visit(
        graph, root,
        goal="Visit the Golden Gate Bridge designer's biography page.",
        url="https://en.wikipedia.org/wiki/Joseph_Strauss",
        title="Joseph Strauss (engineer) - Wikipedia",
    )
    return graph


def _zero_subject_token_graph():
    """A leaf whose own goal text names no proper noun at all (a boilerplate template),
    visiting a page unrelated to the mandate's real subject."""
    graph = IdeaDag(root_title=MANDATE, root_details={"mandate": MANDATE})
    root = graph.root_id()
    _add_visit(
        graph, root,
        goal="visit the page to confirm the figure",
        url="https://en.wikipedia.org/wiki/Random_Page",
        title="Random Page - Wikipedia",
    )
    _add_visit(
        graph, root,
        goal="visit another page to confirm the figure",
        url="https://en.wikipedia.org/wiki/Another_Random_Page",
        title="Another Random Page - Wikipedia",
    )
    return graph


REQ_NAVIGATION = MandateRequirements(navigation=True, grounding=False)


def test_off_topic_visits_pass_when_flag_off_byte_identical():
    graph = _off_topic_graph()
    result = evaluate_grounding(graph, REQ_NAVIGATION)
    assert result.grounded is True
    assert result.distinct_visits == 2


def test_off_topic_visits_pass_when_flag_explicitly_off():
    graph = _off_topic_graph()
    result = evaluate_grounding(graph, REQ_NAVIGATION, require_page_identity=False)
    assert result.grounded is True


def test_off_topic_visits_fail_when_flag_on():
    graph = _off_topic_graph()
    result = evaluate_grounding(graph, REQ_NAVIGATION, require_page_identity=True)
    assert result.grounded is False
    assert result.missing


def test_on_topic_visits_still_pass_when_flag_on():
    graph = _on_topic_graph()
    off_result = evaluate_grounding(graph, REQ_NAVIGATION, require_page_identity=False)
    on_result = evaluate_grounding(graph, REQ_NAVIGATION, require_page_identity=True)
    assert off_result.grounded is True
    assert on_result.grounded is True


def test_zero_subject_tokens_degrades_safely_when_flag_on():
    """A leaf with no extractable subject tokens can't corroborate ANY page — excluding it
    from the relevance check (rather than failing it closed) keeps a boilerplate-goal leaf
    from spuriously breaking grounding that already passed with the flag off."""
    graph = _zero_subject_token_graph()
    off_result = evaluate_grounding(graph, REQ_NAVIGATION, require_page_identity=False)
    on_result = evaluate_grounding(graph, REQ_NAVIGATION, require_page_identity=True)
    assert off_result.grounded is True
    assert on_result.grounded is True
    assert on_result.distinct_visits == off_result.distinct_visits == 2
