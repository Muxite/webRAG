"""The sibling-link fallback must not ground a NAMED visit on another arm's page.

A visit with no declared URL and no usable search pool ends at
``VisitLeafAction._extract_url_from_sibling_results``, whose last two steps are guesses: the
link with the best word overlap, else the first sibling link outright. Live evidence from the
2026-08-25 breadth A/B (276 cells): 13 of 62 such visits opened a page sharing no identity token
with their own leaf -- "Visit the Suez Canal page" reading ``/wiki/Erie_Canal``, "Visit author
page for Toni Morrison" reading ``/wiki/Jane_Austen`` -- and every one reported success.

``run_policy_visit_url_identity_guard`` (default off) restricts the sibling pool to links that
mention a name the leaf itself uses and declines when none does. The flag-off half of these
tests is the regression fence: today's guessing behaviour is preserved byte-for-byte.
"""

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.actions import VisitLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus

GUARD_ON = {"run_policy_visit_url_identity_guard": True}


def _fan_out(visit_title: str, sibling_url: str = "https://en.wikipedia.org/wiki/Erie_Canal"):
    """One completed sibling visit holding a page for a DIFFERENT entity, plus the leaf under
    test (no URL, no search anywhere -- the shape the live failures had)."""
    graph = IdeaDag(root_title="completion year of six canals")
    sibling = graph.add_child(
        graph.root_id(),
        "Visit the Erie Canal page",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    sibling.status = IdeaNodeStatus.DONE
    sibling.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": sibling_url,
        "links": [sibling_url],
    }
    node = graph.add_child(
        graph.root_id(),
        visit_title,
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    return graph, node


def test_a_named_leaf_is_not_handed_another_entitys_page():
    graph, node = _fan_out("Visit the Suez Canal page to find its completion year")

    assert VisitLeafAction(GUARD_ON)._extract_url_from_sibling_results(graph, node) is None


def test_flag_off_still_hands_over_that_page():
    """The bug, pinned: with the flag off the leaf is grounded on the wrong canal."""
    graph, node = _fan_out("Visit the Suez Canal page to find its completion year")

    assert (
        VisitLeafAction()._extract_url_from_sibling_results(graph, node)
        == "https://en.wikipedia.org/wiki/Erie_Canal"
    )


def test_a_leaf_that_names_nothing_keeps_its_only_candidate():
    """"Visit a source page" has no identity to violate, so the guard stays out of the way --
    refusing there would only cost the leaf its one candidate."""
    graph, node = _fan_out("Visit a source page for grounded evidence")

    assert (
        VisitLeafAction(GUARD_ON)._extract_url_from_sibling_results(graph, node)
        == "https://en.wikipedia.org/wiki/Erie_Canal"
    )


def test_the_leafs_own_page_still_comes_through():
    """The guard filters, it does not veto: a sibling link the leaf DOES name is still used."""
    graph, node = _fan_out(
        "Visit the Erie Canal page to find its completion year",
        sibling_url="https://en.wikipedia.org/wiki/Erie_Canal",
    )

    assert (
        VisitLeafAction(GUARD_ON)._extract_url_from_sibling_results(graph, node)
        == "https://en.wikipedia.org/wiki/Erie_Canal"
    )


def test_the_right_page_is_picked_out_of_a_mixed_sibling_pool():
    graph, node = _fan_out("Visit the Suez Canal page to find its completion year")
    sibling = graph.get_node(graph.get_node(graph.root_id()).children[0])
    sibling.details[DetailKey.ACTION_RESULT.value]["links"] = [
        "https://en.wikipedia.org/wiki/Erie_Canal",
        "https://en.wikipedia.org/wiki/Suez_Canal",
    ]

    assert (
        VisitLeafAction(GUARD_ON)._extract_url_from_sibling_results(graph, node)
        == "https://en.wikipedia.org/wiki/Suez_Canal"
    )


def test_an_intent_name_counts_as_the_leafs_own_name():
    """Titles are not the only place a leaf names its entity; the intent is read too."""
    graph, node = _fan_out("Visit the page")
    node.details[DetailKey.INTENT.value] = "the completion year of the Suez Canal"

    assert VisitLeafAction(GUARD_ON)._extract_url_from_sibling_results(graph, node) is None


def test_two_identically_named_leaves_leave_the_guard_nothing_to_judge():
    """Nothing separates this leaf from its sibling, so the guard declines to judge and the
    existing fallback stands — it must not refuse on a tie it cannot break."""
    graph, node = _fan_out("Visit the Erie Canal page")
    node.title = "Visit the Erie Canal page"

    assert (
        VisitLeafAction(GUARD_ON)._extract_url_from_sibling_results(graph, node)
        == "https://en.wikipedia.org/wiki/Erie_Canal"
    )


@pytest.mark.asyncio
async def test_end_to_end_the_visit_fails_instead_of_grounding_on_the_wrong_page():
    """The downstream point of the guard: with no other URL source, the leaf reports a plain
    failure the run can still remediate, in place of ``success=True`` on another arm's page."""
    from agent.app.idea_policies.action_constants import ActionResultKey
    from agent.tests.visit_url_extraction_test import FakeIO

    graph, node = _fan_out("Visit the Suez Canal page to find its completion year")
    node.details["link_count"] = 1

    payload = await VisitLeafAction(GUARD_ON).execute(graph, node.node_id, FakeIO())

    assert payload[ActionResultKey.SUCCESS.value] is False
    assert "erie_canal" not in str(payload).lower()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Visit the Suez Canal Wikipedia page to find its completion year", ["Suez", "Canal"]),
        ("Visit page for Caspian Sea", ["Caspian", "Sea"]),
        ("Visit a source page for grounded evidence", []),
        ("visit page", []),
        ("Open the Toni Morrison author page", ["Toni", "Morrison"]),
    ],
)
def test_named_entities_reads_capitalisation_minus_the_visit_verbs(text, expected):
    assert VisitLeafAction._named_entities(text) == expected


def test_named_entities_ignores_non_strings():
    assert VisitLeafAction._named_entities(None, 17, "Lake Baikal") == ["Lake", "Baikal"]
