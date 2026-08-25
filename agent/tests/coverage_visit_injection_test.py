"""The coverage gate measures VISITS, but every remediation path it drove created SEARCHES.

``candidate_coverage`` counts only nodes with a successful ``visit`` action result -- search
results and node titles are deliberately excluded. Yet the remediation it triggers reaches the
visit-injecting hooks only through ``_grounding_replan``, and each of those hooks early-returns
unless the mandate happens to carry a ``must visit`` phrase or navigation targets. For an
ordinary enumerated-candidate mandate every visit injector declines, so the extension produces
search-shaped work or nothing.

The measurements agree, twice over:

* n=24 A/B with the gate ON: 46 searches, 1 visit, score movement zero (+0.019, p=0.70).
* n=1 with the structural caps lifted (root re-expansion + breadth-aware fan-out): the graph
  widened as intended, and the freed budget went to 7 search nodes / 55 searches against 2
  visits -- visits went DOWN versus the control, and the score with it.

Widening the graph without changing what the new nodes DO just buys more of the wrong action.
This injector closes that: for each candidate the gate reports missing, it deterministically
mints a VISIT -- consuming an already-completed search for that candidate where one exists, and
otherwise a fresh search paired with a dependent visit.

No network: graphs are hand-built.
"""
from __future__ import annotations

import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.post_expansion_hooks import inject_coverage_visits


MANDATE = (
    "For each of the following, find the year of first ascent:\n"
    "1. Mount Everest\n2. Aconcagua\n3. Denali\n4. Kilimanjaro\n"
)
LOG = logging.getLogger("test")


def _graph() -> IdeaDag:
    return IdeaDag(root_title="root", root_details={DetailKey.ORIGINAL_GOAL.value: MANDATE})


def _add_search(graph, query, *, done=True, results=None):
    node = graph.add_child(
        graph.root_id(), f"Search {query}",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: query,
            DetailKey.IS_LEAF.value: True,
        },
        status=IdeaNodeStatus.DONE if done else IdeaNodeStatus.PENDING,
    )
    if done:
        node.details[DetailKey.ACTION_RESULT.value] = {
            "action": "search", "success": True, "query": query,
            "results": results or [{"url": "https://example.org/x", "title": query}],
        }
    return node


def _visits(graph):
    return [n for n in graph.iter_depth_first()
            if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.VISIT.value]


def test_a_visit_is_injected_for_every_missing_candidate():
    graph = _graph()
    injected = inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG)
    assert injected == 4
    assert len(_visits(graph)) == 4


def test_the_injected_visits_name_the_missing_candidates():
    graph = _graph()
    inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG)
    blob = " ".join(n.title for n in _visits(graph))
    for name in ("Mount Everest", "Aconcagua", "Denali", "Kilimanjaro"):
        assert name in blob


def test_an_existing_completed_search_is_reused_rather_than_duplicated():
    """A search for the candidate already ran -- the visit should consume its results."""
    graph = _graph()
    search = _add_search(graph, "Mount Everest first ascent")
    inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG)

    everest = [n for n in _visits(graph) if "Mount Everest" in n.title]
    assert len(everest) == 1
    requires = everest[0].details.get(DetailKey.REQUIRES_DATA.value) or {}
    assert requires.get("source_node_id") == search.node_id


def test_a_candidate_without_a_search_gets_a_search_and_a_dependent_visit():
    graph = _graph()
    inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG)

    denali_visit = [n for n in _visits(graph) if "Denali" in n.title][0]
    requires = denali_visit.details.get(DetailKey.REQUIRES_DATA.value) or {}
    source = graph.get_node(requires.get("source_node_id"))
    assert source is not None
    assert NodeDetailsExtractor.get_action(source.details) == IdeaActionType.SEARCH.value
    assert "Denali" in source.details.get(DetailKey.QUERY.value, "")


def test_an_already_covered_candidate_is_not_re_injected():
    """The gate resolves a candidate from a VISITED page's identity."""
    graph = _graph()
    visited = graph.add_child(
        graph.root_id(), "Visit Mount Everest",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {
                "action": "visit", "success": True,
                "url": "https://en.wikipedia.org/wiki/Mount_Everest",
                "page_title": "Mount Everest", "content": "First climbed in 1953.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    assert visited.status == IdeaNodeStatus.DONE

    injected = inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG)
    assert injected == 3
    titles = " ".join(n.title for n in _visits(graph) if n.node_id != visited.node_id)
    assert "Mount Everest" not in titles


def test_full_coverage_injects_nothing():
    graph = _graph()
    for name in ("Mount Everest", "Aconcagua", "Denali", "Kilimanjaro"):
        graph.add_child(
            graph.root_id(), f"Visit {name}",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.ACTION_RESULT.value: {
                    "action": "visit", "success": True,
                    "url": f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
                    "page_title": name, "content": "x",
                },
            },
            status=IdeaNodeStatus.DONE,
        )
    assert inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG) == 0


def test_a_non_enumerated_mandate_injects_nothing():
    """Fails open, exactly like the gate it serves."""
    graph = IdeaDag(root_title="root")
    chain = "Find the engineer who designed the Pontcysyllte Aqueduct, then their birth year."
    assert inject_coverage_visits(graph, graph.root_id(), 0, chain, LOG) == 0


def test_injection_is_bounded():
    graph = IdeaDag(root_title="root")
    mandate = "Find the founding year of each:\n" + "\n".join(
        f"{i}. Institution {i}" for i in range(1, 20)
    )
    injected = inject_coverage_visits(graph, graph.root_id(), 0, mandate, LOG, max_injections=5)
    assert injected == 5


def test_calling_twice_does_not_duplicate_visits():
    """Remediation can fire more than once in a run; the second pass must be a near no-op."""
    graph = _graph()
    first = inject_coverage_visits(graph, graph.root_id(), 0, MANDATE, LOG)
    second = inject_coverage_visits(graph, graph.root_id(), 1, MANDATE, LOG)
    assert first == 4
    assert second == 0


def test_a_missing_node_is_handled():
    graph = _graph()
    assert inject_coverage_visits(graph, "no-such-node", 0, MANDATE, LOG) == 0
