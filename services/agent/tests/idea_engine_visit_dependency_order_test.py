import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


class _StubIO:
    telemetry = None

    def set_telemetry(self, t):
        return None


def _mk_engine():
    return IdeaDagEngine(io=_StubIO(), settings={})


def test_sequential_reorder_prefers_search_before_url_less_visit():
    """
    Regression test for IDEA-visit scheduling:

    When a visit node has no explicit URL (only link_idea), it typically depends on a sibling
    search to populate actionable URLs. Even in sequential mode, the engine must execute the
    search first (regardless of score) to avoid premature visit failures.
    """
    engine = _mk_engine()
    graph = IdeaDag(root_title="root")
    root_id = graph.root_id()

    search = graph.add_child(
        root_id,
        "Search for something",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
        status=IdeaNodeStatus.PENDING,
        score=0.10,
    )
    visit = graph.add_child(
        root_id,
        "Visit result",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            # No explicit URL, will need sibling search results.
            "link_idea": "Visit the top search result",
            "link_count": 1,
        },
        status=IdeaNodeStatus.PENDING,
        score=0.20,
    )

    # Simulate selection choosing visit first due to score.
    reordered = engine._reorder_for_sequential(graph, visit, [search.node_id, visit.node_id], step_index=0)
    assert reordered is not None
    assert reordered.node_id == search.node_id


def test_sequential_reorder_does_not_override_explicit_url_visit():
    """If visit has an explicit URL, it should not be reordered behind search."""
    engine = _mk_engine()
    graph = IdeaDag(root_title="root")
    root_id = graph.root_id()

    search = graph.add_child(
        root_id,
        "Search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
        status=IdeaNodeStatus.PENDING,
        score=0.10,
    )
    visit = graph.add_child(
        root_id,
        "Visit explicit",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": "https://example.com",
            "link_count": 1,
        },
        status=IdeaNodeStatus.PENDING,
        score=0.20,
    )

    reordered = engine._reorder_for_sequential(graph, visit, [search.node_id, visit.node_id], step_index=0)
    assert reordered is None

