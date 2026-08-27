"""Inline search recovery for a dead declared URL
(``visit_declared_url_search_fallback_enabled``, kill-switch, default ON).

The dead-declared-URL cascade (see ``visit_dead_url_fallback_test.py``) has four sources:
parent search hits, a sibling's result, the stored link index, and the previous hop's page
link menu. Whether the FIRST of those can fire depends on the compiled plan happening to
contain a dedicated search leaf -- a per-task LLM planning choice no arm profile controls.
When a plan has none and no page has been visited yet, all four come back empty and the leaf
re-raises the 404, so ``visit_count`` never increments and every gated downstream validator
collapses to 0.0 (recorded on tasks 130 and 132).

These tests pin the fifth source: one inline ``io.search`` for the leaf's own link idea, fed
through the same dead-URL/chrome/sibling filtering and selection path as the other four.
"""

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.actions import VisitLeafAction
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey, IdeaActionType


DEAD_URL = "https://en.wikipedia.org/wiki/Cincinnati_and_Northern_Kentucky_Suspension_Bridge"
REAL_URL = "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge"


class FakeChroma:
    """An EMPTY stored link index: nothing has been visited, so nothing was ever stored."""

    async def add_to_chroma(self, collection, ids, metadatas, documents):
        return True

    async def list_collections(self):
        return []

    async def query_chroma(self, collection, query_texts, n_results=10):
        return {"metadatas": [[]], "distances": [[]]}


class FakeIO:
    def __init__(self, search_results=()):
        self.fetches = []
        self.searches = []
        self.telemetry = None
        self.connector_chroma = FakeChroma()
        self._search_results = list(search_results)

    async def fetch_url(self, url: str, retries: int = 3, timeout_seconds=None) -> str:
        self.fetches.append(url)
        if url == DEAD_URL:
            raise RuntimeError(f"HTTP fetch failed: {url} status=404")
        return (
            "<html><body><h1>Roebling Suspension Bridge</h1><p>The main span is 1,057 feet "
            "(322 m), the longest in the world when it opened in 1867.</p></body></html>"
        )

    async def visit(self, url: str, timeout_seconds=None) -> str:
        self.fetches.append(url)
        return "page content"

    async def search(self, query: str, count: int = 10, timeout_seconds=None):
        self.searches.append(query)
        return list(self._search_results)

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return None


def _graph_with_dead_url_leaf():
    """The recorded shape: no search leaf, no completed sibling, nothing visited yet."""
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Read the main span of the John A. Roebling Suspension Bridge",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": DEAD_URL,
            "link_count": 1,
        },
    )
    return graph, node.node_id


def _hit(url):
    return {"title": "John A. Roebling Suspension Bridge", "url": url, "description": "Ohio River"}


@pytest.mark.asyncio
async def test_inline_search_recovers_when_all_four_cascade_sources_are_empty():
    graph, node_id = _graph_with_dead_url_leaf()
    io = FakeIO(search_results=[_hit(REAL_URL)])

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is True
    assert result[ActionResultKey.URL.value] == REAL_URL
    assert io.fetches == [DEAD_URL, REAL_URL]
    assert io.searches, "the inline search fallback should have fired"


@pytest.mark.asyncio
async def test_search_recovered_pool_still_drops_the_dead_url():
    """The recovered results go through the SAME filtering path, not around it."""
    graph, node_id = _graph_with_dead_url_leaf()
    io = FakeIO(search_results=[_hit(DEAD_URL), _hit(REAL_URL)])

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.URL.value] == REAL_URL
    assert io.fetches.count(DEAD_URL) == 1


@pytest.mark.asyncio
async def test_flag_off_is_byte_identical_to_the_old_abort():
    graph, node_id = _graph_with_dead_url_leaf()
    io = FakeIO(search_results=[_hit(REAL_URL)])

    result = await VisitLeafAction(
        settings={"visit_declared_url_search_fallback_enabled": False}
    ).execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is False
    assert result[ActionResultKey.HTTP_STATUS.value] == 404
    assert result[ActionResultKey.RETRYABLE.value] is False
    assert io.fetches == [DEAD_URL]
    assert io.searches == []


@pytest.mark.asyncio
async def test_empty_search_preserves_the_original_failure_surface():
    graph, node_id = _graph_with_dead_url_leaf()
    on = FakeIO(search_results=[])
    off = FakeIO(search_results=[])

    recovered = await VisitLeafAction().execute(graph, node_id, on)
    baseline = await VisitLeafAction(
        settings={"visit_declared_url_search_fallback_enabled": False}
    ).execute(graph, node_id, off)

    for payload in (recovered, baseline):
        assert payload[ActionResultKey.SUCCESS.value] is False
        assert payload[ActionResultKey.HTTP_STATUS.value] == 404
        assert payload[ActionResultKey.RETRYABLE.value] is False
    assert recovered[ActionResultKey.ERROR.value] == baseline[ActionResultKey.ERROR.value]
    assert on.searches, "the fallback fired"
    assert on.fetches == [DEAD_URL]


@pytest.mark.asyncio
async def test_search_exception_preserves_the_original_failure_surface():
    class RaisingIO(FakeIO):
        async def search(self, query: str, count: int = 10, timeout_seconds=None):
            self.searches.append(query)
            raise RuntimeError("search backend unavailable")

    graph, node_id = _graph_with_dead_url_leaf()
    io = RaisingIO()

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is False
    assert result[ActionResultKey.HTTP_STATUS.value] == 404
    assert io.searches, "the fallback was attempted"
    assert io.fetches == [DEAD_URL]


@pytest.mark.asyncio
async def test_search_recovery_used_despite_zero_link_count():
    """A planner-emitted ``link_count: 0`` must not zero out an otherwise-recovered pool.

    The consumption gate (``if link_count > len(urls_to_visit)``) used to read the raw
    ``link_count`` straight from the plan node, so ``0 > 0`` was always False and the recovered
    candidates were dropped -- the original dead-URL error was re-raised even though the fallback
    above had just found a real page. See docs/handoffs/DAG_V3_PHASE0_NIGHT3_HANDOFF_2026-08-28.md
    section 4 (task 130).
    """
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Read the main span of the John A. Roebling Suspension Bridge",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": DEAD_URL,
            "link_count": 0,
        },
    )
    io = FakeIO(search_results=[_hit(REAL_URL)])

    result = await VisitLeafAction().execute(graph, node.node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is True
    assert result[ActionResultKey.URL.value] == REAL_URL


@pytest.mark.asyncio
async def test_no_inline_search_when_the_cascade_already_has_candidates():
    """One extra search only when everything else is empty -- no unconditional spend."""
    from agent.app.idea_policies.base import IdeaNodeStatus

    graph = IdeaDag(root_title="root")
    parent = graph.add_child(
        graph.root_id(), "search", details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    parent.status = IdeaNodeStatus.DONE
    parent.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": "hit", "url": REAL_URL}],
    }
    node = graph.add_child(
        parent.node_id,
        "Read the main span of the John A. Roebling Suspension Bridge",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": DEAD_URL,
            "link_count": 1,
        },
    )
    io = FakeIO(search_results=[_hit(REAL_URL)])

    result = await VisitLeafAction().execute(graph, node.node_id, io)

    assert result[ActionResultKey.URL.value] == REAL_URL
    assert io.searches == []
