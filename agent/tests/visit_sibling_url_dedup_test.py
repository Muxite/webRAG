"""Sibling visit-URL dedup (``visit_sibling_url_dedup``, opt-in).

A visit leaf that arrives without a URL of its own is handed one by a sibling-blind
cascade (parent search results, sibling results, Chroma links), and that cascade ranks
the same pool the same way for every such sibling. So a fan-out of per-entity page reads
collapses onto one page: 16.2% of recorded sibling-visit batches contain a duplicate URL
and half of those have EVERY sibling on the same page (ASSUMPTION_AUDIT.md T1-4).

These tests pin both halves: default OFF reproduces the collapse byte-for-byte, and ON
routes the second sibling to the next candidate instead. A URL the planner declared
itself is never dropped -- that duplicate is the planner's call, not the cascade's.
"""

import asyncio

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.actions import VISIT_URL_CLAIMS_KEY, VisitLeafAction
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


class FakeChroma:
    async def add_to_chroma(self, collection, ids, metadatas, documents):
        return True

    async def list_collections(self):
        return []

    async def query_chroma(self, collection, query_texts, n_results=10):
        return {"metadatas": [[]], "distances": [[]]}


class FakeIO:
    """Records every fetch, so "how many pages did the batch actually pull" is assertable."""

    def __init__(self):
        self.fetches = []
        self.telemetry = None
        self.connector_chroma = FakeChroma()

    async def fetch_url(self, url: str, retries: int = 3, timeout_seconds=None) -> str:
        self.fetches.append(url)
        return (
            "<html><body><h1>Test Page</h1><p>Content for a test page with enough text "
            "to look like a real fetch.</p></body></html>"
        )

    async def visit(self, url: str, timeout_seconds=None) -> str:
        self.fetches.append(url)
        return "Test Page Content"

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return None


URL_ONE = "https://example.com/alpha"
URL_TWO = "https://example.com/beta"


def _batch(urls=(URL_ONE, URL_TWO), n_visits=2, visit_details=None):
    """A search parent whose results feed two URL-less visit siblings."""
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(
        graph.root_id(), "search", details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    parent.status = IdeaNodeStatus.DONE
    parent.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": f"Result {i}", "url": u} for i, u in enumerate(urls)],
    }
    children = []
    for i in range(n_visits):
        details = {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_idea": f"entity {i}",
            "link_count": 1,
        }
        details.update(visit_details or {})
        children.append(graph.add_child(parent.node_id, f"visit entity {i}", details=dict(details)))
    return graph, parent, children


def _mark_done(node, url):
    node.status = IdeaNodeStatus.DONE
    node.details["visit_url"] = url
    node.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value, "success": True, "url": url, "content": "x" * 400,
    }


@pytest.mark.asyncio
async def test_siblings_collapse_onto_one_url_by_default():
    """The defect, pinned: default OFF, both siblings fetch the same page."""
    graph, _parent, (first, second) = _batch()
    io = FakeIO()
    action = VisitLeafAction()

    payload_one = await action.execute(graph, first.node_id, io)
    _mark_done(first, payload_one[ActionResultKey.URL.value])
    payload_two = await action.execute(graph, second.node_id, io)

    assert payload_one[ActionResultKey.URL.value] == payload_two[ActionResultKey.URL.value]
    assert io.fetches == [URL_ONE, URL_ONE]


@pytest.mark.asyncio
async def test_second_sibling_takes_the_next_url_when_enabled():
    graph, _parent, (first, second) = _batch()
    io = FakeIO()
    action = VisitLeafAction(settings={"visit_sibling_url_dedup": True})

    payload_one = await action.execute(graph, first.node_id, io)
    _mark_done(first, payload_one[ActionResultKey.URL.value])
    payload_two = await action.execute(graph, second.node_id, io)

    assert payload_one[ActionResultKey.URL.value] == URL_ONE
    assert payload_two[ActionResultKey.URL.value] == URL_TWO
    assert io.fetches == [URL_ONE, URL_TWO]


@pytest.mark.asyncio
async def test_concurrent_siblings_do_not_claim_the_same_url():
    """The claim is written before the fetch is awaited, so the parallel batch path -- the
    one the engine actually uses for a leaf fan-out -- splits too."""
    graph, _parent, children = _batch()
    io = FakeIO()
    action = VisitLeafAction(settings={"visit_sibling_url_dedup": True})

    payloads = await asyncio.gather(
        *[action.execute(graph, child.node_id, io) for child in children]
    )

    assert sorted(p[ActionResultKey.URL.value] for p in payloads) == [URL_ONE, URL_TWO]
    assert sorted(io.fetches) == [URL_ONE, URL_TWO]


@pytest.mark.asyncio
async def test_declared_url_is_never_dropped():
    """An explicit URL is this leaf's own instruction; a duplicate there is the planner's."""
    graph, _parent, (first, second) = _batch(
        visit_details={DetailKey.URL.value: URL_ONE, "link_idea": ""},
    )
    io = FakeIO()
    action = VisitLeafAction(settings={"visit_sibling_url_dedup": True})

    payload_one = await action.execute(graph, first.node_id, io)
    _mark_done(first, payload_one[ActionResultKey.URL.value])
    payload_two = await action.execute(graph, second.node_id, io)

    assert payload_one[ActionResultKey.URL.value] == URL_ONE
    assert payload_two[ActionResultKey.URL.value] == URL_ONE
    assert io.fetches == [URL_ONE, URL_ONE]


@pytest.mark.asyncio
async def test_no_alternative_url_fails_with_a_duplicate_diagnostic():
    """With nothing else to read, the redundant fetch is not paid for and the node says why."""
    graph, _parent, (first, second) = _batch(urls=(URL_ONE,))
    io = FakeIO()
    action = VisitLeafAction(settings={"visit_sibling_url_dedup": True})

    payload_one = await action.execute(graph, first.node_id, io)
    _mark_done(first, payload_one[ActionResultKey.URL.value])
    payload_two = await action.execute(graph, second.node_id, io)

    assert payload_two[ActionResultKey.SUCCESS.value] is False
    assert "already claimed by a sibling" in payload_two[ActionResultKey.ERROR.value]
    assert io.fetches == [URL_ONE]


@pytest.mark.asyncio
async def test_claims_are_recorded_on_the_parent():
    graph, parent, (first, _second) = _batch()
    io = FakeIO()
    action = VisitLeafAction(settings={"visit_sibling_url_dedup": True})

    await action.execute(graph, first.node_id, io)

    claims = parent.details.get(VISIT_URL_CLAIMS_KEY)
    assert claims == {URL_ONE: first.node_id}


@pytest.mark.asyncio
async def test_no_claims_recorded_when_disabled():
    graph, parent, (first, _second) = _batch()
    io = FakeIO()

    await VisitLeafAction().execute(graph, first.node_id, io)

    assert VISIT_URL_CLAIMS_KEY not in parent.details


@pytest.mark.parametrize(
    "left,right,same",
    [
        ("https://EN.Wikipedia.org/wiki/Chuck", "https://en.wikipedia.org/wiki/Chuck", True),
        ("https://en.wikipedia.org/wiki/Chuck/", "https://en.wikipedia.org/wiki/Chuck", True),
        ("https://en.wikipedia.org/wiki/Chuck#cast", "https://en.wikipedia.org/wiki/Chuck", True),
        ("https://hr.wikipedia.org/wiki/Chuck", "https://en.wikipedia.org/wiki/Chuck", False),
        ("https://example.com/p?a=1", "https://example.com/p?a=2", False),
    ],
)
def test_url_normalization_decides_same_page(left, right, same):
    normalize = VisitLeafAction._normalize_visit_url
    assert (normalize(left) == normalize(right)) is same


def test_url_normalization_rejects_non_urls():
    normalize = VisitLeafAction._normalize_visit_url
    assert normalize("") == ""
    assert normalize(None) == ""
    assert normalize("not a url") == ""
