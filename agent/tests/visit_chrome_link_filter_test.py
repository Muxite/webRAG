"""Site-chrome filtering of the visit URL pool (``visit_chrome_link_filter``, opt-in, default OFF).

The dead-URL recovery harvest already refuses page chrome, but that test is applied THERE only.
Every other pool a URL-less visit resolves from -- an ancestor's page links, a sibling's results,
the search hits collected into ``candidate_urls`` -- still offers donation appeals, create-account
forms and portal plumbing as selectable pages. Two things make them win rather than lose: a chrome
link carries the leaf's own words in a campaign/``returnto=`` parameter, so it scores at least as
well as real content, and it sits FIRST in a Wikipedia page's link order, so a score tie resolves
to it. 64 of 2134 executed sibling visits in the recorded corpus (3.0%, 35 runs) fetched one; the
traced case is task 078, where two sibling leaves asking for Chichagof and Flores both read
``donate.wikimedia.org`` and the merge took a donation appeal as island evidence.

These tests pin: ON drops chrome from the parent-link, sibling-result and resolved-candidate pools,
a DECLARED chrome URL is still honoured (the planner's own instruction), and OFF is unchanged.
"""

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.actions import VisitLeafAction
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


DONATE_URL = (
    "https://donate.wikimedia.org/?wmf_source=donate&wmf_medium=sidebar"
    "&wmf_campaign=en.wikipedia.org&uselang=en"
)
CREATE_ACCOUNT_URL = (
    "https://en.wikipedia.org/w/index.php?title=Special%3ACreateAccount"
    "&returnto=Chichagof+Island"
)
REAL_URL = "https://en.wikipedia.org/wiki/Chichagof_Island"
ON = {"visit_chrome_link_filter": True}


class FakeIO:
    def __init__(self):
        self.fetches = []
        self.telemetry = None
        self.connector_chroma = None

    async def fetch_url(self, url: str, retries: int = 3, timeout_seconds=None) -> str:
        self.fetches.append(url)
        return "<html><body><h1>Chichagof Island</h1><p>Area 5,388 km2.</p></body></html>"

    async def visit(self, url: str, timeout_seconds=None) -> str:
        self.fetches.append(url)
        return "page content"

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return None


def _visit_leaf(graph: IdeaDag, parent_id: str, **details) -> str:
    node = graph.add_child(
        parent_id,
        "Visit Chichagof Island (Alaska, USA) Wikipedia page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "link_count": 1,
            **details,
        },
    )
    return node.node_id


def _ancestor_visit(graph: IdeaDag, links) -> str:
    """A completed hop whose page links chrome FIRST, exactly as Wikipedia serves it."""
    hop = graph.add_child(
        graph.root_id(),
        "Visit Bangka Island Wikipedia page",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    hop.status = IdeaNodeStatus.DONE
    hop.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://en.wikipedia.org/wiki/Bangka_Island",
        "links": list(links),
        "link_contexts": {},
    }
    return hop.node_id


@pytest.mark.asyncio
async def test_ancestor_link_pool_skips_chrome_for_the_real_page():
    graph = IdeaDag(root_title="root")
    hop_id = _ancestor_visit(graph, [DONATE_URL, CREATE_ACCOUNT_URL, REAL_URL])
    node_id = _visit_leaf(graph, hop_id)

    result = await VisitLeafAction(settings=ON).execute(graph, node_id, FakeIO())

    assert result[ActionResultKey.URL.value] == REAL_URL


@pytest.mark.asyncio
async def test_flag_off_still_reads_the_donation_page():
    """The recorded behaviour, kept as the OFF baseline: chrome sits first and wins the tie."""
    graph = IdeaDag(root_title="root")
    hop_id = _ancestor_visit(graph, [DONATE_URL, CREATE_ACCOUNT_URL, REAL_URL])
    node_id = _visit_leaf(graph, hop_id)

    result = await VisitLeafAction().execute(graph, node_id, FakeIO())

    assert result[ActionResultKey.URL.value] == DONATE_URL


@pytest.mark.asyncio
async def test_sibling_result_pool_skips_chrome():
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(
        graph.root_id(), "parent", details={DetailKey.ACTION.value: IdeaActionType.THINK.value}
    )
    sibling = graph.add_child(
        parent.node_id, "sibling visit",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    sibling.status = IdeaNodeStatus.DONE
    sibling.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://en.wikipedia.org/wiki/Bangka_Island",
        "links": [DONATE_URL, REAL_URL],
        "link_contexts": {},
    }
    node_id = _visit_leaf(graph, parent.node_id)

    result = await VisitLeafAction(settings=ON).execute(graph, node_id, FakeIO())

    assert result[ActionResultKey.URL.value] == REAL_URL


@pytest.mark.asyncio
async def test_search_candidate_pool_skips_chrome():
    """The dead-declared-URL cascade's own pool: a chrome hit must not be the recovery target."""
    graph = IdeaDag(root_title="root")
    search = graph.add_child(
        graph.root_id(), "search", details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value}
    )
    search.status = IdeaNodeStatus.DONE
    search.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": "hit", "url": u} for u in (DONATE_URL, REAL_URL)],
    }
    node_id = _visit_leaf(graph, search.node_id, link_idea="Chichagof Island page", link_count=2)

    io = FakeIO()
    await VisitLeafAction(settings=ON).execute(graph, node_id, io)

    assert DONATE_URL not in io.fetches
    assert REAL_URL in io.fetches


@pytest.mark.asyncio
async def test_a_declared_chrome_url_is_still_honoured():
    """Same principle as the sibling dedup: an explicit URL is the planner's call, not the pool's."""
    graph = IdeaDag(root_title="root")
    node_id = _visit_leaf(graph, graph.root_id(), optional_url=DONATE_URL)

    io = FakeIO()
    result = await VisitLeafAction(settings=ON).execute(graph, node_id, io)

    assert result[ActionResultKey.URL.value] == DONATE_URL
    assert io.fetches == [DONATE_URL]


@pytest.mark.asyncio
async def test_an_all_chrome_pool_fails_rather_than_reading_chrome():
    graph = IdeaDag(root_title="root")
    hop_id = _ancestor_visit(graph, [DONATE_URL, CREATE_ACCOUNT_URL])
    node_id = _visit_leaf(graph, hop_id)

    io = FakeIO()
    result = await VisitLeafAction(settings=ON).execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is False
    assert io.fetches == []
