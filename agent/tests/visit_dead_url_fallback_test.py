"""Dead declared-URL fallback (``visit_dead_url_fallback_enabled``, kill-switch, default ON).

``io.fetch_url`` RAISES on a permanent HTTP status (404/403). That exception used to escape
``VisitLeafAction.execute`` entirely, so the URL-recovery cascade -- the same cascade a
declared URL that merely RETURNS a failure already falls through to -- never ran. A planner
guessed Wikipedia title is the usual cause (task 135's low reps died on invented titles such
as ``.../Cincinnati_and_Northern_Kentucky_Over-the-Rhine_Suspension_Bridge``), and the
cascade routinely holds the real page: the previous hop's own page links to it.

These tests pin: ON recovers through the cascade (parent search hits and the stored link
index), the dead URL is never re-picked, and when the cascade has nothing NEW the original
404 failure surface is preserved exactly as with the flag OFF.
"""

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.actions import VisitLeafAction
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


DEAD_URL = "https://en.wikipedia.org/wiki/Cincinnati_and_Northern_Kentucky_Suspension_Bridge"
REAL_URL = "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge"


class FakeChroma:
    """Stands in for the stored link index the previous hop's page populated."""

    def __init__(self, urls=()):
        self._urls = list(urls)

    async def add_to_chroma(self, collection, ids, metadatas, documents):
        return True

    async def list_collections(self):
        return ["links_run"] if self._urls else []

    async def query_chroma(self, collection, query_texts, n_results=10):
        return {
            "metadatas": [[{"url": u} for u in self._urls]],
            "distances": [[0.1 * (i + 1) for i in range(len(self._urls))]],
        }


class FakeIO:
    def __init__(self, link_urls=()):
        self.fetches = []
        self.telemetry = None
        self.connector_chroma = FakeChroma(link_urls)

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

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return None


def _node_with_dead_url(graph: IdeaDag, parent_id: str) -> str:
    node = graph.add_child(
        parent_id,
        "Read the main span of the Cincinnati suspension bridge",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            "optional_url": DEAD_URL,
            "link_count": 1,
        },
    )
    return node.node_id


def _graph_with_search_parent(results=(REAL_URL,)):
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(
        graph.root_id(), "search", details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    parent.status = IdeaNodeStatus.DONE
    parent.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": "hit", "url": u} for u in results],
    }
    return graph, parent.node_id


@pytest.mark.asyncio
async def test_dead_declared_url_recovers_through_parent_search_hits():
    graph, parent_id = _graph_with_search_parent()
    node_id = _node_with_dead_url(graph, parent_id)
    io = FakeIO()

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is True
    assert result[ActionResultKey.URL.value] == REAL_URL
    assert io.fetches == [DEAD_URL, REAL_URL]


@pytest.mark.asyncio
async def test_dead_declared_url_recovers_through_the_stored_link_index():
    """Task 135's shape: no search node ran, but the previous hop's page linked the real one."""
    graph = IdeaDag(root_title="root")
    node_id = _node_with_dead_url(graph, graph.root_id())
    io = FakeIO(link_urls=[REAL_URL])

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is True
    assert result[ActionResultKey.URL.value] == REAL_URL


@pytest.mark.asyncio
async def test_dead_url_is_never_re_picked_from_the_candidate_pool():
    graph, parent_id = _graph_with_search_parent(results=(DEAD_URL, REAL_URL))
    node_id = _node_with_dead_url(graph, parent_id)
    io = FakeIO()

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.URL.value] == REAL_URL
    assert io.fetches.count(DEAD_URL) == 1


@pytest.mark.asyncio
async def test_dead_declared_url_recovers_from_the_previous_hop_link_menu():
    """The recorded task-135 shape: hop 1 declared its own URL, so nothing indexed its links.

    The previous hop's page menu lives only in its stored action result; the real page is in it.
    """
    graph = IdeaDag(root_title="root")
    hop_one = graph.add_child(
        graph.root_id(),
        "Identify the engineer of the Brooklyn Bridge",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    hop_one.status = IdeaNodeStatus.DONE
    hop_one.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
        "links_full": [
            "https://donate.wikimedia.org/?wmf_source=donate",
            "https://en.wikipedia.org/wiki/Special:MyContributions",
            REAL_URL,
        ],
        "link_contexts": {REAL_URL: "John A. Roebling Suspension Bridge"},
    }
    node_id = _node_with_dead_url(graph, hop_one.node_id)
    io = FakeIO()

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is True
    assert result[ActionResultKey.URL.value] == REAL_URL


@pytest.mark.asyncio
async def test_page_chrome_links_are_not_harvested():
    """Only the chrome/portal links exist -> nothing names the leaf's goal -> no false recovery."""
    graph = IdeaDag(root_title="root")
    hop_one = graph.add_child(
        graph.root_id(),
        "Identify the engineer of the Brooklyn Bridge",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    hop_one.status = IdeaNodeStatus.DONE
    hop_one.details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
        "links_full": [
            "https://donate.wikimedia.org/?wmf_source=donate",
            "https://en.wikipedia.org/wiki/Special:MyContributions",
            # names the leaf's own words in a returnto= parameter, so a lexical pool would
            # otherwise rank this account page above real content
            "https://en.wikipedia.org/w/index.php?title=Special%3ACreateAccount"
            "&returnto=Cincinnati+suspension+bridge+main+span",
        ],
        "link_contexts": {},
    }
    node_id = _node_with_dead_url(graph, hop_one.node_id)
    io = FakeIO()

    result = await VisitLeafAction().execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is False
    assert result[ActionResultKey.HTTP_STATUS.value] == 404
    assert io.fetches == [DEAD_URL]


@pytest.mark.asyncio
async def test_failure_surface_preserved_when_nothing_is_recoverable():
    """Nothing left to try -> the original 404 result, identical to the flag-OFF abort."""
    graph = IdeaDag(root_title="root")
    node_id = _node_with_dead_url(graph, graph.root_id())

    on = await VisitLeafAction().execute(graph, node_id, FakeIO())
    off = await VisitLeafAction(
        settings={"visit_dead_url_fallback_enabled": False}
    ).execute(graph, node_id, FakeIO())

    for payload in (on, off):
        assert payload[ActionResultKey.SUCCESS.value] is False
        assert payload[ActionResultKey.HTTP_STATUS.value] == 404
        assert payload[ActionResultKey.RETRYABLE.value] is False
    assert on[ActionResultKey.ERROR.value] == off[ActionResultKey.ERROR.value]


@pytest.mark.asyncio
async def test_flag_off_aborts_without_touching_the_cascade():
    graph, parent_id = _graph_with_search_parent()
    node_id = _node_with_dead_url(graph, parent_id)
    io = FakeIO()

    result = await VisitLeafAction(
        settings={"visit_dead_url_fallback_enabled": False}
    ).execute(graph, node_id, io)

    assert result[ActionResultKey.SUCCESS.value] is False
    assert io.fetches == [DEAD_URL]
