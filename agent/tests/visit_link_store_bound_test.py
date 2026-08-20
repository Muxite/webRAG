"""A visit must never hand ChromaDB a whole page's worth of links to embed.

ChromaDB's default embedding function runs client-side, synchronously, inside the
async client's ``add`` coroutine (~19ms/doc measured). A Wikipedia article yields
~990 links after chrome filtering, so the uncapped store blocked the event loop for
18s -- past the 15s ``chroma_op_timeout`` (which cannot fire while the loop is
blocked) and past the 20s visit watchdog, which is what killed task 054's visits
live. These pin both halves of the fix: the per-page cap, and the batching that
keeps any single embedding stall short.
"""

import asyncio

import pytest

from agent.app.idea_policies.actions import VisitLeafAction


class RecordingChroma:
    """Records every add call, batching like the real connector does."""

    PARALLEL_BATCH_SIZE = 50

    def __init__(self):
        self.batches = []

    async def add_to_chroma(self, collection, ids, metadatas, documents):
        self.batches.append(list(documents))
        return True

    async def add_to_chroma_parallel(self, collection, ids, metadatas, documents, batch_size=None):
        bs = batch_size or self.PARALLEL_BATCH_SIZE
        if len(documents) <= bs:
            return await self.add_to_chroma(collection, ids, metadatas, documents)
        ok = True
        for start in range(0, len(documents), bs):
            ok = await self.add_to_chroma(
                collection,
                ids[start:start + bs],
                metadatas[start:start + bs],
                documents[start:start + bs],
            ) and ok
        return ok


class FakeIO:
    def __init__(self, chroma):
        self.connector_chroma = chroma


def _page_links(count: int):
    links = [f"https://en.wikipedia.org/wiki/Page_{i}" for i in range(count)]
    contexts = {url: f"anchor text {i}" for i, url in enumerate(links)}
    return links, contexts


def test_store_caps_links_per_page_and_batches_them():
    action = VisitLeafAction({"visit_link_store_max": 100})
    chroma = RecordingChroma()
    links, contexts = _page_links(988)

    assert asyncio.run(
        action._store_links_in_chroma("https://en.wikipedia.org/wiki/Beloved", links, contexts, FakeIO(chroma))
    ) is True

    stored = [doc for batch in chroma.batches for doc in batch]
    assert len(stored) == 100
    # No single add embeds more than one batch, so the loop stall stays sub-second.
    assert max(len(batch) for batch in chroma.batches) <= RecordingChroma.PARALLEL_BATCH_SIZE


def test_store_keeps_a_deep_link_the_idea_names():
    """Truncating by document order would drop the target; ranking keeps it."""
    action = VisitLeafAction({"visit_link_store_max": 10})
    chroma = RecordingChroma()
    links, contexts = _page_links(200)
    target = "https://en.wikipedia.org/wiki/Garabit_viaduct"
    links.append(target)
    contexts[target] = "Garabit viaduct"

    asyncio.run(
        action._store_links_in_chroma(
            "https://en.wikipedia.org/wiki/Gustave_Eiffel",
            links,
            contexts,
            FakeIO(chroma),
            link_idea="the Garabit viaduct bridge",
        )
    )

    stored_meta_urls = [doc for batch in chroma.batches for doc in batch]
    assert len(stored_meta_urls) == 10
    assert any("Garabit" in doc for doc in stored_meta_urls)


def test_small_pages_are_stored_whole():
    action = VisitLeafAction({"visit_link_store_max": 100})
    chroma = RecordingChroma()
    links, contexts = _page_links(7)

    asyncio.run(action._store_links_in_chroma("https://example.com/a", links, contexts, FakeIO(chroma)))

    assert sum(len(batch) for batch in chroma.batches) == 7


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_disables_the_cap(limit):
    action = VisitLeafAction({"visit_link_store_max": limit})
    chroma = RecordingChroma()
    links, contexts = _page_links(120)

    asyncio.run(action._store_links_in_chroma("https://example.com/a", links, contexts, FakeIO(chroma)))

    assert sum(len(batch) for batch in chroma.batches) == 120
