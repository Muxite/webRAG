"""Two bugs in the visit-link-selection path, both traced to the same root cause: the
model picked the right URL and the engine still fetched the wrong page.

1. `_query_links_from_chroma`/`_store_links_in_chroma` scoped link-index collections
   globally by URL hash only, with no run/task scoping -- a page's outbound-link index
   persisted forever and was visible to any later unrelated run whose ``link_idea``
   happened to embed close to it. Fixed by folding ``io.collection_name`` (the same
   run-scoped identifier this repo already uses for memory isolation, e.g.
   ``idea_test_{test_id}_{run_stamp}``) into the ``links_*`` collection name.

2. `_select_links_with_llm` silently discarded a well-formed model answer that wasn't
   byte-for-byte in ``candidate_urls`` and substituted ``candidate_urls[0]`` instead,
   with no warning -- total loss of a correct answer when the candidate pool itself was
   off-topic (see bug 1). Fixed to prefer a well-formed absolute-URL answer over the
   silent first-candidate fallback, and to always log a warning on this path.
"""

import asyncio
import json
import logging

import pytest

from agent.app.idea_policies.actions import VisitLeafAction


class FakeIO:
    """Minimal AgentIO stand-in carrying a run-scoped ``collection_name``."""

    def __init__(self, chroma, collection_name="agent_memory", llm_response=None):
        self.connector_chroma = chroma
        self.collection_name = collection_name
        self._llm_response = llm_response

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return self._llm_response


class RecordingChroma:
    """A stand-in Chroma connector that actually stores per-collection docs, so a
    query can be answered from real per-collection state instead of a single global
    dict -- this is what would hide the cross-run bleed bug if collapsed."""

    def __init__(self):
        self.collections = {}  # name -> list[(id, metadata, document)]

    async def add_to_chroma(self, collection, ids, metadatas, documents):
        bucket = self.collections.setdefault(collection, [])
        bucket.extend(zip(ids, metadatas, documents))
        return True

    add_to_chroma_parallel = add_to_chroma

    async def list_collections(self):
        return list(self.collections.keys())

    async def query_chroma(self, collection, query_texts, n_results=3, where=None):
        bucket = self.collections.get(collection, [])
        metadatas = [meta for _, meta, _ in bucket][:n_results]
        # Distance doesn't matter for these tests; every match is an equally-close hit.
        distances = [0.1] * len(metadatas)
        return {"metadatas": [metadatas], "distances": [distances]}


def _links(urls):
    return urls, {u: "" for u in urls}


# --- Run-scoped Chroma isolation -------------------------------------------------


def test_link_index_write_is_scoped_to_the_run_collection_namespace():
    action = VisitLeafAction({})
    chroma = RecordingChroma()
    io_a = FakeIO(chroma, collection_name="idea_test_150_run1")

    links, contexts = _links(["https://en.wikipedia.org/wiki/Brooklyn_Bridge"])
    asyncio.run(action._store_links_in_chroma("https://example.com/a", links, contexts, io_a))

    stored_collections = list(chroma.collections.keys())
    assert len(stored_collections) == 1
    assert stored_collections[0].startswith("links_idea_test_150_run1_")


def test_query_under_run_a_scope_never_returns_run_bs_link_entries():
    action = VisitLeafAction({})
    chroma = RecordingChroma()

    io_a = FakeIO(chroma, collection_name="idea_test_150_run1")
    io_b = FakeIO(chroma, collection_name="idea_test_099_run7")

    # Run B indexed a completely unrelated page (Brooklyn Bridge) under its own scope.
    b_links, b_contexts = _links(["https://en.wikipedia.org/wiki/Brooklyn_Bridge"])
    asyncio.run(action._store_links_in_chroma("https://example.com/b-source", b_links, b_contexts, io_b))

    # Run A indexed its own, different page (Hardanger Bridge) under its own scope.
    a_links, a_contexts = _links(["https://en.wikipedia.org/wiki/Hardanger_Bridge"])
    asyncio.run(action._store_links_in_chroma("https://example.com/a-source", a_links, a_contexts, io_a))

    # A query issued under run A's scope must only ever see run A's own link index.
    results = asyncio.run(action._query_links_from_chroma("the bridge", io_a, top_k=10))

    assert results == ["https://en.wikipedia.org/wiki/Hardanger_Bridge"]
    assert "https://en.wikipedia.org/wiki/Brooklyn_Bridge" not in results


def test_query_finds_nothing_when_only_a_different_runs_index_exists():
    action = VisitLeafAction({})
    chroma = RecordingChroma()

    io_b = FakeIO(chroma, collection_name="idea_test_099_run7")
    io_a = FakeIO(chroma, collection_name="idea_test_150_run1")

    b_links, b_contexts = _links(["https://en.wikipedia.org/wiki/Brooklyn_Bridge"])
    asyncio.run(action._store_links_in_chroma("https://example.com/b-source", b_links, b_contexts, io_b))

    # Run A never stored anything -- its query must come back empty, not fall through
    # to run B's collection.
    results = asyncio.run(action._query_links_from_chroma("the bridge", io_a, top_k=10))
    assert results == []


# --- Fallback-preference fix ------------------------------------------------------


CANDIDATES = [
    "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
    "https://en.wikipedia.org/wiki/Brooklyn_Bridge_Park",
    "https://en.wikipedia.org/wiki/Brooklyn_Bridge_history",
]


def test_well_formed_off_list_answer_is_used_not_discarded(caplog):
    """The model answered correctly but its answer isn't in the (poisoned) candidate
    pool -- the fix must use the model's answer, not silently substitute candidates[0]."""
    action = VisitLeafAction({})
    off_list_url = "https://en.wikipedia.org/wiki/Hardanger_Bridge"
    io = FakeIO(
        chroma=RecordingChroma(),
        llm_response=json.dumps({"selected": [off_list_url]}),
    )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(action._select_links_with_llm("the bridge", CANDIDATES, 1, io))

    assert result == [off_list_url]
    assert result[0] != CANDIDATES[0]
    assert any("not in the" in r.message and "candidate pool" in r.message for r in caplog.records)


def test_malformed_answer_still_falls_back_to_first_candidates(caplog):
    action = VisitLeafAction({})
    io = FakeIO(
        chroma=RecordingChroma(),
        llm_response=json.dumps({"selected": ["not-a-url", "also garbage"]}),
    )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(action._select_links_with_llm("the bridge", CANDIDATES, 2, io))

    assert result == CANDIDATES[:2]
    assert any("no usable URL" in r.message for r in caplog.records)


def test_empty_selected_still_falls_back_to_first_candidates(caplog):
    action = VisitLeafAction({})
    io = FakeIO(
        chroma=RecordingChroma(),
        llm_response=json.dumps({"selected": []}),
    )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(action._select_links_with_llm("the bridge", CANDIDATES, 2, io))

    assert result == CANDIDATES[:2]
    assert any("no usable URL" in r.message for r in caplog.records)


def test_in_pool_answer_is_unaffected_by_the_fallback_fix():
    """When the model's answer IS a real candidate, behavior is unchanged: no fallback,
    no off-list warning, the model's own in-pool pick is returned."""
    action = VisitLeafAction({})
    io = FakeIO(
        chroma=RecordingChroma(),
        llm_response=json.dumps({"selected": [CANDIDATES[2], CANDIDATES[0]]}),
    )

    result = asyncio.run(action._select_links_with_llm("the bridge history", CANDIDATES, 2, io))

    assert result == [CANDIDATES[2], CANDIDATES[0]]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://en.wikipedia.org/wiki/Hardanger_Bridge", True),
        ("http://example.com/page", True),
        ("not-a-url", False),
        ("", False),
        ("ftp://example.com/file", False),
        ("https://", False),
    ],
)
def test_looks_like_url_sanity_check(value, expected):
    assert VisitLeafAction._looks_like_url(value) is expected
