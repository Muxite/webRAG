"""Retrieval relevance gates: the memory similarity floor and the finalize context ranking.

ASSUMPTION_AUDIT.md T3-1 and T3-3. A k-NN query returns k rows whether or not any of them
is near, and the finalize prompt pools four query batches and keeps the first
``final_chroma_results`` of them in the order the batches were issued. Both now have a
lever; both levers ship OFF.

The load-bearing assertions are the OFF ones: every measurement taken to date ran without
these, so the default path must stay byte-identical.
"""
from __future__ import annotations

import pytest

from agent.app.idea_finalize import _retrieve_final_chroma_context
from agent.app.idea_memory import MemoryManager
from agent.app.idea_policies.config import MemoryConfig


class _FakeCollection:
    def __init__(self, space: str):
        self.configuration_json = {"hnsw": {"space": space}}
        self.metadata = {"hnsw:space": space}


class _FakeChroma:
    def __init__(self, *, space: str = "cosine", distances=()):
        self.space = space
        self.distances = list(distances)

    async def get_or_create_collection(self, collection, metadata=None):
        return _FakeCollection(self.space)

    async def query_chroma(self, collection, query_texts, n_results=5, where=None):
        n = len(self.distances)
        return {
            "documents": [[f"doc{i}" for i in range(n)]],
            "metadatas": [[{"node_title": f"n{i}"} for i in range(n)]],
            "distances": [list(self.distances)],
            "ids": [[f"id{i}" for i in range(n)]],
        }


# --- the floor ------------------------------------------------------------------------


def test_floor_ships_off():
    """The default admits everything, which is what every past run measured."""
    assert MemoryConfig().retrieval_similarity_floor == 0.0
    assert MemoryManager(connector_chroma=None, namespace="x").similarity_floor == 0.0


@pytest.mark.asyncio
async def test_default_manager_admits_the_far_neighbour():
    chroma = _FakeChroma(distances=[0.05, 0.4, 0.95])
    manager = MemoryManager(connector_chroma=chroma, namespace="idea_dag:test")
    got = await manager.retrieve_relevant_memories("q", n_results=3)
    assert [m["id"] for m in got] == ["id0", "id1", "id2"]


@pytest.mark.asyncio
async def test_floor_drops_below_threshold_and_returns_fewer_than_n():
    chroma = _FakeChroma(distances=[0.05, 0.4, 0.95])
    manager = MemoryManager(
        connector_chroma=chroma, namespace="idea_dag:test", similarity_floor=0.5
    )
    got = await manager.retrieve_relevant_memories("q", n_results=3)
    # cosine space: similarity = 1 - d, so 0.95, 0.60, 0.05 -> the last one falls.
    assert [m["id"] for m in got] == ["id0", "id1"]


@pytest.mark.asyncio
async def test_floor_is_expressed_in_the_collections_own_space():
    """An l2 collection's 0.4 is cosine 0.8, not 0.6 -- the T1-1 hazard, not re-made here."""
    chroma = _FakeChroma(space="l2", distances=[0.4])
    manager = MemoryManager(
        connector_chroma=chroma, namespace="idea_dag:test", similarity_floor=0.7
    )
    got = await manager.retrieve_relevant_memories("q", n_results=1)
    assert [m["id"] for m in got] == ["id0"]
    assert got[0]["similarity"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_similarity_is_exposed_even_with_the_floor_off():
    chroma = _FakeChroma(distances=[0.25])
    manager = MemoryManager(connector_chroma=chroma, namespace="idea_dag:test")
    got = await manager.retrieve_relevant_memories("q", n_results=1)
    assert got[0]["similarity"] == pytest.approx(0.75)


# --- the finalize context ordering ----------------------------------------------------


class _StubMemory:
    """Returns a fixed pool per query text, so batch ARRIVAL order is controllable."""

    def __init__(self, per_query):
        self.per_query = per_query

    async def retrieve_relevant_memories(self, query, n_results=5, **kwargs):
        return list(self.per_query.get(query[:1000], []))

    def format_memories_for_llm(self, memories, max_chars=2000):
        return "|".join(m["id"] for m in memories)


class _StubGraph:
    def root_id(self):
        return "root"

    def iter_depth_first(self):
        return []


def _pool():
    return _StubMemory({
        # The mandate batch is issued FIRST and carries the worse similarities.
        "mandate": [
            {"id": "far_a", "content": "a", "similarity": 0.10, "metadata": {}},
            {"id": "far_b", "content": "b", "similarity": 0.20, "metadata": {}},
        ],
    })


@pytest.mark.asyncio
async def test_finalize_context_keeps_arrival_order_by_default():
    text = await _retrieve_final_chroma_context(
        memory_manager=_pool(), mandate="mandate", graph=_StubGraph(), n_results=2
    )
    assert text == "far_a|far_b"


@pytest.mark.asyncio
async def test_finalize_context_ranks_by_similarity_when_enabled():
    text = await _retrieve_final_chroma_context(
        memory_manager=_pool(), mandate="mandate", graph=_StubGraph(), n_results=2,
        rank_by_similarity=True,
    )
    assert text == "far_b|far_a"


@pytest.mark.asyncio
async def test_finalize_ranking_is_stable_for_equal_similarity():
    memory = _StubMemory({
        "mandate": [
            {"id": "first", "content": "a", "similarity": 0.5, "metadata": {}},
            {"id": "second", "content": "b", "similarity": 0.5, "metadata": {}},
        ],
    })
    text = await _retrieve_final_chroma_context(
        memory_manager=memory, mandate="mandate", graph=_StubGraph(), n_results=2,
        rank_by_similarity=True,
    )
    assert text == "first|second"


def test_finalize_ranking_ships_off():
    assert MemoryConfig().final_context_rank_by_similarity is False
