"""Dedup similarity must be a real cosine similarity, whatever space the collection uses.

ASSUMPTION_AUDIT.md T1-1: ``got_operations.is_duplicate_thought`` used a bare
``1.0 - distance`` against the ``mem_*`` collections, which were created with no metadata
and therefore in chroma's default ``l2`` space (the SQUARED euclidean distance). For the
unit-norm embeddings chroma's bundled MiniLM produces ``d = 2 - 2cos``, so the computed
value was ``2s - 1``: the shipped 0.85 threshold really demanded cosine 0.925.

Pinned here:

* ``MemoryManager`` REQUESTS the cosine space at creation, and reports the space a
  pre-existing collection actually has (per-mandate collections outlive a run);
* the dedup comparison converts through ``similarity_from_distance``, so an l2 distance of
  0.2 (cosine 0.9) is a duplicate at the 0.85 threshold where the old math (0.8) said no.
"""
from __future__ import annotations

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_memory import MEMORY_COLLECTION_METADATA, MemoryManager


class _FakeCollection:
    def __init__(self, space: str):
        self.configuration_json = {"hnsw": {"space": space}}
        self.metadata = {"hnsw:space": space}


class _FakeChroma:
    """``ConnectorChroma``-shaped stand-in: only what MemoryManager calls."""

    def __init__(self, *, space: str = "cosine", distances=(), no_metadata_kwarg: bool = False):
        self.space = space
        self.distances = list(distances)
        self.no_metadata_kwarg = no_metadata_kwarg
        self.created = []
        self.queries = []

    async def get_or_create_collection(self, collection, metadata=None):
        if self.no_metadata_kwarg and metadata is not None:
            raise TypeError("unexpected keyword argument 'metadata'")
        self.created.append((collection, metadata))
        return _FakeCollection(self.space)

    async def query_chroma(self, collection, query_texts, n_results=5, where=None):
        self.queries.append((collection, query_texts, n_results, where))
        n = len(self.distances)
        return {
            "documents": [[f"doc{i}" for i in range(n)]],
            "metadatas": [[{"node_id": f"n{i}"} for i in range(n)]],
            "distances": [list(self.distances)],
            "ids": [[f"id{i}" for i in range(n)]],
        }


def _manager(chroma) -> MemoryManager:
    return MemoryManager(connector_chroma=chroma, namespace="idea_dag:test")


# --- collection creation / space reporting ------------------------------------------


@pytest.mark.asyncio
async def test_memory_collection_is_created_with_the_cosine_space():
    chroma = _FakeChroma()
    manager = _manager(chroma)
    await manager.ensure_collection()
    assert chroma.created[0][1] == MEMORY_COLLECTION_METADATA == {"hnsw:space": "cosine"}
    assert manager.distance_space == "cosine"


@pytest.mark.asyncio
async def test_pre_existing_l2_collection_is_reported_as_l2():
    # mem_* collections are keyed on mandate TEXT, so one created before the cosine
    # metadata landed survives across runs and must still convert correctly.
    manager = _manager(_FakeChroma(space="l2"))
    await manager.ensure_collection()
    assert manager.distance_space == "l2"


@pytest.mark.asyncio
async def test_connector_without_metadata_kwarg_still_yields_a_space():
    manager = _manager(_FakeChroma(space="l2", no_metadata_kwarg=True))
    await manager.ensure_collection()
    assert manager.distance_space == "l2"


@pytest.mark.asyncio
async def test_space_defaults_to_cosine_without_a_connector():
    manager = MemoryManager(connector_chroma=None, namespace="temp")
    assert await manager.ensure_collection() is None
    assert manager.distance_space == "cosine"


@pytest.mark.asyncio
async def test_retrieve_ensures_the_collection_before_querying():
    chroma = _FakeChroma(space="l2", distances=[0.2])
    manager = _manager(chroma)
    await manager.retrieve_relevant_memories(query="q", n_results=3)
    assert chroma.created and chroma.created[0][1] == {"hnsw:space": "cosine"}
    assert manager.distance_space == "l2"


# --- the dedup comparison -------------------------------------------------------------


def _ops(manager) -> GoTOperations:
    return GoTOperations(
        settings={"got_dedup_enabled": True, "got_adaptive_policies": False},
        io=None,
        memory_manager=manager,
    )


@pytest.mark.asyncio
async def test_l2_distance_converts_to_cosine_before_thresholding():
    # d = 0.2 in l2 -> cos 0.9 >= 0.85: a duplicate. The old ``1 - d`` gave 0.8: not one.
    chroma = _FakeChroma(space="l2", distances=[0.2])
    manager = _manager(chroma)
    is_dup, node_id = await _ops(manager).is_duplicate_thought("t", "g", IdeaDag("m"))
    assert (is_dup, node_id) == (True, "n0")


@pytest.mark.asyncio
async def test_l2_distance_below_the_threshold_is_not_a_duplicate():
    # d = 0.5 in l2 -> cos 0.75 < 0.85.
    manager = _manager(_FakeChroma(space="l2", distances=[0.5]))
    is_dup, node_id = await _ops(manager).is_duplicate_thought("t", "g", IdeaDag("m"))
    assert (is_dup, node_id) == (False, None)


@pytest.mark.asyncio
async def test_cosine_collection_uses_the_raw_distance():
    # Same numeric distance, cosine space: 1 - 0.2 = 0.8 < 0.85, so NOT a duplicate.
    manager = _manager(_FakeChroma(space="cosine", distances=[0.2]))
    is_dup, _ = await _ops(manager).is_duplicate_thought("t", "g", IdeaDag("m"))
    assert is_dup is False
    manager = _manager(_FakeChroma(space="cosine", distances=[0.1]))
    is_dup, node_id = await _ops(manager).is_duplicate_thought("t", "g", IdeaDag("m"))
    assert (is_dup, node_id) == (True, "n0")
