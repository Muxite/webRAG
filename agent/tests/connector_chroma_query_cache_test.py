"""Cache-behavior tests for ``ConnectorChroma``'s in-process query memoization.

Chroma embeds query text itself (there is no separate embedding call site in this repo), so
memoizing ``query_chroma`` is what memoizes the embedding work. The contract these lock in:

  * the same (collection, query texts, n_results, where) is answered from cache;
  * ANY of those changing is a miss, and so is a different embedder;
  * a write (add/delete) to that collection, or losing the client, invalidates it — a hit must
    never be an answer the server would no longer give;
  * a hit is still recorded in telemetry (flagged ``cached``), so a trace still shows the ask.

All offline: the fake collection returns a fresh, distinguishable payload per call.
"""
import asyncio

import pytest

from shared.connector_config import ConnectorConfig
from agent.app.connector_chroma import ConnectorChroma


class _CountingCollection:
    """Async collection whose query answer changes on every call and after every write."""

    def __init__(self):
        self.query_calls = 0
        self.generation = 0
        self.raises = False

    async def add(self, ids, metadatas, documents):
        if self.raises:
            raise RuntimeError("simulated chroma error")
        self.generation += 1

    async def delete(self, ids):
        self.generation += 1

    async def query(self, query_texts, n_results, where=None):
        self.query_calls += 1
        tag = f"gen{self.generation}-call{self.query_calls}-{query_texts[0]}-{n_results}-{where}"
        return {"documents": [[tag]], "metadatas": [[{}]], "distances": [[0.1]], "ids": [["1"]]}


class _FakeClient:
    def __init__(self, collection: _CountingCollection):
        self._collection = collection
        self.deleted: list = []

    async def heartbeat(self):
        return 1

    async def get_or_create_collection(self, name, metadata=None, embedding_function=None):
        return self._collection

    async def delete_collection(self, name):
        self.deleted.append(name)


def _connector(collection: _CountingCollection, **cfg_overrides) -> ConnectorChroma:
    cfg = ConnectorConfig()
    cfg.chroma_op_timeout = 5.0
    cfg.chroma_query_cache = True
    cfg.chroma_query_cache_max = 256
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    c = ConnectorChroma(cfg)
    c._chroma = _FakeClient(collection)
    c.chroma_api_ready = True
    return c


@pytest.mark.asyncio
async def test_identical_query_is_a_cache_hit():
    """Same inputs → one round-trip, identical answer, hit counted."""
    coll = _CountingCollection()
    c = _connector(coll)
    first = await c.query_chroma("mem_x", ["who built it"], n_results=3)
    second = await c.query_chroma("mem_x", ["who built it"], n_results=3)
    assert coll.query_calls == 1
    assert second == first
    assert c.query_cache_stats() == {"hits": 1, "misses": 1, "entries": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_texts": ["a different question"], "n_results": 3},
        {"query_texts": ["who built it"], "n_results": 5},
        {"query_texts": ["who built it"], "n_results": 3, "where": {"memory_type": "observation"}},
        {"query_texts": ["who built it", "and when"], "n_results": 3},
    ],
    ids=["text", "n_results", "where", "extra_query_text"],
)
async def test_any_input_change_is_a_miss(kwargs):
    """Every argument that can change the answer is part of the key."""
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["who built it"], n_results=3)
    await c.query_chroma("mem_x", **kwargs)
    assert coll.query_calls == 2
    assert c.query_cache_stats()["hits"] == 0


@pytest.mark.asyncio
async def test_other_collection_is_a_miss():
    """Two collections are two vector stores; the name is part of the key."""
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    await c.query_chroma("mem_y", ["q"], n_results=3)
    assert coll.query_calls == 2


@pytest.mark.asyncio
async def test_embedder_identity_is_part_of_the_key():
    """A different embedding model means different vectors — never reuse across them."""
    coll = _CountingCollection()
    c = _connector(coll)
    cpu_key = c._query_cache_key("mem_x", ["q"], 3, None)
    c.config.chroma_embed_device = "cuda"
    c.config.chroma_embed_model = "all-mpnet-base-v2"
    assert c._query_cache_key("mem_x", ["q"], 3, None) != cpu_key


@pytest.mark.asyncio
async def test_add_invalidates_the_cached_answer():
    """A write changes what the server would answer, so the cached answer must be dropped."""
    coll = _CountingCollection()
    c = _connector(coll)
    before = await c.query_chroma("mem_x", ["q"], n_results=3)
    await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["a new memory"])
    after = await c.query_chroma("mem_x", ["q"], n_results=3)
    assert coll.query_calls == 2
    assert after != before  # the fake's answer moved with the write; the cache did not hide it


@pytest.mark.asyncio
async def test_add_to_another_collection_keeps_the_cache():
    """Invalidation is per collection: an unrelated write must not throw away good work."""
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    await c.add_to_chroma("mem_other", ["1"], [{"k": "v"}], ["unrelated"])
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert coll.query_calls == 1
    assert c.query_cache_stats()["hits"] == 1


@pytest.mark.asyncio
async def test_failed_add_also_invalidates():
    """A timed-out/failed add may still have landed rows, so cached answers are suspect."""
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    coll.raises = True
    assert await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"]) is False
    coll.raises = False
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert coll.query_calls == 2


@pytest.mark.asyncio
async def test_delete_invalidates():
    """Deleting documents changes the answer set exactly like adding does."""
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert await c.delete_from_chroma("mem_x", ["1"]) is True
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert coll.query_calls == 2


@pytest.mark.asyncio
async def test_delete_collection_invalidates():
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert await c.delete_collection("mem_x") is True
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert coll.query_calls == 2


@pytest.mark.asyncio
async def test_client_teardown_clears_everything():
    """Once the connector goes blind, it cannot know what changed while it was gone."""
    coll = _CountingCollection()
    c = _connector(coll)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert c.query_cache_stats()["entries"] == 1
    coll.raises = True
    for _ in range(ConnectorChroma.FAILURE_TEARDOWN_THRESHOLD):
        await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])
    assert c.chroma_api_ready is False
    assert c.query_cache_stats()["entries"] == 0


@pytest.mark.asyncio
async def test_disabled_cache_always_round_trips():
    """CHROMA_QUERY_CACHE=false restores the un-memoized behavior exactly."""
    coll = _CountingCollection()
    c = _connector(coll, chroma_query_cache=False)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    assert coll.query_calls == 2
    assert c.query_cache_stats() == {"hits": 0, "misses": 0, "entries": 0}


@pytest.mark.asyncio
async def test_hit_returns_a_copy_the_caller_cannot_poison():
    """Callers unpack the result dict; mutating it must not corrupt later hits."""
    coll = _CountingCollection()
    c = _connector(coll)
    first = await c.query_chroma("mem_x", ["q"], n_results=3)
    first["documents"][0][0] = "MUTATED"
    second = await c.query_chroma("mem_x", ["q"], n_results=3)
    assert second["documents"][0][0] != "MUTATED"


@pytest.mark.asyncio
async def test_cache_is_bounded_lru():
    """The cache is bounded: the least recently used entry is evicted past the cap."""
    coll = _CountingCollection()
    c = _connector(coll, chroma_query_cache_max=2)
    await c.query_chroma("mem_x", ["q1"], n_results=3)
    await c.query_chroma("mem_x", ["q2"], n_results=3)
    await c.query_chroma("mem_x", ["q1"], n_results=3)  # hit, refreshes q1
    await c.query_chroma("mem_x", ["q3"], n_results=3)  # evicts q2
    assert c.query_cache_stats()["entries"] == 2
    await c.query_chroma("mem_x", ["q1"], n_results=3)  # still cached
    assert c.query_cache_stats()["hits"] == 2
    await c.query_chroma("mem_x", ["q2"], n_results=3)  # evicted -> round-trip
    assert coll.query_calls == 4


@pytest.mark.asyncio
async def test_failed_query_is_not_cached():
    """Only a real answer is cached; a failure must not be replayed as a result."""
    coll = _CountingCollection()
    c = _connector(coll)

    async def _boom(query_texts, n_results, where=None):
        raise RuntimeError("simulated chroma error")

    coll.query = _boom
    assert await c.query_chroma("mem_x", ["q"], n_results=3) is None
    assert c.query_cache_stats()["entries"] == 0


class _RecordingTelemetry:
    def __init__(self):
        self.events = []

    def record_event(self, event, payload):
        self.events.append((event, payload))

    def record_timing(self, **kwargs):
        self.events.append(("timing", kwargs))


@pytest.mark.asyncio
async def test_cache_hit_still_records_telemetry():
    """A hit skips the round-trip, not the trace: the ask is still visible, flagged cached."""
    coll = _CountingCollection()
    c = _connector(coll)
    telemetry = _RecordingTelemetry()
    c.set_telemetry(telemetry)
    c.set_full_capture(True)
    await c.query_chroma("mem_x", ["q"], n_results=3)
    telemetry.events.clear()
    await c.query_chroma("mem_x", ["q"], n_results=3)
    io_events = [p for name, p in telemetry.events if name == "connector_io"]
    assert len(io_events) == 2  # in + out, exactly as an uncached query records
    assert all(e["payload"].get("cached") is True for e in io_events)


@pytest.mark.asyncio
async def test_memory_manager_repeat_probe_is_served_from_cache():
    """The engine-level shape this exists for: the same dedup/retrieval probe, twice.

    ``write_memory`` in between must break the reuse — a candidate written as a thought has to
    be visible to the next duplicate check.
    """
    from agent.app.idea_memory import MemoryManager

    coll = _CountingCollection()
    c = _connector(coll)
    mm = MemoryManager(connector_chroma=c, namespace="idea_dag:deadbeef")
    query = "Find the arch span Find the arch span of the viaduct"
    await mm.retrieve_relevant_memories(query=query, n_results=5, memory_type="internal_thought")
    await mm.retrieve_relevant_memories(query=query, n_results=5, memory_type="internal_thought")
    assert coll.query_calls == 1
    await mm.write_memory(
        content="a newly explored thought", node_id="n1", node_title="Explore",
        memory_type="internal_thought",
    )
    await mm.retrieve_relevant_memories(query=query, n_results=5, memory_type="internal_thought")
    assert coll.query_calls == 2


@pytest.mark.asyncio
async def test_concurrent_identical_queries_agree():
    """Two in-flight identical queries both return a usable answer (one may pre-date the fill)."""
    coll = _CountingCollection()
    c = _connector(coll)
    results = await asyncio.gather(
        c.query_chroma("mem_x", ["q"], n_results=3),
        c.query_chroma("mem_x", ["q"], n_results=3),
    )
    assert all(r is not None for r in results)
    assert c.query_cache_stats()["entries"] == 1
