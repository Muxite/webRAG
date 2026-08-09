"""Concurrency-hygiene regression tests for ConnectorChroma.

These lock in the fixes for the barrage ChromaDB hang epidemic:
  * every raw chroma op is bounded by ``chroma_op_timeout`` and fails OPEN
    (returns False/None) instead of blocking to the cell cap;
  * a single transient op failure does NOT tear the client down (rebuild churn
    worsens SQLite contention) — teardown only after FAILURE_TEARDOWN_THRESHOLD;
  * ``init_chroma`` retries a bounded number of times (no ~302s sleep storm);
  * collections are memoized so add/query don't re-round-trip get_or_create;
  * ``add_to_chroma_parallel`` stores batches SEQUENTIALLY (a SQLite-backed server
    has one global write lock — concurrent writes only amplify contention).

All use an in-memory fake async client; no live ChromaDB server is required.
"""
import asyncio

import pytest

from shared.connector_config import ConnectorConfig
from agent.app.connector_chroma import ConnectorChroma


class _FakeCollection:
    def __init__(self, hang: float = 0.0, raises: bool = False, concurrency_probe=None):
        self.hang = hang
        self.raises = raises
        self._probe = concurrency_probe  # dict tracking {"cur","max"} for overlap checks
        self.add_calls = 0
        self.query_calls = 0

    async def _maybe_block(self):
        if self._probe is not None:
            self._probe["cur"] += 1
            self._probe["max"] = max(self._probe["max"], self._probe["cur"])
        try:
            if self.hang:
                await asyncio.sleep(self.hang)
            if self.raises:
                raise RuntimeError("simulated chroma error (e.g. database is locked)")
        finally:
            if self._probe is not None:
                self._probe["cur"] -= 1

    async def add(self, ids, metadatas, documents):
        self.add_calls += 1
        await self._maybe_block()

    async def query(self, query_texts, n_results, where=None):
        self.query_calls += 1
        await self._maybe_block()
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}


class _FakeAsyncClient:
    def __init__(self, collection: _FakeCollection):
        self._collection = collection
        self.get_or_create_calls = 0

    async def heartbeat(self):
        return 1

    async def get_or_create_collection(self, name):
        self.get_or_create_calls += 1
        return self._collection


def _connector(collection: _FakeCollection, op_timeout: float = 15.0) -> ConnectorChroma:
    """Build a ConnectorChroma pre-wired to a fake client (skips real init)."""
    cfg = ConnectorConfig()
    cfg.chroma_op_timeout = op_timeout
    c = ConnectorChroma(cfg)
    c._chroma = _FakeAsyncClient(collection)
    c.chroma_api_ready = True
    return c


@pytest.mark.asyncio
async def test_add_times_out_and_fails_open():
    """A hung add returns False within the op timeout instead of blocking forever."""
    coll = _FakeCollection(hang=5.0)
    c = _connector(coll, op_timeout=0.05)
    result = await asyncio.wait_for(
        c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"]), timeout=1.0
    )
    assert result is False


@pytest.mark.asyncio
async def test_query_times_out_and_fails_open():
    """A hung query returns None within the op timeout instead of blocking forever."""
    coll = _FakeCollection(hang=5.0)
    c = _connector(coll, op_timeout=0.05)
    result = await asyncio.wait_for(
        c.query_chroma("mem_x", ["q"], n_results=1), timeout=1.0
    )
    assert result is None


@pytest.mark.asyncio
async def test_transient_failure_does_not_tear_down_below_threshold():
    """Failures below the threshold keep the client alive (no rebuild churn)."""
    coll = _FakeCollection(raises=True)
    c = _connector(coll)
    for _ in range(ConnectorChroma.FAILURE_TEARDOWN_THRESHOLD - 1):
        assert await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"]) is False
    assert c.chroma_api_ready is True
    assert c._consecutive_failures == ConnectorChroma.FAILURE_TEARDOWN_THRESHOLD - 1


@pytest.mark.asyncio
async def test_teardown_after_threshold_consecutive_failures():
    """The Nth consecutive failure tears the client down and clears the cache."""
    coll = _FakeCollection(raises=True)
    c = _connector(coll)
    for _ in range(ConnectorChroma.FAILURE_TEARDOWN_THRESHOLD):
        await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])
    assert c.chroma_api_ready is False
    assert c._collections == {}
    assert c._consecutive_failures == 0


@pytest.mark.asyncio
async def test_success_resets_failure_counter():
    """A success between failures resets the counter so teardown needs N in a row."""
    coll = _FakeCollection(raises=True)
    c = _connector(coll)
    await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])  # fail -> 1
    assert c._consecutive_failures == 1
    coll.raises = False
    assert await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"]) is True
    assert c._consecutive_failures == 0


@pytest.mark.asyncio
async def test_collection_is_memoized():
    """Repeated add/query to one collection hit get_or_create exactly once."""
    coll = _FakeCollection()
    c = _connector(coll)
    await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["a"])
    await c.add_to_chroma("mem_x", ["2"], [{"k": "v"}], ["b"])
    await c.query_chroma("mem_x", ["q"], n_results=1)
    assert c._chroma.get_or_create_calls == 1


@pytest.mark.asyncio
async def test_init_chroma_retry_is_bounded():
    """A dead server yields a bounded number of attempts, fast — no ~302s sleep storm."""
    cfg = ConnectorConfig()
    cfg.chroma_init_attempts = 2
    cfg.retry_base_delay = 0.01
    cfg.jitter_seconds = 0.0
    c = ConnectorChroma(cfg)
    attempts = {"n": 0}

    async def _always_fail():
        attempts["n"] += 1
        return False

    c._try_init_chroma = _always_fail
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    ok = await c.init_chroma()
    elapsed = loop.time() - t0
    assert ok is False
    assert attempts["n"] == 2
    assert elapsed < 2.0


class _FakeSyncCollection:
    """Sync collection (embedded PersistentClient shape) — methods are NOT coroutines."""

    def __init__(self):
        self.add_calls = 0

    def add(self, ids, metadatas, documents):
        self.add_calls += 1

    def query(self, query_texts, n_results, where=None):
        return {"documents": [["synced"]], "metadatas": [[{}]], "distances": [[0.0]], "ids": [["1"]]}


class _FakeSyncClient:
    def __init__(self, collection: _FakeSyncCollection):
        self._collection = collection

    def get_or_create_collection(self, name, embedding_function=None):
        return self._collection


@pytest.mark.asyncio
async def test_sync_client_ops_dispatch_via_thread():
    """Embedded (sync) client ops run through asyncio.to_thread and still work."""
    coll = _FakeSyncCollection()
    cfg = ConnectorConfig()
    c = ConnectorChroma(cfg)
    c._chroma = _FakeSyncClient(coll)
    c._is_async_client = False
    c.chroma_api_ready = True
    assert await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"]) is True
    assert coll.add_calls == 1
    res = await c.query_chroma("mem_x", ["q"], n_results=1)
    assert res is not None and res["documents"] == [["synced"]]


@pytest.mark.asyncio
async def test_embedded_mode_end_to_end(tmp_path):
    """Real per-process PersistentClient: init + add + semantic query, no server."""
    try:
        from chromadb.utils import embedding_functions as ef
        ef.DefaultEmbeddingFunction()(["ping"])  # ensure the default model is available
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"default embedding model unavailable: {e}")
    cfg = ConnectorConfig()
    cfg.chroma_mode = "embedded"
    cfg.chroma_embedded_path = str(tmp_path / "db")
    c = ConnectorChroma(cfg)
    assert await c.init_chroma() is True
    assert c._is_async_client is False
    added = await c.add_to_chroma(
        "mem_e", ["1", "2"], [{"k": "a"}, {"k": "b"}],
        ["The capital of France is Paris.", "Photosynthesis converts light to energy."],
    )
    assert added is True
    res = await c.query_chroma("mem_e", ["What is the capital of France?"], n_results=1)
    assert res is not None
    assert res["documents"][0][0].startswith("The capital of France")
    # get-by-id is an exact membership check, not a similarity search: chroma returns ONLY
    # the ids it actually holds, which is how the plan-library sync tells "already indexed
    # here" from "the manifest says so". An absent id is simply missing from the reply.
    got = await c.get_from_chroma("mem_e", ["2", "never_added"])
    assert got is not None and got["ids"] == ["2"]
    assert (await c.get_from_chroma("mem_e", []))["ids"] == []  # no ids, no round-trip


@pytest.mark.asyncio
async def test_parallel_add_is_sequential():
    """add_to_chroma_parallel never overlaps writes (one SQLite write lock)."""
    probe = {"cur": 0, "max": 0}
    coll = _FakeCollection(hang=0.01, concurrency_probe=probe)
    c = _connector(coll)
    n = 6
    ids = [str(i) for i in range(n)]
    metas = [{"k": "v"} for _ in range(n)]
    docs = [f"doc {i}" for i in range(n)]
    ok = await c.add_to_chroma_parallel("mem_x", ids, metas, docs, batch_size=2)
    assert ok is True
    assert probe["max"] == 1  # never two concurrent writes
    assert coll.add_calls == 3  # 6 docs / batch_size 2
