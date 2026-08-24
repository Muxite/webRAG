"""Regression tests for Bug B: Chroma op/init failures were invisible to the infra gate.

``agent/app/testing/utils.py:_is_infra_timing`` classifies a failed timing as infra when
``payload["infra_failed"] is True``, OR ``payload["status"]`` is a retryable code, OR status
is absent and the timing name is in {"http_request", "search_query", "visit"}. Chroma emitted
only ``chroma_add``/``chroma_query`` timings with neither status nor infra_failed set, so a
genuine Chroma outage was silently scored as a task/model failure. Worse, ``init_chroma``'s
exhausted-retry branch and ``get_or_create_collection``'s failure branch emitted no timing at
all, so callers that bail out early (``if coll is None: return False``) left zero trace.

These tests lock in: (1) a transport/timeout-caused failure on add/query/get_or_create/init is
now stamped ``infra_failed=True`` and classifies as infra end-to-end via ``_is_infra_timing``;
(2) a caller/logic error (bad args, malformed input) is NOT stamped and does not classify as
infra — it must stay a genuine task failure.

No live ChromaDB server is required; an in-memory fake async client stands in.
"""
import asyncio

import pytest

from shared.connector_config import ConnectorConfig
from agent.app.connector_chroma import ConnectorChroma
from agent.app.testing.utils import _is_infra_timing, _summarize_infra


class _RecordingTelemetry:
    def __init__(self):
        self.timings = []

    def record_timing(self, **kwargs):
        self.timings.append(kwargs)

    def record_event(self, event, payload):
        pass

    def record_io(self, **kwargs):
        pass


class _ConnectTimeoutError(Exception):
    """Stand-in for httpx.ConnectTimeout / chromadb transport errors: the classifier
    matches on exception CLASS NAME containing "Connect" or "Timeout", not identity."""


class _FakeCollection:
    def __init__(self, exc: Exception = None):
        self._exc = exc

    async def add(self, ids, metadatas, documents):
        if self._exc is not None:
            raise self._exc

    async def query(self, query_texts, n_results, where=None):
        if self._exc is not None:
            raise self._exc
        return {"ids": [[]]}

    async def delete(self, ids):
        if self._exc is not None:
            raise self._exc

    async def get(self, ids, include=None):
        if self._exc is not None:
            raise self._exc
        return {"ids": list(ids)}


class _FakeAsyncClient:
    def __init__(
        self,
        collection=None,
        get_or_create_exc: Exception = None,
        delete_collection_exc: Exception = None,
        list_collections_exc: Exception = None,
    ):
        self._collection = collection
        self._get_or_create_exc = get_or_create_exc
        self._delete_collection_exc = delete_collection_exc
        self._list_collections_exc = list_collections_exc

    async def heartbeat(self):
        return 1

    async def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        if self._get_or_create_exc is not None:
            raise self._get_or_create_exc
        return self._collection or _FakeCollection()

    async def delete_collection(self, name):
        if self._delete_collection_exc is not None:
            raise self._delete_collection_exc

    async def list_collections(self):
        if self._list_collections_exc is not None:
            raise self._list_collections_exc
        return []


def _connector(chroma_client) -> ConnectorChroma:
    cfg = ConnectorConfig()
    c = ConnectorChroma(cfg)
    c._chroma = chroma_client
    c.chroma_api_ready = True
    telemetry = _RecordingTelemetry()
    c.set_telemetry(telemetry)
    return c, telemetry


def _last_timing(telemetry, name):
    for t in reversed(telemetry.timings):
        if t["name"] == name:
            return t
    raise AssertionError(f"no timing named {name!r} recorded; got {telemetry.timings}")


# ---------------------------------------------------------------------------------------
# add_to_chroma
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_transport_failure_stamps_infra_failed():
    """A ConnectTimeout-style failure on add is stamped infra_failed and classifies as infra."""
    coll = _FakeCollection(_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    ok = await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])
    assert ok is False
    timing = _last_timing(telemetry, "chroma_add")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_add_timeout_error_stamps_infra_failed():
    """asyncio.TimeoutError (the _op/_bounded timeout wrapper's own exception) is infra too."""
    coll = _FakeCollection(asyncio.TimeoutError("op timed out"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    ok = await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])
    assert ok is False
    timing = _last_timing(telemetry, "chroma_add")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_add_logic_error_not_stamped_infra():
    """A caller/logic error (e.g. malformed input) must NOT be tagged infra_failed."""
    coll = _FakeCollection(ValueError("embedding dimension mismatch"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    ok = await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])
    assert ok is False
    timing = _last_timing(telemetry, "chroma_add")
    assert timing["payload"].get("infra_failed") is False
    assert _is_infra_timing(timing) is False


# ---------------------------------------------------------------------------------------
# query_chroma
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_transport_failure_stamps_infra_failed():
    coll = _FakeCollection(_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    res = await c.query_chroma("mem_x", ["q"], n_results=1)
    assert res is None
    timing = _last_timing(telemetry, "chroma_query")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_query_logic_error_not_stamped_infra():
    """A genuine bug in the query op (not a transport/timeout signal) stays unflagged."""
    coll = _FakeCollection(RuntimeError("malformed where clause"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    res = await c.query_chroma("mem_x", ["q"], n_results=1)
    assert res is None
    timing = _last_timing(telemetry, "chroma_query")
    assert timing["payload"].get("infra_failed") is False
    assert _is_infra_timing(timing) is False


# ---------------------------------------------------------------------------------------
# get_or_create_collection — previously emitted NO timing at all on failure, so a caller
# that bails on `coll is None` (add_to_chroma / query_chroma) left zero trace.
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_or_create_collection_failure_now_visible_and_infra_tagged():
    client = _FakeAsyncClient(get_or_create_exc=_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(client)
    coll = await c.get_or_create_collection("mem_x")
    assert coll is None
    timing = _last_timing(telemetry, "chroma_get_or_create")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_get_or_create_collection_logic_error_not_infra_tagged():
    client = _FakeAsyncClient(get_or_create_exc=ValueError("bad collection name"))
    c, telemetry = _connector(client)
    coll = await c.get_or_create_collection("mem_x")
    assert coll is None
    timing = _last_timing(telemetry, "chroma_get_or_create")
    assert timing["payload"].get("infra_failed") is False
    assert _is_infra_timing(timing) is False


@pytest.mark.asyncio
async def test_add_to_chroma_surfaces_get_or_create_failure_via_telemetry():
    """Before the fix, add_to_chroma's `if coll is None: return False` bailed ABOVE its own
    _record_timing call, so a get_or_create outage was completely invisible end-to-end. It
    must now show up (via get_or_create_collection's own timing) even though add_to_chroma
    itself still records nothing for this path."""
    client = _FakeAsyncClient(get_or_create_exc=_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(client)
    ok = await c.add_to_chroma("mem_x", ["1"], [{"k": "v"}], ["doc"])
    assert ok is False
    timing = _last_timing(telemetry, "chroma_get_or_create")
    assert _is_infra_timing(timing) is True


# ---------------------------------------------------------------------------------------
# init_chroma — exhausted retries previously emitted NO timing at all (log + return False).
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_init_chroma_exhausted_retries_emits_infra_timing():
    cfg = ConnectorConfig()
    cfg.chroma_init_attempts = 2
    cfg.retry_base_delay = 0.01
    cfg.jitter_seconds = 0.0
    c = ConnectorChroma(cfg)
    telemetry = _RecordingTelemetry()
    c.set_telemetry(telemetry)

    async def _always_fail():
        return False

    c._try_init_chroma = _always_fail
    ok = await c.init_chroma()
    assert ok is False
    timing = _last_timing(telemetry, "chroma_init")
    assert timing["success"] is False
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_init_chroma_success_emits_no_failure_timing():
    """A successful init must not spuriously emit a failure timing."""
    cfg = ConnectorConfig()
    c = ConnectorChroma(cfg)
    telemetry = _RecordingTelemetry()
    c.set_telemetry(telemetry)

    async def _always_succeed():
        c.chroma_api_ready = True
        return True

    c._try_init_chroma = _always_succeed
    ok = await c.init_chroma()
    assert ok is True
    assert all(t["name"] != "chroma_init" for t in telemetry.timings)


# ---------------------------------------------------------------------------------------
# delete_collection / list_collections / delete_from_chroma / get_from_chroma — these
# previously emitted NO timing at all, success or failure (Bug B coverage gap). Now
# routed through the centralized _op wrapper, so both directions are covered too.
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_collection_transport_failure_stamps_infra_failed():
    client = _FakeAsyncClient(delete_collection_exc=_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(client)
    ok = await c.delete_collection("mem_x")
    assert ok is False
    timing = _last_timing(telemetry, "chroma_delete_collection")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_delete_collection_success_emits_success_timing():
    client = _FakeAsyncClient()
    c, telemetry = _connector(client)
    ok = await c.delete_collection("mem_x")
    assert ok is True
    timing = _last_timing(telemetry, "chroma_delete_collection")
    assert timing["success"] is True


@pytest.mark.asyncio
async def test_list_collections_transport_failure_stamps_infra_failed():
    client = _FakeAsyncClient(list_collections_exc=_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(client)
    result = await c.list_collections()
    assert result == []
    timing = _last_timing(telemetry, "chroma_list_collections")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_list_collections_success_emits_success_timing():
    client = _FakeAsyncClient()
    c, telemetry = _connector(client)
    result = await c.list_collections()
    assert result == []
    timing = _last_timing(telemetry, "chroma_list_collections")
    assert timing["success"] is True


@pytest.mark.asyncio
async def test_delete_from_chroma_transport_failure_stamps_infra_failed():
    coll = _FakeCollection(_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    ok = await c.delete_from_chroma("mem_x", ["1"])
    assert ok is False
    timing = _last_timing(telemetry, "chroma_delete")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_delete_from_chroma_logic_error_not_stamped_infra():
    coll = _FakeCollection(ValueError("bad id"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    ok = await c.delete_from_chroma("mem_x", ["1"])
    assert ok is False
    timing = _last_timing(telemetry, "chroma_delete")
    assert timing["payload"].get("infra_failed") is False
    assert _is_infra_timing(timing) is False


@pytest.mark.asyncio
async def test_get_from_chroma_transport_failure_stamps_infra_failed():
    coll = _FakeCollection(_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    result = await c.get_from_chroma("mem_x", ["1"])
    assert result is None
    timing = _last_timing(telemetry, "chroma_get")
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


@pytest.mark.asyncio
async def test_get_from_chroma_logic_error_not_stamped_infra():
    coll = _FakeCollection(RuntimeError("malformed include"))
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    result = await c.get_from_chroma("mem_x", ["1"])
    assert result is None
    timing = _last_timing(telemetry, "chroma_get")
    assert timing["payload"].get("infra_failed") is False
    assert _is_infra_timing(timing) is False


@pytest.mark.asyncio
async def test_get_from_chroma_success_emits_success_timing():
    coll = _FakeCollection()
    c, telemetry = _connector(_FakeAsyncClient(collection=coll))
    result = await c.get_from_chroma("mem_x", ["1"])
    assert result == {"ids": ["1"]}
    timing = _last_timing(telemetry, "chroma_get")
    assert timing["success"] is True


# ---------------------------------------------------------------------------------------
# THE REGRESSION: chroma_get_or_create (and, before, chroma_init) emitted a timing on
# FAILURE ONLY. A per-op bucket that never records a success has success == 0 forever,
# which trips agent/app/testing/utils.py:_summarize_infra's severity gate
# (`rate > 0.5 or success == 0`) on a single transient failure, flipping the WHOLE cell's
# infra.failed to True even though every other op in the cell was fine. Verified here
# end-to-end through the real (unedited) _summarize_infra.
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_get_or_create_failure_amid_successes_does_not_flag_cell():
    ok_client = _FakeAsyncClient()
    c, telemetry = _connector(ok_client)

    # Five successful get_or_create round-trips on distinct collection names (distinct
    # names avoid the per-connector memoization cache short-circuiting the round trip).
    for i in range(5):
        coll = await c.get_or_create_collection(f"mem_ok_{i}")
        assert coll is not None

    # One transient transport failure on a sixth, distinct collection.
    c._chroma = _FakeAsyncClient(get_or_create_exc=_ConnectTimeoutError("connection reset"))
    coll = await c.get_or_create_collection("mem_fail")
    assert coll is None

    summary = _summarize_infra(telemetry.timings)
    assert summary["ops"] == ["chroma_get_or_create"]
    assert summary["rates"]["chroma_get_or_create"] == pytest.approx(1 / 6, rel=1e-3)
    # Before the fix: success was permanently 0 for this bucket -> failed was always True
    # here regardless of the 5 successes. After the fix: 1/6 is well under the 0.5 rate
    # threshold and success != 0, so the cell must NOT be flagged.
    assert summary["failed"] is False


@pytest.mark.asyncio
async def test_total_get_or_create_outage_still_flags_cell():
    """A genuine total outage (every attempt fails) must still flag -- the fix must not
    weaken detection of a real outage, only stop a single blip from being one.

    Stays under FAILURE_TEARDOWN_THRESHOLD (3 consecutive failures) so the connector
    doesn't tear itself down and fall into a *second*, unrelated failing op
    (``chroma_init``, since this test's ConnectorConfig has no chroma_url) -- that
    teardown path is real behavior, just not what this test is isolating.
    """
    client = _FakeAsyncClient(get_or_create_exc=_ConnectTimeoutError("connection reset"))
    c, telemetry = _connector(client)
    for i in range(2):
        coll = await c.get_or_create_collection(f"mem_{i}")
        assert coll is None

    summary = _summarize_infra(telemetry.timings)
    assert summary["ops"] == ["chroma_get_or_create"]
    assert summary["rates"]["chroma_get_or_create"] == 1.0
    assert summary["failed"] is True
