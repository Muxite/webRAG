"""Per-search provenance recorded through ``AgentIO.search`` into telemetry.

The gap: ``ConnectorSearchCorpus`` tracks whether each call was served by the frozen corpus,
a live fallback, or nothing, but only as an in-memory instance attribute -- it never reaches a
stored cell. That makes ``max_live_fallbacks`` preregistration gates structurally uncheckable.

The fix routes the signal through telemetry instead of widening ``summarize_observability``'s
signature: ``agent_io.search`` already calls ``telemetry.record_timing(name="search", ...)``,
so it reads ``connector_search.provenance`` (when present) and folds the last entry into that
timing's payload as ``search_provenance``. A backend without a ``provenance`` attribute (every
live backend) must behave exactly as today.
"""
import pytest
from unittest.mock import MagicMock

from agent.app.agent_io import AgentIO


def _make_io(mock_search, telemetry=None):
    mock_llm = MagicMock()
    mock_llm.set_telemetry = MagicMock()
    mock_search.set_telemetry = MagicMock()
    mock_chroma = MagicMock()
    mock_chroma.set_telemetry = MagicMock()
    mock_http = MagicMock()
    mock_http.set_telemetry = MagicMock()

    io = AgentIO(
        connector_llm=mock_llm,
        connector_search=mock_search,
        connector_http=mock_http,
        connector_chroma=mock_chroma,
    )
    if telemetry is not None:
        io.set_telemetry(telemetry)
    return io


class _FakeTelemetry:
    def __init__(self):
        self.timings = []

    def record_timing(self, name, started_at=None, success=True, payload=None, error=None):
        self.timings.append({"name": name, "payload": payload or {}, "success": success})

    def record_document_seen(self, source, document):
        pass


class _CorpusLikeSearch:
    """Mimics ConnectorSearchCorpus's provenance-tracking shape without the real BM25 machinery."""

    def __init__(self, results, provenance_entry):
        self._results = results
        self._provenance_entry = provenance_entry
        self.provenance = []

    async def query_search(self, query, count=10):
        self.provenance.append(self._provenance_entry)
        return self._results


class _NonCorpusSearch:
    """A backend with no ``provenance`` attribute at all -- serper/brave/searxng shape."""

    def __init__(self, results):
        self._results = results

    async def query_search(self, query, count=10):
        return self._results


@pytest.mark.asyncio
async def test_corpus_hit_records_corpus_provenance_in_search_timing():
    telemetry = _FakeTelemetry()
    search = _CorpusLikeSearch([{"title": "t", "url": "u", "description": "d"}], "corpus")
    io = _make_io(search, telemetry)
    await io.search("query")
    search_timings = [t for t in telemetry.timings if t["name"] == "search"]
    assert len(search_timings) == 1
    assert search_timings[0]["payload"]["search_provenance"] == "corpus"


@pytest.mark.asyncio
async def test_live_fallback_records_live_provenance_in_search_timing():
    telemetry = _FakeTelemetry()
    search = _CorpusLikeSearch([{"title": "t", "url": "u", "description": "d"}], "live")
    io = _make_io(search, telemetry)
    await io.search("query")
    search_timings = [t for t in telemetry.timings if t["name"] == "search"]
    assert search_timings[0]["payload"]["search_provenance"] == "live"


@pytest.mark.asyncio
async def test_zero_result_search_records_none_provenance_in_search_timing():
    telemetry = _FakeTelemetry()
    search = _CorpusLikeSearch([], "none")
    io = _make_io(search, telemetry)
    await io.search("query")
    search_timings = [t for t in telemetry.timings if t["name"] == "search"]
    assert search_timings[0]["payload"]["search_provenance"] == "none"


@pytest.mark.asyncio
async def test_non_corpus_backend_records_no_provenance_key_and_is_unchanged():
    telemetry = _FakeTelemetry()
    search = _NonCorpusSearch([{"title": "t", "url": "u", "description": "d"}])
    io = _make_io(search, telemetry)
    await io.search("query")
    search_timings = [t for t in telemetry.timings if t["name"] == "search"]
    assert len(search_timings) == 1
    assert "search_provenance" not in search_timings[0]["payload"]
    # `arg_coerced` is new (tool-argument-coercion fix): always present, `None` here since
    # "query" was never a wrapped list/dict -- see agent_io.py's `_coerce_search_query` and
    # agent/tests/tool_argument_coercion_test.py.
    assert search_timings[0]["payload"] == {"query": "query", "result_count": 1, "arg_coerced": None}


@pytest.mark.asyncio
async def test_two_sequential_calls_each_record_their_own_provenance():
    """Guards against reading a stale ``provenance[-1]`` from a previous call."""
    telemetry = _FakeTelemetry()
    search = _CorpusLikeSearch([{"title": "t", "url": "u", "description": "d"}], "corpus")
    io = _make_io(search, telemetry)
    await io.search("first query")
    search._provenance_entry = "live"
    await io.search("second query")
    search_timings = [t for t in telemetry.timings if t["name"] == "search"]
    assert len(search_timings) == 2
    assert search_timings[0]["payload"]["search_provenance"] == "corpus"
    assert search_timings[1]["payload"]["search_provenance"] == "live"
