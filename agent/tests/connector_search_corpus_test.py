"""Offline unit tests for the frozen-corpus search backend (app/connector_search_corpus.py).

Exact-key fixtures cannot serve an adaptive agent: keys hash the literal query text, and the
agent emits a different query every run (measured at ~0 effective hits from a 289 MB record
pass, scripts/BENCHMARK_NATIVE.md:14-19). This backend replaces exact-key lookup with a local
BM25 index over a recorded corpus, so ANY query -- recorded or not -- returns ranked results
from a frozen evidence universe at $0 and without the GPU.
"""
import json

import pytest

from shared.connector_config import ConnectorConfig

from agent.app.connector_search_corpus import ConnectorSearchCorpus

DOCS = [
    {"url": "https://example.org/eiffel", "title": "Eiffel Tower",
     "description": "Wrought-iron lattice tower in Paris.",
     "text": "The Eiffel Tower is 330 m tall and stands on the Champ de Mars in Paris."},
    {"url": "https://example.org/denali", "title": "Denali",
     "description": "Highest peak in North America.",
     "text": "Denali rises to 6190 metres and was resurveyed in 2015 in Alaska."},
    {"url": "https://example.org/goals", "title": "Career goals",
     "description": "Scoring record.",
     "text": "He scored 400 goals for the club and 424 goals in total."},
]


def _corpus(tmp_path, docs=DOCS):
    path = tmp_path / "corpus"
    path.mkdir()
    with (path / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return path


def _connector(tmp_path, docs=DOCS):
    return ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(_corpus(tmp_path, docs)))


@pytest.mark.asyncio
async def test_unrecorded_query_still_returns_the_relevant_document(tmp_path):
    """The whole point: a query never seen during recording must still rank the corpus.

    No fixture key for this phrasing exists, and an exact-key cache would miss entirely.
    """
    connector = _connector(tmp_path)
    results = await connector.query_search("how tall is the tower in Paris", count=3)
    assert results is not None
    assert results[0]["url"] == "https://example.org/eiffel"


@pytest.mark.asyncio
async def test_query_never_recorded_verbatim_ranks_by_content_not_by_key(tmp_path):
    """A second unseen phrasing hits a different document -- ranking, not key lookup."""
    connector = _connector(tmp_path)
    results = await connector.query_search("mountain resurveyed in Alaska", count=3)
    assert results[0]["url"] == "https://example.org/denali"


@pytest.mark.asyncio
async def test_results_carry_exactly_the_backend_contract_keys(tmp_path):
    """Downstream code must not be able to tell this backend from Serper."""
    connector = _connector(tmp_path)
    results = await connector.query_search("Paris tower", count=1)
    assert set(results[0]) == {"title", "url", "description"}
    assert all(isinstance(value, str) for value in results[0].values())


@pytest.mark.asyncio
async def test_ranking_is_identical_across_repeated_calls(tmp_path):
    """Replay fidelity: the same query must not drift between calls in one process."""
    connector = _connector(tmp_path)
    first = await connector.query_search("goals scored for the club", count=3)
    second = await connector.query_search("goals scored for the club", count=3)
    assert first == second


@pytest.mark.asyncio
async def test_ranking_is_identical_across_separate_connector_instances(tmp_path):
    """Replay fidelity across process restarts: a fresh index must rank the same way."""
    corpus = _corpus(tmp_path)
    one = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(corpus))
    two = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(corpus))
    assert await one.query_search("Paris", count=3) == await two.query_search("Paris", count=3)


@pytest.mark.asyncio
async def test_count_is_honoured_client_side(tmp_path):
    """count never enters a cache key here, so any value replays against one recording."""
    connector = _connector(tmp_path)
    assert len(await connector.query_search("goals metres tower", count=1)) == 1
    assert len(await connector.query_search("goals metres tower", count=2)) == 2


@pytest.mark.asyncio
async def test_empty_corpus_returns_empty_list_not_none(tmp_path):
    """``None`` means "backend failed to initialise"; an empty corpus is not a failure.

    The live backends conflate these: a cold fixture cache fails init_search_api, query_search
    returns None, and the agent silently proceeds as though the web held nothing.
    """
    connector = _connector(tmp_path, docs=[])
    results = await connector.query_search("anything at all", count=5)
    assert results == []


@pytest.mark.asyncio
async def test_query_matching_nothing_returns_empty_list(tmp_path):
    connector = _connector(tmp_path)
    assert await connector.query_search("zzzzz qqqqq", count=5) == []


@pytest.mark.asyncio
async def test_malformed_corpus_line_is_skipped_not_fatal(tmp_path):
    """An autonomous replay cannot stop to ask a human about one bad line."""
    path = tmp_path / "corpus"
    path.mkdir()
    (path / "documents.jsonl").write_text(
        json.dumps(DOCS[0]) + "\n" + "{not valid json\n" + json.dumps(DOCS[1]) + "\n",
        encoding="utf-8")
    connector = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(path))
    assert len(connector.documents) == 2


def test_factory_selects_the_corpus_backend(tmp_path, monkeypatch):
    """SEARCH_PROVIDER=corpus must route through the same factory as every other backend."""
    from agent.app.connector_search import create_search_backend

    monkeypatch.setenv("SEARCH_PROVIDER", "corpus")
    monkeypatch.setenv("LEDGER_CORPUS_DIR", str(_corpus(tmp_path)))
    backend = create_search_backend(ConnectorConfig())
    assert isinstance(backend, ConnectorSearchCorpus)
    assert len(backend.documents) == 3


class _RecordingFallback:
    """Stands in for a live paid backend, counting how often it is actually reached."""

    def __init__(self, results=None):
        self.calls = []
        self._results = results if results is not None else [
            {"title": "Live", "url": "https://live.example/new", "description": "fetched live"}]

    async def query_search(self, query, count=10):
        self.calls.append((query, count))
        return list(self._results)


@pytest.mark.asyncio
async def test_corpus_hit_never_reaches_the_live_fallback(tmp_path):
    """The point of a corpus is that a hit costs nothing. A hit must not touch the network."""
    fallback = _RecordingFallback()
    connector = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(_corpus(tmp_path)),
                                      fallback=fallback, max_live_fallbacks=5)
    await connector.query_search("tower in Paris", count=3)
    assert fallback.calls == []
    assert connector.live_fallbacks == 0


@pytest.mark.asyncio
async def test_corpus_miss_falls_back_to_live_and_counts_the_call(tmp_path):
    """Chosen policy: a miss goes live rather than breaking the run. The cost must be counted."""
    fallback = _RecordingFallback()
    connector = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(_corpus(tmp_path)),
                                      fallback=fallback, max_live_fallbacks=5)
    results = await connector.query_search("zzzzz qqqqq", count=3)
    assert results[0]["url"] == "https://live.example/new"
    assert connector.live_fallbacks == 1


@pytest.mark.asyncio
async def test_live_fallback_stops_at_the_cap(tmp_path):
    """Spend is bounded in code. Past the cap a miss returns empty instead of billing again."""
    fallback = _RecordingFallback()
    connector = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(_corpus(tmp_path)),
                                      fallback=fallback, max_live_fallbacks=2)
    for i in range(4):
        # Distinct queries: an identical repeat is served from the absorbed result and never
        # reaches the fallback, so repeating one query would not exercise the cap at all.
        await connector.query_search(f"zzzzz{i} qqqqq{i}", count=3)
    assert len(fallback.calls) == 2
    assert connector.live_fallbacks == 2


@pytest.mark.asyncio
async def test_no_fallback_configured_means_a_miss_stays_offline(tmp_path):
    """Default construction must never reach the network, whatever the query."""
    connector = _connector(tmp_path)
    assert await connector.query_search("zzzzz qqqqq", count=3) == []
    assert connector.live_fallbacks == 0


@pytest.mark.asyncio
async def test_live_results_are_added_to_the_corpus_for_later_replay(tmp_path):
    """Record-on-miss: the second identical query must be served from the corpus, not billed."""
    fallback = _RecordingFallback()
    connector = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(_corpus(tmp_path)),
                                      fallback=fallback, max_live_fallbacks=5)
    await connector.query_search("zzzzz qqqqq", count=3)
    results = await connector.query_search("zzzzz qqqqq", count=3)
    assert len(fallback.calls) == 1
    assert results[0]["url"] == "https://live.example/new"


@pytest.mark.asyncio
async def test_each_result_records_which_source_served_it(tmp_path):
    """Replay fidelity must be auditable per cell, not assumed."""
    fallback = _RecordingFallback()
    connector = ConnectorSearchCorpus(ConnectorConfig(), corpus_dir=str(_corpus(tmp_path)),
                                      fallback=fallback, max_live_fallbacks=5)
    await connector.query_search("tower in Paris", count=1)
    await connector.query_search("zzzzz qqqqq", count=1)
    assert connector.provenance == ["corpus", "live"]
