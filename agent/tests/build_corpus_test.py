"""Unit tests for scripts/build_corpus.py -- harvesting a frozen search corpus from run results.

The first corpus should cost nothing: hundreds of stored cells already carry the pages the agent
visited and the URLs it cited. Harvesting those turns existing spend into a reusable, deterministic
evidence universe, so a live recording pass is a top-up rather than a prerequisite.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_corpus  # noqa: E402


def _cell(tmp_path, name, pages=None, extractions=None):
    """Write a per-cell result JSON in the real nested shape the runner emits."""
    payload = {"execution": {"output": {
        "pages": pages or [],
        "extractions": extractions or [],
    }}}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


PAGE = {"page_id": "p1", "url": "https://example.org/eiffel",
        "text": "The Eiffel Tower is 330 m tall.", "content_hash": "abc", "truncated": False}


def test_harvest_reads_stored_pages_from_a_result_cell(tmp_path):
    _cell(tmp_path, "run_122_model_variant_r1.json", pages=[PAGE])
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["url"] == "https://example.org/eiffel"
    assert "Eiffel Tower" in docs[0]["text"]


def test_harvest_counts_only_canonical_result_files(tmp_path):
    """A naive glob over the results dir is inflated by summary and trace files.

    ``*_summary.json`` reflects only the last cell of a multi-invocation run and ``*.jsonl`` are
    traces; counting either once produced a throughput figure 2.1x too high.
    """
    _cell(tmp_path, "run_122_model_variant_r1.json", pages=[PAGE])
    _cell(tmp_path, "run_summary.json", pages=[dict(PAGE, url="https://example.org/summary")])
    (tmp_path / "run_122_model_variant_r1.jsonl").write_text("{}\n", encoding="utf-8")
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert [doc["url"] for doc in docs] == ["https://example.org/eiffel"]


def test_harvest_deduplicates_the_same_url_across_cells(tmp_path):
    """The same page appears in every cell that visited it; the corpus should hold it once."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[PAGE])
    _cell(tmp_path, "b_122_m_v_r1.json", pages=[PAGE])
    assert len(build_corpus.harvest_documents(str(tmp_path))) == 1


def test_harvest_prefers_the_longest_text_for_a_repeated_url(tmp_path):
    """Cells truncate pages differently; the corpus should keep the most complete copy."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[dict(PAGE, text="short")])
    _cell(tmp_path, "b_122_m_v_r1.json", pages=[dict(PAGE, text="a considerably longer body")])
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert docs[0]["text"] == "a considerably longer body"


def test_harvest_skips_pages_without_a_url(tmp_path):
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[dict(PAGE, url="")])
    assert build_corpus.harvest_documents(str(tmp_path)) == []


def test_harvest_survives_a_corrupt_result_file(tmp_path):
    """One unreadable cell must not abort an unattended corpus build."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[PAGE])
    (tmp_path / "b_122_m_v_r1.json").write_text("{not json", encoding="utf-8")
    assert len(build_corpus.harvest_documents(str(tmp_path))) == 1


def test_write_corpus_emits_documents_jsonl_readable_by_the_backend(tmp_path):
    """The builder's output must load straight into the connector without a translation step."""
    from agent.app.connector_search_corpus import load_documents

    out = tmp_path / "corpus"
    build_corpus.write_corpus(str(out), [
        {"url": "https://example.org/a", "title": "A", "description": "d", "text": "body"}])
    loaded = load_documents(str(out))
    assert len(loaded) == 1
    assert loaded[0].url == "https://example.org/a"


def test_harvest_collapses_urls_differing_only_by_fragment(tmp_path):
    """ADVERSARIAL: a fragment names a position on a page, not a different page."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[PAGE])
    _cell(tmp_path, "b_122_m_v_r1.json",
          pages=[dict(PAGE, url=PAGE["url"] + "#geology")])
    assert len(build_corpus.harvest_documents(str(tmp_path))) == 1


def test_harvest_collapses_the_same_page_reached_by_different_query_strings(tmp_path):
    """ADVERSARIAL: measured on the real corpus -- the USGS Denali release appeared twice,
    once bare and once with `?qt-science_center_objects=0#...`, burning two of three result
    slots with one page. URL canonicalisation keeps the query, so identical content must
    collapse on content, not on URL.
    """
    body = "Denali's new elevation is 20,310 feet after the 2015 resurvey."
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[dict(PAGE, text=body)])
    _cell(tmp_path, "b_122_m_v_r1.json",
          pages=[dict(PAGE, url=PAGE["url"] + "?qt-science_center_objects=0", text=body)])
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert len(docs) == 1
    assert "?" not in docs[0]["url"], "the cleaner URL should win"


def test_harvest_keeps_genuinely_different_pages_on_one_host(tmp_path):
    """The collapse must not over-merge: different content is different documents."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[dict(PAGE, text="Denali is in Alaska.")])
    _cell(tmp_path, "b_122_m_v_r1.json",
          pages=[dict(PAGE, url="https://example.org/eiffel2", text="The Eiffel Tower is 330 m.")])
    assert len(build_corpus.harvest_documents(str(tmp_path))) == 2


# ---------------------------------------------------------------------------
# --live mode: query derivation
# ---------------------------------------------------------------------------

def test_derive_queries_leads_with_the_mandate_itself():
    mandate = "What is the height of Denali after the 2015 resurvey?"
    queries = build_corpus.derive_queries(mandate, 3)
    assert queries[0] == mandate


def test_derive_queries_adds_distinct_sub_queries():
    mandate = ("Find the resurveyed height of Denali. Compare it to the pre-2015 figure. "
               "Cite the USGS source.")
    queries = build_corpus.derive_queries(mandate, 5)
    assert len(queries) > 1
    assert len(queries) == len(set(q.lower() for q in queries)), "queries must be distinct"


def test_derive_queries_caps_at_requested_count():
    mandate = ("One. Two sentence here. Three sentence here too. Four sentence here as well. "
               "Five sentence here also. Six sentence goes here.")
    queries = build_corpus.derive_queries(mandate, 3)
    assert len(queries) <= 3


def test_derive_queries_is_deterministic():
    mandate = "Find the tallest mountain in North America and its official height in feet."
    assert build_corpus.derive_queries(mandate, 4) == build_corpus.derive_queries(mandate, 4)


def test_derive_queries_handles_a_mandate_with_no_sentence_breaks():
    mandate = "denali height"
    queries = build_corpus.derive_queries(mandate, 5)
    assert queries == ["denali height"]


# ---------------------------------------------------------------------------
# --live mode: generic document dedup (shared with harvest)
# ---------------------------------------------------------------------------

def test_dedupe_documents_collapses_same_content_different_url():
    body = "Denali's new elevation is 20,310 feet after the 2015 resurvey."
    docs = [
        {"url": "https://example.org/denali", "title": "Denali", "description": "", "text": body},
        {"url": "https://example.org/denali?utm=1", "title": "Denali", "description": "", "text": body},
    ]
    merged = build_corpus.dedupe_documents(docs)
    assert len(merged) == 1
    assert "?" not in merged[0]["url"]


def test_dedupe_documents_keeps_distinct_content():
    docs = [
        {"url": "https://a.example/1", "title": "A", "description": "", "text": "Alpha content here."},
        {"url": "https://b.example/2", "title": "B", "description": "", "text": "Beta content here."},
    ]
    assert len(build_corpus.dedupe_documents(docs)) == 2


def test_dedupe_documents_preserves_search_result_description():
    docs = [{"url": "https://a.example/1", "title": "A", "description": "a snippet from search",
             "text": "Full page body text goes here."}]
    merged = build_corpus.dedupe_documents(docs)
    assert merged[0]["description"] == "a snippet from search"


# ---------------------------------------------------------------------------
# --live mode: live_harvest (network injected as fakes, never real)
# ---------------------------------------------------------------------------

class _FakeSearch:
    """Deterministic fake search backend: records every query it was asked."""

    def __init__(self, results_by_query=None, default=None):
        self.calls = []
        self.results_by_query = results_by_query or {}
        self.default = default if default is not None else []

    async def __call__(self, query, count):
        self.calls.append((query, count))
        return self.results_by_query.get(query, self.default)[:count]


class _FakeVisit:
    """Deterministic fake page fetcher: records every URL it was asked to visit."""

    def __init__(self, text_by_url=None):
        self.calls = []
        self.text_by_url = text_by_url or {}

    async def __call__(self, url):
        self.calls.append(url)
        return self.text_by_url.get(url, f"body for {url}")


def test_live_harvest_visits_top_n_results_per_query():
    search = _FakeSearch(default=[
        {"url": f"https://example.org/{i}", "title": f"T{i}", "description": f"D{i}"}
        for i in range(10)
    ])
    visit = _FakeVisit()
    docs, searches_used, exhausted = _run_live_harvest(
        [("999", "one query only")], search, visit,
        queries_per_task=1, visits_per_query=3, max_searches=10)
    assert searches_used == 1
    assert len(visit.calls) == 3
    assert len(docs) == 3
    assert not exhausted


def test_live_harvest_builds_documents_from_search_metadata_and_visited_text():
    search = _FakeSearch(default=[
        {"url": "https://example.org/x", "title": "X title", "description": "X desc"}])
    visit = _FakeVisit(text_by_url={"https://example.org/x": "the fetched page body"})
    docs, _, _ = _run_live_harvest(
        [("999", "q")], search, visit, queries_per_task=1, visits_per_query=1, max_searches=5)
    assert docs == [{"url": "https://example.org/x", "title": "X title",
                     "description": "X desc", "text": "the fetched page body"}]


def test_live_harvest_stops_at_max_searches_and_reports_exhaustion():
    search = _FakeSearch(default=[{"url": "https://example.org/a", "title": "", "description": ""}])
    visit = _FakeVisit()
    mandates = [("100", "alpha query one. alpha query two. alpha query three."),
                ("200", "beta query one. beta query two. beta query three.")]
    docs, searches_used, exhausted = _run_live_harvest(
        mandates, search, visit, queries_per_task=3, visits_per_query=1, max_searches=2)
    assert searches_used == 2
    assert exhausted is True
    assert len(search.calls) == 2


def test_live_harvest_never_exceeds_max_searches_across_multiple_tasks():
    search = _FakeSearch(default=[])
    visit = _FakeVisit()
    mandates = [(str(i), f"task {i} query one. task {i} query two.") for i in range(5)]
    _docs, searches_used, _exhausted = _run_live_harvest(
        mandates, search, visit, queries_per_task=2, visits_per_query=1, max_searches=4)
    assert searches_used <= 4
    assert len(search.calls) <= 4


def test_live_harvest_skips_a_url_already_visited_this_run():
    search = _FakeSearch(results_by_query={
        "q1": [{"url": "https://example.org/dup", "title": "", "description": ""}],
        "q2": [{"url": "https://example.org/dup", "title": "", "description": ""}],
    })
    visit = _FakeVisit()
    mandates = [("999", "unused mandate text")]

    async def _direct():
        return await build_corpus.live_harvest(
            [(mandates[0][0], mandates[0][1])],
            search, visit,
            queries_override={"999": ["q1", "q2"]},
            visits_per_query=5, max_searches=10)

    import asyncio
    docs, searches_used, exhausted = asyncio.run(_direct())
    assert searches_used == 2
    assert len(visit.calls) == 1, "the duplicate URL across two queries must be visited once"


def _run_live_harvest(mandates, search, visit, *, queries_per_task, visits_per_query, max_searches):
    import asyncio
    return asyncio.run(build_corpus.live_harvest(
        mandates, search, visit,
        queries_per_task=queries_per_task,
        visits_per_query=visits_per_query,
        max_searches=max_searches))
