"""The off-the-shelf arm must PERSIST the pages it read, not merely the fact that it read them.

Measured on the stored corpus before this change: of 40 sampled ``langgraph_react`` cells, 36 had
``observability.visit.count > 0`` but ZERO carried recoverable page text -- ``documents_seen`` is
stripped by ``testing/utils.slim_telemetry_raw`` before the result JSON is written, this arm never
emitted ``output["pages"]``, and it never populates ``result["graph"]``. The archive therefore
records THAT the arm visited and never WHAT it read, which makes any post-hoc claim-grounding,
arm-symmetric verdict or risk-coverage curve impossible for it -- not hard, impossible.

``evidence_loop`` and ``sequential_react_extract`` already freeze their pages this way; this
brings the third arm onto the same contract so a three-arm comparison can exist at all.
"""
from __future__ import annotations

from agent.app.testing.execution_langgraph import pages_from_telemetry


class _Telemetry:
    def __init__(self, documents_seen):
        self.documents_seen = documents_seen


def _visit(url, content):
    return {"source": "visit", "document": {"url": url, "content": content}}


def _search(url):
    return {"source": "search", "document": {"url": url, "title": "t", "description": "d"}}


def test_visited_documents_become_frozen_pages():
    tel = _Telemetry([_visit("https://example.org/a", "alpha body text")])
    pages = pages_from_telemetry(tel, max_chars=6000)
    assert len(pages) == 1
    page = pages[0]
    assert page["url"] == "https://example.org/a"
    assert page["text"] == "alpha body text"
    assert page["page_id"] == "p1"
    assert page["content_hash"] and page["chars"] == len("alpha body text")


def test_search_results_are_not_mistaken_for_visited_pages():
    tel = _Telemetry([_search("https://example.org/s"), _visit("https://example.org/a", "body")])
    pages = pages_from_telemetry(tel, max_chars=6000)
    assert [p["url"] for p in pages] == ["https://example.org/a"]


def test_the_same_url_visited_twice_is_frozen_once():
    tel = _Telemetry([_visit("https://example.org/a", "body one"),
                      _visit("https://example.org/a", "body one")])
    assert len(pages_from_telemetry(tel, max_chars=6000)) == 1


def test_page_ids_are_sequential_and_stable():
    tel = _Telemetry([_visit("https://example.org/a", "one"),
                      _visit("https://example.org/b", "two")])
    assert [p["page_id"] for p in pages_from_telemetry(tel, max_chars=6000)] == ["p1", "p2"]


def test_the_stored_window_is_capped_but_the_hash_covers_the_whole_text():
    body = "x" * 500
    pages = pages_from_telemetry(_Telemetry([_visit("https://e.org/a", body)]), max_chars=100)
    page = pages[0]
    assert page["stored_chars"] == 100 and page["chars"] == 500
    assert page["truncated"] is True
    from agent.app.testing.execution_evidence_loop import store_page
    assert page["content_hash"] == store_page("p1", "https://e.org/a", body, 100)["content_hash"]


def test_an_empty_or_contentless_visit_is_not_frozen():
    tel = _Telemetry([_visit("https://example.org/a", ""), _visit("", "orphan body")])
    assert pages_from_telemetry(tel, max_chars=6000) == []


def test_no_telemetry_yields_no_pages_rather_than_raising():
    assert pages_from_telemetry(None, max_chars=6000) == []
    assert pages_from_telemetry(_Telemetry([]), max_chars=6000) == []


def test_the_frozen_pages_are_what_visited_evidence_can_read_back():
    """The whole point: `visited_evidence` must recover this arm's evidence from a STORED cell."""
    from agent.app.idea_test_utils import visited_evidence
    tel = _Telemetry([_visit("https://example.org/a", "alpha body")])
    stored_cell_execution = {"output": {"pages": pages_from_telemetry(tel, max_chars=6000)}}
    evidence = visited_evidence(stored_cell_execution, None)
    assert [e["url"] for e in evidence] == ["https://example.org/a"]
    assert "alpha body" in evidence[0]["content"]
