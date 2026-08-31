"""Offline tests for the variant-agnostic audit layer (testing/audit_layer.py) — free.

Builds synthetic result dicts shaped like each variant's REAL stored output (graph,
evidence_loop, sequential_react/langgraph_react) and checks that the layer recovers exactly what
each variant actually records — including reporting "not recoverable" honestly rather than
reporting an empty set when a variant simply never wrote the data down.
"""
from __future__ import annotations

from agent.app.testing import audit_layer as al


def _graph_result(deliverable, sources, node_urls):
    nodes = {}
    for i, url in enumerate(node_urls):
        nodes[f"n{i}"] = {"details": {"action": "visit", "visit_url": url}}
    nodes["plan0"] = {"details": {"action": "plan"}}  # non-visit node, must be ignored
    return {
        "execution_variant": "graph",
        "execution": {
            "graph": {"nodes": nodes, "edges": []},
            "output": {"final_deliverable": deliverable, "sources": sources},
        },
    }


def _evidence_loop_result(deliverable, extractions):
    return {
        "execution_variant": "evidence_loop",
        "execution": {
            "graph": {"nodes": {}, "edges": []},
            "output": {"final_deliverable": deliverable, "extractions": extractions},
        },
    }


def _seq_or_lg_result(variant, deliverable):
    return {
        "execution_variant": variant,
        "execution": {
            "graph": {"nodes": {}, "edges": []},
            "output": {"final_deliverable": deliverable, "success": True, "goal_achieved": None},
            "observability": {"visit": {"count": 2, "chars": 100, "words": 10, "kilobytes": 0.1}},
        },
    }


# --- extract_cited_urls ---------------------------------------------------

def test_extract_cited_urls_dedupes_and_orders():
    text = ("See https://en.wikipedia.org/wiki/A and https://en.wikipedia.org/wiki/B. "
            "Also https://en.wikipedia.org/wiki/A again.")
    assert al.extract_cited_urls(text) == [
        "https://en.wikipedia.org/wiki/A", "https://en.wikipedia.org/wiki/B",
    ]


def test_extract_cited_urls_strips_trailing_punctuation():
    text = "(see https://example.com/page.) and https://example.com/other),"
    urls = al.extract_cited_urls(text)
    assert urls == ["https://example.com/page", "https://example.com/other"]


def test_extract_cited_urls_empty_input():
    assert al.extract_cited_urls("") == []
    assert al.extract_cited_urls(None) == []


# --- fetched_urls_from_execution -------------------------------------------

def test_fetched_urls_graph_nodes_ignores_non_visit_nodes():
    result = _graph_result("answer", sources=[], node_urls=["https://a.example/x", "https://b.example/y"])
    urls, source = al.fetched_urls_from_execution(result["execution"])
    assert urls == {"https://a.example/x", "https://b.example/y"}
    assert source == al.FETCHED_SOURCE_GRAPH_NODES


def test_fetched_urls_falls_back_to_output_sources_when_no_visit_nodes():
    execution = {
        "graph": {"nodes": {}, "edges": []},
        "output": {"sources": [{"url": "https://a.example/x", "title": ""}]},
    }
    urls, source = al.fetched_urls_from_execution(execution)
    assert urls == {"https://a.example/x"}
    assert source == al.FETCHED_SOURCE_OUTPUT_SOURCES


def test_fetched_urls_evidence_loop_extractions():
    result = _evidence_loop_result("answer", extractions=[
        {"entity": "e1", "field": "f1", "source_url": "https://wiki.example/e1", "quote": "q"},
        {"entity": "e2", "field": "f1", "source_url": "https://wiki.example/e1", "quote": "q2"},
        {"entity": "e3", "field": "f1", "source_url": "https://wiki.example/e3", "quote": "q3"},
    ])
    urls, source = al.fetched_urls_from_execution(result["execution"])
    assert urls == {"https://wiki.example/e1", "https://wiki.example/e3"}
    assert source == al.FETCHED_SOURCE_EXTRACTIONS


def test_fetched_urls_unrecoverable_for_sequential_and_langgraph_shapes():
    for variant in ("sequential_react", "langgraph_react"):
        result = _seq_or_lg_result(variant, "Per https://a.example/x, the answer is 42.")
        urls, source = al.fetched_urls_from_execution(result["execution"])
        assert urls is None
        assert source is None


# --- cited_vs_fetched --------------------------------------------------------

def test_cited_vs_fetched_computes_both_diffs_when_recoverable():
    deliverable = "Cited https://a.example/x and https://c.example/never-fetched."
    execution = {
        "graph": {"nodes": {}, "edges": []},
        "output": {"sources": [
            {"url": "https://a.example/x"}, {"url": "https://b.example/fetched-not-cited"},
        ]},
    }
    report = al.cited_vs_fetched(deliverable, execution)
    assert report["recoverable"] is True
    assert report["cited"] == {"https://a.example/x", "https://c.example/never-fetched"}
    assert report["fetched"] == {"https://a.example/x", "https://b.example/fetched-not-cited"}
    assert report["cited_never_fetched"] == {"https://c.example/never-fetched"}
    assert report["fetched_never_cited"] == {"https://b.example/fetched-not-cited"}


def test_cited_vs_fetched_none_diffs_when_unrecoverable():
    deliverable = "Per https://a.example/x, the answer is 42."
    execution = {"graph": {"nodes": {}, "edges": []}, "output": {}}
    report = al.cited_vs_fetched(deliverable, execution)
    assert report["recoverable"] is False
    assert report["fetched"] is None
    assert report["cited_never_fetched"] is None
    assert report["fetched_never_cited"] is None
    # cited itself is still always recoverable — it comes from the deliverable, not the executor.
    assert report["cited"] == {"https://a.example/x"}


# --- audit_quotes: tri-state, reuses verify_quote/strip_quote_wrapper -------

def test_audit_quotes_true_when_quote_literally_in_page():
    extractions = [{"entity": "e", "field": "f", "source_url": "https://p", "page_id": "p1",
                     "quote": '"the bridge span is 1991 metres"'}]
    pages = {"p1": "Sources state the bridge span is 1991 metres long overall."}
    rows = al.audit_quotes(extractions, pages)
    assert rows[0]["verified"] is True
    assert rows[0]["fail_reason"] is None


def test_audit_quotes_false_when_page_in_hand_and_quote_absent():
    extractions = [{"entity": "e", "field": "f", "source_url": "https://p", "page_id": "p1",
                     "quote": "the span is definitely one mile long"}]
    pages = {"p1": "Sources state the bridge span is 1991 metres long overall."}
    rows = al.audit_quotes(extractions, pages)
    assert rows[0]["verified"] is False
    assert rows[0]["fail_reason"] == "absent"


def test_audit_quotes_none_when_page_text_unavailable_never_reported_as_failed():
    extractions = [{"entity": "e", "field": "f", "source_url": "https://p", "page_id": "p1",
                     "quote": "the span is 1991 metres"}]
    rows = al.audit_quotes(extractions, pages=None)
    assert rows[0]["verified"] is None
    assert rows[0]["verified"] is not False
    assert rows[0]["fail_reason"] == "no_page"


def test_audit_quotes_falls_back_to_source_url_key_when_no_page_id():
    extractions = [{"entity": "e", "field": "f", "source_url": "https://p", "quote": "hello world"}]
    pages = {"https://p": "well, hello world, indeed"}
    rows = al.audit_quotes(extractions, pages)
    assert rows[0]["verified"] is True


def test_audit_quotes_does_not_promote_a_paraphrase_no_fuzzy_matching():
    extractions = [{"entity": "e", "field": "f", "source_url": "https://p", "page_id": "p1",
                     "quote": "roughly two kilometres in length"}]
    pages = {"p1": "The main span measures 1991 metres, or about 1.2 miles."}
    rows = al.audit_quotes(extractions, pages)
    assert rows[0]["verified"] is False


def test_audit_quotes_preserves_stored_verified_for_comparison():
    extractions = [{"entity": "e", "field": "f", "source_url": "https://p", "page_id": "p1",
                     "quote": "x", "quote_verified": True}]
    rows = al.audit_quotes(extractions, pages=None)
    assert rows[0]["stored_verified"] is True
    assert rows[0]["verified"] is None  # this layer's own re-check, independent of the stored flag


# --- audit_result: the full per-cell surface --------------------------------

def test_audit_result_graph_variant_fully_recoverable():
    result = _graph_result(
        "Per https://a.example/x this holds.",
        sources=[{"url": "https://a.example/x"}],
        node_urls=["https://a.example/x"],
    )
    report = al.audit_result(result)
    assert report["variant"] == "graph"
    assert report["recoverable"] is True
    assert report["cited_never_fetched"] == set()
    assert report["quote_audit_applicable"] is False  # graph has no extractions[] structure


def test_audit_result_evidence_loop_variant_quote_audit_applicable():
    result = _evidence_loop_result("Per https://wiki.example/e1 this holds.", extractions=[
        {"entity": "e1", "field": "f1", "source_url": "https://wiki.example/e1",
         "page_id": "p1", "quote": "the value is 42", "quote_verified": True},
    ])
    report = al.audit_result(result)
    assert report["variant"] == "evidence_loop"
    assert report["recoverable"] is True
    assert report["quote_audit_applicable"] is True
    assert len(report["quotes"]) == 1
    assert report["quotes"][0]["verified"] is None  # no page text handed to audit_result here
    assert report["quotes"][0]["stored_verified"] is True


def test_audit_result_evidence_loop_with_page_text_reaudits_independently():
    result = _evidence_loop_result("answer", extractions=[
        {"entity": "e1", "field": "f1", "source_url": "https://wiki.example/e1",
         "page_id": "p1", "quote": "the value is 42", "quote_verified": True},
    ])
    report = al.audit_result(result, pages={"p1": "we confirm the value is 42 exactly"})
    assert report["quotes"][0]["verified"] is True


def test_audit_result_sequential_and_langgraph_report_unrecoverable_fetch():
    for variant in ("sequential_react", "langgraph_react"):
        result = _seq_or_lg_result(variant, "Per https://a.example/x, the answer is 42.")
        report = al.audit_result(result)
        assert report["variant"] == variant
        assert report["recoverable"] is False
        assert report["fetched"] is None
        assert report["cited"] == {"https://a.example/x"}
        assert report["quote_audit_applicable"] is False
