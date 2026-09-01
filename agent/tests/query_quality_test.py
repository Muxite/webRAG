"""Tests for the search-behaviour metric (`scripts/query_quality.py`).

Measured on the `ledgernum22` campaign: high-scoring `evidence_loop` cells averaged 2.60 search
queries, low-scoring ones 5.75 -- bad searching is real and expensive. But the naive signal is a
trap. Of five flagged near-duplicate query pairs, three were
``"Eiffel Tower original construction cost historical record"`` ->
``"Statue of Liberty ... historical record"`` -> ``"Great Pyramid ... historical record"``: the
SAME template applied to DIFFERENT entities, which is exactly correct on a comparison task.
Penalising string similarity would punish correct breadth.

So the signal is yield-conditioned instead: a requery issued after a search that produced nothing
is adaptation, and a requery issued when the previous search already produced a usable visit is
thrash. This module only MEASURES -- nothing here gates or scores.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import query_quality as qq


def _cell(timings, variant="evidence_loop", score=1.0, test_id="210"):
    return {
        "execution_variant": variant,
        "validation": {"overall_score": score},
        "execution": {"telemetry_raw": {"timings": timings}},
        "test_metadata": {"id": test_id},
    }


def _search(q, ok=True):
    return {"name": "search", "success": ok, "payload": {"query": q}}


def _visit(url, ok=True):
    return {"name": "visit", "success": ok, "payload": {"url": url}}


def _llm():
    return {"name": "llm_call", "success": True, "payload": {}}


class TestEventExtraction:
    def test_searches_and_visits_are_read_in_order(self):
        events = qq.search_events(_cell([_search("a"), _llm(), _visit("u1"), _search("b")]))
        assert [(e.kind, e.text) for e in events] == [
            ("search", "a"), ("visit", "u1"), ("search", "b")]

    def test_a_cell_with_no_telemetry_yields_nothing_rather_than_raising(self):
        assert qq.search_events({"execution": {}}) == []
        assert qq.search_events({}) == []

    def test_a_failed_visit_is_recorded_but_marked_unsuccessful(self):
        events = qq.search_events(_cell([_visit("u1", ok=False)]))
        assert events[0].success is False


class TestYieldConditionedRequery:
    def test_a_requery_after_a_dry_search_is_adaptation(self):
        """No visit happened after the first search, so reformulating is the right move."""
        m = qq.query_quality(_cell([_search("a"), _search("b")]))
        assert m["requery_after_dry"] == 1 and m["requery_after_yield"] == 0

    def test_a_requery_after_a_search_that_already_yielded_is_thrash(self):
        m = qq.query_quality(_cell([_search("a"), _visit("u1"), _search("b")]))
        assert m["requery_after_yield"] == 1 and m["requery_after_dry"] == 0

    def test_the_first_search_is_never_a_requery(self):
        m = qq.query_quality(_cell([_search("a")]))
        assert m["requery_after_dry"] == 0 and m["requery_after_yield"] == 0
        assert m["queries"] == 1

    def test_a_failed_visit_does_not_count_as_yield(self):
        m = qq.query_quality(_cell([_search("a"), _visit("u1", ok=False), _search("b")]))
        assert m["requery_after_dry"] == 1

    def test_template_reuse_across_entities_is_not_penalised(self):
        """The trap: same template, different entity, each search yielding its own page. This is
        correct breadth on a comparison task and must NOT read as thrash."""
        m = qq.query_quality(_cell([
            _search("Eiffel Tower original construction cost historical record"), _visit("u1"),
            _search("Statue of Liberty original construction cost historical record"), _visit("u2"),
            _search("Great Pyramid original construction cost historical record"), _visit("u3"),
        ]))
        assert m["requery_after_yield"] == 2, "these ARE after-yield by the raw rule"
        assert m["distinct_urls"] == 3 and m["redundant_visits"] == 0
        assert m["productive_requeries"] == 2, (
            "a requery that goes on to open a NEW page is productive, not thrash")
        assert m["thrash_requeries"] == 0

    def test_requerying_the_same_entity_and_reopening_one_page_is_thrash(self):
        """Task 227's real trace: search -> visit page -> search -> search -> visit SAME page."""
        m = qq.query_quality(_cell([
            _search("Eiffel Tower construction cost"), _visit("u1"),
            _search("historical construction cost of the Eiffel Tower"),
            _search("historical construction cost of the Eiffel Tower 1889"), _visit("u1"),
        ]))
        assert m["redundant_visits"] == 1
        assert m["thrash_requeries"] >= 1
        assert m["productive_requeries"] == 0


class TestRedundantVisits:
    def test_revisiting_a_url_is_counted(self):
        m = qq.query_quality(_cell([_visit("u1"), _visit("u1"), _visit("u2")]))
        assert m["visits"] == 3 and m["distinct_urls"] == 2 and m["redundant_visits"] == 1

    def test_a_failed_revisit_is_not_counted_as_redundant(self):
        m = qq.query_quality(_cell([_visit("u1"), _visit("u1", ok=False)]))
        assert m["redundant_visits"] == 0


class TestReporting:
    def test_the_metric_never_gates_or_scores(self):
        """This module reports. It must expose no pass/fail and no score adjustment -- the signal
        has not been shown to track ground truth yet, and the repo has been bitten by an inverted
        gate before (the roster gate blocked 46 of 48 eligible cells)."""
        m = qq.query_quality(_cell([_search("a")]))
        assert not any(k in m for k in ("passed", "gate", "penalty", "score_delta"))

    def test_cells_aggregate_by_arm(self):
        cells = [_cell([_search("a"), _search("b")], variant="evidence_loop", score=0.2),
                 _cell([_search("a"), _visit("u")], variant="evidence_loop", score=0.9),
                 _cell([_search("a")], variant="langgraph_react", score=0.5)]
        summary = qq.summarize([qq.query_quality(c) for c in cells])
        assert set(summary) == {"evidence_loop", "langgraph_react"}
        assert summary["evidence_loop"]["cells"] == 2
        assert summary["evidence_loop"]["mean_queries"] == 1.5

    def test_summary_is_empty_for_no_cells(self):
        assert qq.summarize([]) == {}
