"""The storyboard: one run projected into drawable beats.

These tests pin the properties a renderer is allowed to rely on -- a closed kind vocabulary, a
truthful narrative order, cumulative rail state, and every beat traceable back to the run -- plus
the honesty rules that make the deck worth showing at all: the tri-state survives, absent is never
zero, and a refusal is a beat rather than an omission.

Structural claims are exercised against the REAL stored cells in ``agent/idea_test_results`` as
well as fixtures, following ``ledger_trace_test.py``: a fixture only proves the projector handles
the shape its author imagined. No network, no model.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from agent.app.testing.ledger_story import (
    BEAT_KINDS,
    KIND_AUDIT,
    KIND_DERIVE,
    KIND_LOCATE,
    KIND_MINT,
    KIND_QUESTION,
    KIND_RANK,
    KIND_REFUSE,
    KIND_SEARCH,
    KIND_VERDICT,
    KIND_VISIT,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_REFUSED,
    STATUS_UNKNOWN,
    Beat,
    LedgerSnapshot,
    build,
    build_file,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "idea_test_results")


def _cell(**output):
    """A minimal stored-cell shape carrying one output block."""
    return {
        "model": "test-model",
        "execution_variant": "langgraph_react",
        "test_metadata": {"test_id": "999"},
        "validation": {"overall_score": 0.5},
        "execution": {"telemetry_raw": {"mandate": "How tall is it?", "duration": 1.0, "timings": []},
                      "output": output},
    }


PAGE = {"page_id": "p1", "url": "https://en.wikipedia.org/wiki/Tower",
        "chars": 60, "stored_chars": 60, "truncated": False,
        "text": "The tower reaches a roof height of 508.2 m above the plaza."}


def _source(node_id="n1", value="508.2", start=35, end=40, **kw):
    node = {"id": node_id, "kind": "source", "value": value, "page_id": "p1",
            "source_url": PAGE["url"], "start": start, "end": end, "quote": value,
            "quote_verified": True, "verified": True, "occurrences": 1, "unit": "m",
            "unit_bearing": True, "input_ids": [], "operation": ""}
    node.update(kw)
    return node


# --------------------------------------------------------------------- vocabulary and order

def test_every_beat_kind_is_in_the_closed_vocabulary():
    """A renderer switches on ``kind``; an unlisted kind would be drawn by no branch."""
    story = build_file(_a_real_cell())
    assert story.beats, "the sample cell should produce beats"
    for beat in story.beats:
        assert beat.kind in BEAT_KINDS, f"beat {beat.index} has unlisted kind {beat.kind!r}"


def test_story_opens_on_the_question_and_closes_on_the_verdict():
    story = build_file(_a_real_cell())
    assert story.beats[0].kind == KIND_QUESTION
    assert story.beats[-1].kind == KIND_VERDICT


def test_beat_indexes_are_dense_and_ordered():
    story = build_file(_a_real_cell())
    assert [b.index for b in story.beats] == list(range(len(story.beats)))


def test_a_span_is_never_located_before_its_page_is_fetched():
    """The nesting is a fact about the run: you cannot read a span off a page you have not got."""
    story = build_file(_a_real_cell())
    seen_pages = set()
    for beat in story.beats:
        if beat.kind == KIND_VISIT and beat.target:
            seen_pages.add(beat.target)
        if beat.kind == KIND_MINT:
            page_id = beat.detail.get("page_id")
            assert page_id in seen_pages, f"beat {beat.index} mints off unfetched page {page_id!r}"


# --------------------------------------------------------------------- honesty rules

def test_unchecked_quote_is_not_reported_as_a_failed_one():
    """``verified is None`` means the check never ran. Folding it into ``False`` would report a
    check that never happened as a check that failed."""
    story = build(_cell(evidence_graph={"pages": [PAGE], "nodes": [_source(verified=None)],
                                        "rejections": [], "derivation_refusals": []}))
    mint = story.of_kind(KIND_MINT)[0]
    assert mint.status == STATUS_UNKNOWN
    assert mint.detail["verified"] is None
    assert mint.ledger_after.counts["unchecked"] == 1
    assert mint.ledger_after.counts["verified"] == 0


def test_absent_value_is_reported_as_an_error_not_as_unknown():
    story = build(_cell(evidence_graph={"pages": [PAGE], "nodes": [_source(verified=False)],
                                        "rejections": [], "derivation_refusals": []}))
    mint = story.of_kind(KIND_MINT)[0]
    assert mint.status == STATUS_ERROR
    assert mint.ledger_after.counts["unverified"] == 1


def test_a_beat_with_no_interval_has_no_duration_rather_than_a_zero_one():
    """Derivations live on the evidence plane and cost no time. ``0.0s`` would be a lie."""
    story = build(_cell(evidence_graph={
        "pages": [PAGE],
        "nodes": [_source(), _source("n2", "94", 0, 2, unit="count"),
                  {"id": "d1", "kind": "derived", "value": "5.4", "operation": "quotient",
                   "input_ids": ["n1", "n2"], "derivation_valid": True, "unit": ""}],
        "rejections": [], "derivation_refusals": []}))
    derive = story.of_kind(KIND_DERIVE)[0]
    assert derive.duration is None
    assert "t_start" not in derive.as_dict() and "t_end" not in derive.as_dict()


def test_a_refusal_is_a_beat_and_carries_its_typed_code_and_message():
    """The Ledger declining to assert is the behaviour the subsystem exists to produce. A demo
    that silently dropped it would be showing a different, better-behaved system."""
    story = build(_cell(evidence_graph={
        "pages": [PAGE], "nodes": [], "rejections": [],
        "derivation_refusals": [{"operation": "max", "input_ids": [], "code": "INCOMPLETE_ROSTER",
                                 "message": "max over 5 entities, 3 resolved: no operands for X, Y"}]}))
    refuse = story.of_kind(KIND_REFUSE)
    assert len(refuse) == 1
    assert refuse[0].status == STATUS_REFUSED
    assert refuse[0].detail["code"] == "INCOMPLETE_ROSTER"
    assert "no operands for X, Y" in refuse[0].detail["message"]
    assert refuse[0].ledger_after.counts["refusals"] == 1


def test_audit_names_the_condition_that_blocked_support():
    """``answer_supported`` is a conjunction. Reporting only the backing term produces a frame
    that reads "all backed" beside "not supported" -- which looks like a bug in the system rather
    than the ambiguity finding it actually is."""
    story = build(_cell(answer_audit={
        "numbers_total": 2, "answer_supported": False,
        "numbers": [{"text": "5.47", "value": 5.47, "unit": "", "status": "derived", "trivial": False,
                     "ambiguity": 4, "unit_consistent": None, "page_id": "p1", "node_id": "d1"},
                    {"text": "508.2", "value": 508.2, "unit": "m", "status": "backed", "trivial": False,
                     "ambiguity": 0, "unit_consistent": True, "page_id": "p1", "node_id": "n1"}]}))
    audit = story.of_kind(KIND_AUDIT)[0]
    assert audit.detail["blockers"] == {"unbacked": 0, "unit inconsistent": 0, "ambiguously backed": 1}
    assert "ambiguously backed" in audit.target
    assert "2 of 2 non-trivial numbers backed" == audit.subject


def test_a_losing_operand_slot_says_why_it_lost():
    """"below_min_score" and "no_candidate_page" are different failures with different fixes."""
    story = build(_cell(host_derive={
        "reason": "operand_not_found", "min_score": 0.93,
        "slots": [{"index": 0, "entity": "Tower", "field_phrase": "its height", "page_id": "p1",
                   "url": PAGE["url"], "entry": None, "score": 0.12, "reason": "below_min_score"}]}))
    rank = story.of_kind(KIND_RANK)[0]
    assert rank.status == STATUS_REFUSED
    assert "below_min_score" in rank.subject
    assert rank.detail["min_score"] == 0.93
    assert rank.detail["score"] == 0.12


# --------------------------------------------------------------------- the drawable properties

def test_locate_carries_the_span_and_the_text_either_side_of_it():
    """The window is the one place page text is allowed through, because showing the span in
    context IS the beat. It is handed over as three strings so the renderer picks its own
    emphasis rather than parsing markup back out."""
    story = build(_cell(evidence_graph={"pages": [PAGE], "nodes": [_source()],
                                        "rejections": [], "derivation_refusals": []}))
    window = story.of_kind(KIND_LOCATE)[0].detail["window"]
    assert window["span"] == "508.2"
    assert window["before"].endswith("roof height of ")
    assert window["after"].startswith(" m above")
    assert PAGE["text"][window["start"]:window["end"]] == window["span"]


def test_handles_are_named_in_admission_order_like_the_ledger_names_them():
    story = build(_cell(evidence_graph={
        "pages": [PAGE],
        "nodes": [_source("n1", "508.2", 35, 40), _source("n2", "tower", 4, 9, unit=""),
                  {"id": "d1", "kind": "derived", "value": "5.4", "operation": "quotient",
                   "input_ids": ["n1", "n2"], "derivation_valid": True, "unit": ""}],
        "rejections": [], "derivation_refusals": []}))
    handles = story.beats[-1].ledger_after.handles
    assert [h.ref for h in handles] == ["E1", "E2", "D1"]
    # Spans are minted in DOCUMENT order, so the span earlier on the page is E1 regardless of the
    # order the evidence graph happens to list its nodes in. That is what makes the rail readable
    # beside the highlighted page: the handles count down the page.
    assert [h.value for h in handles] == ["tower", "508.2", "5.4"]
    derive = story.of_kind(KIND_DERIVE)[0]
    assert derive.origin == "quotient(E2, E1)", "a derivation names its operands by handle, not by hash"


def test_the_rail_only_ever_grows():
    """Cumulative snapshots are what let a renderer draw frame n without replaying the story."""
    story = build_file(_a_real_cell())
    previous = 0
    for beat in story.beats:
        assert len(beat.ledger_after.handles) >= previous
        previous = len(beat.ledger_after.handles)


def test_every_beat_is_serializable_and_omits_absent_fields():
    story = build_file(_a_real_cell())
    payload = story.as_dict()
    json.dumps(payload)  # must not raise
    for row in payload["beats"]:
        assert "subject" not in row or row["subject"] != ""
        assert row["status"] in ("ok", "error", "empty", "refused", "invalid", "unknown")


# --------------------------------------------------------------------- degenerate inputs

@pytest.mark.parametrize("cell", [None, {}, {"execution": None}, {"execution": {"output": None}}])
def test_a_cell_with_nothing_in_it_still_projects(cell):
    """A run that found nothing is a result. Refusing to draw it would hide the case most worth
    seeing."""
    story = build(cell)
    assert len(story) >= 1
    assert story.beats[0].kind == KIND_QUESTION


# --------------------------------------------------------------------- the real corpus

def _stored_cells(limit):
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    paths = [p for p in paths if not p.endswith(("_summary.json", "_report_v3.json"))]
    return paths[:limit]


def _a_real_cell():
    """One stored cell with a populated evidence graph, for the structural tests."""
    target = os.path.join(RESULTS_DIR, "mint04_gq7b_221_qwen2.5:7b_langgraph_react_cfgfbb154cf_r1.json")
    if os.path.exists(target):
        return target
    for path in _stored_cells(400):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cell = json.load(handle)
        except Exception:
            continue
        graph = (((cell.get("execution") or {}).get("output") or {}).get("evidence_graph")) or {}
        if graph.get("nodes"):
            return path
    pytest.skip("no stored cell with an evidence graph available")


@pytest.mark.skipif(not os.path.isdir(RESULTS_DIR), reason="no stored results on this machine")
def test_projects_every_stored_cell_without_raising():
    """The projector must survive every shape already on disk, not only the ones a fixture
    author imagined."""
    checked = 0
    for path in _stored_cells(120):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cell = json.load(handle)
        except Exception:
            continue
        story = build(cell, source_file=path)
        assert story.beats[0].kind == KIND_QUESTION
        for beat in story.beats:
            assert beat.kind in BEAT_KINDS
            assert isinstance(beat.ledger_after, LedgerSnapshot)
        checked += 1
    assert checked > 0, "expected at least one readable stored cell"


@pytest.mark.skipif(not os.path.isdir(RESULTS_DIR), reason="no stored results on this machine")
def test_the_sample_cell_tells_the_whole_story():
    """The flagship demo cell must exercise every narrative stage, or it is the wrong demo."""
    story = build_file(_a_real_cell())
    counts = story.counts()
    for kind in (KIND_QUESTION, KIND_SEARCH, KIND_VISIT, KIND_LOCATE, KIND_MINT, KIND_DERIVE, KIND_VERDICT):
        assert counts.get(kind), f"the sample run has no {kind} beat"
