"""Offline tests for the ``evidence_loop`` variant (testing/execution_evidence_loop.py) — free.

The arm is ReAct's winning loop (one flat step loop, one JSON decision per step, a 12-step
scratchpad window, whole-page observations) plus exactly four additions: a never-expiring ledger
sidecar, per-hop typed extraction, mechanical quote-offset grounding, and table-first
finalization. These tests pin each addition AND pin that the base loop still behaves like the
sequential control it was copied from.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_evidence_loop as el


ENUMERATED = (
    "Which of these rivers empties into the English Channel?\n"
    "1. River Avon, Bristol — flows west\n"
    "2. River Avon, Hampshire — flows south\n"
    "3. River Avon, Warwickshire — flows into the Severn"
)


def _io(replies, page_text="PAGE CONTENT", synth="SYNTH ANSWER"):
    """A mocked ``AgentIO``. ``replies`` are the LLM replies IN CALL ORDER — step decisions and
    the per-hop extraction replies interleaved exactly as the loop consumes them."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=lambda **kw: {"messages": kw.get("messages", [])})
    encoded = [r if isinstance(r, str) else json.dumps(r) for r in replies]
    io.query_llm = AsyncMock(side_effect=[*encoded, synth, synth, synth])
    io.search = AsyncMock(return_value=[
        {"title": "Avon", "url": "https://en.wikipedia.org/wiki/River_Avon", "description": "river"}])
    io.visit = AsyncMock(return_value=page_text)
    return io


def _extraction(entity, field, value, quote, verdict="SUPPORTED"):
    return json.dumps({"extractions": [
        {"entity": entity, "field": field, "value": value, "verdict": verdict, "quote": quote}]})


# --------------------------------------------------------------------------------------
# row minting — routes on COUNT, never on shape
# --------------------------------------------------------------------------------------


def test_enumerated_mandate_mints_one_row_per_candidate():
    rows = el.mint_rows(ENUMERATED)
    assert len(rows) == 3
    assert any("Bristol" in r.entity for r in rows)
    assert all(r.status == el.STATUS_OPEN for r in rows)


def test_unenumerated_mandate_mints_exactly_one_row():
    rows = el.mint_rows("Who wrote Beloved and where did she get her MA?")
    assert len(rows) == 1


def test_roster_is_capped():
    mandate = "Compare:\n" + "\n".join(f"{i}. Candidate {i}" for i in range(1, 30))
    assert len(el.mint_rows(mandate)) == el._MAX_ROWS


def test_empty_mandate_still_mints_one_row():
    assert len(el.mint_rows("")) == 1


# --------------------------------------------------------------------------------------
# the ledger never expires, the scratchpad does
# --------------------------------------------------------------------------------------


def test_ledger_rows_survive_past_the_scratchpad_window():
    ledger = el.Ledger.mint(ENUMERATED)
    scratchpad = [f"STEP {i}: observation=obs-{i}" for i in range(1, 21)]
    prompt = el.compose_user_prompt("task", ledger, scratchpad)
    assert "obs-1\n" not in prompt and "obs-8" not in prompt   # outside the 12-step window
    assert "obs-20" in prompt                                  # inside it
    for row in ledger.rows:
        assert row.entity[:20] in prompt


def test_a_ledger_row_renders_far_shorter_than_the_observation_it_replaces():
    ledger = el.Ledger.mint(ENUMERATED)
    ledger.rows[0].status = el.STATUS_SUPPORTED
    ledger.rows[0].value = "English Channel"
    ledger.rows[0].source_url = "https://en.wikipedia.org/wiki/River_Avon,_Hampshire"
    assert len(ledger.rows[0].render()) < el._UNCAPPED_OBSERVATION_CHARS / 4


# --------------------------------------------------------------------------------------
# quote-offset grounding — mechanical, no model judgment
# --------------------------------------------------------------------------------------


def test_a_literal_quote_verifies_with_offsets():
    page = "The River Avon in Hampshire empties into the English Channel at Christchurch."
    match = el.verify_quote(page, "empties into the English Channel")
    assert match.verified is True
    assert page[match.start:match.end] == "empties into the English Channel"


def test_a_quote_absent_from_the_page_does_not_verify():
    match = el.verify_quote("The river flows west into the Severn.", "empties into the English Channel")
    assert match.verified is False
    assert (match.start, match.end) == (-1, -1)


def test_a_quote_differing_only_in_whitespace_still_verifies():
    page = "empties  into\nthe English   Channel"
    match = el.verify_quote(page, "empties into the English Channel")
    assert match.verified is True
    assert match.start == 0 and match.end == len(page)


def test_an_empty_quote_is_unchecked_not_failed():
    match = el.verify_quote("some page", "   ")
    assert match.verified is None
    assert match.fail_reason == el.QUOTE_FAIL_EMPTY


# --------------------------------------------------------------------------------------
# wrapper punctuation the MODEL added is stripped; the quote's CONTENT never is
# --------------------------------------------------------------------------------------

_PAGE = "Longest span: 1,991 metres (6,532 ft) — Akashi Kaikyo Bridge, Japan."


def test_a_quote_wrapped_in_straight_double_quotes_verifies():
    match = el.verify_quote(_PAGE, '"Longest span: 1,991 metres (6,532 ft)"')
    assert match.verified is True
    assert _PAGE[match.start:match.end] == "Longest span: 1,991 metres (6,532 ft)"


def test_a_quote_wrapped_in_curly_double_quotes_verifies():
    match = el.verify_quote(_PAGE, "\u201cLongest span: 1,991 metres (6,532 ft)\u201d")
    assert match.verified is True


def test_a_quote_wrapped_in_curly_single_quotes_verifies():
    match = el.verify_quote(_PAGE, "\u2018Longest span: 1,991 metres (6,532 ft)\u2019")
    assert match.verified is True


def test_a_quote_wrapped_in_straight_single_quotes_verifies():
    match = el.verify_quote(_PAGE, "'Longest span: 1,991 metres (6,532 ft)'")
    assert match.verified is True


def test_surrounding_whitespace_is_stripped():
    match = el.verify_quote(_PAGE, "\n   Longest span: 1,991 metres (6,532 ft)  \n")
    assert match.verified is True


def test_wrapper_quotes_combined_with_line_wrapping_verify():
    page = "Longest span\n1,624 metres (5,328\nft)\nHumber Bridge"
    match = el.verify_quote(page, '"Longest span 1,624 metres (5,328 ft)"')
    assert match.verified is True


def test_nested_wrappers_are_stripped():
    match = el.verify_quote(_PAGE, '"\'Longest span: 1,991 metres (6,532 ft)\'"')
    assert match.verified is True


def test_an_unmatched_leading_quote_character_is_not_stripped():
    # A lone quote char is content, not a wrapper: stripping it would invent a match.
    page = 'He said "Longest span" once.'
    match = el.verify_quote(page, '"Longest span')
    assert match.verified is True
    assert page[match.start:match.end] == '"Longest span'


def test_a_quote_that_is_only_wrapper_characters_is_unchecked():
    match = el.verify_quote(_PAGE, '""')
    assert match.verified is None
    assert match.fail_reason == el.QUOTE_FAIL_EMPTY


def test_a_page_quote_that_is_genuinely_wrapped_on_the_page_still_verifies_verbatim():
    page = 'The plaque reads "Longest span: 1,991 metres" in bronze.'
    match = el.verify_quote(page, '"Longest span: 1,991 metres"')
    assert match.verified is True
    assert page[match.start:match.end] == '"Longest span: 1,991 metres"'


def test_a_paraphrase_of_a_true_fact_never_verifies():
    # LOAD-BEARING. The value below is factually correct and the page states it in other words.
    # Quote verification exists precisely to fail this: the model composed a sentence instead of
    # copying one. NEVER relax this by adding fuzzy / token-overlap / edit-distance matching --
    # the pass rate is a measurement, not a target.
    match = el.verify_quote(_PAGE, "The longest span is 1,991 metres (6,532 ft).")
    assert match.verified is False
    assert match.fail_reason == el.QUOTE_FAIL_ABSENT
    assert (match.start, match.end) == (-1, -1)


def test_a_quote_with_no_page_text_is_unchecked_not_failed():
    match = el.verify_quote("", "Longest span: 1,991 metres")
    assert match.verified is None
    assert match.fail_reason == el.QUOTE_FAIL_NO_PAGE


def test_an_absent_quote_records_the_absent_reason():
    match = el.verify_quote(_PAGE, "spans the Golden Gate")
    assert match.verified is False
    assert match.fail_reason == el.QUOTE_FAIL_ABSENT


def test_a_verified_quote_records_no_fail_reason():
    assert el.verify_quote(_PAGE, "Akashi Kaikyo Bridge").fail_reason is None


# --------------------------------------------------------------------------------------
# the fail reason reaches the extraction record and the result payload
# --------------------------------------------------------------------------------------


def _extract_with(quote, page="The River Avon, Bristol flows west into the Severn Estuary."):
    ledger = el.Ledger.mint(ENUMERATED)
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Bristol", ledger.rows[0].field, "Severn", quote))
    records = asyncio.run(el.extract_from_page(
        io, "m", ENUMERATED, ledger, page_id="p1", page_url="https://example.com",
        page_text=page))
    return ledger, records


def test_a_wrapped_quote_promotes_the_row_after_the_wrapper_is_stripped():
    ledger, records = _extract_with('"flows west into the Severn Estuary"')
    assert records[0].quote_verified is True
    assert records[0].quote_fail_reason is None
    assert ledger.find("River Avon, Bristol").status == el.STATUS_SUPPORTED


def test_a_paraphrased_quote_resolves_the_row_but_leaves_it_unverified():
    # The two axes are separate: the model DID read a value off the page (resolution), and its
    # supporting sentence is not on that page (verification). Neither answer is thrown away.
    ledger, records = _extract_with("The Avon flows westward towards the Severn Estuary.")
    assert records[0].quote_verified is False
    assert records[0].quote_fail_reason == el.QUOTE_FAIL_ABSENT
    row = ledger.find("River Avon, Bristol")
    assert row.status == el.STATUS_SUPPORTED
    assert row.value == "Severn"
    assert row.quote_verified is False
    assert row.confidence_tier == el.TIER_RESOLVED_UNVERIFIED


def test_an_empty_quote_is_recorded_unchecked_and_resolves_the_row_unverified():
    ledger, records = _extract_with("")
    assert records[0].quote_verified is None
    assert records[0].quote_fail_reason == el.QUOTE_FAIL_EMPTY
    row = ledger.find("River Avon, Bristol")
    assert row.status == el.STATUS_SUPPORTED
    assert row.quote_verified is False
    assert row.confidence_tier == el.TIER_RESOLVED_UNVERIFIED
    assert records[0].as_dict()["quote_fail_reason"] == el.QUOTE_FAIL_EMPTY


def test_the_result_payload_splits_the_three_causes():
    ledger = el.Ledger.mint(ENUMERATED)
    ledger.extractions = [
        el.Extraction("a", "f", "v", "SUPPORTED", "u", "q", True, "p1", 0, 1),
        el.Extraction("b", "f", "v", "SUPPORTED", "u", "q", False, "p1",
                      quote_fail_reason=el.QUOTE_FAIL_ABSENT),
        el.Extraction("c", "f", "v", "SUPPORTED", "u", "", None, "p1",
                      quote_fail_reason=el.QUOTE_FAIL_EMPTY),
        el.Extraction("d", "f", "v", "SUPPORTED", "u", "q", None, "p1",
                      quote_fail_reason=el.QUOTE_FAIL_NO_PAGE),
    ]
    counts = el.quote_verification_counts(ledger)
    assert counts == {"verified": 1, "failed": 1, "unchecked": 2,
                      "absent": 1, "empty": 1, "no_page": 1}


# --------------------------------------------------------------------------------------
# per-hop typed extraction
# --------------------------------------------------------------------------------------


def test_a_page_yielding_a_value_marks_the_row_supported_with_a_verified_quote():
    ledger = el.Ledger.mint(ENUMERATED)
    page = "The River Avon, Hampshire empties into the English Channel at Christchurch."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Hampshire", ledger.rows[1].field, "English Channel",
        "empties into the English Channel"))

    records = asyncio.run(el.extract_from_page(
        io, "m", ENUMERATED, ledger, page_id="p1",
        page_url="https://en.wikipedia.org/wiki/River_Avon,_Hampshire", page_text=page))

    assert len(records) == 1
    rec = records[0]
    assert rec.quote_verified is True
    assert page[rec.quote_start:rec.quote_end] == "empties into the English Channel"
    assert rec.source_url.endswith("Hampshire")
    row = ledger.find("River Avon, Hampshire")
    assert row.status == el.STATUS_SUPPORTED
    assert row.value == "English Channel"


def test_a_quote_not_on_the_page_resolves_the_row_and_flags_it_unverified():
    ledger = el.Ledger.mint(ENUMERATED)
    page = "The River Avon, Bristol flows west into the Severn Estuary."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Bristol", ledger.rows[0].field, "English Channel",
        "empties into the English Channel"))

    records = asyncio.run(el.extract_from_page(
        io, "m", ENUMERATED, ledger, page_id="p1",
        page_url="https://en.wikipedia.org/wiki/River_Avon,_Bristol", page_text=page))

    assert records[0].quote_verified is False
    row = ledger.find("River Avon, Bristol")
    assert row.status == el.STATUS_SUPPORTED
    assert row.confidence_tier == el.TIER_RESOLVED_UNVERIFIED


def test_an_absent_verdict_marks_the_row_absent():
    ledger = el.Ledger.mint(ENUMERATED)
    io = _io([], page_text="unrelated")
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Warwickshire", ledger.rows[2].field, "", "", verdict="ABSENT"))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                     page_url="https://example.com", page_text="unrelated"))
    assert ledger.find("River Avon, Warwickshire").status == el.STATUS_ABSENT


def _supported(ledger, index, value, url="https://a.example"):
    row = ledger.rows[index]
    row.status, row.value, row.source_url = el.STATUS_SUPPORTED, value, url
    row.quote, row.quote_verified = f"{value} quote", True


def test_a_blocked_verdict_marks_the_row_blocked():
    ledger = el.Ledger.mint(ENUMERATED)
    io = _io([], page_text="Please log in to continue.")
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Bristol", ledger.rows[0].field, "", "", verdict="BLOCKED"))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                     page_url="https://example.com",
                                     page_text="Please log in to continue."))
    assert ledger.find("River Avon, Bristol").status == el.STATUS_BLOCKED


def test_a_supported_row_is_not_demoted_by_a_later_absent_page():
    ledger = el.Ledger.mint(ENUMERATED)
    _supported(ledger, 0, "English Channel")
    io = _io([], page_text="unrelated")
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Bristol", ledger.rows[0].field, "", "", verdict="ABSENT"))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p2",
                                     page_url="https://example.com", page_text="unrelated"))
    assert ledger.find("River Avon, Bristol").status == el.STATUS_SUPPORTED


def test_a_second_differing_verified_value_conflicts_the_row():
    ledger = el.Ledger.mint(ENUMERATED)
    page_a = "The River Avon, Bristol empties into the Severn Estuary."
    page_b = "The River Avon, Bristol empties into the English Channel."
    io = _io([], page_text=page_a)
    io.query_llm = AsyncMock(side_effect=[
        _extraction("River Avon, Bristol", ledger.rows[0].field, "Severn Estuary",
                    "empties into the Severn Estuary"),
        _extraction("River Avon, Bristol", ledger.rows[0].field, "English Channel",
                    "empties into the English Channel"),
    ])
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                     page_url="https://a.example", page_text=page_a))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p2",
                                     page_url="https://b.example", page_text=page_b))
    assert ledger.find("River Avon, Bristol").status == el.STATUS_CONFLICTED


def test_malformed_extraction_json_yields_no_records_and_does_not_crash():
    ledger = el.Ledger.mint(ENUMERATED)
    io = _io([], page_text="page")
    io.query_llm = AsyncMock(return_value="not json at all {{{")
    records = asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                               page_url="https://a.example", page_text="page"))
    assert records == []
    assert all(r.status == el.STATUS_OPEN for r in ledger.rows)


def test_extraction_records_are_append_only_and_keep_the_raw_pointer():
    ledger = el.Ledger.mint("Who wrote Beloved?")
    page = "Beloved was written by Toni Morrison in 1987."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(return_value=_extraction(
        ledger.rows[0].entity, ledger.rows[0].field, "Toni Morrison",
        "written by Toni Morrison"))
    asyncio.run(el.extract_from_page(io, "m", "Who wrote Beloved?", ledger, page_id="p1",
                                     page_url="https://a.example", page_text=page))
    io.query_llm = AsyncMock(return_value=_extraction(
        ledger.rows[0].entity, ledger.rows[0].field, "Toni Morrison",
        "written by Toni Morrison"))
    asyncio.run(el.extract_from_page(io, "m", "Who wrote Beloved?", ledger, page_id="p2",
                                     page_url="https://b.example", page_text=page))
    assert len(ledger.extractions) == 2
    assert ledger.extractions[0].page_id == "p1"
    assert ledger.extractions[0].quote_start >= 0


# --------------------------------------------------------------------------------------
# the loop — ReAct's shape, kept
# --------------------------------------------------------------------------------------


def test_n1_reduces_to_the_plain_react_loop():
    decisions = [
        {"thought": "find author", "action": "search", "args": {"query": "Beloved author"}},
        {"thought": "read", "action": "visit", "args": {"url": "https://en.wikipedia.org/wiki/Toni_Morrison"}},
        {"thought": "answer", "action": "finish", "args": {"answer": "Toni Morrison; MA Cornell."}},
    ]
    replies = [decisions[0], decisions[1],
               _extraction("Beloved", "answer", "Toni Morrison", "written by Toni Morrison"),
               decisions[2]]
    io = _io(replies, page_text="Beloved was written by Toni Morrison.")
    result = asyncio.run(el.run_evidence_loop(io, "Who wrote Beloved?", "m", max_steps=6, max_tokens=512))

    assert len(result.ledger.rows) == 1
    assert "Toni Morrison" in result.deliverable
    io.search.assert_awaited_once()
    assert io.visit.await_args.args[0] == "https://en.wikipedia.org/wiki/Toni_Morrison"


def test_forced_synthesis_when_the_model_never_finishes():
    decisions = [{"thought": "search", "action": "search", "args": {"query": "q"}}]
    io = _io(decisions, synth="FORCED SYNTHESIS")
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=1, max_tokens=512))
    assert "FORCED SYNTHESIS" in result.deliverable


def test_malformed_decision_json_does_not_crash_the_loop():
    io = _io(["not json", "[]", "12", {"action": "finish", "args": {"answer": "DONE"}}])
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=8, max_tokens=512))
    assert "DONE" in result.deliverable
    io.search.assert_not_awaited()


def test_fenced_decision_is_recovered():
    # A model that wraps its decision in a ```json fence must still be parsed via the
    # improved _loads_first_object (no longer a bare json.loads / greedy-regex fallback).
    io = _io([], page_text="")
    fenced = '```json\n{"thought": "fenced", "action": "finish", "args": {"answer": "FENCED ANSWER"}}\n```'
    io.query_llm = AsyncMock(side_effect=[fenced])
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=6, max_tokens=512))
    assert "FENCED ANSWER" in result.deliverable
    io.search.assert_not_awaited()
    io.visit.assert_not_awaited()


def test_fenced_extraction_json_is_recovered():
    # Same fence-recovery fix on the extraction call site (phase="evidence_loop_extract").
    ledger = el.Ledger.mint("Who wrote Beloved?")
    page = "Beloved was written by Toni Morrison in 1987."
    io = _io([], page_text=page)
    fenced = (
        '```json\n{"extractions": [{"entity": "' + ledger.rows[0].entity + '", "field": "'
        + ledger.rows[0].field + '", "value": "Toni Morrison", "verdict": "SUPPORTED", '
        '"quote": "written by Toni Morrison"}]}\n```'
    )
    io.query_llm = AsyncMock(return_value=fenced)
    records = asyncio.run(el.extract_from_page(io, "m", "Who wrote Beloved?", ledger, page_id="p1",
                                               page_url="https://a.example", page_text=page))
    assert len(records) == 1
    assert records[0].value == "Toni Morrison"


def test_a_repeated_search_nudges_instead_of_researching():
    decisions = [
        {"action": "search", "args": {"query": "deepest lake"}},
        {"action": "search", "args": {"query": "Deepest Lake "}},
        {"action": "finish", "args": {"answer": "Lake Baikal"}},
    ]
    io = _io(decisions)
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=6, max_tokens=512))
    assert io.search.await_count == 1
    assert "ALREADY SEARCHED" in "\n".join(result.scratchpad)


def test_an_invalid_action_is_reported_without_crashing():
    io = _io([{"action": "teleport", "args": {}}, {"action": "finish", "args": {"answer": "A"}}])
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=6, max_tokens=512))
    assert "INVALID ACTION" in "\n".join(result.scratchpad)


def test_a_visit_error_blocks_nothing_and_the_loop_continues():
    io = _io([{"action": "visit", "args": {"url": "https://a.example"}},
              {"action": "finish", "args": {"answer": "A"}}])
    io.visit = AsyncMock(side_effect=RuntimeError("boom"))
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=6, max_tokens=512))
    assert "VISIT ERROR" in "\n".join(result.scratchpad)
    assert result.deliverable


def test_the_whole_page_reaches_the_observation_under_the_react_cap():
    page = "X" * 5000
    io = _io([{"action": "visit", "args": {"url": "https://a.example"}},
              "{}",
              {"action": "finish", "args": {"answer": "A"}}], page_text=page)
    result = asyncio.run(el.run_evidence_loop(io, "task", "m", max_steps=6, max_tokens=512))
    entry = next(e for e in result.scratchpad if e.startswith("STEP 1"))
    # whole page into the observation, under ReAct's own 1500-char entry cap (minus the header)
    assert 1400 <= entry.count("X") < el._UNCAPPED_OBSERVATION_CHARS


# --------------------------------------------------------------------------------------
# table-first finalization
# --------------------------------------------------------------------------------------


def test_the_table_renders_every_row_even_when_excerpts_are_truncated():
    mandate = "Compare:\n" + "\n".join(f"{i}. Entity Number {i}" for i in range(1, 7))
    ledger = el.Ledger.mint(mandate)
    for i in range(6):
        _supported(ledger, i, f"value {i}")
    pages = [{"page_id": f"p{i}", "url": f"https://p{i}.example", "text": "Y" * 40000}
             for i in range(6)]
    context = el.render_finalization_context(ledger, pages, char_budget=1200)

    for row in ledger.rows:
        assert row.entity in context
    assert context.index("EVIDENCE TABLE") < context.index("RAW EXCERPTS")
    assert context.count("Y") < 40000


def test_the_verdict_is_derived_from_row_statuses():
    ledger = el.Ledger.mint(ENUMERATED)
    assert ledger.verdict() == el.VERDICT_ABSTAIN
    _supported(ledger, 0, "a")
    assert ledger.verdict() == el.VERDICT_PARTIAL
    _supported(ledger, 1, "b")
    _supported(ledger, 2, "c")
    assert ledger.verdict() == el.VERDICT_ANSWER
    ledger.rows[2].status = el.STATUS_CONFLICTED
    assert ledger.verdict() == el.VERDICT_PARTIAL


def test_an_absent_row_still_allows_an_answer_free_partial_verdict():
    ledger = el.Ledger.mint(ENUMERATED)
    for row in ledger.rows:
        row.status = el.STATUS_ABSENT
    assert ledger.verdict() == el.VERDICT_ABSTAIN


def test_the_deliverable_carries_the_table_and_the_derived_verdict():
    decisions = [{"action": "finish", "args": {"answer": "The Hampshire Avon."}}]
    io = _io(decisions)
    result = asyncio.run(el.run_evidence_loop(io, ENUMERATED, "m", max_steps=4, max_tokens=512))
    assert "The Hampshire Avon." in result.deliverable
    assert "EVIDENCE TABLE" in result.deliverable
    assert result.verdict == el.VERDICT_ABSTAIN
    assert f"VERDICT: {el.VERDICT_ABSTAIN}" in result.deliverable


# --------------------------------------------------------------------------------------
# end-to-end result shape + dispatch
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_variant_runs_end_to_end_and_returns_the_standard_shape(monkeypatch):
    ledger = el.Ledger.mint(ENUMERATED)
    _supported(ledger, 0, "English Channel")

    async def _fake_loop(agent_io, mandate, model_name, max_steps, max_tokens):
        return el.EvidenceLoopResult(deliverable="STUB ANSWER", ledger=ledger,
                                     scratchpad=[], verdict=ledger.verdict())

    monkeypatch.setattr(el, "run_evidence_loop", _fake_loop)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = ENUMERATED

    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {"visit": {"count": 0}},
    )

    assert set(result) >= {"output", "graph", "observability", "duration_seconds", "telemetry"}
    output = result["output"]
    assert output["final_deliverable"] == "STUB ANSWER"
    assert output["success"] is True
    assert output["action_summary"] == "evidence_loop"
    assert output["ledger_verdict"] == el.VERDICT_PARTIAL
    assert len(output["ledger"]) == 3
    assert output["quote_verified_count"] == 0      # no extraction records in this stub ledger
    assert result["observability"]["visit"] == {"count": 0}
    assert result["telemetry"]["correlation_id"].endswith("evidence_loop_r1")


@pytest.mark.asyncio
async def test_a_crashing_loop_still_returns_a_well_formed_failed_result(monkeypatch):
    async def _boom(agent_io, mandate, model_name, max_steps, max_tokens):
        raise RuntimeError("nope")

    monkeypatch.setattr(el, "run_evidence_loop", _boom)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."

    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    assert result["output"]["final_deliverable"] == ""
    assert result["output"]["success"] is False


def test_the_variant_is_registered_in_the_dispatch():
    from agent.app.testing import runner as harness_runner

    assert harness_runner.EVIDENCE_LOOP_VARIANTS == ("evidence_loop",)
    assert "evidence_loop" in harness_runner.KNOWN_EXECUTION_VARIANTS


def test_the_variant_parser_accepts_the_variant_name():
    from agent.app.idea_test_runner import _parse_execution_variants

    assert _parse_execution_variants("evidence_loop") == ["evidence_loop"]
    assert _parse_execution_variants("ledger") == ["evidence_loop"]


@pytest.mark.asyncio
async def test_run_complete_test_dispatches_the_variant(monkeypatch):
    from agent.app.testing import runner as harness_runner

    dispatched = AsyncMock(return_value={
        "output": {"final_deliverable": "42", "success": True, "action_summary": "evidence_loop"},
        "graph": {"nodes": {}}, "observability": {}})
    monkeypatch.setattr(harness_runner, "run_evidence_loop_execution", dispatched)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."
    tm.validation_runner.run = AsyncMock(return_value={"score": 0.0})

    await harness_runner.run_complete_test(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        idea_settings={}, run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
        execution_variant="evidence_loop",
    )
    dispatched.assert_awaited_once()


# --------------------------------------------------------------------------------------
# persisted pages — a stored cell must be re-auditable without the network
# --------------------------------------------------------------------------------------


def test_a_stored_page_round_trips_with_its_hash_and_full_text():
    page = el.store_page("p1", "https://a.example", "Longest span: 1,991 metres.", max_chars=1000)
    assert page["page_id"] == "p1"
    assert page["url"] == "https://a.example"
    assert page["text"] == "Longest span: 1,991 metres."
    assert page["chars"] == 27
    assert page["truncated"] is False
    assert page["content_hash"] == el.hash_page_text("Longest span: 1,991 metres.")
    assert len(page["content_hash"]) == 64


def test_the_hash_changes_when_the_content_changes():
    a = el.store_page("p1", "https://a.example", "Longest span: 1,991 metres.", max_chars=1000)
    b = el.store_page("p1", "https://a.example", "Longest span: 1,992 metres.", max_chars=1000)
    assert a["content_hash"] != b["content_hash"]


def test_a_page_stored_under_a_window_is_marked_truncated():
    page = el.store_page("p1", "https://a.example", "A" * 50, max_chars=10)
    assert page["truncated"] is True
    assert page["stored_chars"] == 10 and page["chars"] == 50
    # the hash still covers the WHOLE fetched text, so drift stays detectable
    assert page["content_hash"] == el.hash_page_text("A" * 50)


def test_a_quote_re_verifies_from_the_stored_text_alone():
    page = el.store_page("p1", "https://a.example",
                         "The River Avon empties into the English Channel.", max_chars=1000)
    match = el.verify_against_stored_page(page, "empties into the English Channel")
    assert match.verified is True
    assert page["text"][match.start:match.end] == "empties into the English Channel"


def test_a_quote_beyond_the_stored_window_is_unverifiable_never_failed():
    # LOAD-BEARING: head-truncation must never manufacture a false ``absent`` verdict.
    page = el.store_page("p1", "https://a.example", "A" * 50 + "the late quote", max_chars=10)
    match = el.verify_against_stored_page(page, "the late quote")
    assert match.verified is None
    assert match.fail_reason == el.QUOTE_FAIL_NO_PAGE


def test_a_paraphrase_still_fails_against_an_untruncated_stored_page():
    page = el.store_page("p1", "https://a.example",
                         "Longest span: 1,991 metres (6,532 ft).", max_chars=1000)
    match = el.verify_against_stored_page(page, "The longest span is 1,991 metres (6,532 ft).")
    assert match.verified is False
    assert match.fail_reason == el.QUOTE_FAIL_ABSENT


def test_an_extraction_with_no_stored_page_is_unverifiable():
    assert el.verify_against_stored_page(None, "anything").verified is None
    assert el.verify_against_stored_page(None, "anything").fail_reason == el.QUOTE_FAIL_NO_PAGE


def _cell(extractions, pages):
    return {"execution": {"output": {"extractions": extractions, "pages": pages}}}


def test_reverifying_a_cell_resolves_each_extraction_to_its_page_by_page_id():
    pages = [el.store_page("p1", "https://a.example", "Toni Morrison wrote Beloved.", 1000),
             el.store_page("p2", "https://b.example", "Cornell University, MA 1955.", 1000)]
    extractions = [
        {"page_id": "p1", "quote": "Toni Morrison wrote Beloved"},
        {"page_id": "p2", "quote": "Cornell University"},
        {"page_id": "p2", "quote": "Toni Morrison wrote Beloved"},   # right quote, wrong page
    ]
    report = el.reverify_cell(_cell(extractions, pages))
    assert [row["quote_verified"] for row in report["extractions"]] == [True, True, False]
    assert report["extractions"][2]["quote_fail_reason"] == el.QUOTE_FAIL_ABSENT
    assert report["counts"]["verified"] == 2
    assert report["counts"]["failed"] == 1


def test_reverifying_a_cell_splits_the_three_causes():
    pages = [el.store_page("p1", "https://a.example", "Longest span: 1,991 metres.", 1000),
             el.store_page("p2", "https://b.example", "A" * 40 + "late quote", max_chars=10)]
    extractions = [
        {"page_id": "p1", "quote": '"Longest span: 1,991 metres"'},      # wrapper-stripped hit
        {"page_id": "p1", "quote": "The longest span is 1,991 metres."},  # paraphrase
        {"page_id": "p1", "quote": ""},                                   # nothing to check
        {"page_id": "p2", "quote": "late quote"},                         # beyond the window
        {"page_id": "p9", "quote": "orphan"},                             # no such page
    ]
    report = el.reverify_cell(_cell(extractions, pages))
    assert report["counts"] == {"verified": 1, "failed": 1, "unchecked": 3,
                                "absent": 1, "no_page": 2, "empty": 1}
    assert report["pages"] == 2
    assert report["extractions"][0]["quote_verified"] is True


def test_reverifying_a_cell_without_stored_pages_reports_everything_unverifiable():
    report = el.reverify_cell(_cell([{"page_id": "p1", "quote": "anything"}], []))
    assert report["counts"]["failed"] == 0
    assert report["counts"]["no_page"] == 1


@pytest.mark.asyncio
async def test_the_loop_persists_every_visited_page_for_later_audit():
    decisions = [
        {"thought": "read", "action": "visit", "args": {"url": "https://a.example"}},
        {"thought": "done", "action": "finish", "args": {"answer": "A"}},
    ]
    page = "Beloved was written by Toni Morrison in 1987."
    io = _io([decisions[0], _extraction("task", "f", "Toni Morrison", "written by Toni Morrison"),
              decisions[1]], page_text=page)
    result = await el.run_evidence_loop(io, "Who wrote Beloved?", "m", max_steps=4, max_tokens=512)
    assert len(result.pages) == 1
    stored = result.pages[0]
    assert stored["page_id"] == "p1" and stored["url"] == "https://a.example"
    assert stored["content_hash"] == el.hash_page_text(page)
    record = result.ledger.extractions[0]
    assert el.verify_against_stored_page(stored, record.quote).verified is True


@pytest.mark.asyncio
async def test_the_result_payload_carries_the_pages_and_the_cell_reverifies(monkeypatch):
    tm = MagicMock()
    tm.metadata = {"test_id": "001"}
    tm.get_task_statement = MagicMock(return_value="Who wrote Beloved?")
    ledger = el.Ledger.mint("Who wrote Beloved?")
    stored = el.store_page("p1", "https://a.example", "Toni Morrison wrote Beloved.", 1000)
    ledger.extractions.append(el.Extraction(
        ledger.rows[0].entity, "f", "Toni Morrison", "SUPPORTED", "https://a.example",
        "Toni Morrison wrote Beloved", True, "p1", 0, 27))

    async def fake_loop(*a, **kw):
        return el.EvidenceLoopResult(deliverable="STUB", ledger=ledger, scratchpad=[],
                                     verdict=el.VERDICT_PARTIAL, pages=[stored])

    monkeypatch.setattr(el, "run_evidence_loop", fake_loop)
    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    assert result["output"]["pages"] == [stored]
    report = el.reverify_cell({"execution": result})
    assert report["counts"]["verified"] == 1


# --------------------------------------------------------------------------------------
# resolution and verification are two independent axes
# --------------------------------------------------------------------------------------


def _resolve(ledger, index, value, quote_verified, url="https://a.example"):
    """Resolve row ``index`` directly, at the requested verification tier."""
    row = ledger.rows[index]
    row.status, row.value, row.source_url = el.STATUS_SUPPORTED, value, url
    row.quote, row.quote_verified = f"{value} quote", quote_verified


def test_a_row_carries_resolution_and_verification_independently():
    ledger = el.Ledger.mint(ENUMERATED)
    assert ledger.rows[0].resolved is False
    assert ledger.rows[0].confidence_tier == el.TIER_UNRESOLVED
    _resolve(ledger, 0, "English Channel", False)
    assert ledger.rows[0].resolved is True
    assert ledger.rows[0].quote_verified is False
    assert ledger.rows[0].confidence_tier == el.TIER_RESOLVED_UNVERIFIED
    _resolve(ledger, 1, "Severn", True)
    assert ledger.rows[1].confidence_tier == el.TIER_RESOLVED_VERIFIED


def test_a_run_whose_values_are_all_paraphrased_does_not_abstain():
    ledger = el.Ledger.mint(ENUMERATED)
    for i in range(3):
        _resolve(ledger, i, f"value {i}", False)
    assert ledger.verdict() == el.VERDICT_ANSWER
    assert ledger.resolution_counts()["resolved_verified"] == 0


def test_a_run_that_found_nothing_still_abstains():
    ledger = el.Ledger.mint(ENUMERATED)
    assert ledger.verdict() == el.VERDICT_ABSTAIN
    for row in ledger.rows:
        row.status = el.STATUS_ABSENT
    assert ledger.verdict() == el.VERDICT_ABSTAIN
    ledger.rows[0].status = el.STATUS_BLOCKED
    assert ledger.verdict() == el.VERDICT_ABSTAIN


def test_resolution_counts_split_the_two_axes():
    ledger = el.Ledger.mint(ENUMERATED)
    _resolve(ledger, 0, "a", True)
    _resolve(ledger, 1, "b", False)
    assert ledger.resolution_counts() == {
        "rows": 3, "resolved": 2, "resolved_verified": 1, "resolved_unverified": 1,
        "unresolved": 1,
    }


def test_a_conflicted_row_counts_as_unresolved():
    ledger = el.Ledger.mint(ENUMERATED)
    for i in range(3):
        _resolve(ledger, i, f"v{i}", True)
    ledger.rows[2].status = el.STATUS_CONFLICTED
    assert ledger.rows[2].resolved is False
    assert ledger.resolution_counts()["unresolved"] == 1
    assert ledger.verdict() == el.VERDICT_PARTIAL


def test_quote_verification_counts_are_untouched_by_resolution():
    # Same extraction stream, same numbers as before the split -- only the ROW status moved.
    ledger, records = _extract_with("The Avon flows westward towards the Severn Estuary.")
    assert ledger.find("River Avon, Bristol").resolved is True
    assert el.quote_verification_counts(ledger) == {
        "verified": 0, "failed": 1, "unchecked": 0, "absent": 1, "no_page": 0, "empty": 0}


def test_a_verified_record_upgrades_a_row_resolved_from_a_paraphrase():
    ledger = el.Ledger.mint(ENUMERATED)
    page = "The River Avon, Bristol empties into the Severn Estuary."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(side_effect=[
        _extraction("River Avon, Bristol", ledger.rows[0].field, "Severn Estuary",
                    "flows towards the Severn"),                     # paraphrase
        _extraction("River Avon, Bristol", ledger.rows[0].field, "Severn Estuary",
                    "empties into the Severn Estuary"),              # verbatim
    ])
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                     page_url="https://a.example", page_text=page))
    assert ledger.rows[0].confidence_tier == el.TIER_RESOLVED_UNVERIFIED
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p2",
                                     page_url="https://b.example", page_text=page))
    assert ledger.rows[0].confidence_tier == el.TIER_RESOLVED_VERIFIED
    assert ledger.rows[0].source_url == "https://b.example"


def test_an_unverified_record_never_overturns_a_verified_row():
    ledger = el.Ledger.mint(ENUMERATED)
    _resolve(ledger, 0, "English Channel", True)
    page = "The River Avon, Bristol flows west."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(return_value=_extraction(
        "River Avon, Bristol", ledger.rows[0].field, "Severn Estuary", "a paraphrase"))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p2",
                                     page_url="https://b.example", page_text=page))
    assert ledger.rows[0].status == el.STATUS_SUPPORTED
    assert ledger.rows[0].value == "English Channel"


def test_two_differing_unverified_values_conflict_the_row():
    ledger = el.Ledger.mint(ENUMERATED)
    page = "The River Avon, Bristol flows west."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(side_effect=[
        _extraction("River Avon, Bristol", ledger.rows[0].field, "Severn Estuary", "para one"),
        _extraction("River Avon, Bristol", ledger.rows[0].field, "English Channel", "para two"),
    ])
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                     page_url="https://a.example", page_text=page))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p2",
                                     page_url="https://b.example", page_text=page))
    assert ledger.rows[0].status == el.STATUS_CONFLICTED


def test_a_supported_record_without_a_value_never_resolves_a_row():
    ledger, records = _extract_with("flows west into the Severn Estuary")
    ledger.rows[0].status, ledger.rows[0].value = el.STATUS_OPEN, ""
    record = el.Extraction("River Avon, Bristol", "f", "", "SUPPORTED", "u", "q", True, "p1")
    ledger.apply(record)
    assert ledger.rows[0].status == el.STATUS_OPEN


# --------------------------------------------------------------------------------------
# an unverified row is reported honestly, never silently
# --------------------------------------------------------------------------------------


def test_an_unverified_row_is_flagged_in_the_prompt_line_and_the_table():
    ledger = el.Ledger.mint(ENUMERATED)
    _resolve(ledger, 0, "English Channel", False)
    _resolve(ledger, 1, "Severn", True)
    assert "unverified" in ledger.rows[0].render().lower()
    assert "unverified" not in ledger.rows[1].render().lower()
    assert len(ledger.rows[0].render()) < el._UNCAPPED_OBSERVATION_CHARS / 4
    table = ledger.render_table()
    assert "UNVERIFIED QUOTE" in table


def test_the_deliverable_reports_both_axes_and_flags_unverified_provenance():
    ledger = el.Ledger.mint(ENUMERATED)
    for i in range(3):
        _resolve(ledger, i, f"v{i}", i == 0)
    text = el._decorate("PROSE", ledger)
    assert f"VERDICT: {el.VERDICT_ANSWER}" in text
    assert "3/3" in text and "1/3" in text
    assert "UNVERIFIED PROVENANCE" in text


def test_a_fully_verified_deliverable_carries_no_unverified_flag():
    ledger = el.Ledger.mint(ENUMERATED)
    for i in range(3):
        _resolve(ledger, i, f"v{i}", True)
    assert "UNVERIFIED PROVENANCE" not in el._decorate("PROSE", ledger)


def test_the_row_payload_carries_both_axes():
    ledger = el.Ledger.mint(ENUMERATED)
    _resolve(ledger, 0, "English Channel", False)
    row = ledger.rows[0].as_dict()
    assert row["resolved"] is True
    assert row["quote_verified"] is False
    assert row["confidence_tier"] == el.TIER_RESOLVED_UNVERIFIED


@pytest.mark.asyncio
async def test_the_result_payload_exposes_resolution_and_verification_separately(monkeypatch):
    ledger = el.Ledger.mint(ENUMERATED)
    _resolve(ledger, 0, "English Channel", False)
    _resolve(ledger, 1, "Severn", True)
    ledger.extractions = [
        el.Extraction("a", "f", "v", "SUPPORTED", "u", "q", True, "p1", 0, 1),
        el.Extraction("b", "f", "v", "SUPPORTED", "u", "q", False, "p1",
                      quote_fail_reason=el.QUOTE_FAIL_ABSENT),
    ]

    async def _fake_loop(agent_io, mandate, model_name, max_steps, max_tokens):
        return el.EvidenceLoopResult(deliverable="STUB", ledger=ledger, scratchpad=[],
                                     verdict=ledger.verdict())

    monkeypatch.setattr(el, "run_evidence_loop", _fake_loop)
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = ENUMERATED

    output = (await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {}))["output"]

    assert output["ledger_verdict"] == el.VERDICT_PARTIAL
    assert output["ledger_resolution_counts"] == {
        "rows": 3, "resolved": 2, "resolved_verified": 1, "resolved_unverified": 1,
        "unresolved": 1}
    assert output["rows_resolved"] == 2
    assert output["rows_quote_backed"] == 1
    assert output["unverified_provenance"] is True
    # the verification axis is reported over EXTRACTIONS, unchanged by the split
    assert output["quote_verified_count"] == 1
    assert output["quote_unverified_count"] == 1
    assert output["quote_fail_reasons"]["absent"] == 1


def test_a_verified_record_breaks_a_conflict_made_of_paraphrases():
    # A weak model's paraphrases must not deadlock a row it later proves off a page.
    ledger = el.Ledger.mint(ENUMERATED)
    page = "The River Avon, Bristol empties into the Severn Estuary."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(side_effect=[
        _extraction("River Avon, Bristol", "f", "English Channel", "para one"),
        _extraction("River Avon, Bristol", "f", "Bristol Channel", "para two"),
        _extraction("River Avon, Bristol", "f", "Severn Estuary",
                    "empties into the Severn Estuary"),
    ])
    for page_id in ("p1", "p2"):
        asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id=page_id,
                                         page_url="https://a.example", page_text=page))
    assert ledger.rows[0].status == el.STATUS_CONFLICTED
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p3",
                                     page_url="https://c.example", page_text=page))
    assert ledger.rows[0].confidence_tier == el.TIER_RESOLVED_VERIFIED
    assert ledger.rows[0].value == "Severn Estuary"


def test_a_conflict_between_verified_values_is_never_broken():
    ledger = el.Ledger.mint(ENUMERATED)
    page = "The River Avon, Bristol empties into the Severn Estuary and the Bristol Channel."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(side_effect=[
        _extraction("River Avon, Bristol", "f", "Severn Estuary", "empties into the Severn"),
        _extraction("River Avon, Bristol", "f", "Bristol Channel", "the Bristol Channel"),
        _extraction("River Avon, Bristol", "f", "Severn Estuary", "empties into the Severn"),
    ])
    for page_id in ("p1", "p2", "p3"):
        asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id=page_id,
                                         page_url="https://a.example", page_text=page))
    assert ledger.rows[0].status == el.STATUS_CONFLICTED
    assert ledger.rows[0].resolved is False


def test_two_values_reaching_a_single_row_ledger_by_the_catch_all_never_conflict():
    # On an N=1 ledger every record matches, so differing values are different FACTS, not a
    # disagreement about one fact -- calling that CONFLICTED would abstain on a solved run.
    ledger = el.Ledger.mint("Who wrote Beloved and where did she get her MA?")
    page = "Beloved was written by Toni Morrison. She took her MA at Cornell University."
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(side_effect=[
        _extraction("Beloved", "f", "Toni Morrison", "Beloved was written by Toni Morrison"),
        _extraction("MA", "f", "Cornell University", "her MA at Cornell University"),
    ])
    for page_id in ("p1", "p2"):
        asyncio.run(el.extract_from_page(io, "m", "Who wrote Beloved?", ledger, page_id=page_id,
                                         page_url="https://a.example", page_text=page))
    assert ledger.rows[0].status == el.STATUS_SUPPORTED
    assert ledger.verdict() == el.VERDICT_ANSWER


def test_a_record_naming_the_single_row_can_still_conflict_it():
    ledger = el.Ledger.mint("How deep is Lake Baikal?")
    entity = ledger.rows[0].entity
    ledger.apply(el.Extraction(entity, "f", "1642 m", "SUPPORTED", "u", "q", True, "p1"))
    ledger.apply(el.Extraction(entity, "f", "1620 m", "SUPPORTED", "u", "q", True, "p2"))
    assert ledger.rows[0].status == el.STATUS_CONFLICTED


def test_an_all_conflicted_run_reports_partial_not_abstain():
    # ABSTAIN means "we did not obtain the values". Two disagreeing values are still values.
    ledger = el.Ledger.mint(ENUMERATED)
    for row in ledger.rows:
        row.status, row.value = el.STATUS_CONFLICTED, "disputed"
    assert ledger.verdict() == el.VERDICT_PARTIAL
    assert ledger.resolution_counts()["resolved"] == 0


# --------------------------------------------------------------------------------------
# unit-bearing values — a bare number is ~28x easier to false-verify than "590 m"
# --------------------------------------------------------------------------------------

DEPTH_PAGE = "Lake Baikal. Maximum depth 1,642 m (5,387 ft). Surface elevation 455 m."


def _extraction_with_unit(entity, field, value, unit, quote, verdict="SUPPORTED"):
    return json.dumps({"extractions": [{"entity": entity, "field": field, "value": value,
                                        "unit": unit, "verdict": verdict, "quote": quote}]})


def _extract_page(payload, page, mandate="How deep is Lake Baikal?"):
    ledger = el.Ledger.mint(mandate)
    io = _io([], page_text=page)
    io.query_llm = AsyncMock(return_value=payload)
    records = asyncio.run(el.extract_from_page(io, "m", mandate, ledger, page_id="p1",
                                               page_url="https://a.example", page_text=page))
    return ledger, records, io


def test_the_extraction_prompt_asks_for_the_value_with_its_unit():
    prompt = el.extract_system_prompt(True)
    assert "\"unit\"" in prompt
    assert "unit" in prompt.lower() and "590 m" in prompt


def test_the_extraction_prompt_reverts_to_the_bare_value_when_the_flag_is_off():
    assert "\"unit\"" not in el.extract_system_prompt(False)


def test_the_extraction_call_carries_the_unit_prompt_by_default():
    _, _, io = _extract_page(
        _extraction_with_unit("task", "depth", "1,642 m", "m", "Maximum depth 1,642 m"),
        DEPTH_PAGE)
    system = io.build_llm_payload.call_args.kwargs["messages"][0]["content"]
    assert "\"unit\"" in system


def test_a_unit_bearing_value_verifies_as_unit_bearing():
    _, records, _ = _extract_page(
        _extraction_with_unit("task", "depth", "1,642 m", "m", "Maximum depth 1,642 m"),
        DEPTH_PAGE)
    rec = records[0]
    assert rec.unit == "m"
    assert rec.value_verified is True
    assert rec.value_unit_bearing is True
    assert rec.value_shape == "number_with_unit"
    assert rec.as_dict()["value_unit_bearing"] is True


def test_a_bare_value_plus_a_separate_unit_field_still_verifies_unit_bearing():
    _, records, _ = _extract_page(
        _extraction_with_unit("task", "depth", "1,642", "m", "Maximum depth 1,642 m"),
        DEPTH_PAGE)
    assert records[0].value_verified is True
    assert records[0].value_unit_bearing is True
    # FIX 1: a bare value with the unit ONLY in the separate `unit` field is real unit-bearing
    # coverage too (38.0% of the gpu0831 block's numeric records) — the stored `value_shape`
    # must reflect that, not just a unit embedded in the value string itself.
    assert records[0].value_shape == "number_with_unit"


def test_a_value_with_no_natural_unit_is_reported_never_refused():
    page = "Beloved was written by Toni Morrison in 1987."
    _, records, _ = _extract_page(
        _extraction_with_unit("task", "author", "Toni Morrison", "", "written by Toni Morrison"),
        page, mandate="Who wrote Beloved?")
    rec = records[0]
    assert rec.value_verified is True
    assert rec.value_unit_bearing is False
    assert rec.value_shape == "text"
    assert rec.value == "Toni Morrison"


def test_a_feet_figure_quoted_as_metres_does_not_verify_against_the_metre_span():
    _, records, _ = _extract_page(
        _extraction_with_unit("task", "depth", "5,387 m", "m", "Maximum depth 1,642 m"),
        DEPTH_PAGE)
    assert records[0].value_verified is False
    assert records[0].value_fail_reason == "absent"


def test_the_same_feet_figure_bare_would_have_false_verified():
    # The motivation, pinned: the bare number is on the page (as feet) and sails through.
    _, records, _ = _extract_page(
        _extraction_with_unit("task", "depth", "5,387", "", "Maximum depth 1,642 m"),
        DEPTH_PAGE)
    assert records[0].value_verified is True
    assert records[0].value_unit_bearing is False


def test_a_resolved_row_keeps_the_unit_for_downstream_arithmetic():
    ledger, _, _ = _extract_page(
        _extraction_with_unit("task", "depth", "1,642 m", "m", "Maximum depth 1,642 m"),
        DEPTH_PAGE)
    assert ledger.rows[0].unit == "m"
    assert ledger.rows[0].as_dict()["unit"] == "m"


# --------------------------------------------------------------------------------------
# closed-roster completeness gate — the roster is stated in the mandate, so this is arithmetic
# --------------------------------------------------------------------------------------

def _roster_mandate(n):
    items = "\n".join(f"{i}. Lake Number {i} — a lake" for i in range(1, n + 1))
    return f"How many of these lakes are deeper than 480 m?\n{items}"


ROSTER_7 = _roster_mandate(7)


def test_the_roster_size_is_read_from_the_mandate():
    ledger = el.Ledger.mint(ROSTER_7)
    status = ledger.roster()
    assert status["roster_named"] == 7
    assert status["roster_resolved"] == 0
    assert status["roster_complete"] is False
    assert status["roster_truncated"] is False


def test_a_complete_roster_answers_without_a_banner():
    ledger = el.Ledger.mint(ROSTER_7)
    for index in range(7):
        _resolve(ledger, index, f"{index} m", True)
    assert ledger.roster()["roster_complete"] is True
    assert ledger.verdict() == el.VERDICT_ANSWER
    assert "ROSTER INCOMPLETE" not in el._decorate("3", ledger)


def test_a_count_over_a_short_roster_is_flagged_rather_than_asserted(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_ROSTER_GATE", "1")
    ledger = el.Ledger.mint(ROSTER_7)
    for index in range(6):
        _resolve(ledger, index, f"{index} m", True)
    status = ledger.roster()
    assert (status["roster_named"], status["roster_resolved"]) == (7, 6)
    assert status["roster_complete"] is False
    assert ledger.verdict() == el.VERDICT_PARTIAL
    deliverable = el._decorate("3", ledger)
    assert "ROSTER INCOMPLETE" in deliverable
    assert "6" in deliverable and "7" in deliverable


def test_the_incomplete_roster_warning_reaches_the_synthesis_context():
    ledger = el.Ledger.mint(ROSTER_7)
    for index in range(6):
        _resolve(ledger, index, f"{index} m", True)
    context = el.render_finalization_context(ledger, [], char_budget=4000)
    assert "ROSTER INCOMPLETE" in context


def test_roster_truncation_at_the_row_cap_is_visible_in_the_payload():
    ledger = el.Ledger.mint(_roster_mandate(16))
    assert len(ledger.rows) == el._MAX_ROWS
    status = ledger.roster()
    assert status["roster_named"] == 16
    assert status["roster_truncated"] is True
    assert status["roster_complete"] is False


def test_a_truncated_roster_cannot_reach_an_answer_verdict(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_ROSTER_GATE", "1")
    ledger = el.Ledger.mint(_roster_mandate(16))
    for index in range(len(ledger.rows)):
        _resolve(ledger, index, f"{index} m", True)
    assert ledger.verdict() == el.VERDICT_PARTIAL
    assert "ROSTER TRUNCATED" in el._decorate("4", ledger)


def test_an_n16_mandate_is_not_silently_capped_when_the_row_cap_is_raised(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_MAX_ROWS", "16")
    ledger = el.Ledger.mint(_roster_mandate(16))
    assert len(ledger.rows) == 16
    assert ledger.roster()["roster_truncated"] is False


def test_the_roster_gate_is_inert_on_a_mandate_that_names_no_roster():
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", True)
    status = ledger.roster()
    assert status["roster_named"] == 0
    assert status["roster_complete"] is True
    assert ledger.verdict() == el.VERDICT_ANSWER


def test_the_roster_gate_is_off_by_default_and_only_the_verdict_downgrade_is_gated(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_ROSTER_GATE", raising=False)
    ledger = el.Ledger.mint(_roster_mandate(16))
    for index in range(len(ledger.rows)):
        _resolve(ledger, index, f"{index} m", True)
    assert ledger.verdict() == el.VERDICT_ANSWER
    assert "ROSTER TRUNCATED" in el._decorate("4", ledger)


@pytest.mark.asyncio
async def test_the_result_payload_reports_the_roster_counts(monkeypatch):
    tm = MagicMock()
    tm.metadata = {"test_id": "072"}
    tm.get_task_statement = MagicMock(return_value=ROSTER_7)
    ledger = el.Ledger.mint(ROSTER_7)
    for index in range(6):
        _resolve(ledger, index, f"{index} m", True)

    async def fake_loop(*a, **kw):
        return el.EvidenceLoopResult(deliverable="STUB", ledger=ledger, scratchpad=[],
                                     verdict=ledger.verdict(), pages=[])

    monkeypatch.setattr(el, "run_evidence_loop", fake_loop)
    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    assert output["roster_named"] == 7
    assert output["roster_resolved"] == 6
    assert output["roster_complete"] is False
    assert output["roster_truncated"] is False


# --------------------------------------------------------------------------------------
# layer 4: the derivation graph is BUILT during the run and `derive` is a typed action
# --------------------------------------------------------------------------------------

TOWERS = "Tower A is 590 m tall. Tower B is 566 m tall. Its architect was Gustave Eiffel."


def _extractions(*records):
    """One extraction reply carrying several records, as the extractor really emits them."""
    return json.dumps({"extractions": [
        {"entity": e, "field": f, "value": v, "verdict": "SUPPORTED", "quote": q, "unit": u}
        for e, f, v, q, u in records]})


_TOWER_RECORDS = (
    ("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
    ("Tower B", "height", "566 m", "Tower B is 566 m tall.", "m"),
)


def _visit_then(*decisions, page=TOWERS):
    """Replies for: visit -> extraction -> each further decision in turn."""
    return [{"action": "visit", "args": {"url": "https://example.org/towers"}},
            _extractions(*_TOWER_RECORDS), *decisions]


def _run(replies, mandate="How much taller is Tower A than Tower B?", page=TOWERS):
    io = _io(replies, page_text=page)
    return asyncio.run(el.run_evidence_loop(io, mandate, "m", max_steps=6, max_tokens=64))


class TestDerivationGraphIsBuilt:
    def test_a_visited_page_and_its_verified_values_enter_the_graph(self):
        result = _run(_visit_then({"action": "finish", "args": {"answer": "24 m"}}))
        graph = result.ledger.graph
        assert graph is not None
        assert [p["page_id"] for p in graph.pages()] == ["p1"]
        values = sorted(n.value for n in graph.nodes())
        assert values == ["566 m", "590 m"]

    def test_a_value_absent_from_the_page_is_refused_a_source_node(self):
        replies = [{"action": "visit", "args": {"url": "https://example.org/towers"}},
                   _extractions(("Tower C", "height", "999 m", "Tower C is 999 m tall.", "m")),
                   {"action": "finish", "args": {"answer": "x"}}]
        result = _run(replies)
        assert result.ledger.graph.nodes() == []
        assert result.ledger.graph.rejections

    def test_each_admitted_source_gets_a_stable_prompt_handle(self):
        result = _run(_visit_then({"action": "finish", "args": {"answer": "24 m"}}))
        handles = result.ledger.handles
        assert sorted(handles) == ["E1", "E2"]
        assert all(result.ledger.graph.node(nid) is not None for nid in handles.values())


class TestDeriveAction:
    def test_difference_is_recomputed_and_reported_with_its_unit(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference",
                                          "input_refs": ["E1", "E2"], "expected_unit": "m"}},
            {"action": "finish", "args": {"answer": "24 m"}}))
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVED D1 = 24 m" in step
        assert any(n.operation == "difference" and n.value == "24" for n in result.ledger.graph.nodes())

    def test_a_models_proposed_value_never_replaces_the_recomputation(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"],
                                          "proposed_value": "1594"}},
            {"action": "finish", "args": {"answer": "x"}}))
        derived = [n for n in result.ledger.graph.nodes() if n.operation == "difference"][0]
        assert derived.value == "24"
        assert derived.derivation_valid is False
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "disagrees" in step

    def test_mismatched_units_are_refused_with_a_typed_code_the_model_can_act_on(self):
        page = "Tower A is 590 m tall. Tower B is 1,940 ft tall."
        replies = [{"action": "visit", "args": {"url": "https://example.org/towers"}},
                   _extractions(("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
                                ("Tower B", "height", "1,940 ft", "Tower B is 1,940 ft tall.", "ft")),
                   {"action": "derive", "args": {"operation": "difference",
                                                 "input_refs": ["E1", "E2"]}},
                   {"action": "finish", "args": {"answer": "x"}}]
        result = _run(replies, page=page)
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED [UNIT_MISMATCH]" in step
        assert "does not convert" in step
        assert not [n for n in result.ledger.graph.nodes() if n.operation == "difference"]

    def test_an_unknown_reference_is_refused_as_a_missing_operand(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "sum", "input_refs": ["E1", "E9"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED [MISSING_OPERAND]" in step

    def test_dividing_by_a_zero_operand_is_refused_not_crashed(self):
        page = "Team A scored 400 goals. Team B played 0 games."
        replies = [{"action": "visit", "args": {"url": "https://example.org/t"}},
                   _extractions(("A", "goals", "400 goals", "Team A scored 400 goals.", "goals"),
                                ("B", "games", "0 games", "Team B played 0 games.", "games")),
                   {"action": "derive", "args": {"operation": "quotient",
                                                 "input_refs": ["E1", "E2"]}},
                   {"action": "finish", "args": {"answer": "x"}}]
        result = _run(replies, page=page)
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED [DIVISION_BY_ZERO]" in step

    def test_a_non_numeric_operand_is_refused(self):
        replies = [{"action": "visit", "args": {"url": "https://example.org/towers"}},
                   _extractions(("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
                                ("Tower A", "architect", "Gustave Eiffel",
                                 "Its architect was Gustave Eiffel.", "")),
                   {"action": "derive", "args": {"operation": "sum", "input_refs": ["E1", "E2"]}},
                   {"action": "finish", "args": {"answer": "x"}}]
        result = _run(replies)
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED [NON_NUMERIC]" in step

    def test_an_unknown_operation_is_refused_and_names_the_vocabulary(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "integrate", "input_refs": ["E1"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED [UNKNOWN_OPERATION]" in step

    def test_derive_with_no_evidence_yet_is_an_observation_not_a_crash(self):
        result = _run([{"action": "derive", "args": {"operation": "sum", "input_refs": ["E1"]}},
                       {"action": "finish", "args": {"answer": "x"}}])
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED" in step

    def test_the_invalid_action_nudge_now_offers_derive(self):
        result = _run([{"action": "nonsense", "args": {}},
                       {"action": "finish", "args": {"answer": "x"}}])
        assert any("derive" in s for s in result.scratchpad if "INVALID ACTION" in s)


class TestDerivedValuesReachSynthesis:
    def test_the_finalization_context_carries_the_locked_derived_values(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {}}))
        context = el.render_finalization_context(result.ledger, [], 4000)
        assert "DERIVED VALUES" in context
        assert "24" in context

    def test_a_ledger_with_no_derivations_renders_no_derived_block(self):
        ledger = el.Ledger.mint("Who wrote Beloved?")
        assert "DERIVED VALUES" not in el.render_finalization_context(ledger, [], 2000)


class TestDerivationGate:
    def _invalid_run(self):
        return _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"],
                                          "proposed_value": "1594"}},
            {"action": "finish", "args": {"answer": "x"}}))

    def test_an_invalid_derivation_does_not_downgrade_the_verdict_by_default(self, monkeypatch):
        monkeypatch.delenv("LEDGER_DERIVATION_GATE", raising=False)
        assert el.derivation_gate_enabled() is False

    def test_the_gate_downgrades_an_answer_to_partial_when_enabled(self, monkeypatch):
        result = self._invalid_run()
        ledger = result.ledger
        for row in ledger.rows:
            row.status = el.STATUS_SUPPORTED
            row.value = row.value or "590 m"
        monkeypatch.delenv("LEDGER_DERIVATION_GATE", raising=False)
        assert ledger.verdict() == el.VERDICT_ANSWER
        monkeypatch.setenv("LEDGER_DERIVATION_GATE", "1")
        assert ledger.verdict() == el.VERDICT_PARTIAL


class TestGraphArtifact:
    def test_the_graph_is_emitted_and_reverifies_offline(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {"answer": "24 m"}}))
        artifact = result.ledger.graph.to_dict()
        from agent.app.testing.evidence_graph import reverify_graph
        report = reverify_graph(artifact)
        assert report["pages"] == 1
        assert report["counts"]["page_drift"] == 0
        assert report["counts"]["derived"] == 1


class TestEvidenceBlockRendering:
    def test_an_already_unit_bearing_value_does_not_get_its_unit_twice(self):
        result = _run(_visit_then({"action": "finish", "args": {"answer": "x"}}))
        lines = result.ledger.evidence_lines()
        assert any(line.startswith("E1 = 590 m ") for line in lines)
        assert not any("590 m m" in line for line in lines)

    def test_a_bare_derived_value_still_gets_its_composed_unit(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        assert any(line.startswith("D1 = 24 m ") for line in result.ledger.evidence_lines())

    def test_the_source_block_is_bounded_but_never_drops_a_derived_value(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        ledger = result.ledger
        # Stand in 60 further sources past the cap. They point at a real admitted node, because
        # a handle whose node id is not in the graph is skipped rather than counted.
        real = ledger.handles["E1"]
        for i in range(60):
            ledger.handles[f"E{i + 100}"] = real
        lines = ledger.evidence_lines(max_sources=3)
        assert any("not shown" in line for line in lines)
        assert any(line.startswith("D1 = 24 m") for line in lines), "a derived value was elided"

    def test_the_cap_is_a_display_bound_not_a_retention_bound(self):
        result = _run(_visit_then({"action": "finish", "args": {"answer": "x"}}))
        ledger = result.ledger
        assert ledger.evidence_lines(max_sources=1) != ledger.evidence_lines(max_sources=40)
        # both handles still resolve, so a `derive` call can still reference the elided one
        assert ledger.resolve_ref("E1") in {n.id for n in ledger.graph.nodes()}
        assert ledger.resolve_ref("E2") in {n.id for n in ledger.graph.nodes()}


class TestRefusalsReachTheArtifact:
    """A live `derive` refusal must be countable from the STORED cell, not only in the prompt.

    The observation already told the model what went wrong; the artifact told nobody. On the
    `ledgernum22` campaign that made the unit-mismatch refusal endpoint unmeasurable: tasks
    222-224 exist to prove incompatible units are refused rather than converted, and a refusal is
    invisible because it creates no node and the scratchpad is not persisted.
    """

    def _mismatch_run(self):
        page = "Tower A is 590 m tall. Tower B is 1,940 ft tall."
        replies = [{"action": "visit", "args": {"url": "https://example.org/towers"}},
                   _extractions(("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
                                ("Tower B", "height", "1,940 ft", "Tower B is 1,940 ft tall.", "ft")),
                   {"action": "derive", "args": {"operation": "difference",
                                                 "input_refs": ["E1", "E2"]}},
                   {"action": "finish", "args": {"answer": "they cannot be combined"}}]
        return _run(replies, page=page)

    def test_a_refused_derivation_is_recorded_on_the_graph(self):
        result = self._mismatch_run()
        refusals = result.ledger.graph.derivation_refusals
        assert len(refusals) == 1
        assert refusals[0]["code"] == "UNIT_MISMATCH"
        assert refusals[0]["operation"] == "difference"

    def test_the_refusal_is_countable_from_the_artifact_alone(self):
        artifact = self._mismatch_run().ledger.graph.to_dict()
        assert artifact["refusal_counts"] == {"UNIT_MISMATCH": 1}

    def test_a_refusal_still_creates_no_derived_node(self):
        result = self._mismatch_run()
        assert not [n for n in result.ledger.graph.nodes() if n.kind == "derived"]

    def test_a_successful_derivation_records_no_refusal(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {"answer": "24 m"}}))
        assert result.ledger.graph.derivation_refusals == []
        assert result.ledger.graph.to_dict()["refusal_counts"] == {}

    def test_the_model_still_sees_the_actionable_observation(self):
        """Recording the refusal must not replace telling the model what to do about it."""
        step = [s for s in self._mismatch_run().scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED [UNIT_MISMATCH]" in step
        assert "does not convert" in step


# --------------------------------------------------------------------------------------
# repeat-refusal guard — a live cell retried the SAME impossible derivation 10 times,
# each attempt burning one of the run's limited steps for an identical NON_NUMERIC refusal
# --------------------------------------------------------------------------------------


def _run_steps(replies, mandate="How much taller is Tower A than Tower B?", page=TOWERS,
              max_steps=8):
    io = _io(replies, page_text=page)
    return asyncio.run(el.run_evidence_loop(io, mandate, "m", max_steps=max_steps, max_tokens=64))


def _non_numeric_derive_replies(n, operation="sum"):
    """visit -> extraction (one numeric, one non-numeric field) -> N identical derive attempts."""
    return [{"action": "visit", "args": {"url": "https://example.org/towers"}},
            _extractions(("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
                        ("Tower A", "architect", "Gustave Eiffel",
                         "Its architect was Gustave Eiffel.", "")),
            *[{"action": "derive", "args": {"operation": operation, "input_refs": ["E1", "E2"]}}
              for _ in range(n)],
            {"action": "finish", "args": {"answer": "x"}}]


class TestDeriveRepeatGuard:
    def test_a_repeated_refused_derivation_short_circuits_after_the_first_attempt(self):
        result = _run_steps(_non_numeric_derive_replies(3))
        derive_steps = [s for s in result.scratchpad if "action=derive" in s]
        assert len(derive_steps) == 3
        assert "DERIVE REFUSED [NON_NUMERIC]" in derive_steps[0]
        assert "ALREADY REFUSED" in derive_steps[1]
        assert "ALREADY REFUSED" in derive_steps[2]

    def test_the_repeat_notice_names_the_original_refusal_code_and_advice(self):
        result = _run_steps(_non_numeric_derive_replies(2))
        derive_steps = [s for s in result.scratchpad if "action=derive" in s]
        repeat = derive_steps[1]
        assert "NON_NUMERIC" in repeat
        assert "numeric value" in repeat

    def test_the_repeat_is_not_double_counted_on_the_artifact(self):
        result = _run_steps(_non_numeric_derive_replies(3))
        assert result.ledger.graph.refusal_counts() == {"NON_NUMERIC": 1}
        assert len(result.ledger.graph.derivation_refusals) == 1

    def test_a_different_operand_set_for_the_same_operation_is_not_a_repeat(self):
        page = ("Tower A is 590 m tall. Its architect was Gustave Eiffel. "
                "Tower B's architect was Someone Else.")
        replies = [{"action": "visit", "args": {"url": "https://example.org/towers"}},
                   _extractions(("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
                                ("Tower A", "architect", "Gustave Eiffel",
                                 "Its architect was Gustave Eiffel.", ""),
                                ("Tower B", "architect", "Someone Else",
                                 "Tower B's architect was Someone Else.", "")),
                   {"action": "derive", "args": {"operation": "sum", "input_refs": ["E1", "E2"]}},
                   {"action": "derive", "args": {"operation": "sum", "input_refs": ["E1", "E3"]}},
                   {"action": "finish", "args": {"answer": "x"}}]
        result = _run_steps(replies, page=page)
        derive_steps = [s for s in result.scratchpad if "action=derive" in s]
        assert len(derive_steps) == 2
        assert "DERIVE REFUSED [NON_NUMERIC]" in derive_steps[0]
        assert "DERIVE REFUSED [NON_NUMERIC]" in derive_steps[1]
        assert "ALREADY REFUSED" not in derive_steps[1]

    def test_the_guard_can_be_disabled_by_flag(self, monkeypatch):
        monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_DEDUP_DERIVE", "0")
        result = _run_steps(_non_numeric_derive_replies(2))
        derive_steps = [s for s in result.scratchpad if "action=derive" in s]
        assert len(derive_steps) == 2
        assert "DERIVE REFUSED [NON_NUMERIC]" in derive_steps[0]
        assert "DERIVE REFUSED [NON_NUMERIC]" in derive_steps[1]
        assert "ALREADY REFUSED" not in derive_steps[1]


# --------------------------------------------------------------------------------------
# operation aliases — the same weak model emitted `multiply` (spelled `product`) and
# `convert_to_numeric`, both rejected as UNKNOWN_OPERATION despite the system prompt
# --------------------------------------------------------------------------------------


class TestDeriveOperationAliases:
    def test_multiply_aliases_to_product_with_an_identical_node(self):
        result_product = _run(_visit_then(
            {"action": "derive", "args": {"operation": "product", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        result_multiply = _run(_visit_then(
            {"action": "derive", "args": {"operation": "multiply", "input_refs": ["E1", "E2"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        derived_p = [n for n in result_product.ledger.graph.nodes() if n.kind == "derived"][0]
        derived_m = [n for n in result_multiply.ledger.graph.nodes() if n.kind == "derived"][0]
        assert derived_p.operation == "product"
        assert derived_m.operation == "product"
        assert derived_p.value == derived_m.value
        assert derived_p.id == derived_m.id

    def test_add_subtract_divide_alias_to_the_named_ops(self):
        for alias, canonical in (("add", "sum"), ("subtract", "difference"),
                                 ("divide", "quotient")):
            result = _run(_visit_then(
                {"action": "derive", "args": {"operation": alias, "input_refs": ["E1", "E2"]}},
                {"action": "finish", "args": {"answer": "x"}}))
            derived = [n for n in result.ledger.graph.nodes() if n.kind == "derived"]
            assert derived, f"{alias} produced no node"
            assert derived[0].operation == canonical

    def test_convert_to_numeric_is_refused_with_conversion_specific_advice(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "convert_to_numeric",
                                          "input_refs": ["E1"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "DERIVE REFUSED" in step
        assert "convert" in step.lower()
        assert "not available" in step.lower() or "does not convert" in step.lower()
        assert not [n for n in result.ledger.graph.nodes() if n.kind == "derived"]

    def test_convert_to_numeric_is_not_silently_mapped_onto_an_arithmetic_op(self):
        result = _run(_visit_then(
            {"action": "derive", "args": {"operation": "convert_to_numeric",
                                          "input_refs": ["E1"]}},
            {"action": "finish", "args": {"answer": "x"}}))
        step = [s for s in result.scratchpad if "action=derive" in s][0]
        assert "UNKNOWN_OPERATION" not in step
