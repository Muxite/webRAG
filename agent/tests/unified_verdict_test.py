"""Offline tests for S3's unified verdict (``testing/execution_evidence_loop.unified_verdict``) —
free, $0.

``Ledger.verdict()`` is RESOLUTION-only by its own docstring: a row whose quote FAILED
verification is still ``ANSWER``. ``unified_verdict`` is a SECOND, additive computation over the
same mechanical signals already present in ``output`` (quote verification, derivation validity,
derivation refusals, resolution) that does not ignore verification. It is gated by
``unified_verdict_enabled`` (default OFF) and never changes ``Ledger.verdict()`` or
``output["ledger_verdict"]``.
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


def _resolve(ledger, index, value, quote_verified, url="https://a.example"):
    row = ledger.rows[index]
    row.status, row.value, row.source_url = el.STATUS_SUPPORTED, value, url
    row.quote, row.quote_verified = f"{value} quote", quote_verified


def _extraction_record(entity, value, quote_verified, quote_fail_reason=None):
    return el.Extraction(entity, "field", value, "SUPPORTED", "https://a.example",
                         "quote text", quote_verified, "p1",
                         quote_fail_reason=quote_fail_reason)


# --------------------------------------------------------------------------------------
# flag: default OFF, opt-in via IDEA_TEST_EVIDENCE_LOOP_UNIFIED_VERDICT
# --------------------------------------------------------------------------------------


def test_unified_verdict_disabled_by_default(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_UNIFIED_VERDICT", raising=False)
    assert el.unified_verdict_enabled() is False


def test_unbacked_number_check_disabled_by_default(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_UNBACKED_NUMBER_CHECK", raising=False)
    assert el.unbacked_number_check_enabled() is False


# --------------------------------------------------------------------------------------
# resolution — mirrors Ledger.verdict()'s own ABSTAIN / PARTIAL clauses
# --------------------------------------------------------------------------------------


def test_abstains_when_nothing_was_obtained():
    ledger = el.Ledger.mint(ENUMERATED)
    counts = el.quote_verification_counts(ledger)
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_ABSTAIN, "reason": None}
    assert ledger.verdict() == el.VERDICT_ABSTAIN  # agrees with the legacy verdict here


def test_partial_when_not_every_row_resolved():
    ledger = el.Ledger.mint(ENUMERATED)
    _resolve(ledger, 0, "English Channel", True)
    counts = el.quote_verification_counts(ledger)
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_PARTIAL, "reason": "unresolved_rows"}


def test_answer_when_every_row_resolved_and_verified_with_no_other_signal():
    ledger = el.Ledger.mint(ENUMERATED)
    for i in range(3):
        _resolve(ledger, i, f"value {i}", True)
    counts = el.quote_verification_counts(ledger)
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_ANSWER, "reason": None}
    assert ledger.verdict() == el.VERDICT_ANSWER  # agrees with the legacy verdict here


# --------------------------------------------------------------------------------------
# the core disagreement: a failed quote does NOT downgrade Ledger.verdict(), but DOES
# downgrade unified_verdict — this is the defect S3 exists to surface.
# --------------------------------------------------------------------------------------


def test_a_run_whose_quote_failed_verification_still_answers_under_the_legacy_verdict():
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", False)  # resolved, quote NOT verified
    assert ledger.verdict() == el.VERDICT_ANSWER  # the exact defect this module's S3 report names


def test_unified_verdict_downgrades_when_a_quote_mechanically_failed():
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", True)
    ledger.extractions.append(
        _extraction_record("(task)", "Toni Morrison", False, el.QUOTE_FAIL_ABSENT))
    counts = el.quote_verification_counts(ledger)
    assert counts["failed"] == 1
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_PARTIAL, "reason": "quote_failed"}


def test_unified_verdict_does_not_downgrade_on_an_unchecked_quote():
    # "unknown" (no page, or an empty quote) is NOT the same claim as "failed" — the tri-state
    # discipline verify_quote already keeps. An unchecked extraction must not read as a failure.
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", True)
    ledger.extractions.append(
        _extraction_record("(task)", "Toni Morrison", None, el.QUOTE_FAIL_NO_PAGE))
    ledger.extractions.append(
        _extraction_record("(task)", "Toni Morrison", None, el.QUOTE_FAIL_EMPTY))
    counts = el.quote_verification_counts(ledger)
    assert counts["failed"] == 0
    assert counts["unchecked"] == 2
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_ANSWER, "reason": None}


def test_unified_verdict_agrees_with_legacy_when_every_quote_verified():
    ledger = el.Ledger.mint(ENUMERATED)
    for i in range(3):
        _resolve(ledger, i, f"v{i}", True)
        ledger.extractions.append(_extraction_record(ledger.rows[i].entity, f"v{i}", True))
    counts = el.quote_verification_counts(ledger)
    assert el.unified_verdict(ledger, counts)["verdict"] == el.VERDICT_ANSWER == ledger.verdict()


# --------------------------------------------------------------------------------------
# derivation validity / refusals — unconditional, unlike Ledger.verdict()'s gated downgrade
# --------------------------------------------------------------------------------------


TOWERS = "Tower A is 590 m tall. Tower B is 566 m tall."


def _extractions(*records):
    return json.dumps({"extractions": [
        {"entity": e, "field": f, "value": v, "verdict": "SUPPORTED", "quote": q, "unit": u}
        for e, f, v, q, u in records]})


_TOWER_RECORDS = (
    ("Tower A", "height", "590 m", "Tower A is 590 m tall.", "m"),
    ("Tower B", "height", "566 m", "Tower B is 566 m tall.", "m"),
)


def _io(replies, page_text=TOWERS, synth="SYNTH"):
    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=lambda **kw: {"messages": kw.get("messages", [])})
    encoded = [r if isinstance(r, str) else json.dumps(r) for r in replies]
    io.query_llm = AsyncMock(side_effect=[*encoded, synth, synth, synth])
    io.search = AsyncMock(return_value=[
        {"title": "t", "url": "https://example.org/towers", "description": "d"}])
    io.visit = AsyncMock(return_value=page_text)
    return io


def _visit_then(*decisions, page=TOWERS):
    return [{"action": "visit", "args": {"url": "https://example.org/towers"}},
            _extractions(*_TOWER_RECORDS), *decisions]


def _run(replies, mandate="How much taller is Tower A than Tower B?", page=TOWERS):
    io = _io(replies, page_text=page)
    return asyncio.run(el.run_evidence_loop(io, mandate, "m", max_steps=6, max_tokens=64))


def test_unified_verdict_downgrades_on_a_known_invalid_derivation_even_though_the_gate_is_off(
    monkeypatch,
):
    monkeypatch.delenv("LEDGER_DERIVATION_GATE", raising=False)
    result = _run(_visit_then(
        {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"],
                                      "proposed_value": "1594"}},
        {"action": "finish", "args": {"answer": "x"}}))
    ledger = result.ledger
    for row in ledger.rows:
        row.status = el.STATUS_SUPPORTED
        row.value = row.value or "590 m"
        row.quote_verified = True
    assert el.derivation_gate_enabled() is False
    assert ledger.verdict() == el.VERDICT_ANSWER  # legacy verdict: unaffected, gate is off
    counts = el.quote_verification_counts(ledger)
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_PARTIAL, "reason": "invalid_derivation"}


def test_unified_verdict_downgrades_on_a_refused_derivation():
    ledger = el.Ledger.mint("How much taller is Tower A than Tower B?")
    _resolve(ledger, 0, "590 m", True)
    graph = ledger.ensure_graph()
    from agent.app.testing.evidence_graph import UnitMismatch
    graph.record_refusal("difference", ["E1", "E2"], UnitMismatch("units differ"))
    counts = el.quote_verification_counts(ledger)
    result = el.unified_verdict(ledger, counts)
    assert result == {"verdict": el.VERDICT_PARTIAL, "reason": "derivation_refused"}


# --------------------------------------------------------------------------------------
# the "163" case: unbacked_numeric_claims / the opt-in unified_verdict check
# --------------------------------------------------------------------------------------


def test_unbacked_numeric_claims_flags_a_number_stated_in_prose_but_never_a_row_or_node_value():
    ledger = el.Ledger.mint("How many floors does the Burj Khalifa have?")
    _resolve(ledger, 0, "154 + 9 maintenance", True)
    deliverable = "The Burj Khalifa has 163 floors (154 + 9 maintenance)."
    unbacked = el.unbacked_numeric_claims(deliverable, ledger)
    assert unbacked == ["163"]


def test_unbacked_numeric_claims_credits_numbers_present_in_the_row_value():
    ledger = el.Ledger.mint("How many floors does the Burj Khalifa have?")
    _resolve(ledger, 0, "154 + 9 maintenance", True)
    deliverable = "154 occupied floors and 9 maintenance floors are reported."
    assert el.unbacked_numeric_claims(deliverable, ledger) == []


def test_unbacked_numeric_claims_credits_a_derived_nodes_own_value():
    # A value the model obtained through the typed `derive` action IS a real D-handle in the
    # graph — mechanically distinct from doing the same arithmetic silently in prose.
    result = _run(_visit_then(
        {"action": "derive", "args": {"operation": "difference", "input_refs": ["E1", "E2"]}},
        {"action": "finish", "args": {"answer": "The difference is 24 m."}}))
    ledger = result.ledger
    assert el.unbacked_numeric_claims("The difference is 24 m.", ledger) == []


def test_unbacked_numeric_claims_is_empty_when_the_deliverable_has_no_numeric_claim():
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", True)
    assert el.unbacked_numeric_claims("Toni Morrison wrote it.", ledger) == []


def test_unified_verdict_ignores_unbacked_numbers_when_the_check_flag_is_off():
    ledger = el.Ledger.mint("How many floors does the Burj Khalifa have?")
    _resolve(ledger, 0, "154 + 9 maintenance", True)
    ledger.extractions.append(
        _extraction_record(ledger.rows[0].entity, "154 + 9 maintenance", True))
    counts = el.quote_verification_counts(ledger)
    deliverable = "The Burj Khalifa has 163 floors (154 + 9 maintenance)."
    result = el.unified_verdict(ledger, counts, deliverable=deliverable,
                                check_unbacked_numbers=False)
    assert result == {"verdict": el.VERDICT_ANSWER, "reason": None}


def test_unified_verdict_catches_the_163_case_when_the_check_flag_is_on():
    ledger = el.Ledger.mint("How many floors does the Burj Khalifa have?")
    _resolve(ledger, 0, "154 + 9 maintenance", True)
    ledger.extractions.append(
        _extraction_record(ledger.rows[0].entity, "154 + 9 maintenance", True))
    counts = el.quote_verification_counts(ledger)
    deliverable = "The Burj Khalifa has 163 floors (154 + 9 maintenance)."
    result = el.unified_verdict(ledger, counts, deliverable=deliverable,
                                check_unbacked_numbers=True)
    assert result == {"verdict": el.VERDICT_PARTIAL, "reason": "unbacked_numeric_claim",
                      "unbacked_numbers": ["163"]}


def test_unified_verdict_reads_the_flag_when_check_unbacked_numbers_is_not_passed(monkeypatch):
    ledger = el.Ledger.mint("How many floors does the Burj Khalifa have?")
    _resolve(ledger, 0, "154 + 9 maintenance", True)
    ledger.extractions.append(
        _extraction_record(ledger.rows[0].entity, "154 + 9 maintenance", True))
    counts = el.quote_verification_counts(ledger)
    deliverable = "The Burj Khalifa has 163 floors (154 + 9 maintenance)."
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_UNBACKED_NUMBER_CHECK", raising=False)
    assert el.unified_verdict(ledger, counts, deliverable=deliverable) == {
        "verdict": el.VERDICT_ANSWER, "reason": None}
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_UNBACKED_NUMBER_CHECK", "1")
    assert el.unified_verdict(ledger, counts, deliverable=deliverable)["reason"] == \
        "unbacked_numeric_claim"


# --------------------------------------------------------------------------------------
# wiring: the new verdict rides alongside the legacy one in `output`, only when the flag is on
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_carries_no_unified_verdict_key_by_default(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_UNIFIED_VERDICT", raising=False)
    tm = MagicMock()
    tm.metadata = {"test_id": "001"}
    tm.get_task_statement = MagicMock(return_value="Who wrote Beloved?")
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", False)  # unverified — would disagree if the flag were on

    async def fake_loop(*a, **kw):
        return el.EvidenceLoopResult(deliverable="Toni Morrison", ledger=ledger, scratchpad=[],
                                     verdict=ledger.verdict(), pages=[])

    monkeypatch.setattr(el, "run_evidence_loop", fake_loop)
    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    assert output["ledger_verdict"] == el.VERDICT_ANSWER  # legacy verdict: unchanged
    assert "unified_verdict" not in output
    assert "unified_verdict_reason" not in output


@pytest.mark.asyncio
async def test_output_carries_the_unified_verdict_when_the_flag_is_on_and_it_disagrees(
    monkeypatch,
):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_UNIFIED_VERDICT", "1")
    tm = MagicMock()
    tm.metadata = {"test_id": "001"}
    tm.get_task_statement = MagicMock(return_value="Who wrote Beloved?")
    ledger = el.Ledger.mint("Who wrote Beloved?")
    _resolve(ledger, 0, "Toni Morrison", True)
    ledger.extractions.append(
        _extraction_record(ledger.rows[0].entity, "Toni Morrison", False, el.QUOTE_FAIL_ABSENT))

    async def fake_loop(*a, **kw):
        return el.EvidenceLoopResult(deliverable="Toni Morrison", ledger=ledger, scratchpad=[],
                                     verdict=ledger.verdict(), pages=[])

    monkeypatch.setattr(el, "run_evidence_loop", fake_loop)
    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    assert output["ledger_verdict"] == el.VERDICT_ANSWER
    assert output["unified_verdict"] == el.VERDICT_PARTIAL
    assert output["unified_verdict_reason"] == "quote_failed"
