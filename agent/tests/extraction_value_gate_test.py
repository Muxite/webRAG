"""Offline tests for the extraction-time value gate (``IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE``).

Measured on the tuning split of the stored ``ledgernum22r3`` campaign (re-checking each
extraction's ``value`` against the page it cites via
``evidence_graph.verify_value_against_stored_page``): ``evidence_loop`` mints 332 extractions
of which 138 (41.6%) cite a page that does NOT contain the value; ``sequential_react_extract``
mints 334 of which 110 (32.9%) do the same. Both arms already compute this per-record (the
``value_verified`` field) and mint the extraction as a supported record regardless.

These tests pin the gate that stops that: with the flag OFF (the shipped default) behaviour is
byte-for-byte the pre-existing one; with it ON, a record whose value was not literally located on
the page it cites (``value_verified is not True`` — covers a mechanically confirmed ``absent``
value as well as an empty/junk one, since neither is a real citation) no longer resolves a ledger
row to SUPPORTED, while the record itself is still appended so nothing is deleted from the audit
trail. The companion test below is the anti-gaming check the brief calls out by name: a value that
IS on the page under a different surface form (comma grouping, unit spacing, a scale word) must
NOT be gated, because ``verify_value`` already normalizes for exactly that and over-refusing here
would trade real accuracy for a flattered KPI.
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


def _io(page_text, reply):
    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=lambda **kw: {"messages": kw.get("messages", [])})
    io.query_llm = AsyncMock(return_value=reply)
    io.visit = AsyncMock(return_value=page_text)
    return io


def _extraction(entity, field, value, quote, verdict="SUPPORTED", unit=""):
    return json.dumps({"extractions": [
        {"entity": entity, "field": field, "value": value, "verdict": verdict,
         "quote": quote, "unit": unit}]})


def _run_extract(page, value, quote, entity_index=0, unit=""):
    ledger = el.Ledger.mint(ENUMERATED)
    entity = ledger.rows[entity_index].entity
    field = ledger.rows[entity_index].field
    io = _io(page, _extraction(entity, field, value, quote, unit=unit))
    records = asyncio.run(el.extract_from_page(
        io, "m", ENUMERATED, ledger, page_id="p1", page_url="https://a.example",
        page_text=page))
    return ledger, records


# --------------------------------------------------------------------------------------
# flag plumbing
# --------------------------------------------------------------------------------------


def test_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", raising=False)
    assert el.extraction_value_gate_enabled() is False


def test_gate_flag_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "1")
    assert el.extraction_value_gate_enabled() is True
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "0")
    assert el.extraction_value_gate_enabled() is False


# --------------------------------------------------------------------------------------
# gate OFF (shipped default) — behaviour is unchanged from before the gate existed
# --------------------------------------------------------------------------------------


def test_gate_off_a_fabricated_value_still_resolves_the_row(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", raising=False)
    page = "The River Avon, Bristol flows west into the Severn Estuary."
    ledger, records = _run_extract(page, "English Channel", "flows west")
    assert records[0].value_verified is False
    assert records[0].excluded_from_support is False
    assert ledger.rows[0].status == el.STATUS_SUPPORTED
    assert ledger.rows[0].value == "English Channel"


# --------------------------------------------------------------------------------------
# gate ON — a value that is NOT on the page it cites is excluded from support
# --------------------------------------------------------------------------------------


def test_gate_on_a_fabricated_value_is_recorded_but_does_not_resolve_the_row(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "1")
    page = "The River Avon, Bristol flows west into the Severn Estuary."
    ledger, records = _run_extract(page, "English Channel", "flows west")
    # The record survives -- nothing is deleted from the append-only audit trail.
    assert len(records) == 1
    assert records[0].value == "English Channel"
    assert records[0].value_verified is False
    assert records[0].excluded_from_support is True
    # But the ledger row it named is NOT resolved to SUPPORTED by it.
    assert ledger.rows[0].status == el.STATUS_OPEN
    assert ledger.rows[0].value == ""
    assert ledger.extractions[0] is records[0]


def test_gate_on_an_empty_or_junk_value_is_also_excluded_from_support(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "1")
    page = "The River Avon, Bristol flows west into the Severn Estuary."
    ledger, records = _run_extract(page, "https://example.com/some-list", "flows west")
    assert records[0].value_verified is None
    assert records[0].excluded_from_support is True
    assert ledger.rows[0].status == el.STATUS_OPEN


def test_gate_on_still_lets_a_genuinely_grounded_value_resolve_the_row(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "1")
    page = "The River Avon, Hampshire empties into the English Channel at Christchurch."
    ledger, records = _run_extract(
        page, "English Channel", "empties into the English Channel", entity_index=1)
    assert records[0].value_verified is True
    assert records[0].excluded_from_support is False
    assert ledger.rows[1].status == el.STATUS_SUPPORTED
    assert ledger.rows[1].value == "English Channel"


def test_gate_on_an_absent_verdict_is_unaffected(monkeypatch):
    # The gate governs SUPPORTED records with a value; a model-declared ABSENT/BLOCKED verdict
    # never carried a value claim to gate in the first place.
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "1")
    ledger = el.Ledger.mint(ENUMERATED)
    io = _io("unrelated", _extraction(
        ledger.rows[2].entity, ledger.rows[2].field, "", "", verdict="ABSENT"))
    asyncio.run(el.extract_from_page(io, "m", ENUMERATED, ledger, page_id="p1",
                                     page_url="https://example.com", page_text="unrelated"))
    assert ledger.rows[2].status == el.STATUS_ABSENT


# --------------------------------------------------------------------------------------
# anti-gaming: a value present under a DIFFERENT SURFACE FORM must never be gated
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("page,value,unit,quote,label", [
    ("The Akashi Kaikyo Bridge spans 1,991 metres.", "1991 metres", "",
     "spans 1,991 metres", "digit-group comma"),
    ("Elevation: 590m above sea level.", "590 m", "",
     "Elevation: 590m", "unit spacing"),
    # A value whose UNIT is spelled differently from the ``unit`` field the model declared
    # alongside it ("ft" in the value, "feet" reported separately): verify_value's own
    # documented case 3 (evidence_graph._candidates) respells the value with the declared unit
    # before giving up, so this is a real formatting difference, not a fabrication.
    ("The peak rises to 20,310 feet above sea level.", "20,310 ft", "feet",
     "rises to 20,310 feet", "unit field respelling"),
])
def test_over_refusal_a_reformatted_value_still_passes_the_gate(monkeypatch, page, value, unit,
                                                                 quote, label):
    monkeypatch.setenv("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", "1")
    ledger, records = _run_extract(page, value, quote, unit=unit)
    assert records[0].excluded_from_support is False, label
    assert ledger.rows[0].status == el.STATUS_SUPPORTED, label
