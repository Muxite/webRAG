"""Offline tests for `ledger_api.recheck_claim` -- the consumer-side audit of a pinned claim.

Motivation, measured on a real acceptance run (qwen2.5:7b, offline corpus, LLM_SEED=12345):

    claim value='154 + 9 maintenance'  quote_verified=True  pin=p1[936:967]
    page_text[936:967] == 'Floor\ncount\n154 + 9 maintenance'
    claim.quote        == 'Floor count\n154 + 9 maintenance'
    text[start:end] == quote  ->  False

The ledger is right: `_locate` matches with whitespace runs collapsed and returns RAW page
offsets, so the WORDS are identical and only whitespace differs. But a consumer handed
`quote` + `quote_start` + `quote_end` and doing the obvious exact comparison concludes the
evidence was tampered with. These tests pin the rule a consumer should actually use.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.app.ledger_api import Claim, recheck_claim  # noqa: E402

_PAGE = {"page_id": "p1", "url": "https://example.org/a",
         "text": "Burj Khalifa\nFloor\ncount\n154 + 9 maintenance\nend"}
_START = _PAGE["text"].index("Floor")
_END = _PAGE["text"].index("\nend")


def _claim(**over):
    base = dict(entity="Burj Khalifa", field="floors", value="154 + 9 maintenance", unit="",
                status="SUPPORTED", resolved=True, confidence_tier="", source_url=_PAGE["url"],
                quote="Floor count\n154 + 9 maintenance", quote_verified=True,
                quote_verification="verified", page_id="p1",
                quote_start=_START, quote_end=_END, evidence_node_id="n1")
    base.update(over)
    return Claim(**base)


def test_whitespace_variant_at_the_pin_rechecks_true():
    """The exact real-run shape: page has `Floor\\ncount`, model quoted `Floor count`."""
    ok, detail = recheck_claim(_claim(), [_PAGE])
    assert ok is True, detail
    assert detail == "pinned"


def test_exact_slice_equality_would_have_failed_this_claim():
    """Guards the reason this function exists -- if this ever becomes equal, the pin changed."""
    c = _claim()
    assert _PAGE["text"][c.quote_start:c.quote_end] != c.quote


def test_a_quote_whose_words_differ_from_the_pin_rechecks_false():
    ok, detail = recheck_claim(_claim(quote="Floor count 155 + 9 maintenance"), [_PAGE])
    assert ok is False
    assert detail == "pin_mismatch"


def test_offsets_outside_the_page_recheck_false_rather_than_raising():
    ok, detail = recheck_claim(_claim(quote_start=9000, quote_end=9100), [_PAGE])
    assert ok is False
    assert detail == "pin_mismatch"


def test_an_unpinned_but_verified_claim_falls_back_to_scanning_the_page():
    """`evidence_node_id` set with offsets -1 is a real observed shape (value located
    mechanically, quote paraphrased) -- the words are still checkable."""
    ok, detail = recheck_claim(_claim(quote_start=-1, quote_end=-1), [_PAGE])
    assert ok is True
    assert detail == "unpinned_match"


def test_an_unpinned_claim_whose_words_are_absent_rechecks_false():
    ok, detail = recheck_claim(_claim(quote_start=-1, quote_end=-1, quote="not on this page"),
                               [_PAGE])
    assert ok is False
    assert detail == "absent"


def test_a_claim_naming_a_page_that_was_not_supplied_is_unknown_not_false():
    """Absent is never zero: no page in hand is not evidence of a bad quote."""
    ok, detail = recheck_claim(_claim(page_id="p9"), [_PAGE])
    assert ok is None
    assert detail == "no_page"


def test_an_unverified_claim_stays_unknown():
    ok, detail = recheck_claim(_claim(quote_verified=None, quote=""), [_PAGE])
    assert ok is None
    assert detail == "no_quote"


def test_recheck_disagreeing_with_the_stored_verdict_is_reported_not_silently_trusted():
    """A row stored as verified whose quote is NOT on the page must recheck False -- this is the
    tampering/drift detector, so it must never inherit `quote_verified`."""
    ok, _ = recheck_claim(_claim(quote="entirely fabricated text", quote_start=-1, quote_end=-1),
                          [_PAGE])
    assert ok is False


def test_pages_may_be_objects_with_attributes_not_only_dicts():
    class _P:
        page_id, text = "p1", _PAGE["text"]
    ok, _ = recheck_claim(_claim(), [_P()])
    assert ok is True
