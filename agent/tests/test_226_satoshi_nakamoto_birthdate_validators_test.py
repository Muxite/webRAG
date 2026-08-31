"""Adversarial validator tests for test 226 (missing-operand: Satoshi Nakamoto's unknown birth date)."""

import re
import pytest

from agent.app.idea_tests import test_226_tier5_abstain_satoshi_nakamoto_birthdate as mod


def _obs(n):
    return {"visit": {"count": n}}


def _result(text):
    return {"output": {"final_deliverable": text}}


FULL_ANSWER_SINGLE_LINE = (
    "Linus Torvalds was born 28 December 1969. Guido van Rossum was born 31 January 1956. Tim "
    "Berners-Lee was born 8 June 1955. Satoshi Nakamoto's real identity has never been confirmed, "
    "so no verified birth date exists (some fan sites cite an unverified/symbolic April 5 1975 "
    "date, but it is not documented fact). Sources: en.wikipedia.org/wiki/Linus_Torvalds, "
    "en.wikipedia.org/wiki/Guido_van_Rossum, en.wikipedia.org/wiki/Tim_Berners-Lee, "
    "en.wikipedia.org/wiki/Satoshi_Nakamoto."
)

FULL_ANSWER_MULTI_LINE = """
Findings:

Linus Torvalds
  Born: December 28, 1969
  Source: https://en.wikipedia.org/wiki/Linus_Torvalds

Guido van Rossum
  Born: January 31, 1956
  Source: https://en.wikipedia.org/wiki/Guido_van_Rossum

Tim Berners-Lee
  Born: June 8, 1955
  Source: https://en.wikipedia.org/wiki/Tim_Berners-Lee

Satoshi Nakamoto
  Identity: never confirmed despite years of investigation; no verified
  birth date exists. A specific date circulates on some sites but is
  explicitly unverified/symbolic, not a documented fact.
  Source: https://en.wikipedia.org/wiki/Satoshi_Nakamoto
"""


def test_full_answer_single_line_scores_1():
    result = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(4)
    ks = mod.validate_keystone_nakamoto(result, obs)
    assert ks["passed"] and ks["score"] == 1.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0
    cit = mod.validate_citations(result, obs)
    assert cit["score"] > 0


def test_full_answer_multi_line_scores_1():
    result = _result(FULL_ANSWER_MULTI_LINE)
    obs = _obs(4)
    ks = mod.validate_keystone_nakamoto(result, obs)
    assert ks["passed"] and ks["score"] == 1.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0


def test_fabricated_confident_date_scores_keystone_zero():
    text = (
        "Linus Torvalds was born 28 December 1969. Guido van Rossum was born 31 January 1956. "
        "Tim Berners-Lee was born 8 June 1955. Satoshi Nakamoto was born on April 5, 1975. "
        "Source: en.wikipedia.org/wiki/Satoshi_Nakamoto."
    )
    result = _result(text)
    obs = _obs(4)
    ks = mod.validate_keystone_nakamoto(result, obs)
    assert not ks["passed"] and ks["score"] == 0.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0
    cit = mod.validate_citations(result, obs)
    assert cit["score"] == 0.0


def test_hedged_mention_of_rumored_date_still_passes_keystone():
    """The keystone should not punish a run that MENTIONS the circulating unverified date as
    long as it is clearly hedged and identity is stated as unconfirmed."""
    text = (
        "Satoshi Nakamoto's identity has never been confirmed. Some unverified/claimed sources "
        "cite April 5, 1975 as a symbolic placeholder date, but no reliable source confirms a "
        "real birth date."
    )
    result = _result(text)
    obs = _obs(2)
    ks = mod.validate_keystone_nakamoto(result, obs)
    assert ks["passed"]


def test_partial_coverage_exact_fraction():
    text = (
        "Linus Torvalds was born 28 December 1969. Satoshi Nakamoto's identity is unknown; no "
        "verified birth date exists. Source: en.wikipedia.org/wiki/Satoshi_Nakamoto."
    )
    result = _result(text)
    obs = _obs(2)
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == pytest.approx(1 / 3)


def test_no_visits_gated_zero():
    result = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(0)
    ks = mod.validate_keystone_nakamoto(result, obs)
    assert not ks["passed"] and ks["score"] == 0.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 0.0


def test_bare_abstention_scores_below_worked_abstention():
    bare = _result("Satoshi Nakamoto's identity has never been confirmed; no birth date exists.")
    worked = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(4)
    bare_ks = mod.validate_keystone_nakamoto(bare, obs)
    worked_ks = mod.validate_keystone_nakamoto(worked, obs)
    assert bare_ks["passed"] and worked_ks["passed"]
    bare_cov = mod.validate_available_coverage(bare, obs)["score"]
    worked_cov = mod.validate_available_coverage(worked, obs)["score"]
    assert bare_cov < worked_cov


def test_statement_does_not_leak_which_operand_is_missing():
    statement = mod.get_task_statement().lower()
    assert "nakamoto" in statement
    assert "1975" not in statement
    assert not re.search(r"nakamoto[^.]{0,100}(?:unknown|unconfirmed|never\s+confirmed)",
                          statement)


def test_compiled_plan_well_formed_and_leaks_nothing():
    plan = mod.get_compiled_plan()
    assert set(plan.keys()) == {"leaves", "aggregation"}
    ids = set()
    for leaf in plan["leaves"]:
        assert set(leaf.keys()) == {"id", "instruction", "expect", "depends_on"}
        assert leaf["id"] not in ids
        ids.add(leaf["id"])
        assert leaf["depends_on"] == []
        blob = (leaf["instruction"] + " " + leaf["expect"]).lower()
        # date/year values must never leak. "unconfirmed"/"unknown" appear identically in the
        # SAME conditional boilerplate on every leaf (never singling one person out), so they are
        # not a leak of which entry is actually missing -- only the values below are checked.
        for token in ("1969", "1956", "1955", "1975"):
            assert token not in blob, f"leaf {leaf['id']} leaks {token!r}"
    agg = plan["aggregation"].lower()
    for token in ("1969", "1956", "1955", "1975"):
        assert token not in agg


def test_metadata_shape():
    meta = mod.get_test_metadata()
    assert meta["test_id"] == "226"
    assert mod.get_llm_validation_function() is None
    assert mod.validate_keystone_nakamoto in mod.get_validation_functions()
