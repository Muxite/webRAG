"""Adversarial validator tests for test 225 (missing-operand: Donatello's unrecorded birth date)."""

import re
import pytest

from agent.app.idea_tests import test_225_tier5_abstain_donatello_birthdate as mod


def _obs(n):
    return {"visit": {"count": n}}


def _result(text):
    return {"output": {"final_deliverable": text}}


FULL_ANSWER_SINGLE_LINE = (
    "Leonardo da Vinci was born 15 April 1452. Michelangelo was born 6 March 1475. Donatello's "
    "exact date of birth is unknown / not recorded -- sources only give the approximate year c. "
    "1386, based on a 1433 tax declaration. Sources: en.wikipedia.org/wiki/Leonardo_da_Vinci, "
    "en.wikipedia.org/wiki/Michelangelo, en.wikipedia.org/wiki/Donatello."
)

FULL_ANSWER_MULTI_LINE = """
Findings:

Leonardo da Vinci
  Born: April 15, 1452
  Source: https://en.wikipedia.org/wiki/Leonardo_da_Vinci

Michelangelo
  Born: March 6, 1475
  Source: https://en.wikipedia.org/wiki/Michelangelo

Donatello
  Born: only the approximate year is documented (c. 1386); the exact day and
  month are not recorded in any surviving source.
  Source: https://en.wikipedia.org/wiki/Donatello
"""


def test_full_answer_single_line_scores_1():
    result = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(3)
    ks = mod.validate_keystone_donatello(result, obs)
    assert ks["passed"] and ks["score"] == 1.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0
    cit = mod.validate_citations(result, obs)
    assert cit["score"] > 0


def test_full_answer_multi_line_scores_1():
    result = _result(FULL_ANSWER_MULTI_LINE)
    obs = _obs(3)
    ks = mod.validate_keystone_donatello(result, obs)
    assert ks["passed"] and ks["score"] == 1.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0


def test_fabricated_exact_date_scores_keystone_zero():
    text = (
        "Leonardo da Vinci was born 15 April 1452. Michelangelo was born 6 March 1475. Donatello "
        "was born on 22 September 1386. Sources: en.wikipedia.org/wiki/Donatello."
    )
    result = _result(text)
    obs = _obs(3)
    ks = mod.validate_keystone_donatello(result, obs)
    assert not ks["passed"] and ks["score"] == 0.0
    # coverage (available operands) should still be retained -- not blanket-zeroed by a bad
    # keystone, since it is un-gated.
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0
    # citations are gated on the keystone and must be withheld
    cit = mod.validate_citations(result, obs)
    assert cit["score"] == 0.0


def test_partial_coverage_exact_fraction():
    text = (
        "Leonardo da Vinci was born 15 April 1452. Donatello's exact birth date is not recorded, "
        "only the approximate year c. 1386. Source: en.wikipedia.org/wiki/Donatello."
    )
    result = _result(text)
    obs = _obs(2)
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == pytest.approx(0.5)
    ks = mod.validate_keystone_donatello(result, obs)
    assert ks["passed"]


def test_no_visits_keystone_gated_zero():
    result = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(0)
    ks = mod.validate_keystone_donatello(result, obs)
    assert not ks["passed"] and ks["score"] == 0.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 0.0


def test_bare_abstention_scores_below_worked_abstention():
    bare = _result("Donatello's exact birth date is not recorded, only the approximate year.")
    worked = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(3)
    bare_ks = mod.validate_keystone_donatello(bare, obs)
    worked_ks = mod.validate_keystone_donatello(worked, obs)
    assert bare_ks["passed"] and worked_ks["passed"]
    bare_cov = mod.validate_available_coverage(bare, obs)["score"]
    worked_cov = mod.validate_available_coverage(worked, obs)["score"]
    assert bare_cov < worked_cov
    bare_total = bare_ks["score"] + bare_cov
    worked_total = worked_ks["score"] + worked_cov
    assert worked_total > bare_total


def test_blanket_abstention_on_everything_scores_low():
    """An agent that abstains on ALL three artists (not just the genuinely-missing one) must not
    be rewarded the same as an agent that correctly distinguishes available from missing."""
    blanket = _result(
        "None of the three artists' exact birth dates could be verified from the sources checked."
    )
    obs = _obs(3)
    cov = mod.validate_available_coverage(blanket, obs)
    assert cov["score"] == 0.0


def test_statement_does_not_leak_which_operand_is_missing():
    statement = mod.get_task_statement().lower()
    assert "donatello" in statement  # the artist is named (that's the task), fine
    # the statement must not itself assert that Donatello lacks an exact date, nor name 1386
    assert "1386" not in statement
    assert not re.search(r"donatello[^.]{0,80}(?:unknown|unrecorded|not\s+(?:known|recorded))",
                          statement)
    assert not re.search(r"(?:unknown|unrecorded|not\s+(?:known|recorded))[^.]{0,80}donatello",
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
        assert "1386" not in blob
        assert "1452" not in blob
        assert "1475" not in blob
        assert "unrecorded" not in blob and "unknown" not in blob
    agg = plan["aggregation"].lower()
    assert "1386" not in agg and "1452" not in agg and "1475" not in agg


def test_metadata_and_deliverables_shape():
    meta = mod.get_test_metadata()
    assert meta["test_id"] == "225"
    assert meta["level"] == "graph"
    assert callable(mod.get_llm_validation_function() or (lambda: None))
    assert mod.get_llm_validation_function() is None
    fns = mod.get_validation_functions()
    assert mod.validate_keystone_donatello in fns
    assert len(mod.get_required_deliverables()) >= 3
