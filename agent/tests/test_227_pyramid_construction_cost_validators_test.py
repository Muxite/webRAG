"""Adversarial validator tests for test 227 (missing-operand: Great Pyramid's unrecorded cost)."""

import re
import pytest

from agent.app.idea_tests import test_227_tier5_abstain_pyramid_construction_cost as mod


def _obs(n):
    return {"visit": {"count": n}}


def _result(text):
    return {"output": {"final_deliverable": text}}


FULL_ANSWER_SINGLE_LINE = (
    "The Eiffel Tower's construction cost was 6.5 million francs. The Statue of Liberty's "
    "pedestal cost the United States up to $300,000 to build. The Sydney Opera House's final "
    "construction cost was $102 million. The Great Pyramid of Giza's original construction cost "
    "is not recorded -- no financial records survive from its construction (modern estimates of "
    "what it would cost to rebuild today range from about $1.2 billion to $7 billion, but that is "
    "a different, hypothetical modern figure, not the ancient original cost). Sources: "
    "en.wikipedia.org/wiki/Eiffel_Tower, en.wikipedia.org/wiki/Statue_of_Liberty, "
    "en.wikipedia.org/wiki/Sydney_Opera_House, en.wikipedia.org/wiki/Great_Pyramid_of_Giza."
)

FULL_ANSWER_MULTI_LINE = """
Findings:

Eiffel Tower
  Original construction cost: 6.5 million francs
  Source: https://en.wikipedia.org/wiki/Eiffel_Tower

Statue of Liberty (pedestal)
  Original construction cost: up to $300,000
  Source: https://en.wikipedia.org/wiki/Statue_of_Liberty

Sydney Opera House
  Original construction cost: $102 million
  Source: https://en.wikipedia.org/wiki/Sydney_Opera_House

Great Pyramid of Giza
  No original construction cost is recorded; no financial records
  survive from antiquity. A modern rebuild-cost estimate (roughly
  $5 billion in today's money) circulates, but that is a hypothetical
  modern figure, not the ancient original cost.
  Source: https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza
"""


def test_full_answer_single_line_scores_1():
    result = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(4)
    ks = mod.validate_keystone_pyramid(result, obs)
    assert ks["passed"] and ks["score"] == 1.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0
    cit = mod.validate_citations(result, obs)
    assert cit["score"] > 0


def test_full_answer_multi_line_scores_1():
    result = _result(FULL_ANSWER_MULTI_LINE)
    obs = _obs(4)
    ks = mod.validate_keystone_pyramid(result, obs)
    assert ks["passed"] and ks["score"] == 1.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0


def test_fabricated_original_cost_scores_keystone_zero():
    text = (
        "The Eiffel Tower's construction cost was 6.5 million francs. The Statue of Liberty's "
        "pedestal cost up to $300,000. The Sydney Opera House cost $102 million. The Great "
        "Pyramid of Giza's original construction cost was $5 billion. Source: "
        "en.wikipedia.org/wiki/Great_Pyramid_of_Giza."
    )
    result = _result(text)
    obs = _obs(4)
    ks = mod.validate_keystone_pyramid(result, obs)
    assert not ks["passed"] and ks["score"] == 0.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 1.0
    cit = mod.validate_citations(result, obs)
    assert cit["score"] == 0.0


def test_hedged_modern_estimate_mention_still_passes_keystone():
    text = (
        "The Great Pyramid's original construction cost is not recorded. A modern estimate of "
        "what it would cost to rebuild today is roughly $5 billion, but that is a hypothetical "
        "present-day figure, not the ancient original cost."
    )
    result = _result(text)
    obs = _obs(2)
    ks = mod.validate_keystone_pyramid(result, obs)
    assert ks["passed"]


def test_partial_coverage_exact_fraction():
    text = (
        "The Eiffel Tower's construction cost was 6.5 million francs. The Great Pyramid's "
        "original construction cost is not recorded. Source: "
        "en.wikipedia.org/wiki/Great_Pyramid_of_Giza."
    )
    result = _result(text)
    obs = _obs(2)
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == pytest.approx(1 / 3)


def test_no_visits_gated_zero():
    result = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(0)
    ks = mod.validate_keystone_pyramid(result, obs)
    assert not ks["passed"] and ks["score"] == 0.0
    cov = mod.validate_available_coverage(result, obs)
    assert cov["score"] == 0.0


def test_bare_abstention_scores_below_worked_abstention():
    bare = _result("The Great Pyramid's original construction cost is not recorded anywhere.")
    worked = _result(FULL_ANSWER_SINGLE_LINE)
    obs = _obs(4)
    bare_ks = mod.validate_keystone_pyramid(bare, obs)
    worked_ks = mod.validate_keystone_pyramid(worked, obs)
    assert bare_ks["passed"] and worked_ks["passed"]
    bare_cov = mod.validate_available_coverage(bare, obs)["score"]
    worked_cov = mod.validate_available_coverage(worked, obs)["score"]
    assert bare_cov < worked_cov


def test_statement_does_not_leak_which_operand_is_missing():
    statement = mod.get_task_statement().lower()
    assert "pyramid" in statement
    assert not re.search(r"pyramid[^.]{0,100}(?:unrecorded|not\s+recorded|unknown)", statement)
    assert "6.5 million" not in statement
    assert "$102 million" not in statement
    assert "300,000" not in statement


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
        for token in ("6.5 million", "300,000", "102 million"):
            assert token not in blob, f"leaf {leaf['id']} leaks {token!r}"
    agg = plan["aggregation"].lower()
    for token in ("6.5 million", "300,000", "102 million"):
        assert token not in agg


def test_metadata_shape():
    meta = mod.get_test_metadata()
    assert meta["test_id"] == "227"
    assert mod.get_llm_validation_function() is None
    assert mod.validate_keystone_pyramid in mod.get_validation_functions()
