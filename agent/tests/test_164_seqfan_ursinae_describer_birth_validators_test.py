"""
Offline adversarial tests for task 164 (sequential prefix -> 6-way fan-out -> merge).

Free (no LLM, no network): synthetic answers + observability payloads assert the designed
score shape — bimodal on the keystone, un-gated on the breadth diagnostic:

  * a full answer scores 1.0 on every check, in BOTH a single-line and a multi-line layout;
  * the wrong-AXIS answer (description year instead of describer birth year) loses the keystone
    and the gated secondaries while KEEPING its coverage credit;
  * partial branch resolution scores the exact fraction and fails the keystone;
  * a fabricated 0-visit answer scores 0 on the visit gate and the keystone;
  * a bare table row (no superlative) does not trip the keystone regex;
  * the compiled plan is schema-valid and leaks no member, describer, year or the answer.
"""
import re

import pytest

from agent.app.idea_tests import test_164_tier5_seqfan_ursinae_describer_birth as t
from agent.app.testing.compiled_plan import normalize_plan, topological_waves, validate_plan


def _result(text: str):
    return {"output": {"final_deliverable": text}}


_EVIDENCE = {
    "visited": [
        {"url": "https://en.wikipedia.org/wiki/Wojtek", "content": "was a Syrian brown bear"},
        {"url": "https://en.wikipedia.org/wiki/Syrian_brown_bear",
         "content": "Ursus arctos syriacus is a subspecies of brown bear"},
        {"url": "https://en.wikipedia.org/wiki/Brown_bear",
         "content": "Subfamily: Ursinae Genus: Ursus Ursus arctos Linnaeus, 1758"},
        {"url": "https://en.wikipedia.org/wiki/Sun_bear", "content": "Helarctos malayanus"},
        {"url": "https://en.wikipedia.org/wiki/Stamford_Raffles", "content": "5 July 1781"},
    ]
}


def _obs(n_visits: int = 15, evidence: bool = True):
    obs = {"visit": {"count": n_visits}}
    if evidence:
        obs["evidence"] = _EVIDENCE
    return obs


_TABLE = (
    "Brown bear (Ursus arctos) - Carl Linnaeus, 1758 - born 1707 - "
    "https://en.wikipedia.org/wiki/Carl_Linnaeus\n"
    "Polar bear (Ursus maritimus) - Constantine Phipps, 1774 - born 1744 - "
    "https://en.wikipedia.org/wiki/Constantine_Phipps,_2nd_Baron_Mulgrave\n"
    "American black bear (Ursus americanus) - Peter Simon Pallas, 1780 - born 1741 - "
    "https://en.wikipedia.org/wiki/Peter_Simon_Pallas\n"
    "Sloth bear (Melursus ursinus) - George Shaw, 1791 - born 1751 - "
    "https://en.wikipedia.org/wiki/George_Shaw_(biologist)\n"
    "Asian black bear (Ursus thibetanus) - Georges Cuvier, 1823 - born 1769 - "
    "https://en.wikipedia.org/wiki/Georges_Cuvier\n"
    "Sun bear (Helarctos malayanus) - Stamford Raffles, 1821 - born 1781 - "
    "https://en.wikipedia.org/wiki/Stamford_Raffles\n"
    "Chain: Wojtek -> Syrian brown bear (Ursus arctos syriacus) -> brown bear -> subfamily Ursinae."
)

_FULL_ONELINE = (
    "The describer born most recently is Stamford Raffles (born 1781), who described the sun bear "
    "(Helarctos malayanus) in 1821.\n" + _TABLE
)

_FULL_MULTILINE = (
    "Latest-born describer:\n"
    "Stamford Raffles, born 1781, describer of the sun bear (Helarctos malayanus).\n" + _TABLE
)


def _checks(result, obs):
    return {c["check"]: c for c in (
        t.validate_visits(result, obs),
        t.validate_keystone_latest_born_describer(result, obs),
        t.validate_coverage(result, obs),
        t.validate_chain_prefix(result, obs),
        t.validate_citations(result, obs),
    )}


@pytest.mark.parametrize("text", [_FULL_ONELINE, _FULL_MULTILINE])
def test_full_answer_scores_every_check(text):
    c = _checks(_result(text), _obs())
    assert c["keystone_latest_born_describer"]["score"] == 1.0
    assert c["coverage"]["score"] == 1.0
    assert c["chain_prefix"]["score"] == 1.0
    assert c["citations"]["score"] == 1.0
    assert c["visit_count"]["score"] == 1.0
    assert all(v["passed"] for v in c.values())


def test_wrong_axis_answer_loses_keystone_but_keeps_coverage():
    text = ("The most recently described species is the Asian black bear (Ursus thibetanus), "
            "described latest by Georges Cuvier in 1823.\n" + _TABLE)
    c = _checks(_result(text), _obs())
    assert c["keystone_latest_born_describer"]["score"] == 0.0
    assert c["coverage"]["score"] == 1.0
    assert c["chain_prefix"]["score"] == 0.0
    assert c["citations"]["score"] == 0.0


def test_bare_table_row_does_not_trip_the_keystone():
    c = _checks(_result(_TABLE), _obs())
    assert c["keystone_latest_born_describer"]["score"] == 0.0
    assert c["coverage"]["score"] == 1.0


def test_partial_branch_resolution_scores_the_exact_fraction():
    rows = _TABLE.splitlines()
    partial = "\n".join(rows[:3])
    c = _checks(_result(partial), _obs(n_visits=6))
    assert c["coverage"]["score"] == pytest.approx(3 / 6)
    assert not c["coverage"]["passed"]
    assert c["keystone_latest_born_describer"]["score"] == 0.0
    assert not c["visit_count"]["passed"]


def test_five_of_six_branches_still_fails_coverage():
    rows = _TABLE.splitlines()
    partial = "\n".join(rows[:5])
    c = _checks(_result(partial), _obs())
    assert c["coverage"]["score"] == pytest.approx(5 / 6)
    assert not c["coverage"]["passed"]
    assert c["keystone_latest_born_describer"]["score"] == 0.0


def test_fabricated_zero_visit_answer_scores_zero_on_gates():
    c = _checks(_result(_FULL_ONELINE), _obs(n_visits=0, evidence=False))
    assert c["visit_count"]["score"] == 0.0 and not c["visit_count"]["passed"]
    assert c["keystone_latest_born_describer"]["score"] == 0.0
    assert c["chain_prefix"]["score"] == 0.0
    assert c["citations"]["score"] == 0.0
    assert c["coverage"]["score"] == 1.0


def test_keystone_without_the_birth_year_is_not_credited():
    text = "The latest-born describer is Stamford Raffles, who described the sun bear.\n"
    c = _checks(_result(text), _obs())
    assert c["keystone_latest_born_describer"]["score"] == 0.0


def test_chain_prefix_needs_page_evidence_not_just_names():
    obs = _obs()
    obs["evidence"] = {"visited": [{"url": "https://en.wikipedia.org/wiki/Stamford_Raffles",
                                    "content": "5 July 1781"}]}
    c = _checks(_result(_FULL_ONELINE), obs)
    assert c["keystone_latest_born_describer"]["score"] == 1.0
    assert c["chain_prefix"]["score"] == 0.0


def test_validator_dicts_have_the_required_shape():
    c = _checks(_result(_FULL_ONELINE), _obs())
    for v in c.values():
        assert set(v) == {"check", "passed", "score", "reason"}
        assert isinstance(v["passed"], bool) and 0.0 <= v["score"] <= 1.0 and v["reason"]


def test_metadata_and_api_surface():
    m = t.get_test_metadata()
    assert m["test_id"] == "164" and m["level"] == "graph" and m["weight"] == "long"
    assert t.get_llm_validation_function() is None
    assert len(t.get_validation_functions()) == 5
    assert len(t.get_required_deliverables()) >= 3 and len(t.get_success_criteria()) >= 3


def test_compiled_plan_is_schema_valid_with_a_dependent_prefix_and_six_branches():
    plan = validate_plan(normalize_plan(t.get_compiled_plan()))
    ids = [leaf["id"] for leaf in plan["leaves"]]
    assert len(ids) == 4 + 2 * t.SET_SIZE == len(set(ids))
    waves = topological_waves(plan["leaves"])
    assert [len(w) for w in waves] == [1, 1, 1, 1, t.SET_SIZE, t.SET_SIZE]
    for leaf in plan["leaves"]:
        for dep in leaf["depends_on"]:
            assert "{" + dep + "}" in leaf["instruction"]


_FORBIDDEN = [
    "raffles", "1781", "sun bear", "helarctos", "malayanus", "ursinae", "ursus", "bear",
    "syrian", "linnaeus", "pallas", "phipps", "mulgrave", "shaw", "cuvier", "sloth", "polar",
    "panda", "melursus", "thibetanus", "arctos", "maritimus", "americanus",
    "1758", "1774", "1780", "1791", "1821", "1823", "1707", "1741", "1744", "1751", "1769",
]


def _plan_text(plan):
    parts = [plan["aggregation"]]
    for leaf in plan["leaves"]:
        parts += [leaf["id"], leaf["instruction"], leaf["expect"]]
    return " ".join(parts).lower()


def test_compiled_plan_leaks_nothing():
    text = _plan_text(normalize_plan(t.get_compiled_plan()))
    leaked = [w for w in _FORBIDDEN if re.search(r"\b" + re.escape(w) + r"\b", text)]
    assert leaked == [], f"compiled plan leaks: {leaked}"


def test_task_statement_leaks_nothing_beyond_the_given_start():
    text = t.get_task_statement().lower()
    leaked = [w for w in _FORBIDDEN if re.search(r"\b" + re.escape(w) + r"\b", text)]
    assert leaked == [], f"task statement leaks: {leaked}"
    assert "wojtek" in text
