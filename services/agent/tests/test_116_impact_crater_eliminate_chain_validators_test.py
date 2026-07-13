"""Offline unit tests for test 116 (impact craters -> Vredefort age). Free, no LLM.

Covers the leak-resistant keystone gate (Vredefort age 2.023 Ga), the UN-gated candidate-coverage
diagnostic (retained even when the terminus is wrong), the keystone-gated survivor/citation
secondaries, the correct answer in single- and multi-line layout, the famous-decoy failure mode
(picking Chicxulub -> keystone 0 but coverage retained), keystone token rejecting near-miss numbers,
visit gating, and the compiled plan (branch-then-chain, templated, self-describing, leaks nothing).
"""
import re

from agent.app.idea_tests import test_116_tier5_impact_crater_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: Chicxulub crater is the largest mostly INTACT crater, the K-Pg dinosaur crater, ~66 Ma "
    "(https://en.wikipedia.org/wiki/Chicxulub_crater); the Vredefort impact structure in South Africa "
    "is the largest verified impact structure, ~300 km "
    "(https://en.wikipedia.org/wiki/Vredefort_impact_structure); the Sudbury Basin in Ontario is ~130 km "
    "(https://en.wikipedia.org/wiki/Sudbury_Basin); the Chesapeake Bay impact crater in Virginia is "
    "~85 km, Eocene (https://en.wikipedia.org/wiki/Chesapeake_Bay_impact_crater). Stage 2: survivor is "
    "Vredefort. Stage 3: its age is 2.023 Ga (2,023 Ma)."
)

_FULL_MULTI = (
    "STAGE 1 rankings:\n"
    "  Chicxulub crater -> largest mostly intact, dinosaur k-pg, 66 Ma\n"
    "    https://en.wikipedia.org/wiki/Chicxulub_crater\n"
    "  Vredefort impact structure -> largest verified, 300 km, South Africa\n"
    "    https://en.wikipedia.org/wiki/Vredefort_impact_structure\n"
    "  Sudbury Basin -> 130 km, Ontario\n"
    "    https://en.wikipedia.org/wiki/Sudbury_Basin\n"
    "  Chesapeake Bay impact crater -> 85 km, eocene, Virginia\n"
    "    https://en.wikipedia.org/wiki/Chesapeake_Bay_impact_crater\n"
    "STAGE 2 survivor: Vredefort\n"
    "STAGE 3 age:\n"
    "  2.023\n  Ga\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_survivor_gates_to_zero_but_keeps_coverage():
    wrong = (
        "Chicxulub crater -> largest mostly intact, dinosaur k-pg 66 Ma; Vredefort impact structure -> "
        "300 km South Africa; Sudbury Basin -> 130 km Ontario; Chesapeake Bay impact crater -> 85 km "
        "eocene Virginia. I take the famous Chicxulub dinosaur crater as the largest; its age is 66 Ma."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_near_miss_numbers():
    assert t.validate_keystone(_r("code 12.023 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("in the year 2023 it was dated"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("Chicxulub is 66 Ma old"), _OBS)["score"] == 0.0
    # the correct age with the Ma unit still matches
    assert t.validate_keystone(_r("dated to 2,023 Ma"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = (
        "Chicxulub crater -> dinosaur k-pg 66 Ma; Vredefort impact structure -> largest verified 300 km "
        "South Africa. I did not check Sudbury or Chesapeake."
    )
    r = _r(text)
    assert abs(t.validate_candidate_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0


def test_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_candidate_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_candidate_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_candidate_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone(r, ungrounded_obs)["passed"] is False
    assert t.validate_survivor(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone(r, ungrounded_obs)["score"],
        t.validate_candidate_coverage(r, ungrounded_obs)["score"],
        t.validate_survivor(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Vredefort age: 2.023 Ga", "survivor: Vredefort"]}
    assert t.validate_keystone(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["election"]
    assert struct["waves"][2] == ["keystone_age"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_chicxulub", "cand_vredefort", "cand_sudbury", "cand_chesapeake"):
        assert "{" + key + "}" in by_id["election"]["instruction"]
    assert "{election}" in by_id["keystone_age"]["instruction"]
    assert "age" in by_id["keystone_age"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("2.023", "2,023", "2023"):
        assert leak not in blob, f"plan leaks {leak!r}"
