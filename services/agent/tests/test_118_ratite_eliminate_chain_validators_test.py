"""Offline unit tests for test 118 (ratites -> southern cassowary claw length). Free, no LLM.

Covers the leak-resistant keystone gate (claw 12 cm / 4.7 in), the UN-gated candidate-coverage
diagnostic, the keystone-gated survivor/citation secondaries, single- and multi-line layout, the
famous-decoy failure mode (picking the ostrich), keystone token rejecting near-miss numbers (casque /
egg confusions), visit gating, and the compiled plan (branch-then-chain, templated, self-describing,
leaks nothing).
"""
from agent.app.idea_tests import test_118_tier5_ratite_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: the common ostrich is the largest living bird of the African savanna "
    "(https://en.wikipedia.org/wiki/Common_ostrich); the emu is the second-largest bird of open "
    "Australia (https://en.wikipedia.org/wiki/Emu); the southern cassowary is the rainforest ratite of "
    "New Guinea with a dagger claw, a dangerous bird "
    "(https://en.wikipedia.org/wiki/Southern_cassowary); the greater rhea lives in the South American "
    "grasslands of Argentina (https://en.wikipedia.org/wiki/Greater_rhea). Stage 2: survivor is the "
    "southern cassowary. Stage 3: its inner-toe claw is up to 12 cm (4.7 in) long."
)

_FULL_MULTI = (
    "STAGE 1:\n"
    "  Common ostrich -> largest living bird, africa savanna\n"
    "    https://en.wikipedia.org/wiki/Common_ostrich\n"
    "  Emu -> second-largest, australia, dromaius\n"
    "    https://en.wikipedia.org/wiki/Emu\n"
    "  Southern cassowary -> rainforest, new guinea, casque, dagger, dangerous\n"
    "    https://en.wikipedia.org/wiki/Southern_cassowary\n"
    "  Greater rhea -> south america grassland argentina\n"
    "    https://en.wikipedia.org/wiki/Greater_rhea\n"
    "STAGE 2 survivor: southern cassowary\n"
    "STAGE 3 claw:\n  4.7 in\n  (12 cm)\n"
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
        "Common ostrich -> largest living bird africa savanna; emu -> second-largest australia dromaius; "
        "southern cassowary -> rainforest new guinea casque dagger dangerous; greater rhea -> south "
        "america grassland argentina. I pick the famous largest bird, the ostrich; it can be 2.8 m tall."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_near_miss_numbers():
    assert t.validate_keystone(_r("the casque is 13 to 20 cm high"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("the egg is 11.8-15.8 cm long"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("112 cm feathers"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("the claw is up to 12 cm long"), _OBS)["score"] == 1.0
    assert t.validate_keystone(_r("the claw is 4.7 in long"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = (
        "Common ostrich -> largest living bird africa; southern cassowary -> rainforest new guinea "
        "dagger. I did not check the emu or the rhea."
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


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Inner-toe claw: up to 12 cm", "survivor: cassowary"]}
    assert t.validate_keystone(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["election"]
    assert struct["waves"][2] == ["keystone_claw"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_ostrich", "cand_emu", "cand_cassowary", "cand_rhea"):
        assert "{" + key + "}" in by_id["election"]["instruction"]
    assert "{election}" in by_id["keystone_claw"]["instruction"]
    assert "claw" in by_id["keystone_claw"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("12 cm", "4.7 in", "4.7in"):
        assert leak not in blob, f"plan leaks {leak!r}"
