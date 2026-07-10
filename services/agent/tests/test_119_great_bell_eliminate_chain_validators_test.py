"""Offline unit tests for test 119 (great bells -> Mingun Bell weight in viss). Free, no LLM.

Covers the leak-resistant keystone gate (55,555 viss / 90,718 kg), the UN-gated candidate-coverage
diagnostic, the keystone-gated survivor/citation secondaries, single- and multi-line layout, the
famous-decoy failure mode (picking the cracked Tsar Bell), keystone token rejecting near-miss numbers,
visit gating, and the compiled plan (branch-then-chain, templated, self-describing, leaks nothing).
"""
from agent.app.idea_tests import test_119_tier5_great_bell_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: the Tsar Bell in the Moscow Kremlin is the largest ever cast but is cracked and has never "
    "been rung (https://en.wikipedia.org/wiki/Tsar_Bell); the Great Bell of Dhammazedi from the "
    "Shwedagon Pagoda was lost, sunk in a river in 1608 "
    "(https://en.wikipedia.org/wiki/Great_Bell_of_Dhammazedi); the Mingun Bell in Myanmar is intact and "
    "rings (https://en.wikipedia.org/wiki/Mingun_Bell); the Liberty Bell in Philadelphia is cracked and "
    "no longer rung (https://en.wikipedia.org/wiki/Liberty_Bell). Stage 2: survivor is the Mingun Bell. "
    "Stage 3: its weight is 55,555 viss (90,718 kg)."
)

_FULL_MULTI = (
    "STAGE 1:\n"
    "  Tsar Bell -> largest ever cast, cracked, never rung, kremlin moscow\n"
    "    https://en.wikipedia.org/wiki/Tsar_Bell\n"
    "  Great Bell of Dhammazedi -> lost, sank in river, 1608, shwedagon\n"
    "    https://en.wikipedia.org/wiki/Great_Bell_of_Dhammazedi\n"
    "  Mingun Bell -> intact, rings, myanmar\n"
    "    https://en.wikipedia.org/wiki/Mingun_Bell\n"
    "  Liberty Bell -> cracked, philadelphia, 1752\n"
    "    https://en.wikipedia.org/wiki/Liberty_Bell\n"
    "STAGE 2 survivor: Mingun Bell\n"
    "STAGE 3 weight:\n  55,555\n  viss\n"
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
        "Tsar Bell -> largest ever cast, cracked, never rung, kremlin moscow; Great Bell of Dhammazedi "
        "-> lost, sank river 1608 shwedagon; Mingun Bell -> intact rings myanmar; Liberty Bell -> cracked "
        "philadelphia 1752. I pick the famous largest bell, the Tsar Bell; it weighs about 202 tons."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_near_miss_numbers():
    assert t.validate_keystone(_r("code 155,555 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("the Tsar Bell weighs 202 tons"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("weight 55,555 viss"), _OBS)["score"] == 1.0
    assert t.validate_keystone(_r("weight 90,718 kg"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = (
        "Tsar Bell -> largest ever cast, cracked, never rung kremlin; Mingun Bell -> intact rings myanmar. "
        "I did not investigate the remaining two bells."
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
         "deliverables": ["Weight: 55,555 viss", "survivor: Mingun Bell"]}
    assert t.validate_keystone(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["election"]
    assert struct["waves"][2] == ["keystone_weight"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_tsar", "cand_dhammazedi", "cand_mingun", "cand_liberty"):
        assert "{" + key + "}" in by_id["election"]["instruction"]
    assert "{election}" in by_id["keystone_weight"]["instruction"]
    assert "viss" in by_id["keystone_weight"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("55,555", "55555", "90,718", "90718", "199,999"):
        assert leak not in blob, f"plan leaks {leak!r}"
