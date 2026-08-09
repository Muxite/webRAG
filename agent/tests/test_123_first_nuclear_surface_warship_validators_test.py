"""Offline unit tests for test 123 (first nuclear-powered surface warship -> USS Long Beach -> displacement)."""
from agent.app.idea_tests import test_123_tier5_first_nuclear_surface_warship_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Stage 1: USS Nautilus is the first nuclear submarine (SSN-571) "
    "(https://en.wikipedia.org/wiki/USS_Nautilus_(SSN-571)); NS Savannah is the first nuclear "
    "merchant ship, a civilian cargo-passenger vessel (https://en.wikipedia.org/wiki/NS_Savannah); "
    "USS Enterprise is the first nuclear aircraft carrier (CVN-65) "
    "(https://en.wikipedia.org/wiki/USS_Enterprise_(CVN-65)); USS Long Beach is the world's first "
    "nuclear-powered surface combatant warship, commissioned 1961 "
    "(https://en.wikipedia.org/wiki/USS_Long_Beach_(CGN-9)) — the survivor. Stage 2: its "
    "displacement is 15,540 tons."
)

_FULL_MULTI = (
    "STAGE 1 — vessels:\n"
    "  USS Nautilus -> nuclear submarine (SSN-571)\n"
    "    https://en.wikipedia.org/wiki/USS_Nautilus_(SSN-571)\n"
    "  NS Savannah -> civilian merchant ship\n"
    "    https://en.wikipedia.org/wiki/NS_Savannah\n"
    "  USS Enterprise -> nuclear aircraft carrier (CVN-65)\n"
    "    https://en.wikipedia.org/wiki/USS_Enterprise_(CVN-65)\n"
    "  USS Long Beach -> first nuclear surface combatant warship, 1961  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/USS_Long_Beach_(CGN-9)\n"
    "STAGE 2 — displacement:\n"
    "  15,540\n"
    "  tons\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_displacement(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_displacement(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_displacement_alt_forms_satisfy_keystone():
    assert t.validate_keystone_displacement(_r("displacement 15540 tons"), _OBS)["score"] == 1.0
    assert t.validate_keystone_displacement(_r("15 540 t full load"), _OBS)["score"] == 1.0


def test_ungrounded_correct_value_scores_near_zero():
    """Right keystone value present, but zero visits (no grounding) -> keystone and every
    keystone-gated secondary must collapse to 0, even though the value string matches."""
    r = _r(_FULL_SINGLE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_displacement(r, ungrounded)["score"] == 0.0
    assert t.validate_survivor(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    overall = sum(v["score"] for v in [
        t.validate_keystone_displacement(r, ungrounded),
        t.validate_survivor(r, ungrounded),
        t.validate_citations(r, ungrounded),
    ]) / 3.0
    assert overall < 0.75


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "USS Nautilus -> nuclear submarine; NS Savannah -> civilian merchant ship; USS Enterprise -> "
        "nuclear aircraft carrier; USS Long Beach -> nuclear surface warship 1961. I take the famous "
        "USS Nautilus; it displaced about 4,092 tons submerged."
    )
    r = _r(wrong)
    assert t.validate_keystone_displacement(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_displacement(_r("Nautilus 4,092 tons"), _OBS)["score"] == 0.0
    assert t.validate_keystone_displacement(_r("Enterprise 93,000 tons"), _OBS)["score"] == 0.0
    assert t.validate_keystone_displacement(_r("Savannah 13,599 tons"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("USS Nautilus -> nuclear submarine; USS Long Beach -> surface warship 1961. Did not "
            "check NS Savannah or USS Enterprise.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_displacement(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visits_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is False


def test_compiled_plan_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 4
    assert struct["wave_widths"] == [4, 1]
    assert struct["waves"][1] == ["survivor_displacement"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_nautilus", "cand_savannah", "cand_enterprise", "cand_longbeach"):
        assert "{" + key + "}" in by_id["survivor_displacement"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("15,540", "15540", "15 540"):
        assert leak not in blob, f"plan leaks {leak!r}"
