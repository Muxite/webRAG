"""Offline unit tests for test 122 (largest operational filled-aperture single dish -> FAST -> illuminated aperture)."""
from agent.app.idea_tests import test_122_tier5_filled_aperture_telescope_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Stage 1: Arecibo Telescope collapsed in 2020 and is no longer operational "
    "(https://en.wikipedia.org/wiki/Arecibo_Telescope); RATAN-600 is a ring reflector, not a filled "
    "aperture (https://en.wikipedia.org/wiki/RATAN-600); the Green Bank Telescope is the largest "
    "fully steerable 100 m dish (https://en.wikipedia.org/wiki/Green_Bank_Telescope); FAST is the "
    "500 m filled-aperture dish, operational since 2020 and the survivor "
    "(https://en.wikipedia.org/wiki/Five-hundred-meter_Aperture_Spherical_Telescope). Stage 2: only "
    "a 300 m circle is illuminated at any one time."
)

_FULL_MULTI = (
    "STAGE 1 — telescopes:\n"
    "  Arecibo Telescope -> collapsed 2020, not operational\n"
    "    https://en.wikipedia.org/wiki/Arecibo_Telescope\n"
    "  RATAN-600 -> ring reflector, not a filled aperture\n"
    "    https://en.wikipedia.org/wiki/RATAN-600\n"
    "  Green Bank Telescope -> largest fully steerable 100 m dish\n"
    "    https://en.wikipedia.org/wiki/Green_Bank_Telescope\n"
    "  FAST -> 500 m filled-aperture dish, operational  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Five-hundred-meter_Aperture_Spherical_Telescope\n"
    "STAGE 2 — illuminated aperture:\n"
    "  300\n"
    "  m\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_imperial_alternative_satisfies_keystone():
    assert t.validate_keystone_aperture(_r("illuminated diameter 984 ft 3 in"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Arecibo Telescope -> collapsed 2020; RATAN-600 -> ring reflector; Green Bank Telescope -> "
        "steerable 100 m dish; FAST -> 500 m filled-aperture dish. I take the famous Arecibo; its "
        "dish was 305 m across."
    )
    r = _r(wrong)
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_aperture(_r("Arecibo dish 305 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_aperture(_r("physical dish 500 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_aperture(_r("ring 1300 m long"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Arecibo Telescope -> collapsed 2020; FAST -> 500 m filled-aperture operational dish. "
            "Did not open the other two telescopes.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 0.0


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
    assert struct["waves"][1] == ["survivor_aperture"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_arecibo", "cand_ratan", "cand_greenbank", "cand_fast"):
        assert "{" + key + "}" in by_id["survivor_aperture"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("300 m", "984"):
        assert leak not in blob, f"plan leaks {leak!r}"
