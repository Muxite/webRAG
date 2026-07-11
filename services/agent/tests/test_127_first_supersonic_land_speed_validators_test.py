"""Offline unit tests for test 127 (first land vehicle to break the sound barrier -> ThrustSSC -> record speed)."""
from agent.app.idea_tests import test_127_tier5_first_supersonic_land_speed_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Stage 1: the Bugatti Chiron holds a production-car speed record (~304 mph), not the outright "
    "land speed record (https://en.wikipedia.org/wiki/Bugatti_Chiron); Thrust2 held the former "
    "outright record but was subsonic at ~633 mph (https://en.wikipedia.org/wiki/Thrust2); the "
    "Bloodhound LSR has only run test passes and set no official record "
    "(https://en.wikipedia.org/wiki/Bloodhound_LSR); ThrustSSC broke the sound barrier and set the "
    "outright land speed record in 1997 (https://en.wikipedia.org/wiki/ThrustSSC) — the survivor. "
    "Stage 2: its record speed is 763.035 mph."
)

_FULL_MULTI = (
    "STAGE 1 — vehicles:\n"
    "  Bugatti Chiron -> production-car record ~304 mph, not land speed record\n"
    "    https://en.wikipedia.org/wiki/Bugatti_Chiron\n"
    "  Thrust2 -> former outright record, subsonic ~633 mph\n"
    "    https://en.wikipedia.org/wiki/Thrust2\n"
    "  Bloodhound LSR -> test runs only, no official record\n"
    "    https://en.wikipedia.org/wiki/Bloodhound_LSR\n"
    "  ThrustSSC -> broke the sound barrier, outright land speed record 1997  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/ThrustSSC\n"
    "STAGE 2 — record speed:\n"
    "  763.035\n"
    "  mph\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_speed(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_speed(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_speed_alt_forms_satisfy_keystone():
    assert t.validate_keystone_speed(_r("record 763 mph"), _OBS)["score"] == 1.0
    assert t.validate_keystone_speed(_r("1,227.985 km/h flying mile"), _OBS)["score"] == 1.0
    assert t.validate_keystone_speed(_r("about 1,228 km/h"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Bugatti Chiron -> production-car record ~304 mph not land speed record; Thrust2 -> former "
        "outright subsonic ~633 mph; Bloodhound LSR -> test runs only no official record; ThrustSSC "
        "-> broke the sound barrier land speed record 1997. I take the famous Bugatti Chiron; it hit "
        "about 304 mph."
    )
    r = _r(wrong)
    assert t.validate_keystone_speed(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_speed(_r("Thrust2 633 mph"), _OBS)["score"] == 0.0
    assert t.validate_keystone_speed(_r("Bugatti 304 mph"), _OBS)["score"] == 0.0
    assert t.validate_keystone_speed(_r("set on 15 October 1997"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Bugatti Chiron -> production-car record ~304 mph; ThrustSSC -> broke the sound barrier "
            "land speed record 1997. Did not check Thrust2 or Bloodhound.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_speed(r, _OBS)["score"] == 0.0


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
    assert struct["waves"][1] == ["survivor_speed"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_bugatti", "cand_thrust2", "cand_bloodhound", "cand_thrustssc"):
        assert "{" + key + "}" in by_id["survivor_speed"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("763", "1,227.985", "1227.985", "1,228"):
        assert leak not in blob, f"plan leaks {leak!r}"
