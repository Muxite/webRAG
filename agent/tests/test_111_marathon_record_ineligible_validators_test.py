"""Offline unit tests for test 111 (World Marathon Majors -> Boston -> net elevation drop)."""
from agent.app.idea_tests import test_111_tier5_marathon_record_ineligible_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: the Berlin Marathon is a flat, fast, record-eligible course "
    "(https://en.wikipedia.org/wiki/Berlin_Marathon); the London Marathon (Greenwich to The Mall) is "
    "record-eligible (https://en.wikipedia.org/wiki/London_Marathon); the Chicago Marathon is a flat "
    "and fast loop (https://en.wikipedia.org/wiki/Chicago_Marathon); the Boston Marathon "
    "(Hopkinton) is NOT eligible, being net-downhill and point-to-point "
    "(https://en.wikipedia.org/wiki/Boston_Marathon) — the survivor. Stage 2: the course drops 459 "
    "feet (140 m) from start to finish."
)

_FULL_MULTI = (
    "STAGE 1 — Majors:\n"
    "  Berlin Marathon -> flat, fast, record course\n"
    "    https://en.wikipedia.org/wiki/Berlin_Marathon\n"
    "  London Marathon -> record-eligible (Greenwich / The Mall)\n"
    "    https://en.wikipedia.org/wiki/London_Marathon\n"
    "  Chicago Marathon -> flat and fast\n"
    "    https://en.wikipedia.org/wiki/Chicago_Marathon\n"
    "  Boston Marathon -> ineligible (net-downhill, point-to-point, Hopkinton)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Boston_Marathon\n"
    "STAGE 2 — net drop:\n"
    "  459\n"
    "  feet\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_drop(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_drop(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_metres_alternative_satisfies_keystone():
    assert t.validate_keystone_drop(_r("net drop of 140 m from Hopkinton"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Berlin Marathon -> flat, fast world-record course; London Marathon -> record-eligible via "
        "The Mall; Chicago Marathon -> flat and fast; Boston Marathon -> ineligible net-downhill "
        "point-to-point. I take the famous Berlin course; it is dead flat."
    )
    r = _r(wrong)
    assert t.validate_keystone_drop(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_drop(_r("elevation gain of 300 feet"), _OBS)["score"] == 0.0
    assert t.validate_keystone_drop(_r("42.195 km course"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Berlin Marathon -> world-record course; Boston Marathon -> ineligible net-downhill "
            "point-to-point. The remaining two Majors were not checked.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_drop(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visits_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False


def test_compiled_plan_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 4
    assert struct["wave_widths"] == [4, 1]
    assert struct["waves"][1] == ["survivor_drop"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_berlin", "cand_boston", "cand_london", "cand_chicago"):
        assert "{" + key + "}" in by_id["survivor_drop"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("459", "140 m", "140m"):
        assert leak not in blob, f"plan leaks {leak!r}"
