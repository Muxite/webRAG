"""Offline unit tests for test 113 (largest castles -> Malbork -> enclosed land area)."""
from agent.app.idea_tests import test_113_tier5_largest_castle_by_area_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: Windsor Castle is the largest inhabited castle "
    "(https://en.wikipedia.org/wiki/Windsor_Castle); Prague Castle is the largest ancient castle "
    "complex (https://en.wikipedia.org/wiki/Prague_Castle); Mehrangarh Fort is a hilltop fort in "
    "Jodhpur, Rajasthan, India (https://en.wikipedia.org/wiki/Mehrangarh); Malbork Castle is the "
    "largest castle in the world by land area and the largest brick castle "
    "(https://en.wikipedia.org/wiki/Malbork_Castle) — the survivor. Stage 2: its outer walls "
    "enclose 21 ha (52 acres)."
)

_FULL_MULTI = (
    "STAGE 1 — castles:\n"
    "  Windsor Castle -> largest inhabited castle\n"
    "    https://en.wikipedia.org/wiki/Windsor_Castle\n"
    "  Prague Castle -> largest ancient castle complex\n"
    "    https://en.wikipedia.org/wiki/Prague_Castle\n"
    "  Mehrangarh Fort -> hilltop fort in Jodhpur, Rajasthan\n"
    "    https://en.wikipedia.org/wiki/Mehrangarh\n"
    "  Malbork Castle -> largest by land area, largest brick castle  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Malbork_Castle\n"
    "STAGE 2 — enclosed area:\n"
    "  18.038\n"
    "  ha\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_area(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_area(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_whs_and_acre_alternatives_satisfy_keystone():
    assert t.validate_keystone_area(_r("World Heritage area 18.038 ha (44.57 acres)"), _OBS)["score"] == 1.0
    assert t.validate_keystone_area(_r("enclosed 52 acres"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Windsor Castle -> largest inhabited castle; Prague Castle -> largest ancient complex; "
        "Mehrangarh Fort -> Jodhpur, Rajasthan fort; Malbork Castle -> largest by land area brick "
        "castle. I take the famous Windsor; it covers about 5.3 ha."
    )
    r = _r(wrong)
    assert t.validate_keystone_area(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_area(_r("Windsor about 5.3 ha"), _OBS)["score"] == 0.0
    assert t.validate_keystone_area(_r("built in 1406"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Windsor Castle -> largest inhabited; Malbork Castle -> largest by land area brick "
            "castle. Did not check Prague or Mehrangarh.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_area(r, _OBS)["score"] == 0.0


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
    assert struct["waves"][1] == ["survivor_area"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_windsor", "cand_prague", "cand_malbork", "cand_mehrangarh"):
        assert "{" + key + "}" in by_id["survivor_area"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("18.038", "21 ha", "52 acre", "44.57"):
        assert leak not in blob, f"plan leaks {leak!r}"
