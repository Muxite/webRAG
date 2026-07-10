"""Offline unit tests for test 110 (Mark-1 computers -> Ferranti Mark 1 -> valve count)."""
import re

from agent.app.idea_tests import test_110_tier5_mark1_commercial_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: the Harvard Mark I (IBM ASCC) was an electromechanical relay machine, a one-off "
    "(https://en.wikipedia.org/wiki/Harvard_Mark_I); the Manchester Baby (SSEM) was an experimental "
    "prototype (https://en.wikipedia.org/wiki/Manchester_Baby); the Manchester Mark 1 was a "
    "university research machine with index registers "
    "(https://en.wikipedia.org/wiki/Manchester_Mark_1); the Ferranti Mark 1 was the first "
    "commercially available electronic computer (https://en.wikipedia.org/wiki/Ferranti_Mark_1) — "
    "the survivor. Stage 2: it contained 4,050 vacuum tubes."
)

_FULL_MULTI = (
    "STAGE 1 — candidates:\n"
    "  Harvard Mark I -> electromechanical relay one-off\n"
    "    https://en.wikipedia.org/wiki/Harvard_Mark_I\n"
    "  Manchester Baby (SSEM) -> experimental prototype\n"
    "    https://en.wikipedia.org/wiki/Manchester_Baby\n"
    "  Manchester Mark 1 -> university research machine (index registers)\n"
    "    https://en.wikipedia.org/wiki/Manchester_Mark_1\n"
    "  Ferranti Mark 1 -> first commercially available electronic computer  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Ferranti_Mark_1\n"
    "STAGE 2 — valves:\n"
    "  4,050\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_valves(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_valves(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Harvard Mark I -> electromechanical relay one-off; Manchester Baby (SSEM) -> experimental "
        "prototype; Manchester Mark 1 -> university research machine with index registers; Ferranti "
        "Mark 1 -> commercially available. I take the famous Harvard Mark I as the answer; it had "
        "765,000 electromechanical components."
    )
    r = _r(wrong)
    assert t.validate_keystone_valves(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_numbers():
    assert t.validate_keystone_valves(_r("part 14,050x code"), _OBS)["score"] == 0.0
    assert t.validate_keystone_valves(_r("765,000 components"), _OBS)["score"] == 0.0
    assert t.validate_keystone_valves(_r("4050 valves"), _OBS)["score"] == 1.0


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Harvard Mark I -> electromechanical relay one-off; Ferranti Mark 1 -> commercially "
        "available. I did not check the two Manchester machines."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_valves(r, _OBS)["score"] == 0.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_branch_exploration(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visits_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Ferranti Mark 1 valves: 4,050", "survivor: Ferranti"]}
    assert t.validate_keystone_valves(r, _OBS)["score"] == 1.0


def test_compiled_plan_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 4
    assert struct["wave_widths"] == [4, 1]
    assert struct["waves"][1] == ["survivor_valves"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_harvard", "cand_baby", "cand_manchester", "cand_ferranti"):
        assert "{" + key + "}" in by_id["survivor_valves"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("4,050", "4050"):
        assert leak not in blob, f"plan leaks {leak!r}"
