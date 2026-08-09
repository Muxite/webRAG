"""
Offline unit tests for the branch-to-eliminate-then-chain task (test 103) — free, no LLM.

Covers the leak-resistant keystone gate (Montreal Tower height 165 m / 541 ft), the UN-gated branch-
exploration diagnostic (how many of the four Olympic stadiums were resolved, retained even when the
terminus is wrong), the keystone-gated survivor/tower and citation secondaries, single- and multi-line
layout, and the adversarial failure modes (electing famous Berlin, embedded/bare numbers). Plus the
compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1).
"""
import re

from agent.app.idea_tests import test_103_tier5_olympic_stadium_tower as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: of the four Olympic stadiums, the Olympiastadion (Berlin) hosted the 1936 Games and has "
    "no inclined tower (https://en.wikipedia.org/wiki/Olympiastadion_(Berlin)); the Olympiastadion "
    "(Munich) hosted the 1972 Games (https://en.wikipedia.org/wiki/Olympiastadion_(Munich)); the "
    "Athens Olympic Stadium hosted the 2004 Games (https://en.wikipedia.org/wiki/Athens_Olympic_Stadium); "
    "and the Stade olympique de Montreal (1976 Games) is dominated by the Montreal Tower, the world's "
    "tallest inclined tower (https://en.wikipedia.org/wiki/Olympic_Stadium_(Montreal)) — the survivor. "
    "Stage 2: the Montreal Tower (https://en.wikipedia.org/wiki/Montreal_Tower) is 165 m (541 ft) tall, "
    "leaning at 45 degrees."
)

_FULL_MULTI = (
    "STAGE 1 — Olympic stadiums and their tower situation:\n"
    "  Olympiastadion (Berlin) -> 1936 Games, no inclined tower\n"
    "    https://en.wikipedia.org/wiki/Olympiastadion_(Berlin)\n"
    "  Olympiastadion (Munich) -> 1972 Games\n"
    "    https://en.wikipedia.org/wiki/Olympiastadion_(Munich)\n"
    "  Athens Olympic Stadium -> 2004 Games\n"
    "    https://en.wikipedia.org/wiki/Athens_Olympic_Stadium\n"
    "  Stade olympique de Montreal -> 1976 Games, world's tallest inclined tower  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Olympic_Stadium_(Montreal)\n"
    "STAGE 2 — Montreal Tower height:\n"
    "  165\n"
    "  m\n"
    "    https://en.wikipedia.org/wiki/Montreal_Tower\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_feet_alternative_satisfies_keystone():
    r = _r("The Montreal Tower is 541 ft tall.")
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Stadiums: Olympiastadion (Berlin) 1936; Olympiastadion (Munich) 1972; Athens Olympic Stadium "
        "2004; Stade olympique de Montreal 1976. I take the famous Berlin 1936 stadium as the "
        "survivor; its bell tower is 77 m tall."
    )
    r = _r(wrong)
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0       # no 165 m / 541 ft
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0    # all four resolved
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0           # gated on keystone


def test_keystone_token_rejects_bare_and_embedded_numbers():
    assert t.validate_keystone_height(_r("crowd 165 thousand"), _OBS)["score"] == 0.0   # 165 without metre unit
    assert t.validate_keystone_height(_r("row 1165 m marker"), _OBS)["score"] == 0.0    # 1165 not 165


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Olympiastadion (Berlin) 1936; Stade olympique de Montreal 1976 with the inclined Montreal "
        "Tower. I did not check the other two stadiums."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_branch_exploration(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Montreal Tower height: 165 m", "survivor: Montreal"]}
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_tower"]
    assert struct["waves"][2] == ["tower_height"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("stadium_berlin", "stadium_munich", "stadium_athens", "stadium_montreal"):
        assert "{" + key + "}" in by_id["survivor_tower"]["instruction"]
    assert "{survivor_tower}" in by_id["tower_height"]["instruction"]
    assert "height" in by_id["tower_height"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("montreal tower", "tour de montr", "165", "541", "45 degree",
                 "montreal wins", "montreal is the survivor"):
        assert leak not in blob, f"plan leaks {leak!r}"
