"""
Offline unit tests for the branch-to-eliminate-then-chain task (test 102) — free, no LLM.

Covers the leak-resistant keystone gate (Louis Armstrong airport longest runway 10,104 ft / 3,080 m),
the UN-gated branch-exploration diagnostic (how many of the four musician-airports were resolved to
their namesake genre, retained even when the terminus is wrong), the keystone-gated survivor/runway
and citation secondaries, single- and multi-line layout, and the adversarial failure modes (electing
famous Lennon, embedded/near-miss numbers). Plus the compiled plan is a genuine branch-then-chain DAG.
"""
import re

from agent.app.idea_tests import test_102_tier5_musician_airport_runway as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}


_FULL_SINGLE = (
    "Stage 1: of the four musician-named airports, Liverpool John Lennon Airport is named after a "
    "rock musician (The Beatles) (https://en.wikipedia.org/wiki/Liverpool_John_Lennon_Airport); "
    "Rio de Janeiro/Galeão Antonio Carlos Jobim International is named after a bossa nova composer "
    "(https://en.wikipedia.org/wiki/Rio_de_Janeiro/Galeão_International_Airport); Salzburg Airport "
    "W. A. Mozart is named after a classical composer "
    "(https://en.wikipedia.org/wiki/Salzburg_Airport); and Louis Armstrong New Orleans International "
    "is named after a jazz trumpeter "
    "(https://en.wikipedia.org/wiki/Louis_Armstrong_New_Orleans_International_Airport) — the survivor. "
    "Stage 2: its longest runway, 11/29, is 10,104 ft (3,080 m)."
)

_FULL_MULTI = (
    "STAGE 1 — musician airports and namesake genres:\n"
    "  Liverpool John Lennon Airport -> rock\n"
    "    https://en.wikipedia.org/wiki/Liverpool_John_Lennon_Airport\n"
    "  Rio/Galeão Antonio Carlos Jobim International -> bossa nova\n"
    "    https://en.wikipedia.org/wiki/Rio_de_Janeiro/Galeão_International_Airport\n"
    "  Salzburg Airport W. A. Mozart -> classical\n"
    "    https://en.wikipedia.org/wiki/Salzburg_Airport\n"
    "  Louis Armstrong New Orleans International -> jazz trumpeter  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Louis_Armstrong_New_Orleans_International_Airport\n"
    "STAGE 2 — longest runway (11/29):\n"
    "  10,104\n"
    "  ft\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_runway(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_runway(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_metric_alternative_satisfies_keystone():
    r = _r("The longest runway at Louis Armstrong New Orleans is 3,080 m.")
    assert t.validate_keystone_runway(r, _OBS)["score"] == 1.0


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Airports: Liverpool John Lennon (rock); Rio/Galeão Antonio Carlos Jobim (bossa nova); "
        "Salzburg W. A. Mozart (classical); Louis Armstrong New Orleans (jazz trumpeter). I take the "
        "famous Liverpool John Lennon as the survivor; its longest runway is 7,500 ft."
    )
    r = _r(wrong)
    assert t.validate_keystone_runway(r, _OBS)["score"] == 0.0       # no 10,104 / 3,080 m
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0    # all four genres resolved
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0           # gated on keystone


def test_keystone_token_rejects_embedded_numbers():
    assert t.validate_keystone_runway(_r("id 110,104 series"), _OBS)["score"] == 0.0   # 110,104 not 10,104
    assert t.validate_keystone_runway(_r("elevation 3,080 ft"), _OBS)["score"] == 0.0  # 3,080 needs metre unit


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Liverpool John Lennon Airport (rock); Louis Armstrong New Orleans (jazz trumpeter). I did "
        "not check the other two airports."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_runway(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_branch_exploration(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["longest runway 11/29: 10,104 ft", "survivor: Louis Armstrong"]}
    assert t.validate_keystone_runway(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_airport"]
    assert struct["waves"][2] == ["longest_runway"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("airport_lennon", "airport_jobim", "airport_mozart", "airport_armstrong"):
        assert "{" + key + "}" in by_id["survivor_airport"]["instruction"]
    assert "{survivor_airport}" in by_id["longest_runway"]["instruction"]
    assert "runway" in by_id["longest_runway"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("10,104", "10104", "3,080", "11/29", "armstrong is the survivor",
                 "armstrong wins"):
        assert leak not in blob, f"plan leaks {leak!r}"
