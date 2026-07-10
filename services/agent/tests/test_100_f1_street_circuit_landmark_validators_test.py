"""
Offline unit tests for the branch-to-eliminate(argmax)-then-chain task (test 100) — free, no LLM.

Covers the leak-resistant keystone gate (Maiden Tower height 29.5 m / 97 ft), the UN-gated branch-
exploration diagnostic (how many of the four circuits were resolved to their lap length, retained
even when the terminus is wrong), the keystone-gated survivor/tower and citation secondaries, single-
and multi-line layout, and the adversarial failure modes (electing famous Monaco, embedded/near-miss
numbers). Plus the compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1).
"""
import re

from agent.app.idea_tests import test_100_tier5_f1_street_circuit_landmark as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: of the four F1 street circuits, Monaco is 3.337 km "
    "(https://en.wikipedia.org/wiki/Circuit_de_Monaco) — the shortest; the Marina Bay Street Circuit "
    "(Singapore) is 5.073 km (https://en.wikipedia.org/wiki/Marina_Bay_Street_Circuit); the Valencia "
    "Street Circuit is 5.419 km (https://en.wikipedia.org/wiki/Valencia_Street_Circuit); and the Baku "
    "City Circuit is 6.003 km (https://en.wikipedia.org/wiki/Baku_City_Circuit) — the longest, the "
    "survivor. Stage 2: Baku threads around the Old City past the Maiden Tower "
    "(https://en.wikipedia.org/wiki/Maiden_Tower_(Baku)). Stage 3: the Maiden Tower is 29.5 m (97 ft) tall."
)

_FULL_MULTI = (
    "STAGE 1 — street circuits and lap lengths:\n"
    "  Circuit de Monaco -> 3.337 km\n"
    "    https://en.wikipedia.org/wiki/Circuit_de_Monaco\n"
    "  Marina Bay Street Circuit (Singapore) -> 5.073 km\n"
    "    https://en.wikipedia.org/wiki/Marina_Bay_Street_Circuit\n"
    "  Valencia Street Circuit -> 5.419 km\n"
    "    https://en.wikipedia.org/wiki/Valencia_Street_Circuit\n"
    "  Baku City Circuit -> 6.003 km  [SURVIVOR / longest]\n"
    "    https://en.wikipedia.org/wiki/Baku_City_Circuit\n"
    "STAGE 2 — landmark:\n"
    "  Maiden Tower\n"
    "    https://en.wikipedia.org/wiki/Maiden_Tower_(Baku)\n"
    "STAGE 3 — height:\n"
    "  29.5\n"
    "  m\n"
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
    r = _r("The Maiden Tower stands 97 ft tall.")
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Street circuits: Circuit de Monaco 3.337 km; Marina Bay Street Circuit 5.073 km; Valencia "
        "Street Circuit 5.419 km; Baku City Circuit 6.003 km. I take the famous Monaco as the "
        "survivor; it passes the Monte Carlo Casino, elevation 20 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0      # no 29.5 / 97 ft
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0   # all four lap lengths resolved
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0          # gated on keystone


def test_keystone_token_rejects_embedded_and_bare_numbers():
    assert t.validate_keystone_height(_r("lap 129.5 something"), _OBS)["score"] == 0.0   # 129.5 not 29.5
    assert t.validate_keystone_height(_r("turn 97 of the season"), _OBS)["score"] == 0.0  # bare 97, no ft


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Circuit de Monaco 3.337 km; Baku City Circuit 6.003 km. I did not check Marina Bay or "
        "Valencia."
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
         "deliverables": ["Maiden Tower height: 29.5 m", "survivor: Baku"]}
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_landmark"]
    assert struct["waves"][2] == ["tower_height"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("circuit_monaco", "circuit_singapore", "circuit_valencia", "circuit_baku"):
        assert "{" + key + "}" in by_id["survivor_landmark"]["instruction"]
    assert "{survivor_landmark}" in by_id["tower_height"]["instruction"]
    assert "height" in by_id["tower_height"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("maiden tower", "29.5", "97 ft", "6.003", "5.419", "5.073", "3.337",
                 "baku is the longest", "baku wins"):
        assert leak not in blob, f"plan leaks {leak!r}"
