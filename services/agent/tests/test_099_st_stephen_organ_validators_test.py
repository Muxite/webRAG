"""
Offline unit tests for the branch-to-eliminate-then-chain task (test 099) — free, no LLM.

Covers the leak-resistant keystone gate (Passau organ = 17,774 pipes), the UN-gated branch-
exploration diagnostic (how many of the four St. Stephen's churches were resolved, retained even
when the terminus is wrong), the keystone-gated survivor/organ and citation secondaries, single-
and multi-line layout, and the adversarial failure modes (electing famous Vienna, the mis-cited
17,974, embedded/near-miss numbers). Plus the compiled plan is a genuine branch-then-chain DAG
(4 -> 1 -> 1), templates upstream, is self-describing, and leaks no verdict / survivor / pipe count.
"""
import re

from agent.app.idea_tests import test_099_tier5_st_stephen_organ as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}


_FULL_SINGLE = (
    "Stage 1: of the four St. Stephen's churches, Vienna's Stephansdom (Austria) is the famous "
    "Gothic cathedral but not the largest organ (https://en.wikipedia.org/wiki/St._Stephen's_Cathedral,_Vienna); "
    "St. Stephen's Basilica, Budapest (Hungary) is neoclassical and not the largest "
    "(https://en.wikipedia.org/wiki/St._Stephen's_Basilica); St. Stephen's Cathedral, Brisbane "
    "(Australia) is not the largest (https://en.wikipedia.org/wiki/St._Stephen's_Cathedral,_Brisbane); "
    "and Passau Cathedral (Bavaria, Germany) has the LARGEST church organ "
    "(https://en.wikipedia.org/wiki/Passau_Cathedral) — the survivor. Stage 2: that organ has "
    "17,774 pipes across 233 registers."
)

_FULL_MULTI = (
    "STAGE 1 — St. Stephen's churches and their organ status:\n"
    "  Vienna (Austria, Gothic) -> not the largest\n"
    "    https://en.wikipedia.org/wiki/St._Stephen's_Cathedral,_Vienna\n"
    "  Budapest (Hungary, neoclassical) -> not the largest\n"
    "    https://en.wikipedia.org/wiki/St._Stephen's_Basilica\n"
    "  Brisbane (Australia) -> not the largest\n"
    "    https://en.wikipedia.org/wiki/St._Stephen's_Cathedral,_Brisbane\n"
    "  Passau (Bavaria, Germany) -> LARGEST church organ  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Passau_Cathedral\n"
    "STAGE 2 — organ pipes:\n"
    "  17,774\n"
    "  233 registers\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_pipes(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_pipes(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "St. Stephen's churches: Vienna (Austria, Gothic); Budapest (Hungary, neoclassical); "
        "Passau (Bavaria, Germany); Brisbane (Australia). I take the famous Vienna Stephansdom as the "
        "largest-organ survivor; its organ has about 10,000 pipes."
    )
    r = _r(wrong)
    assert t.validate_keystone_pipes(r, _OBS)["score"] == 0.0      # no 17,774
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0  # all four resolved
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0  # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0         # gated on keystone


def test_miscited_number_fails_keystone():
    # The oft-repeated erroneous "17,974" must NOT satisfy the 17,774 keystone.
    r = _r("Passau Cathedral organ has 17,974 pipes.")
    assert t.validate_keystone_pipes(r, _OBS)["passed"] is False


def test_keystone_token_rejects_embedded_numbers():
    assert t.validate_keystone_pipes(_r("code 117,774 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_pipes(_r("about 10,000 pipes"), _OBS)["score"] == 0.0


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Vienna (Austria, Gothic); Passau (Bavaria, Germany) has the largest organ. I did not check "
        "Budapest or Brisbane."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_pipes(r, _OBS)["score"] == 0.0


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
         "deliverables": ["Passau organ: 17,774 pipes", "survivor: Passau"]}
    assert t.validate_keystone_pipes(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_cathedral"]
    assert struct["waves"][2] == ["organ_pipes"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cath_vienna", "cath_budapest", "cath_passau", "cath_brisbane"):
        assert "{" + key + "}" in by_id["survivor_cathedral"]["instruction"]
    assert "{survivor_cathedral}" in by_id["organ_pipes"]["instruction"]
    assert "pipe" in by_id["organ_pipes"]["instruction"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("17,774", "17774", "233", "passau wins", "passau is the largest",
                 "the survivor is passau"):
        assert leak not in blob, f"plan leaks {leak!r}"
