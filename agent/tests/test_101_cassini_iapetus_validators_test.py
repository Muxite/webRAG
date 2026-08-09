"""
Offline unit tests for the branch-to-eliminate-then-chain task (test 101) — free, no LLM.

Covers the leak-resistant keystone gate (Iapetus mean radius 734.4 km / diameter ~1,469 km), the
UN-gated branch-exploration diagnostic (how many of the four 'Cassini' features were resolved to
their body, retained even when the terminus is wrong), the keystone-gated survivor/moon and citation
secondaries, single- and multi-line layout, and the adversarial failure modes (electing the famous
lunar crater, embedded/near-miss numbers). Plus the compiled plan is a genuine branch-then-chain DAG.
"""
import re

from agent.app.idea_tests import test_101_tier5_cassini_iapetus as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: of the four 'Cassini' features, the lunar Cassini crater is on the Moon in Mare Imbrium "
    "(https://en.wikipedia.org/wiki/Cassini_(lunar_crater)); the Martian Cassini crater is on Mars in "
    "the Arabia quadrangle (https://en.wikipedia.org/wiki/Cassini_(Martian_crater)); the Cassini "
    "spacecraft/Huygens probe explored Saturn (https://en.wikipedia.org/wiki/Cassini-Huygens); and "
    "Cassini Regio is a dark region on Iapetus, a moon of Saturn "
    "(https://en.wikipedia.org/wiki/Cassini_Regio) — the survivor. Stage 2: opening Iapetus "
    "(https://en.wikipedia.org/wiki/Iapetus_(moon)), its mean radius is 734.4 km (mean diameter ~1,469 km)."
)

_FULL_MULTI = (
    "STAGE 1 — 'Cassini' features and their bodies:\n"
    "  Cassini (lunar crater) -> the Moon (Imbrium)\n"
    "    https://en.wikipedia.org/wiki/Cassini_(lunar_crater)\n"
    "  Cassini (Martian crater) -> Mars (martian)\n"
    "    https://en.wikipedia.org/wiki/Cassini_(Martian_crater)\n"
    "  Cassini spacecraft / Huygens probe -> a probe\n"
    "    https://en.wikipedia.org/wiki/Cassini-Huygens\n"
    "  Cassini Regio -> Iapetus (moon of Saturn)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Cassini_Regio\n"
    "STAGE 2 — moon mean radius:\n"
    "  734.4\n"
    "  km\n"
    "    https://en.wikipedia.org/wiki/Iapetus_(moon)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_radius(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_radius(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_diameter_alternative_satisfies_keystone():
    r = _r("Cassini Regio is on Iapetus, whose mean diameter is 1,469 km.")
    assert t.validate_keystone_radius(r, _OBS)["score"] == 1.0


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "'Cassini' features: lunar Cassini crater on the Moon (Imbrium); Martian Cassini crater on "
        "Mars (martian); Cassini spacecraft/Huygens probe; Cassini Regio on Iapetus. I take the "
        "famous lunar Cassini crater as the survivor; its diameter is 57 km."
    )
    r = _r(wrong)
    assert t.validate_keystone_radius(r, _OBS)["score"] == 0.0       # no 734 / 1,469
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0    # all four bodies resolved
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0           # gated on keystone


def test_keystone_token_rejects_embedded_numbers():
    assert t.validate_keystone_radius(_r("catalog 2734 xj"), _OBS)["score"] == 0.0    # 2734 not 734
    assert t.validate_keystone_radius(_r("id 21,469 series"), _OBS)["score"] == 0.0   # 21,469 not 1,469


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Cassini lunar crater on the Moon (Imbrium); Cassini Regio on Iapetus. I did not check the "
        "other two features."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_radius(r, _OBS)["score"] == 0.0


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
         "deliverables": ["Iapetus mean radius: 734.4 km", "survivor: Cassini Regio"]}
    assert t.validate_keystone_radius(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_moon"]
    assert struct["waves"][2] == ["moon_radius"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("feat_lunar", "feat_martian", "feat_spacecraft", "feat_regio"):
        assert "{" + key + "}" in by_id["survivor_moon"]["instruction"]
    assert "{survivor_moon}" in by_id["moon_radius"]["instruction"]
    assert "radius" in by_id["moon_radius"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("iapetus", "734", "1,469", "1469", "regio wins", "regio is the survivor"):
        assert leak not in blob, f"plan leaks {leak!r}"
