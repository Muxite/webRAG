"""
Offline unit tests for the giant-titanosaur branch-then-chain task (test 104) — free, no LLM.

Covers the leak-resistant keystone gate (the survivor's measured scapula length, 1.74 m), the
UN-gated elimination-coverage diagnostic (how many of the four titanosaurs were investigated,
retained even when the terminus is wrong and capped by visits), the keystone-gated survivor/bone
and citation secondaries, the correct answer in single- and multi-line layout, and the adversarial
failure modes: the fame decoy (electing Argentinosaurus and reporting an estimated figure) and a
near-miss number. Plus the compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1) that
templates upstream, is self-describing, and leaks nothing.
"""
import re

from agent.app.idea_tests import test_104_tier5_titanosaur_completeness_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: Argentinosaurus (https://en.wikipedia.org/wiki/Argentinosaurus) is very fragmentary; "
    "Patagotitan (https://en.wikipedia.org/wiki/Patagotitan) is partial; Puertasaurus "
    "(https://en.wikipedia.org/wiki/Puertasaurus) is four vertebrae; Dreadnoughtus "
    "(https://en.wikipedia.org/wiki/Dreadnoughtus) is ~70.4% complete — the survivor. Stage 2/3: "
    "its scapula (shoulder blade) is 1.74 m long, the longest of any titanosaur."
)

_FULL_MULTI = (
    "STAGE 1 — completeness:\n"
    "  Argentinosaurus -> fragmentary\n"
    "    https://en.wikipedia.org/wiki/Argentinosaurus\n"
    "  Patagotitan -> partial\n"
    "    https://en.wikipedia.org/wiki/Patagotitan\n"
    "  Puertasaurus -> four vertebrae\n"
    "    https://en.wikipedia.org/wiki/Puertasaurus\n"
    "  Dreadnoughtus -> most complete  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Dreadnoughtus\n"
    "STAGE 3 — scapula length:\n"
    "  1.74\n"
    "  metres\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_scapula(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_bone(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_scapula(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_bone(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Argentinosaurus, Patagotitan, Puertasaurus, Dreadnoughtus all investigated. I take the "
        "famous Argentinosaurus as the answer; its estimated femur is about 2.5 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_scapula(r, _OBS)["score"] == 0.0     # no 1.74
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0  # all four named
    assert t.validate_survivor_and_bone(r, _OBS)["score"] == 0.0     # gated
    assert t.validate_citations(r, _OBS)["score"] == 0.0            # gated


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_scapula(_r("code 11.74 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_scapula(_r("length 1.740 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_scapula(_r("scapula 1.31 m (ilium)"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = (
        "I investigated only Argentinosaurus and Dreadnoughtus; I did not check the other two "
        "giant sauropods on the list."
    )
    r = _r(text)
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_scapula(r, _OBS)["score"] == 0.0


def test_elimination_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_elimination_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_elimination_coverage(r, {"visit": {"count": 0}})["passed"] is False
    assert abs(t.validate_elimination_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_elimination_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_scapula(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_scapula(r, ungrounded_obs)["passed"] is False
    assert t.validate_survivor_and_bone(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_scapula(r, ungrounded_obs)["score"],
        t.validate_elimination_coverage(r, ungrounded_obs)["score"],
        t.validate_survivor_and_bone(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Dreadnoughtus scapula: 1.74 m", "survivor: Dreadnoughtus"]}
    assert t.validate_keystone_scapula(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor"]
    assert struct["waves"][2] == ["scapula_length"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("titan_argentinosaurus", "titan_patagotitan", "titan_puertasaurus", "titan_dreadnoughtus"):
        assert "{" + key + "}" in by_id["survivor"]["instruction"]
    assert "{survivor}" in by_id["scapula_length"]["instruction"]
    assert "scapula" in by_id["scapula_length"]["expect"].lower()
    assert "metres" in by_id["scapula_length"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1.74", "1.31", "70.4", "70 %", "45.3", "26 m"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # must not leak WHICH genus survives as the answer beyond naming it as a GIVEN candidate:
    # 'dreadnoughtus' appears only as a candidate id/name, never labelled the most-complete one.
    assert "most complete" not in by_survivor_text(plan)


def by_survivor_text(plan):
    # helper: text that would reveal the answer if the survivor were pre-labelled
    return " ".join(
        l["instruction"].lower() for l in plan["leaves"] if l["id"].startswith("titan_")
    )
