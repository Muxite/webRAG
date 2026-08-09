"""
Offline unit tests for the multi-hop dependent chain task (test 040) — free, no LLM.

Covers the GROUNDING-GATE fix: keystone credit for the final district requires the agent to
have actually visited at least one page (visit.count > 0); a correct-but-ungrounded
(parametric-memory) answer must collapse to <0.75 overall. Also covers the gated
intermediate-hop and source-URL secondaries, and that a grounded-correct answer scores
exactly as before.
"""
from agent.app.idea_tests import test_040_multihop_chain as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}

_FULL = (
    "Chain: Nineteen Eighty-Four -> George Orwell -> Motihari -> East Champaran. "
    "Sources: https://en.wikipedia.org/wiki/Nineteen_Eighty-Four "
    "https://en.wikipedia.org/wiki/George_Orwell "
    "https://en.wikipedia.org/wiki/Motihari"
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_district(r, _OBS)["score"] == 1.0
    assert t.validate_chain_intermediate(r, _OBS)["score"] == 1.0
    assert t.validate_chain_urls(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_below_075():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_district(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_district(r, ungrounded_obs)["passed"] is False
    assert t.validate_chain_intermediate(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_chain_urls(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_district(r, ungrounded_obs)["score"],
        t.validate_chain_intermediate(r, ungrounded_obs)["score"],
        t.validate_chain_urls(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_district_gates_to_zero():
    r = _r("Chain: Nineteen Eighty-Four -> George Orwell -> Motihari -> West Champaran.")
    assert t.validate_keystone_district(r, _OBS)["score"] == 0.0
    assert t.validate_chain_intermediate(r, _OBS)["score"] == 0.0
    assert t.validate_chain_urls(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
