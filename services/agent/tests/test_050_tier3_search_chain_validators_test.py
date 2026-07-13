"""
Offline unit tests for the tier-3 URL-free 2-hop search chain task (test 050) — free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (Toni Morrison's master's university, Cornell)
requires the agent to have actually visited at least one page (visit.count > 0); a
correct-but-ungrounded (parametric-memory) answer must collapse to <0.75 overall. Also covers
the gated author/citation secondaries, and that a grounded-correct answer scores exactly as
before.
"""
from agent.app.idea_tests import test_050_tier3_search_chain as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}

_FULL = (
    "The author of 'Beloved' is Toni Morrison. She earned her master's (MA) from Cornell "
    "University. Source: https://en.wikipedia.org/wiki/Toni_Morrison"
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_university(r, _OBS)["score"] == 1.0
    assert t.validate_author(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_below_075():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_university(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_university(r, ungrounded_obs)["passed"] is False
    assert t.validate_author(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_university(r, ungrounded_obs)["score"],
        t.validate_author(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_university_gates_to_zero():
    r = _r("The author of 'Beloved' is Toni Morrison. She earned her master's from Harvard.")
    assert t.validate_keystone_university(r, _OBS)["score"] == 0.0
    assert t.validate_author(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
