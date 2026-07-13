"""
Offline unit tests for the tier-2 two-page fact combination task (test 049) — free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (both completion years + the derived 3-year gap)
requires the agent to have actually visited at least one page (visit.count > 0); a
correct-but-ungrounded (parametric-memory) answer must collapse to <0.75 overall. Also covers
the gated ordering/grounding/visited-both secondaries, and that a grounded-correct answer
scores exactly as before.
"""
from agent.app.idea_tests import test_049_tier2_two_page_combine as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}

_FULL = (
    "The Eiffel Tower was completed in 1889 (https://en.wikipedia.org/wiki/Eiffel_Tower). "
    "The Statue of Liberty was dedicated in 1886 (https://en.wikipedia.org/wiki/Statue_of_Liberty). "
    "The Statue of Liberty was first, 3 years earlier than the Eiffel Tower."
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_combination(r, _OBS)["score"] == 1.0
    assert t.validate_ordering(r, _OBS)["score"] == 1.0
    assert t.validate_grounding(r, _OBS)["score"] == 1.0
    assert t.validate_visited_both(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_below_075():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_combination(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_combination(r, ungrounded_obs)["passed"] is False
    assert t.validate_ordering(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_grounding(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_visited_both(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_keystone_combination(r, ungrounded_obs)["score"],
        t.validate_ordering(r, ungrounded_obs)["score"],
        t.validate_grounding(r, ungrounded_obs)["score"],
        t.validate_visited_both(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_gap_gates_to_zero():
    r = _r("The Eiffel Tower was completed in 1889 and the Statue of Liberty in 1886, a gap of 5 years.")
    assert t.validate_keystone_combination(r, _OBS)["score"] == 0.0
    assert t.validate_ordering(r, _OBS)["score"] == 0.0
    assert t.validate_grounding(r, _OBS)["score"] == 0.0


def test_visited_both_requires_two_visits():
    r = _r(_FULL)
    assert t.validate_visited_both(r, {"visit": {"count": 1}})["score"] == 0.5
    assert t.validate_visited_both(r, {"visit": {"count": 0}})["score"] == 0.0
