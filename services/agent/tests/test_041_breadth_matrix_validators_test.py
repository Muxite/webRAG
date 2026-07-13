"""
Offline unit tests for the wide-breadth source matrix task (test 041) — free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (longest-span bridge = Akashi Kaikyo, 1,991 m)
requires the agent to have actually visited at least one page (visit.count > 0); a
correct-but-ungrounded (parametric-memory) answer must collapse to <0.75 overall. Also
covers the gated span-values/table secondaries and that a grounded-correct answer scores
exactly as before.
"""
from agent.app.idea_tests import test_041_breadth_matrix as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}

_FULL = (
    "| Bridge | Main span (m) | Source URL |\n"
    "|---|---|---|\n"
    "| Akashi Kaikyo Bridge | 1,991 | https://en.wikipedia.org/wiki/Akashi_Kaikyo_Bridge |\n"
    "| Great Belt Bridge | 1,624 | https://en.wikipedia.org/wiki/Great_Belt_Bridge |\n"
    "| Humber Bridge | 1,410 | https://en.wikipedia.org/wiki/Humber_Bridge |\n"
    "| Verrazzano-Narrows Bridge | 1,298 | https://en.wikipedia.org/wiki/Verrazzano-Narrows_Bridge |\n"
    "| Golden Gate Bridge | 1,280 | https://en.wikipedia.org/wiki/Golden_Gate_Bridge |\n"
    "| Mackinac Bridge | 1,158 | https://en.wikipedia.org/wiki/Mackinac_Bridge |\n"
    "The longest main span is Akashi Kaikyo Bridge at 1,991 m."
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_longest(r, _OBS)["score"] == 1.0
    assert t.validate_span_values(r, _OBS)["score"] == 1.0
    assert t.validate_table(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_below_075():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_longest(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_longest(r, ungrounded_obs)["passed"] is False
    assert t.validate_span_values(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_table(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_longest(r, ungrounded_obs)["score"],
        t.validate_span_values(r, ungrounded_obs)["score"],
        t.validate_table(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_longest_bridge_gates_to_zero():
    r = _r("The longest main span is the Golden Gate Bridge at 1,280 m.")
    assert t.validate_keystone_longest(r, _OBS)["score"] == 0.0
    assert t.validate_span_values(r, _OBS)["score"] == 0.0
    assert t.validate_table(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 6}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
