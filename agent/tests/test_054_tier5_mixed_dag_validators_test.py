"""
Offline unit tests for the tier-5 mixed DAG (parallel gather + dependent hop) task
(test 054) — free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (Toni Morrison's master's university, Cornell —
the dependent hop's fact) requires the agent to have actually visited at least one page
(visit.count > 0); a correct-but-ungrounded (parametric-memory) answer must collapse the
overall (harness-average) score to <0.75. The UN-gated ``validate_breadth_authors``
diagnostic is deliberately preserved (it measures the independent parallel-wave fan-out
regardless of the dependent keystone/grounding). Also covers the gated citation secondary,
that a grounded-correct answer scores exactly as before, and the compiled plan.
"""
from agent.app.idea_tests import test_054_tier5_mixed_dag as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}

_FULL = (
    "Author of 'Beloved': Toni Morrison. Author of 'The Old Man and the Sea': Ernest Hemingway. "
    "Toni Morrison earned her master's (MA) from Cornell University. "
    "Source: https://en.wikipedia.org/wiki/Toni_Morrison"
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_university(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_authors(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_keystone_but_keeps_breadth():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_university(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_university(r, ungrounded_obs)["passed"] is False
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # UN-gated breadth diagnostic: both parallel authors still credited.
    assert t.validate_breadth_authors(r, ungrounded_obs)["score"] == 1.0
    all_scores = [f(r, ungrounded_obs)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_wrong_university_gates_to_zero_but_keeps_breadth():
    r = _r(_FULL.replace("Cornell University", "Harvard University"))
    assert t.validate_keystone_university(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_breadth_authors(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_compiled_plan_validates_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("morrison", "hemingway", "cornell"):
        assert leak not in blob, f"plan leaks {leak!r}"
