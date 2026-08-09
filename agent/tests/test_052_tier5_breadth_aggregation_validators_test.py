"""
Offline unit tests for the tier-5 6-way fan-out & aggregation (argmin) task (test 052) —
free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (Jane Austen identified as earliest-born, 1775)
requires the agent to have actually visited at least one page (visit.count > 0); a
correct-but-ungrounded (parametric-memory) answer must collapse the OVERALL average of the
credit-bearing checks to <0.75. The UN-gated ``validate_coverage`` breadth diagnostic is
deliberately preserved (it is NOT gated on grounding, by design — it measures whether the
agent actually fanned out and gathered facts, independent of the final argmin/grounding
verdict), so it is excluded from the <0.75 collapse average per the task's own design intent.
Also covers the gated citations secondary and that a grounded-correct answer scores exactly
as before, plus the compiled plan.
"""
from agent.app.idea_tests import test_052_tier5_breadth_aggregation as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}

_FULL = (
    "Earliest-born author: Jane Austen, born 1775.\n"
    "Pride and Prejudice -> Jane Austen -> 1775 (https://en.wikipedia.org/wiki/Jane_Austen)\n"
    "Crime and Punishment -> Fyodor Dostoevsky -> 1821 (https://en.wikipedia.org/wiki/Fyodor_Dostoevsky)\n"
    "Mrs Dalloway -> Virginia Woolf -> 1882 (https://en.wikipedia.org/wiki/Virginia_Woolf)\n"
    "The Great Gatsby -> F. Scott Fitzgerald -> 1896 (https://en.wikipedia.org/wiki/F._Scott_Fitzgerald)\n"
    "The Old Man and the Sea -> Ernest Hemingway -> 1899 (https://en.wikipedia.org/wiki/Ernest_Hemingway)\n"
    "Beloved -> Toni Morrison -> 1931 (https://en.wikipedia.org/wiki/Toni_Morrison)\n"
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_keystone_and_citations_but_keeps_coverage():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_earliest(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_earliest(r, ungrounded_obs)["passed"] is False
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    # UN-gated breadth diagnostic: by design NOT gated on grounding — it measures raw fact
    # coverage in the text regardless of visits, so it stays full here.
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 1.0
    # But the credit-bearing (gated) checks collapse well below 0.75.
    gated_scores = [
        t.validate_keystone_earliest(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(gated_scores) / len(gated_scores) < 0.75
    # And the full harness-average (all 4 registered validators) also collapses <0.75.
    all_scores = [f(r, ungrounded_obs)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_wrong_earliest_gates_to_zero():
    # No Austen-earliest cue -> keystone fails even though all six facts are still reported.
    r = _r(_FULL.replace("Earliest-born author: Jane Austen, born 1775.\n", ""))
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    # coverage remains ungated (breadth still fully gathered)
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 6}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_compiled_plan_validates_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("austen", "1775", "dostoevsky", "1821"):
        assert leak not in blob, f"plan leaks {leak!r}"
