"""
Offline unit tests for the faithfulness & contradiction task (test 042) — free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (both planted-false claims refuted with the
correct obscure year) requires the agent to have actually visited at least one page
(visit.count > 0); a correct-but-ungrounded (parametric-memory) refutation must collapse to
<0.75 overall. Also covers the gated true-control and citation secondaries, and that a
grounded-correct answer scores exactly as before.
"""
from agent.app.idea_tests import test_042_contradiction_verify as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL = (
    "C1: FALSE. 4 Vesta was actually discovered in 1807 by Olbers, not 1850. "
    "See https://en.wikipedia.org/wiki/4_Vesta\n"
    "C2: TRUE. The Terracotta Army was discovered in 1974, confirmed.\n"
    "C3: FALSE. The Antikythera mechanism was recovered in 1901, not 1955. "
    "See https://en.wikipedia.org/wiki/Antikythera_mechanism\n"
    "C4: TRUE. The Lascaux cave paintings were discovered in 1940, confirmed."
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_refutations(r, _OBS)["score"] == 1.0
    assert t.validate_true_claims(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] > 0.0
    assert t.validate_visits(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_below_075():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_refutations(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_refutations(r, ungrounded_obs)["passed"] is False
    assert t.validate_true_claims(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_refutations(r, ungrounded_obs)["score"],
        t.validate_true_claims(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_year_refutation_gates_to_zero():
    r = _r("C1: FALSE, actually discovered in 1801. C3: FALSE, actually recovered in 1900.")
    assert t.validate_keystone_refutations(r, _OBS)["score"] == 0.0
    assert t.validate_true_claims(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
