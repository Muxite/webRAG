"""
Offline unit tests for the terminal-arithmetic river-length-difference task (test 085) — free.

Covers the keystone gate (computed difference 966 - 647 = 319) in single- and multi-line layout;
the raw-input trap (reporting 966 or 647 instead of the computed difference) gating every
credit-bearing check to zero while the UN-gated breadth diagnostic is retained; the grounding
requirement (a correct-value answer with zero visits must not earn credit); the visit process
metric; and the compiled plan being a well-formed 2-way independent fan-out that leaks no length
or the difference.
"""
from agent.app.idea_tests import test_085_tier5_terminal_arithmetic_b as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}

_FULL_SINGLE = (
    "Tisza River: 966 km (https://en.wikipedia.org/wiki/Tisza). Siret River: 647 km "
    "(https://en.wikipedia.org/wiki/Siret_(river)). Absolute difference: 966 - 647 = 319 km."
)

_FULL_MULTI = (
    "Tisza River: 966 km\n"
    "  (source: https://en.wikipedia.org/wiki/Tisza)\n"
    "Siret River: 647 km\n"
    "  (source: https://en.wikipedia.org/wiki/Siret_(river))\n"
    "Difference: 966 - 647 = 319 km\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_lengths(r, _OBS)["score"] == 1.0
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_lengths(r, _OBS)["score"] == 1.0
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_raw_input_trap_gates_to_zero_but_keeps_breadth():
    # Reporting a raw river length instead of the computed difference fails the keystone.
    wrong = "Tisza River: 966 km. Siret River: 647 km. The difference is 966 km."
    r = _r(wrong)
    assert t.validate_keystone_difference(r, _OBS)["passed"] is False
    assert t.validate_chains_resolved(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_breadth_lengths(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_difference_gates_to_zero():
    """Grounding requirement: the correct keystone DIFFERENCE alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory
    guess must collapse the keystone gate (and everything gated on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_difference(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_difference(r, ungrounded_obs)["passed"] is False
    assert t.validate_chains_resolved(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # Breadth is CAPPED BY visit count (F29-style fix): a 0-visit run banks 0 breadth credit
    # even though the raw text names both lengths, closing the "parametric recall" leak.
    assert t.validate_breadth_lengths(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_difference(r, ungrounded_obs)["score"],
        t.validate_breadth_lengths(r, ungrounded_obs)["score"],
        t.validate_chains_resolved(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 2}})["score"] == 1.0


def test_compiled_plan_is_well_formed_two_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 2
    assert struct["edge_count"] == 0


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("966", "647", "319"):
        assert leak not in blob, f"plan leaks {leak!r}"
