"""
Offline unit tests for the k-th-largest (3rd deepest fjord) task (test 075) — free.

Covers the keystone gate (Jervis Inlet = 3rd deepest of six fjords) in single- and multi-line
layout; the max/min decoy traps (Sognefjorden deepest, Milford Sound shallowest) gating every
credit-bearing check to zero while the UN-gated coverage diagnostic is retained; the grounding
requirement (a correct-value answer with zero visits must not earn credit); partial coverage
scoring an exact fraction; the visit process metric; and the compiled plan being a well-formed
6-way independent fan-out that leaks no depth figure or rank.
"""
from agent.app.idea_tests import test_075_tier5_kth_largest as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}

_FULL_SINGLE = (
    "Sognefjorden 1,308 m; Hardangerfjord 860 m; Jervis Inlet 670 m; Romsdalsfjord 550 m; "
    "Lysefjord 422 m; Milford Sound 291 m. Sorted deepest to shallowest, Jervis Inlet is the "
    "3rd deepest at 670 m. Sources: https://en.wikipedia.org/wiki/Sognefjorden "
    "https://en.wikipedia.org/wiki/Hardangerfjord https://en.wikipedia.org/wiki/Jervis_Inlet "
    "https://en.wikipedia.org/wiki/Romsdalsfjord https://en.wikipedia.org/wiki/Lysefjord "
    "https://en.wikipedia.org/wiki/Milford_Sound"
)

_FULL_MULTI = (
    "Ranked (deepest to shallowest):\n"
    "  1. Sognefjorden - 1308 m - https://en.wikipedia.org/wiki/Sognefjorden\n"
    "  2. Hardangerfjord - 860 m - https://en.wikipedia.org/wiki/Hardangerfjord\n"
    "  3. Jervis Inlet - 670 m - https://en.wikipedia.org/wiki/Jervis_Inlet\n"
    "  4. Romsdalsfjord - 550 m - https://en.wikipedia.org/wiki/Romsdalsfjord\n"
    "  5. Lysefjord - 422 m - https://en.wikipedia.org/wiki/Lysefjord\n"
    "  6. Milford Sound - 291 m - https://en.wikipedia.org/wiki/Milford_Sound\n"
    "The 3rd deepest is:\n"
    "  Jervis Inlet\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_kth(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_depth(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_kth(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_depth(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_max_decoy_gates_to_zero_but_keeps_coverage():
    wrong = _FULL_MULTI.replace(
        "The 3rd deepest is:\n  Jervis Inlet\n",
        "The deepest (and my pick for 3rd) is:\n  Sognefjorden\n",
    )
    r = _r(wrong)
    assert t.validate_keystone_kth(r, _OBS)["score"] == 0.0
    assert t.validate_winner_depth(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_answer_gates_to_zero():
    """Grounding requirement: the correct keystone WINNER alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess must
    collapse the keystone gate (and everything gated on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_kth(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_kth(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_depth(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_kth(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_winner_depth(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_scores_fraction():
    partial = "Sognefjorden 1308 m. Jervis Inlet 670 m."
    r = _r(partial)
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 2 / 6) < 1e-9


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 6}})["score"] == 1.0


def test_compiled_plan_is_well_formed_six_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 0


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1308", "860", "670", "550", "422", "291"):
        assert leak not in blob, f"plan leaks {leak!r}"
