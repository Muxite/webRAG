"""
Offline unit tests for the closest-to-reference argmin task (test 071) — free.

Covers the keystone gate (Great Slave Lake = closest candidate to Lake Matano by max depth) in
single- and multi-line layout; the deepest/shallowest decoy traps gating every credit-bearing
check to zero while the UN-gated coverage diagnostic is retained; the grounding requirement (a
correct-value answer with zero visits must not earn credit); partial coverage scoring an exact
fraction; the visit process metric; and the compiled plan being a well-formed 6-way independent
fan-out that leaks no depth figure or winner.
"""
from agent.app.idea_tests import test_071_tier5_closest_to_reference as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}

_FULL_SINGLE = (
    "Lake Matano (reference) 590 m; Lake Superior 406 m; Lake Kivu 480 m; Lake Tahoe 501 m; "
    "Great Slave Lake 614 m; Lake Malawi 706 m. Great Slave Lake is closest to Lake Matano's "
    "depth (|614-590|=24 m). Sources: https://en.wikipedia.org/wiki/Lake_Matano "
    "https://en.wikipedia.org/wiki/Lake_Superior https://en.wikipedia.org/wiki/Lake_Kivu "
    "https://en.wikipedia.org/wiki/Lake_Tahoe https://en.wikipedia.org/wiki/Great_Slave_Lake "
    "https://en.wikipedia.org/wiki/Lake_Malawi"
)

_FULL_MULTI = (
    "Depths (m):\n"
    "  Lake Matano (ref) - 590 - https://en.wikipedia.org/wiki/Lake_Matano\n"
    "  Lake Superior - 406 - https://en.wikipedia.org/wiki/Lake_Superior\n"
    "  Lake Kivu - 480 - https://en.wikipedia.org/wiki/Lake_Kivu\n"
    "  Lake Tahoe - 501 - https://en.wikipedia.org/wiki/Lake_Tahoe\n"
    "  Great Slave Lake - 614 - https://en.wikipedia.org/wiki/Great_Slave_Lake\n"
    "  Lake Malawi - 706 - https://en.wikipedia.org/wiki/Lake_Malawi\n"
    "Closest to reference:\n"
    "  Great Slave Lake (diff 24 m)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_closest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_closest_value(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_closest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_closest_value(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deepest_decoy_gates_to_zero_but_keeps_coverage():
    wrong = _FULL_MULTI.replace(
        "Closest to reference:\n  Great Slave Lake (diff 24 m)\n",
        "Deepest (and closest, in my view):\n  Lake Malawi is closest.\n",
    )
    r = _r(wrong)
    assert t.validate_keystone_closest(r, _OBS)["score"] == 0.0
    assert t.validate_closest_value(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_winner_gates_to_zero():
    """Grounding requirement: the correct keystone WINNER alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess must
    collapse the keystone gate (and everything gated on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_closest(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_closest(r, ungrounded_obs)["passed"] is False
    assert t.validate_closest_value(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_closest(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_closest_value(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_scores_fraction():
    partial = "Lake Matano 590 m. Great Slave Lake 614 m."
    r = _r(partial)
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 2 / 6) < 1e-9


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 6}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is True


def test_compiled_plan_is_well_formed_six_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 0


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("590", "406", "480", "501", "614", "706", "24"):
        assert leak not in blob, f"plan leaks {leak!r}"
