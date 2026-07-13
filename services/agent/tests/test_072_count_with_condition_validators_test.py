"""
Offline unit tests for the count-with-condition lake-depth task (test 072) — free.

Mirrors the adversarial suite shared with 073/090 for the COUNT keystone (KEYSTONE_COUNT = 4
lakes with max depth > 480 m, reported in deliverables[0]):

  * a correct full answer scores 1.0 on every validator (single- and multi-line layout);
  * a wrong count (3, 5, or the naive 'all seven' 7) gates the keystone to 0 AND short-circuits
    the gated secondaries (passing-lake names, citation) to 0, while the UN-gated coverage
    diagnostic is retained (all seven depths still credited);
  * the grounding requirement: a correct-value answer with zero visits must not earn credit;
  * partial coverage yields the exact gathered fraction;
  * a no-visits run gates the visit process metric to 0;
  * the compiled plan is a well-formed 7-way independent fan-out that leaks no depth value.
"""
from agent.app.idea_tests import test_072_tier5_count_with_condition as t
from agent.app.testing import compiled_plan as cp


_OBS = {"visit": {"count": 7}}

_BODY_SINGLE = (
    "Passing (> 480 m): Lake Matano 590 m, Hornindalsvatnet 514 m, Quesnel Lake 511 m, "
    "Sarez Lake 505 m. Failing: Tinnsjå 460 m, Lake Manapouri 444 m, Lake Ohrid 288 m. "
    "Sources: https://en.wikipedia.org/wiki/Lake_Matano "
    "https://en.wikipedia.org/wiki/Hornindalsvatnet "
    "https://en.wikipedia.org/wiki/Quesnel_Lake "
    "https://en.wikipedia.org/wiki/Sarez_Lake "
    "https://en.wikipedia.org/wiki/Tinnsj%C3%A5 "
    "https://en.wikipedia.org/wiki/Lake_Manapouri "
    "https://en.wikipedia.org/wiki/Lake_Ohrid"
)

_BODY_MULTI = (
    "Restated depths:\n"
    "  Lake Matano: 590 m — EXCEEDS 480 m\n"
    "  Hornindalsvatnet: 514 m — EXCEEDS 480 m\n"
    "  Quesnel Lake: 511 m — EXCEEDS 480 m\n"
    "  Sarez Lake: 505 m — EXCEEDS 480 m\n"
    "  Tinnsjå: 460 m — below 480 m\n"
    "  Lake Manapouri: 444 m — below 480 m\n"
    "  Lake Ohrid: 288 m — below 480 m\n"
    "Sources:\n"
    "  https://en.wikipedia.org/wiki/Lake_Matano\n"
    "  https://en.wikipedia.org/wiki/Hornindalsvatnet\n"
    "  https://en.wikipedia.org/wiki/Quesnel_Lake\n"
    "  https://en.wikipedia.org/wiki/Sarez_Lake\n"
    "  https://en.wikipedia.org/wiki/Tinnsj%C3%A5\n"
    "  https://en.wikipedia.org/wiki/Lake_Manapouri\n"
    "  https://en.wikipedia.org/wiki/Lake_Ohrid\n"
)


def _result(count_primary, body):
    return {"deliverables": [count_primary, body], "output": {"final_deliverable": f"{count_primary}. {body}"}}


def test_full_answer_single_line_scores_all():
    r = _result("4", _BODY_SINGLE)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_passing_lakes(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _result("4", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_passing_lakes(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_wrong_count_gates_keystone_and_secondaries_but_keeps_coverage():
    r = _result("5", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False
    assert t.validate_passing_lakes(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_naive_all_seven_fails_keystone():
    r = _result("7", _BODY_MULTI)
    assert t.validate_keystone_count(r, _OBS)["passed"] is False


def test_ungrounded_correct_count_gates_to_zero():
    """Grounding requirement: the correct keystone COUNT alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess must
    collapse the keystone gate (and everything gated on it) to 0."""
    r = _result("4", _BODY_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_count(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_count(r, ungrounded_obs)["passed"] is False
    assert t.validate_passing_lakes(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_count(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_passing_lakes(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_scores_fraction():
    partial = "Lake Matano 590 m, Hornindalsvatnet 514 m."
    r = _result("4", partial)
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 2 / 7) < 1e-9


def test_no_visits_gates_visit_metric():
    r = _result("4", _BODY_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False


def test_compiled_plan_is_well_formed_seven_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 7
    assert struct["edge_count"] == 0


def test_compiled_plan_leaks_no_depth_value():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for depth in ("590", "514", "511", "505", "460", "444", "288"):
        assert depth not in blob, f"plan leaks depth {depth!r}"
