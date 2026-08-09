"""
Offline unit tests for the bounded subset-sum + aggregate-distractor task (test 070) — free.

Covers the keystone gate (the computed first-four-season sum, 78 = 13+22+19+24, band [77,79]) in
single- and multi-line layout; the wrong-keystone (aggregate distractor 91) trap gating every
credit-bearing check to zero while the UN-gated coverage diagnostic is retained; the grounding
requirement (a correct-value answer with zero visits must not earn credit); partial coverage
scoring an exact fraction; the visit process metric; and the compiled plan being a well-formed
4-way independent fan-out that leaks no per-season figure or total.
"""
from agent.app.idea_tests import test_070_tier5_subset_sum_distractor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Chuck season 1: 13 episodes; season 2: 22 episodes; season 3: 19 episodes; season 4: "
    "24 episodes. Sum of the first four seasons: 78 (13+22+19+24=78). Sources: "
    "https://en.wikipedia.org/wiki/Chuck_(season_1) https://en.wikipedia.org/wiki/Chuck_(season_2) "
    "https://en.wikipedia.org/wiki/Chuck_(season_3) https://en.wikipedia.org/wiki/Chuck_(season_4)"
)

_FULL_MULTI = (
    "Per-season episode counts:\n"
    "  Season 1: 13 — https://en.wikipedia.org/wiki/Chuck_(season_1)\n"
    "  Season 2: 22 — https://en.wikipedia.org/wiki/Chuck_(season_2)\n"
    "  Season 3: 19 — https://en.wikipedia.org/wiki/Chuck_(season_3)\n"
    "  Season 4: 24 — https://en.wikipedia.org/wiki/Chuck_(season_4)\n"
    "13 + 22 + 19 + 24 = 78\n"
)


def _result(primary, body=""):
    return {"deliverables": [primary, body], "output": {"final_deliverable": f"{primary}. {body}"}}


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_subset_sum(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_perseason_figures(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_subset_sum(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_perseason_figures(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_distractor_total_gates_to_zero_but_keeps_coverage():
    # Grabbing the whole-series infobox total (91, the distractor) fails the keystone band.
    r = _result("91", _FULL_MULTI)
    assert t.validate_keystone_subset_sum(r, _OBS)["passed"] is False
    assert t.validate_perseason_figures(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_sum_gates_to_zero():
    """Grounding requirement: the correct keystone SUM alone must NOT earn credit if the agent
    never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess must
    collapse the keystone gate (and everything gated on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_subset_sum(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_subset_sum(r, ungrounded_obs)["passed"] is False
    assert t.validate_perseason_figures(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_subset_sum(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_perseason_figures(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_scores_fraction():
    partial = "Season 1: 13. Season 2: 22."
    r = _result("78", partial)
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 2 / 4) < 1e-9


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 4}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True


def test_compiled_plan_is_well_formed_four_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 4
    assert struct["edge_count"] == 0


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("13", "22", "19", "24", "78", "91"):
        assert leak not in blob, f"plan leaks {leak!r}"
