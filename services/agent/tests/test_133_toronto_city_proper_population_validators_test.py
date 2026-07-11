"""
Offline unit tests for the Toronto city-proper conflicting-source task (test 133) — free, no LLM.

Covers the keystone gate (city-proper 2,794,356) that MUST reject the metro-area bait (6,202,225 /
6,712,341) and any averaged value; the UN-gated reconciliation coverage diagnostic (both scopes
surfaced, retained when the pick is wrong, gated on read-evidence); the keystone-gated scope-
identification and citation secondaries; single- and multi-line layout; and the compiled plan
(2 -> 1) that leaks nothing.
"""
from agent.app.idea_tests import test_133_tier5_toronto_city_proper_population as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "The ~6.2 million figure is the metropolitan area (Census Metropolitan Area 6,202,225). The "
    "city-proper administrative City of Toronto 2021 census population is 2,794,356. Source: "
    "https://en.wikipedia.org/wiki/Toronto"
)

_FULL_MULTI = (
    "Toronto 2021 census reconciliation:\n"
    "  metropolitan area (CMA): 6,202,225\n"
    "  city proper (City of Toronto): 2,794,356\n"
    "The city-proper administrative figure is the answer.\n"
    "  https://en.wikipedia.org/wiki/Toronto\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_population(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_population(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True


def test_wrong_source_pick_gates_but_keeps_coverage():
    wrong = (
        "The city proper is 2,794,356 and the metro is 6,202,225. I report Toronto's population as "
        "6,202,225. https://en.wikipedia.org/wiki/Toronto"
    )
    r = {"output": {"final_deliverable": "Toronto: 6,202,225"},
         "deliverables": ["Toronto: 6,202,225", wrong]}
    assert t.validate_keystone_population(r, _OBS)["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_averaged_value_gates_to_zero():
    avg = _r("Averaging city and metro gives about 4,498,290. https://en.wikipedia.org/wiki/Toronto")
    assert t.validate_keystone_population(avg, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_and_near_miss():
    assert t.validate_keystone_population(_r("6,202,225"), _OBS)["score"] == 0.0
    assert t.validate_keystone_population(_r("6,712,341 (GTA)"), _OBS)["score"] == 0.0
    assert t.validate_keystone_population(_r("about 4.5 million"), _OBS)["score"] == 0.0
    assert t.validate_keystone_population(_r("2,794,356 city proper"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    r = _r("The City of Toronto has 2,794,356 residents.")
    assert abs(t.validate_reconciliation_coverage(r, _OBS)["score"] - 0.5) < 1e-9


def test_coverage_requires_read_evidence():
    r = _r(_FULL_SINGLE.replace("https://en.wikipedia.org/wiki/Toronto", ""))
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 1}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["City of Toronto population: 2,794,356", "metro is 6,202,225"]}
    assert t.validate_keystone_population(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_two_then_one():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["wave_widths"] == [2, 1]
    assert struct["waves"][1] == ["reconcile"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("2,794,356", "2794356", "2,794", "794,356"):
        assert leak not in blob, f"plan leaks {leak!r}"
