"""
Offline unit tests for the One World Trade Center conflicting-source task (test 131) — free, no LLM.

Covers the keystone gate (architectural incl. spire 1,776 ft / 541.3 m) that MUST reject the roof
scope (1,368 ft / 417 m) and any averaged value; the UN-gated reconciliation coverage diagnostic
(both scopes surfaced, retained when the pick is wrong, gated on read-evidence); the keystone-
gated scope-identification and citation secondaries; single- and multi-line layout; and the
compiled plan (2 -> 1) that leaks nothing.
"""
from agent.app.idea_tests import test_131_tier5_wtc_spire_architectural_height as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "The 1,368 ft (417 m) figure is only the roof height. The CTBUH ruled the mast is a spire, so "
    "the official architectural height including the spire is 1,776 ft (541.3 m). Source: "
    "https://en.wikipedia.org/wiki/One_World_Trade_Center"
)

_FULL_MULTI = (
    "1 WTC height reconciliation:\n"
    "  roof (occupied top, spire excluded): 1,368 ft / 417 m\n"
    "  official architectural (CTBUH, spire counts): 1,776 ft\n"
    "    541.3 m\n"
    "The architectural height is the ranked height.\n"
    "  https://en.wikipedia.org/wiki/One_World_Trade_Center\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True


def test_wrong_source_pick_gates_but_keeps_coverage():
    wrong = (
        "The roof is 1,368 ft and the architectural top is 1,776 ft. I report 1 WTC as 1,368 ft "
        "(417 m) tall. https://en.wikipedia.org/wiki/One_World_Trade_Center"
    )
    r = {"output": {"final_deliverable": "1 WTC: 1,368 ft (417 m)"},
         "deliverables": ["1 WTC: 1,368 ft (417 m)", wrong]}
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_averaged_value_gates_to_zero():
    avg = _r("Splitting the difference gives about 1,572 ft (479 m). https://en.wikipedia.org/wiki/One_World_Trade_Center")
    assert t.validate_keystone_height(avg, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_and_near_miss():
    assert t.validate_keystone_height(_r("1,368 ft (417 m)"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("observatory 1,268 ft (386 m)"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("about 1,572 ft"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("541.3 m architectural"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    r = _r("The architectural height is 1,776 ft (541.3 m).")
    assert abs(t.validate_reconciliation_coverage(r, _OBS)["score"] - 0.5) < 1e-9


def test_coverage_requires_read_evidence():
    r = _r(_FULL_SINGLE.replace("https://en.wikipedia.org/wiki/One_World_Trade_Center", ""))
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 1}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["1 WTC architectural height: 1,776 ft (541.3 m)", "roof is 1,368 ft"]}
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0


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
    for leak in ("1,776", "1776", "541"):
        assert leak not in blob, f"plan leaks {leak!r}"
