"""
Offline unit tests for the Denali-resurvey conflicting-source task (test 130) — free, no LLM.

Covers the keystone gate (2015 USGS 20,310 ft / 6,190 m) that MUST reject the older 1952 value
(20,320 ft / 6,194 m) and any averaged value; the UN-gated reconciliation coverage diagnostic
(both figures surfaced, retained when the pick is wrong, gated on read-evidence); the keystone-
gated recency-identification and citation secondaries; single- and multi-line layout; and the
compiled plan (2 -> 1) that leaks nothing.
"""
from agent.app.idea_tests import test_130_tier5_denali_resurvey_height as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "The 20,320 ft (6,194 m) figure is the older 1952 photogrammetric measurement; the 2015 USGS "
    "GPS resurvey found Denali to be 20,310 ft (6,190 m), which supersedes it. Source: "
    "https://en.wikipedia.org/wiki/Denali"
)

_FULL_MULTI = (
    "Denali elevation reconciliation:\n"
    "  older (1952 photogrammetry): 20,320 ft / 6,194 m\n"
    "  current (2015 USGS GPS resurvey): 20,310 ft\n"
    "    6,190 m\n"
    "The 2015 resurvey supersedes the 1952 figure.\n"
    "  https://en.wikipedia.org/wiki/Denali\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True


def test_wrong_source_pick_gates_but_keeps_coverage():
    wrong = (
        "The 1952 survey gave 20,320 ft and the 2015 survey 20,310 ft. I report Denali as 20,320 ft "
        "(6,194 m). https://en.wikipedia.org/wiki/Denali"
    )
    r = {"output": {"final_deliverable": "Denali: 20,320 ft (6,194 m)"},
         "deliverables": ["Denali: 20,320 ft (6,194 m)", wrong]}
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_averaged_value_gates_to_zero():
    avg = _r("Averaging the two surveys gives about 20,315 ft (6,192 m). https://en.wikipedia.org/wiki/Denali")
    assert t.validate_keystone_elevation(avg, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_and_near_miss():
    assert t.validate_keystone_elevation(_r("20,320 ft (6,194 m)"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("about 20,315 ft"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("6,192 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("6,190 m current"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    r = _r("Denali's current elevation is 20,310 ft (6,190 m).")
    assert abs(t.validate_reconciliation_coverage(r, _OBS)["score"] - 0.5) < 1e-9


def test_coverage_requires_read_evidence():
    r = _r(_FULL_SINGLE.replace("https://en.wikipedia.org/wiki/Denali", ""))
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 1}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Denali current elevation: 20,310 ft (6,190 m)", "older was 20,320 ft"]}
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0


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
    for leak in ("20,310", "20310", "6,190", "6190"):
        assert leak not in blob, f"plan leaks {leak!r}"
