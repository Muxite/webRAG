"""
Offline unit tests for the Pluto-diameter conflicting-source task (test 128) — free, no LLM.

Covers the keystone gate (refined REX diameter 2,376.6 km / radius 1,188.3 km) that MUST reject
the preliminary first-hit value (2,370 km) and any averaged value; the UN-gated reconciliation
coverage diagnostic (both figures surfaced, retained when the pick is wrong, gated on read-
evidence); the keystone-gated source-identification and citation secondaries; the correct answer
in single- and multi-line layout; and the compiled plan (2 -> 1) that leaks nothing.
"""
import re

from agent.app.idea_tests import test_128_tier5_pluto_diameter_revision as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "New Horizons first announced ~2,370 km on 13 July 2015, but that was the PRELIMINARY figure; "
    "it was later refined by the Radio Science Experiment (REX) to a diameter of 2,376.6 km (mean "
    "radius 1,188.3 km), which supersedes the first-hit value. Source: "
    "https://en.wikipedia.org/wiki/Pluto"
)

_FULL_MULTI = (
    "Reconciliation of the two figures:\n"
    "  preliminary (first announced 13 Jul 2015): 2,370 km\n"
    "  refined (REX radio-occultation, final): 2,376.6 km\n"
    "    mean radius 1,188.3 km\n"
    "The refined REX measurement supersedes the preliminary one.\n"
    "  https://en.wikipedia.org/wiki/Pluto\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_scores_near_zero():
    """Right keystone value present, but zero visits (no grounding) and no source citation in text
    -> keystone and every keystone-gated secondary must collapse to 0, even though the value string
    matches."""
    r = _r(_FULL_SINGLE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_diameter(r, ungrounded)["score"] == 0.0
    assert t.validate_identifies_correct_source(r, ungrounded)["score"] == 0.0
    assert t.validate_citation(r, ungrounded)["score"] == 0.0
    overall = sum(v["score"] for v in [
        t.validate_keystone_diameter(r, ungrounded),
        t.validate_identifies_correct_source(r, ungrounded),
        t.validate_citation(r, ungrounded),
    ]) / 3.0
    assert overall < 0.75


def test_wrong_source_pick_gates_but_keeps_coverage():
    # Agent saw both but chose the first-hit preliminary value as the answer.
    wrong = (
        "New Horizons announced 2,370 km and later 2,376 km. I report Pluto's diameter as 2,370 km. "
        "https://en.wikipedia.org/wiki/Pluto"
    )
    r = {"output": {"final_deliverable": "Pluto diameter: 2,370 km"},
         "deliverables": ["Pluto diameter: 2,370 km", wrong]}
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 0.0     # picked wrong
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0  # both figures present, retained
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 0.0  # gated
    assert t.validate_citation(r, _OBS)["score"] == 0.0             # gated


def test_averaged_value_gates_to_zero():
    avg = _r("Reconciling the two figures, Pluto's diameter is about 2,373 km. https://en.wikipedia.org/wiki/Pluto")
    assert t.validate_keystone_diameter(avg, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_and_near_miss():
    assert t.validate_keystone_diameter(_r("diameter 2,370 km"), _OBS)["score"] == 0.0
    assert t.validate_keystone_diameter(_r("2,372 km"), _OBS)["score"] == 0.0
    assert t.validate_keystone_diameter(_r("2,374 km"), _OBS)["score"] == 0.0
    assert t.validate_keystone_diameter(_r("about 2,373 km average"), _OBS)["score"] == 0.0
    assert t.validate_keystone_diameter(_r("radius 1,188.3 km"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    r = _r("Pluto's refined diameter is 2,376.6 km.")  # only correct side, no preliminary
    assert abs(t.validate_reconciliation_coverage(r, _OBS)["score"] - 0.5) < 1e-9


def test_coverage_requires_read_evidence():
    r = _r(_FULL_SINGLE.replace("https://en.wikipedia.org/wiki/Pluto", ""))
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 1}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Pluto refined diameter: 2,376.6 km", "preliminary was 2,370 km"]}
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 1.0


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
    for leak in ("2,376", "2376", "1,188", "1188"):
        assert leak not in blob, f"plan leaks {leak!r}"
