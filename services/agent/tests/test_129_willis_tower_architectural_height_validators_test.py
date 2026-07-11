"""
Offline unit tests for the Willis-Tower conflicting-source task (test 129) — free, no LLM.

Covers the keystone gate (architectural 1,451 ft / 442 m) that MUST reject the antenna-tip bait
(1,729 ft / 527 m) and any averaged value; the UN-gated reconciliation coverage diagnostic (both
scopes surfaced, retained when the pick is wrong, gated on read-evidence); the keystone-gated
scope-identification and citation secondaries; single- and multi-line layout; and the compiled
plan (2 -> 1) that leaks nothing.
"""
from agent.app.idea_tests import test_129_tier5_willis_tower_architectural_height as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "The 1,729 ft (527 m) figure is the antenna TIP height after the 2000 extension, not the "
    "architectural height. The official CTBUH architectural height (antennas excluded) is 1,451 ft "
    "(442 m). Source: https://en.wikipedia.org/wiki/Willis_Tower"
)

_FULL_MULTI = (
    "Willis Tower heights:\n"
    "  antenna tip (incl. broadcast antennas): 1,729 ft / 527 m\n"
    "  official architectural (CTBUH, antennas excluded): 1,451 ft\n"
    "    442 m\n"
    "The architectural height is the ranked height.\n"
    "  https://en.wikipedia.org/wiki/Willis_Tower\n"
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
        "The tip is 1,729 ft and the architectural top is 1,451 ft. I report the Willis Tower as "
        "1,729 ft (527 m) tall. https://en.wikipedia.org/wiki/Willis_Tower"
    )
    r = {"output": {"final_deliverable": "Willis Tower: 1,729 ft (527 m)"},
         "deliverables": ["Willis Tower: 1,729 ft (527 m)", wrong]}
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_averaged_value_gates_to_zero():
    avg = _r("Splitting the difference, the height is about 1,590 ft (484.5 m). https://en.wikipedia.org/wiki/Willis_Tower")
    assert t.validate_keystone_height(avg, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_and_near_miss():
    assert t.validate_keystone_height(_r("1,729 ft (527 m)"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("roof 1,354 ft (413 m)"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("about 1,590 ft"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("442 m architectural"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    r = _r("The architectural height is 1,451 ft (442 m).")
    assert abs(t.validate_reconciliation_coverage(r, _OBS)["score"] - 0.5) < 1e-9


def test_coverage_requires_read_evidence():
    r = _r(_FULL_SINGLE.replace("https://en.wikipedia.org/wiki/Willis_Tower", ""))
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 1}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Willis Tower architectural height: 1,451 ft (442 m)", "tip is 1,729 ft"]}
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
    for leak in ("1,451", "1451", "442"):
        assert leak not in blob, f"plan leaks {leak!r}"
