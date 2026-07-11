"""
Offline unit tests for the Brunel -> SS Great Eastern stop/continue chain (test 136) — no LLM.

Keystone gate (SS Great Eastern length 692 ft / 211 m), UN-gated chain-coverage (capped by visits),
gated terminal-resolution + citations, single/multi-line layouts, and the two Bucket-C failure
modes: STOP-EARLY (Clifton bridge's own span 702 ft) and OVER-HOP (a different Brunel ship, SS Great
Britain 322 ft). Compiled plan is a genuine dag chain that templates its predecessor and leaks nothing.
"""
from agent.app.idea_tests import test_136_tier5_brunel_great_eastern_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Hop 1: the Clifton Suspension Bridge (https://en.wikipedia.org/wiki/Clifton_Suspension_Bridge) "
    "was designed by Isambard Kingdom Brunel (https://en.wikipedia.org/wiki/Isambard_Kingdom_Brunel). "
    "Hop 2: he designed the SS Great Eastern (https://en.wikipedia.org/wiki/SS_Great_Eastern), the "
    "largest ship ever built at her 1858 launch. Hop 3: her length is 692 ft (211 m)."
)

_FULL_MULTI = (
    "HOP 1 — engineer:\n"
    "  Clifton Suspension Bridge -> Isambard Kingdom Brunel\n"
    "    https://en.wikipedia.org/wiki/Clifton_Suspension_Bridge\n"
    "    https://en.wikipedia.org/wiki/Isambard_Kingdom_Brunel\n"
    "HOP 2 — terminal ship:\n"
    "  SS Great Eastern\n"
    "    https://en.wikipedia.org/wiki/SS_Great_Eastern\n"
    "HOP 3 — length:\n"
    "  692 ft\n"
    "  (211 m)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_length(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_length(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_stop_early_gates_to_zero_but_keeps_coverage():
    wrong = (
        "The Clifton Suspension Bridge, designed by Isambard Kingdom Brunel, has a main span of "
        "702 ft (214 m)."
    )
    r = _r(wrong)
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_over_hop_gates_to_zero():
    wrong = "Brunel's earlier famous ship, the SS Great Britain, was 322 ft (98 m) long."
    r = _r(wrong)
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_length(_r("code 6920 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("marker 2115"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("ratio 2.11 units"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I only investigated the Clifton Suspension Bridge and Brunel; I never reached the ship."
    r = _r(text)
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0


def test_chain_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, {"visit": {"count": 2}})["score"] - 2 / 3) < 1e-9
    assert t.validate_chain_coverage(r, {"visit": {"count": 3}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["SS Great Eastern length: 692 ft (211 m)", "engineer: Brunel"]}
    assert t.validate_keystone_length(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_dag_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 2
    assert struct["wave_widths"] == [1, 1, 1]
    assert struct["is_dag_chain"] is True
    assert struct["is_pure_fanout"] is False


def test_compiled_plan_templates_upstream_and_leaks_nothing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{creator}" in by_id["other_work"]["instruction"]
    assert "{other_work}" in by_id["figure"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("692", "211"):
        assert leak not in blob, f"plan leaks {leak!r}"
