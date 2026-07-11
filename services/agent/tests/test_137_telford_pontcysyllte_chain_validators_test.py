"""
Offline unit tests for the Telford -> Pontcysyllte stop/continue chain (test 137) — no LLM.

Keystone gate (Pontcysyllte length 307 m / 336 yd), UN-gated chain-coverage (capped by visits),
gated terminal-resolution + citations, single/multi-line layouts, and the two Bucket-C failure
modes: STOP-EARLY (Menai bridge's own span) and OVER-HOP (a different Telford work, the Caledonian
Canal). Compiled plan is a genuine dag chain that templates its predecessor and leaks nothing.
"""
from agent.app.idea_tests import test_137_tier5_telford_pontcysyllte_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Hop 1: the Menai Suspension Bridge (https://en.wikipedia.org/wiki/Menai_Suspension_Bridge) was "
    "engineered by Thomas Telford (https://en.wikipedia.org/wiki/Thomas_Telford). Hop 2: he built "
    "the Pontcysyllte Aqueduct (https://en.wikipedia.org/wiki/Pontcysyllte_Aqueduct) over the River "
    "Dee, completed 1805. Hop 3: its total length is 336 yd (307 m)."
)

_FULL_MULTI = (
    "HOP 1 — engineer:\n"
    "  Menai Suspension Bridge -> Thomas Telford\n"
    "    https://en.wikipedia.org/wiki/Menai_Suspension_Bridge\n"
    "    https://en.wikipedia.org/wiki/Thomas_Telford\n"
    "HOP 2 — terminal (over the River Dee):\n"
    "  Pontcysyllte Aqueduct\n"
    "    https://en.wikipedia.org/wiki/Pontcysyllte_Aqueduct\n"
    "HOP 3 — total length:\n"
    "  336 yd\n"
    "  (307 m)\n"
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
    wrong = "The Menai Suspension Bridge, by Thomas Telford, has a main span of 577 ft (176 m)."
    r = _r(wrong)
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_over_hop_gates_to_zero():
    wrong = "Thomas Telford also built the Caledonian Canal, about 97 km (60 mi) long."
    r = _r(wrong)
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_length(_r("code 3070 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("marker 3365"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("code 30755 units"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I only investigated the Menai Suspension Bridge and Thomas Telford; not the aqueduct."
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
         "deliverables": ["Pontcysyllte Aqueduct length: 336 yd (307 m)", "engineer: Telford"]}
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
    for leak in ("307", "336", "127"):
        assert leak not in blob, f"plan leaks {leak!r}"
