"""
Offline unit tests for the 5535 Annefrank eponym RE-EXPANSION task (test 142) — free, no LLM.

Bucket D (under-grounded re-expansion trigger). Covers the leak-resistant keystone gate (the
asteroid's orbital period), the UN-gated re-expansion coverage diagnostic (both targeted steps,
capped by visits), the keystone-gated resolution/citation secondaries, single- and multi-line
correct layouts, and the adversarial failure modes: the wrong-PROPERTY trap (reporting the diameter
off the correct page), an UNKNOWN/insufficient-first-page answer, and a near-miss number. Plus the
compiled plan is a 2-hop re-expansion chain that templates upstream and leaks nothing.
"""
from agent.app.idea_tests import test_142_tier5_annefrank_asteroid_period as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_FULL_SINGLE = (
    "The Anne Frank biography does not give the asteroid's orbit. Resolving the eponym: asteroid 5535 "
    "Annefrank; https://en.wikipedia.org/wiki/5535_Annefrank lists an orbital period of 3.29 yr."
)

_FULL_MULTI = (
    "STEP 1 (insufficient): Anne Frank bio — no orbital period.\n"
    "STEP 2 (re-expanded): 5535 Annefrank minor-planet page\n"
    "  https://en.wikipedia.org/wiki/5535_Annefrank\n"
    "  orbital period:\n"
    "  3.29\n"
    "  yr\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_period(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_period(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_wrong_property_trap_gates_to_zero_but_keeps_coverage():
    wrong = (
        "Anne Frank -> asteroid 5535 Annefrank; https://en.wikipedia.org/wiki/5535_Annefrank. Its "
        "mean diameter is 4.34 km and it rotates every 15.12 h."
    )
    r = _r(wrong)
    assert t.validate_keystone_period(r, _OBS)["score"] == 0.0       # diameter/rotation not the keystone
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0  # both steps still evidenced
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0     # gated
    assert t.validate_citations(r, _OBS)["score"] == 0.0            # gated


def test_unknown_insufficient_first_page_answer_gates_to_zero():
    r = _r("The Anne Frank biography names the asteroid but gives no orbital period; UNKNOWN.")
    assert t.validate_keystone_period(r, _OBS)["score"] == 0.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_period(_r("code 13.29 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_period(_r("value 3.290 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_period(_r("diameter 4.34 km"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_half_and_gate_zero():
    r = _r("I only read the Anne Frank biography and stopped there.")
    assert abs(t.validate_reexpansion_coverage(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_keystone_period(r, _OBS)["score"] == 0.0


def test_coverage_capped_by_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_reexpansion_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_reexpansion_coverage(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_reexpansion_coverage(r, {"visit": {"count": 2}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["5535 Annefrank orbital period: 3.29 yr", "asteroid: 5535 Annefrank"]}
    assert t.validate_keystone_period(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_two_hop_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 2
    assert struct["edge_count"] == 1
    assert struct["wave_widths"] == [1, 1]
    assert struct["is_dag_chain"] is True
    assert struct["is_pure_fanout"] is False


def test_compiled_plan_templates_upstream_and_leaks_nothing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{anne_frank}" in by_id["asteroid_period"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("3.29", "1,202", "1202"):
        assert leak not in blob, f"plan leaks {leak!r}"
