"""
Offline unit tests for the Beethoven-crater eponym RE-EXPANSION task (test 143) — free, no LLM.

Bucket D (under-grounded re-expansion trigger). Covers the leak-resistant keystone gate (the
Mercurian crater's diameter), the UN-gated re-expansion coverage diagnostic (both targeted steps,
capped by visits), the keystone-gated resolution/citation secondaries, single- and multi-line correct
layouts, and the adversarial failure modes: an UNKNOWN/insufficient-first-page answer (the composer
bio has no crater) and a near-miss number. Plus the compiled plan is a 2-hop re-expansion chain that
templates upstream and leaks nothing.
"""
from agent.app.idea_tests import test_143_tier5_beethoven_crater_diameter as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_FULL_SINGLE = (
    "The Beethoven biography says nothing about a crater. Resolving the eponym: Beethoven is a large "
    "impact basin on Mercury; https://en.wikipedia.org/wiki/Beethoven_(crater) gives a diameter of "
    "630 km."
)

_FULL_MULTI = (
    "STEP 1 (insufficient): Ludwig van Beethoven bio — no crater.\n"
    "STEP 2 (re-expanded): Beethoven crater on Mercury\n"
    "  https://en.wikipedia.org/wiki/Beethoven_(crater)\n"
    "  diameter:\n"
    "  630\n"
    "  km\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_diameter(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_diameter(r, ungrounded_obs)["passed"] is False
    assert t.validate_target_resolution(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_reexpansion_coverage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_diameter(r, ungrounded_obs)["score"],
        t.validate_reexpansion_coverage(r, ungrounded_obs)["score"],
        t.validate_target_resolution(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_unknown_insufficient_first_page_answer_gates_to_zero():
    r = _r("The Beethoven biography focuses on his music and mentions no crater on Mercury; UNKNOWN.")
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 0.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_diameter(_r("code 1630 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_diameter(_r("value 6300"), _OBS)["score"] == 0.0
    assert t.validate_keystone_diameter(_r("about 63.0 km"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_half_and_gate_zero():
    r = _r("I only read the Beethoven biography and stopped there.")
    assert abs(t.validate_reexpansion_coverage(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 0.0


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
         "deliverables": ["Beethoven crater (Mercury) diameter: 630 km", "crater on Mercury"]}
    assert t.validate_keystone_diameter(r, _OBS)["score"] == 1.0


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
    assert "{beethoven}" in by_id["crater_diameter"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("630", "390"):
        assert leak not in blob, f"plan leaks {leak!r}"
