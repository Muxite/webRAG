"""
Offline unit tests for the Mount Adams same-name RE-EXPANSION task (test 140) — free, no LLM.

Bucket D (under-grounded re-expansion trigger). Covers the leak-resistant keystone gate (the New
Hampshire peak's elevation), the UN-gated re-expansion coverage diagnostic (both targeted steps,
capped by visits, retained even when the number is wrong), the keystone-gated resolution/citation
secondaries, the correct answer in single- and multi-line layout, and the adversarial failure modes:
the famous WRONG entity (the 12,281 ft Washington volcano), an UNKNOWN/insufficient-first-page answer,
and a near-miss number. Plus the compiled plan is a 2-hop re-expansion chain that templates upstream
and leaks nothing.
"""
from agent.app.idea_tests import test_140_tier5_mount_adams_disambiguation as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_FULL_SINGLE = (
    "A plain search hits Mount Adams (Washington), the 12,281 ft Cascade volcano — the wrong one. "
    "The question wants Mount Adams in the Presidential Range of New Hampshire (Coos County): "
    "https://en.wikipedia.org/wiki/Mount_Adams_(New_Hampshire) — its elevation is 5,793 ft (1,766 m)."
)

_FULL_MULTI = (
    "STEP 1 (obvious/wrong): Mount Adams, Washington -> 12,281 ft Cascade volcano.\n"
    "STEP 2 (re-expanded): Mount Adams, New Hampshire (Presidential Range)\n"
    "  https://en.wikipedia.org/wiki/Mount_Adams_(New_Hampshire)\n"
    "  elevation:\n"
    "  5,793\n"
    "  ft (1,766 m)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_elevation(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_elevation(r, ungrounded_obs)["passed"] is False
    assert t.validate_target_resolution(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_reexpansion_coverage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_elevation(r, ungrounded_obs)["score"],
        t.validate_reexpansion_coverage(r, ungrounded_obs)["score"],
        t.validate_target_resolution(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_famous_wrong_entity_gates_to_zero_but_keeps_coverage():
    wrong = (
        "Mount Adams is the 12,281 ft (3,743 m) stratovolcano in the Cascade Range of Washington "
        "state. https://en.wikipedia.org/wiki/Mount_Adams_(Washington)"
    )
    r = _r(wrong)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0     # 12,281 is not the keystone
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0      # gated
    assert t.validate_citations(r, _OBS)["score"] == 0.0             # gated


def test_unknown_insufficient_first_page_answer_gates_to_zero():
    r = _r("I found the Washington Mount Adams page but it doesn't match; elevation UNKNOWN.")
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_elevation(_r("code 15,793 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("value 57930"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("elev 11,766 m mark"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_half_and_gate_zero():
    r = _r("I only read Mount Adams (the obvious page) and stopped there.")
    assert abs(t.validate_reexpansion_coverage(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0


def test_coverage_capped_by_visits():
    r = _r(_FULL_SINGLE)  # names both entities
    assert t.validate_reexpansion_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reexpansion_coverage(r, {"visit": {"count": 0}})["passed"] is False
    assert abs(t.validate_reexpansion_coverage(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_reexpansion_coverage(r, {"visit": {"count": 2}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Mount Adams (NH) elevation: 5,793 ft (1,766 m)", "Presidential Range"]}
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0


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
    assert "{mount_adams}" in by_id["adams_elevation"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("5,793", "5793", "1,766", "1766"):
        assert leak not in blob, f"plan leaks {leak!r}"
