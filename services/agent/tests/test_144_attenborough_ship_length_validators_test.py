"""
Offline unit tests for the RRS Sir David Attenborough eponym RE-EXPANSION task (test 144) — free, no LLM.

Bucket D (under-grounded re-expansion trigger). Covers the leak-resistant keystone gate (the ship's
length), the UN-gated re-expansion coverage diagnostic (both targeted steps, capped by visits), the
keystone-gated resolution/citation secondaries, single- and multi-line correct layouts, and the
adversarial failure modes: the wrong-DIMENSION trap (reporting the beam/tonnage off the correct page),
an UNKNOWN/insufficient-first-page answer, and a near-miss number. Plus the compiled plan is a 2-hop
re-expansion chain that templates upstream and leaks nothing.
"""
from agent.app.idea_tests import test_144_tier5_attenborough_ship_length as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_FULL_SINGLE = (
    "The David Attenborough biography names the ship but not its length. Resolving the eponym: the "
    "RRS Sir David Attenborough (a polar research vessel); "
    "https://en.wikipedia.org/wiki/RRS_Sir_David_Attenborough gives a length of 128.9 m (423 ft)."
)

_FULL_MULTI = (
    "STEP 1 (insufficient): David Attenborough bio — names the ship, no length.\n"
    "STEP 2 (re-expanded): RRS Sir David Attenborough (research ship)\n"
    "  https://en.wikipedia.org/wiki/RRS_Sir_David_Attenborough\n"
    "  length:\n"
    "  128.9\n"
    "  m\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_length(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_length(r, _OBS)["score"] == 1.0
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_length(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_length(r, ungrounded_obs)["passed"] is False
    assert t.validate_target_resolution(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_reexpansion_coverage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_length(r, ungrounded_obs)["score"],
        t.validate_reexpansion_coverage(r, ungrounded_obs)["score"],
        t.validate_target_resolution(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_dimension_trap_gates_to_zero_but_keeps_coverage():
    wrong = (
        "Attenborough -> the RRS Sir David Attenborough research ship; "
        "https://en.wikipedia.org/wiki/RRS_Sir_David_Attenborough. Its beam is 24 m and gross tonnage "
        "is 15,000 GT."
    )
    r = _r(wrong)
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0       # beam/tonnage not the keystone
    assert t.validate_reexpansion_coverage(r, _OBS)["score"] == 1.0  # both steps still evidenced
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0     # gated
    assert t.validate_citations(r, _OBS)["score"] == 0.0            # gated


def test_unknown_insufficient_first_page_answer_gates_to_zero():
    r = _r("The David Attenborough biography names the ship but not its length, so length is UNKNOWN.")
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0
    assert t.validate_target_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_near_miss_but_accepts_standard_roundings():
    """Unit-tolerance fix (F26): a standard rounding ("129 m") or extra precision ("128.90") is a
    CORRECT grounded answer and must now score 1.0 -- only a genuinely different value (an
    embedded/near-miss digit run, or the beam/tonnage decoys) is rejected."""
    assert t.validate_keystone_length(_r("value 128.90 marker"), _OBS)["score"] == 1.0
    assert t.validate_keystone_length(_r("length 129 m"), _OBS)["score"] == 1.0
    assert t.validate_keystone_length(_r("code 1128.9 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("beam 24 m"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_half_and_gate_zero():
    r = _r("I only read the David Attenborough biography and stopped there.")
    assert abs(t.validate_reexpansion_coverage(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0


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
         "deliverables": ["RRS Sir David Attenborough length: 128.9 m", "research vessel"]}
    assert t.validate_keystone_length(r, _OBS)["score"] == 1.0


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
    assert "{attenborough}" in by_id["ship_length"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("128.9", "423"):
        assert leak not in blob, f"plan leaks {leak!r}"
