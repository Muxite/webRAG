"""
Offline unit tests for the Everest -> Waugh -> 1856 declared height chain (test 138) — no LLM.

Keystone gate (the 1856 publicly declared height, 29,002 ft), UN-gated chain-coverage (capped by
visits), gated terminal-resolution + citations, single/multi-line layouts, and the two Bucket-C
failure modes: STOP-EARLY (the rounded computed 29,000 ft / namesake) and OVER-HOP (the modern
re-surveyed 29,032 ft). Compiled plan is a genuine dag chain that templates its predecessor and
leaks nothing.
"""
from agent.app.idea_tests import test_138_tier5_everest_waugh_survey_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Hop 1: Mount Everest (https://en.wikipedia.org/wiki/Mount_Everest) is named after George "
    "Everest (https://en.wikipedia.org/wiki/George_Everest), but the name was proposed by his "
    "successor Andrew Scott Waugh (https://en.wikipedia.org/wiki/Andrew_Scott_Waugh). Hop 2/3: "
    "Waugh's Great Trigonometrical Survey publicly declared the height in 1856 as 29,002 ft, two "
    "feet above the exact 29,000 ft computed."
)

_FULL_MULTI = (
    "HOP 1 — who proposed the name:\n"
    "  named after George Everest; proposed by Andrew Scott Waugh\n"
    "    https://en.wikipedia.org/wiki/Mount_Everest\n"
    "    https://en.wikipedia.org/wiki/George_Everest\n"
    "    https://en.wikipedia.org/wiki/Andrew_Scott_Waugh\n"
    "HOP 2/3 — the 1856 declared survey height:\n"
    "  29,002\n"
    "  feet\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_stop_early_gates_to_zero_but_keeps_coverage():
    wrong = "Andrew Scott Waugh's survey computed Mount Everest at exactly 29,000 ft."
    r = _r(wrong)
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0            # 29,000 != 29,002
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9  # peak + namer
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_over_hop_gates_to_zero():
    wrong = "Andrew Scott Waugh named Mount Everest; its modern official height is 29,032 ft (8,849 m)."
    r = _r(wrong)
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0            # 29,032 over-hop
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_height(_r("computed 29,000 ft"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("1955 value 29,029 ft"), _OBS)["score"] == 0.0
    assert t.validate_keystone_height(_r("2020 value 29,032 ft"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I looked at Mount Everest and Andrew Scott Waugh, but not the announcement details."
    r = _r(text)
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_keystone_height(r, _OBS)["score"] == 0.0


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
         "deliverables": ["1856 declared height: 29,002 ft", "namer: Andrew Scott Waugh"]}
    assert t.validate_keystone_height(r, _OBS)["score"] == 1.0


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
    assert "{namer}" in by_id["survey"]["instruction"]
    assert "{survey}" in by_id["figure"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("29,002", "29002"):
        assert leak not in blob, f"plan leaks {leak!r}"
