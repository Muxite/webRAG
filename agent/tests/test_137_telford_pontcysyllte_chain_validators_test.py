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


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage now checks
# instead of an aggregate visit count -- see idea_test_utils.waypoint_chain_coverage).
_EV_START = {"url": "https://en.wikipedia.org/wiki/Menai_Suspension_Bridge",
             "content": "The Menai Suspension Bridge was engineered by Thomas Telford."}
_EV_CREATOR = {"url": "https://en.wikipedia.org/wiki/Thomas_Telford",
               "content": "Thomas Telford was a Scottish civil engineer, architect and stonemason."}
_EV_TERMINAL = {"url": "https://en.wikipedia.org/wiki/Pontcysyllte_Aqueduct",
                "content": "The Pontcysyllte Aqueduct carries the Llangollen Canal over the River "
                           "Dee, with a total length of 336 yd (307 m)."}
_FULL_EVIDENCE = [_EV_START, _EV_CREATOR, _EV_TERMINAL]


def _obs(visited=None, n=4):
    return {"visit": {"count": n}, "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()

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


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_length(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_length(r, ungrounded_obs)["passed"] is False
    assert t.validate_terminal_resolution(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_chain_coverage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_length(r, ungrounded_obs)["score"],
        t.validate_chain_coverage(r, ungrounded_obs)["score"],
        t.validate_terminal_resolution(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_stop_early_gates_to_zero_but_keeps_coverage():
    wrong = "The Menai Suspension Bridge, by Thomas Telford, has a main span of 577 ft (176 m)."
    r = _r(wrong)
    partial_obs = _obs([_EV_START, _EV_CREATOR])  # terminal page never visited
    assert t.validate_keystone_length(r, partial_obs)["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, partial_obs)["score"] - 2 / 3) < 1e-9
    assert t.validate_terminal_resolution(r, partial_obs)["score"] == 0.0
    assert t.validate_citations(r, partial_obs)["score"] == 0.0


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
    partial_obs = _obs([_EV_START, _EV_CREATOR])
    assert abs(t.validate_chain_coverage(r, partial_obs)["score"] - 2 / 3) < 1e-9
    assert t.validate_keystone_length(r, partial_obs)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """GROUNDING fix (2026-08-16): a waypoint named in the answer with NO supporting visited-page
    evidence must not be credited, regardless of how many OTHER pages were visited (no more
    aggregate visit-count cap). Real corpus example (task 137, csnopar_g flash/good_adaptive): the
    model visited the START page TWICE and never opened Telford's or the aqueduct's own page, yet
    confidently named and cited both -- the old aggregate-visit cap (2 visits >= 2 named) credited
    the full 3/3 anyway. A non-Wikipedia source that genuinely covers the fact (e.g. a heritage
    society article about the aqueduct) still earns credit via content, not just a wiki/ slug."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    # Visited only the start page (twice) -- it names Telford itself (HOP 1's own design), so
    # "start"/"creator" are grounded but "terminal" (Pontcysyllte) is not.
    twice_start_only = [_EV_START, _EV_START]
    assert abs(t.validate_chain_coverage(r, _obs(twice_start_only, n=2))["score"] - 2 / 3) < 1e-9
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0
    non_wiki_terminal = {"url": "https://www.canalrivertrust.org.uk/pontcysyllte-aqueduct",
                          "content": "The Pontcysyllte Aqueduct, engineered by Thomas Telford, has "
                                     "a total length of 336 yd (307 m)."}
    assert t.validate_chain_coverage(
        r, _obs([_EV_START, _EV_CREATOR, non_wiki_terminal])
    )["score"] == 1.0


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
