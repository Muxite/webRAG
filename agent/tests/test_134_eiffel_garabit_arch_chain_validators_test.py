"""
Offline unit tests for the Eiffel -> Garabit stop/continue chain (test 134) — free, no LLM.

Covers the leak-resistant keystone gate (Garabit length 565 m / arch 165 m), the UN-gated
chain-coverage diagnostic (waypoints traversed, capped by visits, retained when the answer is
wrong), the keystone-gated terminal-resolution and citation secondaries, single- and multi-line
layouts, and the two Bucket-C failure modes: STOP-EARLY (report the Statue of Liberty's own figure)
and OVER-HOP (report the Eiffel Tower). Plus the compiled plan is a genuine dag chain (3 leaves,
one per wave) that templates its direct predecessor and leaks nothing.
"""
from agent.app.idea_tests import test_134_tier5_eiffel_garabit_arch_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage now checks
# instead of an aggregate visit count -- see idea_test_utils.waypoint_chain_coverage).
_EV_START = {"url": "https://en.wikipedia.org/wiki/Statue_of_Liberty",
             "content": "The Statue of Liberty's internal iron frame was engineered by Gustave Eiffel."}
_EV_CREATOR = {"url": "https://en.wikipedia.org/wiki/Gustave_Eiffel",
               "content": "Gustave Eiffel was a French structural engineer who also designed the "
                          "internal frame of the Statue of Liberty."}
_EV_TERMINAL = {"url": "https://en.wikipedia.org/wiki/Garabit_viaduct",
                "content": "The Garabit Viaduct's total length is 565 m (main arch span 165 m)."}
_FULL_EVIDENCE = [_EV_START, _EV_CREATOR, _EV_TERMINAL]


def _obs(visited=None, n=4):
    return {"visit": {"count": n}, "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()

_FULL_SINGLE = (
    "Hop 1: the Statue of Liberty (https://en.wikipedia.org/wiki/Statue_of_Liberty) internal iron "
    "frame was engineered by Gustave Eiffel (https://en.wikipedia.org/wiki/Gustave_Eiffel). Hop 2: "
    "his Garabit Viaduct (https://en.wikipedia.org/wiki/Garabit_viaduct), a wrought-iron railway "
    "arch viaduct over the Truyère, completed 1884. Hop 3: its total length is 565 m (main arch "
    "span 165 m)."
)

_FULL_MULTI = (
    "HOP 1 — engineer:\n"
    "  Statue of Liberty -> Gustave Eiffel\n"
    "    https://en.wikipedia.org/wiki/Statue_of_Liberty\n"
    "    https://en.wikipedia.org/wiki/Gustave_Eiffel\n"
    "HOP 2 — terminal:\n"
    "  Garabit Viaduct (railway arch viaduct over the Truyère)\n"
    "    https://en.wikipedia.org/wiki/Garabit_viaduct\n"
    "HOP 3 — length:\n"
    "  565\n"
    "  metres\n"
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
    wrong = (
        "The Statue of Liberty's internal frame was engineered by Gustave Eiffel. The Statue of "
        "Liberty stands 46 m tall (93 m with the pedestal)."
    )
    r = _r(wrong)
    partial_obs = _obs([_EV_START, _EV_CREATOR])  # terminal page never visited
    assert t.validate_keystone_length(r, partial_obs)["score"] == 0.0           # no 565/165
    assert abs(t.validate_chain_coverage(r, partial_obs)["score"] - 2 / 3) < 1e-9  # start + engineer
    assert t.validate_terminal_resolution(r, partial_obs)["score"] == 0.0       # gated
    assert t.validate_citations(r, partial_obs)["score"] == 0.0                 # gated


def test_over_hop_gates_to_zero():
    wrong = (
        "Engineered by Gustave Eiffel; his most famous work, the Eiffel Tower, stands about 330 m "
        "tall (300 m to the roof)."
    )
    r = _r(wrong)
    assert t.validate_keystone_length(r, _OBS)["score"] == 0.0           # 330/300 != 565/165
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_length(_r("code 5650 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("marker 1655"), _OBS)["score"] == 0.0
    assert t.validate_keystone_length(_r("ratio 1.65 units"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I only investigated the Statue of Liberty and Gustave Eiffel; I did not reach the viaduct."
    r = _r(text)
    partial_obs = _obs([_EV_START, _EV_CREATOR])
    assert abs(t.validate_chain_coverage(r, partial_obs)["score"] - 2 / 3) < 1e-9
    assert t.validate_keystone_length(r, partial_obs)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """GROUNDING fix (2026-08-16): a waypoint named in the answer with NO supporting visited-page
    evidence must not be credited, regardless of how many OTHER pages were visited (no more
    aggregate visit-count cap) -- and a page that genuinely supports a waypoint restores credit
    even when the model's own answer would otherwise be capped by a low visit count."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    assert t.validate_chain_coverage(r, _obs([]))["passed"] is False
    assert abs(t.validate_chain_coverage(r, _obs([_EV_START, _EV_CREATOR]))["score"] - 2 / 3) < 1e-9
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0
    # A page that supports NONE of the named waypoints (off-topic junk visit) contributes nothing,
    # even though it is a real, successful visit -- see the donation/Reddit/TikTok regression in
    # idea_test_utils_test.py for the corpus-derived version of this case.
    junk = [{"url": "https://www.reddit.com/r/todayilearned/comments/abc123/", "content": "TIL something unrelated."}]
    assert t.validate_chain_coverage(r, _obs(junk, n=5))["score"] == 0.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Garabit Viaduct length: 565 m", "engineer: Gustave Eiffel"]}
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
    for leak in ("565", "165", "1,854", "541"):
        assert leak not in blob, f"plan leaks {leak!r}"
