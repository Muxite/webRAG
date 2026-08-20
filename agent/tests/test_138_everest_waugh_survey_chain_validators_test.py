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


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage now checks
# instead of an aggregate visit count -- see idea_test_utils.waypoint_chain_coverage).
_EV_START = {"url": "https://en.wikipedia.org/wiki/Mount_Everest",
             "content": "Mount Everest is named in honour of Sir George Everest, though the name "
                        "was proposed by his successor."}
_EV_CREATOR = {"url": "https://en.wikipedia.org/wiki/Andrew_Scott_Waugh",
               "content": "Andrew Scott Waugh was the British Surveyor General of India who "
                          "succeeded George Everest in that post."}
_EV_TERMINAL = {"url": "https://en.wikipedia.org/wiki/George_Everest",
                "content": "Waugh's Great Trigonometrical Survey publicly declared the peak's "
                           "height in 1856 to be 29,002 ft."}
_FULL_EVIDENCE = [_EV_START, _EV_CREATOR, _EV_TERMINAL]


def _obs(visited=None, n=4):
    return {"visit": {"count": n}, "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()

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


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_height(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_height(r, ungrounded_obs)["passed"] is False
    assert t.validate_terminal_resolution(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_chain_coverage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_height(r, ungrounded_obs)["score"],
        t.validate_chain_coverage(r, ungrounded_obs)["score"],
        t.validate_terminal_resolution(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_stop_early_gates_to_zero_but_keeps_coverage():
    wrong = "Andrew Scott Waugh's survey computed Mount Everest at exactly 29,000 ft."
    r = _r(wrong)
    partial_obs = _obs([_EV_START, _EV_CREATOR])  # never visited a page for the 1856 declaration
    assert t.validate_keystone_height(r, partial_obs)["score"] == 0.0            # 29,000 != 29,002
    assert abs(t.validate_chain_coverage(r, partial_obs)["score"] - 2 / 3) < 1e-9  # peak + namer
    assert t.validate_terminal_resolution(r, partial_obs)["score"] == 0.0
    assert t.validate_citations(r, partial_obs)["score"] == 0.0


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
    partial_obs = _obs([_EV_START, _EV_CREATOR])
    assert abs(t.validate_chain_coverage(r, partial_obs)["score"] - 2 / 3) < 1e-9
    assert t.validate_keystone_height(r, partial_obs)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """GROUNDING fix (2026-08-16): a waypoint named in the answer with NO supporting visited-page
    evidence must not be credited, regardless of how many OTHER pages were visited (no more
    aggregate visit-count cap)."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, _obs([_EV_START, _EV_CREATOR]))["score"] - 2 / 3) < 1e-9
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0


def test_real_corpus_junk_visits_do_not_credit_waypoints_alone():
    """Regression, built from the real over-crediting cell this repair targets (task 138,
    csnopar_g/flash/good_adaptive): a run with 12 total visits -- Mount Everest (opened 4x), a
    Wikimedia donation page, a Reddit thread, and a TikTok video -- where the OLD validator
    credited "3/3 chain waypoints traversed" purely because the aggregate visit count (12) was
    >= 3, with zero regard for WHICH pages were visited.

    Visiting a donation page, a Reddit thread and a TikTok video contributes NOTHING on its own
    (this is the literal donation/Reddit/TikTok-must-not-credit-a-waypoint case)."""
    junk_only = [
        {"url": "https://wikimediafoundation.org/give", "content": "Donate to the Wikimedia Foundation..."},
        {"url": "https://www.reddit.com/r/todayilearned/comments/picihv/til/", "content": "TIL something else entirely."},
        {"url": "https://www.tiktok.com/@justin_danger_nunley/video/7644377213382495502", "content": "unrelated video caption"},
        {"url": "https://en.wikipedia.org/wiki/Mount_Everest", "content": "Mount Everest is Earth's highest mountain."},
    ]
    r = _r(_FULL_SINGLE)  # names all three waypoints in the answer text
    result_junk = t.validate_chain_coverage(r, _obs(junk_only, n=12))
    assert abs(result_junk["score"] - 1 / 3) < 1e-9  # only "start" (Everest) is genuinely grounded
    assert result_junk["passed"] is False

    # The REAL cell was not actually a bug once the multi-page visit is read correctly: the
    # "give" action's fetch was a two-URL batch that ALSO fetched a genuine, on-topic secondary
    # source (a history.com article covering both Waugh and the 1856 survey) bundled into the
    # SAME visit's content -- so full 3/3 credit there is legitimate, not an artifact.
    bundled_secondary_source = {
        "url": "https://wikimediafoundation.org/give",
        "content": (
            "Donate to the Wikimedia Foundation... "
            "=== https://www.history.com/articles/who-is-mount-everest-named-after ===\n"
            "The British initially referred to the mountain as Peak XV until Andrew Waugh, the "
            "surveyor general of India, proposed that it be named for his predecessor, Sir George "
            "Everest. Waugh wrote to the Royal Geographical Society in 1856."
        ),
    }
    with_real_source = junk_only[1:] + [bundled_secondary_source, junk_only[-1]]
    result_real = t.validate_chain_coverage(r, _obs(with_real_source, n=12))
    assert result_real["score"] == 1.0
    assert result_real["passed"] is True


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
