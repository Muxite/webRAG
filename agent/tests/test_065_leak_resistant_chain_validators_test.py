"""
Offline unit tests for the URL-free 3-hop chain with a leak-resistant terminus (test 065) — free.

Cover the leak-resistant keystone gate (the birthplace town's infobox elevation, 162 m), the
UN-gated breadth diagnostic (poet + town, retained even when the terminus is wrong), the
keystone-gated citation secondary, the correct answer in both single- and multi-line layout, and
the three adversarial failure modes the task is built to expose:
  * the famous homonym (Hidalgo del Parral, Mexico @ 1,620 m) -> keystone token must reject "1,620"
    AND "1620" while breadth is retained;
  * a near-miss transposed number (168 m) -> keystone fails;
  * a confident from-memory parametric guess (a round 150 m, no visit) -> keystone fails.
Plus the compiled plan is a genuine 3-hop chain DAG (three waves, two edges) that templates the
upstream hops and leaks no poet / town / country / elevation.
"""
from agent.app.idea_tests import test_065_tier5_leak_resistant_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage now checks
# instead of an aggregate visit count -- see idea_test_utils.waypoint_chain_coverage). The poet's
# OWN page already names the birthplace town (that is how hop 2 is meant to work), so visiting it
# alone grounds both waypoints -- matching the task's own hop mechanics.
_EV_POET = {"url": "https://en.wikipedia.org/wiki/Pablo_Neruda",
            "content": "Pablo Neruda was a Chilean poet born in Parral, Chile, awarded the Nobel "
                       "Prize in Literature in 1971."}
_EV_TOWN = {"url": "https://en.wikipedia.org/wiki/Parral,_Chile",
            "content": "Parral is a Chilean town in Linares Province, Maule Region. Elevation: "
                       "162 m (531 ft)."}
# The famous Mexican homonym -- a WRONG page that nonetheless genuinely mentions "Parral" in its
# own content, so a validator matching on content (not just the correct slug) still grounds the
# "town" waypoint from it; only the separate keystone check (exact elevation figure) rejects it.
_EV_WRONG_TOWN = {"url": "https://en.wikipedia.org/wiki/Hidalgo_del_Parral",
                  "content": "Hidalgo del Parral is a city in Chihuahua, Mexico, at an elevation "
                             "of 1,620 m (5,310 ft)."}
_FULL_EVIDENCE = [_EV_POET, _EV_TOWN]


def _obs(visited=None, n=3):
    return {"visit": {"count": n}, "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()


_FULL_SINGLE = (
    "Chain: 'Twenty Love Poems and a Song of Despair' -> Pablo Neruda "
    "(https://en.wikipedia.org/wiki/Pablo_Neruda), born in Parral, Chile "
    "(https://en.wikipedia.org/wiki/Parral,_Chile). Parral's elevation is 162 m (531 ft) "
    "above sea level."
)

_FULL_MULTI = (
    "Hop 1 - Poet:\n"
    "  Pablo Neruda (https://en.wikipedia.org/wiki/Pablo_Neruda), Nobel laureate 1971.\n"
    "Hop 2 - Birthplace town:\n"
    "  Parral, Chile (https://en.wikipedia.org/wiki/Parral,_Chile).\n"
    "Hop 3 - Elevation:\n"
    "  162\n"
    "  metres above sea level.\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b162\b on its own line), the breadth
    # poet/town and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_homonym_wrong_town_gates_to_zero_but_keeps_breadth():
    # The classic slip: the agent resolves the poet + town name but searches the bare name and
    # lands on the famous Mexican homonym (Hidalgo del Parral, 1,620 m). The keystone token must
    # reject "1,620" (comma breaks the token); breadth is retained (poet + a "Parral" both named);
    # the gated citation zeroes out.
    wrong = (
        "Pablo Neruda (https://en.wikipedia.org/wiki/Pablo_Neruda) was born in Parral; the page "
        "for Hidalgo del Parral gives an elevation of 1,620 m (5,310 ft)."
    )
    r = _r(wrong)
    wrong_town_obs = _obs([_EV_POET, _EV_WRONG_TOWN])  # visited the WRONG "Parral" page
    assert t.validate_keystone_elevation(r, wrong_town_obs)["score"] == 0.0     # "1,620" != 162
    assert t.validate_chain_coverage(r, wrong_town_obs)["score"] == 1.0          # poet + town name retained
    assert t.validate_citations(r, wrong_town_obs)["score"] == 0.0              # gated on keystone


def test_keystone_token_rejects_unbounded_homonym():
    # Harden the token directly: the comma-free homonym form "1620 m" must NOT satisfy the keystone
    # (no word boundary between "162" and the trailing "0").
    r = _r("Neruda was born in Parral; its elevation is 1620 m above sea level.")
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0


def test_near_miss_value_fails():
    # A transposed/typo'd elevation (168 m instead of 162 m) must fail the exact-match gate even
    # though the rest of the chain is correct and cited.
    text = _FULL_SINGLE.replace("162 m (531 ft)", "168 m (551 ft)")
    r = _r(text)
    assert not t.validate_keystone_elevation(r, _OBS)["passed"]
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0          # chain still walked
    assert t.validate_citations(r, _OBS)["score"] == 0.0              # gated on keystone


def test_parametric_guess_gates_to_zero():
    # A confident from-memory answer: correct poet + town named, but a plausible ROUND elevation
    # guessed without ever opening the town's page. The keystone must be 0 and citations gated 0,
    # while the un-gated breadth still credits that the poet + town were named.
    guess = (
        "From memory: 'Twenty Love Poems and a Song of Despair' is by Pablo Neruda, who was born "
        "in Parral, Chile. Its elevation is roughly 150 metres above sea level."
    )
    r = _r(guess)
    # Only the poet's page was ever opened -- it already names the birthplace town itself (hop
    # 2's own mechanic), so breadth still credits both waypoints; the town's OWN page (and thus
    # its elevation) was never visited, so the keystone stays 0.
    poet_only_obs = _obs([_EV_POET])
    assert t.validate_keystone_elevation(r, poet_only_obs)["score"] == 0.0
    assert t.validate_citations(r, poet_only_obs)["score"] == 0.0
    assert t.validate_chain_coverage(r, poet_only_obs)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only the poet resolved (hop 1), the birthplace town never reached -> breadth registers
    # exactly one of two intermediate hops.
    text = (
        "The poet is Pablo Neruda (https://en.wikipedia.org/wiki/Pablo_Neruda); I could not "
        "determine the birthplace town or its elevation."
    )
    r = _r(text)
    poet_only_obs = _obs([_EV_POET])
    assert abs(t.validate_chain_coverage(r, poet_only_obs)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_elevation(r, poet_only_obs)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """GROUNDING fix (2026-08-16): a waypoint named in the answer with NO supporting visited-page
    evidence must not be credited, regardless of how many OTHER pages were visited (no more
    aggregate visit-count cap)."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, _obs([_EV_POET]))["score"] - 1.0) < 1e-9  # town revealed on poet's page
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0
    # A real, successful visit to a page unrelated to either waypoint contributes nothing.
    junk = [{"url": "https://www.reddit.com/r/books/comments/xyz/", "content": "unrelated book discussion"}]
    assert t.validate_chain_coverage(r, _obs(junk, n=5))["score"] == 0.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 2}})["score"] - (2 / 3)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of the town's elevation (162 m) must collapse the keystone gate (and everything gated on it)
    to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_elevation(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_elevation(r, ungrounded_obs)["passed"] is False
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    # Coverage requires PER-WAYPOINT visited-page EVIDENCE (2026-08-16 grounding fix): a run with
    # no evidence at all banks 0 coverage credit even though the raw text names the poet and town,
    # closing the "parametric recall" leak.
    assert t.validate_chain_coverage(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_elevation(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_validates_and_is_a_chain_dag():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 2
    # A pure 3-hop chain: one leaf per wave, chained.
    assert struct["waves"] == [["poet"], ["birthplace"], ["elevation"]]
    # plan_structure sorts edges by (dep, child); "birthplace" sorts before "poet".
    assert struct["edges"] == ["birthplace->elevation", "poet->birthplace"]
    assert struct["is_dag_chain"] is True
    # Each dependent leaf templates its upstream hop (real edges, not duplicated text).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{poet}" in by_id["birthplace"]["instruction"]
    assert "{birthplace}" in by_id["elevation"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the GIVEN collection but no poet, town, country, or the elevation.
    for leak in ("neruda", "parral", "chile", "162", "1620", "531", "reyes", "basoalto"):
        assert leak not in blob, f"plan leaks {leak!r}"
