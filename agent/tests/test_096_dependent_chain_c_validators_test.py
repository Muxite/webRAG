"""
Offline unit tests for the URL-free 3-hop chain C with a leak-resistant terminus (test 096) — free.

Cover the leak-resistant keystone gate (the birthplace town's infobox elevation, 265 m), the
UN-gated breadth diagnostic (aviator + town, retained even when the terminus is wrong), the
keystone-gated citation secondary, the correct answer in both single- and multi-line layout, and
the adversarial failure modes the task is built to expose:
  * the fame-decoy (a larger nearby city, Kansas City @ 276 m) -> keystone token must reject "276"
    while breadth is retained;
  * a near-miss transposed number (256 m) -> keystone fails;
  * a confident from-memory parametric guess (a round 300 m, no visit) -> keystone fails.
Plus the compiled plan is a genuine 3-hop chain DAG (three waves, two edges) that templates the
upstream hops, is self-describing, and leaks no aviator / town / state / elevation.
"""
from agent.app.idea_tests import test_096_tier5_dependent_chain_c as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage now checks
# instead of having no grounding at all -- see idea_test_utils.waypoint_chain_coverage). The
# aviator's OWN page already names the birthplace town (that is how hop 2 is meant to work), so
# visiting it alone grounds both waypoints -- matching the task's own hop mechanics.
_EV_AVIATOR = {"url": "https://en.wikipedia.org/wiki/Amelia_Earhart",
               "content": "Amelia Earhart was an American aviation pioneer, born in Atchison, "
                          "Kansas."}
_EV_TOWN = {"url": "https://en.wikipedia.org/wiki/Atchison,_Kansas",
            "content": "Atchison is a city in Atchison County, Kansas. Elevation: 869 ft (265 m)."}
# A fame-decoy -- a larger nearby city page that mentions neither the aviator nor the town, so it
# cannot ground either waypoint on its own.
_EV_KANSAS_CITY = {"url": "https://en.wikipedia.org/wiki/Kansas_City,_Missouri",
                   "content": "Kansas City is a major city in Missouri at an elevation of about "
                              "276 m."}
_FULL_EVIDENCE = [_EV_AVIATOR, _EV_TOWN]


def _obs(visited=None, n=3):
    return {"visit": {"count": n}, "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()


_FULL_SINGLE = (
    "Chain: first solo nonstop transatlantic flight by a woman -> Amelia Earhart "
    "(https://en.wikipedia.org/wiki/Amelia_Earhart), born in Atchison, Kansas "
    "(https://en.wikipedia.org/wiki/Atchison,_Kansas). Atchison's elevation is 869 ft (265 m) "
    "above sea level."
)

_FULL_MULTI = (
    "Hop 1 - Aviator:\n"
    "  Amelia Earhart (https://en.wikipedia.org/wiki/Amelia_Earhart), first woman to fly the Atlantic solo.\n"
    "Hop 2 - Birthplace town:\n"
    "  Atchison, Kansas (https://en.wikipedia.org/wiki/Atchison,_Kansas), U.S.\n"
    "Hop 3 - Elevation:\n"
    "  265\n"
    "  metres above sea level.\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b265\b on its own line), the breadth
    # aviator/town and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_fame_decoy_wrong_city_gates_to_zero_but_keeps_breadth():
    # The classic slip: the agent resolves the aviator + birth town but reports the elevation of a
    # larger nearby city (Kansas City, MO @ 276 m). The keystone token must reject "276"; breadth is
    # retained (aviator + town both named); the gated citation zeroes out.
    wrong = (
        "Amelia Earhart (https://en.wikipedia.org/wiki/Amelia_Earhart) was born in Atchison; "
        "nearby Kansas City sits at an elevation of about 276 m above sea level."
    )
    r = _r(wrong)
    decoy_obs = _obs([_EV_AVIATOR, _EV_KANSAS_CITY])  # visited the decoy city, not Atchison itself
    assert t.validate_keystone_elevation(r, decoy_obs)["score"] == 0.0     # "276" != 265
    assert t.validate_chain_coverage(r, decoy_obs)["score"] == 1.0          # aviator + town retained
    assert t.validate_citations(r, decoy_obs)["score"] == 0.0             # gated on keystone


def test_keystone_token_rejects_embedded_number():
    # Harden the token directly: a larger number that merely contains "265" ("2650", "1265") must
    # NOT satisfy the keystone, and the imperial "869" ft form must not either.
    assert t.validate_keystone_elevation(_r("elevation code 2650 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("station 1265 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("elevation 869 ft only"), _OBS)["score"] == 0.0


def test_near_miss_value_fails():
    # A transposed/typo'd elevation (256 m instead of 265 m) must fail the exact-match gate even
    # though the rest of the chain is correct and cited.
    text = _FULL_SINGLE.replace("869 ft (265 m)", "840 ft (256 m)")
    r = _r(text)
    assert not t.validate_keystone_elevation(r, _OBS)["passed"]
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0          # chain still walked
    assert t.validate_citations(r, _OBS)["score"] == 0.0             # gated on keystone


def test_parametric_guess_gates_to_zero():
    # A confident from-memory answer: correct aviator + town named, but a plausible ROUND elevation
    # guessed without ever opening the town's page. The keystone must be 0 and citations gated 0,
    # while the un-gated breadth still credits that the aviator + town were named.
    guess = (
        "From memory: the first woman to fly the Atlantic solo is Amelia Earhart, who was born in "
        "Atchison, Kansas. Its elevation is roughly 300 metres above sea level."
    )
    r = _r(guess)
    # Only the aviator's page was ever opened -- it already names the birthplace town itself (hop
    # 2's own mechanic), so breadth still credits both waypoints; the town's OWN page (and thus
    # its elevation) was never visited, so the keystone stays 0.
    aviator_only_obs = _obs([_EV_AVIATOR])
    assert t.validate_keystone_elevation(r, aviator_only_obs)["score"] == 0.0
    assert t.validate_citations(r, aviator_only_obs)["score"] == 0.0
    assert t.validate_chain_coverage(r, aviator_only_obs)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only the aviator resolved (hop 1), the birthplace town never reached -> breadth registers
    # exactly one of two intermediate hops.
    text = (
        "The aviator is Amelia Earhart (https://en.wikipedia.org/wiki/Amelia_Earhart); "
        "I could not determine the birthplace town or its elevation."
    )
    r = _r(text)
    aviator_only_obs = _obs([_EV_AVIATOR])
    assert abs(t.validate_chain_coverage(r, aviator_only_obs)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_elevation(r, aviator_only_obs)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """GROUNDING fix (2026-08-16): a waypoint named in the answer with NO supporting visited-page
    evidence must not be credited -- this task's chain_coverage had NO grounding requirement at
    all before this fix."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    assert t.validate_chain_coverage(r, _obs([_EV_AVIATOR]))["score"] == 1.0  # town revealed on aviator's page
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0
    junk = [{"url": "https://www.tiktok.com/@someone/video/999", "content": "unrelated video caption"}]
    assert t.validate_chain_coverage(r, _obs(junk, n=5))["score"] == 0.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 2}})["score"] - (2 / 3)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_elevation(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_elevation(r, ungrounded_obs)["passed"] is False
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_elevation(r, ungrounded_obs)["score"],
        t.validate_chain_coverage(r, ungrounded_obs)["score"],
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
    assert struct["waves"] == [["aviator"], ["birthplace"], ["elevation"]]
    # plan_structure sorts edges by (dep, child); "aviator" sorts before "birthplace".
    assert struct["edges"] == ["aviator->birthplace", "birthplace->elevation"]
    assert struct["is_dag_chain"] is True
    # Each dependent leaf templates its upstream hop (real edges, not duplicated text).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{aviator}" in by_id["birthplace"]["instruction"]
    assert "{birthplace}" in by_id["elevation"]["instruction"]


def test_compiled_plan_leaves_are_self_describing():
    # Each leaf must restate its own hop-subject in its instruction/expect so the aggregator can
    # bind the entity after leaf ids are stripped (confirmed necessary for premium aggregators).
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "aviator" in by_id["birthplace"]["instruction"].lower()
    assert "town" in by_id["elevation"]["expect"].lower()
    assert "elevation" in by_id["elevation"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the GIVEN deed but no aviator, town, state, or the elevation.
    for leak in ("earhart", "amelia", "atchison", "kansas", "265", "869", "276"):
        assert leak not in blob, f"plan leaks {leak!r}"
