"""
Offline unit tests for the URL-free 3-hop chain B with a leak-resistant terminus (test 092) — free.

Cover the leak-resistant keystone gate (the birthplace town's infobox elevation, 564 m), the
UN-gated breadth diagnostic (person + town, retained even when the terminus is wrong), the
keystone-gated citation secondary, the correct answer in both single- and multi-line layout, and
the adversarial failure modes the task is built to expose:
  * the famous homonym (Trujillo, Peru @ 34 m — named after the Spanish hometown) -> keystone
    token must reject "34" while breadth is retained;
  * a near-miss transposed number (546 m) -> keystone fails;
  * a confident from-memory parametric guess (a round 500 m, no visit) -> keystone fails.
Plus the compiled plan is a genuine 3-hop chain DAG (three waves, two edges) that templates the
upstream hops, is self-describing, and leaks no person / town / country / elevation.
"""
from agent.app.idea_tests import test_092_tier5_dependent_chain_b as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Per-waypoint visited-page evidence (the grounding channel validate_chain_coverage now checks
# instead of having no grounding at all -- see idea_test_utils.waypoint_chain_coverage). The
# person's OWN page already names the birthplace town (that is how hop 2 is meant to work), so
# visiting it alone grounds both waypoints -- matching the task's own hop mechanics.
_EV_PERSON = {"url": "https://en.wikipedia.org/wiki/Francisco_Pizarro",
              "content": "Francisco Pizarro was a Spanish conquistador who led the conquest of the "
                         "Inca Empire; he was born in Trujillo, Spain."}
_EV_TOWN = {"url": "https://en.wikipedia.org/wiki/Trujillo,_Spain",
            "content": "Trujillo is a town in Extremadura, Province of Cáceres, Spain. Elevation: "
                       "564 m (1,850 ft)."}
# The famous Peruvian homonym -- a WRONG page that nonetheless genuinely mentions "Trujillo" in
# its own content, so a validator matching on content (not just the correct slug) still grounds
# the "town" waypoint from it; only the separate keystone check (exact elevation) rejects it.
_EV_WRONG_TOWN = {"url": "https://en.wikipedia.org/wiki/Trujillo,_Peru",
                  "content": "Trujillo is a city in northwestern Peru, named after the Spanish "
                             "hometown of Francisco Pizarro, at an elevation of 34 m (112 ft)."}
_FULL_EVIDENCE = [_EV_PERSON, _EV_TOWN]


def _obs(visited=None, n=3):
    return {"visit": {"count": n}, "evidence": {"visited": _FULL_EVIDENCE if visited is None else visited}}


_OBS = _obs()


_FULL_SINGLE = (
    "Chain: conquest of the Inca Empire -> Francisco Pizarro "
    "(https://en.wikipedia.org/wiki/Francisco_Pizarro), born in Trujillo, Spain "
    "(https://en.wikipedia.org/wiki/Trujillo,_Spain). Trujillo's elevation is 564 m (1,850 ft) "
    "above sea level."
)

_FULL_MULTI = (
    "Hop 1 - Conquistador:\n"
    "  Francisco Pizarro (https://en.wikipedia.org/wiki/Francisco_Pizarro), conquered the Inca.\n"
    "Hop 2 - Birthplace town:\n"
    "  Trujillo, Spain (https://en.wikipedia.org/wiki/Trujillo,_Spain), Extremadura.\n"
    "Hop 3 - Elevation:\n"
    "  564\n"
    "  metres above sea level.\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b564\b on its own line), the breadth
    # person/town and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_homonym_wrong_town_gates_to_zero_but_keeps_breadth():
    # The classic slip: the agent resolves the person + town name but searches the bare name and
    # lands on the famous Peruvian homonym (Trujillo, Peru, 34 m — itself named after the Spanish
    # hometown). The keystone token must reject "34"; breadth is retained (person + a "Trujillo"
    # both named); the gated citation zeroes out.
    wrong = (
        "Francisco Pizarro (https://en.wikipedia.org/wiki/Francisco_Pizarro) was born in Trujillo; "
        "the page for Trujillo, Peru gives an elevation of 34 m (112 ft)."
    )
    r = _r(wrong)
    wrong_town_obs = _obs([_EV_PERSON, _EV_WRONG_TOWN])  # visited the WRONG "Trujillo" page
    assert t.validate_keystone_elevation(r, wrong_town_obs)["score"] == 0.0     # "34" != 564
    assert t.validate_chain_coverage(r, wrong_town_obs)["score"] == 1.0          # person + town name retained
    assert t.validate_citations(r, wrong_town_obs)["score"] == 0.0             # gated on keystone


def test_keystone_token_rejects_embedded_number():
    # Harden the token directly: a larger number that merely contains "564" ("5643", "1564") must
    # NOT satisfy the keystone, and the imperial "1,850" ft form must not either.
    assert t.validate_keystone_elevation(_r("elevation code 5643 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("station 1564 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("elevation 1,850 ft only"), _OBS)["score"] == 0.0


def test_near_miss_value_fails():
    # A transposed/typo'd elevation (546 m instead of 564 m) must fail the exact-match gate even
    # though the rest of the chain is correct and cited.
    text = _FULL_SINGLE.replace("564 m (1,850 ft)", "546 m (1,791 ft)")
    r = _r(text)
    assert not t.validate_keystone_elevation(r, _OBS)["passed"]
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0          # chain still walked
    assert t.validate_citations(r, _OBS)["score"] == 0.0             # gated on keystone


def test_parametric_guess_gates_to_zero():
    # A confident from-memory answer: correct person + town named, but a plausible ROUND elevation
    # guessed without ever opening the town's page. The keystone must be 0 and citations gated 0,
    # while the un-gated breadth still credits that the person + town were named.
    guess = (
        "From memory: the conquistador of the Inca Empire is Francisco Pizarro, who was born in "
        "Trujillo, Spain. Its elevation is roughly 500 metres above sea level."
    )
    r = _r(guess)
    # Only the person's page was ever opened -- it already names the birthplace town itself (hop
    # 2's own mechanic), so breadth still credits both waypoints; the town's OWN page (and thus
    # its elevation) was never visited, so the keystone stays 0.
    person_only_obs = _obs([_EV_PERSON])
    assert t.validate_keystone_elevation(r, person_only_obs)["score"] == 0.0
    assert t.validate_citations(r, person_only_obs)["score"] == 0.0
    assert t.validate_chain_coverage(r, person_only_obs)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only the person resolved (hop 1), the birthplace town never reached -> breadth registers
    # exactly one of two intermediate hops.
    text = (
        "The conquistador is Francisco Pizarro (https://en.wikipedia.org/wiki/Francisco_Pizarro); "
        "I could not determine the birthplace town or its elevation."
    )
    r = _r(text)
    person_only_obs = _obs([_EV_PERSON])
    assert abs(t.validate_chain_coverage(r, person_only_obs)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_elevation(r, person_only_obs)["score"] == 0.0


def test_chain_coverage_requires_page_evidence_not_just_text():
    """GROUNDING fix (2026-08-16): a waypoint named in the answer with NO supporting visited-page
    evidence must not be credited -- this task's chain_coverage had NO grounding requirement at
    all before this fix."""
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, _obs([]))["score"] == 0.0
    assert t.validate_chain_coverage(r, _obs([_EV_PERSON]))["score"] == 1.0  # town revealed on person's page
    assert t.validate_chain_coverage(r, _obs(_FULL_EVIDENCE))["score"] == 1.0
    junk = [{"url": "https://www.reddit.com/r/history/comments/xyz/", "content": "unrelated discussion"}]
    assert t.validate_chain_coverage(r, _obs(junk, n=5))["score"] == 0.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 2}})["score"] - (2 / 3)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]


def test_compiled_plan_validates_and_is_a_chain_dag():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 2
    # A pure 3-hop chain: one leaf per wave, chained.
    assert struct["waves"] == [["conquistador"], ["birthplace"], ["elevation"]]
    # plan_structure sorts edges by (dep, child); "birthplace" sorts before "conquistador".
    assert struct["edges"] == ["birthplace->elevation", "conquistador->birthplace"]
    assert struct["is_dag_chain"] is True
    # Each dependent leaf templates its upstream hop (real edges, not duplicated text).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{conquistador}" in by_id["birthplace"]["instruction"]
    assert "{birthplace}" in by_id["elevation"]["instruction"]


def test_compiled_plan_leaves_are_self_describing():
    # Each leaf must restate its own hop-subject in its instruction/expect so the aggregator can
    # bind the entity after leaf ids are stripped (confirmed necessary for premium aggregators).
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "conquistador" in by_id["birthplace"]["instruction"].lower()
    assert "town" in by_id["elevation"]["expect"].lower()
    assert "elevation" in by_id["elevation"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the GIVEN deed but no person, town, country, or the elevation.
    for leak in ("pizarro", "trujillo", "spain", "castile", "extremadura", "peru",
                 "564", "34", "1850", "caceres", "cáceres"):
        assert leak not in blob, f"plan leaks {leak!r}"
