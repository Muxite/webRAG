"""
Offline unit tests for the URL-free 3-hop chain D with a leak-resistant terminus (test 097) — free.

Cover the leak-resistant keystone gate (the birthplace village's infobox elevation, 750 m), the
UN-gated breadth diagnostic (painter + village, retained even when the terminus is wrong), the
keystone-gated citation secondary, the correct answer in both single- and multi-line layout, and
the adversarial failure modes the task is built to expose:
  * the fame-decoy (Zaragoza @ 243 m, the city where Goya trained) -> keystone token must reject
    "243" while breadth is retained;
  * a near-miss transposed number (705 m) -> keystone fails;
  * a confident from-memory parametric guess (a round 700 m, no visit) -> keystone fails.
Plus the compiled plan is a genuine 3-hop chain DAG (three waves, two edges) that templates the
upstream hops, is self-describing, and leaks no painter / village / country / elevation.
"""
from agent.app.idea_tests import test_097_tier5_dependent_chain_d as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "Chain: 'The Third of May 1808' + the Black Paintings -> Francisco Goya "
    "(https://en.wikipedia.org/wiki/Francisco_Goya), born in Fuendetodos, Spain "
    "(https://en.wikipedia.org/wiki/Fuendetodos). Fuendetodos's elevation is 750 m (2,460 ft) "
    "above sea level."
)

_FULL_MULTI = (
    "Hop 1 - Painter:\n"
    "  Francisco Goya (https://en.wikipedia.org/wiki/Francisco_Goya), painter of the Black Paintings.\n"
    "Hop 2 - Birthplace village:\n"
    "  Fuendetodos, Aragon, Spain (https://en.wikipedia.org/wiki/Fuendetodos).\n"
    "Hop 3 - Elevation:\n"
    "  750\n"
    "  metres above sea level.\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b750\b on its own line), the breadth
    # painter/village and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_fame_decoy_wrong_city_gates_to_zero_but_keeps_breadth():
    # The classic slip: the agent resolves the painter + birth village but reports the elevation of
    # Zaragoza (243 m), the city where Goya trained and is strongly associated. The keystone token
    # must reject "243"; breadth is retained (painter + village both named); the citation zeroes out.
    wrong = (
        "Francisco Goya (https://en.wikipedia.org/wiki/Francisco_Goya) was born in Fuendetodos; "
        "the page for Zaragoza, where he trained, gives an elevation of 243 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0     # "243" != 750
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0          # painter + village retained
    assert t.validate_citations(r, _OBS)["score"] == 0.0             # gated on keystone


def test_keystone_token_rejects_embedded_number():
    # Harden the token directly: a larger number that merely contains "750" ("7500", "1750") must
    # NOT satisfy the keystone, and the imperial "2,460" ft form must not either.
    assert t.validate_keystone_elevation(_r("elevation code 7500 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("station 1750 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_elevation(_r("elevation 2,460 ft only"), _OBS)["score"] == 0.0


def test_near_miss_value_fails():
    # A transposed/typo'd elevation (705 m instead of 750 m) must fail the exact-match gate even
    # though the rest of the chain is correct and cited.
    text = _FULL_SINGLE.replace("750 m (2,460 ft)", "705 m (2,313 ft)")
    r = _r(text)
    assert not t.validate_keystone_elevation(r, _OBS)["passed"]
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0          # chain still walked
    assert t.validate_citations(r, _OBS)["score"] == 0.0             # gated on keystone


def test_parametric_guess_gates_to_zero():
    # A confident from-memory answer: correct painter + village named, but a plausible ROUND
    # elevation guessed without ever opening the village's page. The keystone must be 0 and citations
    # gated 0, while the un-gated breadth still credits that the painter + village were named.
    guess = (
        "From memory: the painter of the Black Paintings is Francisco Goya, who was born in "
        "Fuendetodos, Spain. Its elevation is roughly 700 metres above sea level."
    )
    r = _r(guess)
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only the painter resolved (hop 1), the birthplace village never reached -> breadth registers
    # exactly one of two intermediate hops.
    text = (
        "The painter is Francisco Goya (https://en.wikipedia.org/wiki/Francisco_Goya); "
        "I could not determine the birthplace village or its elevation."
    )
    r = _r(text)
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_elevation(r, _OBS)["score"] == 0.0


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
    assert struct["waves"] == [["painter"], ["birthplace"], ["elevation"]]
    # plan_structure sorts edges by (dep, child); "birthplace" sorts before "painter".
    assert struct["edges"] == ["birthplace->elevation", "painter->birthplace"]
    assert struct["is_dag_chain"] is True
    # Each dependent leaf templates its upstream hop (real edges, not duplicated text).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{painter}" in by_id["birthplace"]["instruction"]
    assert "{birthplace}" in by_id["elevation"]["instruction"]


def test_compiled_plan_leaves_are_self_describing():
    # Each leaf must restate its own hop-subject in its instruction/expect so the aggregator can
    # bind the entity after leaf ids are stripped (confirmed necessary for premium aggregators).
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "painter" in by_id["birthplace"]["instruction"].lower()
    assert "village" in by_id["elevation"]["expect"].lower()
    assert "elevation" in by_id["elevation"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the GIVEN works but no painter, village, country, or the elevation.
    for leak in ("goya", "francisco", "fuendetodos", "aragon", "spain", "zaragoza", "madrid",
                 "750", "243", "657", "2460"):
        assert leak not in blob, f"plan leaks {leak!r}"
