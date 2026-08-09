"""
Offline unit tests for the two-constraint numeric AND-filter task (test 076) — free, no LLM.

Cover the keystone gate (Lake Winnipegosis = the unique lake with area > 5,000 km² AND max depth
< 20 m) across its three acceptance paths (terse-only-winner, explicit assertion, enumeration
table); the single-constraint decoys (Lake Athabasca / Lake Okeechobee / Lake Khanka) gating the
keystone to zero; two regressions shared with test_062/test_064's identical assertion-regex
machinery (a wide superlative-to-winner gap that used to exceed the old 55-char window, and a
rival's citation URL that must not pollute the keystone check); and that the compiled plan is a
well-formed twelve-leaf pure-fan-out DAG that leaks no attribute value or winner.
"""
from agent.app.idea_tests import test_076_tier5_numeric_and_filter as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}


def test_terse_only_winner_named_passes():
    assert t.validate_keystone_filter(_r("Lake Winnipegosis."), _OBS)["passed"]


def test_explicit_assertion_passes():
    text = (
        "Of the six lakes, only Lake Winnipegosis satisfies both constraints: area 5,370 km² "
        "(over 5,000) and max depth 12 m (under 20)."
    )
    assert t.validate_keystone_filter(_r(text), _OBS)["passed"]


def test_enumeration_table_passes():
    text = (
        "Lake Winnipegosis: area=5,370 km² (>5,000 km²? yes), max depth=12 m (<20 m? yes)\n"
        "Nettilling Lake: area=5,542 km² (>5,000 km²? yes), max depth=132 m (<20 m? no)\n"
        "Reindeer Lake: area=6,650 km² (>5,000 km²? yes), max depth=219 m (<20 m? no)\n"
        "Lake Athabasca: area=7,849 km² (>5,000 km²? yes), max depth=124 m (<20 m? no)\n"
        "Lake Okeechobee: area=1,900 km² (>5,000 km²? no), max depth=3.7 m (<20 m? yes)\n"
        "Lake Khanka: area=4,070 km² (>5,000 km²? no), max depth=10.6 m (<20 m? yes)\n"
    )
    r = _r(text)
    assert t.validate_keystone_filter(r, _OBS)["passed"]
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_area_only_decoy_gates_to_zero():
    # Chases the biggest lake by area (Athabasca, 7,849 km²) but it's 124 m deep -- drops the
    # depth constraint.
    text = "Lake Athabasca is the answer: it is the largest lake by area among the six."
    assert not t.validate_keystone_filter(_r(text), _OBS)["passed"]
    assert t.validate_winner_attributes(_r(text), _OBS)["score"] == 0.0  # gated on keystone


def test_depth_only_decoy_gates_to_zero():
    # Chases the shallowest lake (Okeechobee, 3.7 m) but it's only 1,900 km² -- drops the area
    # constraint.
    text = "Lake Okeechobee is the answer: it is the shallowest of the six lakes."
    assert not t.validate_keystone_filter(_r(text), _OBS)["passed"]


def test_wide_gap_between_assertion_and_winner_passes():
    # Regression (same bug CLASS confirmed live on test_062's identical regex machinery): a
    # natural "assertion ... is <winner>" sentence can put more than 55 characters between the
    # assertion phrase and the winner's name -- this used to exceed the direction-2 proximity
    # window and wrongly fail an otherwise-correct answer. The window is now 90 (matching
    # direction 1), symmetric with test_062/test_064.
    text = (
        "The only lake among the six that satisfies both constraints simultaneously is Lake "
        "Winnipegosis."
    )
    gap_start = text.lower().index("only") + len("only")
    gap_end = text.lower().index("lake winnipegosis")
    assert 55 < (gap_end - gap_start) <= 90, (gap_end - gap_start)  # sanity: exercises the fix
    assert t.validate_keystone_filter(_r(text), _OBS)["passed"]


def test_citation_url_for_a_rival_does_not_pollute_the_keystone_check():
    # Defensive regression: same bug class as test_069's confirmed indexmundi/slovakia false
    # positive -- a rival's citation URL must not perturb a correct keystone assertion.
    text = (
        "Lake Winnipegosis is the only lake satisfying both constraints (area 5,370 km², depth "
        "12 m). Source for Lake Athabasca: https://en.wikipedia.org/wiki/Lake_Athabasca"
    )
    assert t.validate_keystone_filter(_r(text), _OBS)["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    # Regression: 076's _keystone_ok previously never checked visit.count at all (unlike every
    # sibling reachable-tier task) -- an ungrounded parametric-memory guess of Lake Winnipegosis
    # would have earned full keystone credit. Now gated like its siblings.
    text = "Lake Winnipegosis is the only lake satisfying both constraints."
    ungrounded_obs = {"visit": {"count": 0}}
    assert not t.validate_keystone_filter(_r(text), ungrounded_obs)["passed"]
    assert t.validate_winner_attributes(_r(text), ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(_r(text), ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_filter(_r(text), _OBS)["passed"]  # grounded -> still passes


def test_visit_gate():
    n = t.validate_visits(_r("x"), {"visit": {"count": 3}})
    assert n["score"] == 0.5
    assert not n["passed"]
    assert t.validate_visits(_r("x"), {"visit": {"count": 6}})["passed"]


def test_compiled_plan_validates_and_is_twelve_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 12
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("5,370", "5370", "12 m", "winnipegosis is the", "winnipegosis satisfies"):
        assert leak not in blob, f"plan leaks {leak!r}"
    assert not t._WINNIPEGOSIS_WINS.search(plan["aggregation"])
