"""
Offline unit tests for web-grounded task 221 (skyscraper avg-floor-height argmax) — free, no LLM,
no network. Same structure as siblings 218/219/220, plus an extra check for the tightened +/-1%
tolerance band (this task's margin, 8.1%, is thinner than its siblings').
"""
from fractions import Fraction

from agent.app.idea_tests import test_221_tier5_skyscraper_floor_height_argmax as t


def _independent_ranking():
    # heights carry one decimal place; scale by 10 to keep the cross-multiplication solver
    # exact-integer (never touches floating-point division).
    items = [(e["name"], round(e["height_m"] * 10), e["floors"]) for e in t.ENTITIES]
    ranked = []
    for it in items:
        pos = 0
        while pos < len(ranked) and ranked[pos][1] * it[2] > it[1] * ranked[pos][2]:
            pos += 1
        ranked.insert(pos, it)
    values = {name: float(Fraction(h10, f)) / 10 for name, h10, f in items}
    return [r[0] for r in ranked], values


def test_ground_truth_matches_independent_cross_multiplication_solver():
    order, values = _independent_ranking()
    assert order[0] == "One World Trade Center"
    assert order[1] == "Burj Khalifa"
    module_values = {e["name"]: e["ratio"] for e in t.ENTITIES}
    for name, v in values.items():
        assert abs(module_values[name] - v) < 1e-6


def test_raw_winner_differs_from_derived_winner():
    by_height = max(t.ENTITIES, key=lambda e: e["height_m"])
    assert by_height["name"] != t.WINNER["name"]
    assert by_height["name"] == "Burj Khalifa"


def test_keystone_margin_and_tolerance_are_disjoint():
    assert 0.06 <= t._MARGIN <= 0.15
    assert t._MARGIN > 8 * t.RATIO_TOL


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=5):
    return {"visit": {"count": n}}


FULL_ANSWER_ONELINE = (
    "One World Trade Center has the highest average floor height at 5.81 m/floor, higher than "
    "any of the others. One World Trade Center: height 546.2 m, 94 floors. Burj Khalifa: height "
    "828 m, 154 floors. Taipei 101: height 509.2 m, 101 floors. Shanghai Tower: height 632 m, "
    "128 floors. Willis Tower: height 442 m, 110 floors. "
    "Sources: https://en.wikipedia.org/wiki/One_World_Trade_Center "
    "https://en.wikipedia.org/wiki/Burj_Khalifa https://en.wikipedia.org/wiki/Taipei_101 "
    "https://en.wikipedia.org/wiki/Shanghai_Tower https://en.wikipedia.org/wiki/Willis_Tower"
)

FULL_ANSWER_MULTILINE = """
Highest average floor height:
One World Trade Center
5.81 m/floor, the highest of the five.

All buildings:
One World Trade Center: height 546.2 m, 94 floors
Burj Khalifa: height 828 m, 154 floors
Taipei 101: height 509.2 m, 101 floors
Shanghai Tower: height 632 m, 128 floors
Willis Tower: height 442 m, 110 floors

Sources:
https://en.wikipedia.org/wiki/One_World_Trade_Center
https://en.wikipedia.org/wiki/Burj_Khalifa
https://en.wikipedia.org/wiki/Taipei_101
https://en.wikipedia.org/wiki/Shanghai_Tower
https://en.wikipedia.org/wiki/Willis_Tower
"""


def test_full_answer_oneline_scores_all_checks_at_1():
    r, o = _r(FULL_ANSWER_ONELINE), _obs()
    for fn in t.get_validation_functions():
        res = fn(r, o)
        assert res["score"] == 1.0, (res["check"], res["reason"])


def test_full_answer_multiline_scores_all_checks_at_1():
    r, o = _r(FULL_ANSWER_MULTILINE), _obs()
    for fn in t.get_validation_functions():
        res = fn(r, o)
        assert res["score"] == 1.0, (res["check"], res["reason"])


def test_wrong_keystone_raw_winner_named_instead():
    # Names Burj Khalifa (tallest, the salient decoy) as the winner, with full data.
    text = (
        "Burj Khalifa has the highest average floor height, since it is the world's tallest "
        "building. One World Trade Center: height 546.2 m, 94 floors. Burj Khalifa: height 828 "
        "m, 154 floors. Taipei 101: height 509.2 m, 101 floors. Shanghai Tower: height 632 m, "
        "128 floors. Willis Tower: height 442 m, 110 floors. "
        "https://en.wikipedia.org/wiki/Burj_Khalifa https://en.wikipedia.org/wiki/One_World_Trade_Center "
        "https://en.wikipedia.org/wiki/Taipei_101 https://en.wikipedia.org/wiki/Shanghai_Tower "
        "https://en.wikipedia.org/wiki/Willis_Tower"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_argmax"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0
    assert results["winner_ratio"]["score"] == 0.0
    assert results["citation"]["score"] == 0.0


def test_partial_coverage_is_exact_fraction():
    text = (
        "One World Trade Center has the highest average floor height at 5.81 m/floor. "
        "One World Trade Center: height 546.2 m, 94 floors. Burj Khalifa: height 828 m, 154 "
        "floors. https://en.wikipedia.org/wiki/One_World_Trade_Center"
    )
    r, o = _r(text), _obs()
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 2 / 5


def test_no_visits_zeroes_keystone():
    r, o = _r(FULL_ANSWER_ONELINE), _obs(0)
    ks = t.validate_keystone_argmax(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0


def test_statement_leak_raw_inputs_present_derived_absent():
    statement = t.get_task_statement()
    for e in t.ENTITIES:
        assert e["name"] in statement
        assert str(e["height_m"]) not in statement
        # "Taipei 101" the building's PROPER NAME literally contains its own floor count (101),
        # by construction (it is named for its floor count) -- this is not a leak, it's the given.
        if e["floors"] >= 10 and e["key"] != "taipei_101":
            assert str(e["floors"]) not in statement
    assert "5.81" not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 5
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    for e in t.ENTITIES:
        assert str(e["height_m"]) not in blob
        if e["key"] != "taipei_101":
            assert str(e["floors"]) not in blob
    assert "one world trade center has the highest" not in plan["aggregation"].lower()
