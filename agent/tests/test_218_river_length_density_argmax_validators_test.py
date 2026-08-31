"""
Offline unit tests for web-grounded task 218 (river channel-length-density argmax) — free, no
LLM, no network.

Jobs:
  1. GROUND-TRUTH RE-DERIVATION with a second, independently implemented computation (integer
     cross-multiplication -- never division) that must agree with the module's table.
  2. IMPORT-TIME TRAP ASSERTION: the raw-value winner (longest river / largest basin) must differ
     from the derived-value winner (verified again here, redundantly, on top of the module's own
     assertions).
  3. ADVERSARIAL VALIDATOR HARDENING: full answer (single- and multi-line), wrong-keystone (the
     raw-argmax decoy), partial coverage, no-visits, leak checks.
"""
from fractions import Fraction

from agent.app.idea_tests import test_218_tier5_river_length_density_argmax as t


def _independent_ranking():
    items = [(e["name"], e["length_km"] * 1000, e["basin_km2"]) for e in t.ENTITIES]
    ranked = []
    for it in items:
        pos = 0
        while pos < len(ranked) and ranked[pos][1] * it[2] > it[1] * ranked[pos][2]:
            pos += 1
        ranked.insert(pos, it)
    values = {name: float(Fraction(length_m, basin)) for name, length_m, basin in items}
    return [r[0] for r in ranked], values


def test_ground_truth_matches_independent_cross_multiplication_solver():
    order, values = _independent_ranking()
    assert order[0] == "Mekong"
    assert order[1] == "Yangtze"
    module_values = {e["name"]: e["density"] for e in t.ENTITIES}
    for name, v in values.items():
        assert abs(module_values[name] - v) < 1e-9


def test_raw_winner_differs_from_derived_winner():
    by_length = max(t.ENTITIES, key=lambda e: e["length_km"])
    by_basin = max(t.ENTITIES, key=lambda e: e["basin_km2"])
    assert by_length["name"] != t.WINNER["name"]
    assert by_basin["name"] != t.WINNER["name"]
    assert by_length["name"] == "Nile"
    assert by_basin["name"] == "Amazon"


def test_keystone_margin_wide():
    assert t._MARGIN > 0.5


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=5):
    return {"visit": {"count": n}}


FULL_ANSWER_ONELINE = (
    "Mekong has the highest channel-length density at 5.47 m/km^2, higher than any other river. "
    "Yangtze: length 6,300 km, basin 1,808,500 km^2. Nile: length 7,088 km, basin 2,927,843 km^2. "
    "Mississippi: length 3,766 km, basin 2,980,000 km^2. Amazon: length 6,575 km, basin 6,925,674 "
    "km^2. Mekong: length 4,350 km, basin 795,000 km^2. "
    "Sources: https://en.wikipedia.org/wiki/Mekong https://en.wikipedia.org/wiki/Yangtze "
    "https://en.wikipedia.org/wiki/Nile https://en.wikipedia.org/wiki/Mississippi_River "
    "https://en.wikipedia.org/wiki/Amazon_River"
)

FULL_ANSWER_MULTILINE = """
Highest channel-length density:
Mekong
5.47 m/km^2, the highest of the five.

All rivers:
Mekong: length 4,350 km, basin 795,000 km^2
Yangtze: length 6,300 km, basin 1,808,500 km^2
Nile: length 7,088 km, basin 2,927,843 km^2
Mississippi: length 3,766 km, basin 2,980,000 km^2
Amazon: length 6,575 km, basin 6,925,674 km^2

Sources:
https://en.wikipedia.org/wiki/Mekong
https://en.wikipedia.org/wiki/Yangtze
https://en.wikipedia.org/wiki/Nile
https://en.wikipedia.org/wiki/Mississippi_River
https://en.wikipedia.org/wiki/Amazon_River
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
    # Names Nile (longest river, the salient decoy) as the winner with a full data table.
    text = (
        "Nile has the highest channel-length density, since it is the longest river. "
        "Mekong: length 4,350 km, basin 795,000 km^2. Yangtze: length 6,300 km, basin 1,808,500 "
        "km^2. Nile: length 7,088 km, basin 2,927,843 km^2. Mississippi: length 3,766 km, basin "
        "2,980,000 km^2. Amazon: length 6,575 km, basin 6,925,674 km^2. "
        "https://en.wikipedia.org/wiki/Nile https://en.wikipedia.org/wiki/Amazon_River "
        "https://en.wikipedia.org/wiki/Mekong https://en.wikipedia.org/wiki/Yangtze "
        "https://en.wikipedia.org/wiki/Mississippi_River"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_argmax"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0          # coverage un-gated, all 10 figures present
    assert results["winner_density"]["score"] == 0.0    # gated -> 0
    assert results["citation"]["score"] == 0.0           # gated -> 0


def test_partial_coverage_is_exact_fraction():
    text = (
        "Mekong has the highest channel-length density at 5.47 m/km^2. "
        "Mekong: length 4,350 km, basin 795,000 km^2. Yangtze: length 6,300 km, basin 1,808,500 "
        "km^2. https://en.wikipedia.org/wiki/Mekong"
    )
    r, o = _r(text), _obs()
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 2 / 5


def test_no_visits_zeroes_keystone():
    r, o = _r(FULL_ANSWER_ONELINE), _obs(0)
    ks = t.validate_keystone_argmax(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0  # coverage is un-gated even with zero visits


def test_statement_leak_raw_inputs_present_derived_absent():
    statement = t.get_task_statement()
    # every raw input (the given river names) is legitimately in the statement
    for e in t.ENTITIES:
        assert e["name"] in statement
    # the derived density value (the winner's ratio) must NOT be leaked anywhere in the statement
    assert f"{t.WINNER_DENSITY:.2f}" not in statement
    assert "5.47" not in statement
    # no raw length/basin figure is missing (they are legitimately absent -- NO URLs, no figures
    # are given at all in this task's statement; only the entity names are)
    for e in t.ENTITIES:
        assert str(e["length_km"]) not in statement
        assert str(e["basin_km2"]) not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 5
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    blob_lower = blob.lower()
    for e in t.ENTITIES:
        assert str(e["length_km"]) not in blob
        assert str(e["basin_km2"]) not in blob
    assert "mekong" in blob_lower  # naming the given rivers is fine
    assert f"{t.WINNER_DENSITY:.2f}" not in blob
    # aggregation must not declare the winner
    agg_lower = plan["aggregation"].lower()
    assert "mekong has the highest" not in agg_lower
