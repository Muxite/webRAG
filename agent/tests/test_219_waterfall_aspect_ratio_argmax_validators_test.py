"""
Offline unit tests for web-grounded task 219 (waterfall aspect-ratio argmax) — free, no LLM, no
network. Same structure as the 218 sibling: independent cross-multiplication re-derivation,
raw-winner-vs-derived-winner trap check, adversarial validator hardening, leak checks.
"""
from fractions import Fraction

from agent.app.idea_tests import test_219_tier5_waterfall_aspect_ratio_argmax as t


def _independent_ranking():
    items = [(e["name"], e["height_m"], e["width_m"]) for e in t.ENTITIES]
    ranked = []
    for it in items:
        pos = 0
        while pos < len(ranked) and ranked[pos][1] * it[2] > it[1] * ranked[pos][2]:
            pos += 1
        ranked.insert(pos, it)
    values = {name: float(Fraction(h, w)) for name, h, w in items}
    return [r[0] for r in ranked], values


def test_ground_truth_matches_independent_cross_multiplication_solver():
    order, values = _independent_ranking()
    assert order[0] == "Multnomah Falls"
    assert order[1] == "Kaieteur Falls"
    module_values = {e["name"]: e["ratio"] for e in t.ENTITIES}
    for name, v in values.items():
        assert abs(module_values[name] - v) < 1e-9


def test_raw_winner_differs_from_derived_winner():
    by_height = max(t.ENTITIES, key=lambda e: e["height_m"])
    by_width = max(t.ENTITIES, key=lambda e: e["width_m"])
    assert by_height["name"] != t.WINNER["name"]
    assert by_width["name"] != t.WINNER["name"]
    assert by_height["name"] == "Kaieteur Falls"
    assert by_width["name"] == "Victoria Falls"


def test_keystone_margin_wide():
    assert t._MARGIN > 10.0  # 3050% relative


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=5):
    return {"visit": {"count": n}}


FULL_ANSWER_ONELINE = (
    "Multnomah Falls has the highest aspect ratio at 63.0, higher than any of the others. "
    "Multnomah Falls: height 189 m, width 3 m. Kaieteur Falls: height 226 m, width 113 m. "
    "Dettifoss: height 44 m, width 100 m. Niagara Horseshoe Falls: height 57 m, width 790 m. "
    "Victoria Falls: height 108 m, width 1,708 m. "
    "Sources: https://en.wikipedia.org/wiki/Multnomah_Falls "
    "https://en.wikipedia.org/wiki/Kaieteur_Falls https://en.wikipedia.org/wiki/Dettifoss "
    "https://en.wikipedia.org/wiki/Niagara_Falls https://en.wikipedia.org/wiki/Victoria_Falls"
)

FULL_ANSWER_MULTILINE = """
Highest aspect ratio:
Multnomah Falls
63.0, the highest of the five.

All waterfalls:
Multnomah Falls: height 189 m, width 3 m
Kaieteur Falls: height 226 m, width 113 m
Dettifoss: height 44 m, width 100 m
Niagara Horseshoe Falls: height 57 m, width 790 m
Victoria Falls: height 108 m, width 1,708 m

Sources:
https://en.wikipedia.org/wiki/Multnomah_Falls
https://en.wikipedia.org/wiki/Kaieteur_Falls
https://en.wikipedia.org/wiki/Dettifoss
https://en.wikipedia.org/wiki/Niagara_Falls
https://en.wikipedia.org/wiki/Victoria_Falls
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
    # Names Kaieteur Falls (the tallest, the salient decoy) as the winner, with full data.
    text = (
        "Kaieteur Falls has the highest aspect ratio, since it is the tallest of the five. "
        "Multnomah Falls: height 189 m, width 3 m. Kaieteur Falls: height 226 m, width 113 m. "
        "Dettifoss: height 44 m, width 100 m. Niagara Horseshoe Falls: height 57 m, width 790 m. "
        "Victoria Falls: height 108 m, width 1,708 m. "
        "https://en.wikipedia.org/wiki/Kaieteur_Falls https://en.wikipedia.org/wiki/Multnomah_Falls "
        "https://en.wikipedia.org/wiki/Dettifoss https://en.wikipedia.org/wiki/Niagara_Falls "
        "https://en.wikipedia.org/wiki/Victoria_Falls"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_argmax"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0
    assert results["winner_ratio"]["score"] == 0.0
    assert results["citation"]["score"] == 0.0


def test_partial_coverage_is_exact_fraction():
    text = (
        "Multnomah Falls has the highest aspect ratio at 63.0. "
        "Multnomah Falls: height 189 m, width 3 m. Kaieteur Falls: height 226 m, width 113 m. "
        "https://en.wikipedia.org/wiki/Multnomah_Falls"
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
        # multi-digit raw figures are unambiguous leak checks (width=3 is too short/ambiguous,
        # since it trivially collides with the "1./2./3." entity-list numbering)
        if e["height_m"] >= 10:
            assert str(e["height_m"]) not in statement
        if e["width_m"] >= 10:
            assert str(e["width_m"]) not in statement
    assert f"{t.WINNER_RATIO:.1f}" not in statement
    assert "63.0" not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 5
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    for e in t.ENTITIES:
        assert str(e["height_m"]) not in blob
        assert str(e["width_m"]) not in blob
    assert "multnomah falls has the highest" not in plan["aggregation"].lower()
