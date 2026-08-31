"""
Offline unit tests for web-grounded task 220 (bridge span-fraction argmax) — free, no LLM, no
network. Same structure as siblings 218/219.
"""
from fractions import Fraction

from agent.app.idea_tests import test_220_tier5_bridge_span_fraction_argmax as t


def _independent_ranking():
    items = [(e["name"], e["span_m"], e["total_m"]) for e in t.ENTITIES]
    ranked = []
    for it in items:
        pos = 0
        while pos < len(ranked) and ranked[pos][1] * it[2] > it[1] * ranked[pos][2]:
            pos += 1
        ranked.insert(pos, it)
    values = {name: float(Fraction(span, total)) for name, span, total in items}
    return [r[0] for r in ranked], values


def test_ground_truth_matches_independent_cross_multiplication_solver():
    order, values = _independent_ranking()
    assert order[0] == "Humber Bridge"
    assert order[1] == "Akashi Kaikyo Bridge"
    module_values = {e["name"]: e["fraction"] for e in t.ENTITIES}
    for name, v in values.items():
        assert abs(module_values[name] - v) < 1e-9


def test_raw_winner_differs_from_derived_winner():
    by_span = max(t.ENTITIES, key=lambda e: e["span_m"])
    by_total = max(t.ENTITIES, key=lambda e: e["total_m"])
    assert by_span["name"] != t.WINNER["name"]
    assert by_total["name"] != t.WINNER["name"]
    assert by_span["name"] == "Akashi Kaikyo Bridge"
    assert by_total["name"] == "Mackinac Bridge"


def test_keystone_margin_wide():
    assert t._MARGIN > 0.20


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=5):
    return {"visit": {"count": n}}


FULL_ANSWER_ONELINE = (
    "Humber Bridge has the highest span fraction at 63.5%, higher than any of the others. "
    "Humber Bridge: total length 2,220 m, longest span 1,410 m. Akashi Kaikyo Bridge: total "
    "length 3,911 m, longest span 1,991 m. Golden Gate Bridge: total length 2,737 m, longest "
    "span 1,280 m. Verrazzano-Narrows Bridge: total length 4,176 m, longest span 1,298 m. "
    "Mackinac Bridge: total length 8,038 m, longest span 1,158 m. "
    "Sources: https://en.wikipedia.org/wiki/Humber_Bridge "
    "https://en.wikipedia.org/wiki/Akashi_Kaikyo_Bridge https://en.wikipedia.org/wiki/Golden_Gate_Bridge "
    "https://en.wikipedia.org/wiki/Verrazzano-Narrows_Bridge https://en.wikipedia.org/wiki/Mackinac_Bridge"
)

FULL_ANSWER_MULTILINE = """
Highest span fraction:
Humber Bridge
63.5%, the highest of the five.

All bridges:
Humber Bridge: total length 2,220 m, longest span 1,410 m
Akashi Kaikyo Bridge: total length 3,911 m, longest span 1,991 m
Golden Gate Bridge: total length 2,737 m, longest span 1,280 m
Verrazzano-Narrows Bridge: total length 4,176 m, longest span 1,298 m
Mackinac Bridge: total length 8,038 m, longest span 1,158 m

Sources:
https://en.wikipedia.org/wiki/Humber_Bridge
https://en.wikipedia.org/wiki/Akashi_Kaikyo_Bridge
https://en.wikipedia.org/wiki/Golden_Gate_Bridge
https://en.wikipedia.org/wiki/Verrazzano-Narrows_Bridge
https://en.wikipedia.org/wiki/Mackinac_Bridge
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
    # Names Akashi Kaikyo Bridge (longest single span, the salient decoy) as the winner.
    text = (
        "Akashi Kaikyo Bridge has the highest span fraction, since it has the longest single "
        "span. Humber Bridge: total length 2,220 m, longest span 1,410 m. Akashi Kaikyo Bridge: "
        "total length 3,911 m, longest span 1,991 m. Golden Gate Bridge: total length 2,737 m, "
        "longest span 1,280 m. Verrazzano-Narrows Bridge: total length 4,176 m, longest span "
        "1,298 m. Mackinac Bridge: total length 8,038 m, longest span 1,158 m. "
        "https://en.wikipedia.org/wiki/Akashi_Kaikyo_Bridge https://en.wikipedia.org/wiki/Humber_Bridge "
        "https://en.wikipedia.org/wiki/Golden_Gate_Bridge https://en.wikipedia.org/wiki/Verrazzano-Narrows_Bridge "
        "https://en.wikipedia.org/wiki/Mackinac_Bridge"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_argmax"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0
    assert results["winner_fraction"]["score"] == 0.0
    assert results["citation"]["score"] == 0.0


def test_partial_coverage_is_exact_fraction():
    text = (
        "Humber Bridge has the highest span fraction at 63.5%. "
        "Humber Bridge: total length 2,220 m, longest span 1,410 m. Akashi Kaikyo Bridge: total "
        "length 3,911 m, longest span 1,991 m. https://en.wikipedia.org/wiki/Humber_Bridge"
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
        assert str(e["total_m"]) not in statement
        assert str(e["span_m"]) not in statement
    assert "0.635" not in statement
    assert "63.5" not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 5
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    for e in t.ENTITIES:
        assert str(e["total_m"]) not in blob
        assert str(e["span_m"]) not in blob
    assert "humber bridge has the highest" not in plan["aggregation"].lower()
