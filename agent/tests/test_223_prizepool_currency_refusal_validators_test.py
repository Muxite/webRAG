"""
Offline unit tests for web-grounded task 223 (snooker/poker prize-money currency-mismatch
refusal) — free, no LLM, no network. Structure mirrors sibling 222.
"""
from agent.app.idea_tests import test_223_tier5_prizepool_currency_refusal as t


def test_ground_truth_figures_are_distinct_and_verified():
    assert t.SNOOKER_GBP == 2_395_000
    assert t.WSOP_USD == 90_535_500
    assert t.SNOOKER_GBP != t.WSOP_USD


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=2):
    return {"visit": {"count": n}}


FULL_REFUSAL_ONELINE = (
    "World Snooker Championship 2025 total prize fund: £2,395,000 (GBP). WSOP Main Event 2025 "
    "total prize pool: $90,535,500 (USD). These two figures cannot be directly combined or summed "
    "because they are published in different currencies (GBP and USD) and neither source gives "
    "an exchange rate, so no valid single combined total can be computed here. "
    "Sources: https://en.wikipedia.org/wiki/2025_World_Snooker_Championship "
    "https://www.wsop.com/news/2025-wsop-main-event-official-prize-pool-and-total-entries/"
)

FULL_REFUSAL_MULTILINE = """
World Snooker Championship 2025 total prize fund:
£2,395,000 (GBP)

WSOP Main Event 2025 total prize pool:
$90,535,500 (USD)

Verdict:
These figures are in different currencies. No exchange rate is given by either source, so I
cannot combine them into a single total -- doing so would require guessing an exchange rate,
which this system does not do.

Sources:
https://en.wikipedia.org/wiki/2025_World_Snooker_Championship
https://www.wsop.com/news/2025-wsop-main-event-official-prize-pool-and-total-entries/
"""


def test_full_refusal_oneline_scores_all_checks_at_1():
    r, o = _r(FULL_REFUSAL_ONELINE), _obs()
    for fn in t.get_validation_functions():
        res = fn(r, o)
        assert res["score"] == 1.0, (res["check"], res["reason"])


def test_full_refusal_multiline_scores_all_checks_at_1():
    r, o = _r(FULL_REFUSAL_MULTILINE), _obs()
    for fn in t.get_validation_functions():
        res = fn(r, o)
        assert res["score"] == 1.0, (res["check"], res["reason"])


def test_confidently_converted_combined_total_scores_keystone_zero():
    text = (
        "World Snooker Championship 2025 total prize fund: £2,395,000 (GBP). WSOP Main Event 2025 "
        "total prize pool: $90,535,500 (USD). Converting the snooker figure to USD at an exchange "
        "rate of approximately 1.34, the combined total prize money of both events is "
        "approximately $93.7 million. "
        "https://en.wikipedia.org/wiki/2025_World_Snooker_Championship "
        "https://www.wsop.com/news/2025-wsop-main-event-official-prize-pool-and-total-entries/"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_refusal"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0
    assert results["citations"]["score"] == 0.0


def test_missing_refusal_language_scores_keystone_zero():
    text = (
        "World Snooker Championship 2025 total prize fund: £2,395,000 (GBP). WSOP Main Event 2025 "
        "total prize pool: $90,535,500 (USD). "
        "https://en.wikipedia.org/wiki/2025_World_Snooker_Championship "
        "https://www.wsop.com/news/2025-wsop-main-event-official-prize-pool-and-total-entries/"
    )
    r, o = _r(text), _obs()
    ks = t.validate_keystone_refusal(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0


def test_partial_coverage_is_exact_fraction():
    text = "World Snooker Championship 2025 total prize fund: £2,395,000 (GBP)."
    r, o = _r(text), _obs()
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 0.5


def test_no_visits_zeroes_keystone():
    r, o = _r(FULL_REFUSAL_ONELINE), _obs(0)
    ks = t.validate_keystone_refusal(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0


def test_statement_leak_raw_inputs_present_derived_absent():
    statement = t.get_task_statement()
    assert "World Snooker Championship" in statement
    assert "WSOP" in statement
    assert "2,395,000" not in statement
    assert "90,535,500" not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 2
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    assert "2,395,000" not in blob
    assert "90,535,500" not in blob
    assert "£" not in blob and "$" not in blob
