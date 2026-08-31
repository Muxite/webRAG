"""
Offline unit tests for web-grounded task 224 (India/US box-office currency-mismatch refusal) —
free, no LLM, no network. Structure mirrors siblings 222/223, plus an extra check for the
crore-vs-dollar magnitude asymmetry in the "bogus combined total" veto.
"""
from agent.app.idea_tests import test_224_tier5_boxoffice_currency_refusal as t


def test_ground_truth_figures_are_distinct_and_verified():
    assert t.CHHAAVA_INR_CRORE == 121
    assert t.MINECRAFT_USD == 162_753_003
    assert t.CHHAAVA_INR_CRORE != t.MINECRAFT_USD


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=2):
    return {"visit": {"count": n}}


FULL_REFUSAL_ONELINE = (
    "Chhaava's India opening weekend collection: ₹121 crore (INR). A Minecraft Movie's US "
    "domestic opening weekend gross: $162,753,003 (USD). These two figures cannot be directly "
    "combined or summed because they are published in different currencies (INR crore and USD) "
    "and neither source gives an exchange rate, so no valid single combined total can be computed "
    "here. "
    "Sources: https://www.hollywoodreporterindia.com/features/insight/"
    "chhaava-box-office-collection-vicky-kaushal-starrer-smashes-records-with-121-crore-opening-weekend "
    "https://www.boxofficemojo.com/release/rl746096129/"
)

FULL_REFUSAL_MULTILINE = """
Chhaava India opening weekend collection:
121 crore (INR)

A Minecraft Movie US domestic opening weekend gross:
$162,753,003 (USD)

Verdict:
These figures are in different currencies. No exchange rate is given by either source, so I
cannot combine them into a single total -- doing so would require guessing an exchange rate,
which this system does not do.

Sources:
https://www.hollywoodreporterindia.com/features/insight/chhaava-box-office-collection-vicky-kaushal-starrer-smashes-records-with-121-crore-opening-weekend
https://www.boxofficemojo.com/release/rl746096129/
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
        "Chhaava's India opening weekend collection: ₹121 crore (INR). A Minecraft Movie's US "
        "domestic opening weekend gross: $162,753,003 (USD). Converting the rupee figure to USD "
        "at an exchange rate of approximately 83.5, the combined total of both opening weekends is "
        "approximately $177.2 million. "
        "https://www.hollywoodreporterindia.com/features/insight/chhaava-box-office-collection-vicky-kaushal-starrer-smashes-records-with-121-crore-opening-weekend "
        "https://www.boxofficemojo.com/release/rl746096129/"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_refusal"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0
    assert results["citations"]["score"] == 0.0


def test_missing_refusal_language_scores_keystone_zero():
    text = (
        "Chhaava's India opening weekend collection: ₹121 crore (INR). A Minecraft Movie's US "
        "domestic opening weekend gross: $162,753,003 (USD). "
        "https://www.hollywoodreporterindia.com/features/insight/chhaava-box-office-collection-vicky-kaushal-starrer-smashes-records-with-121-crore-opening-weekend "
        "https://www.boxofficemojo.com/release/rl746096129/"
    )
    r, o = _r(text), _obs()
    ks = t.validate_keystone_refusal(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0


def test_partial_coverage_is_exact_fraction():
    text = "Chhaava's India opening weekend collection: ₹121 crore (INR)."
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
    assert "Chhaava" in statement
    assert "A Minecraft Movie" in statement
    assert "121 crore" not in statement
    assert "162,753,003" not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 2
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    assert "121 crore" not in blob
    assert "162,753,003" not in blob
    assert "₹" not in blob and "$" not in blob
