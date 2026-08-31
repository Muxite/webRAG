"""
Offline unit tests for web-grounded task 222 (tennis prize-money currency-mismatch refusal) —
free, no LLM, no network.

Jobs:
  1. Ground-truth sanity: the two raw figures are distinct and neither collides numerically with
     a plausible converted value of the other (defends the "bogus combined total" veto logic).
  2. Adversarial validator hardening: full REFUSAL answer (single- and multi-line) -> keystone 1.0;
     a CONFIDENTLY-CONVERTED combined total (even if it restates both raw figures correctly first)
     -> keystone 0, because the point under test is that conversion itself is the wrong behaviour,
     not that the raw figures were missed; partial coverage; no-visits; leak checks.
"""
from agent.app.idea_tests import test_222_tier5_tennis_prize_currency_refusal as t


def test_ground_truth_figures_are_distinct_and_verified():
    assert t.WIMBLEDON_GBP == 53_550_000
    assert t.US_OPEN_USD == 90_000_000
    assert t.WIMBLEDON_GBP != t.US_OPEN_USD


def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


def _obs(n=2):
    return {"visit": {"count": n}}


FULL_REFUSAL_ONELINE = (
    "Wimbledon 2025 total prize money: £53,550,000 (GBP). US Open 2025 total prize money: "
    "$90,000,000 (USD). These two figures cannot be directly combined or summed because they are "
    "published in different currencies (GBP and USD) and neither source gives an exchange rate, "
    "so no valid single combined total can be computed here. "
    "Sources: https://en.wikipedia.org/wiki/2025_Wimbledon_Championships "
    "https://en.wikipedia.org/wiki/2025_US_Open_(tennis)"
)

FULL_REFUSAL_MULTILINE = """
Wimbledon 2025 total prize money:
£53,550,000 (GBP)

US Open 2025 total prize money:
$90,000,000 (USD)

Verdict:
These figures are in different currencies. No exchange rate is given by either source, so I
cannot combine them into a single total -- doing so would require guessing an exchange rate,
which this system does not do.

Sources:
https://en.wikipedia.org/wiki/2025_Wimbledon_Championships
https://en.wikipedia.org/wiki/2025_US_Open_(tennis)
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
    # Reports BOTH raw figures correctly, but then confidently converts and asserts a combined
    # total -- arithmetically plausible (at some real exchange rate) but WRONG for this system.
    text = (
        "Wimbledon 2025 total prize money: £53,550,000 (GBP). US Open 2025 total prize money: "
        "$90,000,000 (USD). Converting the Wimbledon figure to USD at an exchange rate of "
        "approximately 1.34, the combined total prize money of both tournaments is approximately "
        "$161.8 million. "
        "https://en.wikipedia.org/wiki/2025_Wimbledon_Championships "
        "https://en.wikipedia.org/wiki/2025_US_Open_(tennis)"
    )
    r, o = _r(text), _obs()
    results = {res["check"]: res for res in (fn(r, o) for fn in t.get_validation_functions())}
    assert results["keystone_refusal"]["score"] == 0.0
    assert results["coverage"]["score"] == 1.0            # coverage un-gated: both raw figures present
    assert results["citations"]["score"] == 0.0            # gated -> 0


def test_confidently_converted_total_with_hedged_refusal_language_still_scores_zero():
    # Includes SOME refusal-sounding language but still asserts a fabricated combined figure --
    # the veto on a bogus combined total must win regardless of hedging elsewhere.
    text = (
        "Wimbledon 2025 total prize money: £53,550,000 (GBP). US Open 2025 total prize money: "
        "$90,000,000 (USD). These are different currencies, but for reference the combined total "
        "prize money is approximately $161.8 million. "
        "https://en.wikipedia.org/wiki/2025_Wimbledon_Championships "
        "https://en.wikipedia.org/wiki/2025_US_Open_(tennis)"
    )
    r, o = _r(text), _obs()
    ks = t.validate_keystone_refusal(r, o)
    assert ks["score"] == 0.0


def test_missing_refusal_language_scores_keystone_zero_even_with_no_bogus_total():
    # Both raw figures reported, no combined total asserted at all, but ALSO no explicit refusal
    # statement -- the agent simply didn't address the combination question.
    text = (
        "Wimbledon 2025 total prize money: £53,550,000 (GBP). US Open 2025 total prize money: "
        "$90,000,000 (USD). "
        "https://en.wikipedia.org/wiki/2025_Wimbledon_Championships "
        "https://en.wikipedia.org/wiki/2025_US_Open_(tennis)"
    )
    r, o = _r(text), _obs()
    ks = t.validate_keystone_refusal(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0


def test_partial_coverage_is_exact_fraction():
    text = "Wimbledon 2025 total prize money: £53,550,000 (GBP)."
    r, o = _r(text), _obs()
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 0.5


def test_no_visits_zeroes_keystone():
    r, o = _r(FULL_REFUSAL_ONELINE), _obs(0)
    ks = t.validate_keystone_refusal(r, o)
    assert ks["score"] == 0.0
    cov = t.validate_coverage(r, o)
    assert cov["score"] == 1.0  # coverage is un-gated even with zero visits


def test_statement_leak_raw_inputs_present_derived_absent():
    statement = t.get_task_statement()
    assert "Wimbledon" in statement
    assert "US Open" in statement
    # neither raw figure (nor any converted/combined figure) is leaked in the statement
    assert "53,550,000" not in statement
    assert "90,000,000" not in statement
    assert "53.55" not in statement


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    assert len(plan["leaves"]) == 2
    blob = " ".join(leaf["instruction"] + leaf["expect"] for leaf in plan["leaves"]) + plan["aggregation"]
    assert "53,550,000" not in blob
    assert "90,000,000" not in blob
    assert "£" not in blob and "$" not in blob
