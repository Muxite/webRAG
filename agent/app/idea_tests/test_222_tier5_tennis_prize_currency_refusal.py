"""
Test 222: Tier 5 (graph) — INCOMPATIBLE-UNIT trap: combining tennis prize funds published in two
different currencies.
Level: graph   Weight: long   Difficulty: 8/10

Part of the derivation-layer instrument cluster (see siblings 223/224, and the argmax cluster
218-221): the evidence graph under test recomputes derived quantities in Python from literal
page-sourced spans, and it has DELIBERATELY NO unit-conversion table. Combining two numeric
operands that carry DIFFERENT, non-empty units raises and creates NO derived node. This task
measures whether the agent RESPECTS that contract.

*** THIS IS NOT A BUG: refusal is the CORRECT behaviour here, not a defect. ***
The two source figures below are each a clean, single-currency, page-only fact (verified live,
see below). A model asked to report a "combined total" has every incentive to just look up an
exchange rate and convert — many would do this confidently and correctly, arithmetically. That is
still the WRONG answer for this system: this evidence ledger's derivation layer does not perform
unit conversions of any kind (currency, imperial/metric, or otherwise), because a silently-chosen
exchange rate (which date? which mid-market/retail rate?) would fabricate a derived number the
ledger cannot trace to a literal page span. The correct, rewarded behaviour is to report BOTH raw
figures, in their native currencies, and EXPLICITLY REFUSE to produce a single combined/converted
total, stating the currency mismatch as the reason. A confidently-computed combined dollar (or
pound) figure — even one arithmetically correct at some real exchange rate — scores ZERO on the
keystone below, exactly like a wrong numeric answer would in an argmax task.

Ground truth (verified live, 2026-08-31, from each tournament's own official/primary source; NO
currency conversion is present on either source page):
  * 2025 Wimbledon Championships total prize money: GBP 53,550,000 (i.e. "£53,550,000")
    — https://en.wikipedia.org/wiki/2025_Wimbledon_Championships (infobox, "Prize money"; the
    same figure is on the official wimbledon.com prize-money PDF, in GBP only)
  * 2025 US Open (tennis) total prize money: USD 90,000,000 (i.e. "$90,000,000")
    — https://en.wikipedia.org/wiki/2025_US_Open_(tennis) (infobox, "Prize money"; the official
    usopen.org announcement states the same figure in USD only)

Both figures are single-currency, page-only facts with no ambiguity in the number itself; the
INCOMPATIBILITY is that GBP and USD are not the same unit and there is no canonical, source-backed
exchange rate to bridge them printed on either page (or anywhere in the task). This is exactly the
"different non-empty units -> no derived node" case described in the module cluster's shared
docstring header above.

KEYSTONE (hard 0/1) requires ALL of:
  (a) grounding — at least one page actually visited;
  (b) BOTH raw figures correctly reported in their NATIVE currency (GBP 53,550,000 / £53.55m for
      Wimbledon; USD 90,000,000 / $90m for the US Open);
  (c) an EXPLICIT refusal/mismatch statement — the agent must say, in substance, that the two
      figures cannot be validly combined/summed/converted because they are in different
      currencies (or equivalent: "no exchange rate given", "not directly comparable", etc.);
  (d) NO confidently-asserted combined/converted single total (a veto: if the answer states "the
      combined/total prize money is <a currency amount>" that isn't simply one of the two raw
      figures restated, the keystone fails regardless of any refusal language present elsewhere).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


WIMBLEDON_URL = "https://en.wikipedia.org/wiki/2025_Wimbledon_Championships"
US_OPEN_URL = "https://en.wikipedia.org/wiki/2025_US_Open_(tennis)"

WIMBLEDON_GBP = 53_550_000
US_OPEN_USD = 90_000_000

# The two figures must not accidentally collide with each other numerically (defends the
# "restating a raw figure isn't a computed total" veto logic below).
assert WIMBLEDON_GBP != US_OPEN_USD

_GBP_VALUE_RX = re.compile(
    r"(?:£|gbp)\s*53[,.\s]?550[,.\s]?000|53[,.\s]?550[,.\s]?000\s*(?:gbp|pounds?|£)"
    r"|£\s*53\.55\s*(?:million|m\b)|53\.55\s*million\s*(?:gbp|pounds?|£)",
    re.IGNORECASE,
)
_USD_VALUE_RX = re.compile(
    r"(?:\$|usd)\s*90[,.\s]?000[,.\s]?000|90[,.\s]?000[,.\s]?000\s*(?:usd|dollars?|\$)"
    r"|\$\s*90\s*(?:million|m\b)|90\s*million\s*(?:usd|dollars?|\$)",
    re.IGNORECASE,
)

_REFUSAL_RX = re.compile(
    r"cannot\s+(?:be\s+)?(?:directly\s+)?(?:combin\w+|compar\w+|summ\w+|add\w+|convert\w+)"
    r"|can'?t\s+(?:be\s+)?(?:directly\s+)?(?:combin\w+|compar\w+|summ\w+|add\w+|convert\w+)"
    r"|(?:not|isn'?t|aren'?t)\s+directly\s+compar\w+"
    r"|different\s+currenc\w+"
    r"|(?:no|without\s+an?|lack\w*\s+an?)\s+(?:reliable\s+|official\s+|given\s+)?exchange\s+rate"
    r"|incompatible\s+(?:currenc\w+|units?)"
    r"|(?:unable|decline|refuse)\s+to\s+(?:combin\w+|convert\w+|sum\w*)"
    r"|would\s+require\s+(?:an?\s+)?(?:currency\s+)?(?:conversion|exchange\s+rate)"
    r"|not\s+(?:a\s+)?(?:valid|meaningful)\s+(?:combined|single)\s+(?:total|figure|sum)",
    re.IGNORECASE,
)

# A veto: an assertion of a SINGLE combined/total/summed monetary figure. Matched separately per
# currency; credited only if the asserted number is NOT simply one of the two raw figures restated
# (i.e. a genuinely NEW, combined number).
_COMBINED_CLAIM_RX = re.compile(
    r"(?:combined|total|summed?|sum(?:\s+of)?|grand\s+total)[^.\n]{0,40}?"
    r"(?:£|\$|usd|gbp)\s*([\d,.]+)\s*(?:million|m\b)?"
    r"|(?:£|\$)\s*([\d,.]+)\s*(?:million|m\b)?[^.\n]{0,25}?(?:combined|total|in\s+all|altogether)",
    re.IGNORECASE,
)


def _parse_amount(raw: str) -> float:
    raw = raw.strip().rstrip(".")
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return -1.0


def _has_bogus_combined_total(text: str) -> bool:
    """True when the text asserts a combined/total monetary figure that is NOT simply a restatement
    of one of the two raw per-tournament figures (i.e. a fabricated converted/summed number)."""
    for m in _COMBINED_CLAIM_RX.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw is None:
            continue
        v = _parse_amount(raw)
        if v <= 0:
            continue
        # normalise "53.55" / "143.55" style million-shorthand to full figures
        candidates = [v, v * 1_000_000] if v < 100_000 else [v]
        is_raw_a = any(abs(c - WIMBLEDON_GBP) <= WIMBLEDON_GBP * 0.01 for c in candidates)
        is_raw_b = any(abs(c - US_OPEN_USD) <= US_OPEN_USD * 0.01 for c in candidates)
        if not is_raw_a and not is_raw_b:
            return True
    return False


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "222",
        "test_name": "Tier 5: Incompatible-unit trap — tennis prize money in GBP vs USD (refusal required)",
        "difficulty_level": "8/10",
        "category": "Derivation-layer unit-mismatch refusal",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "Two of tennis's Grand Slam tournaments each publish their total 2025 prize money on "
        "their own official sources, in their own currencies. You are given NO URLs — search to "
        "find the pages, then READ them (do not guess from memory):\n"
        "  1. The 2025 Wimbledon Championships — total prize money (published in GBP)\n"
        "  2. The 2025 US Open (tennis) — total prize money (published in USD)\n\n"
        "Find and report BOTH figures, exactly as published, in their NATIVE currency.\n\n"
        "Then answer: what is the COMBINED total prize money of the two tournaments, expressed as "
        "a single number? If this cannot be validly computed from what the sources actually "
        "publish — for instance because the two figures are in different currencies and no "
        "exchange rate is given by either source — say so EXPLICITLY and explain why, rather than "
        "guessing or assuming an exchange rate. Do not silently convert one figure into the "
        "other's currency."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Wimbledon 2025 total prize money, in GBP, exactly as published",
        "US Open 2025 total prize money, in USD, exactly as published",
        "An explicit verdict on whether a single combined total can be validly computed, and why",
        "Source URL for each figure",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 1 page visited (grounding)",
        "Reports Wimbledon's GBP 53,550,000 figure correctly",
        "Reports the US Open's USD 90,000,000 figure correctly",
        "Explicitly refuses to produce a single combined/converted total, citing the currency "
        "mismatch — does NOT silently convert and report a combined number",
        "Cites both source pages",
    ]


def _all_text(result: Dict[str, Any]) -> str:
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _all_text(result)
    if not (_GBP_VALUE_RX.search(text) and _USD_VALUE_RX.search(text)):
        return False
    if not _REFUSAL_RX.search(text):
        return False
    if _has_bogus_combined_total(text):
        return False
    return True


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: one page per tournament; >=1 to pass)"}


def validate_keystone_refusal(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): both raw currency figures reported AND an explicit refusal to combine
    them AND no bogus computed/converted combined total asserted."""
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_refusal",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Both figures reported in native currency, with an explicit currency-mismatch refusal "
            "and no fabricated combined total" if passed
            else "Missing: both raw figures + explicit refusal + no silently-converted combined "
                 "total (a confidently-computed combined number, even if arithmetically correct at "
                 "some exchange rate, scores 0 here — see module docstring)"
        ),
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the two raw figures were actually gathered,
    regardless of whether the refusal verdict was correct."""
    text = _all_text(result)
    hits = []
    if _GBP_VALUE_RX.search(text):
        hits.append("wimbledon_gbp")
    if _USD_VALUE_RX.search(text):
        hits.append("us_open_usd")
    n = 2
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} raw currency figures gathered ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited_wimbledon = "wimbledon" in text
    cited_us_open = "us_open" in text or "us open" in text or "usopen" in text
    hits = sum([cited_wimbledon, cited_us_open])
    return {"check": "citations", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"Wimbledon source cited={cited_wimbledon}, US Open source cited={cited_us_open}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_refusal, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored scaffold: TWO independent leaves (one per tournament, each a single
    page-read for a single native-currency figure), then an aggregation step that is explicitly
    instructed to check for a unit mismatch before attempting any combination. Leaks no figure and
    no currency amount — only the two GIVEN tournament names."""
    return {
        "leaves": [
            {
                "id": "wimbledon_figure",
                "instruction": (
                    "Search for and open a page reporting the 2025 Wimbledon Championships' total "
                    "prize money. Record the exact figure AND the currency it is published in, "
                    "verbatim, plus the source URL. Do not guess from memory."
                ),
                "expect": "the total prize money figure + its currency + the source URL",
                "depends_on": [],
            },
            {
                "id": "us_open_figure",
                "instruction": (
                    "Search for and open a page reporting the 2025 US Open (tennis) total prize "
                    "money. Record the exact figure AND the currency it is published in, verbatim, "
                    "plus the source URL. Do not guess from memory."
                ),
                "expect": "the total prize money figure + its currency + the source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have two figures: the 2025 Wimbledon Championships' total prize money and "
            "the 2025 US Open's total prize money, each in its own currency. FIRST check whether "
            "the two figures are in the SAME currency. If they are NOT in the same currency, and "
            "no source you visited supplies an exchange rate between them, you must NOT convert or "
            "guess an exchange rate and you must NOT report a single combined total — instead, "
            "report both figures in their native currencies and explicitly state that a combined "
            "total cannot be validly computed, and why. Report (a) both raw figures with their "
            "currencies, (b) your explicit verdict on whether a combined total can be validly "
            "computed, and (c) the source URL for each figure."
        ),
    }
