"""
Test 223: Tier 5 (graph) — INCOMPATIBLE-UNIT trap: combining a snooker prize fund (GBP) and a
poker tournament prize pool (USD).
Level: graph   Weight: long   Difficulty: 8/10

Sibling of 222/224 (see 222's docstring for the full rationale of the derivation-layer's
no-unit-conversion contract). *** REFUSAL IS THE CORRECT BEHAVIOUR HERE, NOT A BUG. *** A
confidently-computed combined total — even one arithmetically correct at some real exchange rate —
scores ZERO on the keystone below.

Ground truth (verified live, 2026-08-31, from each event's own official/primary source; NO
currency conversion is present on either source page):
  * 2025 World Snooker Championship total prize fund: GBP 2,395,000 (i.e. "£2,395,000")
    — https://en.wikipedia.org/wiki/2025_World_Snooker_Championship (infobox, "Total prize fund")
  * 2025 WSOP (World Series of Poker) Main Event total prize pool: USD 90,535,500
    (i.e. "$90,535,500") — https://www.wsop.com/news/2025-wsop-main-event-official-prize-pool-and-total-entries/
    (official WSOP announcement, in USD only)

Both figures are single-currency, page-only facts; the two events (a UK cue-sports championship
and a US poker tournament) have no natural reason to share a currency, and neither source supplies
a conversion.

KEYSTONE (hard 0/1) requires ALL of:
  (a) grounding — at least one page actually visited;
  (b) BOTH raw figures correctly reported in their NATIVE currency (GBP 2,395,000 for the World
      Snooker Championship; USD 90,535,500 for the WSOP Main Event);
  (c) an EXPLICIT refusal/mismatch statement — the agent must say, in substance, that the two
      figures cannot be validly combined/summed/converted because they are in different
      currencies;
  (d) NO confidently-asserted combined/converted single total (a veto: any assertion of a
      combined/total figure that is not simply a restatement of one of the two raw figures fails
      the keystone, regardless of any refusal language present elsewhere).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


SNOOKER_URL = "https://en.wikipedia.org/wiki/2025_World_Snooker_Championship"
WSOP_URL = "https://www.wsop.com/news/2025-wsop-main-event-official-prize-pool-and-total-entries/"

SNOOKER_GBP = 2_395_000
WSOP_USD = 90_535_500

assert SNOOKER_GBP != WSOP_USD

_GBP_VALUE_RX = re.compile(
    r"(?:£|gbp)\s*2[,.\s]?395[,.\s]?000|2[,.\s]?395[,.\s]?000\s*(?:gbp|pounds?|£)"
    r"|£\s*2\.395\s*(?:million|m\b)|2\.395\s*million\s*(?:gbp|pounds?|£)",
    re.IGNORECASE,
)
_USD_VALUE_RX = re.compile(
    r"(?:\$|usd)\s*90[,.\s]?535[,.\s]?500|90[,.\s]?535[,.\s]?500\s*(?:usd|dollars?|\$)"
    r"|\$\s*90\.5\d?\s*(?:million|m\b)|90\.5\d?\s*million\s*(?:usd|dollars?|\$)",
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
    for m in _COMBINED_CLAIM_RX.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw is None:
            continue
        v = _parse_amount(raw)
        if v <= 0:
            continue
        candidates = [v, v * 1_000_000] if v < 100_000 else [v]
        is_raw_a = any(abs(c - SNOOKER_GBP) <= SNOOKER_GBP * 0.01 for c in candidates)
        is_raw_b = any(abs(c - WSOP_USD) <= WSOP_USD * 0.01 for c in candidates)
        if not is_raw_a and not is_raw_b:
            return True
    return False


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "223",
        "test_name": "Tier 5: Incompatible-unit trap — snooker prize fund (GBP) vs poker prize pool (USD)",
        "difficulty_level": "8/10",
        "category": "Derivation-layer unit-mismatch refusal",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "Two major 2025 competitions each publish their total prize money on their own official "
        "sources, in their own currencies. You are given NO URLs — search to find the pages, then "
        "READ them (do not guess from memory):\n"
        "  1. The 2025 World Snooker Championship — total prize fund (published in GBP)\n"
        "  2. The 2025 WSOP (World Series of Poker) Main Event — total prize pool (published in "
        "USD)\n\n"
        "Find and report BOTH figures, exactly as published, in their NATIVE currency.\n\n"
        "Then answer: what is the COMBINED total prize money of the two events, expressed as a "
        "single number? If this cannot be validly computed from what the sources actually publish "
        "— for instance because the two figures are in different currencies and no exchange rate "
        "is given by either source — say so EXPLICITLY and explain why, rather than guessing or "
        "assuming an exchange rate. Do not silently convert one figure into the other's currency."
    )


def get_required_deliverables() -> List[str]:
    return [
        "World Snooker Championship 2025 total prize fund, in GBP, exactly as published",
        "WSOP Main Event 2025 total prize pool, in USD, exactly as published",
        "An explicit verdict on whether a single combined total can be validly computed, and why",
        "Source URL for each figure",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 1 page visited (grounding)",
        "Reports the World Snooker Championship's GBP 2,395,000 figure correctly",
        "Reports the WSOP Main Event's USD 90,535,500 figure correctly",
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
            "reason": f"{n} visit(s) (target >=2: one page per event; >=1 to pass)"}


def validate_keystone_refusal(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
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
    text = _all_text(result)
    hits = []
    if _GBP_VALUE_RX.search(text):
        hits.append("snooker_gbp")
    if _USD_VALUE_RX.search(text):
        hits.append("wsop_usd")
    n = 2
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} raw currency figures gathered ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited_snooker = "snooker" in text
    cited_wsop = "wsop" in text
    hits = sum([cited_snooker, cited_wsop])
    return {"check": "citations", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"Snooker source cited={cited_snooker}, WSOP source cited={cited_wsop}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_refusal, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    return {
        "leaves": [
            {
                "id": "snooker_figure",
                "instruction": (
                    "Search for and open a page reporting the 2025 World Snooker Championship's "
                    "total prize fund. Record the exact figure AND the currency it is published "
                    "in, verbatim, plus the source URL. Do not guess from memory."
                ),
                "expect": "the total prize fund figure + its currency + the source URL",
                "depends_on": [],
            },
            {
                "id": "wsop_figure",
                "instruction": (
                    "Search for and open a page reporting the 2025 WSOP (World Series of Poker) "
                    "Main Event total prize pool. Record the exact figure AND the currency it is "
                    "published in, verbatim, plus the source URL. Do not guess from memory."
                ),
                "expect": "the total prize pool figure + its currency + the source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have two figures: the World Snooker Championship's total prize fund and the "
            "WSOP Main Event's total prize pool, each in its own currency. FIRST check whether the "
            "two figures are in the SAME currency. If they are NOT in the same currency, and no "
            "source you visited supplies an exchange rate between them, you must NOT convert or "
            "guess an exchange rate and you must NOT report a single combined total — instead, "
            "report both figures in their native currencies and explicitly state that a combined "
            "total cannot be validly computed, and why. Report (a) both raw figures with their "
            "currencies, (b) your explicit verdict on whether a combined total can be validly "
            "computed, and (c) the source URL for each figure."
        ),
    }
