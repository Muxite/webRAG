"""
Test 224: Tier 5 (graph) — INCOMPATIBLE-UNIT trap: combining a Bollywood film's India opening
weekend (INR crore) and a Hollywood film's US opening weekend (USD).
Level: graph   Weight: long   Difficulty: 8/10

Sibling of 222/223 (see 222's docstring for the full rationale of the derivation-layer's
no-unit-conversion contract). *** REFUSAL IS THE CORRECT BEHAVIOUR HERE, NOT A BUG. *** A
confidently-computed combined total — even one arithmetically correct at some real exchange rate —
scores ZERO on the keystone below.

Ground truth (verified live, 2026-08-31, from each figure's own primary trade-press source; NO
currency conversion is present on either source page):
  * Chhaava (2025) — India opening weekend (3-day) net box office collection: INR 121 crore
    (i.e. "₹121 crore") — https://www.hollywoodreporterindia.com/features/insight/chhaava-box-office-collection-vicky-kaushal-starrer-smashes-records-with-121-crore-opening-weekend
    (no US dollar equivalent given anywhere in that article)
  * A Minecraft Movie (2025) — US domestic opening weekend box office gross: USD 162,753,003
    (i.e. "$162,753,003") — https://www.boxofficemojo.com/release/rl746096129/ (Box Office Mojo,
    no rupee equivalent given)

Both figures are single-currency, page-only facts: one denominated in Indian rupees (using the
"crore" = 10,000,000 unit convention standard in Indian trade press), the other in US dollars.
Neither source supplies a conversion, and — unlike a fixed physical-unit conversion — currency
exchange rates fluctuate daily, so there is no single canonical "correct" INR/USD rate to apply
even if one wanted to convert.

KEYSTONE (hard 0/1) requires ALL of:
  (a) grounding — at least one page actually visited;
  (b) BOTH raw figures correctly reported in their NATIVE currency (INR 121 crore / ₹121 crore for
      Chhaava; USD 162,753,003 / $162,753,003 (or ~$162.75 million) for A Minecraft Movie);
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


CHHAAVA_URL = (
    "https://www.hollywoodreporterindia.com/features/insight/"
    "chhaava-box-office-collection-vicky-kaushal-starrer-smashes-records-with-121-crore-opening-weekend"
)
MINECRAFT_URL = "https://www.boxofficemojo.com/release/rl746096129/"

CHHAAVA_INR_CRORE = 121
MINECRAFT_USD = 162_753_003

assert CHHAAVA_INR_CRORE != MINECRAFT_USD

_INR_VALUE_RX = re.compile(r"(?:₹|inr|rs\.?)\s*121\s*crore|121\s*crore", re.IGNORECASE)
_USD_VALUE_RX = re.compile(
    r"(?:\$|usd)\s*162[,.\s]?753[,.\s]?003|162[,.\s]?753[,.\s]?003\s*(?:usd|dollars?|\$)"
    r"|\$\s*162\.7\d?\s*(?:million|m\b)|162\.7\d?\s*million\s*(?:usd|dollars?|\$)",
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

# A "combined" claim expressed as a dollar/crore amount attached to combined/total wording.
_COMBINED_CLAIM_RX = re.compile(
    r"(?:combined|total|summed?|sum(?:\s+of)?|grand\s+total)[^.\n]{0,40}?"
    r"(?:£|\$|₹|usd|inr)\s*([\d,.]+)\s*(?:million|crore|m\b)?"
    r"|(?:\$|₹)\s*([\d,.]+)\s*(?:million|crore|m\b)?[^.\n]{0,25}?(?:combined|total|in\s+all|altogether)",
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
        # candidate renderings: the bare number, as millions, or (small numbers) as crore
        candidates = {v}
        if v < 100_000:
            candidates.add(v * 1_000_000)   # "$162.75 million" style
        if v < 100_000:
            candidates.add(v)               # crore count itself (e.g. "121")
        is_raw_a = any(abs(c - CHHAAVA_INR_CRORE) <= max(1.0, CHHAAVA_INR_CRORE * 0.01) for c in candidates)
        is_raw_b = any(abs(c - MINECRAFT_USD) <= MINECRAFT_USD * 0.01 for c in candidates)
        if not is_raw_a and not is_raw_b:
            return True
    return False


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "224",
        "test_name": "Tier 5: Incompatible-unit trap — India box office (INR crore) vs US box office (USD)",
        "difficulty_level": "8/10",
        "category": "Derivation-layer unit-mismatch refusal",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "Two 2025 films each had a widely reported opening weekend, each denominated in the local "
        "box-office currency convention. You are given NO URLs — search to find the pages, then "
        "READ them (do not guess from memory):\n"
        "  1. 'Chhaava' (2025 Hindi-language film) — India opening weekend (3-day) net box office "
        "collection (published in INR crore)\n"
        "  2. 'A Minecraft Movie' (2025) — US domestic opening weekend box office gross (published "
        "in USD)\n\n"
        "Find and report BOTH figures, exactly as published, in their NATIVE currency/unit.\n\n"
        "Then answer: what is the COMBINED total of the two opening weekends, expressed as a "
        "single number? If this cannot be validly computed from what the sources actually publish "
        "— for instance because the two figures are in different currencies and no exchange rate "
        "is given by either source — say so EXPLICITLY and explain why, rather than guessing or "
        "assuming an exchange rate. Do not silently convert one figure into the other's currency."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Chhaava's India opening weekend collection, in INR crore, exactly as published",
        "A Minecraft Movie's US domestic opening weekend gross, in USD, exactly as published",
        "An explicit verdict on whether a single combined total can be validly computed, and why",
        "Source URL for each figure",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 1 page visited (grounding)",
        "Reports Chhaava's INR 121 crore figure correctly",
        "Reports A Minecraft Movie's USD 162,753,003 figure correctly",
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
    if not (_INR_VALUE_RX.search(text) and _USD_VALUE_RX.search(text)):
        return False
    if not _REFUSAL_RX.search(text):
        return False
    if _has_bogus_combined_total(text):
        return False
    return True


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: one page per film; >=1 to pass)"}


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
    if _INR_VALUE_RX.search(text):
        hits.append("chhaava_inr_crore")
    if _USD_VALUE_RX.search(text):
        hits.append("minecraft_usd")
    n = 2
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} raw currency figures gathered ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited_chhaava = "hollywoodreporterindia" in text or "chhaava" in text
    cited_minecraft = "boxofficemojo" in text
    hits = sum([cited_chhaava, cited_minecraft])
    return {"check": "citations", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"Chhaava source cited={cited_chhaava}, Box Office Mojo cited={cited_minecraft}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_refusal, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    return {
        "leaves": [
            {
                "id": "chhaava_figure",
                "instruction": (
                    "Search for and open a page reporting the India opening weekend (3-day) box "
                    "office collection of the 2025 film 'Chhaava'. Record the exact figure AND the "
                    "currency/unit it is published in (e.g. INR crore), verbatim, plus the source "
                    "URL. Do not guess from memory."
                ),
                "expect": "the opening weekend figure + its currency/unit + the source URL",
                "depends_on": [],
            },
            {
                "id": "minecraft_figure",
                "instruction": (
                    "Search for and open a page reporting the US domestic opening weekend box "
                    "office gross of the 2025 film 'A Minecraft Movie'. Record the exact figure "
                    "AND the currency it is published in, verbatim, plus the source URL. Do not "
                    "guess from memory."
                ),
                "expect": "the opening weekend figure + its currency + the source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have two figures: Chhaava's India opening weekend collection and A Minecraft "
            "Movie's US domestic opening weekend gross, each in its own currency/unit. FIRST check "
            "whether the two figures are in the SAME currency. If they are NOT in the same "
            "currency, and no source you visited supplies an exchange rate between them, you must "
            "NOT convert or guess an exchange rate and you must NOT report a single combined "
            "total — instead, report both figures in their native currencies/units and explicitly "
            "state that a combined total cannot be validly computed, and why. Report (a) both raw "
            "figures with their currencies/units, (b) your explicit verdict on whether a combined "
            "total can be validly computed, and (c) the source URL for each figure."
        ),
    }
