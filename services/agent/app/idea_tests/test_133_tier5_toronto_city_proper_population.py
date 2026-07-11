"""
Test 133: Tier 5 (adaptive-targeted) — CONFLICTING-SOURCE RECONCILIATION (trust decision).
Level: integration   Weight: long   Difficulty: 9/10

LOW-CONTEXT source-trust decision task (Bucket B). A good ADAPTIVE agent must reconcile two
reputable-looking "population of Toronto" figures by applying a checkable SCOPE rule, on a narrow
golden path of 2-3 reads — NOT by breadth, NOT by averaging, NOT by grabbing the biggest number.

The quantity: the population of the City of Toronto (city proper) at the 2021 Canadian census.
  * WRONG (metro-area scope, biggest-number bait): the metropolitan figure — Census Metropolitan
    Area 6,202,225 (or Greater Toronto Area 6,712,341). A naive agent grabs the larger metro number
    as "the population of Toronto."
  * CORRECT (keystone): the CITY-PROPER (administrative City of Toronto) 2021 census population,
    2,794,356 — the infobox "City" figure.
  The reconciliation RULE is scope: the question asks for the CITY-PROPER (administrative) census
  population, not the metropolitan area / CMA / GTA. Do not grab the metro figure; do not average.

Ground truth (verified against live English Wikipedia https://en.wikipedia.org/wiki/Toronto,
2026-07-10):
  * Infobox (2021 census): City 2,794,356; CMA 6,202,225; GTA 6,712,341.
  * Margin/distinctness: correct-value token regex ``2[,\\s]?794[,\\s]?356`` matches only the
    city-proper figure, never the metro 6,202,225 / 6,712,341 or their average (~4.5M). The
    figures share no leading digits, so one noisy extraction of a metro figure fails the keystone
    (bimodal) rather than silently flipping it.

Leak-resistant: the precise city-proper 2,794,356 is a page-only census figure a naive/parametric
agent replaces with a rounded metro "~6 million." No answer string appears anywhere the executor
reads.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/Toronto"
_SLUG_RX = r"wiki/toronto"

# CORRECT (keystone): City of Toronto (city proper), 2021 census 2,794,356.
_CORRECT_RE = re.compile(r"2[,\s]?794[,\s]?356")
# WRONG (metro scope): CMA 6,202,225 OR GTA 6,712,341.
_WRONG_RE = re.compile(r"6[,\s]?202[,\s]?225|6[,\s]?712[,\s]?341")

# Rule marker: city proper / administrative vs metro / CMA / GTA.
_RULE_MARK = (
    r"(?:city\s+proper|city\s+of\s+toronto|administrativ\w+|municipal\w+|proper|"
    r"metro\w*|metropolitan|census\s+metropolitan|c\.?m\.?a\.?|greater\s+toronto|g\.?t\.?a\.?|"
    r"agglomerat\w+|urban\s+area|not\s+the\s+metro|surrounding)"
)
_WRONG_TOK = r"(?:6[,\s]?202[,\s]?225|6[,\s]?712[,\s]?341)"
_IDENT_RE = re.compile(
    rf"{_RULE_MARK}[^.]{{0,90}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,90}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "133",
        "test_name": "Tier 5 targeted: conflicting-source reconciliation (Toronto — city proper vs metropolitan area)",
        "difficulty_level": "9/10",
        "category": "adaptive_targeted",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker making a SOURCE-TRUST decision with FEW reads (do not fan out; do "
        "not answer from memory). Two population figures circulate for Toronto (2021 Canadian "
        "census):\n\n"
        '  CLAIM (the biggest figure quoted): "Toronto\'s population is about 6.2 million."\n\n'
        "Open the authoritative Wikipedia article for Toronto and read the population figures. Apply "
        "this rule: the question asks for the CITY-PROPER (administrative City of Toronto) census "
        "population, NOT the metropolitan area (Census Metropolitan Area / Greater Toronto Area). "
        "The ~6.2 million figure is the metro-area population. Do NOT grab the biggest number and do "
        "NOT average. Then report:\n"
        "  (a) the CORRECT city-proper (City of Toronto) 2021 census population — the keystone;\n"
        "  (b) identify the ~6.2 million figure as the metropolitan-area population (a different "
        "scope).\n\n"
        "Cite the exact authoritative source URL you read the figures from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct city-proper (City of Toronto) 2021 census population — the keystone",
        "The ~6.2 million (or GTA) figure identified as the metropolitan-area population (different scope)",
        "A statement of the rule (city-proper administrative census, not metro/CMA/GTA)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 reads (the city-proper figure must be read, not recalled)",
        "Correct city-proper population reported (2,794,356)",
        "Metro-area figure (6,202,225 / 6,712,341) identified as a different scope (not chosen, not averaged)",
        "Authoritative Toronto page cited",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(_CORRECT_RE.search(_primary_text(result)))


def _read_evidence(result: Dict[str, Any], observability: Dict[str, Any]) -> bool:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return n > 0 or bool(re.search(_SLUG_RX, _all_text(result).lower()))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=2; the city-proper figure must be read, not recalled)"}


def validate_keystone_population(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the city-proper 2021 census population (2,794,356). Rejects the metro
    6,202,225 / 6,712,341 and the average (~4.5M); the figures share no leading digits."""
    passed = _keystone_ok(result)
    return {"check": "keystone_population", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "City-proper population 2,794,356 present" if passed
                      else "City-proper population (2,794,356, 2021 census) missing/incorrect"}


def validate_reconciliation_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic (retained even when the final pick is wrong): did the agent
    surface BOTH the city-proper figure AND a metro figure — proving it read the source rather than
    grabbing one. Gated ONLY on read-evidence so a parametric answer banks nothing."""
    if not _read_evidence(result, observability):
        return {"check": "reconciliation_coverage", "passed": False, "score": 0.0,
                "reason": "No read-evidence (no visit, no citation) -> coverage not credited"}
    text = _all_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "reconciliation_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"city_proper_value={has_correct}, metro_value={has_wrong}"}


def validate_identifies_correct_source(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the agent FLAGS the ~6.2M (or GTA 6,712,341) figure as the metropolitan-area
    population (a different scope). Short-circuits to 0 without the keystone or read-evidence."""
    if not _keystone_ok(result):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> scope reconciliation not credited"}
    if not _read_evidence(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "No read-evidence -> reconciliation not credited"}
    flagged = bool(_IDENT_RE.search(_all_text(result)))
    return {"check": "identifies_correct_source", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"metro figure flagged as metropolitan-area scope={flagged}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: authoritative Toronto page cited. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(_SLUG_RX, _all_text(result).lower()))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative Toronto page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_population,
        validate_reconciliation_coverage,
        validate_identifies_correct_source,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold (2 -> 1) for the graph_compiled arm.

    Two parallel visit leaves — the METRO-area figure and the CITY-PROPER figure — plus a dependent
    verify leaf that applies the scope rule (``depends_on`` both, templating ``{metro}`` and
    ``{city_proper}``). Leak-free: names each figure by its SCOPE, never its number; the correct
    2,794,356 never appears."""
    return {
        "leaves": [
            {
                "id": "metro",
                "instruction": (
                    "Open the authoritative Wikipedia article for Toronto. Read the METROPOLITAN-AREA "
                    "population at the 2021 census (the Census Metropolitan Area / Greater Toronto "
                    "Area figure — the larger number covering surrounding regions). Do NOT answer "
                    "from memory; report the exact figure and the source URL."
                ),
                "expect": "METRO-area population (2021 census) — source URL",
                "depends_on": [],
            },
            {
                "id": "city_proper",
                "instruction": (
                    "On the authoritative Wikipedia article for Toronto, read the CITY-PROPER "
                    "population — the administrative City of Toronto figure at the 2021 census (the "
                    "'City' line in the infobox, NOT the metropolitan area). Do NOT answer from "
                    "memory; report the exact figure and the source URL."
                ),
                "expect": "CITY-PROPER population (2021 census) — source URL",
                "depends_on": [],
            },
            {
                "id": "reconcile",
                "action": "verify",
                "details": {
                    "claim": "Toronto's population is about 6.2 million.",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Reconcile the two figures using this rule: the question asks for the "
                    "CITY-PROPER (administrative) census population, NOT the metropolitan area; do "
                    "not average them. Metro value gathered: {metro}. City-proper value gathered: "
                    "{city_proper}. Report the correct city-proper population, identify the metro "
                    "figure as a different scope, and cite the source URL."
                ),
                "expect": "Correct city-proper population + metro figure flagged as different scope — source URL",
                "depends_on": ["metro", "city_proper"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT city-proper (City of Toronto) 2021 census population — the "
            "keystone — and (b) identify the ~6.2 million (or Greater Toronto Area) figure as the "
            "metropolitan-area population (a different scope). Do not grab the biggest number and do "
            "not average. Cite the authoritative source URL."
        ),
    }
