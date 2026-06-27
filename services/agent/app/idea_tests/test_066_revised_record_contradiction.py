"""
Test 066: Cross-source contradiction — popular-wrong vs authoritative revised record
Level: integration   Weight: long   Difficulty: 8/10

A single quantity whose COMMONLY-CITED value is an under-count (and effectively WRONG as a
*total*) while the AUTHORITATIVE source carries a precise, comprehensively-surveyed value. The
agent must report the CORRECT surveyed value AND name the popular value as outdated/incomplete,
resolving the contradiction by consulting the primary source instead of memory. This is the
``verify`` axis (clone of test 042 faithfulness/contradiction and test 056): a cheap model that
answers parametrically confidently emits the iconic popular number; a structured agent
cross-checks the popular claim against the authoritative page and corrects it.

The fact: the total length of the Great Wall of China.
  * Commonly-cited / popular value (WRONG as a *total*, outdated): ~8,850 km (5,500 miles) — the
    figure for the Ming-dynasty wall alone, endlessly repeated as "the length of the Great Wall"
    and the default a weak model recalls.
  * Authoritative / correct comprehensive value (KEYSTONE): 21,196.18 km (13,170.70 mi) —
    established by China's National Cultural Heritage Administration 2012 comprehensive
    archaeological survey of ALL dynasties' walls; this is the figure in the live Wikipedia
    infobox. The precise surveyed digits (21,196 km / 13,170 mi) are page-only — a weak
    parametric model recalls the round popular 5,500 mi / 8,850 km, or a vague "about 21,000 km",
    not the surveyed 21,196.18.

Scoring is gated/bimodal: the KEYSTONE is the authoritative-comprehensive value (the precise
surveyed figure the popular value lacks). The secondary checks (identifies the outdated popular
value, cites the authoritative URL) short-circuit to 0 when the keystone is absent.

Anti-parametric gate (Route B tightening, 2026-06-27): Both the contradiction-coverage and the
wrong-value-identification secondaries are now GATED on evidence of actually reading the source
(visit_count > 0 OR the authoritative URL cited in the answer). This prevents a model that
recalls both figures from memory (0 page visits) from banking those checks without visiting the
authoritative page. A purely parametric answer (0 visits, no citation) can score at most 1/5 on
the keystone alone — ≤0.20 — even if it knows all the facts from training data.

Ground truth (verified against live English Wikipedia, 2026-06-26):
  * Authoritative (correct): https://en.wikipedia.org/wiki/Great_Wall_of_China — infobox
    ``size = {{cvt|21196.18|km|mi}}`` -> rendered "21,196.18 km (13,170.70 mi)"; body: "...the
    walls and trenches spanning a total length of 21,196.18 km (13,170.70 mi)" (2012 National
    Cultural Heritage Administration comprehensive survey). The same page documents the popular
    subset: "the Ming Great Wall measures 8,850 km (5,500 mi)".
  * Margin: 21,196.18 km vs 8,850 km differ by ~12,346 km (>2.39x); the figures share NO tokens
    (21,196 / 13,170 vs 8,850 / 5,500), so a single noisy extraction of the popular figure fails
    the keystone (correct, bimodal) rather than silently flipping it. A vague rounded recall
    ("about 21,000 km" / "13,000 miles") also fails the keystone — only the surveyed 21,196 /
    13,170 passes.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/Great_Wall_of_China"

# Correct / comprehensive authoritative value: 21,196.18 km OR 13,170.70 mi. Requiring the precise
# surveyed thousands+hundreds (21,196 / 13,170) is the page-only crux: it distinguishes the
# surveyed total from the popular 8,850 km / 5,500 mi AND from a vague rounded "about 21,000 km /
# 13,000 miles" (both of which lack these digits).
_CORRECT_RE = re.compile(r"21[,\s]?196|13[,\s]?170")

# Wrong / outdated popular value: the iconic ~8,850 km (5,500 mi) Ming-only figure, repeated as
# THE length of the Great Wall.
_WRONG_RE = re.compile(r"\b8[,\s]?85[01]\b|\b5[,\s]?500\b")

# A "this value is outdated / commonly-cited but superseded / only a subset" marker, for the gated
# secondary that requires the agent to FLAG the popular figure (not merely echo it). Includes the
# Great-Wall-specific framing (only the Ming wall / incomplete / underestimate).
_OUTDATED_MARK = (
    r"(?:outdated|out\s*of\s*date|old(?:er)?|previous(?:ly)?|prior|former(?:ly)?|supersed\w+|"
    r"commonly|widely|long[-\s]?(?:cited|standing|held|believed)|no\s+longer|incorrect|wrong|"
    r"mistaken|misconception|myth|underestimat\w+|incomplete|partial|only|just|ming|earlier|"
    r"subset|used\s+to|often\s+(?:cited|quoted))"
)
_WRONG_TOK = r"(?:\b8[,\s]?85[01]\b|\b5[,\s]?500\b)"
# Proximity (newline-tolerant via [^.], sentence-bounded): the outdated marker within a sentence of
# the wrong value, in either order.
_IDENT_RE = re.compile(
    rf"{_OUTDATED_MARK}[^.]{{0,80}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,80}}{_OUTDATED_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "066",
        "test_name": "Cross-source contradiction — popular-wrong vs authoritative revised record",
        "difficulty_level": "8/10",
        "category": "Cross-source contradiction / verify",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker. A popular figure is circulating that you must VERIFY against the "
        "AUTHORITATIVE source — do NOT answer from memory, because the commonly-cited value may be "
        "OUTDATED or INCOMPLETE:\n\n"
        '  CLAIM (commonly cited): "The Great Wall of China is about 8,850 km (5,500 miles) '
        'long."\n\n'
        "Consult the authoritative source for the Great Wall of China's length (the most "
        "comprehensive official archaeological survey, as reflected in the authoritative article) "
        "and then:\n"
        "  (a) Report the CORRECT total length of the Great Wall as established by the most "
        "comprehensive authoritative survey (this is your primary answer).\n"
        "  (b) Identify the commonly-cited figure that is now OUTDATED / an underestimate, and "
        "explain the discrepancy (what the comprehensive survey counted that the popular figure "
        "omits).\n\n"
        "Cite the exact authoritative source URL you read the total length from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct comprehensive total length of the Great Wall (the keystone / primary answer)",
        "The commonly-cited figure now identified as outdated / an underestimate",
        "Explanation of the discrepancy (the 2012 comprehensive all-dynasties survey vs Ming-only)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 sources/pages consulted (the authoritative figure must be read, not recalled)",
        "Correct comprehensive total length reported (21,196.18 km / 13,170.70 mi)",
        "Commonly-cited outdated value (~8,850 km / 5,500 mi) identified as a Ming-only underestimate",
        "Authoritative source URL cited",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE: the authoritative comprehensive total length (21,196.18 km / 13,170.70 mi)."""
    return bool(_CORRECT_RE.search(extract_final_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=2; the authoritative figure must be read, not recalled)"}


def validate_keystone_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: authoritative comprehensive total length present. Hard 0/1."""
    passed = _keystone_ok(result)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Comprehensive total length (21,196.18 km / 13,170.70 mi) present" if passed
                      else "Authoritative comprehensive total length (21,196.18 km / 13,170.70 mi) missing"}


def validate_contradiction_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED coverage diagnostic: how much of the CONTRADICTION was actually surfaced.

    Counts whether BOTH sides were gathered — the authoritative comprehensive value AND the
    commonly-cited Ming-only value. Gated on evidence of reading the authoritative source
    (visit_count > 0 OR the authoritative URL cited): a purely parametric answer (0 visits,
    no citation) that happens to recall both figures from training data scores 0, preventing
    memory-only answers from banking this check.
    """
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    cited = bool(re.search(r"wiki/great_wall_of_china", extract_final_text(result).lower()))
    if n == 0 and not cited:
        return {"check": "contradiction_coverage", "passed": False, "score": 0.0,
                "reason": "No visits and no authoritative citation — coverage not credited without reading the source"}
    text = extract_final_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "contradiction_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"comprehensive_value={has_correct}, popular_value={has_wrong}"}


def validate_identifies_wrong_value(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated): the agent identifies the commonly-cited ~8,850 km / 5,500 mi as the
    OUTDATED / incomplete (Ming-only) figure (flagged near an outdated/subset marker), not merely
    echoed. Short-circuits to 0 when the keystone is absent OR when no evidence of reading the
    authoritative source is present (visit_count > 0 OR authoritative URL cited). The dual gate
    ensures a model that recalls both figures from parametric memory — but never visits the page —
    cannot claim credit for identifying the popular value as wrong."""
    if not _keystone_ok(result):
        return {"check": "identifies_wrong_value", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> wrong-value identification not credited"}
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    cited = bool(re.search(r"wiki/great_wall_of_china", extract_final_text(result).lower()))
    if n == 0 and not cited:
        return {"check": "identifies_wrong_value", "passed": False, "score": 0.0,
                "reason": "No visits and no authoritative citation -> wrong-value identification not credited without reading the source"}
    flagged = bool(_IDENT_RE.search(extract_final_text(result)))
    return {"check": "identifies_wrong_value", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"outdated popular value (~8,850 km / 5,500 mi) flagged={flagged}"}


def validate_authoritative_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated): the authoritative Great Wall page is cited. Short-circuits to 0 when the
    keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "authoritative_citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(r"wiki/great_wall_of_china", extract_final_text(result).lower()))
    return {"check": "authoritative_citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative Great Wall of China page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_length,
        validate_contradiction_coverage,
        validate_identifies_wrong_value,
        validate_authoritative_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold for the ``graph_compiled`` variant.

    A parallel pair of visit leaves (the COMPREHENSIVE surveyed total length and the Ming-only
    subset length) plus a dependent ``verify`` leaf that cross-checks the popular CLAIM against
    both gathered facts (``depends_on`` the two visit leaves, templating ``{comprehensive_total}``
    and ``{ming_only}``). The verify leaf carries ``action: "verify"`` and ``details.claim`` (the
    popular claim) per the ``verify`` action contract; the executor normalizes leaves to
    id/instruction/expect/depends_on, so the self-contained ``instruction`` also drives the generic
    leaf loop.

    Leak-free: the popular CLAIM (~8,850 km / 5,500 mi — the value to fact-check) is present by
    design, but the authoritative-comprehensive value (21,196.18 km / 13,170.70 mi) appears
    NOWHERE — the cheap model must read it off the page.
    """
    return {
        "leaves": [
            {
                "id": "comprehensive_total",
                "instruction": (
                    "Open the AUTHORITATIVE Wikipedia article for the Great Wall of China and read "
                    "the TOTAL length of the wall established by the most comprehensive official "
                    "archaeological survey — the all-sections, all-dynasties figure in the infobox "
                    "/ lead (NOT the figure for any single dynasty). Do NOT answer from memory; "
                    "report the exact figure with units."
                ),
                "expect": "COMPREHENSIVE total length with units — source URL",
                "depends_on": [],
            },
            {
                "id": "ming_only",
                "instruction": (
                    "On the AUTHORITATIVE Wikipedia article for the Great Wall of China, read the "
                    "length attributed to ONLY the Ming-dynasty wall (the shorter, commonly-cited "
                    "figure often quoted as 'the length of the Great Wall'). Do NOT answer from "
                    "memory; report the exact figure with units."
                ),
                "expect": "MING-ONLY wall length with units — source URL",
                "depends_on": [],
            },
            {
                "id": "verify_popular",
                "action": "verify",
                "details": {
                    "claim": "The Great Wall of China is about 8,850 km (5,500 miles) long.",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Fact-check this commonly-cited CLAIM against the evidence already gathered: "
                    '"The Great Wall of China is about 8,850 km (5,500 miles) long." '
                    "The COMPREHENSIVE surveyed total length gathered is: {comprehensive_total}. "
                    "The MING-ONLY wall length is: {ming_only}. Decide whether the claim states the "
                    "comprehensive total or only a subset, and give a verdict (TRUE / OUTDATED). "
                    "Report the correct comprehensive total length, the figure that is now an "
                    "underestimate, and cite the authoritative source URL."
                ),
                "expect": "VERDICT (TRUE/OUTDATED) + comprehensive total + underestimate figure — source URL",
                "depends_on": ["comprehensive_total", "ming_only"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT total length of the Great Wall of China as established by the "
            "most comprehensive authoritative survey (this is the primary answer / keystone), and "
            "(b) identify the commonly-cited figure that is now OUTDATED / an underestimate, "
            "briefly explaining the discrepancy (what the comprehensive survey counted that the "
            "popular figure omits). Cite the authoritative source URL for the total length."
        ),
    }
