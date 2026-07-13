"""
Test 129: Tier 5 (adaptive-targeted) — CONFLICTING-SOURCE RECONCILIATION (trust decision).
Level: integration   Weight: long   Difficulty: 9/10

LOW-CONTEXT source-trust decision task (Bucket B). A good ADAPTIVE agent must reconcile two
reputable-looking "height of the Willis Tower" figures by applying a checkable SCOPE rule, on a
narrow golden path of 2-3 reads — NOT by breadth, NOT by averaging, NOT by grabbing the tallest
number it sees.

The quantity: the height of the Willis Tower (formerly Sears Tower), Chicago.
  * WRONG (biggest-number bait): the pinnacle / tip height including the broadcast antennas,
    1,729 ft (527.0 m) after the 2000 antenna extension. A naive agent grabs the largest figure.
  * CORRECT (keystone): the official CTBUH ARCHITECTURAL height (ground to structural/architectural
    top, excluding antennas), 1,451 ft (442 m) — the figure used to rank the building.
  The reconciliation RULE is scope: the OFFICIAL CTBUH architectural height (antennas excluded) is
  the building's ranked height — not the antenna tip, and not an average of the two.

Ground truth (verified against live English Wikipedia https://en.wikipedia.org/wiki/Willis_Tower,
2026-07-10):
  * Infobox: architectural 1,451 ft (442 m); tip 1,729 ft (527.0 m) after antenna extension in
    2000; top floor / roof 1,354 ft (413 m).
  * Margin/distinctness: correct-value token regex ``1[,\\s]?451|\\b442\\b`` matches only the
    architectural height, never the tip 1,729 / 527, the roof 1,354 / 413, or the averages
    (1,590 ft / 484.5 m). The families share no digits, so a single noisy extraction of the tip
    fails the keystone (bimodal) rather than silently flipping it.

Leak-resistant: the exact architectural 1,451 ft / 442 m is a page/CTBUH figure; a parametric
model reaches for the famous tip-with-antenna "1,729 ft". No answer string appears anywhere the
executor reads.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/Willis_Tower"
_SLUG_RX = r"wiki/willis_tower"

# CORRECT (keystone): architectural height 1,451 ft OR 442 m.
_CORRECT_RE = re.compile(r"1[,\s]?451|\b442\b")
# WRONG (biggest-number bait): tip/antenna 1,729 ft OR 527 m.
_WRONG_RE = re.compile(r"1[,\s]?729|\b527\b")

# Rule marker: architectural / official / excludes antenna vs tip.
_RULE_MARK = (
    r"(?:antenna\w*|tip|pinnacle|spire|mast|broadcast|architectural|ctbuh|official|"
    r"excludes?|excluding|includes?|including|structural|not\s+the\s+(?:tip|antenna))"
)
_WRONG_TOK = r"(?:1[,\s]?729|\b527\b)"
_IDENT_RE = re.compile(
    rf"{_RULE_MARK}[^.]{{0,80}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,80}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "129",
        "test_name": "Tier 5 targeted: conflicting-source reconciliation (Willis Tower — architectural vs antenna tip)",
        "difficulty_level": "9/10",
        "category": "adaptive_targeted",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker making a SOURCE-TRUST decision with FEW reads (do not fan out; do "
        "not answer from memory). Sources quote two different 'heights' for the Willis Tower "
        "(formerly Sears Tower) in Chicago:\n\n"
        '  CLAIM (the biggest figure quoted): "The Willis Tower is 1,729 ft (527 m) tall."\n\n'
        "Open the authoritative Wikipedia article for the Willis Tower and read its height figures. "
        "Apply this rule: the OFFICIAL architectural height (CTBUH — ground to architectural top, "
        "EXCLUDING broadcast antennas) is the building's ranked height. The 1,729 ft figure is the "
        "antenna TIP height, not the architectural height. Do NOT grab the biggest number and do "
        "NOT average the two. Then report:\n"
        "  (a) the CORRECT architectural height of the Willis Tower (ft and/or m) — the keystone;\n"
        "  (b) identify the 1,729 ft / 527 m figure as the antenna-TIP height (not the architectural "
        "height).\n\n"
        "Cite the exact authoritative source URL you read the figures from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct architectural height of the Willis Tower — the keystone",
        "The 1,729 ft / 527 m figure identified as the antenna-tip height (not architectural)",
        "A statement of the rule (official CTBUH architectural height excludes antennas)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 reads (the architectural figure must be read, not recalled)",
        "Correct architectural height reported (1,451 ft / 442 m)",
        "Antenna-tip 1,729 ft / 527 m identified as a different scope (not chosen, not averaged)",
        "Authoritative Willis Tower page cited",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(_CORRECT_RE.search(_primary_text(result)))


def _read_evidence(result: Dict[str, Any], observability: Dict[str, Any]) -> bool:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return n > 0 or bool(re.search(_SLUG_RX, _all_text(result).lower()))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=2; the architectural figure must be read, not recalled)"}


def validate_keystone_height(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the architectural height (1,451 ft / 442 m). Rejects the antenna-tip
    1,729 ft / 527 m, the roof 1,354 / 413, and the averages (the token families share no digits)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Architectural height 1,451 ft / 442 m present" if passed
                      else "Architectural height (1,451 ft / 442 m) missing/incorrect"}


def validate_reconciliation_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic (retained even when the final pick is wrong): did the agent
    surface BOTH figures — the architectural height AND the antenna-tip height — proving it read the
    source rather than grabbing one. Gated ONLY on read-evidence so a parametric answer banks
    nothing."""
    if not _read_evidence(result, observability):
        return {"check": "reconciliation_coverage", "passed": False, "score": 0.0,
                "reason": "No read-evidence (no visit, no citation) -> coverage not credited"}
    text = _all_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "reconciliation_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"architectural_value={has_correct}, tip_value={has_wrong}"}


def validate_identifies_correct_source(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the agent FLAGS 1,729 ft / 527 m as the antenna-TIP (not architectural)
    height. Short-circuits to 0 without the keystone or read-evidence."""
    if not _keystone_ok(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> scope reconciliation not credited"}
    if not _read_evidence(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "No read-evidence -> reconciliation not credited"}
    flagged = bool(_IDENT_RE.search(_all_text(result)))
    return {"check": "identifies_correct_source", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"1,729 ft / 527 m flagged as antenna-tip scope={flagged}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: authoritative Willis Tower page cited. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(_SLUG_RX, _all_text(result).lower()))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative Willis Tower page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_height,
        validate_reconciliation_coverage,
        validate_identifies_correct_source,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold (2 -> 1) for the graph_compiled arm.

    Two parallel visit leaves — the antenna-TIP height and the OFFICIAL ARCHITECTURAL height — plus
    a dependent verify leaf that applies the scope rule (``depends_on`` both, templating ``{tip}``
    and ``{architectural}``). Leak-free: names each figure by its SCOPE, never its number; the
    correct 1,451 ft / 442 m never appears."""
    return {
        "leaves": [
            {
                "id": "tip",
                "instruction": (
                    "Open the authoritative Wikipedia article for the Willis Tower (Chicago). Read "
                    "the TIP / pinnacle height that INCLUDES the broadcast antennas (the tallest "
                    "figure, after the 2000 antenna extension). Do NOT answer from memory; report "
                    "the exact figure with units and the source URL."
                ),
                "expect": "Antenna-TIP height with units — source URL",
                "depends_on": [],
            },
            {
                "id": "architectural",
                "instruction": (
                    "On the authoritative Wikipedia article for the Willis Tower, read the OFFICIAL "
                    "ARCHITECTURAL height (CTBUH: ground to architectural top, EXCLUDING the "
                    "broadcast antennas). Do NOT answer from memory; report the exact figure with "
                    "units and the source URL."
                ),
                "expect": "Official ARCHITECTURAL height with units — source URL",
                "depends_on": [],
            },
            {
                "id": "reconcile",
                "action": "verify",
                "details": {
                    "claim": "The Willis Tower is 1,729 ft (527 m) tall.",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Reconcile the two figures using this rule: the OFFICIAL architectural height "
                    "(antennas excluded) is the building's ranked height, NOT the antenna tip; do "
                    "not average them. Tip value gathered: {tip}. Architectural value gathered: "
                    "{architectural}. Report the correct architectural height, identify the tip "
                    "figure as a different scope, and cite the source URL."
                ),
                "expect": "Correct architectural height + tip figure flagged as different scope — source URL",
                "depends_on": ["tip", "architectural"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT official architectural height of the Willis Tower (ft and/or m) "
            "— the keystone — and (b) identify the 1,729 ft / 527 m figure as the antenna-tip height "
            "(a different scope, not the architectural height). Do not grab the biggest number and "
            "do not average. Cite the authoritative source URL."
        ),
    }
