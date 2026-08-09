"""
Test 131: Tier 5 (adaptive-targeted) — CONFLICTING-SOURCE RECONCILIATION (trust decision).
Level: integration   Weight: long   Difficulty: 9/10

LOW-CONTEXT source-trust decision task (Bucket B). A good ADAPTIVE agent must reconcile two
reputable-looking "height of One World Trade Center" figures by applying a checkable SCOPE rule,
on a narrow golden path of 2-3 reads — NOT by breadth, NOT by averaging, NOT by grabbing whichever
number appears first.

The quantity: the height of One World Trade Center (1 WTC / Freedom Tower), New York City.
  * WRONG (roof-only scope): the roof height, 1,368 ft (417 m) — the top of the occupied structure,
    a number an agent may grab as "the height of the building" and which happens to match the
    original Twin Towers' roof.
  * CORRECT (keystone): the official CTBUH ARCHITECTURAL height INCLUDING the spire, 1,776 ft
    (541.3 m) — the ranked height (CTBUH ruled the mast is a permanent architectural spire, so it
    counts).
  The reconciliation RULE is scope: the OFFICIAL CTBUH architectural height COUNTS the spire; the
  roof height (1,368 ft) is a different, smaller scope. Do not report the roof height; do not
  average the two.

Ground truth (verified against live English Wikipedia
https://en.wikipedia.org/wiki/One_World_Trade_Center, 2026-07-10):
  * Infobox: architectural 1,776 ft (541.3 m); roof 1,368 ft (417.0 m); top floor / observatory
    1,268 ft (386.5 m).
  * Body: on 12 Nov 2013 the CTBUH ruled the mast is a spire (a permanent part of the architecture),
    so the full 1,776 ft counts as the official architectural height; without that ruling the tower
    would measure only 1,368 ft to its roof.
  * Margin/distinctness: correct-value token regex ``1[,\\s]?776|\\b541\\b`` matches only the
    architectural height, never the roof 1,368 / 417, the observatory 1,268 / 386, or the averages
    (1,572 ft / 479 m). The families share no digits (bimodal, not silently flippable).

Leak-resistant: the executor never sees the answer; the CTBUH spire ruling and the exact 541.3 m
are page facts a naive agent must read rather than recall.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/One_World_Trade_Center"
_SLUG_RX = r"wiki/one_world_trade_center"

# CORRECT (keystone): architectural height INCLUDING the spire, 1,776 ft OR 541 m.
_CORRECT_RE = re.compile(r"1[,\s]?776|\b541\b")
# WRONG (roof scope): 1,368 ft OR 417 m.
_WRONG_RE = re.compile(r"1[,\s]?368|\b417\b")

# Rule marker: architectural / spire counts vs roof / occupied top.
_RULE_MARK = (
    r"(?:roof|architectural|spire|mast|ctbuh|official|includ\w+|exclud\w+|counts?|"
    r"occupied|structural|not\s+the\s+(?:roof|spire)|to\s+the\s+roof|ranked)"
)
_WRONG_TOK = r"(?:1[,\s]?368|\b417\b)"
_IDENT_RE = re.compile(
    rf"{_RULE_MARK}[^.]{{0,80}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,80}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "131",
        "test_name": "Tier 5 targeted: conflicting-source reconciliation (One World Trade Center — architectural spire vs roof)",
        "difficulty_level": "9/10",
        "category": "adaptive_targeted",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker making a SOURCE-TRUST decision with FEW reads (do not fan out; do "
        "not answer from memory). Two 'heights' circulate for One World Trade Center (1 WTC, New "
        "York City):\n\n"
        '  CLAIM (a figure quoted as the building\'s height): "One World Trade Center is 1,368 ft '
        '(417 m) tall."\n\n'
        "Open the authoritative Wikipedia article for One World Trade Center and read its height "
        "figures. Apply this rule: the OFFICIAL CTBUH ARCHITECTURAL height COUNTS the spire (the "
        "CTBUH ruled the mast is a permanent architectural spire), so it is the ranked height. The "
        "1,368 ft figure is only the ROOF height (a different, smaller scope). Do NOT report the "
        "roof height and do NOT average the two. Then report:\n"
        "  (a) the CORRECT official architectural height of 1 WTC (ft and/or m) — the keystone;\n"
        "  (b) identify the 1,368 ft / 417 m figure as the ROOF height (not the architectural "
        "height).\n\n"
        "Cite the exact authoritative source URL you read the figures from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct official architectural height of 1 WTC (including the spire) — the keystone",
        "The 1,368 ft / 417 m figure identified as the roof height (not architectural)",
        "A statement of the rule (CTBUH architectural height counts the spire)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 reads (the architectural figure must be read, not recalled)",
        "Correct architectural height reported (1,776 ft / 541.3 m)",
        "Roof height 1,368 ft / 417 m identified as a different scope (not chosen, not averaged)",
        "Authoritative One World Trade Center page cited",
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
    """KEYSTONE (hard 0/1): the architectural height incl. spire (1,776 ft / 541.3 m). Rejects the
    roof 1,368 ft / 417 m, the observatory 1,268 / 386, and the averages (families share no digits)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Architectural height 1,776 ft / 541.3 m present" if passed
                      else "Architectural height (1,776 ft / 541.3 m, incl. spire) missing/incorrect"}


def validate_reconciliation_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic (retained even when the final pick is wrong): did the agent
    surface BOTH figures — the architectural height AND the roof height — proving it read the source
    rather than grabbing one. Gated ONLY on read-evidence so a parametric answer banks nothing."""
    if not _read_evidence(result, observability):
        return {"check": "reconciliation_coverage", "passed": False, "score": 0.0,
                "reason": "No read-evidence (no visit, no citation) -> coverage not credited"}
    text = _all_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "reconciliation_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"architectural_value={has_correct}, roof_value={has_wrong}"}


def validate_identifies_correct_source(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the agent FLAGS 1,368 ft / 417 m as the ROOF height (not architectural).
    Short-circuits to 0 without the keystone or read-evidence."""
    if not _keystone_ok(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> scope reconciliation not credited"}
    if not _read_evidence(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "No read-evidence -> reconciliation not credited"}
    flagged = bool(_IDENT_RE.search(_all_text(result)))
    return {"check": "identifies_correct_source", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"1,368 ft / 417 m flagged as roof scope={flagged}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: authoritative 1 WTC page cited. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(_SLUG_RX, _all_text(result).lower()))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative One World Trade Center page cited={cited}"}


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

    Two parallel visit leaves — the ROOF height and the OFFICIAL ARCHITECTURAL height (incl. spire) —
    plus a dependent verify leaf that applies the scope rule (``depends_on`` both, templating
    ``{roof}`` and ``{architectural}``). Leak-free: names each figure by its SCOPE, never its
    number; the correct 1,776 ft / 541.3 m never appears."""
    return {
        "leaves": [
            {
                "id": "roof",
                "instruction": (
                    "Open the authoritative Wikipedia article for One World Trade Center (New York "
                    "City). Read the ROOF height (the top of the occupied structure, EXCLUDING the "
                    "spire). Do NOT answer from memory; report the exact figure with units and the "
                    "source URL."
                ),
                "expect": "ROOF height with units — source URL",
                "depends_on": [],
            },
            {
                "id": "architectural",
                "instruction": (
                    "On the authoritative Wikipedia article for One World Trade Center, read the "
                    "OFFICIAL CTBUH ARCHITECTURAL height that INCLUDES the spire (the ranked height "
                    "in the infobox). Do NOT answer from memory; report the exact figure with units "
                    "and the source URL."
                ),
                "expect": "Official ARCHITECTURAL height (incl. spire) with units — source URL",
                "depends_on": [],
            },
            {
                "id": "reconcile",
                "action": "verify",
                "details": {
                    "claim": "One World Trade Center is 1,368 ft (417 m) tall.",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Reconcile the two figures using this rule: the OFFICIAL CTBUH architectural "
                    "height COUNTS the spire and is the ranked height, NOT the roof; do not average "
                    "them. Roof value gathered: {roof}. Architectural value gathered: "
                    "{architectural}. Report the correct architectural height, identify the roof "
                    "figure as a different scope, and cite the source URL."
                ),
                "expect": "Correct architectural height + roof figure flagged as different scope — source URL",
                "depends_on": ["roof", "architectural"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT official architectural height of One World Trade Center (ft "
            "and/or m, including the spire) — the keystone — and (b) identify the 1,368 ft / 417 m "
            "figure as the roof height (a different, smaller scope). Do not report the roof height "
            "and do not average. Cite the authoritative source URL."
        ),
    }
