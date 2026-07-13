"""
Test 128: Tier 5 (adaptive-targeted) — CONFLICTING-SOURCE RECONCILIATION (trust decision).
Level: integration   Weight: long   Difficulty: 9/10

LOW-CONTEXT source-trust decision task (Bucket B). A good ADAPTIVE agent must reconcile two
reputable-looking figures for the SAME quantity by applying a checkable rule, on a narrow golden
path of 2-3 reads — NOT by breadth, and NOT by averaging or grabbing the first hit.

The quantity: the diameter of the dwarf planet Pluto, as measured by NASA's New Horizons flyby.
  * WRONG (first-hit / preliminary): the FIRST value NASA announced on 13 July 2015, ~2,370 km —
    a widely-reproduced press-release figure that was later refined. A naive/early-stopping agent
    reports this first number.
  * CORRECT (keystone): the REFINED FINAL value from the New Horizons Radio Science Experiment
    (REX) radio-occultation measurement, 2,376.6 km diameter (mean radius 1,188.3 km) — the figure
    in the live Wikipedia infobox.
  The reconciliation RULE is recency/refinement: the refined FINAL measurement supersedes the
  PRELIMINARY press-release figure. Do not take the first hit; do not average the two.

Ground truth (verified against live English Wikipedia https://en.wikipedia.org/wiki/Pluto,
2026-07-10):
  * Infobox: mean radius 1,188.3 +/- 0.8 km  ->  diameter 2,376.6 +/- 1.6 km.
  * Body: "...determined Pluto's diameter to be 2,370 km (1,473 mi), which was later revised to be
    2,372 km ... on July 24, and later to 2374 +/- 8 km. Using ... the New Horizons Radio Science
    Experiment (REX), the diameter was found to be 2,376.6 +/- 1.6 km."
  * Margin/distinctness: the correct-value token regex ``2[,\\s]?37[67]`` (2,376 / 2,377) matches
    the refined figure but NEVER the preliminary 2,370 / 2,372 / 2,374, and NEVER the average
    2,373; the two families of digits share no token, so one noisy extraction of the first-hit
    figure fails the keystone (bimodal) rather than silently flipping it. A vague "about 2,370 km"
    also fails — only the refined 2,376.6 (or radius 1,188.3) passes.

Leak-resistant: the refined REX diameter is a page-only oddity; a parametric model reaches for the
round preliminary "2,370 km". No answer string appears anywhere the executor reads.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/Pluto"
_SLUG_RX = r"wiki/pluto"

# CORRECT (keystone): refined REX diameter 2,376.6 km OR mean radius 1,188.3 km.
_CORRECT_RE = re.compile(r"2[,\s]?37[67]|1[,\s]?188")
# WRONG (first-hit / preliminary): 2,370 / 2,372 / 2,374 km.
_WRONG_RE = re.compile(r"2[,\s]?37[024]")

# Rule marker: refined/final supersedes preliminary/first.
_RULE_MARK = (
    r"(?:refin\w+|final|revis\w+|updated|supersed\w+|later|correct\w+|precise|"
    r"occultation|rex|radio\s+science|not\s+preliminary|preliminary|initial|first|press)"
)
_WRONG_TOK = r"(?:2[,\s]?37[024])"
_IDENT_RE = re.compile(
    rf"{_RULE_MARK}[^.]{{0,80}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,80}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "128",
        "test_name": "Tier 5 targeted: conflicting-source reconciliation (Pluto diameter — refined REX vs preliminary)",
        "difficulty_level": "9/10",
        "category": "adaptive_targeted",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker making a SOURCE-TRUST decision with FEW reads (do not fan out; do "
        "not answer from memory). NASA's New Horizons mission reported Pluto's diameter more than "
        "once, and two different values circulate:\n\n"
        '  CLAIM (widely reproduced, the FIRST value announced 13 July 2015): "Pluto\'s diameter is '
        'about 2,370 km."\n\n'
        "Open the authoritative Wikipedia article for Pluto and read how the measurement was "
        "REFINED. Apply this rule: the REFINED FINAL measurement (from the New Horizons Radio "
        "Science Experiment / radio-occultation) SUPERSEDES the preliminary press-release figure. "
        "Do NOT take the first hit and do NOT average the two values. Then report:\n"
        "  (a) the CORRECT refined final diameter of Pluto in km (or its mean radius) — the keystone;\n"
        "  (b) identify the ~2,370 km figure as the preliminary value that was later refined, and "
        "state which measurement supersedes it.\n\n"
        "Cite the exact authoritative source URL you read the figure from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct refined final diameter of Pluto (or mean radius) — the keystone",
        "The ~2,370 km figure identified as the preliminary value that was later refined",
        "A statement of the rule (refined REX measurement supersedes the preliminary figure)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 reads (the refined figure must be read, not recalled)",
        "Correct refined diameter reported (2,376.6 km / radius 1,188.3 km)",
        "Preliminary ~2,370 km value identified as superseded (not chosen, not averaged)",
        "Authoritative Pluto page cited",
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
            "reason": f"{n} visit(s) (target >=2; the refined figure must be read, not recalled)"}


def validate_keystone_diameter(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the REFINED final diameter (2,376.6 km / radius 1,188.3 km). Rejects the
    preliminary 2,370/2,372/2,374 and the average 2,373 (the correct token family shares no digits)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_diameter", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Refined diameter 2,376.6 km / radius 1,188.3 km present" if passed
                      else "Refined final diameter (2,376.6 km / 1,188.3 km radius) missing/incorrect"}


def validate_reconciliation_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic (retained even when the final pick is wrong): did the agent
    surface BOTH conflicting figures — the refined value AND the preliminary one — proving it looked
    at the source rather than grabbing one. Gated ONLY on read-evidence (a visit or the authoritative
    citation) so a parametric answer banks nothing."""
    if not _read_evidence(result, observability):
        return {"check": "reconciliation_coverage", "passed": False, "score": 0.0,
                "reason": "No read-evidence (no visit, no citation) -> coverage not credited"}
    text = _all_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "reconciliation_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"refined_value={has_correct}, preliminary_value={has_wrong}"}


def validate_identifies_correct_source(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the agent FLAGS the ~2,370 km value as the preliminary/superseded figure
    (rule marker within a sentence of the wrong token). Short-circuits to 0 without the keystone or
    without read-evidence."""
    if not _keystone_ok(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source reconciliation not credited"}
    if not _read_evidence(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "No read-evidence -> reconciliation not credited"}
    flagged = bool(_IDENT_RE.search(_all_text(result)))
    return {"check": "identifies_correct_source", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"preliminary ~2,370 km flagged as superseded={flagged}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: authoritative Pluto page cited. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(_SLUG_RX, _all_text(result).lower()))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative Pluto page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_diameter,
        validate_reconciliation_coverage,
        validate_identifies_correct_source,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold (2 -> 1) for the graph_compiled arm.

    Two parallel visit leaves — the PRELIMINARY first-announced value and the REFINED final value —
    plus a dependent verify leaf that applies the recency rule (``depends_on`` both, templating
    ``{preliminary}`` and ``{refined}``). Leak-free: names each figure by its RECENCY (preliminary
    vs refined), never its number; the correct 2,376.6 km / 1,188.3 km never appears."""
    return {
        "leaves": [
            {
                "id": "preliminary",
                "instruction": (
                    "Open the authoritative Wikipedia article for Pluto. Read the FIRST diameter "
                    "value NASA's New Horizons mission announced on 13 July 2015 (the preliminary "
                    "press-release figure, later refined). Do NOT answer from memory; report the "
                    "exact figure with units and the source URL."
                ),
                "expect": "PRELIMINARY first-announced diameter with units — source URL",
                "depends_on": [],
            },
            {
                "id": "refined",
                "instruction": (
                    "On the authoritative Wikipedia article for Pluto, read the REFINED FINAL "
                    "diameter (or mean radius) from the New Horizons Radio Science Experiment (REX) "
                    "radio-occultation measurement — the value in the infobox. Do NOT answer from "
                    "memory; report the exact figure with units and the source URL."
                ),
                "expect": "REFINED final diameter/radius with units — source URL",
                "depends_on": [],
            },
            {
                "id": "reconcile",
                "action": "verify",
                "details": {
                    "claim": "Pluto's diameter is about 2,370 km (the first value announced).",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Reconcile the two figures using this rule: the REFINED FINAL measurement "
                    "supersedes the PRELIMINARY first-announced figure; do not average them. "
                    "Preliminary value gathered: {preliminary}. Refined value gathered: {refined}. "
                    "Report the correct refined diameter, identify the preliminary figure as "
                    "superseded, and cite the source URL."
                ),
                "expect": "Correct refined diameter + preliminary figure flagged superseded — source URL",
                "depends_on": ["preliminary", "refined"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT refined final diameter of Pluto (or its mean radius) — the "
            "keystone — and (b) identify the ~2,370 km figure as the preliminary value that was "
            "later refined, stating which measurement supersedes it. Do not average the two. Cite "
            "the authoritative source URL."
        ),
    }
