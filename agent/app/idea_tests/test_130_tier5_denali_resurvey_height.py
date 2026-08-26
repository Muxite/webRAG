"""
Test 130: Tier 5 (adaptive-targeted) — CONFLICTING-SOURCE RECONCILIATION (trust decision).
Level: integration   Weight: long   Difficulty: 9/10

LOW-CONTEXT source-trust decision task (Bucket B). A good ADAPTIVE agent must reconcile two
reputable-looking elevation figures for the SAME mountain by applying a checkable RECENCY rule, on
a narrow golden path of 2-3 reads — NOT by breadth, NOT by averaging, NOT by grabbing the older
familiar number.

The quantity: the summit elevation of Denali (Mount McKinley), Alaska — the highest peak in North
America.
  * WRONG (older, long-familiar): 20,320 ft (6,194 m) — the 1952 photogrammetric figure, cited for
    decades and the value a parametric model recalls.
  * CORRECT (keystone): 20,310 ft (6,190 m) — the value from the 2015 U.S. Geological Survey GPS
    resurvey, 10 ft lower; the current official elevation in the live Wikipedia infobox.
  The reconciliation RULE is recency: the MOST RECENT official survey (2015 USGS GPS) supersedes
  the 1952 figure. Do not report the old familiar number; do not average the two.

Ground truth (verified against live English Wikipedia https://en.wikipedia.org/wiki/Denali,
2026-07-10):
  * Infobox: 20,310 ft (6,190 m).
  * Body: "On September 2, 2015, the U.S. Geological Survey measured the mountain at 20,310 feet
    (6,190 m) high, 10 ft lower than the 20,320 feet (6,194 m) measured in 1952."
  * Margin/distinctness: correct-value token regex ``20[,\\s]?310|6[,\\s]?190`` matches only the
    2015 figure, never the 1952 20,320 / 6,194 or the averages (20,315 ft / 6,192 m). The
    thousands+hundreds differ, so a single noisy extraction of the old figure fails the keystone
    (bimodal) rather than silently flipping it.

Leak-resistant: the resurveyed 20,310 ft / 6,190 m is a page/USGS figure; a parametric model
reaches for the classic "20,320 ft". No answer string appears anywhere the executor reads.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/Denali"
_SLUG_RX = r"wiki/denali"

# CORRECT (keystone): 2015 USGS resurvey 20,310 ft OR 6,190 m.
_CORRECT_RE = re.compile(r"20[,\s]?310|6[,\s]?190")
# WRONG (older 1952): 20,320 ft OR 6,194 m.
_WRONG_RE = re.compile(r"20[,\s]?320|6[,\s]?194")

# Rule marker: 2015 / most recent / resurvey supersedes 1952 / older.
_RULE_MARK = (
    r"(?:2015|resurvey\w*|re-?survey\w*|revis\w+|updated|supersed\w+|newer|latest|"
    r"most\s+recent|older|previous\w*|former\w*|1952|photogramm\w+|no\s+longer|outdated)"
)
_WRONG_TOK = r"(?:20[,\s]?320|6[,\s]?194)"
_IDENT_RE = re.compile(
    rf"{_RULE_MARK}[^.]{{0,80}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,80}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "130",
        "test_name": "Tier 5 targeted: conflicting-source reconciliation (Denali — 2015 USGS resurvey vs 1952)",
        "difficulty_level": "9/10",
        "category": "adaptive_targeted",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker making a SOURCE-TRUST decision with FEW reads (do not fan out; do "
        "not answer from memory). Two elevations circulate for Denali (Mount McKinley), the highest "
        "peak in North America:\n\n"
        '  CLAIM (the long-familiar figure): "Denali is 20,320 ft (6,194 m) high."\n\n'
        "Open the authoritative Wikipedia article for Denali and read how the elevation was "
        "re-measured. Apply this rule: the MOST RECENT official survey supersedes the older figure. "
        "The 20,320 ft value is the 1952 photogrammetric measurement; a newer official GPS survey "
        "gives a slightly lower figure. Do NOT report the old familiar number and do NOT average "
        "the two. Then report:\n"
        "  (a) the CORRECT current official elevation of Denali (ft and/or m) — the keystone;\n"
        "  (b) identify the 20,320 ft / 6,194 m figure as the older (1952) value that was "
        "superseded, and name the survey that supersedes it.\n\n"
        "Cite the exact authoritative source URL you read the figures from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct current official elevation of Denali — the keystone",
        "The 20,320 ft / 6,194 m figure identified as the older (1952) superseded value",
        "A statement of the rule (the 2015 USGS resurvey supersedes the 1952 figure)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 reads (the resurveyed figure must be read, not recalled)",
        "Correct current elevation reported (20,310 ft / 6,190 m)",
        "Older 1952 figure (20,320 ft / 6,194 m) identified as superseded (not chosen, not averaged)",
        "Authoritative Denali page cited",
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


def _sources_cited(result: Dict[str, Any], pattern: "re.Pattern") -> bool:
    """Check the engine's structured citations array (result['output']['sources']) for the
    expected slug/URL, in addition to the prose-text check. The engine can populate the exact
    correct source URL there even when the agent's prose answer never repeats the literal
    string -- relying on prose alone is a false-negative grading bug."""
    if not isinstance(result, dict):
        return False
    output = result.get("output")
    if not isinstance(output, dict):
        return False
    sources = output.get("sources")
    if not isinstance(sources, list):
        return False
    for src in sources:
        if isinstance(src, dict):
            for field in ("url", "title"):
                val = src.get(field)
                if isinstance(val, str) and pattern.search(val.lower()):
                    return True
        elif isinstance(src, str) and pattern.search(src.lower()):
            return True
    return False


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(_CORRECT_RE.search(_primary_text(result)))


def _read_evidence(result: Dict[str, Any], observability: Dict[str, Any]) -> bool:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return n > 0 or bool(re.search(_SLUG_RX, _all_text(result).lower())) or _sources_cited(
        result, re.compile(_SLUG_RX)
    )


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=2; the resurveyed figure must be read, not recalled)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the 2015 resurvey elevation (20,310 ft / 6,190 m). Rejects the older
    20,320 ft / 6,194 m and the average 20,315 ft / 6,192 m (the token families share no digits)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Current elevation 20,310 ft / 6,190 m present" if passed
                      else "Current elevation (20,310 ft / 6,190 m, 2015 USGS) missing/incorrect"}


def validate_reconciliation_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic (retained even when the final pick is wrong): did the agent
    surface BOTH figures — the 2015 value AND the 1952 value — proving it read the source rather
    than grabbing one. Gated ONLY on read-evidence so a parametric answer banks nothing."""
    if not _read_evidence(result, observability):
        return {"check": "reconciliation_coverage", "passed": False, "score": 0.0,
                "reason": "No read-evidence (no visit, no citation) -> coverage not credited"}
    text = _all_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "reconciliation_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"resurvey_value={has_correct}, older_value={has_wrong}"}


def validate_identifies_correct_source(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the agent FLAGS 20,320 ft / 6,194 m as the older (1952) superseded value.
    Short-circuits to 0 without the keystone or read-evidence."""
    if not _keystone_ok(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> recency reconciliation not credited"}
    if not _read_evidence(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "No read-evidence -> reconciliation not credited"}
    flagged = bool(_IDENT_RE.search(_all_text(result)))
    return {"check": "identifies_correct_source", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"20,320 ft / 6,194 m flagged as older (1952) superseded value={flagged}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: authoritative Denali page cited. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(_SLUG_RX, _all_text(result).lower())) or _sources_cited(
        result, re.compile(_SLUG_RX)
    )
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative Denali page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_elevation,
        validate_reconciliation_coverage,
        validate_identifies_correct_source,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold (2 -> 1) for the graph_compiled arm.

    Two parallel visit leaves — the OLDER 1952 figure and the MOST-RECENT official resurvey — plus a
    dependent verify leaf that applies the recency rule (``depends_on`` both, templating ``{older}``
    and ``{resurvey}``). Leak-free: names each figure by its RECENCY, never its number; the correct
    20,310 ft / 6,190 m never appears."""
    return {
        "leaves": [
            {
                "id": "older",
                "instruction": (
                    "Open the authoritative Wikipedia article for Denali (Mount McKinley). Read the "
                    "OLDER, long-cited elevation from the 1952 photogrammetric survey. Do NOT answer "
                    "from memory; report the exact figure with units and the source URL."
                ),
                "expect": "OLDER (1952) elevation with units — source URL",
                "depends_on": [],
            },
            {
                "id": "resurvey",
                "instruction": (
                    "On the authoritative Wikipedia article for Denali, read the CURRENT official "
                    "elevation from the MOST RECENT survey (the 2015 U.S. Geological Survey GPS "
                    "resurvey) — the figure in the infobox. Do NOT answer from memory; report the "
                    "exact figure with units and the source URL."
                ),
                "expect": "CURRENT elevation from the most recent survey with units — source URL",
                "depends_on": [],
            },
            {
                "id": "reconcile",
                "action": "verify",
                "details": {
                    "claim": "Denali is 20,320 ft (6,194 m) high (the long-familiar figure).",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Reconcile the two figures using this rule: the MOST RECENT official survey "
                    "supersedes the older figure; do not average them. Older value gathered: "
                    "{older}. Most-recent resurvey value gathered: {resurvey}. Report the correct "
                    "current elevation, identify the older figure as superseded, and cite the "
                    "source URL."
                ),
                "expect": "Correct current elevation + older figure flagged superseded — source URL",
                "depends_on": ["older", "resurvey"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT current official elevation of Denali (ft and/or m) — the "
            "keystone — and (b) identify the 20,320 ft / 6,194 m figure as the older (1952) value "
            "that was superseded by the most recent official survey. Do not report the old number "
            "and do not average. Cite the authoritative source URL."
        ),
    }
