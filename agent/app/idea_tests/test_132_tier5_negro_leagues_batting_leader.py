"""
Test 132: Tier 5 (adaptive-targeted) — CONFLICTING-SOURCE RECONCILIATION (trust decision).
Level: integration   Weight: long   Difficulty: 10/10

LOW-CONTEXT source-trust decision task (Bucket B). A good ADAPTIVE agent must reconcile two
reputable-looking answers to "who holds the MLB career batting-average record" by applying a
checkable OFFICIAL-REVISION rule, on a narrow golden path of 2-3 reads — NOT by breadth, NOT by
averaging, and NOT by reflexively naming the historical answer.

The quantity: the all-time Major League Baseball career batting-average leader and their average.
  * WRONG (historical consensus, the reflex answer): Ty Cobb, .367 (long the consensus leader; the
    name/number a parametric model recalls).
  * CORRECT (keystone): Josh Gibson, .371 — after MLB officially incorporated Negro Leagues
    statistics into major-league records in 2024, Gibson became the all-time leader, supplanting
    Cobb; this is the #1 line in the live Wikipedia leaders table.
  The reconciliation RULE is official revision / recency: MLB's 2024 incorporation of Negro Leagues
  statistics REVISED the official record; the revised leader supersedes the pre-2024 consensus. Do
  not name the historical leader; do not average the two batting averages.

Ground truth (verified against live English Wikipedia
https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_career_batting_average_leaders,
2026-07-10):
  * #1 Josh Gibson, .371; #2 Ty Cobb, .367.  Article: "Until the incorporation of statistics from
    Negro league baseball into major-league records in 2024, Ty Cobb was the consensus leader;
    subsequently, he was supplanted by Josh Gibson."
  * Margin/distinctness: correct-value token regex ``\\.37[12]`` (.371, allowing MLB.com's .372
    rounding) matches only Gibson, never Cobb's .366/.367 or the average .369; the keystone answer
    is ALSO the name Josh Gibson (rejecting the reflex "Ty Cobb"). One noisy extraction of Cobb's
    figure fails the keystone (bimodal) rather than silently flipping it.

Leak-resistant: the executor never sees the answer; the post-2024 revised leader is a page fact a
naive/parametric agent overrides with the memorized "Ty Cobb .367".
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = (
    "https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_career_batting_average_leaders"
)
_SLUG_RX = r"batting[_\s-]?average[_\s-]?leaders|wiki/josh_gibson"

# CORRECT (keystone): Josh Gibson's average .371 (accept .372 rounding).
_CORRECT_RE = re.compile(r"\.37[12]\b")
# WRONG (historical consensus): Ty Cobb's .366/.367.
_WRONG_RE = re.compile(r"\.36[67]\b")
_GIBSON_RE = re.compile(r"josh\s+gibson", re.IGNORECASE)
_COBB_RE = re.compile(r"ty\s+cobb", re.IGNORECASE)

# Rule marker: 2024 Negro Leagues incorporation / revised / supplanted vs older consensus.
_RULE_MARK = (
    r"(?:2024|negro\s+league\w*|incorporat\w+|revis\w+|updated|supersed\w+|supplant\w+|"
    r"official\w*|previous\w*|former\w*|consensus|until\s+the|no\s+longer|new\s+leader)"
)
_WRONG_TOK = r"(?:ty\s+cobb|\.36[67]\b)"
_IDENT_RE = re.compile(
    rf"{_RULE_MARK}[^.]{{0,90}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,90}}{_RULE_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "132",
        "test_name": "Tier 5 targeted: conflicting-source reconciliation (MLB career batting leader — 2024 Negro Leagues revision)",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker making a SOURCE-TRUST decision with FEW reads (do not fan out; do "
        "not answer from memory). Two answers circulate for 'the all-time MLB career batting-average "
        "leader':\n\n"
        '  CLAIM (the long-standing consensus): "Ty Cobb is the all-time MLB career batting-average '
        'leader, at .367."\n\n'
        "Open the authoritative Wikipedia list of MLB career batting-average leaders and read the "
        "current #1. Apply this rule: MLB's OFFICIAL incorporation of Negro Leagues statistics into "
        "major-league records in 2024 REVISED the record — the revised official leader supersedes "
        "the pre-2024 consensus. Do NOT reflexively name the historical leader and do NOT average "
        "the two batting averages. Then report:\n"
        "  (a) the CORRECT current all-time career batting-average leader AND that player's batting "
        "average — the keystone;\n"
        "  (b) identify Ty Cobb (.367) as the former (pre-2024) consensus leader who was supplanted, "
        "naming the 2024 revision.\n\n"
        "Cite the exact authoritative source URL you read the record from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct current career batting-average leader and their average — the keystone",
        "Ty Cobb (.367) identified as the former (pre-2024) consensus leader who was supplanted",
        "A statement of the rule (2024 MLB incorporation of Negro Leagues statistics revised the record)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 reads (the current #1 must be read, not recalled)",
        "Correct current leader and average reported (Josh Gibson, .371)",
        "Ty Cobb (.367) identified as the superseded former leader (not chosen, not averaged)",
        "Authoritative leaders list (or Josh Gibson page) cited",
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
    """KEYSTONE: the revised leader's batting average (.371/.372) AND the name Josh Gibson — both
    required so neither a bare number nor a bare name (nor the reflex Ty Cobb) passes. Also requires
    grounding evidence (at least one visit) so a parametric guess cannot bank keystone credit."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    if not grounded:
        return False
    txt = _primary_text(result)
    return bool(_CORRECT_RE.search(txt)) and bool(_GIBSON_RE.search(txt))


def _read_evidence(result: Dict[str, Any], observability: Dict[str, Any]) -> bool:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return n > 0 or bool(re.search(_SLUG_RX, _all_text(result).lower())) or _sources_cited(
        result, re.compile(_SLUG_RX)
    )


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=2; the current #1 must be read, not recalled)"}


def validate_keystone_leader(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Josh Gibson AND .371/.372. Rejects the reflex Ty Cobb / .367 and the
    average .369 (the correct token family shares no digits with the wrong one)."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_leader", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Revised leader Josh Gibson (.371) present" if passed
                      else "Revised leader (Josh Gibson, .371) missing/incorrect"}


def validate_reconciliation_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic (retained even when the final pick is wrong): did the agent
    surface BOTH the revised leader AND the former consensus leader — proving it read the source
    rather than grabbing the reflex answer. Credits each side by NAME (Josh Gibson / Ty Cobb). Gated
    ONLY on read-evidence so a parametric answer banks nothing."""
    if not _read_evidence(result, observability):
        return {"check": "reconciliation_coverage", "passed": False, "score": 0.0,
                "reason": "No read-evidence (no visit, no citation) -> coverage not credited"}
    text = _all_text(result)
    has_correct = bool(_GIBSON_RE.search(text))
    has_wrong = bool(_COBB_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "reconciliation_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"revised_leader(Gibson)={has_correct}, former_leader(Cobb)={has_wrong}"}


def validate_identifies_correct_source(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the agent FLAGS Ty Cobb / .367 as the former (pre-2024) consensus leader who
    was supplanted (rule marker within a sentence of Cobb / .367). Short-circuits to 0 without the
    keystone or read-evidence."""
    if not _keystone_ok(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> revision reconciliation not credited"}
    if not _read_evidence(result, observability):
        return {"check": "identifies_correct_source", "passed": False, "score": 0.0,
                "reason": "No read-evidence -> reconciliation not credited"}
    flagged = bool(_IDENT_RE.search(_all_text(result)))
    return {"check": "identifies_correct_source", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"Ty Cobb (.367) flagged as former (pre-2024) leader={flagged}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: authoritative leaders list (or Josh Gibson page) cited. Gated on keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(_SLUG_RX, _all_text(result).lower())) or _sources_cited(
        result, re.compile(_SLUG_RX)
    )
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative source cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_leader,
        validate_reconciliation_coverage,
        validate_identifies_correct_source,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold (2 -> 1) for the graph_compiled arm.

    Two parallel visit leaves — the FORMER pre-2024 consensus leader and the CURRENT #1 in the
    revised record — plus a dependent verify leaf that applies the official-revision rule
    (``depends_on`` both, templating ``{former}`` and ``{current}``). Leak-free: names each by its
    ROLE (former vs current), never the answer; 'Josh Gibson' and '.371' never appear."""
    return {
        "leaves": [
            {
                "id": "former",
                "instruction": (
                    "Open the authoritative Wikipedia list of MLB career batting-average leaders. "
                    "Read who was the CONSENSUS all-time leader BEFORE the 2024 incorporation of "
                    "Negro Leagues statistics, and that player's batting average. Do NOT answer from "
                    "memory; report the name, the average, and the source URL."
                ),
                "expect": "FORMER (pre-2024) consensus leader + average — source URL",
                "depends_on": [],
            },
            {
                "id": "current",
                "instruction": (
                    "On the authoritative Wikipedia list of MLB career batting-average leaders, read "
                    "the CURRENT #1 (top row) after the 2024 incorporation of Negro Leagues "
                    "statistics into major-league records, and that player's batting average. Do NOT "
                    "answer from memory; report the name, the average, and the source URL."
                ),
                "expect": "CURRENT #1 leader + average — source URL",
                "depends_on": [],
            },
            {
                "id": "reconcile",
                "action": "verify",
                "details": {
                    "claim": "Ty Cobb is the all-time MLB career batting-average leader, at .367.",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Reconcile the two answers using this rule: MLB's 2024 incorporation of Negro "
                    "Leagues statistics REVISED the official record, so the current #1 supersedes "
                    "the pre-2024 consensus; do not average the batting averages. Former leader "
                    "gathered: {former}. Current #1 gathered: {current}. Report the correct current "
                    "leader and average, identify the former leader as supplanted, and cite the "
                    "source URL."
                ),
                "expect": "Correct current leader + average, former leader flagged supplanted — source URL",
                "depends_on": ["former", "current"],
            },
        ],
        "aggregation": (
            "Report (a) the CORRECT current all-time MLB career batting-average leader AND that "
            "player's average — the keystone — and (b) identify the former (pre-2024) consensus "
            "leader as the one who was supplanted, naming the 2024 revision. Do not name the "
            "historical leader as the answer and do not average. Cite the authoritative source URL."
        ),
    }
