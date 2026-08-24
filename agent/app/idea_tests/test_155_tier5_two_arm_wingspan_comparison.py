"""
Test 155: Tier 5 (breadth) — Two-arm independent comparison: which aircraft has the greater
wingspan?
Level: graph   Weight: medium   Difficulty: 6/10

NO URLs are given. TWO aircraft; each has its wingspan on its own encyclopedia page. The two
lookups are completely independent — neither page mentions a value the other lookup needs —
so the whole task is one parallel wave of two page-reads followed by a single numeric
comparison. This is the minimal-width member of the breadth family (test 052 is 6-way,
test 152 is 7-way): it isolates the "two independent arms, then merge" shape without any
aggregation-over-many bookkeeping, so a failure here is a *shape* failure rather than a
scratchpad-capacity failure.

Domain deliberately disjoint from the rest of the suite (aviation specs; the suite's other
comparisons use lakes/rivers/bridges/telescopes/mountains).

Parametric-leak resistance: the An-225 Mriya is popularly billed as "the largest aircraft
ever built" (it holds the maximum-takeoff-weight and length records), so a model answering
from memory rather than from the pages is actively pulled toward the WRONG arm. The wingspan
record among these two belongs to the 1947 Hughes H-4 Hercules ("Spruce Goose"). The keystone
therefore separates page-reading from recall.

Ground truth (both figures verified against the live English Wikipedia specifications
sections, 2026-08-22):
  Antonov An-225 Mriya   -> wingspan 88.4 m (290 ft 0 in)
    https://en.wikipedia.org/wiki/Antonov_An-225_Mriya
  Hughes H-4 Hercules    -> wingspan 97.51 m (319 ft 11 in)   <-- WINNER / keystone
    https://en.wikipedia.org/wiki/Hughes_H-4_Hercules

Margin: 97.51 - 88.4 = 9.11 m, i.e. the H-4 is 10.3% wider. Unit-conversion noise (m<->ft),
rounding to whole metres, or the occasional secondary source quoting the H-4 at 320 ft 11 in
all move the numbers by well under a metre, so no single noisy extraction can flip the
verdict. Both figures are stable spec-table values, not evolving statistics.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# The two arms — single source of truth for the task statement, validators and the compiled
# plan, so they can never drift apart. ``name_re``/``value_re``/``slug`` are regexes.
ARMS: List[Dict[str, str]] = [
    {
        "aircraft": "Antonov An-225 Mriya",
        "name_re": r"an[-\s]?225|mriya",
        "wingspan": "88.4 m",
        # 88.4 m or its 290 ft equivalent.
        "value_re": r"\b88[.,]4\b|\b290\s*ft",
        "slug": r"wiki/antonov[_\s]?an[-_\s]?225",
    },
    {
        "aircraft": "Hughes H-4 Hercules",
        "name_re": r"h[-\s]?4\b|hercules|spruce\s+goose",
        "wingspan": "97.51 m",
        # 97.51/97.5 m or its 319/320 ft equivalent (secondary sources round differently).
        "value_re": r"\b97[.,]5\d?\b|\b(?:319|320)\s*ft",
        "slug": r"wiki/hughes[_\s]?h[-_\s]?4[_\s]?hercules",
    },
]

_HERC = r"(?:hughes\s+)?h[-\s]?4\b|hercules|spruce\s+goose"
_AN = r"an[-\s]?225|mriya"
_BIGGER = r"larger|wider|greater|bigger|longer"
_SMALLER = r"smaller|narrower|shorter|less"

# KEYSTONE verdict regex. Direction is encoded structurally, never by a bare comparative word:
# a flipped verdict ("the An-225 has a larger wingspan than the H-4") must NOT match. The three
# accepted forms are (1) WINNER ... bigger ... than ... LOSER, (2) LOSER ... smaller ... than ...
# WINNER, and (3) a labelled verdict line "Larger wingspan: <winner>". Proximity uses [^.] (not
# [^.\n]) so a line break inside a report layout still matches, while a sentence-ending period
# still bounds the window.
#
# Every intermediate gap is TEMPERED — it may not run past an occurrence of the OTHER arm's
# name — so a sentence that mentions both aircraft cannot be stitched into a spurious match
# ("Compared with the Hercules, the Boeing is wider than the An-225" must not pass).
def _gap_not(pattern: str, n: int) -> str:
    return rf"(?:(?!(?:{pattern}))[^.]){{0,{n}}}"


_SEP = r"(?::|=|->|—|\bbelongs\s+to\b|\bis\b|\bwas\b)"
_VERDICT = re.compile(
    # (1) WINNER ... bigger ... than ... LOSER
    rf"(?:{_HERC}){_gap_not(_AN, 70)}\b(?:{_BIGGER})\b{_gap_not(_AN, 50)}\bthan\b[^.]{{0,40}}(?:{_AN})"
    # (2) LOSER ... smaller ... than ... WINNER
    rf"|(?:{_AN}){_gap_not(_HERC, 70)}\b(?:{_SMALLER})\b{_gap_not(_HERC, 50)}\bthan\b[^.]{{0,40}}(?:{_HERC})"
    # (3) labelled verdict line: "Larger wingspan: <winner>"
    rf"|\b(?:{_BIGGER}|largest|widest|greatest|biggest)\b\s+wingspan\s*{_SEP}\s*{_gap_not(_AN, 40)}(?:{_HERC})"
    # (4) "Winner: <winner>"
    rf"|\b(?:winner|verdict|answer)\b\s*(?::|=|->|—)\s*{_gap_not(_AN, 40)}(?:{_HERC})"
    # (5) "<winner> ... has the larger wingspan"
    rf"|(?:{_HERC}){_gap_not(_AN, 40)}\bhas\s+the\s+"
    rf"(?:{_BIGGER}|largest|widest|greatest|biggest)\b\s+wingspan",
    re.IGNORECASE,
)

# A comparison FRAME ("compared with X, Y is wider than Z") can smuggle a third subject between
# the winner's name and the comparative, which tempering alone cannot see. Guard: when a frame
# cue immediately precedes a candidate match AND a comma separates the winner's name from the
# comparative, the winner was the frame's OBJECT, not the subject -> not a verdict for it.
# "Compared to the An-225, the H-4 Hercules has a larger wingspan" (no comma in that gap) is
# still credited.
_FRAME_CUE = re.compile(
    r"\b(?:compared\s+(?:with|to)|versus|vs\.?|unlike|relative\s+to|against|"
    r"in\s+contrast\s+(?:with|to)|next\s+to)\b[^.]{0,40}$",
    re.IGNORECASE,
)


def _verdict_stated(text: str) -> bool:
    """True iff the report states the CORRECT comparison direction (winner = the H-4)."""
    for m in _VERDICT.finditer(text):
        before = text[max(0, m.start() - 44):m.start()]
        if _FRAME_CUE.search(before) and "," in m.group(0):
            continue  # framed third-party sentence, not a verdict about the winner
        return True
    return False
# The verdict must be GROUNDED in the winner's actual page figure, not asserted bare.
_KEYSTONE_VALUE = re.compile(ARMS[1]["value_re"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "155",
        "test_name": "Tier 5: Two-arm independent comparison (aircraft wingspan)",
        "difficulty_level": "6/10",
        "category": "Breadth Two-Arm Comparison",
        "level": "graph",
        "weight": "medium",
    }


def get_task_statement() -> str:
    # Deliberately SHAPE-AGNOSTIC: this is posed as one open comparison question, with an
    # explicit note that the two facts are independent and may be gathered in any order or at
    # the same time. It never prescribes "first do A, then do B" — a linear agent and a
    # parallel one are both free to choose their own execution shape.
    return (
        "You are given NO URLs.\n\n"
        "Which aircraft has the GREATER WINGSPAN: the Antonov An-225 Mriya, or the Hughes "
        "H-4 Hercules?\n\n"
        "Each aircraft's wingspan is stated on that aircraft's own encyclopedia page; the two "
        "figures are independent of each other and may be looked up in any order, or at the "
        "same time. Read both from the pages themselves — do not answer from memory.\n\n"
        "Report (a) which of the two has the greater wingspan, (b) BOTH wingspans in metres, "
        "and (c) the exact source URL of each aircraft page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which aircraft has the greater wingspan (the comparison verdict)",
        "Both wingspan figures in metres",
        "Source URL for each aircraft page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (one per arm)",
        "Correct verdict: the Hughes H-4 Hercules (97.51 m) is wider than the An-225 (88.4 m)",
        "Both wingspan values reported",
        "Cites both aircraft pages",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires GROUNDING: the verdict alone is insufficient — the agent must
    have actually visited at least one page (visit.count > 0) and must quote the winner's page
    figure, else an ungrounded parametric-memory coin-flip would earn credit. (The un-gated
    ``validate_coverage`` diagnostic below is deliberately NOT gated on grounding — it measures
    how much was actually gathered regardless.)"""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    return bool(_verdict_stated(text) and _KEYSTONE_VALUE.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2, one page per independent arm)"}


def validate_keystone_verdict(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the comparison verdict — the Hughes H-4 Hercules has the greater wingspan,
    grounded in its page figure (97.51 m). Hard 0/1; a flipped verdict scores 0."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_verdict", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Correct verdict: Hughes H-4 Hercules wider (97.51 m vs 88.4 m)" if passed
                      else "Verdict missing, flipped, or ungrounded (expected: H-4 Hercules, 97.51 m)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the two arms were actually gathered, i.e.
    aircraft name AND its own wingspan figure both present.

    Deliberately not short-circuited on the keystone — it measures whether the agent visited
    and extracted BOTH independent arms, which is the axis that separates a structured agent
    from a linear one even when the final comparison is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for a in ARMS:
        if re.search(a["name_re"], text) and re.search(a["value_re"], text):
            covered += 1
            hits.append(a["aircraft"])
    n = len(ARMS)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} aircraft+wingspan pairs reported ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for a in ARMS if re.search(a["slug"], text))
    n = len(ARMS)
    return {"check": "citations", "passed": cited == n, "score": cited / n,
            "reason": f"{cited}/{n} aircraft pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_verdict, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored two-arm scaffold for the ``graph_compiled`` variant.

    Schema v2 DAG with exactly TWO leaves, each with an EMPTY ``depends_on``: the arms are
    genuinely independent (shape fairness — neither leaf's instruction may reference the other
    arm's aircraft or value), so the plan is a single parallel wave followed by the comparison.
    It encodes only the STRUCTURE — no wingspan figures and no verdict — so the cheap runtime
    model still does both page-reads, both extractions and the comparison itself.
    """
    leaves = [
        {
            # id keyed on the AIRCRAFT (a GIVEN in the task statement) — never on the wingspan
            # or the verdict (the unknowns), so the scaffold leaks no part of the answer.
            "id": re.sub(r"[^a-z0-9]+", "_", a["aircraft"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for the {a['aircraft']} and read its WINGSPAN "
                "directly from that page's specifications/infobox (do not answer from memory). "
                "Report the wingspan in metres."
            ),
            "expect": "AIRCRAFT NAME — wingspan in metres — the aircraft's exact Wikipedia URL",
            "depends_on": [],
        }
        for a in ARMS
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given the wingspan of two aircraft, each with its source URL. Compare the "
            "two numbers and determine which aircraft has the greater wingspan value. Report "
            "(a) that aircraft, stating explicitly that its wingspan is the greater of the two, "
            "(b) both wingspans in metres, and (c) each aircraft's source URL."
        ),
    }
