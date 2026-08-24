"""
Test 154: Tier 5 (breadth) — URL-free TWO-ARM independent comparison (dam height)
Level: graph   Weight: medium   Difficulty: 7/10

NO URLs are given. TWO dams; each dam's structural height is stated in the "Height" field of
that dam's OWN encyclopedia page. The two lookups are wholly independent — neither value is
needed to obtain the other — so the task can be discharged as two arms resolved in any order
(or simultaneously) followed by a single comparison/merge step.

Shape-agnostic by construction: the task statement, the deliverables and the validators all
speak of "each dam" and "the two heights", and never of a first/second lookup. Nothing in the
wording implies or rewards a "look up A, then look up B" narrative over independently
resolving both arms and merging at the end. This doubles as the pilot for a wider reframe of
existing two-entity comparison tasks, so the neutral wording is part of the fixture.

Ground truth (each value verified against the live English Wikipedia infobox "Height" field,
2026-08-22):
  Grande Dixence Dam (Valais, Switzerland)  -> "285 m (935 ft)"        gravity dam, 1961
  Hoover Dam (Nevada/Arizona, USA)          -> "726.4 ft (221.4 m)"    arch-gravity, 1936
  KEYSTONE verdict: Grande Dixence is TALLER, by 63.6 m (285 - 221.4), i.e. ~29% taller.

Margin robustness: the two metric figures (285 vs 221.4) are separated by 63.6 m; the two
imperial figures (935 vs 726.4 ft) by 208.6 ft. No rounding convention, unit conversion or
alternative "structural vs hydraulic height" reading on either page moves either value
anywhere near the other, so a single noisy extraction cannot flip the verdict.

Leak resistance: common knowledge over-weights Hoover Dam's fame, which points at the WRONG
answer, so a parametric guess is more likely to be flipped than right; the correct verdict
needs both pages actually read.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# The two arms — single source of truth for the task statement, validators and compiled plan,
# so they can never drift apart. ``name_re``/``value_re``/``slug`` are regexes.
ARMS: List[Dict[str, str]] = [
    {
        "dam": "Grande Dixence Dam",
        "where": "Valais, Switzerland",
        "name_re": r"grande\s+dixence",
        # metric 285 m, or the imperial 935 ft the same infobox prints
        "value_re": r"\b285\b|\b935\b",
        "height_m": "285",
        "slug": r"wiki/grande[_%20]?dixence",
    },
    {
        "dam": "Hoover Dam",
        "where": "Nevada / Arizona, United States",
        "name_re": r"hoover",
        # metric 221.4 m (or 221), or the imperial 726.4 ft
        "value_re": r"\b221\b|\b726\b",
        "height_m": "221.4",
        "slug": r"wiki/hoover[_%20]?dam",
    },
]

# --- Keystone regex -----------------------------------------------------------------------
# The verdict must be DIRECTIONAL: "Grande Dixence is taller" earns credit, "Hoover is taller"
# must not. Plain proximity ("taller" near "grande dixence") is unusable here because a flipped
# answer mentions both names in the same sentence; every window below is therefore *tempered*
# — ``(?:(?!hoover)[^.])`` consumes any non-period character EXCEPT the start of the other
# dam's name — so the loser's name can never sit inside a window that is supposed to bind the
# winner to the comparative. ``[^.]`` (not ``[^.\n]``) keeps the windows newline-tolerant: a
# report that puts "Taller dam:" and the name on separate lines still matches, while a
# sentence-ending period still bounds the window.
_NOT_B = r"(?:(?!hoover)[^.])"
_NOT_A = r"(?:(?!grande)[^.])"
_MORE = r"taller|higher|tallest|highest|greater\s+height"
_LESS = r"shorter|lower|smaller|less\s+tall"

_VERDICT_A_TALLER = re.compile(
    # (1) "Grande Dixence ... is taller ..."  (loser may not intervene)
    rf"grande\s+dixence{_NOT_B}{{0,60}}(?:{_MORE})"
    # (2) "Hoover ... is shorter than ... Grande Dixence"
    rf"|hoover{_NOT_A}{{0,60}}(?:{_LESS}){_NOT_B}{{0,60}}grande\s+dixence"
    # (3) label layout: a line that OPENS with the comparative — "Taller dam: Grande Dixence"
    #     (the name may sit on the next line; only the comparative is line-anchored, the window
    #     after it stays newline-tolerant).  Anchoring the comparative to line-start is what
    #     stops "Hoover Dam is taller than the Grande Dixence Dam" from matching here.
    rf"|(?:^|\n)\s*(?:the\s+)?(?:{_MORE}){_NOT_B}{{0,40}}grande\s+dixence",
    re.IGNORECASE,
)
# The winner's own height must be present and grounded, not just its name.
_KEYSTONE_VALUE = re.compile(r"\b285\b|\b935\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "154",
        "test_name": "Tier 5: Two-arm independent comparison (which dam is taller)",
        "difficulty_level": "7/10",
        "category": "Breadth Fan-out & Aggregation",
        "level": "graph",
        "weight": "medium",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  - {a['dam']} ({a['where']})" for a in ARMS)
    return (
        "You are given NO URLs. Two dams are named below. The structural HEIGHT of each dam "
        "is stated in the 'Height' field of the infobox on that dam's own encyclopedia page:\n"
        f"{listing}\n\n"
        "The two height lookups are wholly independent of each other: neither dam's height is "
        "needed in order to obtain the other's, and the two pages may be read in any order or "
        "at the same time. Do not answer from memory — the height of each dam must be read off "
        "that dam's own page.\n\n"
        "Compare the two heights and report:\n"
        "  (a) which of the two dams is TALLER, and by how many metres;\n"
        "  (b) the height in metres of EACH of the two dams;\n"
        "  (c) the exact source URL of each page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which of the two dams is taller (the comparison verdict) and the height difference",
        "The height in metres of each of the two dams",
        "Source URL for each dam page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (one per dam)",
        "Correct verdict: the Grande Dixence Dam is taller (285 m vs 221.4 m)",
        "Reports the height of both dams",
        "Cites both dam pages",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires GROUNDING: the verdict string alone is insufficient — the agent
    must have actually visited at least one page (visit.count > 0), else an ungrounded
    parametric-memory guess would earn credit. (The un-gated ``validate_coverage`` diagnostic
    below is deliberately NOT gated on grounding — it measures breadth regardless.)"""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    return bool(_VERDICT_A_TALLER.search(text) and _KEYSTONE_VALUE.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: one page per dam)"}


def validate_keystone_taller(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the comparison verdict — the Grande Dixence Dam named as the TALLER of the
    two, with its height (285 m). Directional: a flipped verdict scores 0. Hard 0/1."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_taller", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Verdict correct: Grande Dixence Dam is taller (285 m)" if passed
                      else "Correct verdict (Grande Dixence Dam taller, 285 m) missing/flipped"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the two dam+height pairs are present.

    Deliberately not short-circuited on the keystone — it measures whether the agent actually
    resolved BOTH arms, which is the axis that separates a structured agent from a linear one
    even when the final comparison is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for a in ARMS:
        if re.search(a["name_re"], text) and re.search(a["value_re"], text):
            covered += 1
            hits.append(a["dam"])
    n = len(ARMS)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} dam+height pairs reported ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for a in ARMS if re.search(a["slug"], text))
    n = len(ARMS)
    return {"check": "citations", "passed": cited == n, "score": cited / n,
            "reason": f"{cited}/{n} dam pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_taller, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored two-arm/compare scaffold for the ``graph_compiled`` variant.

    Schema v2 DAG with exactly TWO leaves and an EMPTY ``depends_on`` on BOTH: the arms are
    genuinely independent, so the plan is one parallel wave followed by the comparison. Neither
    leaf's instruction references the other arm's entity or value — that is the shape-fairness
    property this fixture exists to test. The scaffold encodes only STRUCTURE; it leaks no
    height and no verdict, and the aggregation names neither dam. The runtime model still does
    both page-reads, both extractions and the comparison itself.
    """
    leaves = [
        {
            # id keyed on the DAM (a GIVEN) — never on the height or the verdict (the unknowns).
            "id": re.sub(r"[^a-z0-9]+", "_", a["dam"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for the {a['dam']} ({a['where']}) and read its "
                "structural height directly from the infobox 'Height' field (do not answer "
                "from memory). Report that height in metres."
            ),
            "expect": "DAM NAME — height in metres — the dam's exact Wikipedia URL",
            "depends_on": [],
        }
        for a in ARMS
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given the structural height in metres of each of the two dams, each with "
            "its source URL. Compare the two heights and report (a) which of the two dams is "
            "taller and by how many metres, (b) the height in metres of each of the two dams, "
            "and (c) the source URL of each dam's page."
        ),
    }
