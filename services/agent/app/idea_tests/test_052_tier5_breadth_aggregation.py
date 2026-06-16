"""
Test 052: Tier 5 (breadth) — URL-free 6-way Fan-out & Aggregation (argmin)
Level: graph   Weight: long   Difficulty: 8/10

NO URLs are given. SIX independent novels; for each, the agent must find the author and
that author's year of birth (one page-read per author), then AGGREGATE across all six to
report which author was born EARLIEST. This is the canonical Graph-of-Thoughts win
condition — fan-out + merge — and the deliberate counterpoint to the linear chains
(050/051): a *linear* ReAct agent must serialize six gather-hops into a capped step budget
and hold all six facts in one degrading scratchpad to compute the argmin, whereas a graph
fans the six hops out in parallel and aggregates structurally. It is the discriminator for
the "expensive-model-authored scaffold lets a cheap model recover top-tier accuracy"
thesis: see ``get_compiled_plan`` (the offline-authored fan-out/aggregate plan the
``graph_compiled`` variant executes).

Ground truth (verified against live English Wikipedia, 2026-06):
  Pride and Prejudice      -> Jane Austen          -> born 1775   (EARLIEST / keystone)
  Crime and Punishment     -> Fyodor Dostoevsky    -> born 1821
  Mrs Dalloway             -> Virginia Woolf       -> born 1882
  The Great Gatsby         -> F. Scott Fitzgerald  -> born 1896
  The Old Man and the Sea  -> Ernest Hemingway     -> born 1899
  Beloved                  -> Toni Morrison        -> born 1931
The argmin is unambiguous: Austen (1775) precedes the runner-up (Dostoevsky, 1821) by 46
years, so noisy single-year extraction errors cannot flip the keystone.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# The breadth set — single source of truth for the task statement, validators and the
# compiled plan, so they can never drift apart. ``surname``/``year``/``slug`` are regexes.
ENTRIES: List[Dict[str, str]] = [
    {"novel": "Pride and Prejudice", "author": "Jane Austen",
     "surname": r"austen", "year": "1775", "slug": r"wiki/jane_austen"},
    {"novel": "Crime and Punishment", "author": "Fyodor Dostoevsky",
     "surname": r"dostoevsky|dostoyevsky", "year": "1821",
     "slug": r"wiki/fyodor_dostoevsky|wiki/fyodor_dostoyevsky"},
    {"novel": "Mrs Dalloway", "author": "Virginia Woolf",
     "surname": r"woolf", "year": "1882", "slug": r"wiki/virginia_woolf"},
    {"novel": "The Great Gatsby", "author": "F. Scott Fitzgerald",
     "surname": r"fitzgerald", "year": "1896", "slug": r"wiki/f\._?scott_fitzgerald"},
    {"novel": "The Old Man and the Sea", "author": "Ernest Hemingway",
     "surname": r"hemingway", "year": "1899", "slug": r"wiki/ernest_hemingway"},
    {"novel": "Beloved", "author": "Toni Morrison",
     "surname": r"morrison", "year": "1931", "slug": r"wiki/toni_morrison"},
]

# Keystone: the aggregate (argmin) answer must name Austen AS the earliest, with 1775.
# Proximity uses [^.] (not [^.\n]) so a line break between "Earliest-born author:" and the name —
# a common report layout — still matches; a sentence-ending period still bounds the window.
_EARLIEST_NEAR_AUSTEN = re.compile(
    r"(earliest|oldest|born\s+first|first\s+to\s+be\s+born)[^.]{0,60}austen"
    r"|austen[^.]{0,80}(earliest|oldest|born\s+first|born\s+the\s+earliest)",
    re.IGNORECASE,
)
_KEYSTONE_YEAR = re.compile(r"\b1775\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "052",
        "test_name": "Tier 5: 6-way Fan-out & Aggregation (earliest-born author)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. '{e['novel']}'" for i, e in enumerate(ENTRIES, 1))
    return (
        "You are given NO URLs. For EACH of the following six novels, identify its author "
        "and read that author's year of birth from the author's page (do not guess from "
        "memory — open each author's page):\n"
        f"{listing}\n\n"
        "Then AGGREGATE across all six: determine which author was born EARLIEST.\n"
        "Report (a) the EARLIEST-born author and their birth year, and (b) the full list "
        "novel -> author -> birth year for all six, citing the exact source URL of every "
        "author page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Earliest-born author + birth year (the aggregate answer)",
        "All six novel -> author -> birth year rows",
        "Source URL per author page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (six-way fan-out)",
        "Correctly identifies the earliest-born author (Jane Austen, 1775)",
        "Reports all six author/birth-year pairs",
        "Cites each author's source page",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result)
    return bool(_EARLIEST_NEAR_AUSTEN.search(text) and _KEYSTONE_YEAR.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
            "reason": f"{n} visit(s) (target >=6 for a six-way fan-out; >=4 to pass)"}


def validate_keystone_earliest(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the argmin answer — Jane Austen identified as the earliest-born (1775). Hard 0/1."""
    passed = _keystone_ok(result)
    return {"check": "keystone_earliest", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Earliest-born = Jane Austen (1775)" if passed
                      else "Earliest-born author (Jane Austen, 1775) missing/incorrect"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the six author+birth-year pairs are present.

    Deliberately not short-circuited on the keystone — it measures whether the agent
    actually fanned out and gathered all six facts, which is the axis that separates the
    graph (parallel fan-out) from a linear agent even when the final argmin is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for e in ENTRIES:
        has_author = bool(re.search(e["surname"], text))
        has_year = bool(re.search(r"\b" + e["year"] + r"\b", text))
        if has_author and has_year:
            covered += 1
            hits.append(e["author"])
    n = len(ENTRIES)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} author+birth-year pairs reported ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for e in ENTRIES if re.search(e["slug"], text))
    n = len(ENTRIES)
    return {"check": "citations", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} author pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_earliest, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    This is the artifact the *expensive* model (Claude Code, paid by subscription) produces
    ONCE, offline: the decomposition into six independent parallel leaves plus the
    aggregation recipe. It encodes only the STRUCTURE (what to fan out into, how to merge)
    — it deliberately leaks no authors, birth years or the argmin. The cheap runtime model
    still does every page-read, extraction and the final argmin reasoning. Moving the
    *planning* off the cheap model is the whole point: the linear/native-graph arms make
    the cheap model plan at runtime, which is where it flails on breadth tasks.
    """
    leaves = [
        {
            # id keyed on the NOVEL (the given) — never the author (the unknown to find),
            # so the scaffold leaks no part of the answer.
            "id": re.sub(r"[^a-z0-9]+", "_", e["novel"].lower()).strip("_"),
            "instruction": (
                f"Identify the author of the novel '{e['novel']}', then open that author's "
                "Wikipedia page and read their YEAR OF BIRTH directly from the page (do not "
                "guess from memory)."
            ),
            "expect": "AUTHOR FULL NAME — born YEAR — author's exact Wikipedia URL",
        }
        for e in ENTRIES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given the author and birth year for six novels, each with its source "
            "URL. AGGREGATE across all six: determine which author has the MINIMUM (earliest) "
            "birth year. Report (a) the EARLIEST-born author and their birth year, stating "
            "explicitly that they were born earliest, and (b) the full list "
            "novel -> author -> birth year for all six, citing each author's source URL."
        ),
    }
