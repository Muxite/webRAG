"""
Test 054: Tier 5 (mixed DAG) — URL-free parallel gather + one dependent final hop
Level: graph   Weight: long   Difficulty: 8/10

The structural counterpoint to the pure fan-out (052/053) and the pure dependent chain
(050/051): a MIXED DAG. Two independent author look-ups fan out in parallel, then a third hop
DEPENDS on one of them (it needs the resolved author before its target page is knowable). NO
URLs are given. This is the task that exercises the compiled executor's full topology —
parallel wave THEN a dependent wave with upstream-result templating — and is where an
auto-compiled plan must emit both an independent pair and a real dependency edge.

Ground truth (verified against live English Wikipedia / repo tests 050 & 052, 2026-06):
  'Beloved'                -> author Toni Morrison      (parallel hop A)
  'The Old Man and the Sea'-> author Ernest Hemingway  (parallel hop B)
  Toni Morrison            -> master's (MA) from Cornell University   (dependent hop C, on A)
The keystone (Cornell) is reachable only by chaining hop C off hop A's resolved author.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


KEYSTONE_UNIV = r"cornell"
HOP_A_AUTHOR = r"toni\s+morrison|morrison"
HOP_B_AUTHOR = r"ernest\s+hemingway|hemingway"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "054",
        "test_name": "Tier 5: Mixed DAG (parallel gather + dependent final hop)",
        "difficulty_level": "8/10",
        "category": "Mixed DAG (fan-out + dependent hop)",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — search to find the pages you need, then read them (do not guess "
        "from memory):\n"
        "  1. Identify the AUTHOR of the novel 'Beloved' (the Pulitzer-winning novel).\n"
        "  2. Identify the AUTHOR of the novel 'The Old Man and the Sea'.\n"
        "  3. For the author of 'Beloved' (from step 1), read that author's page and identify the "
        "UNIVERSITY where that author earned their master's (MA) degree.\n\n"
        "Report (a) the master's-degree UNIVERSITY of Beloved's author, and (b) BOTH authors, "
        "citing the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Master's-degree university of Beloved's author (the dependent keystone)",
        "Author of 'Beloved'",
        "Author of 'The Old Man and the Sea'",
        "Source URLs",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (the dependent hop needs the author's own page)",
        "Correct master's university (Cornell University)",
        "Both authors identified (Toni Morrison, Ernest Hemingway)",
        "Cites the source pages",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_UNIV, extract_final_text(result).lower()))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3; the dependent hop needs Beloved's author page)"}


def validate_keystone_university(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the dependent hop's fact — the master's university (Cornell). Hard 0/1."""
    passed = _keystone_ok(result)
    return {"check": "keystone_university", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Cornell present" if passed else "Master's university (Cornell) missing/incorrect"}


def validate_breadth_authors(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the two parallel-hop authors were gathered.

    Deliberately not gated on the keystone — it measures whether the agent actually ran the
    independent fan-out (both authors), the axis the parallel wave provides even when the
    dependent keystone hop is botched.
    """
    text = extract_final_text(result).lower()
    has_a = bool(re.search(HOP_A_AUTHOR, text))
    has_b = bool(re.search(HOP_B_AUTHOR, text))
    hits = int(has_a) + int(has_b)
    return {"check": "breadth_authors", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"morrison={has_a}, hemingway={has_b}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/toni_morrison", text)) or bool(re.search(r"wiki/beloved", text))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"Beloved/Morrison page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_university, validate_breadth_authors, validate_citation]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored MIXED-DAG scaffold for the ``graph_compiled`` variant.

    Two independent leaves (the parallel wave) plus a third that ``depends_on`` the first and
    templates its resolved author via ``{author_beloved}`` (the dependent wave). Encodes only
    STRUCTURE — it names the given novels but leaks no author or university. The hand reference
    the auto-compiler is measured against: it must reproduce one independent pair + one edge.
    """
    return {
        "leaves": [
            {
                "id": "author_beloved",
                "instruction": "Identify the AUTHOR of the novel 'Beloved' (the Pulitzer-winning novel).",
                "expect": "AUTHOR FULL NAME — source URL",
                "depends_on": [],
            },
            {
                "id": "author_old_man",
                "instruction": "Identify the AUTHOR of the novel 'The Old Man and the Sea'.",
                "expect": "AUTHOR FULL NAME — source URL",
                "depends_on": [],
            },
            {
                "id": "masters_university",
                "instruction": (
                    "The author of 'Beloved' is: {author_beloved}. Open that author's Wikipedia "
                    "page and read the UNIVERSITY where they earned their master's (MA) degree "
                    "(do not guess from memory)."
                ),
                "expect": "UNIVERSITY NAME — source URL",
                "depends_on": ["author_beloved"],
            },
        ],
        "aggregation": (
            "Report (a) the UNIVERSITY where the author of 'Beloved' earned their master's (MA) "
            "degree (this is the keystone), and (b) BOTH authors gathered. Cite every source URL."
        ),
    }
