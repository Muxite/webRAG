"""
Test 050: Tier 3 (hard) — URL-free 2-hop Search Chain
Level: navigation   Weight: long   Difficulty: 7/10

NO URLs are given (explicit URLs structurally favor naive_rag). The agent must SEARCH to
find the right pages, then follow a dependency: identify the author of a novel, then read
THAT author's page to find a fact (master's university) that is not in the novel's
snippets. A single fixed search round (naive_rag) cannot pre-plan the second hop, so this
is where graph/sequential agents should pull ahead.

Ground truth (verified against live English Wikipedia, 2026-06):
  novel 'Beloved'  ->  author Toni Morrison  ->  master's (MA) from Cornell University
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "050",
        "test_name": "Tier 3: URL-free 2-hop Search Chain",
        "difficulty_level": "7/10",
        "category": "Search-Driven Multi-Hop",
        "level": "navigation",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — search to find the pages you need, then read them "
        "(do not guess from memory).\n"
        "  1. Identify the AUTHOR of the novel 'Beloved' (the Pulitzer-winning novel).\n"
        "  2. Read that author's page and identify the UNIVERSITY where that author earned "
        "their master's (MA) degree.\n\n"
        "Report (a) the author, (b) the master's-degree university, and cite the exact URL of "
        "every page you read."
    )


def get_required_deliverables() -> List[str]:
    return ["Author of 'Beloved'", "University of the author's master's degree", "Source URLs"]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (search-driven, not given)",
        "Correct author (Toni Morrison)",
        "Correct master's university (Cornell University)",
        "Cites the source pages",
    ]


KEYSTONE_UNIV = r"cornell"
HOP_AUTHOR = r"toni\s+morrison"


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_UNIV, extract_final_text(result).lower()))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2; second hop needs the author's own page)"}


def validate_keystone_university(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the master's university (Cornell). Hard 0/1."""
    passed = _keystone_ok(result)
    return {"check": "keystone_university", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Cornell present" if passed else "Master's university (Cornell) missing/incorrect"}


def validate_author(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "author", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> author hop not credited"}
    has = bool(re.search(HOP_AUTHOR, extract_final_text(result).lower()))
    return {"check": "author", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"author Toni Morrison identified={has}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/toni_morrison", text)) or bool(re.search(r"wiki/beloved", text))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_university, validate_author, validate_citation]


def get_llm_validation_function() -> callable:
    return None
