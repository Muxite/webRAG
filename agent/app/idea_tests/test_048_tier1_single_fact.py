"""
Test 048: Tier 1 (easy) — Single-Page Fact Retrieval, citation required
Level: micro   Weight: short   Difficulty: 2/10

A deliberately small, deterministic sanity task for the cost-vs-accuracy benchmark's
*foundation*: one named page, one well-known fact, cite the URL. The fact (Eiffel
Tower completed in 1889) is well known, so ALL tooling rungs (minimal / partial /
full) should pass near 100% — that is the point. Tier 1 validates the pipeline end to
end (search/visit -> synthesis -> validation -> cost accounting) rather than
discriminating between rungs.

Keep context small: run with a visit char cap (e.g. IDEA_TEST_NAIVE_RAG_PAGE_CHARS)
so the page is trimmed; the fact is near the top of the infobox/lead.

Ground truth (verified against live English Wikipedia, 2026-06):
  Eiffel Tower completed = 1889
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


PAGE_URL = "https://en.wikipedia.org/wiki/Eiffel_Tower"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "048",
        "test_name": "Tier 1: Single-Page Fact Retrieval",
        "difficulty_level": "2/10",
        "category": "Tier 1 Sanity",
        "level": "micro",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Visit this page: {PAGE_URL}\n"
        "Report the year the Eiffel Tower was completed (see its infobox/lead). "
        "Give the year and cite the page URL."
    )


def get_required_deliverables() -> List[str]:
    return ["Completion year", "Source URL"]


def get_success_criteria() -> List[str]:
    return [
        "Reports the year 1889",
        "Cites the source URL",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return ("eiffel" in text) and bool(re.search(r"\b1889\b", text))


def validate_keystone_year(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the correct completion year. Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_year",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Reported 1889" if passed else "Year 1889 missing/incorrect",
    }


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Cited the source page. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/eiffel_tower", text))
    return {
        "check": "grounding",
        "passed": cited,
        "score": 1.0 if cited else 0.0,
        "reason": f"source cited={cited}",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_year, validate_grounding]


def get_llm_validation_function() -> callable:
    return None
