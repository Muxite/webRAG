"""
Test 049: Tier 2 (medium) — Two-Page Fact Combination
Level: integration   Weight: short   Difficulty: 4/10

A small *integration* task: read TWO named pages, pull one fact from each, and combine
them into a derived answer (which structure was completed first, and by how many
years). The combined value (the year gap) cannot be lifted from a single search
snippet — it requires actually reading both pages and computing — so this is the first
rung where ``full``/``partial`` (which visit) should beat ``minimal`` (snippets only).
Still deliberately small: two pages, two integers, capped visit chars.

Ground truth (verified against live English Wikipedia, 2026-06):
  Eiffel Tower completed/opened = 1889
  Statue of Liberty dedicated   = 1886
  => Statue of Liberty was completed FIRST, by 3 years.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


EIFFEL_URL = "https://en.wikipedia.org/wiki/Eiffel_Tower"
LIBERTY_URL = "https://en.wikipedia.org/wiki/Statue_of_Liberty"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "049",
        "test_name": "Tier 2: Two-Page Fact Combination",
        "difficulty_level": "4/10",
        "category": "Tier 2 Integration",
        "level": "integration",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        "Compare the completion dates of two landmarks using these pages:\n"
        f"- {EIFFEL_URL} (Eiffel Tower)\n"
        f"- {LIBERTY_URL} (Statue of Liberty)\n"
        "Determine which of the two was completed FIRST and by HOW MANY YEARS. "
        "Report both completion years, the winner, and the year gap. Cite both URLs."
    )


def get_required_deliverables() -> List[str]:
    return ["Both completion years", "Which was first", "Year gap", "Both source URLs"]


def get_success_criteria() -> List[str]:
    return [
        "Both years present (1886 and 1889)",
        "Identifies the Statue of Liberty as completed first",
        "States the gap of 3 years",
        "Cites both source URLs",
    ]


def _both_years_present(text: str) -> bool:
    return bool(re.search(r"\b1886\b", text)) and bool(re.search(r"\b1889\b", text))


def _gap_ok(text: str) -> bool:
    return bool(re.search(r"(\b3\b|three)\s*year", text))


def _keystone_ok(result: Dict[str, Any]) -> bool:
    """KEYSTONE: the combined answer — both years AND the 3-year gap."""
    text = extract_final_text(result).lower()
    return _both_years_present(text) and _gap_ok(text)


def validate_keystone_combination(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: both years present and the derived 3-year gap. Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_combination",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Both years + 3-year gap reported" if passed else "Missing a year or the 3-year gap",
    }


def validate_ordering(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Statue of Liberty named as the earlier of the two. Short-circuits on keystone."""
    if not _keystone_ok(result):
        return {"check": "ordering", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> ordering not credited"}
    text = extract_final_text(result).lower()
    # Liberty earlier: mentioned alongside a "first/earlier/before" cue.
    cue = re.search(r"statue of liberty[^.]{0,60}\b(first|earlier|before|older)\b", text) \
        or re.search(r"\b(first|earlier|before|older)\b[^.]{0,60}statue of liberty", text)
    passed = bool(cue)
    return {
        "check": "ordering",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": f"Statue of Liberty identified as earlier={passed}",
    }


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Cited BOTH source pages. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    cited_eiffel = bool(re.search(r"wiki/eiffel_tower", text))
    cited_liberty = bool(re.search(r"wiki/statue_of_liberty", text))
    n = int(cited_eiffel) + int(cited_liberty)
    return {
        "check": "grounding",
        "passed": n == 2,
        "score": n / 2.0,
        "reason": f"cited {n}/2 sources (eiffel={cited_eiffel}, liberty={cited_liberty})",
    }


def validate_visited_both(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence the answer came from reading both pages (>=2 visits). Discriminates the
    snippet-only ``minimal`` rung from rungs that actually crawl."""
    if not _keystone_ok(result):
        return {"check": "visited_both", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> visit credit withheld"}
    visits = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visited_both",
        "passed": visits >= 2,
        "score": 1.0 if visits >= 2 else (0.5 if visits == 1 else 0.0),
        "reason": f"{visits} page visit(s) (need 2)",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_combination, validate_ordering, validate_grounding, validate_visited_both]


def get_llm_validation_function() -> callable:
    return None
