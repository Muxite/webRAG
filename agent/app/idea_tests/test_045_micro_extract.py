"""
Test 045: Micro — Single-Page Fact Extraction (+ stop early)
Level: micro   Weight: short   Difficulty: 2/10

A micro test of a single local decision: visit ONE named page, extract ONE specific value
from noisy text, and STOP (do not wander to other pages). Cheap, fast, repeatable — for
debugging the browse/extract policy, not for the cost-recovery headline. Discrimination
here is mostly about efficiency (did it stop when the evidence was found?) and grounding.

Ground truth (verified against live English Wikipedia, 2026-06):
  Burj Khalifa architectural height = 828 m
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


PAGE_URL = "https://en.wikipedia.org/wiki/Burj_Khalifa"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "045",
        "test_name": "Micro: Single-Page Fact Extraction",
        "difficulty_level": "2/10",
        "category": "Micro Extraction",
        "level": "micro",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Visit ONLY this page: {PAGE_URL}\n"
        "Report the architectural height of the Burj Khalifa in metres (the value in its infobox). "
        "Do not visit any other pages — stop as soon as you have the value. Cite the page URL."
    )


def get_required_deliverables() -> List[str]:
    return ["Architectural height in metres", "Source URL"]


def get_success_criteria() -> List[str]:
    return [
        "Reports the height 828 m",
        "Visits only the one page (efficient — no wandering)",
        "Cites the source URL",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return ("burj khalifa" in text) and bool(re.search(r"\b828\b", text))


def validate_keystone_fact(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the correct height value. Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_height",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Reported 828 m" if passed else "Height 828 m missing/incorrect",
    }


def validate_efficiency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Micro tests reward stopping early. Penalize wandering beyond the one needed page."""
    if not _keystone_ok(result):
        return {"check": "efficiency", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> efficiency not credited"}
    visits = observability.get("visit", {}).get("count", 0)
    if visits <= 1:
        score = 1.0
    elif visits == 2:
        score = 0.7
    else:
        score = max(0.0, 1.0 - (visits - 1) * 0.3)
    return {
        "check": "efficiency",
        "passed": visits <= 2,
        "score": round(score, 3),
        "reason": f"{visits} visit(s) (ideal 1; stop when found)",
    }


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Cited the source page. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/burj_khalifa", text))
    return {
        "check": "grounding",
        "passed": cited,
        "score": 1.0 if cited else 0.0,
        "reason": f"source cited={cited}",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_fact, validate_efficiency, validate_grounding]


def get_llm_validation_function() -> callable:
    return None
