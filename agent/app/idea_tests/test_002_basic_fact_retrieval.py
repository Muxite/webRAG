"""
Test 002: Basic Fact Retrieval
Difficulty: 1/10 (Easy)
Category: Simple Information Retrieval
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "002",
        "test_name": "Basic Fact Retrieval",
        "difficulty_level": "1/10",
        "category": "Simple Information Retrieval",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return "What is the capital city of Australia? Provide the answer and cite one authoritative source URL."


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Capital city name",
        "One authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Correct capital city identified (Canberra)",
        "At least one valid URL cited",
        "Answer is clear and direct",
    ]


def validate_answer(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate correct answer provided."""
    final_text = extract_final_text(result).lower()
    has_canberra = "canberra" in final_text
    return {
        "check": "correct_answer",
        "passed": has_canberra,
        "score": 1.0 if has_canberra else 0.0,
        "reason": "Canberra found" if has_canberra else "Canberra not found",
    }


def validate_url(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate URL present."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    has_url = len(urls) >= 1
    return {
        "check": "url_present",
        "passed": has_url,
        "score": min(1.0, len(urls)),
        "url_count": len(urls),
        "reason": f"Found {len(urls)} URL(s)" if has_url else "No URLs found",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_answer, validate_url]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
