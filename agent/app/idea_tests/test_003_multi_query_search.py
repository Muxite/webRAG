"""
Test 003: Multi-Query Search
Difficulty: 2/10 (Easy-Moderate)
Category: Information Retrieval
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "003",
        "test_name": "Multi-Query Search",
        "difficulty_level": "2/10",
        "category": "Information Retrieval",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return "Find the first webpage result URL for each of the following search terms: 'python programming', 'machine learning', 'data science'. Return exactly 3 URLs, one for each term."


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Three distinct URLs",
        "One URL per search term",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 3 search actions executed",
        "Three distinct URLs provided",
        "URLs are valid HTTP/HTTPS links",
    ]


def validate_search_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sufficient searches."""
    search_count = observability.get("search", {}).get("count", 0)
    passed = search_count >= 3
    return {
        "check": "search_count",
        "passed": passed,
        "score": min(1.0, search_count / 3.0),
        "actual": search_count,
        "expected": 3,
        "reason": f"Found {search_count} searches" if passed else f"Only {search_count} searches",
    }


def validate_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate URLs present."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    passed = len(urls) >= 3
    return {
        "check": "url_count",
        "passed": passed,
        "score": min(1.0, len(urls) / 3.0),
        "url_count": len(urls),
        "reason": f"Found {len(urls)} URLs" if passed else f"Only {len(urls)} URLs, expected 3",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_search_count, validate_urls]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
