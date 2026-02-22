"""
Test 012: Wikipedia Link Collection
Difficulty: 3/10 (Moderate)
Category: Link Collection & Summarization
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "012",
        "test_name": "Wikipedia Link Collection",
        "difficulty_level": "3/10",
        "category": "Link Collection & Summarization",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Go to the Wikipedia main page (https://en.wikipedia.org/wiki/Main_Page) and collect 10 links "
        "from the page. For each link, provide: the link URL, the link text/title, and a brief description "
        "(1-2 sentences) of what the page is about. Organize the results clearly."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "10 distinct Wikipedia links",
        "URL for each link",
        "Link text/title for each",
        "Brief description (1-2 sentences) for each",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 10 links collected",
        "All links are valid Wikipedia URLs",
        "Descriptions provided for each link",
        "At least 1 visit action executed",
    ]


def validate_link_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sufficient links collected."""
    final_text = extract_final_text(result)
    wikipedia_urls = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text)
    unique_urls = len(set(wikipedia_urls))
    passed = unique_urls >= 10
    return {
        "check": "link_count",
        "passed": passed,
        "score": min(1.0, unique_urls / 10.0),
        "url_count": unique_urls,
        "reason": f"Found {unique_urls} Wikipedia URLs",
    }


def validate_descriptions(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate descriptions present."""
    final_text = extract_final_text(result)
    sentences = re.findall(r"[A-Z][^.!?]{20,}[.!?]", final_text)
    passed = len(sentences) >= 8
    return {
        "check": "descriptions",
        "passed": passed,
        "score": min(1.0, len(sentences) / 10.0),
        "sentence_count": len(sentences),
        "reason": f"Found {len(sentences)} descriptive sentences",
    }


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Wikipedia page visited."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 1
    return {
        "check": "wikipedia_visit",
        "passed": passed,
        "score": min(1.0, visit_count),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit actions",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_link_count, validate_descriptions, validate_visits]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
