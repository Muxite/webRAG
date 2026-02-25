"""
Test 026: Deterministic Page Facts
Difficulty: 1/10 (Easy)
Category: Deterministic Retrieval
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "026",
        "test_name": "Deterministic Page Facts",
        "difficulty_level": "1/10",
        "category": "Deterministic Retrieval",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Visit https://example.com and https://www.iana.org/domains/reserved. "
        "Return: (1) the exact H1 text from example.com, (2) the exact domain of the external link from example.com, "
        "and (3) one URL citation for each visited page."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Exact H1 text from example.com",
        "External link domain from example.com",
        "Two citation URLs (example.com and iana.org/domains/reserved)",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 visit actions executed",
        "Output contains 'Example Domain'",
        "Output contains iana.org domain reference",
    ]


def _extract_visit_urls(result: Dict[str, Any]) -> List[str]:
    """
    Extract visited URLs from graph action results.
    :param result: Test result payload.
    :return: Visited URL list.
    """
    graph = result.get("graph") or {}
    nodes = graph.get("nodes") or {}
    node_items = nodes.values() if isinstance(nodes, dict) else (nodes if isinstance(nodes, list) else [])
    urls: List[str] = []
    for node in node_items:
        if not isinstance(node, dict):
            continue
        details = node.get("details") or {}
        action_result = details.get("action_result") or details.get("actionResult") or {}
        if isinstance(action_result, dict) and action_result.get("action") == "visit" and action_result.get("success"):
            url = str(action_result.get("url") or "").strip()
            if url.startswith("http"):
                urls.append(url)
    return urls


def validate_visit_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate visit count."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 2
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 2.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)",
    }


def validate_example_domain_text(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate deterministic text from example.com."""
    final_text = extract_final_text(result)
    passed = "Example Domain" in final_text
    return {
        "check": "example_domain_text",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Found deterministic text 'Example Domain'" if passed else "Missing 'Example Domain'",
    }


def validate_known_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate known URLs are present in output or visited graph."""
    final_text = extract_final_text(result)
    output_urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    visited_urls = _extract_visit_urls(result)
    combined = " ".join(output_urls + visited_urls).lower()
    has_example = "example.com" in combined
    has_iana = "iana.org/domains/reserved" in combined
    checks = int(has_example) + int(has_iana)
    return {
        "check": "known_urls",
        "passed": checks == 2,
        "score": checks / 2.0,
        "has_example": has_example,
        "has_iana_reserved": has_iana,
        "reason": f"known URLs present: example={has_example}, iana_reserved={has_iana}",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_visit_count, validate_example_domain_text, validate_known_urls]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
