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
        "At least 10 distinct Wikipedia links from a visited page",
        "URL for each link",
        "Short label for each link",
        "One-line note for each link",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 10 links collected",
        "All links are valid Wikipedia URLs",
        "At least 1 visit action executed",
        "At least 1 returned link is present in visited-page link evidence",
    ]


def _visit_link_evidence(result: Dict[str, Any]) -> List[str]:
    """
    Extract visited-page link evidence from graph action results.
    :param result: Test result payload.
    :return: Collected link URLs.
    """
    graph = result.get("graph") or {}
    nodes = graph.get("nodes") or {}
    node_items = nodes.values() if isinstance(nodes, dict) else (nodes if isinstance(nodes, list) else [])
    links: List[str] = []
    for node in node_items:
        if not isinstance(node, dict):
            continue
        details = node.get("details") or {}
        action_result = details.get("action_result") or details.get("actionResult") or {}
        if not isinstance(action_result, dict):
            continue
        if action_result.get("action") != "visit" or not action_result.get("success"):
            continue
        result_links = action_result.get("links") or []
        if isinstance(result_links, list):
            for url in result_links:
                text = str(url or "").strip()
                if text.startswith("http"):
                    links.append(text)
    return links


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


def validate_link_evidence(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate output links overlap with visited-page link evidence.
    :param result: Test result.
    :param observability: Observability data.
    :return: Validation result.
    """
    final_text = extract_final_text(result)
    output_urls = set(re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text))
    visit_links = set(url for url in _visit_link_evidence(result) if "wikipedia.org" in url.lower())
    overlap = output_urls.intersection(visit_links)
    passed = len(overlap) >= 1
    return {
        "check": "link_evidence_overlap",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "output_url_count": len(output_urls),
        "visit_link_count": len(visit_links),
        "overlap_count": len(overlap),
        "reason": f"Found {len(overlap)} overlapping link(s)",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_link_count, validate_descriptions, validate_visits, validate_link_evidence]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
