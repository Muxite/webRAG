"""
Test 025: Wikipedia Link Chain Game
Difficulty: 2/10 (Easy-Moderate)
Category: Link Navigation & Verification
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "025",
        "test_name": "Wikipedia Link Chain Game",
        "difficulty_level": "2/10",
        "category": "Link Navigation & Verification",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Start at https://en.wikipedia.org/wiki/Main_Page and build a chain of at least 4 Wikipedia URLs. "
        "Each next URL must come from links found on the previous visited page. "
        "Return the chain in order, and include one verified adjacency pair near the end "
        "(show that URL[i] links to URL[i+1])."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "At least 4 ordered Wikipedia URLs",
        "A verified adjacency pair near the end of the chain",
        "Citation URLs for the visited pages",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 3 visit actions executed",
        "At least 4 distinct Wikipedia URLs in output",
        "At least one linked adjacency pair exists in visit-link evidence",
    ]


def _extract_visit_graph_data(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract visit nodes with URL and discovered links.
    :param result: Test result payload.
    :return: Visit node list.
    """
    graph = result.get("graph") or {}
    nodes = graph.get("nodes") or {}
    node_items = nodes.values() if isinstance(nodes, dict) else (nodes if isinstance(nodes, list) else [])
    visits: List[Dict[str, Any]] = []
    for node in node_items:
        if not isinstance(node, dict):
            continue
        details = node.get("details") or {}
        action_result = details.get("action_result") or details.get("actionResult") or {}
        if not isinstance(action_result, dict):
            continue
        if action_result.get("action") != "visit" or not action_result.get("success"):
            continue
        url = str(action_result.get("url") or "").strip()
        links = action_result.get("links") or []
        clean_links = [str(item).strip() for item in links if str(item).strip().startswith("http")]
        visits.append({"url": url, "links": clean_links})
    return visits


def validate_visit_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate visit count."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 3
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 3.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)",
    }


def validate_chain_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate chain URL count in output."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text)
    unique_urls = len(set(urls))
    passed = unique_urls >= 4
    return {
        "check": "chain_urls",
        "passed": passed,
        "score": min(1.0, unique_urls / 4.0),
        "url_count": unique_urls,
        "reason": f"Found {unique_urls} Wikipedia URL(s) in output",
    }


def validate_linked_adjacency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate at least one adjacency pair from visited page links.
    :param result: Test result payload.
    :param observability: Observability data.
    :return: Validation result.
    """
    visits = _extract_visit_graph_data(result)
    visited_urls = [item.get("url", "") for item in visits if item.get("url", "").startswith("http")]
    visited_set = set(visited_urls)
    linked_pairs = 0
    for visit in visits:
        src = visit.get("url", "")
        if not src:
            continue
        links = visit.get("links", [])
        for candidate in links:
            if candidate in visited_set and candidate != src:
                linked_pairs += 1
                break
    passed = linked_pairs >= 1
    return {
        "check": "linked_adjacency_evidence",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "visited_url_count": len(visited_set),
        "linked_pairs_found": linked_pairs,
        "reason": f"Found {linked_pairs} linked adjacency pair(s)",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_visit_count, validate_chain_urls, validate_linked_adjacency]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
