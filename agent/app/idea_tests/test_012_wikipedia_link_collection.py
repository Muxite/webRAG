"""
Test 012: Wikipedia Link Collection
Difficulty: 3/10 (Moderate)
Category: Link Collection & Summarization
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, normalize_url, visited_link_urls


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


def _claimed_wikipedia_urls(result: Dict[str, Any]) -> List[str]:
    """Wikipedia URLs the answer claims, normalized so they can be compared against the evidence.

    Trailing punctuation matters: the extraction regex stops only at whitespace/`)`/`\\`/`"`, so a
    markdown or prose answer emitting ``.../wiki/Foo,`` or ``.../wiki/Foo.`` would otherwise never
    equal the normalized link the page really contained, and an honest agent would be marked
    ungrounded for its formatting.
    """
    raw = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", extract_final_text(result))
    return list(dict.fromkeys(normalize_url(u.rstrip(".,;:!?*|]}'\"")) for u in raw))


def _grounded_wikipedia_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> tuple:
    """(claimed, grounded) — grounded = claimed URLs the visited pages really contained."""
    claimed = _claimed_wikipedia_urls(result)
    evidence = visited_link_urls(result, observability)
    return claimed, [u for u in claimed if u in evidence]


def validate_link_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: only URLs genuinely evidenced by a visited page count toward the 10 required —
    otherwise padding 1 real link with 9 fabricated ones (which still pass the URL-shape regex)
    scores as if all 10 were real.

    Evidence is the page's UNCAPPED ``links_full`` (plus URLs appearing verbatim in the fetched
    text) rather than ``links``, which is capped to ``max_links_per_visit`` — as low as 5 for the
    sequential variant and under lean mode, where requiring 10 grounded links from the capped
    field alone would be arithmetically impossible for even a perfect agent.
    """
    claimed, grounded = _grounded_wikipedia_urls(result, observability)
    passed = len(grounded) >= 10
    return {
        "check": "link_count",
        "passed": passed,
        "score": min(1.0, len(grounded) / 10.0),
        "url_count": len(grounded),
        "reason": f"{len(grounded)} genuinely-evidenced Wikipedia URL(s) of {len(claimed)} claimed",
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
    GROUNDING (hard, ratio-scored): most of the claimed links must actually appear in the
    visited page's own link evidence, not just >=1 of them — a single real visit padded with
    9 fabricated links must NOT score full credit here. Score is the fraction of claimed links
    that are genuinely evidenced; passing requires that fraction to be high (>=0.8) AND at
    least 8 links evidenced, so bulk fabrication on top of one token-effort visit fails.
    :param result: Test result.
    :param observability: Observability data.
    :return: Validation result.
    """
    claimed, grounded = _grounded_wikipedia_urls(result, observability)
    ratio = (len(grounded) / len(claimed)) if claimed else 0.0
    passed = len(grounded) >= 8 and ratio >= 0.8
    return {
        "check": "link_evidence_overlap",
        "passed": passed,
        "score": ratio,
        "output_url_count": len(claimed),
        "overlap_count": len(grounded),
        "reason": f"{len(grounded)}/{len(claimed) or 0} claimed link(s) genuinely evidenced by a visited page",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_link_count, validate_descriptions, validate_visits, validate_link_evidence]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
