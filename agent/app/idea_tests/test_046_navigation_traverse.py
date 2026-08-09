"""
Test 046: Navigation — Link-Following Traversal
Level: navigation   Weight: long   Difficulty: 7/10

Tests browsing intelligence, not just retrieval: starting from ONE page, the agent must
follow hyperlinks to reach a target page and extract a fact there. Adjacency is verified
objectively from the agent's OWN visit data (each visited page's full outgoing links), so
the agent cannot get credit by search-shortcutting or by reciting the answer from memory —
it must actually traverse the link from Apollo 11 to the launch vehicle's page.

The destination fact (the rocket's total height) is something the agent cannot answer from
the START page or memory — it must actually OPEN the Saturn V page — which is what forces
real navigation rather than a parametric shortcut (naming "Boeing" from memory is not enough).

Chain (verified against live English Wikipedia, 2026-06):
  Apollo 11  --(hyperlink)-->  Saturn V  (total height 111 m / 363 ft; first stage by Boeing)
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, build_visit_link_graph, normalize_url


START_URL = "https://en.wikipedia.org/wiki/Apollo_11"
HOP_URL = "https://en.wikipedia.org/wiki/Saturn_V"
TARGET_COMPANY = r"boeing"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "046",
        "test_name": "Navigation: Link-Following Traversal",
        "difficulty_level": "7/10",
        "category": "Navigation / Traversal",
        "level": "navigation",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        f"Start ONLY at this page: {START_URL}\n"
        "Navigate by FOLLOWING HYPERLINKS (do not use web search). Follow the link to the Wikipedia "
        "page of the rocket that launched the mission, OPEN that page, and read its infobox.\n\n"
        "Report: (a) the rocket's TOTAL HEIGHT in metres (from its infobox), (b) the company that "
        "built the rocket's FIRST STAGE, and (c) the ordered list of Wikipedia page URLs you visited "
        "(your path). Base the height on the page you open — do not guess."
    )


def get_required_deliverables() -> List[str]:
    return ["The rocket's total height in metres", "The first-stage contractor company",
            "Ordered list of visited page URLs (the path)"]


def get_success_criteria() -> List[str]:
    return [
        "Follows the link from Apollo 11 to the Saturn V page (verified adjacency)",
        "Reports the rocket height (111 m / 363 ft) from the opened page",
        "Identifies the company Boeing",
        "Efficient path (no excessive wandering)",
    ]


def _hop_visited(result: Dict[str, Any]) -> bool:
    _, visited = build_visit_link_graph(result)
    return normalize_url(HOP_URL) in visited


def _has_height(result: Dict[str, Any]) -> bool:
    low = extract_final_text(result).lower()
    return bool(re.search(r"\b363\b", low)) or bool(re.search(r"\b1(10|11)(\.\d+)?\s*m", low))


def _keystone_ok(result: Dict[str, Any]) -> bool:
    # Grounded navigation: the destination page was actually opened AND its height fact reported.
    return _hop_visited(result) and _has_height(result)


def validate_keystone_target(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: actually opened the Saturn V page AND reported its height fact. Hard 0/1."""
    visited = _hop_visited(result)
    height = _has_height(result)
    passed = visited and height
    return {
        "check": "keystone_destination_grounded",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": f"saturn_v_visited={visited}, height_reported={height}",
    }


def validate_path_adjacency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the Apollo 11 -> Saturn V hyperlink was really followed. Short-circuits on keystone."""
    if not _keystone_ok(result):
        return {"check": "path_adjacency", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> adjacency not credited"}
    link_map, visited = build_visit_link_graph(result)
    start = normalize_url(START_URL)
    hop = normalize_url(HOP_URL)
    start_visited = start in visited
    adjacency = hop in link_map.get(start, set())
    hits = int(start_visited) + int(adjacency)
    return {
        "check": "path_adjacency",
        "passed": adjacency and start_visited,
        "score": hits / 2.0,
        "reason": f"start_visited={start_visited}, apollo11_links_saturnV={adjacency}",
    }


def validate_company(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Secondary: the first-stage contractor. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "first_stage_company", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> company not credited"}
    named = bool(re.search(TARGET_COMPANY, extract_final_text(result).lower()))
    return {
        "check": "first_stage_company",
        "passed": named,
        "score": 1.0 if named else 0.0,
        "reason": f"boeing_named={named}",
    }


def validate_efficiency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Good surfers are efficient. Reward a short path. Short-circuits on keystone."""
    if not _keystone_ok(result):
        return {"check": "efficiency", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> efficiency not credited"}
    visits = observability.get("visit", {}).get("count", 0)
    # Ideal path is ~2 pages (Apollo 11 -> Saturn V). Allow some slack, decay after 4.
    if visits <= 3:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (visits - 3) * 0.25)
    return {
        "check": "efficiency",
        "passed": visits <= 4,
        "score": round(score, 3),
        "reason": f"{visits} visit(s) (ideal ~2)",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_target, validate_path_adjacency, validate_company, validate_efficiency]


def get_llm_validation_function() -> callable:
    return None
