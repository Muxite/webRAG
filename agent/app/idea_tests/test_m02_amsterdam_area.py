"""
Test m02 (bad-model lab, MICRO tier) — atomic single-page numeric extraction.

Anti-parametric: an OBSCURE sub-Antarctic French island's page-only land area.
A weak model recalling from memory tends toward a round "55" — the precise
page value discriminates. Ground truth live-verified against English Wikipedia
(2026-07-22): Amsterdam Island land area = 56.6 km2 (21.9 sq mi).
"""
from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text

PAGE_URL = "https://en.wikipedia.org/wiki/Amsterdam_Island"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "m02",
        "test_name": "Micro: obscure island land area (single-page)",
        "difficulty_level": "1/10",
        "category": "Bad-model lab micro tier",
        "level": "micro",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Visit this page: {PAGE_URL}\n"
        "Report the land area of Amsterdam Island in square kilometres (see its infobox). "
        "Give the number and cite the page URL."
    )


def get_required_deliverables() -> List[str]:
    return ["Land area in km2", "Source URL"]


def get_success_criteria() -> List[str]:
    return ["Reports 56.6 (km2)", "Cites the source URL"]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return bool(re.search(r"(?<!\d)56\.6(?!\d)", text))


def validate_keystone_area(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the page-only land area (56.6 km2). Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_area",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Reported 56.6 km2" if passed else "Area 56.6 missing/incorrect",
    }


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Cited the source page. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    # The island's actual canonical English Wikipedia title is "Île Amsterdam" (the French
    # name) -- "Amsterdam_Island" is a redirect. A real, correct visit can land on either URL
    # (confirmed live: en.wikipedia.org/wiki/%C3%8Ele_Amsterdam returns HTTP 200, title "Île
    # Amsterdam - Wikipedia", not a broken link) and both slugs contain "amsterdam", so match
    # either rather than only the assumed-canonical one.
    cited = bool(re.search(r"wiki/[a-z0-9%_-]*amsterdam", text))
    return {"check": "grounding", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source cited={cited}"}


def get_compiled_plan() -> Dict[str, Any]:
    """Trivial single-leaf hand plan so graph_compiled runs directly (no LLM scaffold
    authoring — which would misroute to the local ollama endpoint)."""
    return {
        "leaves": [{
            "id": "land_area",
            "instruction": (f"Visit {PAGE_URL} and read the infobox. Report the land area "
                            "of Amsterdam Island in square kilometres."),
            "expect": "The land area in km2 (e.g. '56.6 km2').",
            "depends_on": [],
        }],
        "aggregation": "Report the land area in square kilometres exactly as on the page, and cite the source URL.",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_area, validate_grounding]


def get_llm_validation_function() -> callable:
    return None
