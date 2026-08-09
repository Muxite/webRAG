"""
Test m01 (bad-model lab, MICRO tier) — atomic single-page numeric extraction.

Anti-parametric by design (test-design memory): an OBSCURE lake's page-only max
depth that a weak model cannot recall — it must actually visit the page. Keystone
is a single specific number, so it is also cheap-executor-reliable. One entity,
so no mis-binding risk. Ground truth live-verified against English Wikipedia
(2026-07-22): Quesnel Lake maximum depth = 511 m (1,677 ft).
"""
from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text

PAGE_URL = "https://en.wikipedia.org/wiki/Quesnel_Lake"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "m01",
        "test_name": "Micro: obscure lake max depth (single-page)",
        "difficulty_level": "1/10",
        "category": "Bad-model lab micro tier",
        "level": "micro",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Visit this page: {PAGE_URL}\n"
        "Report the maximum depth of Quesnel Lake in metres (see its infobox). "
        "Give the number and cite the page URL."
    )


def get_required_deliverables() -> List[str]:
    return ["Maximum depth in metres", "Source URL"]


def get_success_criteria() -> List[str]:
    return ["Reports 511 (metres)", "Cites the source URL"]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return bool(re.search(r"(?<!\d)511(?!\d)", text))


def validate_keystone_depth(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the page-only max depth (511 m). Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_depth",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Reported 511 m" if passed else "Max depth 511 missing/incorrect",
    }


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Cited the source page. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/quesnel_lake", text))
    return {"check": "grounding", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source cited={cited}"}


def get_compiled_plan() -> Dict[str, Any]:
    """Trivial single-leaf hand plan so graph_compiled runs directly (no LLM scaffold
    authoring — which would misroute to the local ollama endpoint)."""
    return {
        "leaves": [{
            "id": "max_depth",
            "instruction": (f"Visit {PAGE_URL} and read the infobox. Report the maximum "
                            "depth of Quesnel Lake in metres."),
            "expect": "The maximum depth in metres (e.g. '511 m').",
            "depends_on": [],
        }],
        "aggregation": "Report the maximum depth in metres exactly as on the page, and cite the source URL.",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_depth, validate_grounding]


def get_llm_validation_function() -> callable:
    return None
