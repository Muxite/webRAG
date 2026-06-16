"""
Test 051: Tier 4 (extreme) — URL-free 3-hop Dependent Chain
Level: graph   Weight: long   Difficulty: 9/10

NO URLs are given. A 3-hop chain where each hop's target page is only knowable after
reading the previous page, and the FINAL fact (a university's founding year) is not present
in the author's or novel's snippets — it requires reaching the third page. This is
structurally impossible for a one-shot naive_rag round and is the headline graph-vs-linear
discriminator (model on test_040, but fully search-driven / no seed URL).

Ground truth (verified against live English Wikipedia, 2026-06):
  novel 'Things Fall Apart' -> author Chinua Achebe
  Chinua Achebe -> attended University College Ibadan (now University of Ibadan)
  University of Ibadan -> founded 1948
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "051",
        "test_name": "Tier 4: URL-free 3-hop Dependent Chain",
        "difficulty_level": "9/10",
        "category": "Search-Driven Dependent Chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs. Follow a research chain — each step can only be answered by "
        "reading the previous entity's page (do not guess from memory):\n"
        "  1. Identify the AUTHOR of the novel 'Things Fall Apart'.\n"
        "  2. Read that author's page; identify the UNIVERSITY they attended as an undergraduate.\n"
        "  3. Read that university's page; identify the YEAR it was founded/established.\n\n"
        "Report (a) the founding YEAR, and (b) the full chain AUTHOR -> UNIVERSITY -> FOUNDING "
        "YEAR, citing the exact URL of every page you visited."
    )


def get_required_deliverables() -> List[str]:
    return ["University founding year", "Author (hop 1)", "University (hop 2)", "Source URL per page"]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (search-driven chain)",
        "Correct author (Chinua Achebe)",
        "Correct university (University of Ibadan / University College Ibadan)",
        "Correct founding year (1948)",
        "Each hop's source URL cited",
    ]


KEYSTONE_YEAR = r"\b1948\b"
HOP_AUTHOR = r"chinua\s+achebe|achebe"
HOP_UNIV = r"ibadan"


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_YEAR, extract_final_text(result).lower()))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3 for a 3-hop chain; founding year needs the 3rd page)"}


def validate_keystone_year(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the university founding year (1948). Hard 0/1."""
    passed = _keystone_ok(result)
    return {"check": "keystone_year", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Founding year 1948 present" if passed else "Founding year (1948) missing/incorrect"}


def validate_chain_intermediate(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "chain_intermediate", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> intermediate hops not credited"}
    text = extract_final_text(result).lower()
    has_author = bool(re.search(HOP_AUTHOR, text))
    has_univ = bool(re.search(HOP_UNIV, text))
    hits = int(has_author) + int(has_univ)
    return {"check": "chain_intermediate", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"author={has_author}, university={has_univ}"}


def validate_chain_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "chain_urls", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    has_achebe = bool(re.search(r"wiki/chinua_achebe", text))
    has_ibadan = bool(re.search(r"wiki/university_of_ibadan|wiki/university_college_ibadan", text))
    hits = int(has_achebe) + int(has_ibadan)
    return {"check": "chain_urls", "passed": hits >= 1, "score": hits / 2.0,
            "reason": f"cited: achebe={has_achebe}, ibadan={has_ibadan}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_year, validate_chain_intermediate, validate_chain_urls]


def get_llm_validation_function() -> callable:
    return None
