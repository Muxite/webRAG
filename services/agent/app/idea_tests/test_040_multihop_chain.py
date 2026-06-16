"""
Test 040: Multi-Hop Dependent Chain
Difficulty: 8/10 (Hard)
Category: Multi-Hop Dependent Chain

The agent must follow a 3-page dependency chain where the URL for each hop is only
knowable after reading the previous page:

    Nineteen Eighty-Four (novel)  ->  author (George Orwell)
    George Orwell (person)        ->  birth town (Motihari)
    Motihari (town)               ->  administrative district (East Champaran)

This is structurally impossible for `naive_rag` (one fixed search round cannot discover
the Motihari hop, which only exists after reading Orwell's page) and yields ~0 for a
no-visit `parametric` baseline that cannot cite the chain of source pages. Scoring is
gated/bimodal: secondary checks short-circuit to 0.0 when the keystone (the final
district) is absent, so chain-completers separate sharply from everyone else.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


START_URL = "https://en.wikipedia.org/wiki/Nineteen_Eighty-Four"

# Ground truth (verified against live English Wikipedia, 2026-06):
#   author       = George Orwell
#   birth town   = Motihari
#   district     = East Champaran (a.k.a. Purba Champaran), Bihar
KEYSTONE_DISTRICT = r"(east\s+champaran|purba\s+champaran)"
HOP_AUTHOR = r"george\s+orwell"
HOP_TOWN = r"motihari"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "040",
        "test_name": "Multi-Hop Dependent Chain",
        "difficulty_level": "8/10",
        "category": "Multi-Hop Dependent Chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "Follow a chain of Wikipedia pages. Each step can only be answered by reading the "
        "previous page — do NOT guess from memory.\n"
        f"  1. Start at the novel's page: {START_URL}. Identify the novel's AUTHOR.\n"
        "  2. Visit the author's Wikipedia page. Identify the TOWN where the author was born.\n"
        "  3. Visit that town's Wikipedia page. Identify the administrative DISTRICT that the "
        "town is the headquarters of.\n\n"
        "Report: (a) the final district name, and (b) the full chain "
        "AUTHOR -> BIRTH TOWN -> DISTRICT, citing the exact URL of every page you visited."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Final administrative district name",
        "The author (hop 1)",
        "The author's birth town (hop 2)",
        "Source URL for each of the 3 pages visited",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 visit actions executed",
        "Correct author identified (George Orwell)",
        "Correct birth town identified (Motihari)",
        "Correct final district reported (East Champaran)",
        "Each hop's source URL cited",
    ]


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    return {
        "check": "visit_count",
        "passed": visit_count >= 3,
        "score": min(1.0, visit_count / 3.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=3 for a 3-hop chain)",
    }


def validate_keystone_district(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the final hop's district must be correct. Hard 0/1."""
    text = extract_final_text(result).lower()
    passed = bool(re.search(KEYSTONE_DISTRICT, text))
    return {
        "check": "keystone_final_district",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Final district 'East Champaran' present" if passed else "Final district missing/incorrect",
    }


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_DISTRICT, extract_final_text(result).lower()))


def validate_chain_intermediate(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Intermediate hops. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "chain_intermediate", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> intermediate hops not credited"}
    text = extract_final_text(result).lower()
    has_author = bool(re.search(HOP_AUTHOR, text))
    has_town = bool(re.search(HOP_TOWN, text))
    hits = int(has_author) + int(has_town)
    return {
        "check": "chain_intermediate",
        "passed": hits == 2,
        "score": hits / 2.0,
        "reason": f"author={has_author}, birth_town={has_town}",
    }


def validate_chain_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """The 3 source URLs. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "chain_urls", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    has_novel = bool(re.search(r"nineteen_eighty-four|/wiki/1984|nineteen-eighty", text))
    has_orwell = bool(re.search(r"wiki/george_orwell", text))
    has_town = bool(re.search(r"wiki/motihari", text))
    hits = int(has_novel) + int(has_orwell) + int(has_town)
    return {
        "check": "chain_urls",
        "passed": hits >= 2,
        "score": hits / 3.0,
        "reason": f"cited source urls: novel={has_novel}, orwell={has_orwell}, motihari={has_town}",
    }


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_district, validate_chain_intermediate, validate_chain_urls]


def get_llm_validation_function() -> callable:
    return None
