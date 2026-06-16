"""
Test 042: Faithfulness & Contradiction (anti-parametric)
Difficulty: 9/10 (Very Hard)
Category: Faithfulness & Contradiction

Four claims about the discovery dates of obscure artefacts/objects; two are subtly FALSE.
The discovery YEARS are deliberately obscure — a small model cannot reliably recall them,
so it MUST visit the authoritative page to refute a false claim with the correct year.
This is exactly what the new `verify` action is for: cross-check a claim against gathered
evidence and surface the contradicting authority + the corrected fact.

`parametric` (no visits) cannot produce the correct obscure year and so fails the keystone;
`naive_rag` tends to visit the obvious page and may miss a claim. Scoring is gated/bimodal:
the keystone requires BOTH planted-false claims refuted *with their correct year*; secondary
checks short-circuit to 0.0 otherwise.

Ground truth (verified against live English Wikipedia, 2026-06):
  C1  4 Vesta discovered 1850          -> FALSE: discovered 1807 (Olbers)
  C2  Terracotta Army discovered 1974  -> TRUE  (control)
  C3  Antikythera mechanism found 1955 -> FALSE: recovered 1901
  C4  Lascaux paintings found 1940     -> TRUE  (control)
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CLAIMS = [
    {"id": "C1", "claim": "The asteroid 4 Vesta was discovered in the year 1850.",
     "truth": "FALSE", "url": "https://en.wikipedia.org/wiki/4_Vesta"},
    {"id": "C2", "claim": "The Terracotta Army was discovered in 1974.",
     "truth": "TRUE", "url": "https://en.wikipedia.org/wiki/Terracotta_Army"},
    {"id": "C3", "claim": "The Antikythera mechanism was recovered from a shipwreck in 1955.",
     "truth": "FALSE", "url": "https://en.wikipedia.org/wiki/Antikythera_mechanism"},
    {"id": "C4", "claim": "The Lascaux cave paintings were discovered in 1940.",
     "truth": "TRUE", "url": "https://en.wikipedia.org/wiki/Lascaux"},
]

_FALSE_MARKER = r"(false|incorrect|untrue|not\s+true|mistaken|debunk|wrong)"
_TRUE_MARKER = r"(true|correct|accurate|verified|confirmed)"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "042",
        "test_name": "Faithfulness & Contradiction",
        "difficulty_level": "9/10",
        "category": "Faithfulness & Contradiction",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    lines = [
        "You are a fact-checker. Some of the claims below are FALSE. The discovery years are "
        "obscure, so you must VERIFY each claim against the authoritative Wikipedia page rather "
        "than rely on memory. For EACH claim:\n"
        "  (a) Visit the relevant Wikipedia page and find the actual discovery year.\n"
        "  (b) Give a verdict: TRUE or FALSE.\n"
        "  (c) If FALSE, state the CORRECT year and cite the contradicting source URL with a short "
        "verbatim quote from that page.\n\n"
        "Claims:",
    ]
    for c in CLAIMS:
        lines.append(f'  {c["id"]}: "{c["claim"]}"  (source: {c["url"]})')
    lines.append(
        "\nProduce a verdict table (Claim | Verdict | Correct year | Source URL) and cite every URL."
    )
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    return [f"Verdict for {c['id']}" for c in CLAIMS] + [
        "Correct discovery year for each false claim",
        "Contradicting source URL per false claim",
        "Verdict table",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 visit actions executed",
        "C1 flagged FALSE with correct year 1807",
        "C3 flagged FALSE with correct year 1901",
        "C2 and C4 flagged TRUE",
        "Contradicting source URLs cited",
    ]


def _vesta_refuted(text: str) -> bool:
    return ("vesta" in text) and ("1807" in text) and bool(re.search(_FALSE_MARKER, text))


def _antikythera_refuted(text: str) -> bool:
    return ("antikythera" in text) and ("1901" in text) and bool(re.search(_FALSE_MARKER, text))


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return _vesta_refuted(text) and _antikythera_refuted(text)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    return {
        "check": "visit_count",
        "passed": visit_count >= 2,
        "score": min(1.0, visit_count / 4.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=2)",
    }


def validate_keystone_refutations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: both planted-false claims refuted with the correct obscure year. Hard 0/1."""
    text = extract_final_text(result).lower()
    v_ok = _vesta_refuted(text)
    a_ok = _antikythera_refuted(text)
    passed = v_ok and a_ok
    return {
        "check": "keystone_refutations",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": f"vesta_refuted(1807)={v_ok}, antikythera_refuted(1901)={a_ok}",
    }


def validate_true_claims(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Control claims must be flagged TRUE. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "true_controls", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> control claims not credited"}
    text = extract_final_text(result).lower()
    c2_ok = ("terracotta" in text) and ("1974" in text) and bool(re.search(_TRUE_MARKER, text))
    c4_ok = ("lascaux" in text) and ("1940" in text) and bool(re.search(_TRUE_MARKER, text))
    hits = int(c2_ok) + int(c4_ok)
    return {
        "check": "true_controls",
        "passed": hits == 2,
        "score": hits / 2.0,
        "reason": f"terracotta_true={c2_ok}, lascaux_true={c4_ok}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Contradicting source URLs for the false claims. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "contradiction_citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = extract_final_text(result).lower()
    has_vesta_url = bool(re.search(r"wiki/4_vesta", text))
    has_anti_url = bool(re.search(r"wiki/antikythera_mechanism", text))
    hits = int(has_vesta_url) + int(has_anti_url)
    return {
        "check": "contradiction_citations",
        "passed": hits >= 1,
        "score": hits / 2.0,
        "reason": f"vesta_url={has_vesta_url}, antikythera_url={has_anti_url}",
    }


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_refutations, validate_true_claims, validate_citations]


def get_llm_validation_function() -> callable:
    return None
