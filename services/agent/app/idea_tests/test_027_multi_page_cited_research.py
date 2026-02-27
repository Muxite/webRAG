"""
Test 027: Multi-Page Cited Research
Difficulty: 5/10 (Moderate)
Category: Heavy Data Collection & Citations

Requires the agent to visit multiple Wikipedia pages about different countries,
extract specific facts from each page, and cite the exact source URL for every fact.
Tests heavy multi-page data collection, accurate fact extraction, and citation quality.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "027",
        "test_name": "Multi-Page Cited Research",
        "difficulty_level": "5/10",
        "category": "Heavy Data Collection & Citations",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Research three countries by visiting their Wikipedia pages. "
        "Visit ALL THREE of these URLs:\n"
        "  1. https://en.wikipedia.org/wiki/France\n"
        "  2. https://en.wikipedia.org/wiki/Japan\n"
        "  3. https://en.wikipedia.org/wiki/Brazil\n\n"
        "For EACH country, extract from the actual page content:\n"
        "  (a) The capital city\n"
        "  (b) The official language(s)\n"
        "  (c) The approximate population (any recent figure from the page)\n\n"
        "Present your findings in a structured format with exactly 3 sections "
        "(one per country). Every fact MUST include a citation showing which URL "
        "it was extracted from. Do NOT use prior knowledge — only report what you "
        "find on the visited pages."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Capital city for each of the 3 countries",
        "Official language(s) for each of the 3 countries",
        "Population figure for each of the 3 countries",
        "URL citation for every fact",
        "Evidence that all 3 pages were visited",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 3 visit actions executed",
        "France Wikipedia page visited",
        "Japan Wikipedia page visited",
        "Brazil Wikipedia page visited",
        "Capital cities correctly identified (Paris, Tokyo, Brasília)",
        "Languages mentioned (French, Japanese, Portuguese)",
        "Population figures present",
        "Citation URLs present for facts",
    ]


# ── helpers ───────────────────────────────────────────────────────────────

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


# ── validation functions ──────────────────────────────────────────────────

def validate_visit_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate at least 3 visit actions executed."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 3
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 3.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)" if passed else f"Only {visit_count} visits (need ≥3)",
    }


def validate_france_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate France facts extracted."""
    final_text = extract_final_text(result).lower()
    has_paris = "paris" in final_text
    has_french = "french" in final_text
    has_france_url = bool(re.search(r"en\.wikipedia\.org/wiki/france", final_text, re.IGNORECASE))
    checks = int(has_paris) + int(has_french) + int(has_france_url)
    passed = checks >= 2
    return {
        "check": "france_facts",
        "passed": passed,
        "score": checks / 3.0,
        "has_paris": has_paris,
        "has_french": has_french,
        "has_france_url": has_france_url,
        "reason": f"France: capital={'✓' if has_paris else '✗'}, language={'✓' if has_french else '✗'}, citation={'✓' if has_france_url else '✗'}",
    }


def validate_japan_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Japan facts extracted."""
    final_text = extract_final_text(result).lower()
    has_tokyo = "tokyo" in final_text
    has_japanese = "japanese" in final_text
    has_japan_url = bool(re.search(r"en\.wikipedia\.org/wiki/japan", final_text, re.IGNORECASE))
    checks = int(has_tokyo) + int(has_japanese) + int(has_japan_url)
    passed = checks >= 2
    return {
        "check": "japan_facts",
        "passed": passed,
        "score": checks / 3.0,
        "has_tokyo": has_tokyo,
        "has_japanese": has_japanese,
        "has_japan_url": has_japan_url,
        "reason": f"Japan: capital={'✓' if has_tokyo else '✗'}, language={'✓' if has_japanese else '✗'}, citation={'✓' if has_japan_url else '✗'}",
    }


def validate_brazil_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Brazil facts extracted."""
    final_text = extract_final_text(result).lower()
    has_brasilia = bool(re.search(r"bras[ií]lia", final_text))
    has_portuguese = "portuguese" in final_text
    has_brazil_url = bool(re.search(r"en\.wikipedia\.org/wiki/brazil", final_text, re.IGNORECASE))
    checks = int(has_brasilia) + int(has_portuguese) + int(has_brazil_url)
    passed = checks >= 2
    return {
        "check": "brazil_facts",
        "passed": passed,
        "score": checks / 3.0,
        "has_brasilia": has_brasilia,
        "has_portuguese": has_portuguese,
        "has_brazil_url": has_brazil_url,
        "reason": f"Brazil: capital={'✓' if has_brasilia else '✗'}, language={'✓' if has_portuguese else '✗'}, citation={'✓' if has_brazil_url else '✗'}",
    }


def validate_population_figures(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate population figures present for all 3 countries."""
    final_text = extract_final_text(result)
    # Match large numbers that look like populations (millions/billions or raw digits)
    population_patterns = re.findall(r"\b\d[\d,. ]{4,}\d\b", final_text)
    million_mentions = len(re.findall(r"\b\d+[\d.]*\s*(million|billion|mil)\b", final_text, re.IGNORECASE))
    total_pop_evidence = len(population_patterns) + million_mentions
    passed = total_pop_evidence >= 3
    return {
        "check": "population_figures",
        "passed": passed,
        "score": min(1.0, total_pop_evidence / 3.0),
        "population_evidence_count": total_pop_evidence,
        "reason": f"Found {total_pop_evidence} population-like figure(s)" if passed else f"Only {total_pop_evidence} population figure(s) (need ≥3)",
    }


def validate_citation_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate citation URLs are present for the three countries."""
    final_text = extract_final_text(result).lower()
    has_france = bool(re.search(r"https?://en\.wikipedia\.org/wiki/france", final_text))
    has_japan = bool(re.search(r"https?://en\.wikipedia\.org/wiki/japan", final_text))
    has_brazil = bool(re.search(r"https?://en\.wikipedia\.org/wiki/brazil", final_text))
    cited = int(has_france) + int(has_japan) + int(has_brazil)
    passed = cited >= 3
    return {
        "check": "citation_urls",
        "passed": passed,
        "score": cited / 3.0,
        "cited_countries": cited,
        "has_france_citation": has_france,
        "has_japan_citation": has_japan,
        "has_brazil_citation": has_brazil,
        "reason": f"{cited}/3 country citation URLs found",
    }


async def validate_with_llm(
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm,
    model_name: str,
) -> Dict[str, Any]:
    """LLM validation for multi-page cited research quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)

    prompt = f"""Validate this multi-page research task:

Task: {task}

Agent Output:
{final_text[:6000]}

Observability:
- Visit actions executed: {visit_count}

Check:
1. Were all 3 Wikipedia pages actually visited (not just cited from memory)?
2. Are capital cities correct (Paris, Tokyo, Brasília)?
3. Are official languages correct (French, Japanese, Portuguese)?
4. Are population figures present and plausible?
5. Does every fact include a citation URL back to its source page?
6. Is the output well-structured with 3 clear sections?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "visit_evidence": boolean,
  "citation_quality": string,
  "fact_accuracy": string
}}"""

    try:
        messages = [
            {"role": "system", "content": "You are a test validator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        payload = connector_llm.build_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=0.1,
        )
        response = await connector_llm.client.chat.completions.create(**payload)
        content = response.choices[0].message.content
        llm_result = json.loads(content)
        return {
            "check": "llm_validation",
            "passed": llm_result.get("passed", False),
            "score": llm_result.get("score", 0.0),
            "reasons": llm_result.get("reasons", []),
            "details": llm_result,
        }
    except Exception as exc:
        return {
            "check": "llm_validation",
            "passed": False,
            "score": 0.0,
            "error": str(exc),
        }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [
        validate_visit_count,
        validate_france_facts,
        validate_japan_facts,
        validate_brazil_facts,
        validate_population_figures,
        validate_citation_urls,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
