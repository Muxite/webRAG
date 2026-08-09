"""
Test 033: Dual-Niche Comparison
Difficulty: 5/10 (Moderate-Hard)
Category: Branching Comparison

Forces the agent to research two unrelated niche topics independently
(tardigrades in astrobiology AND Brutalist architecture) then produce a
structured comparison of their resilience/durability themes.  A good
graph agent will branch into two parallel research paths and merge at
the end.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "033",
        "test_name": "Dual-Niche Comparison",
        "difficulty_level": "5/10",
        "category": "Branching Comparison",
    }


def get_task_statement() -> str:
    return (
        "Research TWO separate topics and then compare them:\n"
        "Topic A – Tardigrades in astrobiology: visit https://en.wikipedia.org/wiki/Tardigrade "
        "and find (1) at least two extreme conditions tardigrades survive, "
        "(2) one named space-exposure experiment.\n"
        "Topic B – Brutalist architecture: visit https://en.wikipedia.org/wiki/Brutalist_architecture "
        "and find (1) at least two defining characteristics of the style, "
        "(2) one specific building cited as an example.\n"
        "Finally, write a short comparative paragraph (100+ words) that draws a thematic parallel "
        "between tardigrade resilience and Brutalist durability. "
        "Cite the exact Wikipedia URL for every fact."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Two extreme conditions tardigrades survive",
        "One named space-exposure experiment",
        "Two defining characteristics of Brutalist architecture",
        "One specific Brutalist building example",
        "Comparative paragraph (100+ words) linking resilience themes",
        "Wikipedia URL citations for each fact",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 visit actions (one per topic)",
        "Tardigrade survival conditions mentioned",
        "Space experiment named",
        "Brutalist characteristics mentioned",
        "Specific building named",
        "Comparative paragraph present",
        "Both Wikipedia URLs cited",
    ]


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 2
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 2.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)",
    }


def validate_tardigrade_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    conditions = sum(1 for kw in [
        "radiation", "vacuum", "dehydration", "desiccation", "temperature",
        "freezing", "pressure", "space", "boiling",
    ] if kw in text)
    has_experiment = bool(re.search(
        r"(tardis|tardigrade.*experiment|biokis|foton|expose|phobos)", text
    ))
    score = min(1.0, (min(conditions, 2) / 2.0) * 0.6 + (0.4 if has_experiment else 0.0))
    return {
        "check": "tardigrade_facts",
        "passed": conditions >= 2 and has_experiment,
        "score": score,
        "conditions_found": conditions,
        "has_experiment": has_experiment,
        "reason": f"conditions={conditions}, experiment={has_experiment}",
    }


def validate_brutalist_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    characteristics = sum(1 for kw in [
        "raw concrete", "béton brut", "beton brut", "monolithic", "massive",
        "exposed concrete", "geometric", "fortress", "modular", "repetitive",
        "unfinished", "rough", "heavyweight",
    ] if kw in text)
    buildings = sum(1 for b in [
        "barbican", "trellick", "habitat 67", "national theatre", "unite",
        "unité", "boston city hall", "breuer", "robarts", "balfron",
        "alexandra road", "sirius", "western city gate",
    ] if b in text)
    score = min(1.0, (min(characteristics, 2) / 2.0) * 0.5 + (min(buildings, 1)) * 0.5)
    return {
        "check": "brutalist_facts",
        "passed": characteristics >= 2 and buildings >= 1,
        "score": score,
        "characteristics_found": characteristics,
        "buildings_found": buildings,
        "reason": f"characteristics={characteristics}, buildings={buildings}",
    }


def validate_comparison(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    lower = text.lower()
    has_tardigrade_ref = "tardigrade" in lower or "water bear" in lower
    has_brutalist_ref = "brutalist" in lower or "brutalism" in lower
    has_parallel = bool(re.search(
        r"(both|similarly|parallel|alike|resilien|durabil|endur|withstand|surviv)", lower
    ))
    comparison_section = ""
    for paragraph in text.split("\n\n"):
        if ("tardigrade" in paragraph.lower() or "water bear" in paragraph.lower()) and \
           ("brutalist" in paragraph.lower() or "brutalism" in paragraph.lower()):
            comparison_section = paragraph
            break
    word_count = len(comparison_section.split()) if comparison_section else 0
    length_ok = word_count >= 80
    score_parts = (
        (0.25 if has_tardigrade_ref else 0.0)
        + (0.25 if has_brutalist_ref else 0.0)
        + (0.25 if has_parallel else 0.0)
        + (0.25 if length_ok else min(0.25, word_count / 100.0 * 0.25))
    )
    return {
        "check": "comparison_paragraph",
        "passed": has_tardigrade_ref and has_brutalist_ref and has_parallel and length_ok,
        "score": score_parts,
        "word_count": word_count,
        "reason": f"tardigrade={has_tardigrade_ref}, brutalist={has_brutalist_ref}, parallel={has_parallel}, words={word_count}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    has_tardigrade_url = "wikipedia.org/wiki/tardigrade" in text
    has_brutalist_url = "wikipedia.org/wiki/brutalist" in text
    score = (0.5 if has_tardigrade_url else 0.0) + (0.5 if has_brutalist_url else 0.0)
    return {
        "check": "citation_urls",
        "passed": has_tardigrade_url and has_brutalist_url,
        "score": score,
        "reason": f"tardigrade_url={has_tardigrade_url}, brutalist_url={has_brutalist_url}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this dual-niche comparison task:

Task: {task}

Agent Output:
{final_text[:5000]}

Check:
1. Are two extreme tardigrade survival conditions named?
2. Is a space-exposure experiment named?
3. Are two Brutalist architectural characteristics identified?
4. Is a specific Brutalist building named?
5. Is there a comparative paragraph (100+ words) linking resilience themes?
6. Are both Wikipedia URLs cited?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "tardigrade_quality": string,
  "brutalist_quality": string,
  "comparison_quality": string
}}"""
    try:
        messages = [
            {"role": "system", "content": "You are a test validator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        payload = connector_llm.build_payload(
            messages=messages, json_mode=True, model_name=model_name, temperature=0.1,
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
        return {"check": "llm_validation", "passed": False, "score": 0.0, "error": str(exc)}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_tardigrade_facts, validate_brutalist_facts, validate_comparison, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
