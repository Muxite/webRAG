"""
Test 013: Wikipedia Exploration
Difficulty: 4/10 (Moderate)
Category: Exploration & Reasoning
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "013",
        "test_name": "Wikipedia Exploration",
        "difficulty_level": "4/10",
        "category": "Exploration & Reasoning",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Start at the Wikipedia page for 'Artificial Intelligence' (https://en.wikipedia.org/wiki/Artificial_intelligence) "
        "and explore by following links. Your goal is to reach a Wikipedia page about 'Quantum Computing' "
        "(https://en.wikipedia.org/wiki/Quantum_computing) by following links from page to page. "
        "Document your path: list each page you visit in order, explain why you chose each link, "
        "and show how the topics connect. You must reach the Quantum Computing page."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Path from AI to Quantum Computing",
        "List of pages visited in order",
        "Explanation for each link choice",
        "Connection between topics explained",
        "Final page is Quantum Computing",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Started at AI page",
        "Reached Quantum Computing page",
        "At least 3 intermediate pages visited",
        "Link choices explained",
        "Topic connections demonstrated",
    ]


def validate_start_page(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate started at AI page."""
    final_text = extract_final_text(result).lower()
    has_ai = bool(re.search(r"\b(artificial intelligence|ai\s+page|started at|beginning)\b", final_text))
    return {
        "check": "start_page",
        "passed": has_ai,
        "score": 1.0 if has_ai else 0.0,
        "reason": "AI page mentioned" if has_ai else "AI page not mentioned",
    }


def validate_end_page(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate reached Quantum Computing page."""
    final_text = extract_final_text(result).lower()
    has_quantum = bool(re.search(r"\b(quantum computing|quantum computer|reached|final|destination)\b", final_text))
    quantum_url = bool(re.search(r"quantum.*comput", final_text))
    passed = has_quantum or quantum_url
    return {
        "check": "end_page",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Quantum Computing page reached" if passed else "Quantum Computing page not reached",
    }


def validate_path(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate path with multiple pages."""
    final_text = extract_final_text(result)
    wikipedia_urls = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text)
    unique_urls = len(set(wikipedia_urls))
    passed = unique_urls >= 3
    return {
        "check": "path_length",
        "passed": passed,
        "score": min(1.0, unique_urls / 4.0),
        "url_count": unique_urls,
        "reason": f"Found {unique_urls} Wikipedia pages in path",
    }


def validate_explanations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate link choice explanations."""
    final_text = extract_final_text(result).lower()
    has_why = bool(re.search(r"\b(why|because|reason|chose|selected|link|connect|relate)\b", final_text))
    has_connection = bool(re.search(r"\b(connect|link|relate|path|journey|route|between)\b", final_text))
    passed = has_why and has_connection
    return {
        "check": "explanations",
        "passed": passed,
        "score": 0.5 if has_why else 0.0 + (0.5 if has_connection else 0.0),
        "reason": f"Explanations: why={has_why}, connections={has_connection}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for path quality and reasoning."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this Wikipedia exploration task:

Task: {task}

Agent Output:
{final_text[:3000]}

Check:
1. Did the agent start at AI page and reach Quantum Computing page?
2. Is there a clear path with multiple intermediate pages?
3. Are link choices explained?
4. Are topic connections demonstrated?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "path_complete": boolean,
  "reasoning_quality": string
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
    return [validate_start_page, validate_end_page, validate_path, validate_explanations]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
