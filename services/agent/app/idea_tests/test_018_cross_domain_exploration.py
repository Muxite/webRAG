"""
Test 018: Cross-Domain Exploration
Difficulty: 7/10 (Very Hard)
Category: Cross-Domain Exploration
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "018",
        "test_name": "Cross-Domain Exploration",
        "difficulty_level": "7/10",
        "category": "Cross-Domain Exploration",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Start at the Wikipedia page about 'Photosynthesis' (https://en.wikipedia.org/wiki/Photosynthesis) "
        "and explore by following links to reach the Wikipedia page about 'Tiger' (https://en.wikipedia.org/wiki/Tiger). "
        "Document your complete path trace: list every page you visit in order with URLs, explain why you chose each link at each step, "
        "and show how the topics connect from photosynthesis to tiger. You must reach the Tiger page. Provide the full path trace for all pages visited."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Started at Photosynthesis Wikipedia page",
        "Reached Tiger Wikipedia page",
        "Complete path trace (all pages visited in order)",
        "URL for each page in the path",
        "Explanation for each link choice",
        "Connection between topics explained",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Photosynthesis page visited",
        "Tiger page reached",
        "Complete path trace provided",
        "At least 3 intermediate pages in path",
        "Link choices explained with reasoning",
        "Topic connections demonstrated",
    ]


def validate_start(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate started at Photosynthesis."""
    final_text = extract_final_text(result).lower()
    has_photo = bool(re.search(r"\b(photosynthesis|started|beginning)\b", final_text))
    return {
        "check": "start_page",
        "passed": has_photo,
        "score": 1.0 if has_photo else 0.0,
        "reason": "Photosynthesis mentioned" if has_photo else "Photosynthesis not mentioned",
    }


def validate_end(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate reached Tiger page."""
    final_text = extract_final_text(result).lower()
    has_tiger = bool(re.search(r"\b(tiger|reached|final|destination|end)\b", final_text))
    tiger_url = bool(re.search(r"tiger", final_text))
    passed = has_tiger or tiger_url
    return {
        "check": "end_page",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Tiger page reached" if passed else "Tiger page not reached",
    }


def validate_path_trace(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete path trace."""
    final_text = extract_final_text(result)
    wikipedia_urls = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text)
    unique_urls = len(set(wikipedia_urls))
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = unique_urls >= 4 or visit_count >= 4
    return {
        "check": "path_trace",
        "passed": passed,
        "score": min(1.0, max(unique_urls / 5.0, visit_count / 5.0)),
        "url_count": unique_urls,
        "visit_count": visit_count,
        "reason": f"Found {unique_urls} URLs, {visit_count} visits in path",
    }


def validate_explanations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate link choice explanations."""
    final_text = extract_final_text(result).lower()
    has_why = bool(re.search(r"\b(why|because|reason|chose|selected|link|step)\b", final_text))
    has_connect = bool(re.search(r"\b(connect|relate|link|bridge|associate|tie|path|journey)\b", final_text))
    passed = has_why and has_connect
    return {
        "check": "explanations",
        "passed": passed,
        "score": 0.5 if has_why else 0.0 + (0.5 if has_connect else 0.0),
        "reason": f"Explanations: why={has_why}, connections={has_connect}",
    }




async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for path exploration quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this Wikipedia exploration task:

Task: {task}

Agent Output:
{final_text[:5000]}

Check:
1. Did agent start at Photosynthesis page?
2. Did agent reach Tiger page?
3. Is a complete path trace provided (all pages in order)?
4. Are link choices explained with reasoning?
5. Are topic connections demonstrated (how photosynthesis connects to tiger)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "path_complete": boolean,
  "reasoning_quality": string,
  "connection_quality": string
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
    return [validate_start, validate_end, validate_path_trace, validate_explanations]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
