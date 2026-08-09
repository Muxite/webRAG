"""
Test 016: Topic Connection Exploration
Difficulty: 6/10 (Hard)
Category: Exploration & Reasoning
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "016",
        "test_name": "Topic Connection Exploration",
        "difficulty_level": "6/10",
        "category": "Exploration & Reasoning",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Start at a Wikipedia page about 'Machine Learning'. Explore by following links to find a connection "
        "to 'Neuroscience'. You must find at least one page that connects these two topics (e.g., neural networks, "
        "computational neuroscience, brain-computer interfaces). Document your exploration: list pages visited, "
        "explain how each link choice relates to finding the connection, and identify the connecting concept. "
        "The path should demonstrate understanding of how these fields relate."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Started at Machine Learning page",
        "Reached a page connecting ML and Neuroscience",
        "List of pages visited",
        "Explanation of link choices",
        "Connecting concept identified",
        "Relationship between fields explained",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Machine Learning page visited",
        "Connection page found",
        "At least 3 pages in exploration path",
        "Link choices explained with reasoning",
        "Connecting concept clearly identified",
    ]


def validate_start(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate started at ML page."""
    final_text = extract_final_text(result).lower()
    has_ml = bool(re.search(r"\b(machine learning|ml\s+page|started|beginning)\b", final_text))
    return {
        "check": "start_page",
        "passed": has_ml,
        "score": 1.0 if has_ml else 0.0,
        "reason": "ML page mentioned" if has_ml else "ML page not mentioned",
    }


def validate_connection(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate connection found."""
    final_text = extract_final_text(result).lower()
    has_neuro = bool(re.search(r"\b(neuroscience|neural|brain|neuro)\b", final_text))
    has_connection = bool(re.search(r"\b(connect|link|relate|bridge|neural network|computational neuroscience|bci|brain.?computer)\b", final_text))
    passed = has_neuro and has_connection
    return {
        "check": "connection",
        "passed": passed,
        "score": 0.5 if has_neuro else 0.0 + (0.5 if has_connection else 0.0),
        "reason": f"Connection: neuro={has_neuro}, connection={has_connection}",
    }


def validate_path(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate exploration path."""
    final_text = extract_final_text(result)
    wikipedia_urls = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text)
    unique_urls = len(set(wikipedia_urls))
    passed = unique_urls >= 3
    return {
        "check": "path",
        "passed": passed,
        "score": min(1.0, unique_urls / 4.0),
        "url_count": unique_urls,
        "reason": f"Found {unique_urls} pages in path",
    }


def validate_reasoning(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate reasoning quality."""
    final_text = extract_final_text(result).lower()
    has_explanation = bool(re.search(r"\b(why|because|reason|chose|select|relate|connect)\b", final_text))
    has_concept = bool(re.search(r"\b(concept|idea|connection|link|bridge|relate|field)\b", final_text))
    passed = has_explanation and has_concept
    return {
        "check": "reasoning",
        "passed": passed,
        "score": 0.5 if has_explanation else 0.0 + (0.5 if has_concept else 0.0),
        "reason": f"Reasoning: explanation={has_explanation}, concept={has_concept}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for exploration quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this topic connection exploration task:

Task: {task}

Agent Output:
{final_text[:4000]}

Check:
1. Did agent start at Machine Learning page?
2. Was a connection to Neuroscience found?
3. Is the connecting concept clearly identified?
4. Are link choices explained with reasoning?
5. Is the relationship between fields demonstrated?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "connection_found": boolean,
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
    return [validate_start, validate_connection, validate_path, validate_reasoning]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
