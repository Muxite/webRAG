"""
Test 006: Obscure Historical Event
Difficulty: 6/10 (Hard)
Category: Historical Research
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "006",
        "test_name": "Obscure Historical Event",
        "difficulty_level": "6/10",
        "category": "Historical Research",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Research the 'Great Emu War' of Australia. Find primary source documents or historical records, "
        "modern analysis of the event, and quote at least one passage from a historical document or "
        "contemporary news article about it. Provide URLs to all sources."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Description of the Great Emu War",
        "Primary source or historical record",
        "Modern analysis",
        "At least one quoted passage from historical document/news",
        "URLs to all sources",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Event correctly identified (1932 Australia)",
        "Primary source or historical record found",
        "Modern analysis included",
        "Historical quote provided",
        "At least 3 authoritative URLs",
    ]


def validate_event(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate event identification."""
    final_text = extract_final_text(result).lower()
    has_emu = "emu" in final_text
    has_war = "war" in final_text
    has_australia = "australia" in final_text or "australian" in final_text
    has_1932 = "1932" in final_text
    checks = sum([has_emu, has_war, has_australia, has_1932])
    return {
        "check": "event_identification",
        "passed": checks >= 3,
        "score": checks / 4.0,
        "reason": f"Event checks: emu={has_emu}, war={has_war}, australia={has_australia}, 1932={has_1932}",
    }


def validate_quote(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate historical quote present."""
    final_text = extract_final_text(result)
    quote_patterns = [
        r'"[^"]{30,}"',
        r"'[^']{30,}'",
        r"according to.*[A-Z][^.!?]{30,}",
        r"reported.*[A-Z][^.!?]{30,}",
    ]
    quotes = []
    for pattern in quote_patterns:
        quotes.extend(re.findall(pattern, final_text, re.IGNORECASE))
    passed = len(quotes) >= 1
    return {
        "check": "historical_quote",
        "passed": passed,
        "score": min(1.0, len(quotes)),
        "quote_count": len(quotes),
        "reason": f"Found {len(quotes)} quotes",
    }


def validate_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate source URLs."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    historical_urls = [u for u in urls if any(x in u.lower() for x in [".edu", ".gov", "archive", "museum", "history", "national"])]
    passed = len(urls) >= 3
    return {
        "check": "source_urls",
        "passed": passed,
        "score": min(1.0, len(urls) / 3.0),
        "url_count": len(urls),
        "historical_urls": len(historical_urls),
        "reason": f"Found {len(urls)} URLs ({len(historical_urls)} historical)",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for primary sources and modern analysis."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this historical research task:

Task: {task}

Agent Output:
{final_text[:3000]}

Check:
1. Is a primary source or historical record mentioned?
2. Is modern analysis included?
3. Is the event correctly described (1932 Australia emu conflict)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "has_primary_source": boolean,
  "has_modern_analysis": boolean,
  "event_correct": boolean
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
    return [validate_event, validate_quote, validate_urls]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
