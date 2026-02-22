"""
Test 005: Social Media Analysis
Difficulty: 5/10 (Moderate-Hard)
Category: Social Media & Content Analysis
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "005",
        "test_name": "Social Media Analysis",
        "difficulty_level": "5/10",
        "category": "Social Media & Content Analysis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Visit a social media platform (Reddit, Twitter/X, or similar) and find a recent discussion "
        "about artificial intelligence. Recite what users are saying, quote at least 2 specific user "
        "comments, identify the users' apparent backgrounds or expertise levels based on their comments, "
        "and provide URLs to the discussion threads."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Social media platform identified",
        "Discussion topic about AI",
        "At least 2 quoted user comments",
        "User background/expertise assessment",
        "URLs to discussion threads",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Social media platform visited",
        "AI discussion found",
        "At least 2 user quotes provided",
        "User backgrounds assessed",
        "Thread URLs provided",
    ]


def validate_platform(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate social media platform visited."""
    final_text = extract_final_text(result).lower()
    platforms = ["reddit", "twitter", "x.com", "facebook", "linkedin", "discord"]
    found_platform = any(p in final_text for p in platforms)
    return {
        "check": "platform_visited",
        "passed": found_platform,
        "score": 1.0 if found_platform else 0.0,
        "reason": "Platform found" if found_platform else "No platform identified",
    }


def validate_quotes(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user quotes present."""
    final_text = extract_final_text(result)
    quote_patterns = [
        r'"[^"]{20,}"',
        r"'[^']{20,}'",
        r"said.*:.*[A-Z][^.!?]{20,}",
    ]
    quotes = []
    for pattern in quote_patterns:
        quotes.extend(re.findall(pattern, final_text, re.IGNORECASE))
    quote_count = len(quotes)
    passed = quote_count >= 2
    return {
        "check": "user_quotes",
        "passed": passed,
        "score": min(1.0, quote_count / 2.0),
        "quote_count": quote_count,
        "reason": f"Found {quote_count} quotes",
    }


def validate_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate discussion URLs."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    social_urls = [u for u in urls if any(x in u.lower() for x in ["reddit", "twitter", "x.com", "thread", "post", "comment"])]
    passed = len(social_urls) >= 1
    return {
        "check": "discussion_urls",
        "passed": passed,
        "score": min(1.0, len(social_urls)),
        "url_count": len(social_urls),
        "reason": f"Found {len(social_urls)} discussion URLs",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [validate_platform, validate_quotes, validate_urls]


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for user background assessment."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this social media analysis task:

Task: {task}

Agent Output:
{final_text[:3000]}

Check:
1. Are user backgrounds/expertise levels identified?
2. Is the AI discussion topic clear?
3. Are quotes properly attributed or contextualized?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "has_background_assessment": boolean,
  "has_ai_discussion": boolean,
  "quotes_contextualized": boolean
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


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
