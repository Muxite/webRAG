"""
Test 017: Recursive Link Analysis
Difficulty: 7/10 (Very Hard)
Category: Deep Analysis
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "017",
        "test_name": "Recursive Link Analysis",
        "difficulty_level": "7/10",
        "category": "Deep Analysis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Find a Wikipedia page about 'Cryptography'. From that page, identify 3 important concepts mentioned "
        "(e.g., encryption, public key, hash function). For each concept, follow links to find a related page, "
        "then from that related page, find one more link that connects back to cryptography or security. "
        "For each page in this recursive exploration, provide: URL, main topic, key information, and explain "
        "how it relates to the original cryptography page. Create a map showing how all these pages connect."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Original Cryptography page",
        "3 important concepts identified",
        "Related page for each concept",
        "Second-level link for each (connecting back)",
        "URL, topic, and key info for each page",
        "Relationship explanations",
        "Connection map",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Cryptography page visited",
        "3 concepts identified",
        "At least 6 pages total (1 start + 3 concepts + 2+ second-level)",
        "All relationships explained",
        "Connection map provided",
    ]


def validate_cryptography(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate cryptography page."""
    final_text = extract_final_text(result).lower()
    has_crypto = bool(re.search(r"\b(cryptograph|encryption|decryption|cipher)\b", final_text))
    return {
        "check": "cryptography_page",
        "passed": has_crypto,
        "score": 1.0 if has_crypto else 0.0,
        "reason": "Cryptography mentioned" if has_crypto else "Cryptography not mentioned",
    }


def validate_concepts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate concepts identified."""
    final_text = extract_final_text(result).lower()
    concepts = ["encryption", "public key", "hash", "cipher", "algorithm", "security", "key", "decryption"]
    found_concepts = sum(1 for c in concepts if c in final_text)
    passed = found_concepts >= 3
    return {
        "check": "concepts",
        "passed": passed,
        "score": min(1.0, found_concepts / 3.0),
        "concept_count": found_concepts,
        "reason": f"Found {found_concepts} concepts",
    }


def validate_pages(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sufficient pages visited."""
    final_text = extract_final_text(result)
    wikipedia_urls = re.findall(r"https?://[^\s)\\\"]*wikipedia\.org[^\s)\\\"]*", final_text)
    unique_urls = len(set(wikipedia_urls))
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = unique_urls >= 6 or visit_count >= 6
    return {
        "check": "page_count",
        "passed": passed,
        "score": min(1.0, max(unique_urls / 6.0, visit_count / 6.0)),
        "url_count": unique_urls,
        "visit_count": visit_count,
        "reason": f"Found {unique_urls} URLs, {visit_count} visits",
    }


def validate_relationships(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate relationships explained."""
    final_text = extract_final_text(result).lower()
    has_relate = bool(re.search(r"\b(relate|connect|link|associate|tie|link back|connect back)\b", final_text))
    has_map = bool(re.search(r"\b(map|diagram|structure|connection|network|graph|tree)\b", final_text))
    passed = has_relate and has_map
    return {
        "check": "relationships",
        "passed": passed,
        "score": 0.5 if has_relate else 0.0 + (0.5 if has_map else 0.0),
        "reason": f"Relationships: relate={has_relate}, map={has_map}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for recursive exploration quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this recursive link analysis task:

Task: {task}

Agent Output:
{final_text[:5000]}

Check:
1. Was Cryptography page visited?
2. Were 3 important concepts identified?
3. Were related pages found for each concept?
4. Were second-level links found (connecting back)?
5. Are all relationships explained?
6. Is a connection map provided?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "recursive_structure": boolean,
  "analysis_quality": string
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
    return [validate_cryptography, validate_concepts, validate_pages, validate_relationships]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
