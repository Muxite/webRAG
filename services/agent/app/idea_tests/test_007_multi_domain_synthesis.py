"""
Test 007: Multi-Domain Esoteric Synthesis
Difficulty: 7/10 (Hard)
Category: Cross-Domain Research
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "007",
        "test_name": "Multi-Domain Esoteric Synthesis",
        "difficulty_level": "7/10",
        "category": "Cross-Domain Research",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Research Teflon (PTFE): its chemical structure, historical development, health controversies, "
        "and current products that use it. Then synthesize information across these domains to explain "
        "how the chemistry relates to the health concerns and how modern products address these concerns. "
        "Cite sources from chemistry, history, health, and product domains."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Chemical structure information",
        "Historical development",
        "Health controversies",
        "Current products using Teflon",
        "Synthesis explaining chemistry-health-product connections",
        "Sources from multiple domains",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All four domains covered (chemistry, history, health, products)",
        "Synthesis connecting domains",
        "At least 5 authoritative URLs from different domains",
        "Technical accuracy",
    ]


def validate_domains(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate multiple domains covered."""
    final_text = extract_final_text(result).lower()
    has_chemistry = bool(re.search(r"\b(chemical|molecule|structure|ptfe|polymer|fluorine)\b", final_text))
    has_history = bool(re.search(r"\b(history|developed|invented|discovered|timeline|year)\b", final_text))
    has_health = bool(re.search(r"\b(health|controvers|toxic|cancer|pfoa|pfas|safety)\b", final_text))
    has_products = bool(re.search(r"\b(product|pan|cookware|coating|application|use)\b", final_text))
    domain_count = sum([has_chemistry, has_history, has_health, has_products])
    return {
        "check": "domain_coverage",
        "passed": domain_count >= 3,
        "score": domain_count / 4.0,
        "chemistry": has_chemistry,
        "history": has_history,
        "health": has_health,
        "products": has_products,
        "reason": f"Domains: chem={has_chemistry}, hist={has_history}, health={has_health}, prod={has_products}",
    }


def validate_synthesis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate synthesis present."""
    final_text = extract_final_text(result).lower()
    has_connection = bool(re.search(r"\b(relate|connect|link|because|due to|explain|synthesis|connect)\b", final_text))
    has_chemistry_health = bool(re.search(r"\b(chemical.*health|structure.*toxic|molecule.*safety)\b", final_text))
    return {
        "check": "synthesis",
        "passed": has_connection or has_chemistry_health,
        "score": 0.5 if has_connection else 0.0 + (0.5 if has_chemistry_health else 0.0),
        "reason": f"Synthesis: connection={has_connection}, chem-health={has_chemistry_health}",
    }


def validate_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate URLs from multiple domains."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    domain_urls = {
        "chemistry": [u for u in urls if any(x in u.lower() for x in [".edu", "chemistry", "scientific", "pubchem"])],
        "history": [u for u in urls if any(x in u.lower() for x in ["history", "museum", "archive", "timeline"])],
        "health": [u for u in urls if any(x in u.lower() for x in ["health", "medical", "fda", "who", "cancer"])],
        "product": [u for u in urls if any(x in u.lower() for x in ["product", "company", "manufacturer", "review"])],
    }
    unique_domains = sum(1 for d in domain_urls.values() if len(d) > 0)
    passed = len(urls) >= 5 and unique_domains >= 2
    return {
        "check": "multi_domain_urls",
        "passed": passed,
        "score": min(1.0, (len(urls) / 5.0) * 0.5 + (unique_domains / 4.0) * 0.5),
        "url_count": len(urls),
        "unique_domains": unique_domains,
        "reason": f"Found {len(urls)} URLs across {unique_domains} domains",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for synthesis quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this multi-domain synthesis task:

Task: {task}

Agent Output:
{final_text[:4000]}

Check:
1. Are all four domains covered (chemistry, history, health, products)?
2. Is there a synthesis connecting chemistry to health concerns?
3. Are modern product solutions mentioned?
4. Is technical accuracy maintained?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "all_domains_covered": boolean,
  "synthesis_quality": string,
  "technical_accuracy": boolean
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
    return [validate_domains, validate_synthesis, validate_urls]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
