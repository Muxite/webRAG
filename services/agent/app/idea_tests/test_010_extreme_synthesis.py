"""
Test 010: Extreme Multi-Domain Synthesis
Difficulty: 10/10 (Extremely Hard)
Category: Extreme Synthesis
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "010",
        "test_name": "Extreme Multi-Domain Synthesis",
        "difficulty_level": "10/10",
        "category": "Extreme Synthesis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Research and synthesize information across 5 domains: (1) Quantum computing principles and current "
        "capabilities, (2) Cryptography and encryption methods, (3) Climate change impact on computing infrastructure, "
        "(4) Economic implications of quantum computing adoption, and (5) Ethical considerations of quantum supremacy. "
        "Then create a comprehensive analysis explaining how these domains interconnect, what conflicts or synergies "
        "exist, and what the future landscape might look like. Each domain must have at least 2 authoritative sources. "
        "The synthesis must demonstrate deep understanding of cross-domain relationships."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Quantum computing principles and capabilities",
        "Cryptography and encryption methods",
        "Climate impact on computing",
        "Economic implications",
        "Ethical considerations",
        "Cross-domain interconnection analysis",
        "Conflict/synergy identification",
        "Future landscape prediction",
        "At least 10 authoritative sources across all domains",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All 5 domains thoroughly covered",
        "Synthesis demonstrating cross-domain understanding",
        "Conflicts and synergies identified",
        "Future predictions with reasoning",
        "At least 10 authoritative URLs",
        "Deep technical accuracy",
    ]


def validate_domains(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all 5 domains covered."""
    final_text = extract_final_text(result).lower()
    domains = {
        "quantum": bool(re.search(r"\b(quantum|qubit|superposition|entanglement|quantum computing)\b", final_text)),
        "crypto": bool(re.search(r"\b(cryptograph|encrypt|decrypt|rsa|aes|security|key)\b", final_text)),
        "climate": bool(re.search(r"\b(climate|carbon|energy|power consumption|environment|green)\b", final_text)),
        "economic": bool(re.search(r"\b(economic|cost|market|investment|adoption|business|financial)\b", final_text)),
        "ethical": bool(re.search(r"\b(ethic|moral|privacy|supremacy|societ|responsibility|concern)\b", final_text)),
    }
    domain_count = sum(domains.values())
    return {
        "check": "domain_coverage",
        "passed": domain_count >= 4,
        "score": domain_count / 5.0,
        "domains_covered": domain_count,
        "details": domains,
        "reason": f"Covered {domain_count}/5 domains",
    }


def validate_synthesis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate cross-domain synthesis."""
    final_text = extract_final_text(result).lower()
    has_interconnection = bool(re.search(r"\b(interconnect|relate|connect|link|relationship|between|across)\b", final_text))
    has_conflict = bool(re.search(r"\b(conflict|tension|challenge|problem|issue|dilemma)\b", final_text))
    has_synergy = bool(re.search(r"\b(synergy|benefit|advantage|opportunity|positive|enhance)\b", final_text))
    has_future = bool(re.search(r"\b(future|predict|forecast|will|might|could|scenario|landscape)\b", final_text))
    checks = sum([has_interconnection, has_conflict, has_synergy, has_future])
    return {
        "check": "synthesis_quality",
        "passed": checks >= 3,
        "score": checks / 4.0,
        "has_interconnection": has_interconnection,
        "has_conflict": has_conflict,
        "has_synergy": has_synergy,
        "has_future": has_future,
        "reason": f"Synthesis: interconnect={has_interconnection}, conflict={has_conflict}, synergy={has_synergy}, future={has_future}",
    }


def validate_sources(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extensive source coverage."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    authoritative = [u for u in urls if any(x in u.lower() for x in [".edu", ".gov", ".org", "nature", "science", "ieee", "acm", "arxiv", "research"])]
    passed = len(urls) >= 10 and len(authoritative) >= 7
    return {
        "check": "source_coverage",
        "passed": passed,
        "score": min(1.0, (len(urls) / 10.0) * 0.5 + (len(authoritative) / 7.0) * 0.5),
        "total_urls": len(urls),
        "authoritative_urls": len(authoritative),
        "reason": f"Found {len(urls)} URLs ({len(authoritative)} authoritative)",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for extreme synthesis quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this extreme multi-domain synthesis task:

Task: {task}

Agent Output:
{final_text[:6000]}

Check:
1. Are all 5 domains thoroughly covered?
2. Is there deep cross-domain interconnection analysis?
3. Are conflicts and synergies identified?
4. Is future landscape prediction present with reasoning?
5. Are at least 10 authoritative sources cited?
6. Is technical accuracy maintained across domains?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "all_domains_covered": boolean,
  "synthesis_depth": string,
  "cross_domain_understanding": string,
  "source_quality": string,
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
    return [validate_domains, validate_synthesis, validate_sources]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
