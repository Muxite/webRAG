"""
Test 015: Multi-Page Synthesis
Difficulty: 6/10 (Hard)
Category: Synthesis & Analysis
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "015",
        "test_name": "Multi-Page Synthesis",
        "difficulty_level": "6/10",
        "category": "Synthesis & Analysis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Find 5 different web pages about 'renewable energy' from different types of sources "
        "(news, government, research, company, organization). Visit each page, extract key information, "
        "and then synthesize the information to identify: (1) common themes across all sources, "
        "(2) conflicting viewpoints or data, (3) emerging trends mentioned, and (4) gaps in coverage. "
        "Provide URLs and citations for all pages."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "5 pages from different source types",
        "Key information from each page",
        "Common themes identified",
        "Conflicting viewpoints identified",
        "Emerging trends identified",
        "Gaps in coverage identified",
        "URLs for all pages",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All 5 source types represented",
        "Information extracted from each",
        "Synthesis demonstrates cross-page analysis",
        "Conflicts and trends identified",
        "At least 5 visit actions",
    ]


def validate_source_types(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate different source types."""
    final_text = extract_final_text(result).lower()
    types = {
        "news": bool(re.search(r"\b(news|article|media|journalism)\b", final_text)),
        "government": bool(re.search(r"\b(government|official|federal|state|\.gov)\b", final_text)),
        "research": bool(re.search(r"\b(research|study|university|academic|\.edu)\b", final_text)),
        "company": bool(re.search(r"\b(company|corporation|business|industry|commercial)\b", final_text)),
        "organization": bool(re.search(r"\b(organization|ngo|foundation|institute|non.?profit)\b", final_text)),
    }
    type_count = sum(types.values())
    return {
        "check": "source_types",
        "passed": type_count >= 4,
        "score": type_count / 5.0,
        "types_found": type_count,
        "details": types,
        "reason": f"Found {type_count}/5 source types",
    }


def validate_synthesis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate synthesis elements."""
    final_text = extract_final_text(result).lower()
    has_themes = bool(re.search(r"\b(theme|common|similar|pattern|consistent)\b", final_text))
    has_conflict = bool(re.search(r"\b(conflict|differ|disagree|oppose|contrast|contradict)\b", final_text))
    has_trends = bool(re.search(r"\b(trend|emerging|future|growing|increasing|developing)\b", final_text))
    has_gaps = bool(re.search(r"\b(gap|missing|lack|absence|uncover|not\s+mention)\b", final_text))
    checks = sum([has_themes, has_conflict, has_trends, has_gaps])
    return {
        "check": "synthesis",
        "passed": checks >= 3,
        "score": checks / 4.0,
        "has_themes": has_themes,
        "has_conflict": has_conflict,
        "has_trends": has_trends,
        "has_gaps": has_gaps,
        "reason": f"Synthesis: themes={has_themes}, conflict={has_conflict}, trends={has_trends}, gaps={has_gaps}",
    }


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate multiple pages visited."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 5
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 5.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit actions",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for synthesis quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this multi-page synthesis task:

Task: {task}

Agent Output:
{final_text[:5000]}

Check:
1. Are 5 different source types represented?
2. Is information extracted from each page?
3. Are common themes identified?
4. Are conflicting viewpoints identified?
5. Are emerging trends and gaps identified?
6. Is the synthesis coherent and analytical?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "source_diversity": boolean,
  "synthesis_quality": string
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
    return [validate_source_types, validate_synthesis, validate_visits]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
