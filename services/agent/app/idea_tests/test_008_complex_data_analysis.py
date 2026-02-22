"""
Test 008: Complex Data Analysis
Difficulty: 8/10 (Very Hard)
Category: Data Analysis & Statistics
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "008",
        "test_name": "Complex Data Analysis",
        "difficulty_level": "8/10",
        "category": "Data Analysis & Statistics",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Find the top 5 programming languages by GitHub repository count in 2024. For each language, "
        "provide: the exact repository count, the percentage change from 2023, the primary use cases, "
        "and identify which language had the largest percentage growth. Then analyze trends and predict "
        "which language might overtake the current leader in the next 2 years, with reasoning. Cite all data sources."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Top 5 languages with repository counts",
        "Percentage change from 2023 for each",
        "Primary use cases for each language",
        "Language with largest growth identified",
        "Trend analysis and prediction",
        "Data source citations",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All 5 languages identified with counts",
        "Percentage changes calculated",
        "Use cases provided",
        "Growth leader identified",
        "Trend analysis with reasoning",
        "At least 3 data source URLs",
    ]


def validate_languages(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate top 5 languages present."""
    final_text = extract_final_text(result).lower()
    common_languages = ["javascript", "python", "java", "typescript", "c++", "c#", "php", "go", "rust", "ruby"]
    found_languages = [lang for lang in common_languages if lang in final_text]
    passed = len(found_languages) >= 5
    return {
        "check": "languages_count",
        "passed": passed,
        "score": min(1.0, len(found_languages) / 5.0),
        "found_count": len(found_languages),
        "reason": f"Found {len(found_languages)} languages",
    }


def validate_numbers(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate repository counts and percentages."""
    final_text = extract_final_text(result)
    large_numbers = re.findall(r"\b\d{1,3}[,\s]?\d{3,}\b", final_text)
    percentages = re.findall(r"\b\d+\.?\d*\s*%", final_text)
    has_counts = len(large_numbers) >= 5
    has_percentages = len(percentages) >= 3
    return {
        "check": "data_numbers",
        "passed": has_counts and has_percentages,
        "score": (0.5 if has_counts else 0.0) + (0.5 if has_percentages else 0.0),
        "counts": len(large_numbers),
        "percentages": len(percentages),
        "reason": f"Found {len(large_numbers)} counts, {len(percentages)} percentages",
    }


def validate_analysis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trend analysis."""
    final_text = extract_final_text(result).lower()
    has_trend = bool(re.search(r"\b(trend|growth|increase|decrease|change|pattern)\b", final_text))
    has_prediction = bool(re.search(r"\b(predict|forecast|future|overtake|surpass|next|years?)\b", final_text))
    has_reasoning = bool(re.search(r"\b(because|reason|due to|likely|probably|evidence)\b", final_text))
    checks = sum([has_trend, has_prediction, has_reasoning])
    return {
        "check": "trend_analysis",
        "passed": checks >= 2,
        "score": checks / 3.0,
        "has_trend": has_trend,
        "has_prediction": has_prediction,
        "has_reasoning": has_reasoning,
        "reason": f"Analysis: trend={has_trend}, prediction={has_prediction}, reasoning={has_reasoning}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for data accuracy and analysis quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this data analysis task:

Task: {task}

Agent Output:
{final_text[:4000]}

Check:
1. Are 5 languages identified with repository counts?
2. Are percentage changes from 2023 provided?
3. Is the growth leader identified?
4. Is there a prediction with reasoning?
5. Are data sources cited?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "languages_complete": boolean,
  "data_accuracy": string,
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
    return [validate_languages, validate_numbers, validate_analysis]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
