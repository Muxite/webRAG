"""
Test 001: Conflicting Information Resolution
Difficulty: 3/10 (Moderate)
Category: Data Verification & Analysis
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """
    Return test metadata.
    :return: Metadata dict.
    """
    return {
        "test_id": "001",
        "test_name": "Conflicting Information Resolution",
        "difficulty_level": "3/10",
        "category": "Data Verification & Analysis",
        "estimated_complexity": "Requires 3+ sources, discrepancy analysis, reasoning",
    }


def get_task_statement() -> str:
    """
    Return task statement for agent.
    :return: Task statement.
    """
    return (
        "Find the current population of San Francisco as reported by three different "
        "authoritative sources (government, news outlet, and research institution). "
        "For each source, provide: the population figure, the date of the data, and a direct citation URL. "
        "Then identify any discrepancies between the sources and explain which figure is most likely accurate and why."
    )


def get_required_deliverables() -> List[str]:
    """
    Return required deliverables.
    :return: List of deliverables.
    """
    return [
        "Three distinct population figures from different source types (government, news, research)",
        "For each source: population figure, date, and citation URL",
        "Discrepancy analysis identifying differences",
        "Explanation of why discrepancies exist",
        "Assessment of which figure is most accurate with reasoning",
    ]


def get_success_criteria() -> List[str]:
    """
    Return success criteria.
    :return: List of criteria.
    """
    return [
        "At least 5 search actions executed",
        "Three different source types identified (government, news, research)",
        "Each source has population figure, date, and URL",
        "Discrepancy analysis is present and coherent",
        "At least 3 distinct authoritative sources cited",
        "URLs are valid and accessible",
        "Analysis demonstrates understanding of data reliability",
    ]


def validate_search_actions(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate sufficient search actions executed.
    :param result: Test result.
    :param observability: Observability data.
    :return: Validation result.
    """
    search_count = observability.get("search", {}).get("count", 0)
    passed = search_count >= 5
    return {
        "check": "search_actions",
        "passed": passed,
        "score": min(1.0, search_count / 5.0),
        "actual": search_count,
        "expected": 5,
        "reason": f"Found {search_count} search actions, expected at least 5" if passed else f"Only {search_count} search actions, expected at least 5",
    }


def validate_sources(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate three different source types present.
    :param result: Test result.
    :param observability: Observability data.
    :return: Validation result.
    """
    final_text = extract_final_text(result)
    final_text_lower = final_text.lower()
    
    has_government = bool(re.search(r"\b(government|census|department of finance|\.gov|official|federal|state)\b", final_text_lower))
    has_news = bool(re.search(r"\b(news|article|reported|journalist|\.com/news|media outlet)\b", final_text_lower))
    has_research = bool(re.search(r"\b(research|university|institute|study|\.edu|academic|think tank)\b", final_text_lower))
    
    source_count = sum([has_government, has_news, has_research])
    passed = source_count >= 3
    
    return {
        "check": "source_types",
        "passed": passed,
        "score": source_count / 3.0,
        "government": has_government,
        "news": has_news,
        "research": has_research,
        "reason": f"Found {source_count} source types (gov: {has_government}, news: {has_news}, research: {has_research})" if passed else f"Only found {source_count} source types, expected 3",
    }


def validate_data_completeness(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate all required data fields present.
    :param result: Test result.
    :param observability: Observability data.
    :return: Validation result.
    """
    final_text = extract_final_text(result)
    
    has_population = bool(re.search(r"\b\d{1,3}[,\s]?\d{3}\b", final_text))
    has_dates = bool(re.search(r"\b(202[0-9]|january|february|march|april|may|june|july|august|september|october|november|december)\b", final_text, re.IGNORECASE))
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    has_urls = len(urls) >= 3
    
    checks_passed = sum([has_population, has_dates, has_urls])
    passed = checks_passed == 3
    
    return {
        "check": "data_completeness",
        "passed": passed,
        "score": checks_passed / 3.0,
        "has_population": has_population,
        "has_dates": has_dates,
        "has_urls": has_urls,
        "url_count": len(urls),
        "reason": f"Data completeness: population={has_population}, dates={has_dates}, urls={has_urls} ({len(urls)} URLs found)",
    }


def validate_discrepancy_analysis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate discrepancy analysis present.
    :param result: Test result.
    :param observability: Observability data.
    :return: Validation result.
    """
    final_text = extract_final_text(result)
    final_text_lower = final_text.lower()
    
    has_discrepancy = bool(re.search(r"\b(discrepan|differ|vary|conflict|inconsisten|disagre)\b", final_text_lower))
    has_explanation = bool(re.search(r"\b(why|because|reason|methodolog|timing|definition|due to)\b", final_text_lower))
    has_assessment = bool(re.search(r"\b(accurate|most likely|best|reliable|trustworthy|correct)\b", final_text_lower))
    
    checks_passed = sum([has_discrepancy, has_explanation, has_assessment])
    passed = checks_passed >= 2
    
    return {
        "check": "discrepancy_analysis",
        "passed": passed,
        "score": checks_passed / 3.0,
        "has_discrepancy": has_discrepancy,
        "has_explanation": has_explanation,
        "has_assessment": has_assessment,
        "reason": f"Analysis quality: discrepancy={has_discrepancy}, explanation={has_explanation}, assessment={has_assessment}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """
    LLM-based validation for complex reasoning.
    :param result: Test result.
    :param observability: Observability data.
    :param connector_llm: LLM connector.
    :param model_name: Model name.
    :return: Validation result.
    """
    final_text = extract_final_text(result)
    
    task = get_task_statement()
    criteria = get_success_criteria()
    
    prompt = f"""You are validating Test 001: Conflicting Information Resolution.

Task: {task}

Success Criteria:
{chr(10).join(f"- {c}" for c in criteria)}

Agent Output:
{final_text[:3000]}

Evaluate whether the agent's output meets the success criteria. Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "missing_requirements": [string],
  "source_verification": {{
    "government_source": boolean,
    "news_source": boolean,
    "research_source": boolean,
    "valid_urls": int
  }},
  "data_completeness": {{
    "all_figures": boolean,
    "all_dates": boolean,
    "all_urls": boolean
  }},
  "analysis_quality": {{
    "discrepancies_identified": boolean,
    "explanation_provided": boolean,
    "accuracy_assessed": boolean
  }}
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
            "missing_requirements": llm_result.get("missing_requirements", []),
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
    """
    Return validation functions to run.
    :return: List of validation functions.
    """
    return [
        validate_search_actions,
        validate_sources,
        validate_data_completeness,
        validate_discrepancy_analysis,
    ]


def get_llm_validation_function() -> callable:
    """
    Return LLM validation function.
    :return: LLM validation function or None.
    """
    return validate_with_llm
