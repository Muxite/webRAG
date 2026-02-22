"""
Test 019: Explicit Visit Requirement
Difficulty: 3/10 (Medium)
Category: URL Visiting Requirement
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "019",
        "test_name": "Explicit Visit Requirement",
        "difficulty_level": "3/10",
        "category": "URL Visiting Requirement",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Visit the Wikipedia page about 'Python (programming language)' at https://en.wikipedia.org/wiki/Python_(programming_language) "
        "and extract the following information directly from the page content: "
        "(1) The year Python was first released, (2) The name of Python's creator, and (3) The current stable version number. "
        "You MUST visit the URL - do not rely on search results. Provide the information with citations from the actual page content."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Year Python was first released",
        "Name of Python's creator",
        "Current stable version number",
        "Evidence that the Wikipedia page was visited",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Wikipedia page URL visited (https://en.wikipedia.org/wiki/Python_(programming_language))",
        "Year of first release extracted (1991)",
        "Creator name extracted (Guido van Rossum)",
        "Stable version number extracted",
        "At least 1 visit action executed",
    ]


def validate_visit_executed(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that visit action was executed."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 1
    return {
        "check": "visit_executed",
        "passed": passed,
        "score": min(1.0, visit_count / 2.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)" if passed else "No visits executed",
    }


def validate_wikipedia_url_visited(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the specific Wikipedia URL was visited."""
    final_text = extract_final_text(result).lower()
    has_python_wiki = "python" in final_text and "wikipedia" in final_text
    has_url = bool(re.search(r"en\.wikipedia\.org/wiki/python", final_text, re.IGNORECASE))
    passed = has_python_wiki or has_url
    return {
        "check": "wikipedia_url_visited",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Wikipedia Python page mentioned" if passed else "Wikipedia Python page not mentioned",
    }


def validate_year_extracted(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate year of first release extracted."""
    final_text = extract_final_text(result).lower()
    has_1991 = "1991" in final_text
    has_year = bool(re.search(r"\b(199[0-9]|200[0-9])\b", final_text))
    passed = has_1991 or (has_year and "first" in final_text and "release" in final_text)
    return {
        "check": "year_extracted",
        "passed": passed,
        "score": 1.0 if has_1991 else (0.5 if has_year else 0.0),
        "reason": "Year 1991 found" if has_1991 else ("Year mentioned" if has_year else "Year not found"),
    }


def validate_creator_extracted(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate creator name extracted."""
    final_text = extract_final_text(result).lower()
    has_guido = "guido" in final_text
    has_rossum = "rossum" in final_text
    has_creator = "creator" in final_text or "created by" in final_text
    passed = has_guido or (has_rossum and has_creator)
    return {
        "check": "creator_extracted",
        "passed": passed,
        "score": 1.0 if has_guido else (0.5 if has_rossum else 0.0),
        "reason": "Guido van Rossum found" if has_guido else ("Rossum mentioned" if has_rossum else "Creator not found"),
    }


def validate_version_extracted(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate stable version number extracted."""
    final_text = extract_final_text(result)
    version_pattern = re.search(r"\b(\d+\.\d+\.\d+|\d+\.\d+)\b", final_text)
    has_version = bool(version_pattern)
    has_stable = "stable" in final_text.lower() or "current" in final_text.lower()
    passed = has_version and has_stable
    return {
        "check": "version_extracted",
        "passed": passed,
        "score": 0.5 if has_version else 0.0 + (0.5 if has_stable else 0.0),
        "version": version_pattern.group(1) if version_pattern else None,
        "reason": f"Version {version_pattern.group(1)} found" if has_version else "Version not found",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for visit requirement."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)
    
    prompt = f"""Validate this explicit visit requirement task:

Task: {task}

Agent Output:
{final_text[:5000]}

Observability:
- Visit actions executed: {visit_count}

Check:
1. Did agent visit the Wikipedia URL (not just search)?
2. Was the year 1991 extracted?
3. Was Guido van Rossum identified as creator?
4. Was a stable version number extracted?
5. Is there evidence the page was actually visited (not just from search results)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "visit_evidence": boolean,
  "information_completeness": string
}}"""
    
    try:
        import json
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
    return [
        validate_visit_executed,
        validate_wikipedia_url_visited,
        validate_year_extracted,
        validate_creator_extracted,
        validate_version_extracted,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
