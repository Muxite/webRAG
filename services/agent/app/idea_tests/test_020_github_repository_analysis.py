"""
Test 020: GitHub Repository Analysis
Difficulty: 4/10 (Moderate)
Category: Code Repository Analysis
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "020",
        "test_name": "GitHub Repository Analysis",
        "difficulty_level": "4/10",
        "category": "Code Repository Analysis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Visit the GitHub repository at https://github.com/python/cpython and extract the following information: "
        "(1) The primary programming language used, (2) The number of stars (approximate), (3) The license type, "
        "(4) A brief description of what the repository is. You must visit the GitHub page directly - do not rely on search results."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Primary programming language",
        "Number of stars",
        "License type",
        "Repository description",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "GitHub repository page visited",
        "Primary language identified (Python/C)",
        "Star count mentioned",
        "License identified",
        "Description provided",
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


def validate_github_visited(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate GitHub page was visited."""
    final_text = extract_final_text(result).lower()
    has_github = "github" in final_text
    has_cpython = "cpython" in final_text or "python" in final_text
    passed = has_github and has_cpython
    return {
        "check": "github_visited",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "GitHub cpython repository mentioned" if passed else "GitHub repository not mentioned",
    }


def validate_language(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate programming language identified."""
    final_text = extract_final_text(result).lower()
    has_python = "python" in final_text
    has_c = "c " in final_text or " c++" in final_text
    passed = has_python or has_c
    return {
        "check": "language",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Programming language mentioned" if passed else "Language not found",
    }


def validate_stars(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate star count mentioned."""
    final_text = extract_final_text(result)
    star_pattern = re.search(r"\b(\d+[kkm]?)\s*(star|⭐)", final_text, re.IGNORECASE)
    has_stars = bool(star_pattern)
    return {
        "check": "stars",
        "passed": has_stars,
        "score": 1.0 if has_stars else 0.0,
        "star_count": star_pattern.group(1) if star_pattern else None,
        "reason": f"Star count found: {star_pattern.group(1)}" if has_stars else "Star count not found",
    }


def validate_license(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate license identified."""
    final_text = extract_final_text(result).lower()
    has_license = bool(re.search(r"\b(license|apache|mit|gpl|bsd|psf)\b", final_text))
    return {
        "check": "license",
        "passed": has_license,
        "score": 1.0 if has_license else 0.0,
        "reason": "License mentioned" if has_license else "License not found",
    }


def validate_description(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate repository description provided."""
    final_text = extract_final_text(result)
    sentences = re.findall(r"[A-Z][^.!?]{30,}[.!?]", final_text)
    has_description = len(sentences) >= 1
    return {
        "check": "description",
        "passed": has_description,
        "score": min(1.0, len(sentences) / 2.0),
        "sentence_count": len(sentences),
        "reason": f"Found {len(sentences)} descriptive sentence(s)" if has_description else "Description not found",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [
        validate_visit_executed,
        validate_github_visited,
        validate_language,
        validate_stars,
        validate_license,
        validate_description,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
