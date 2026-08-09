"""
Test 022: Technical Documentation Analysis
Difficulty: 5/10 (Moderate-Hard)
Category: Technical Documentation
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "022",
        "test_name": "Technical Documentation Analysis",
        "difficulty_level": "5/10",
        "category": "Technical Documentation",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Search for and visit the official documentation page for 'Docker' (not Wikipedia). "
        "Extract the following information: "
        "(1) What Docker is (brief definition), (2) The latest stable version number, "
        "(3) Three key features or capabilities, (4) A link to the installation guide. "
        "You must visit the official documentation website directly."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Definition of Docker",
        "Latest stable version number",
        "Three key features",
        "Installation guide link",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Official Docker documentation visited",
        "Definition provided",
        "Version number extracted",
        "At least 3 features listed",
        "Installation guide link found",
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


def validate_docker_visited(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Docker documentation visited."""
    final_text = extract_final_text(result).lower()
    has_docker = "docker" in final_text
    has_docs = "documentation" in final_text or "docs" in final_text
    passed = has_docker and has_docs
    return {
        "check": "docker_docs",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Docker documentation mentioned" if passed else "Docker docs not mentioned",
    }


def validate_definition(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate definition provided."""
    final_text = extract_final_text(result).lower()
    definition_keywords = ["container", "platform", "software", "application", "deploy"]
    has_definition = any(kw in final_text for kw in definition_keywords) and "docker" in final_text
    sentences = re.findall(r"[A-Z][^.!?]{30,}[.!?]", extract_final_text(result))
    has_description = len(sentences) >= 1
    passed = has_definition and has_description
    return {
        "check": "definition",
        "passed": passed,
        "score": 0.5 if has_definition else 0.0 + (0.5 if has_description else 0.0),
        "reason": "Definition found" if passed else "Definition missing",
    }


def validate_version(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate version number extracted."""
    final_text = extract_final_text(result)
    version_pattern = re.search(r"\b(\d+\.\d+\.\d+|\d+\.\d+)\b", final_text)
    has_version = bool(version_pattern)
    return {
        "check": "version",
        "passed": has_version,
        "score": 1.0 if has_version else 0.0,
        "version": version_pattern.group(1) if version_pattern else None,
        "reason": f"Version {version_pattern.group(1)} found" if has_version else "Version not found",
    }


def validate_features(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate features listed."""
    final_text = extract_final_text(result)
    feature_indicators = re.findall(r"(feature|capability|ability|benefit|advantage)[:]\s*([A-Z][^.!?]{10,})", final_text, re.IGNORECASE)
    numbered_features = re.findall(r"\b([1-3]\.|•|-)\s*([A-Z][^.!?]{15,})", final_text)
    total_features = len(feature_indicators) + len(numbered_features)
    passed = total_features >= 3
    return {
        "check": "features",
        "passed": passed,
        "score": min(1.0, total_features / 3.0),
        "feature_count": total_features,
        "reason": f"Found {total_features} feature(s)" if passed else "Insufficient features",
    }


def validate_installation_link(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate installation guide link found."""
    final_text = extract_final_text(result)
    url_pattern = re.findall(r"https?://[^\s)\\\"]+", final_text)
    has_install_url = any("install" in url.lower() or "guide" in url.lower() or "docs.docker.com" in url.lower() for url in url_pattern)
    has_install_text = "installation" in final_text.lower() and ("link" in final_text.lower() or "url" in final_text.lower() or len(url_pattern) > 0)
    passed = has_install_url or has_install_text
    return {
        "check": "installation_link",
        "passed": passed,
        "score": 1.0 if has_install_url else (0.5 if has_install_text else 0.0),
        "reason": "Installation link found" if passed else "Installation link missing",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [
        validate_visit_executed,
        validate_docker_visited,
        validate_definition,
        validate_version,
        validate_features,
        validate_installation_link,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
