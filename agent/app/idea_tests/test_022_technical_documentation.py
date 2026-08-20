"""
Test 022: Technical Documentation Analysis
Difficulty: 5/10 (Moderate-Hard)
Category: Technical Documentation
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import (
    extract_final_text, evidence_text as _evidence_text, visited_evidence as _visited_evidence,
)


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
        "Find Docker's official documentation (not Wikipedia) and verify the information there. "
        "Extract the following information: "
        "(1) What Docker is (brief definition), (2) The latest stable version number, "
        "(3) Three key features or capabilities, (4) A link to the installation guide."
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
    """GROUNDING: the definition's descriptive sentence must have real word-overlap with the
    actually-fetched page content, not just be a plausible-sounding sentence the model wrote."""
    final_text = extract_final_text(result).lower()
    definition_keywords = ["container", "platform", "software", "application", "deploy"]
    has_definition = any(kw in final_text for kw in definition_keywords) and "docker" in final_text
    sentences = re.findall(r"[A-Z][^.!?]{30,}[.!?]", extract_final_text(result))
    evidence_text = _evidence_text(result, observability)
    grounded_sentences = 0
    for s in sentences:
        words = re.findall(r"[a-zA-Z]{4,}", s.lower())
        hits = sum(1 for w in words if w in evidence_text)
        if words and hits / len(words) >= 0.4:
            grounded_sentences += 1
    has_description = grounded_sentences >= 1
    passed = has_definition and has_description
    return {
        "check": "definition",
        "passed": passed,
        "score": (0.5 if has_definition else 0.0) + (0.5 if has_description else 0.0),
        "reason": "Grounded definition found" if passed else "Definition missing or not grounded in visited content",
    }


def validate_version(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: the claimed version number must literally appear in the real fetched page
    content — a fabricated version string won't match what was actually on the page."""
    final_text = extract_final_text(result)
    version_pattern = re.search(r"\b(\d+\.\d+\.\d+|\d+\.\d+)\b", final_text)
    evidence_text = _evidence_text(result, observability)
    has_version = bool(version_pattern) and version_pattern.group(1) in evidence_text
    return {
        "check": "version",
        "passed": has_version,
        "score": 1.0 if has_version else 0.0,
        "version": version_pattern.group(1) if version_pattern else None,
        "reason": f"Version {version_pattern.group(1)} found and grounded" if has_version else "Version not found or not grounded",
    }


def validate_features(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: a claimed feature only counts if it has real word-overlap with the
    actually-fetched page content — otherwise the model can list plausible-sounding but
    invented "features"."""
    final_text = extract_final_text(result)
    feature_indicators = re.findall(r"(?:feature|capability|ability|benefit|advantage)[:]\s*([A-Z][^.!?]{10,})", final_text, re.IGNORECASE)
    numbered_features = re.findall(r"(?:[1-3]\.|•|-)\s*([A-Z][^.!?]{15,})", final_text)
    evidence_text = _evidence_text(result, observability)
    grounded = 0
    for feat in feature_indicators + numbered_features:
        words = re.findall(r"[a-zA-Z]{4,}", feat.lower())
        hits = sum(1 for w in words if w in evidence_text)
        if words and hits / len(words) >= 0.4:
            grounded += 1
    passed = grounded >= 3
    return {
        "check": "features",
        "passed": passed,
        "score": min(1.0, grounded / 3.0),
        "feature_count": grounded,
        "reason": f"Found {grounded} grounded feature(s)" if passed else "Insufficient grounded features",
    }


def validate_installation_link(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: the reported installation link only counts if it is one of the URLs actually
    visited, or literally appears in the real fetched page content — a plausible-looking
    docs.docker.com URL invented by the model does not count as "found"."""
    final_text = extract_final_text(result)
    url_pattern = re.findall(r"https?://[^\s)\\\"]+", final_text)
    evidence = _visited_evidence(result, observability)
    visited_urls = {e["url"] for e in evidence}
    evidence_text = _evidence_text(result, observability)
    has_install_url = any(
        ("install" in url.lower() or "guide" in url.lower() or "docs.docker.com" in url.lower())
        and (url in visited_urls or url in evidence_text)
        for url in url_pattern
    )
    return {
        "check": "installation_link",
        "passed": has_install_url,
        "score": 1.0 if has_install_url else 0.0,
        "reason": "Grounded installation link found" if has_install_url else "Installation link missing or not grounded",
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
