"""
Test 023: Sequential Data Gathering
Difficulty: 6/10 (Moderate-Hard)
Category: Sequential Processing
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import (
    extract_final_text, evidence_text as _evidence_text, visited_domains as _visited_domains,
    visited_evidence as _visited_evidence,
)


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "023",
        "test_name": "Sequential Data Gathering",
        "difficulty_level": "6/10",
        "category": "Sequential Processing",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Report the following about the 'Rust programming language': "
        "(1) the current stable version number, and (2) the installation method for your operating system. "
        "Both facts must come from official Rust sources (not Wikipedia), with citations."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Official Rust website URL found via search",
        "Official Rust website visited",
        "Current stable version number",
        "Installation guide found via search",
        "Installation guide page visited",
        "Installation method extracted",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 search actions executed",
        "At least 2 visit actions executed",
        "Rust official website mentioned",
        "Version number extracted",
        "Installation guide mentioned",
        "Installation method provided",
        "Sequential pattern evident (search -> visit -> search -> visit)",
    ]


def validate_searches(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate at least one search executed. Loosened from >=2: the task statement no longer
    prescribes a search->visit->search->visit path, so a route that reaches both grounded facts
    with fewer searches must not be scored down for it."""
    search_count = observability.get("search", {}).get("count", 0)
    passed = search_count >= 1
    return {
        "check": "multiple_searches",
        "passed": passed,
        "score": min(1.0, search_count / 1.0),
        "search_count": search_count,
        "reason": f"Found {search_count} search(es)" if passed else "Insufficient searches",
    }


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate at least two visits executed. Kept at >=2 (not relaxed further): the task's
    keystone facts genuinely live on two different pages (official site + install guide), so
    this reflects the destination, not a prescribed route."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 2
    return {
        "check": "multiple_visits",
        "passed": passed,
        "score": min(1.0, visit_count / 2.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)" if passed else "Insufficient visits",
    }


def validate_rust_official(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: the official Rust site only counts as found if rust-lang.org is one of the
    actually-visited domains — merely writing "rust-lang.org" in the answer is not evidence it
    was visited."""
    final_text = extract_final_text(result).lower()
    has_rust = "rust" in final_text
    has_official = "official" in final_text or "rust-lang.org" in final_text or "rustlang.org" in final_text
    domains = _visited_domains(result, observability)
    visited_official = any("rust-lang.org" in d or "rustlang.org" in d for d in domains)
    passed = has_rust and has_official and visited_official
    return {
        "check": "rust_official",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Rust official website genuinely visited" if passed else "Rust official site not actually visited",
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


def validate_installation_guide(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: a real second page visit (distinct from the official-site visit) must have
    happened — the task's own sequential shape requires search->visit->search->visit, so
    claiming an installation guide without a second real visit is unevidenced."""
    final_text = extract_final_text(result).lower()
    has_install = "install" in final_text
    has_guide = "guide" in final_text or "instructions" in final_text or "how to" in final_text
    second_visit = len(_visited_evidence(result, observability)) >= 2
    passed = has_install and has_guide and second_visit
    return {
        "check": "installation_guide",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Installation guide mentioned with a real second visit" if passed else "Installation guide missing or unevidenced by a real second visit",
    }


def validate_installation_method(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: the claimed installation method keyword must literally appear in the real
    fetched page content, not just in the model's own free-text answer."""
    final_text = extract_final_text(result).lower()
    method_keywords = ["rustup", "cargo", "package manager", "homebrew", "apt", "yum", "chocolatey", "download", "binary"]
    evidence_text = _evidence_text(result, observability)
    grounded_methods = [kw for kw in method_keywords if kw in final_text and kw in evidence_text]
    has_method = len(grounded_methods) > 0
    has_steps = "step" in final_text or "command" in final_text or "run" in final_text
    passed = has_method and has_steps
    return {
        "check": "installation_method",
        "passed": passed,
        "score": (0.5 if has_method else 0.0) + (0.5 if has_steps else 0.0),
        "reason": "Grounded installation method found" if passed else "Installation method missing or not grounded",
    }


def validate_sequential_pattern(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate genuine multi-step information gathering (>=1 search, >=2 visits) occurred.
    Loosened from a fixed search->visit->search->visit count: the statement no longer prescribes
    that exact shape, so this now checks that real work happened, not a specific step sequence."""
    search_count = observability.get("search", {}).get("count", 0)
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = search_count >= 1 and visit_count >= 2
    return {
        "check": "sequential_pattern",
        "passed": passed,
        "score": min(1.0, (search_count + visit_count) / 3.0),
        "search_count": search_count,
        "visit_count": visit_count,
        "reason": f"Sequential pattern: {search_count} searches, {visit_count} visits" if passed else "Sequential pattern not evident",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [
        validate_searches,
        validate_visits,
        validate_rust_official,
        validate_version,
        validate_installation_guide,
        validate_installation_method,
        validate_sequential_pattern,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
