"""
Test 021: News Article Extraction
Difficulty: 3/10 (Medium)
Category: News Content Analysis
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "021",
        "test_name": "News Article Extraction",
        "difficulty_level": "3/10",
        "category": "News Content Analysis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Search for recent news articles about 'artificial intelligence regulation' from the past year. "
        "Visit at least 2 different news websites (not Wikipedia) and extract: "
        "(1) The headline of each article, (2) The publication date, (3) The main topic or key point, "
        "(4) The source website name. Provide a summary comparing the different perspectives."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "At least 2 news articles from different sources",
        "Headline for each article",
        "Publication date for each",
        "Main topic/key point for each",
        "Source website names",
        "Comparison summary",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 different news websites visited",
        "Headlines extracted",
        "Dates mentioned",
        "Topics identified",
        "Source names provided",
        "Comparison summary included",
        "At least 2 visit actions executed",
    ]


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate multiple visits executed."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 2
    return {
        "check": "multiple_visits",
        "passed": passed,
        "score": min(1.0, visit_count / 2.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)" if passed else "Insufficient visits",
    }


def validate_news_sources(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate different news sources visited."""
    final_text = extract_final_text(result).lower()
    news_keywords = ["reuters", "bbc", "cnn", "guardian", "nytimes", "wsj", "ap news", "the verge", "techcrunch", "wired"]
    found_sources = [kw for kw in news_keywords if kw in final_text]
    passed = len(found_sources) >= 2
    return {
        "check": "news_sources",
        "passed": passed,
        "score": min(1.0, len(found_sources) / 2.0),
        "sources_found": found_sources,
        "reason": f"Found {len(found_sources)} news source(s)" if passed else "Insufficient news sources",
    }


def validate_headlines(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate headlines extracted."""
    final_text = extract_final_text(result)
    headline_pattern = re.findall(r"(headline|title):\s*([A-Z][^.!?]{20,})", final_text, re.IGNORECASE)
    has_headlines = len(headline_pattern) >= 2
    return {
        "check": "headlines",
        "passed": has_headlines,
        "score": min(1.0, len(headline_pattern) / 2.0),
        "headline_count": len(headline_pattern),
        "reason": f"Found {len(headline_pattern)} headline(s)" if has_headlines else "Headlines not found",
    }


def validate_dates(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate publication dates mentioned."""
    final_text = extract_final_text(result)
    date_pattern = re.findall(r"\b(202[3-5]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", final_text, re.IGNORECASE)
    has_dates = len(date_pattern) >= 2
    return {
        "check": "dates",
        "passed": has_dates,
        "score": min(1.0, len(date_pattern) / 2.0),
        "date_count": len(date_pattern),
        "reason": f"Found {len(date_pattern)} date(s)" if has_dates else "Dates not found",
    }


def validate_comparison(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate comparison summary provided."""
    final_text = extract_final_text(result).lower()
    comparison_keywords = ["compare", "comparison", "different", "perspective", "contrast", "similar", "differ"]
    has_comparison = any(kw in final_text for kw in comparison_keywords)
    sentences = re.findall(r"[A-Z][^.!?]{40,}[.!?]", extract_final_text(result))
    has_summary = len(sentences) >= 2
    passed = has_comparison and has_summary
    return {
        "check": "comparison",
        "passed": passed,
        "score": 0.5 if has_comparison else 0.0 + (0.5 if has_summary else 0.0),
        "reason": "Comparison summary found" if passed else "Comparison summary missing",
    }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [
        validate_visits,
        validate_news_sources,
        validate_headlines,
        validate_dates,
        validate_comparison,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return None
