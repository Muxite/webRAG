"""
Test 021: News Article Extraction
Difficulty: 3/10 (Medium)
Category: News Content Analysis
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import (
    extract_final_text, evidence_text as _evidence_text, visited_domains as _visited_domains,
    visited_evidence as _visited_evidence,
)


_NEWS_DOMAINS = {
    "reuters": "reuters.com", "bbc": "bbc.co", "cnn": "cnn.com", "guardian": "theguardian.com",
    "nytimes": "nytimes.com", "wsj": "wsj.com", "ap news": "apnews.com",
    "the verge": "theverge.com", "techcrunch": "techcrunch.com", "wired": "wired.com",
}


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
    """GROUNDING: distinct real visited domains, not a raw activity counter — visiting the same
    page twice (or a failed visit) must not count as covering two different sources."""
    n = len(_visited_domains(result, observability))
    passed = n >= 2
    return {
        "check": "multiple_visits",
        "passed": passed,
        "score": min(1.0, n / 2.0),
        "visit_count": n,
        "reason": f"Found {n} distinct visited domain(s)" if passed else "Insufficient distinct visits",
    }


def validate_news_sources(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: a claimed news source only counts if that source's domain was actually
    visited — naming "Reuters" in the output text is not evidence Reuters was ever read."""
    final_text = extract_final_text(result).lower()
    domains = _visited_domains(result, observability)
    found_sources = [
        kw for kw, dom in _NEWS_DOMAINS.items()
        if kw in final_text and any(dom in d for d in domains)
    ]
    passed = len(found_sources) >= 2
    return {
        "check": "news_sources",
        "passed": passed,
        "score": min(1.0, len(found_sources) / 2.0),
        "sources_found": found_sources,
        "reason": f"Found {len(found_sources)} genuinely-visited news source(s)" if passed else "Insufficient grounded news sources",
    }


def validate_headlines(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: a reported headline only counts if most of its words actually appear in the
    real fetched content of a visited page — otherwise a model can invent a plausible-shaped
    "Headline: ..." line with no connection to anything it read."""
    final_text = extract_final_text(result)
    headline_pattern = re.findall(r"(?:headline|title):\s*([A-Z][^.!?]{20,})", final_text, re.IGNORECASE)
    evidence_text = _evidence_text(result, observability)
    grounded = 0
    for headline in headline_pattern:
        words = re.findall(r"[a-zA-Z]{4,}", headline.lower())
        hits = sum(1 for w in words if w in evidence_text)
        if words and hits / len(words) >= 0.5:
            grounded += 1
    passed = grounded >= 2
    return {
        "check": "headlines",
        "passed": passed,
        "score": min(1.0, grounded / 2.0),
        "headline_count": grounded,
        "reason": f"Found {grounded} evidence-grounded headline(s)" if passed else "Headlines not grounded in visited content",
    }


def validate_dates(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GROUNDING: a claimed publication date only counts if that literal date string appears
    in the real fetched content of a visited page."""
    final_text = extract_final_text(result)
    date_pattern = re.findall(
        r"\b(?:202[3-5]|jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\s+\d{1,2}\b"
        r"|\b\d{1,2}\s+(?:jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\b",
        final_text, re.IGNORECASE,
    )
    evidence_text = _evidence_text(result, observability)
    grounded = [d for d in date_pattern if d.lower() in evidence_text]
    passed = len(grounded) >= 2
    return {
        "check": "dates",
        "passed": passed,
        "score": min(1.0, len(grounded) / 2.0),
        "date_count": len(grounded),
        "reason": f"Found {len(grounded)} evidence-grounded date(s)" if passed else "Dates not grounded in visited content",
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
