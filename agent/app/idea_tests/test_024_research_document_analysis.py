"""
Test 024: Research Document Analysis
Difficulty: 7/10 (Hard)
Category: Research Document Retrieval
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "024",
        "test_name": "Research Document Analysis",
        "difficulty_level": "7/10",
        "category": "Research Document Retrieval",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Find and analyze research documents about 'machine learning interpretability' or 'explainable AI' from academic sources. "
        "You must: (1) Search for recent research papers or academic articles (from 2020 onwards), "
        "(2) Visit at least 2 different research sources (arXiv, research institution websites, academic journals, or conference papers), "
        "(3) Extract from each source: the title, authors (at least one author name), publication year, and main research contribution or finding. "
        "(4) Provide a summary comparing the different research approaches or findings. "
        "Focus on actual research documents, not Wikipedia or general news articles."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "At least 2 research documents from academic sources",
        "Title for each document",
        "Author name(s) for each",
        "Publication year (2020 or later)",
        "Main research contribution/finding for each",
        "Comparison summary of research approaches",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 research sources visited",
        "Research document titles extracted",
        "Author names identified",
        "Publication years mentioned (2020+)",
        "Research contributions/findings described",
        "Comparison summary provided",
        "Academic/research sources used (not Wikipedia/news)",
    ]


def validate_research_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate multiple research source visits."""
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 2
    return {
        "check": "research_visits",
        "passed": passed,
        "score": min(1.0, visit_count / 2.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)" if passed else "Insufficient visits",
    }


def validate_research_sources(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate academic/research sources used."""
    final_text = extract_final_text(result).lower()
    research_keywords = [
        "arxiv", "research", "paper", "journal", "conference", "academic", "university",
        "institute", "publication", "study", "scholar", "peer-reviewed", "pdf"
    ]
    found_sources = [kw for kw in research_keywords if kw in final_text]
    has_research = len(found_sources) >= 3
    no_wikipedia = "wikipedia" not in final_text or (final_text.count("wikipedia") == 1 and "not wikipedia" in final_text)
    passed = has_research and no_wikipedia
    return {
        "check": "research_sources",
        "passed": passed,
        "score": min(1.0, len(found_sources) / 3.0) if has_research else 0.0,
        "sources_found": found_sources,
        "reason": f"Found {len(found_sources)} research indicators" if passed else "Insufficient research sources",
    }


def validate_titles(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate research document titles extracted."""
    final_text = extract_final_text(result)
    title_patterns = re.findall(r"(title|paper|article):\s*([A-Z][^.!?]{30,})", final_text, re.IGNORECASE)
    quoted_titles = re.findall(r"\"([A-Z][^\"]{30,})\"", final_text)
    total_titles = len(title_patterns) + len(quoted_titles)
    passed = total_titles >= 2
    return {
        "check": "titles",
        "passed": passed,
        "score": min(1.0, total_titles / 2.0),
        "title_count": total_titles,
        "reason": f"Found {total_titles} title(s)" if passed else "Insufficient titles",
    }


def validate_authors(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate author names extracted."""
    final_text = extract_final_text(result)
    author_patterns = re.findall(r"(author|by):\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", final_text, re.IGNORECASE)
    author_names = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", final_text)
    unique_authors = len(set([a[1] if isinstance(a, tuple) else a for a in author_patterns + author_names]))
    passed = unique_authors >= 2
    return {
        "check": "authors",
        "passed": passed,
        "score": min(1.0, unique_authors / 2.0),
        "author_count": unique_authors,
        "reason": f"Found {unique_authors} author name(s)" if passed else "Insufficient authors",
    }


def validate_years(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate publication years (2020+) mentioned."""
    final_text = extract_final_text(result)
    year_pattern = re.findall(r"\b(202[0-5])\b", final_text)
    recent_years = [y for y in year_pattern if int(y) >= 2020]
    passed = len(recent_years) >= 2
    return {
        "check": "years",
        "passed": passed,
        "score": min(1.0, len(recent_years) / 2.0),
        "year_count": len(recent_years),
        "years": recent_years,
        "reason": f"Found {len(recent_years)} recent year(s)" if passed else "Insufficient recent years",
    }


def validate_contributions(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate research contributions/findings described."""
    final_text = extract_final_text(result).lower()
    contribution_keywords = ["contribution", "finding", "result", "method", "approach", "propose", "demonstrate", "show"]
    found_keywords = [kw for kw in contribution_keywords if kw in final_text]
    sentences = re.findall(r"[A-Z][^.!?]{40,}[.!?]", extract_final_text(result))
    has_descriptions = len(sentences) >= 4
    passed = len(found_keywords) >= 2 and has_descriptions
    return {
        "check": "contributions",
        "passed": passed,
        "score": min(1.0, (len(found_keywords) / 2.0 + (1.0 if has_descriptions else 0.0)) / 2.0),
        "keyword_count": len(found_keywords),
        "sentence_count": len(sentences),
        "reason": f"Found {len(found_keywords)} contribution keywords, {len(sentences)} sentences" if passed else "Insufficient contributions",
    }


def validate_comparison(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate comparison summary provided."""
    final_text = extract_final_text(result).lower()
    comparison_keywords = ["compare", "comparison", "different", "approach", "contrast", "similar", "differ", "versus", "vs"]
    has_comparison = any(kw in final_text for kw in comparison_keywords)
    sentences = re.findall(r"[A-Z][^.!?]{50,}[.!?]", extract_final_text(result))
    has_summary = len(sentences) >= 2
    passed = has_comparison and has_summary
    return {
        "check": "comparison",
        "passed": passed,
        "score": 0.5 if has_comparison else 0.0 + (0.5 if has_summary else 0.0),
        "reason": "Comparison summary found" if passed else "Comparison summary missing",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for research document quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)
    
    prompt = f"""Validate this research document analysis task:

Task: {task}

Agent Output:
{final_text[:5000]}

Observability:
- Visit actions executed: {visit_count}

Check:
1. Did agent find actual research documents (not Wikipedia or general news)?
2. Were at least 2 research sources visited?
3. Were titles, authors, and years extracted?
4. Were research contributions/findings described?
5. Was a comparison summary provided?
6. Are the sources academic/research-oriented?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "research_quality": string,
  "academic_sources": boolean,
  "comparison_quality": string
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
        validate_research_visits,
        validate_research_sources,
        validate_titles,
        validate_authors,
        validate_years,
        validate_contributions,
        validate_comparison,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
