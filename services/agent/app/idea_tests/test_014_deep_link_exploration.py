"""
Test 014: Deep Link Exploration
Difficulty: 5/10 (Moderate-Hard)
Category: Deep Exploration
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "014",
        "test_name": "Deep Link Exploration",
        "difficulty_level": "5/10",
        "category": "Deep Exploration",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Start at a news article about climate change (find one via search). From that article, "
        "follow links to find: (1) a scientific research paper or study, (2) a government policy document, "
        "and (3) an organization's position statement. For each, provide the URL, a summary of the content, "
        "and explain how it relates to the original article. Document your exploration path."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Original news article about climate change",
        "Scientific research paper/study",
        "Government policy document",
        "Organization position statement",
        "URL and summary for each",
        "Relationship to original article explained",
        "Exploration path documented",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All 4 document types found",
        "Valid URLs for each",
        "Summaries provided",
        "Relationships explained",
        "At least 4 visit actions",
    ]


def validate_document_types(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all document types found."""
    final_text = extract_final_text(result).lower()
    has_news = bool(re.search(r"\b(news|article|report|journalism)\b", final_text))
    has_research = bool(re.search(r"\b(research|study|paper|scientific|journal|publication)\b", final_text))
    has_government = bool(re.search(r"\b(government|policy|official|federal|state|department)\b", final_text))
    has_organization = bool(re.search(r"\b(organization|position|statement|ngo|institute|foundation)\b", final_text))
    doc_count = sum([has_news, has_research, has_government, has_organization])
    return {
        "check": "document_types",
        "passed": doc_count >= 3,
        "score": doc_count / 4.0,
        "news": has_news,
        "research": has_research,
        "government": has_government,
        "organization": has_organization,
        "reason": f"Found {doc_count}/4 document types",
    }


def validate_urls(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate URLs present."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    passed = len(urls) >= 4
    return {
        "check": "url_count",
        "passed": passed,
        "score": min(1.0, len(urls) / 4.0),
        "url_count": len(urls),
        "reason": f"Found {len(urls)} URLs",
    }


def validate_summaries(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate summaries present."""
    final_text = extract_final_text(result)
    sentences = re.findall(r"[A-Z][^.!?]{30,}[.!?]", final_text)
    passed = len(sentences) >= 4
    return {
        "check": "summaries",
        "passed": passed,
        "score": min(1.0, len(sentences) / 6.0),
        "sentence_count": len(sentences),
        "reason": f"Found {len(sentences)} summary sentences",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for exploration quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this deep link exploration task:

Task: {task}

Agent Output:
{final_text[:4000]}

Check:
1. Are all 4 document types present (news, research, government, organization)?
2. Are summaries provided for each?
3. Are relationships to original article explained?
4. Is the exploration path documented?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "all_types_present": boolean,
  "exploration_quality": string
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
    return [validate_document_types, validate_urls, validate_summaries]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
