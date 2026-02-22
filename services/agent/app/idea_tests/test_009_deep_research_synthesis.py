"""
Test 009: Deep Research Synthesis
Difficulty: 9/10 (Extremely Hard)
Category: Deep Research & Synthesis
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "009",
        "test_name": "Deep Research Synthesis",
        "difficulty_level": "9/10",
        "category": "Deep Research & Synthesis",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Research the controversy surrounding the 'Piltdown Man' hoax. Find: the original discovery claims, "
        "the scientific methods used to expose it as a fraud, the timeline of the hoax and its exposure, "
        "the identity of the perpetrator(s) and their motivations, modern scientific techniques that would "
        "have detected it immediately, and the impact on paleoanthropology. Synthesize this into a coherent "
        "narrative explaining how scientific self-correction works. Cite primary sources, scientific papers, "
        "and historical documents."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Original discovery claims",
        "Methods used to expose fraud",
        "Timeline of hoax and exposure",
        "Perpetrator identity and motivations",
        "Modern detection techniques",
        "Impact on paleoanthropology",
        "Synthesis on scientific self-correction",
        "Primary sources and scientific papers cited",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "All six research areas covered",
        "Synthesis connecting to scientific self-correction",
        "Primary sources and scientific papers cited",
        "At least 8 authoritative URLs",
        "Historical accuracy",
    ]


def validate_research_areas(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all research areas covered."""
    final_text = extract_final_text(result).lower()
    areas = {
        "discovery": bool(re.search(r"\b(discover|found|claim|announce|reveal)\b", final_text)),
        "exposure": bool(re.search(r"\b(expose|fraud|hoax|fake|reveal|uncover|detect)\b", final_text)),
        "timeline": bool(re.search(r"\b(timeline|year|date|period|when|1912|1953)\b", final_text)),
        "perpetrator": bool(re.search(r"\b(perpetrator|who|person|responsible|suspect|dawson)\b", final_text)),
        "modern": bool(re.search(r"\b(modern|technique|method|today|current|would have|detect)\b", final_text)),
        "impact": bool(re.search(r"\b(impact|effect|influence|paleoanthropology|field|science)\b", final_text)),
    }
    area_count = sum(areas.values())
    return {
        "check": "research_areas",
        "passed": area_count >= 5,
        "score": area_count / 6.0,
        "areas_covered": area_count,
        "details": areas,
        "reason": f"Covered {area_count}/6 research areas",
    }


def validate_synthesis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate synthesis present."""
    final_text = extract_final_text(result).lower()
    has_synthesis = bool(re.search(r"\b(synthesis|synthesize|connect|relate|explain|narrative|conclusion)\b", final_text))
    has_self_correction = bool(re.search(r"\b(self.?correct|scientific method|peer review|verification|replication)\b", final_text))
    return {
        "check": "synthesis",
        "passed": has_synthesis and has_self_correction,
        "score": 0.5 if has_synthesis else 0.0 + (0.5 if has_self_correction else 0.0),
        "reason": f"Synthesis: present={has_synthesis}, self-correction={has_self_correction}",
    }


def validate_sources(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate authoritative sources."""
    final_text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", final_text)
    authoritative = [u for u in urls if any(x in u.lower() for x in [".edu", ".gov", ".org", "museum", "archive", "scientific", "journal", "nature", "science"])]
    passed = len(urls) >= 8 and len(authoritative) >= 5
    return {
        "check": "authoritative_sources",
        "passed": passed,
        "score": min(1.0, (len(urls) / 8.0) * 0.5 + (len(authoritative) / 5.0) * 0.5),
        "total_urls": len(urls),
        "authoritative_urls": len(authoritative),
        "reason": f"Found {len(urls)} URLs ({len(authoritative)} authoritative)",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for comprehensive coverage and synthesis."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this deep research synthesis task:

Task: {task}

Agent Output:
{final_text[:5000]}

Check:
1. Are all 6 research areas covered?
2. Is the synthesis on scientific self-correction present and coherent?
3. Are primary sources and scientific papers cited?
4. Is historical accuracy maintained?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "all_areas_covered": boolean,
  "synthesis_quality": string,
  "source_quality": string,
  "historical_accuracy": boolean
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
    return [validate_research_areas, validate_synthesis, validate_sources]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
