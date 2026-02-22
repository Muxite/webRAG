"""
Test 011: Advanced Reasoning Challenge
Difficulty: 9/10 (Extremely Hard)
Category: Advanced Reasoning
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "011",
        "test_name": "Advanced Reasoning Challenge",
        "difficulty_level": "9/10",
        "category": "Advanced Reasoning",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Find three different scientific studies or research papers that reached contradictory conclusions "
        "about the same topic (e.g., coffee and health, screen time effects, exercise frequency). For each study, "
        "identify: the research question, methodology, sample size, key findings, and potential limitations. "
        "Then analyze why the studies reached different conclusions, evaluate which methodology is most robust, "
        "and synthesize what the current scientific consensus is (if any). Cite all papers with URLs or DOIs."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Three contradictory studies on same topic",
        "Research question for each",
        "Methodology for each",
        "Sample size for each",
        "Key findings for each",
        "Limitations for each",
        "Analysis of why conclusions differ",
        "Evaluation of most robust methodology",
        "Synthesis of current consensus",
        "Citations (URLs or DOIs) for all papers",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "Three studies identified with contradictions",
        "All required details for each study",
        "Analysis of methodological differences",
        "Robust methodology evaluation",
        "Consensus synthesis",
        "At least 3 paper citations (DOIs or URLs)",
    ]


def validate_studies(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate three studies present."""
    final_text = extract_final_text(result).lower()
    study_indicators = ["study", "research", "paper", "publication", "journal", "findings", "conclusion"]
    study_count = sum(1 for indicator in study_indicators if indicator in final_text)
    has_contradiction = bool(re.search(r"\b(contradict|differ|opposite|conflict|disagree|contrary)\b", final_text))
    passed = study_count >= 6 and has_contradiction
    return {
        "check": "studies_present",
        "passed": passed,
        "score": min(1.0, (study_count / 6.0) * 0.5 + (0.5 if has_contradiction else 0.0)),
        "study_indicators": study_count,
        "has_contradiction": has_contradiction,
        "reason": f"Found {study_count} study indicators, contradiction={has_contradiction}",
    }


def validate_methodology(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate methodology details."""
    final_text = extract_final_text(result).lower()
    has_methodology = bool(re.search(r"\b(methodolog|method|approach|design|procedure|protocol)\b", final_text))
    has_sample = bool(re.search(r"\b(sample|participant|subject|n\s*=|size|number of)\b", final_text))
    has_limitations = bool(re.search(r"\b(limitation|weakness|bias|confound|flaw|issue)\b", final_text))
    checks = sum([has_methodology, has_sample, has_limitations])
    return {
        "check": "methodology_details",
        "passed": checks >= 2,
        "score": checks / 3.0,
        "has_methodology": has_methodology,
        "has_sample": has_sample,
        "has_limitations": has_limitations,
        "reason": f"Methodology: method={has_methodology}, sample={has_sample}, limitations={has_limitations}",
    }


def validate_analysis(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate analysis quality."""
    final_text = extract_final_text(result).lower()
    has_why = bool(re.search(r"\b(why|because|reason|due to|explain|cause)\b", final_text))
    has_evaluation = bool(re.search(r"\b(evaluat|assess|compare|robust|strong|weak|better|best)\b", final_text))
    has_consensus = bool(re.search(r"\b(consensus|agreement|current|scientific|general|majority)\b", final_text))
    checks = sum([has_why, has_evaluation, has_consensus])
    return {
        "check": "analysis_quality",
        "passed": checks >= 2,
        "score": checks / 3.0,
        "has_why": has_why,
        "has_evaluation": has_evaluation,
        "has_consensus": has_consensus,
        "reason": f"Analysis: why={has_why}, evaluation={has_evaluation}, consensus={has_consensus}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    """LLM validation for comprehensive study analysis."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    
    prompt = f"""Validate this advanced reasoning task:

Task: {task}

Agent Output:
{final_text[:5000]}

Check:
1. Are three contradictory studies on the same topic identified?
2. Are all required details present for each study (question, methodology, sample, findings, limitations)?
3. Is there analysis of why conclusions differ?
4. Is the most robust methodology evaluated?
5. Is current scientific consensus synthesized?
6. Are papers properly cited (DOIs or URLs)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "three_studies": boolean,
  "all_details_present": boolean,
  "analysis_quality": string,
  "citations_present": boolean
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
    return [validate_studies, validate_methodology, validate_analysis]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
