"""
Test 036: Adversarial Compare-and-Contrast
Difficulty: 8/10 (Very Hard)
Category: Adversarial Multi-Branch Reasoning

The agent must research two obscure, superficially similar but
fundamentally different topics (Lamarckian inheritance vs. epigenetics,
AND phlogiston theory vs. oxidation theory), build a compare-and-contrast
table for each pair, then write an analytical essay on how discredited
scientific ideas can contain seeds of later validated theories.
This demands 4+ independent research branches that must be cross-compared
in pairs and then unified.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "036",
        "test_name": "Adversarial Compare-and-Contrast",
        "difficulty_level": "8/10",
        "category": "Adversarial Multi-Branch Reasoning",
    }


def get_task_statement() -> str:
    return (
        "Research and compare two pairs of historically related scientific ideas:\n\n"
        "PAIR 1 — Lamarckism vs. Epigenetics:\n"
        "Visit https://en.wikipedia.org/wiki/Lamarckism and "
        "https://en.wikipedia.org/wiki/Epigenetics\n"
        "Extract: (a) Lamarck's core claim about inheritance of acquired characteristics, "
        "(b) one modern epigenetic mechanism (e.g., DNA methylation, histone modification), "
        "(c) one specific example where epigenetics superficially resembles Lamarckian inheritance.\n\n"
        "PAIR 2 — Phlogiston theory vs. Oxidation:\n"
        "Visit https://en.wikipedia.org/wiki/Phlogiston_theory and "
        "https://en.wikipedia.org/wiki/Combustion\n"
        "Extract: (a) who proposed phlogiston theory and its core claim, "
        "(b) who disproved it and how, "
        "(c) one observation that phlogiston theory correctly predicted despite being wrong.\n\n"
        "Produce:\n"
        "1. A compare-and-contrast table for each pair (similarities vs. differences)\n"
        "2. A 200+ word analytical essay arguing how discredited theories can contain "
        "\"seeds of truth\" that later validated theories build upon, referencing specific "
        "facts from BOTH pairs.\n"
        "Cite all four Wikipedia URLs."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Lamarck's core claim",
        "Modern epigenetic mechanism",
        "Epigenetics-Lamarckism resemblance example",
        "Phlogiston proposer and core claim",
        "Who disproved phlogiston and how",
        "Correct phlogiston prediction",
        "Compare-contrast table for each pair",
        "200+ word analytical essay",
        "4 Wikipedia URL citations",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 visit actions",
        "Both pairs covered with facts",
        "Tables or structured comparisons present",
        "Essay references both pairs",
        "All 4 Wikipedia URLs cited",
    ]


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 3
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 4.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=4)",
    }


def validate_pair1_lamarck(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    has_lamarck = bool(re.search(r"(lamarck|acquired characteristics|inheritance.*acquired|use and disuse)", text))
    has_epigenetic_mechanism = bool(re.search(
        r"(methylation|histone|chromatin|acetylation|non.?coding rna|mirna|epigenetic.*mechanism)", text
    ))
    has_resemblance = bool(re.search(
        r"(resemble|similar|echo|parallel|vindicate|lamarck.*epigenetic|epigenetic.*lamarck|transgenerational)", text
    ))
    score = sum([0.35 if has_lamarck else 0.0, 0.35 if has_epigenetic_mechanism else 0.0, 0.3 if has_resemblance else 0.0])
    return {
        "check": "pair1_lamarck_epigenetics",
        "passed": has_lamarck and has_epigenetic_mechanism,
        "score": score,
        "reason": f"lamarck={has_lamarck}, mechanism={has_epigenetic_mechanism}, resemblance={has_resemblance}",
    }


def validate_pair2_phlogiston(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    has_proposer = bool(re.search(r"(stahl|becher|georg|johann)", text))
    has_disprover = bool(re.search(r"(lavoisier|oxygen|combustion.*oxygen|disprove)", text))
    has_correct_prediction = bool(re.search(
        r"(correct|predict|explain|account|phlogiston.*mass|combusti|calx|metal.*calx|reduction)", text
    ))
    score = sum([0.35 if has_proposer else 0.0, 0.35 if has_disprover else 0.0, 0.3 if has_correct_prediction else 0.0])
    return {
        "check": "pair2_phlogiston_oxidation",
        "passed": has_proposer and has_disprover,
        "score": score,
        "reason": f"proposer={has_proposer}, disprover={has_disprover}, prediction={has_correct_prediction}",
    }


def validate_tables(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    lower = text.lower()
    has_table_markers = bool(re.search(r"(\|.*\||\bsimilarit|\bdifference|\bcompare|\bcontrast)", lower))
    has_pair1_table = ("lamarck" in lower and "epigenetic" in lower and has_table_markers)
    has_pair2_table = ("phlogiston" in lower and ("oxidation" in lower or "combustion" in lower or "lavoisier" in lower) and has_table_markers)
    pipe_rows = len(re.findall(r"^\s*\|.+\|", text, re.MULTILINE))
    has_markdown_table = pipe_rows >= 3
    score = (0.3 if has_pair1_table else 0.0) + (0.3 if has_pair2_table else 0.0) + (0.4 if has_markdown_table else 0.1 if has_table_markers else 0.0)
    return {
        "check": "comparison_tables",
        "passed": has_pair1_table and has_pair2_table,
        "score": score,
        "has_markdown_table": has_markdown_table,
        "pipe_rows": pipe_rows,
        "reason": f"pair1_table={has_pair1_table}, pair2_table={has_pair2_table}, md_rows={pipe_rows}",
    }


def validate_essay(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    lower = text.lower()
    has_seeds = bool(re.search(r"(seed|kernel|germ|precursor|foreshadow|anticipat|hint|prescient|ahead.*time)", lower))
    has_both_pairs = (
        ("lamarck" in lower or "epigenetic" in lower)
        and ("phlogiston" in lower or "lavoisier" in lower)
    )
    paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 40]
    essay_text = max(paragraphs, key=len, default="") if paragraphs else ""
    word_count = len(essay_text.split())
    length_ok = word_count >= 160
    score = (
        (0.3 if has_seeds else 0.0)
        + (0.3 if has_both_pairs else 0.0)
        + (0.4 if length_ok else min(0.4, word_count / 200.0 * 0.4))
    )
    return {
        "check": "analytical_essay",
        "passed": has_seeds and has_both_pairs and length_ok,
        "score": score,
        "word_count": word_count,
        "reason": f"seeds_theme={has_seeds}, both_pairs={has_both_pairs}, words={word_count}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    urls = {
        "lamarckism": "wikipedia.org/wiki/lamarckism" in text,
        "epigenetics": "wikipedia.org/wiki/epigenetics" in text,
        "phlogiston": "wikipedia.org/wiki/phlogiston" in text,
        "combustion": "wikipedia.org/wiki/combustion" in text,
    }
    cited = sum(urls.values())
    return {
        "check": "citation_urls",
        "passed": cited >= 3,
        "score": cited / 4.0,
        "details": urls,
        "reason": f"{cited}/4 URLs cited",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this adversarial compare-and-contrast task:

Task: {task}

Agent Output:
{final_text[:7000]}

Check:
1. Pair 1: Lamarck's core claim? Epigenetic mechanism? Resemblance example?
2. Pair 2: Phlogiston proposer? Disprover (Lavoisier)? Correct prediction?
3. Compare-contrast tables for both pairs?
4. 200+ word analytical essay referencing both pairs?
5. "Seeds of truth" theme present?
6. All 4 Wikipedia URLs cited?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "pair1_quality": string,
  "pair2_quality": string,
  "essay_quality": string,
  "table_quality": string
}}"""
    try:
        messages = [
            {"role": "system", "content": "You are a test validator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        payload = connector_llm.build_payload(
            messages=messages, json_mode=True, model_name=model_name, temperature=0.1,
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
        return {"check": "llm_validation", "passed": False, "score": 0.0, "error": str(exc)}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_pair1_lamarck, validate_pair2_phlogiston, validate_tables, validate_essay, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
