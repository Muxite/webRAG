"""
Test 034: Laundry-List Multi-Topic Extraction
Difficulty: 6/10 (Hard)
Category: Parallel Fact Gathering

Gives the agent a flat list of 6 unrelated niche topics, each requiring one
specific fact extracted from a real web page.  A strong graph agent will
fan out into 6 parallel branches and merge.  Weaker planners will
serialize, time-out, or miss items.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


TOPICS = [
    {
        "label": "Axolotl neoteny",
        "url": "https://en.wikipedia.org/wiki/Axolotl",
        "question": "What is the scientific name of the axolotl?",
        "answer_pattern": r"ambystoma\s+mexicanum",
        "keywords": ["ambystoma", "mexicanum"],
    },
    {
        "label": "Voynich manuscript",
        "url": "https://en.wikipedia.org/wiki/Voynich_manuscript",
        "question": "In which library is the Voynich manuscript currently held?",
        "answer_pattern": r"beinecke|yale",
        "keywords": ["beinecke", "yale"],
    },
    {
        "label": "Pando clonal colony",
        "url": "https://en.wikipedia.org/wiki/Pando_(tree)",
        "question": "In which US state is the Pando clonal colony located?",
        "answer_pattern": r"utah",
        "keywords": ["utah"],
    },
    {
        "label": "Antikythera mechanism",
        "url": "https://en.wikipedia.org/wiki/Antikythera_mechanism",
        "question": "Approximately what century BCE was the Antikythera mechanism built?",
        "answer_pattern": r"(2nd|1st|second|first)\s*(century|c\.?\s*bce?)",
        "keywords": ["century", "bce"],
    },
    {
        "label": "Mantis shrimp vision",
        "url": "https://en.wikipedia.org/wiki/Mantis_shrimp",
        "question": "How many types of photoreceptor cells do mantis shrimp have (approximate)?",
        "answer_pattern": r"(12|16|twelve|sixteen)",
        "keywords": ["photoreceptor", "color"],
    },
    {
        "label": "Svalbard Global Seed Vault",
        "url": "https://en.wikipedia.org/wiki/Svalbard_Global_Seed_Vault",
        "question": "In which country is the Svalbard Global Seed Vault located?",
        "answer_pattern": r"norway|norwegian",
        "keywords": ["norway", "svalbard"],
    },
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "034",
        "test_name": "Laundry-List Multi-Topic Extraction",
        "difficulty_level": "6/10",
        "category": "Parallel Fact Gathering",
    }


def get_task_statement() -> str:
    lines = [
        "Visit each of the following Wikipedia pages and answer the question for each topic. "
        "Return all six answers with the citation URL for each.\n"
    ]
    for i, t in enumerate(TOPICS, 1):
        lines.append(f"{i}. {t['label']}: {t['url']} — {t['question']}")
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    return [f"Answer for {t['label']}" for t in TOPICS] + ["Citation URL per answer"]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 visit actions executed",
        "At least 5 of 6 answers correct",
        "Citation URLs present for each topic",
    ]


def validate_visit_count(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 4
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 6.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=4)",
    }


def validate_answers(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    correct = 0
    details = {}
    for t in TOPICS:
        found = bool(re.search(t["answer_pattern"], text, re.IGNORECASE))
        if not found:
            found = all(kw in text for kw in t["keywords"])
        details[t["label"]] = found
        if found:
            correct += 1
    passed = correct >= 5
    return {
        "check": "answer_correctness",
        "passed": passed,
        "score": correct / len(TOPICS),
        "correct": correct,
        "total": len(TOPICS),
        "details": details,
        "reason": f"{correct}/{len(TOPICS)} answers correct",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    cited = 0
    for t in TOPICS:
        url_fragment = t["url"].replace("https://en.wikipedia.org/wiki/", "").lower()
        if url_fragment in text or t["url"].lower() in text:
            cited += 1
    passed = cited >= 5
    return {
        "check": "citation_coverage",
        "passed": passed,
        "score": cited / len(TOPICS),
        "cited": cited,
        "total": len(TOPICS),
        "reason": f"{cited}/{len(TOPICS)} topics cited",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this laundry-list fact extraction task:

Task: {task}

Agent Output:
{final_text[:6000]}

Expected answers:
1. Axolotl scientific name: Ambystoma mexicanum
2. Voynich manuscript location: Beinecke Rare Book Library at Yale
3. Pando colony state: Utah
4. Antikythera mechanism century: ~2nd century BCE
5. Mantis shrimp photoreceptors: 12-16 types
6. Seed Vault country: Norway

Check:
1. How many of the 6 answers are correct?
2. Are citation URLs provided for each?
3. Is the information sourced from actual page visits (not hallucinated)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "correct_count": int,
  "cited_count": int
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
    return [validate_visit_count, validate_answers, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
