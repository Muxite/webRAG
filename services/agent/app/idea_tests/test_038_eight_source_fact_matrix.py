"""
Test 038: Eight-Source Fact Matrix
Difficulty: 9/10 (Extremely Hard)
Category: Massive Parallel Extraction

The agent must visit eight different Wikipedia pages across unrelated domains,
extract exactly two specific facts from each, and assemble the results into a
structured 8×2 fact matrix.  A graph agent can fan out into 8 parallel
branches; a sequential agent must serialize all visits and will likely run out
of nodes or lose context before finishing.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


SOURCES = [
    {
        "label": "Great Wall of China",
        "url": "https://en.wikipedia.org/wiki/Great_Wall_of_China",
        "facts": [
            {"q": "total length in km or miles", "patterns": [r"(21,?196|13,?170|kilometers|miles|km)"]},
            {"q": "the dynasty that built the most famous sections", "patterns": [r"(ming|qin)"]},
        ],
        "keywords": ["great wall"],
    },
    {
        "label": "CRISPR gene editing",
        "url": "https://en.wikipedia.org/wiki/CRISPR_gene_editing",
        "facts": [
            {"q": "the protein used as molecular scissors", "patterns": [r"cas9|cas12|cas13"]},
            {"q": "the year CRISPR was first used to edit mammalian cells", "patterns": [r"201[23]"]},
        ],
        "keywords": ["crispr"],
    },
    {
        "label": "Mariana Trench",
        "url": "https://en.wikipedia.org/wiki/Mariana_Trench",
        "facts": [
            {"q": "maximum depth in metres or feet", "patterns": [r"(10,?994|10,?935|36,?070|36,?201|metres|feet)"]},
            {"q": "name of the deepest point", "patterns": [r"challenger\s*deep"]},
        ],
        "keywords": ["mariana", "trench"],
    },
    {
        "label": "Voyager 1",
        "url": "https://en.wikipedia.org/wiki/Voyager_1",
        "facts": [
            {"q": "launch year", "patterns": [r"1977"]},
            {"q": "the boundary it crossed to enter interstellar space", "patterns": [r"(heliopause|interstellar)"]},
        ],
        "keywords": ["voyager"],
    },
    {
        "label": "Rosetta Stone",
        "url": "https://en.wikipedia.org/wiki/Rosetta_Stone",
        "facts": [
            {"q": "the three scripts on the stone", "patterns": [r"(hieroglyph|demotic|greek)"]},
            {"q": "the year it was discovered", "patterns": [r"1799"]},
        ],
        "keywords": ["rosetta"],
    },
    {
        "label": "Penicillin",
        "url": "https://en.wikipedia.org/wiki/Penicillin",
        "facts": [
            {"q": "who discovered penicillin", "patterns": [r"(fleming|alexander\s+fleming)"]},
            {"q": "the year it was discovered", "patterns": [r"1928"]},
        ],
        "keywords": ["penicillin"],
    },
    {
        "label": "Aurora (astronomy)",
        "url": "https://en.wikipedia.org/wiki/Aurora_(astronomy)",
        "facts": [
            {"q": "the scientific name for the northern lights", "patterns": [r"aurora\s*borealis"]},
            {"q": "what causes auroras (solar particles interacting with …)", "patterns": [r"(solar wind|magnetosphere|charged particle|magnetic field)"]},
        ],
        "keywords": ["aurora"],
    },
    {
        "label": "Enigma machine",
        "url": "https://en.wikipedia.org/wiki/Enigma_machine",
        "facts": [
            {"q": "who led the British effort to crack Enigma", "patterns": [r"(turing|alan\s+turing|bletchley)"]},
            {"q": "the war in which Enigma was used", "patterns": [r"(world war ii|world war 2|ww2|wwii|second world war)"]},
        ],
        "keywords": ["enigma"],
    },
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "038",
        "test_name": "Eight-Source Fact Matrix",
        "difficulty_level": "9/10",
        "category": "Massive Parallel Extraction",
    }


def get_task_statement() -> str:
    lines = [
        "Visit each of the following eight Wikipedia pages and extract the two "
        "specific facts listed for each topic.  Present your findings as a "
        "structured fact matrix (table) with one row per topic and columns for "
        "each fact.  Cite every Wikipedia URL.\n"
    ]
    for i, s in enumerate(SOURCES, 1):
        fact_descs = " AND ".join(f"({chr(96+j)}) {f['q']}" for j, f in enumerate(s["facts"], 1))
        lines.append(f"{i}. {s['label']}: {s['url']} — Extract: {fact_descs}")
    lines.append(
        "\nAfter gathering all facts, present a MARKDOWN TABLE with columns: "
        "Topic | Fact A | Fact B | Source URL"
    )
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    deliverables = []
    for s in SOURCES:
        for f in s["facts"]:
            deliverables.append(f"{s['label']}: {f['q']}")
    deliverables.append("Structured fact matrix table")
    deliverables.append("Citation URL per topic")
    return deliverables


def get_success_criteria() -> List[str]:
    return [
        "At least 6 visit actions executed",
        "At least 12 of 16 facts extracted correctly",
        "Structured table present in output",
        "At least 6 of 8 Wikipedia URLs cited",
    ]


# ── validators ──────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 6
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 8.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=6)",
    }


def validate_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    correct = 0
    total = 0
    details: Dict[str, List[bool]] = {}
    for s in SOURCES:
        topic_hits = []
        for f in s["facts"]:
            total += 1
            matched = any(bool(re.search(p, text, re.IGNORECASE)) for p in f["patterns"])
            topic_hits.append(matched)
            if matched:
                correct += 1
        details[s["label"]] = topic_hits
    passed = correct >= 12
    return {
        "check": "fact_extraction",
        "passed": passed,
        "score": correct / total,
        "correct": correct,
        "total": total,
        "details": {k: [str(v) for v in vs] for k, vs in details.items()},
        "reason": f"{correct}/{total} facts correct",
    }


def validate_table(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    pipe_rows = len(re.findall(r"^\s*\|.+\|", text, re.MULTILINE))
    has_table = pipe_rows >= 5
    topics_in_table = sum(
        1 for s in SOURCES
        if any(kw in text.lower() for kw in s["keywords"])
    )
    score = (0.5 if has_table else min(0.5, pipe_rows / 10.0)) + (0.5 * topics_in_table / len(SOURCES))
    return {
        "check": "structured_table",
        "passed": has_table and topics_in_table >= 6,
        "score": score,
        "pipe_rows": pipe_rows,
        "topics_in_table": topics_in_table,
        "reason": f"table_rows={pipe_rows}, topics_in_table={topics_in_table}/8",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    cited = 0
    details = {}
    for s in SOURCES:
        fragment = s["url"].replace("https://en.wikipedia.org/wiki/", "").lower()
        present = fragment in text or s["url"].lower() in text
        details[s["label"]] = present
        if present:
            cited += 1
    return {
        "check": "citation_urls",
        "passed": cited >= 6,
        "score": cited / len(SOURCES),
        "cited": cited,
        "total": len(SOURCES),
        "details": details,
        "reason": f"{cited}/{len(SOURCES)} URLs cited",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this eight-source fact matrix task:

Task: {task}

Agent Output:
{final_text[:8000]}

Check:
1. How many of the 16 individual facts (2 per topic × 8 topics) are present and correct?
2. Is there a structured markdown table?
3. Are all 8 Wikipedia URLs cited?
4. Is the information from actual page visits (not hallucinated)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "facts_correct": int,
  "facts_total": 16,
  "has_table": boolean,
  "urls_cited": int
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
    return [validate_visits, validate_facts, validate_table, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
