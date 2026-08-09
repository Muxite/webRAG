"""
Test 037: Five-Topic Convergence Challenge
Difficulty: 9/10 (Extremely Hard)
Category: Massive Branching Convergence

The hardest branching test: the agent must research five completely
disparate niche topics, each from a different Wikipedia page, extract
specific facts, then produce a unified "convergence report" that
identifies at least three cross-cutting themes spanning all five topics.
A naive sequential agent will time-out or miss topics; a good planner
will fan out into 5 branches, gather facts, and perform a multi-way merge.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


TOPICS = [
    {
        "label": "Kowloon Walled City",
        "url": "https://en.wikipedia.org/wiki/Kowloon_Walled_City",
        "fact_q": "peak population density or approximate resident count",
        "keywords": ["kowloon", "walled city"],
        "fact_patterns": [r"(33,000|50,000|densit|resident|population)"],
    },
    {
        "label": "Mycelium networks",
        "url": "https://en.wikipedia.org/wiki/Mycelium",
        "fact_q": "the term for fungal communication networks connecting trees",
        "keywords": ["mycelium", "wood wide web"],
        "fact_patterns": [r"(wood wide web|mycorrhiz|network|connect.*tree)"],
    },
    {
        "label": "Fermi Paradox",
        "url": "https://en.wikipedia.org/wiki/Fermi_paradox",
        "fact_q": "the approximate number of stars in the Milky Way that makes the paradox compelling",
        "keywords": ["fermi", "paradox"],
        "fact_patterns": [r"(100 billion|200 billion|billion.*star|star.*billion|10\^11)"],
    },
    {
        "label": "Dead Internet theory",
        "url": "https://en.wikipedia.org/wiki/Dead_Internet_theory",
        "fact_q": "the core claim of the Dead Internet theory",
        "keywords": ["dead internet", "bot"],
        "fact_patterns": [r"(bot|artificial|generated|manipulat|inauthentic|automated)"],
    },
    {
        "label": "Kintsugi",
        "url": "https://en.wikipedia.org/wiki/Kintsugi",
        "fact_q": "the material used and the philosophy behind kintsugi repair",
        "keywords": ["kintsugi", "gold", "lacquer"],
        "fact_patterns": [r"(gold|lacquer|wabi.?sabi|imperfect|repair.*beauty|beauty.*break)"],
    },
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "037",
        "test_name": "Five-Topic Convergence Challenge",
        "difficulty_level": "9/10",
        "category": "Massive Branching Convergence",
    }


def get_task_statement() -> str:
    lines = [
        "Research each of the following five topics by visiting their Wikipedia pages. "
        "For each, extract the specific fact requested.\n"
    ]
    for i, t in enumerate(TOPICS, 1):
        lines.append(f"{i}. {t['label']}: {t['url']} — Extract: {t['fact_q']}")
    lines.append(
        "\nAfter gathering all facts, write a CONVERGENCE REPORT (250+ words) that:\n"
        "- Identifies at least THREE cross-cutting themes that span all five topics "
        "(e.g., hidden networks, emergent complexity, human perception vs. reality, decay and renewal)\n"
        "- For each theme, references specific facts from at least two of the five topics\n"
        "- Concludes with an original insight that connects all five topics\n\n"
        "Cite every Wikipedia URL used."
    )
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    return [f"Fact from {t['label']}" for t in TOPICS] + [
        "Convergence report (250+ words)",
        "At least 3 cross-cutting themes",
        "Wikipedia URL citations for all 5 topics",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 visit actions",
        "At least 4 of 5 facts extracted",
        "Convergence report with 3+ themes",
        "Report references facts from multiple topics per theme",
        "All 5 Wikipedia URLs cited",
    ]


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 4
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 5.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=5)",
    }


def validate_facts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    found = 0
    details = {}
    for t in TOPICS:
        matched = any(bool(re.search(p, text, re.IGNORECASE)) for p in t["fact_patterns"])
        if not matched:
            matched = all(kw in text for kw in t["keywords"])
        details[t["label"]] = matched
        if matched:
            found += 1
    passed = found >= 4
    return {
        "check": "fact_extraction",
        "passed": passed,
        "score": found / len(TOPICS),
        "found": found,
        "total": len(TOPICS),
        "details": details,
        "reason": f"{found}/{len(TOPICS)} facts extracted",
    }


def validate_convergence_report(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    lower = text.lower()

    theme_indicators = [
        r"(hidden.*network|connect|unseen.*structure)",
        r"(emergent|complex|self.?organiz|bottom.?up)",
        r"(perception|reality|illusion|appear|myth)",
        r"(decay|renewal|repair|transform|resilience|impermanence)",
        r"(density|overcrowd|resource|scarcity|abundance)",
        r"(communicat|signal|information|flow)",
        r"(beauty|aesthetic|philosophy|meaning)",
    ]
    themes_found = sum(1 for pat in theme_indicators if re.search(pat, lower))
    themes_ok = themes_found >= 3

    topic_mentions = sum(1 for t in TOPICS if any(kw in lower for kw in t["keywords"]))
    topics_ok = topic_mentions >= 4

    paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 30]
    report_text = " ".join(paragraphs[-3:]) if paragraphs else ""
    word_count = len(report_text.split())
    length_ok = word_count >= 200

    score = (
        (0.3 if themes_ok else themes_found / 3.0 * 0.3)
        + (0.3 if topics_ok else topic_mentions / 5.0 * 0.3)
        + (0.4 if length_ok else min(0.4, word_count / 250.0 * 0.4))
    )
    return {
        "check": "convergence_report",
        "passed": themes_ok and topics_ok and length_ok,
        "score": score,
        "themes_found": themes_found,
        "topics_mentioned": topic_mentions,
        "word_count": word_count,
        "reason": f"themes={themes_found}, topics={topic_mentions}/5, words={word_count}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    cited = 0
    details = {}
    for t in TOPICS:
        url_fragment = t["url"].replace("https://en.wikipedia.org/wiki/", "").lower()
        present = url_fragment in text or t["url"].lower() in text
        details[t["label"]] = present
        if present:
            cited += 1
    return {
        "check": "citation_urls",
        "passed": cited >= 4,
        "score": cited / len(TOPICS),
        "cited": cited,
        "total": len(TOPICS),
        "details": details,
        "reason": f"{cited}/{len(TOPICS)} URLs cited",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this five-topic convergence challenge:

Task: {task}

Agent Output:
{final_text[:8000]}

Check:
1. Are facts extracted from all 5 topics?
2. Is there a convergence report of 250+ words?
3. Does the report identify at least 3 cross-cutting themes?
4. Does each theme reference facts from at least 2 topics?
5. Is there an original concluding insight?
6. Are all 5 Wikipedia URLs cited?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "facts_found": int,
  "themes_identified": int,
  "report_word_count": int,
  "cross_references_quality": string
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
    return [validate_visits, validate_facts, validate_convergence_report, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
