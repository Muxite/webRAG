"""
Test 035: Cross-Domain Niche Synthesis
Difficulty: 7/10 (Hard)
Category: Diverge-Converge Synthesis

The agent must research three niche topics from completely different
domains (mycology, maritime law, origami mathematics), extract specific
facts from each, and then write an original synthesis essay that
identifies a unifying theme across all three.  This forces a 3-way
branch-and-merge pattern.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "035",
        "test_name": "Cross-Domain Niche Synthesis",
        "difficulty_level": "7/10",
        "category": "Diverge-Converge Synthesis",
    }


def get_task_statement() -> str:
    return (
        "Research three unrelated niche topics, extract the requested facts from "
        "each Wikipedia page, and then write a synthesis.\n\n"
        "1. Cordyceps fungi: visit https://en.wikipedia.org/wiki/Cordyceps — "
        "find (a) the common name for the behaviour Cordyceps induces in ants, "
        "(b) one commercial or medicinal use of Cordyceps.\n\n"
        "2. Admiralty law: visit https://en.wikipedia.org/wiki/Admiralty_law — "
        "find (a) the historical origin or earliest legal code associated with admiralty law, "
        "(b) one modern international maritime convention or treaty.\n\n"
        "3. Mathematics of paper folding: visit https://en.wikipedia.org/wiki/Mathematics_of_paper_folding — "
        "find (a) the name of one theorem or axiom related to origami mathematics, "
        "(b) one practical engineering application of origami folding.\n\n"
        "Finally, write a 150+ word essay identifying a UNIFYING THEME that connects "
        "all three topics (e.g., hidden complexity, emergent structure, nature inspiring engineering). "
        "The essay must reference specific facts from each topic. "
        "Cite the Wikipedia URL for every fact."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Cordyceps: ant behaviour name + commercial/medicinal use",
        "Admiralty law: historical origin + modern convention",
        "Paper folding math: theorem/axiom name + engineering application",
        "Synthesis essay (150+ words) with unifying theme",
        "Wikipedia URL citations",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 visit actions",
        "Facts from all 3 topics present",
        "Synthesis essay with cross-references",
        "All 3 Wikipedia URLs cited",
    ]


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 3
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 3.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s)",
    }


def validate_cordyceps(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    has_zombie = bool(re.search(r"(zombie|mind.?control|parasit|manipulat|puppet)", text))
    has_use = bool(re.search(r"(traditional.*medicine|supplement|sinensis|medicin|caterpillar fungus|energy|athletic)", text))
    score = (0.5 if has_zombie else 0.0) + (0.5 if has_use else 0.0)
    return {
        "check": "cordyceps_facts",
        "passed": has_zombie and has_use,
        "score": score,
        "reason": f"zombie_behaviour={has_zombie}, commercial_use={has_use}",
    }


def validate_admiralty(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    has_origin = bool(re.search(
        r"(rhod(ian|es)|oleron|hanseatic|lex mercatoria|ancient|roman|byzantine|medieval)", text
    ))
    has_convention = bool(re.search(
        r"(unclos|solas|marpol|salvage convention|hague|hamburg rules|rotterdam rules|york.antwerp|convention)", text
    ))
    score = (0.5 if has_origin else 0.0) + (0.5 if has_convention else 0.0)
    return {
        "check": "admiralty_facts",
        "passed": has_origin and has_convention,
        "score": score,
        "reason": f"historical_origin={has_origin}, convention={has_convention}",
    }


def validate_origami_math(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    has_theorem = bool(re.search(
        r"(huzita|kawasaki|haga|flat.?fold|two.?color|maekawa|fold.*axiom|justin|beloch)", text
    ))
    has_engineering = bool(re.search(
        r"(solar.*panel|airbag|stent|deployable|satellite|miura.?fold|telescope|origami.*engineer|space|fold.*structure)", text
    ))
    score = (0.5 if has_theorem else 0.0) + (0.5 if has_engineering else 0.0)
    return {
        "check": "origami_math_facts",
        "passed": has_theorem and has_engineering,
        "score": score,
        "reason": f"theorem={has_theorem}, engineering={has_engineering}",
    }


def validate_synthesis_essay(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    lower = text.lower()
    has_cordyceps = "cordyceps" in lower
    has_admiralty = "admiralty" in lower or "maritime law" in lower
    has_origami = "origami" in lower or "paper folding" in lower or "paper-folding" in lower
    has_theme = bool(re.search(
        r"(unif|common thread|connect|theme|parallel|across|bridge|shared|all three)", lower
    ))
    longest_paragraph = max((p for p in text.split("\n\n")), key=len, default="")
    word_count = len(longest_paragraph.split())
    length_ok = word_count >= 120
    topics_in_essay = sum([has_cordyceps, has_admiralty, has_origami])
    score = (
        (topics_in_essay / 3.0) * 0.4
        + (0.3 if has_theme else 0.0)
        + (0.3 if length_ok else min(0.3, word_count / 150.0 * 0.3))
    )
    return {
        "check": "synthesis_essay",
        "passed": topics_in_essay == 3 and has_theme and length_ok,
        "score": score,
        "topics_referenced": topics_in_essay,
        "has_theme": has_theme,
        "longest_paragraph_words": word_count,
        "reason": f"topics={topics_in_essay}/3, theme={has_theme}, words={word_count}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    urls = {
        "cordyceps": "wikipedia.org/wiki/cordyceps" in text,
        "admiralty": "wikipedia.org/wiki/admiralty" in text,
        "paper_folding": "wikipedia.org/wiki/mathematics_of_paper_folding" in text,
    }
    cited = sum(urls.values())
    return {
        "check": "citation_urls",
        "passed": cited == 3,
        "score": cited / 3.0,
        "details": urls,
        "reason": f"{cited}/3 URLs cited",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this cross-domain niche synthesis task:

Task: {task}

Agent Output:
{final_text[:6000]}

Check:
1. Cordyceps: zombie-ant behaviour named? Commercial/medicinal use?
2. Admiralty law: historical origin? Modern convention?
3. Origami math: theorem/axiom named? Engineering application?
4. Synthesis essay: 150+ words? References all 3 topics? Clear unifying theme?
5. All 3 Wikipedia URLs cited?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "facts_per_topic": {{"cordyceps": int, "admiralty": int, "origami": int}},
  "synthesis_quality": string
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
    return [validate_visits, validate_cordyceps, validate_admiralty, validate_origami_math, validate_synthesis_essay, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
