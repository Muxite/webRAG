"""
Test 043: Capstone Reconciliation & Synthesis
Difficulty: 10/10 (Maximum)
Category: Multi-Hop Breadth + Synthesis

A capstone combining multi-hop chains and wide breadth with a written synthesis. For
each of THREE scientific theories the agent must chain two pages (theory -> originator
-> originator's birth country), then write a 200+ word reconciliation that connects the
three origins with per-source citations.

That is 6 dependent page reads (3 theories + 3 people) — far beyond `naive_rag`'s fixed
3-visit budget — plus a synthesis graded by the fixed `gpt-5-mini` judge. Scoring is
gated/bimodal on the keystone (all three originators AND their birth countries correct);
secondary objective checks short-circuit otherwise, and the LLM judge grades synthesis
quality.

Ground truth (stable):
  General relativity                 -> Albert Einstein  -> Germany (Ulm)
  Evolution by natural selection     -> Charles Darwin   -> England / United Kingdom
  Periodic law (periodic table)      -> Dmitri Mendeleev -> Russia
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


THEORIES = [
    {"id": "T1", "theory": "Theory of general relativity",
     "url": "https://en.wikipedia.org/wiki/General_relativity",
     "originator": r"einstein", "country": r"german"},
    {"id": "T2", "theory": "Theory of evolution by natural selection",
     "url": "https://en.wikipedia.org/wiki/Natural_selection",
     "originator": r"darwin", "country": r"(england|united\s+kingdom|\bu\.?k\.?\b|british)"},
    {"id": "T3", "theory": "Periodic law (the periodic table)",
     "url": "https://en.wikipedia.org/wiki/Periodic_table",
     "originator": r"mendeleev", "country": r"russia"},
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "043",
        "test_name": "Capstone Reconciliation & Synthesis",
        "difficulty_level": "10/10",
        "category": "Multi-Hop Breadth + Synthesis",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    lines = [
        "You are writing a research brief on the origins of three foundational scientific theories. "
        "For EACH theory: visit its Wikipedia page, identify the primary ORIGINATOR, then visit that "
        "person's Wikipedia page and note their BIRTH COUNTRY.\n",
    ]
    for t in THEORIES:
        lines.append(f"  {t['id']}: {t['theory']} — start at {t['url']}")
    lines.append(
        "\nThen produce:\n"
        "  (a) a table: Theory | Originator | Birth country | Source URLs.\n"
        "  (b) a 200+ word synthesis discussing what the three origins have in common and how they "
        "differ (nationality, era, field), citing each source URL.\n"
        "Base every fact on a visited page — do not rely on memory alone."
    )
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    return [f"Originator + birth country for {t['id']}" for t in THEORIES] + [
        "Theory/originator/country table",
        "200+ word synthesis with citations",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 visit actions executed",
        "All three originators correct (Einstein, Darwin, Mendeleev)",
        "All three birth countries correct (Germany, England/UK, Russia)",
        "Table present",
        "200+ word synthesis with citations",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    for t in THEORIES:
        if not re.search(t["originator"], text) or not re.search(t["country"], text):
            return False
    return True


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    return {
        "check": "visit_count",
        "passed": visit_count >= 4,
        "score": min(1.0, visit_count / 6.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=4 of 6 pages)",
    }


def validate_keystone_originators(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: all three originators AND birth countries correct. Hard 0/1."""
    text = extract_final_text(result).lower()
    details = {}
    all_ok = True
    for t in THEORIES:
        o = bool(re.search(t["originator"], text))
        c = bool(re.search(t["country"], text))
        details[t["id"]] = {"originator": o, "country": c}
        if not (o and c):
            all_ok = False
    return {
        "check": "keystone_originators",
        "passed": all_ok,
        "score": 1.0 if all_ok else 0.0,
        "details": details,
        "reason": "All originators+countries correct" if all_ok else "Missing originator/country",
    }


def validate_table(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Table + word count. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "table_and_length", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> table/length not credited"}
    text = extract_final_text(result)
    rows = len(re.findall(r"^\s*\|.*\|\s*$", text, re.MULTILINE))
    words = len(text.split())
    score = 0.5 * (1.0 if rows >= 4 else rows / 4.0) + 0.5 * (1.0 if words >= 200 else words / 200.0)
    return {
        "check": "table_and_length",
        "passed": rows >= 4 and words >= 200,
        "score": round(score, 3),
        "reason": f"table rows={rows}, words={words}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Per-source citations. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for tok in ("einstein", "darwin", "mendeleev") if f"wiki/{tok}" in text)
    url_count = len(re.findall(r"https?://[^\s)\\\"]+", extract_final_text(result)))
    score = max(min(1.0, cited / 3.0), min(1.0, url_count / 6.0))
    return {
        "check": "citations",
        "passed": url_count >= 4,
        "score": round(score, 3),
        "reason": f"person-page citations={cited}, total urls={url_count}",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any],
                            connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    prompt = f"""Grade this research-brief synthesis on the origins of three scientific theories.

Agent Output:
{final_text[:8000]}

Expected facts:
- General relativity -> Albert Einstein -> born in Germany
- Evolution by natural selection -> Charles Darwin -> born in England (United Kingdom)
- Periodic law / periodic table -> Dmitri Mendeleev -> born in Russia

Grade on:
1. Are all three originators and birth countries correct?
2. Is there a clear table?
3. Is the synthesis 200+ words and does it genuinely compare/contrast the three origins
   (not just restate facts)?
4. Are source URLs cited per source (provenance)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "facts_correct": int,
  "has_table": boolean,
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
    return [validate_visits, validate_keystone_originators, validate_table, validate_citations]


def get_llm_validation_function() -> callable:
    return validate_with_llm
