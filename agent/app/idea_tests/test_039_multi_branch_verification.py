"""
Test 039: Multi-Branch Claim Verification
Difficulty: 10/10 (Maximum)
Category: Parallel Verification & Confidence Rating

The agent receives four bold factual claims from different domains.
For each claim it must:
  1. Search for supporting and contradicting evidence.
  2. Visit at least one authoritative source per claim.
  3. Rate the claim as TRUE / PARTIALLY TRUE / FALSE with evidence.

Then produce a structured verdict report with a confidence score per claim
and an overall reliability summary.

This is designed to be near-impossible for a sequential planner: it requires
4 independent investigation branches each with search+visit+analysis, plus a
final synthesis.  The sequential node budget (80) is far too small.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


CLAIMS = [
    {
        "id": "C1",
        "claim": "The speed of light in a vacuum is approximately 299,792,458 metres per second.",
        "domain": "Physics",
        "truth": "TRUE",
        "evidence_patterns": [r"299,?792,?458|speed of light|3\s*×?\s*10\^?8"],
        "url_hint": "https://en.wikipedia.org/wiki/Speed_of_light",
    },
    {
        "id": "C2",
        "claim": "Leonardo da Vinci painted the ceiling of the Sistine Chapel.",
        "domain": "Art History",
        "truth": "FALSE",
        "evidence_patterns": [r"michelangelo|sistine.*michelangelo|not.*da vinci|false|incorrect"],
        "url_hint": "https://en.wikipedia.org/wiki/Sistine_Chapel_ceiling",
    },
    {
        "id": "C3",
        "claim": "Octopuses have three hearts and blue blood.",
        "domain": "Biology",
        "truth": "TRUE",
        "evidence_patterns": [r"three hearts|3 hearts|blue blood|haemocyanin|hemocyanin"],
        "url_hint": "https://en.wikipedia.org/wiki/Octopus",
    },
    {
        "id": "C4",
        "claim": "Mount Everest is located on the border between Nepal and India.",
        "domain": "Geography",
        "truth": "FALSE",
        "evidence_patterns": [r"(china|tibet|nepal.*china|china.*nepal|not.*india|false|incorrect)"],
        "url_hint": "https://en.wikipedia.org/wiki/Mount_Everest",
    },
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "039",
        "test_name": "Multi-Branch Claim Verification",
        "difficulty_level": "10/10",
        "category": "Parallel Verification & Confidence Rating",
    }


def get_task_statement() -> str:
    lines = [
        "You are a fact-checker.  For each of the following four claims:\n"
        "  (a) Search for evidence (both supporting and contradicting).\n"
        "  (b) Visit at least one authoritative source (Wikipedia is acceptable).\n"
        "  (c) Determine whether the claim is TRUE, PARTIALLY TRUE, or FALSE.\n"
        "  (d) Provide a confidence score (0.0–1.0) and a short justification citing the source URL.\n\n"
        "Claims to verify:\n"
    ]
    for c in CLAIMS:
        lines.append(f'  {c["id"]} [{c["domain"]}]: "{c["claim"]}"')
    lines.append(
        "\nAfter verifying all claims, produce a VERDICT REPORT with:\n"
        "- A verdict table: Claim ID | Verdict | Confidence | Key Evidence | Source URL\n"
        "- A 100+ word reliability summary discussing which claims were straightforward "
        "to verify and which required deeper investigation.\n"
        "Cite every URL used."
    )
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    return [
        f"Verdict for {c['id']} ({c['domain']})" for c in CLAIMS
    ] + [
        "Confidence score per claim",
        "Evidence citation per claim",
        "Verdict table",
        "100+ word reliability summary",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 visit actions (one per claim ideally)",
        "At least 3 of 4 verdicts correct",
        "Confidence scores present",
        "Verdict table in output",
        "Reliability summary present (100+ words)",
        "Source URLs cited per claim",
    ]


# ── validators ──────────────────────────────────────────────────────

def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    passed = visit_count >= 3
    return {
        "check": "visit_count",
        "passed": passed,
        "score": min(1.0, visit_count / 4.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=3)",
    }


def validate_verdicts(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    correct = 0
    details = {}
    for c in CLAIMS:
        claim_region_found = c["id"].lower() in text or c["domain"].lower() in text
        evidence_found = any(
            bool(re.search(p, text, re.IGNORECASE)) for p in c["evidence_patterns"]
        )
        if c["truth"] == "TRUE":
            verdict_ok = bool(re.search(
                rf'({c["id"]}[^.]*?(true|correct|confirmed|verified))', text, re.IGNORECASE
            )) or evidence_found
        else:
            verdict_ok = bool(re.search(
                rf'({c["id"]}[^.]*?(false|incorrect|wrong|inaccurate|partially))', text, re.IGNORECASE
            )) or evidence_found
        hit = claim_region_found and evidence_found and verdict_ok
        details[c["id"]] = {
            "claim_found": claim_region_found,
            "evidence_found": evidence_found,
            "verdict_correct": verdict_ok,
            "overall": hit,
        }
        if hit:
            correct += 1
    passed = correct >= 3
    return {
        "check": "verdict_correctness",
        "passed": passed,
        "score": correct / len(CLAIMS),
        "correct": correct,
        "total": len(CLAIMS),
        "details": details,
        "reason": f"{correct}/{len(CLAIMS)} verdicts correct",
    }


def validate_confidence_scores(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    scores_found = re.findall(r"(?:confidence|score)[^0-9]*(0\.\d+|1\.0)", text.lower())
    unique = set(scores_found)
    has_enough = len(unique) >= 2
    return {
        "check": "confidence_scores",
        "passed": has_enough,
        "score": min(1.0, len(unique) / 4.0),
        "scores_found": list(unique),
        "reason": f"Found {len(unique)} distinct confidence scores",
    }


def validate_verdict_table(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    pipe_rows = len(re.findall(r"^\s*\|.+\|", text, re.MULTILINE))
    has_table = pipe_rows >= 3
    claim_ids_in_table = sum(1 for c in CLAIMS if c["id"].lower() in text.lower())
    score = (0.5 if has_table else min(0.5, pipe_rows / 6.0)) + (0.5 * claim_ids_in_table / len(CLAIMS))
    return {
        "check": "verdict_table",
        "passed": has_table and claim_ids_in_table >= 3,
        "score": score,
        "pipe_rows": pipe_rows,
        "claims_in_table": claim_ids_in_table,
        "reason": f"table_rows={pipe_rows}, claims_in_table={claim_ids_in_table}/4",
    }


def validate_reliability_summary(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result)
    lower = text.lower()
    has_summary_marker = bool(re.search(r"(reliab|summary|overall|conclusion|finding)", lower))
    paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 20]
    longest = max(paragraphs, key=len, default="") if paragraphs else ""
    word_count = len(longest.split())
    length_ok = word_count >= 80
    score = (0.4 if has_summary_marker else 0.0) + (0.6 if length_ok else min(0.6, word_count / 100.0 * 0.6))
    return {
        "check": "reliability_summary",
        "passed": has_summary_marker and length_ok,
        "score": score,
        "word_count": word_count,
        "reason": f"summary_marker={has_summary_marker}, words={word_count}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    text = extract_final_text(result).lower()
    cited = 0
    details = {}
    for c in CLAIMS:
        fragment = c["url_hint"].replace("https://en.wikipedia.org/wiki/", "").lower()
        present = fragment in text or c["url_hint"].lower() in text or "wikipedia.org" in text
        details[c["id"]] = present
        if present:
            cited += 1
    return {
        "check": "citation_urls",
        "passed": cited >= 3,
        "score": cited / len(CLAIMS),
        "cited": cited,
        "total": len(CLAIMS),
        "details": details,
        "reason": f"{cited}/{len(CLAIMS)} claims cited",
    }


async def validate_with_llm(result: Dict[str, Any], observability: Dict[str, Any], connector_llm, model_name: str) -> Dict[str, Any]:
    final_text = extract_final_text(result)
    task = get_task_statement()
    prompt = f"""Validate this multi-branch claim verification task:

Task: {task}

Agent Output:
{final_text[:8000]}

Expected verdicts:
- C1 (Speed of light): TRUE — ~299,792,458 m/s
- C2 (Sistine Chapel): FALSE — Michelangelo, not da Vinci
- C3 (Octopus hearts): TRUE — 3 hearts, blue blood (hemocyanin)
- C4 (Everest border): FALSE — Nepal/China (Tibet), not Nepal/India

Check:
1. Are all 4 claims addressed?
2. Are at least 3 verdicts correct?
3. Are confidence scores provided?
4. Is there a verdict table?
5. Is there a 100+ word reliability summary?
6. Are source URLs cited?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "verdicts_correct": int,
  "has_table": boolean,
  "summary_quality": string
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
    return [
        validate_visits,
        validate_verdicts,
        validate_confidence_scores,
        validate_verdict_table,
        validate_reliability_summary,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return validate_with_llm
