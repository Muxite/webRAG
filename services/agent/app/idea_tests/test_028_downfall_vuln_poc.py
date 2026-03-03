"""
Test 028: Downfall CPU Vulnerability + PoC Repository Discovery
Difficulty: 6/10 (Hard-Medium)
Category: Multi-Hop Security Research + GitHub Discovery

Requires the agent to research the Downfall transient execution CPU vulnerability,
correctly identify its attack class, affected hardware, discoverer, and then
independently locate a public proof-of-concept repository on GitHub (without being
given the URL) and extract contributor metadata.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "028",
        "test_name": "Downfall Vulnerability + PoC Repo Discovery",
        "difficulty_level": "6/10",
        "category": "Multi-Hop Security Research + GitHub Discovery",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "I'm trying to understand the Downfall CPU vulnerability in detail.\n\n"
        "1. What class of attack is it, and what CPU feature does it abuse?\n"
        "2. Which instruction set / vector extensions are primarily involved?\n"
        "3. Who discovered it and when was it publicly disclosed?\n"
        "4. Which vendors or CPU generations are affected?\n"
        "5. There's a public proof-of-concept implementation for Downfall that people "
        "use for experiments. Without me telling you where it is, can you find that "
        "kind of repo and tell me: how many contributors it has and what their GitHub "
        "usernames are?\n\n"
        "Search the web, visit pages, and visit the actual GitHub repository to answer."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Attack class identification (transient execution / speculative execution side-channel)",
        "CPU feature abused (AVX gather / vector extensions)",
        "Discoverer name (Daniel Moghimi) and disclosure date (~August 2023)",
        "Affected vendors / CPU generations (Intel, Skylake through Ice Lake era)",
        "PoC repository located on GitHub",
        "Contributor count (3)",
        "Contributor usernames (mcu-administrator, pdxphil, esyr-rh)",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 visit actions executed",
        "At least 1 search action executed",
        "Transient execution / speculative execution identified",
        "AVX / GATHER mentioned",
        "Daniel Moghimi identified as discoverer",
        "August 2023 disclosure mentioned",
        "Intel CPUs identified as affected",
        "PoC repo found and visited on GitHub",
        "3 contributors reported",
        "Contributor usernames listed",
    ]


# ── validation functions ──────────────────────────────────────────────────

def validate_visit_and_search(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the agent performed both search and visit actions."""
    visit_count = observability.get("visit", {}).get("count", 0)
    search_count = observability.get("search", {}).get("count", 0)
    passed = visit_count >= 2 and search_count >= 1
    score = min(1.0, visit_count / 3.0) * 0.5 + min(1.0, search_count / 2.0) * 0.5
    return {
        "check": "visit_and_search",
        "passed": passed,
        "score": score,
        "visit_count": visit_count,
        "search_count": search_count,
        "reason": f"{visit_count} visit(s), {search_count} search(es)",
    }


def validate_attack_class(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate correct identification of the attack class."""
    text = extract_final_text(result).lower()
    has_transient = bool(re.search(r"transient.{0,20}execution", text))
    has_speculative = bool(re.search(r"speculative.{0,20}execution", text))
    has_side_channel = "side.channel" in text.replace("-", ".") or "side channel" in text
    has_gds = "gather data sampling" in text or "gds" in text

    checks = int(has_transient or has_speculative) + int(has_side_channel) + int(has_gds)
    passed = (has_transient or has_speculative) and has_side_channel
    return {
        "check": "attack_class",
        "passed": passed,
        "score": min(1.0, checks / 2.0),
        "has_transient_execution": has_transient,
        "has_speculative_execution": has_speculative,
        "has_side_channel": has_side_channel,
        "has_gds": has_gds,
        "reason": f"transient/spec={has_transient or has_speculative}, side-channel={has_side_channel}, GDS={has_gds}",
    }


def validate_avx_gather(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate AVX / GATHER instruction involvement mentioned."""
    text = extract_final_text(result).lower()
    has_avx = bool(re.search(r"\bavx\b", text))
    has_avx2 = "avx2" in text or "avx-2" in text
    has_avx512 = bool(re.search(r"avx.?512", text))
    has_gather = "gather" in text
    has_vector = "vector" in text

    checks = int(has_avx or has_avx2 or has_avx512) + int(has_gather) + int(has_vector)
    passed = (has_avx or has_avx2 or has_avx512) and has_gather
    return {
        "check": "avx_gather",
        "passed": passed,
        "score": min(1.0, checks / 2.0),
        "has_avx": has_avx or has_avx2 or has_avx512,
        "has_gather": has_gather,
        "has_vector": has_vector,
        "reason": f"AVX={has_avx or has_avx2 or has_avx512}, GATHER={has_gather}, vector={has_vector}",
    }


def validate_discoverer(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate discoverer identification."""
    text = extract_final_text(result).lower()
    has_moghimi = "moghimi" in text
    has_daniel = "daniel" in text
    has_2023 = "2023" in text
    has_august = "august" in text

    name_ok = has_moghimi
    date_ok = has_2023
    passed = name_ok and date_ok
    score = (0.5 if name_ok else 0.0) + (0.25 if has_2023 else 0.0) + (0.25 if has_august else 0.0)
    return {
        "check": "discoverer",
        "passed": passed,
        "score": score,
        "has_moghimi": has_moghimi,
        "has_daniel": has_daniel,
        "has_2023": has_2023,
        "has_august": has_august,
        "reason": f"Moghimi={has_moghimi}, 2023={has_2023}, August={has_august}",
    }


def validate_affected_hardware(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate affected hardware identification (Intel CPUs, specific generations)."""
    text = extract_final_text(result).lower()
    has_intel = "intel" in text
    has_skylake = "skylake" in text
    has_ice_lake = bool(re.search(r"ice.?lake", text))
    has_tiger_lake = bool(re.search(r"tiger.?lake", text))
    has_gen_mention = bool(re.search(r"(6th|7th|8th|9th|10th|11th).{0,10}gen", text))
    has_core = "core" in text

    gen_detail = int(has_skylake) + int(has_ice_lake) + int(has_tiger_lake) + int(has_gen_mention)
    passed = has_intel and gen_detail >= 1
    score = (0.5 if has_intel else 0.0) + min(0.5, gen_detail * 0.2)
    return {
        "check": "affected_hardware",
        "passed": passed,
        "score": score,
        "has_intel": has_intel,
        "has_skylake": has_skylake,
        "has_ice_lake": has_ice_lake,
        "has_tiger_lake": has_tiger_lake,
        "has_gen_mention": has_gen_mention,
        "reason": f"Intel={has_intel}, gen_details={gen_detail}",
    }


def validate_poc_repo(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a PoC repository was found on GitHub."""
    text = extract_final_text(result).lower()
    has_github = "github" in text
    has_downfall_repo = bool(re.search(r"github\.com/\S*downfall", text))
    has_poc = bool(re.search(r"\b(poc|proof.of.concept|proof of concept)\b", text))
    has_repo = "repository" in text or "repo" in text

    passed = has_github and (has_downfall_repo or has_poc)
    score = (0.4 if has_github else 0.0) + (0.4 if has_downfall_repo else 0.0) + (0.2 if has_poc else 0.0)
    return {
        "check": "poc_repo_found",
        "passed": passed,
        "score": min(1.0, score),
        "has_github": has_github,
        "has_downfall_repo": has_downfall_repo,
        "has_poc": has_poc,
        "reason": f"GitHub={has_github}, downfall_repo={has_downfall_repo}, PoC={has_poc}",
    }


def validate_contributors(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate contributor count and usernames from the PoC repo."""
    text = extract_final_text(result).lower()

    expected_usernames = ["mcu-administrator", "pdxphil", "esyr-rh"]
    found_usernames = [u for u in expected_usernames if u in text]

    has_three = bool(re.search(r"\b3\b.*contributor", text)) or bool(re.search(r"three.*contributor", text))
    username_count = len(found_usernames)

    passed = username_count >= 2 and has_three
    score = (0.3 if has_three else 0.0) + (username_count / len(expected_usernames)) * 0.7
    return {
        "check": "contributors",
        "passed": passed,
        "score": min(1.0, score),
        "has_three_count": has_three,
        "found_usernames": found_usernames,
        "username_match_count": username_count,
        "reason": f"3 mentioned={has_three}, usernames found={username_count}/3 ({', '.join(found_usernames)})",
    }


async def validate_with_llm(
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm,
    model_name: str,
) -> Dict[str, Any]:
    """LLM validation for Downfall vulnerability research quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)
    search_count = observability.get("search", {}).get("count", 0)

    prompt = f"""Validate this multi-hop security research task:

Task: {task}

Agent Output:
{final_text[:8000]}

Observability:
- Visit actions: {visit_count}
- Search actions: {search_count}

Check these specific facts:
1. Is Downfall correctly identified as a transient/speculative execution side-channel attack? (not buffer overflow, not generic malware)
2. Is AVX / AVX2 / AVX-512 / GATHER instruction involvement clearly mentioned?
3. Is Daniel Moghimi identified as the discoverer, with disclosure around August 2023?
4. Are Intel CPUs identified as affected, with specific generations (Skylake through Ice Lake era)?
5. Did the agent independently find a PoC repository on GitHub (not given the URL)?
6. Does it report 3 contributors with usernames mcu-administrator, pdxphil, esyr-rh?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "attack_class_correct": boolean,
  "avx_mentioned": boolean,
  "discoverer_correct": boolean,
  "hardware_correct": boolean,
  "repo_found": boolean,
  "contributors_correct": boolean
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
        return {
            "check": "llm_validation",
            "passed": False,
            "score": 0.0,
            "error": str(exc),
        }


def get_validation_functions() -> List[callable]:
    """Return validation functions."""
    return [
        validate_visit_and_search,
        validate_attack_class,
        validate_avx_gather,
        validate_discoverer,
        validate_affected_hardware,
        validate_poc_repo,
        validate_contributors,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
