"""
Test 029: Retbleed Transient Execution Variant + Reference Implementation
Difficulty: 6/10 (Hard-Medium)
Category: Multi-Hop Security Research + GitHub Discovery

Requires the agent to research the Retbleed transient execution vulnerability,
identify affected architectures, research group, and disclosure date, then
independently locate a public reference implementation or evaluation framework
on GitHub and extract repository metadata.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "029",
        "test_name": "Retbleed Transient Execution + Repo Discovery",
        "difficulty_level": "6/10",
        "category": "Multi-Hop Security Research + GitHub Discovery",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Research the Retbleed CPU vulnerability:\n\n"
        "1. What class of CPUs does it impact (architectures, vendors, microarchitectures)?\n"
        "2. How does Retbleed fit into the transient execution vulnerability family? "
        "What specific speculative execution mechanism does it exploit?\n"
        "3. Who published it (research group or institution) and what was the approximate "
        "disclosure date?\n"
        "4. Find a public reference implementation, evaluation framework, or proof-of-concept "
        "for Retbleed on GitHub. Describe its purpose in 2-3 bullet points, list the number "
        "of contributors and their GitHub usernames.\n\n"
        "Search the web, visit relevant pages, and visit the actual GitHub repository."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Affected CPU architectures and vendors (Intel and AMD, specific microarchitectures)",
        "Transient execution family classification (speculative return instructions / retpoline bypass)",
        "Research group / institution and disclosure date (~2022)",
        "GitHub repository located with description",
        "Contributor count and usernames",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 visit actions executed",
        "At least 1 search action executed",
        "Transient execution / speculative execution identified",
        "Return instructions / retpoline mentioned",
        "Both Intel and AMD identified as affected",
        "ETH Zurich / research institution mentioned",
        "2022 disclosure year mentioned",
        "GitHub repo found and visited",
        "Contributors listed",
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


def validate_transient_execution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate correct classification as transient execution attack."""
    text = extract_final_text(result).lower()
    has_transient = bool(re.search(r"transient.{0,20}execution", text))
    has_speculative = bool(re.search(r"speculative.{0,20}execution", text))
    has_return = bool(re.search(r"\breturn\b", text)) and bool(re.search(r"\b(instruction|predict|specul)", text))
    has_retpoline = "retpoline" in text
    has_branch = bool(re.search(r"branch.{0,15}predict", text))

    mechanism_ok = has_return or has_retpoline or has_branch
    class_ok = has_transient or has_speculative
    passed = class_ok and mechanism_ok
    checks = int(class_ok) + int(has_return) + int(has_retpoline) + int(has_branch)
    return {
        "check": "transient_execution",
        "passed": passed,
        "score": min(1.0, checks / 2.5),
        "has_transient": has_transient,
        "has_speculative": has_speculative,
        "has_return": has_return,
        "has_retpoline": has_retpoline,
        "has_branch_prediction": has_branch,
        "reason": f"class={class_ok}, return={has_return}, retpoline={has_retpoline}, branch={has_branch}",
    }


def validate_affected_cpus(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate both Intel and AMD identified as affected."""
    text = extract_final_text(result).lower()
    has_intel = "intel" in text
    has_amd = "amd" in text
    has_zen = bool(re.search(r"zen\s*[123+]?", text))
    has_skylake = "skylake" in text
    has_kaby = "kaby" in text
    has_coffee = "coffee" in text

    vendor_ok = has_intel and has_amd
    detail = int(has_zen) + int(has_skylake) + int(has_kaby) + int(has_coffee)
    passed = vendor_ok
    score = (0.3 if has_intel else 0.0) + (0.3 if has_amd else 0.0) + min(0.4, detail * 0.15)
    return {
        "check": "affected_cpus",
        "passed": passed,
        "score": score,
        "has_intel": has_intel,
        "has_amd": has_amd,
        "microarch_details": detail,
        "reason": f"Intel={has_intel}, AMD={has_amd}, microarch_details={detail}",
    }


def validate_researchers(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate research group / institution and disclosure timing."""
    text = extract_final_text(result).lower()
    has_eth = "eth" in text or "zurich" in text
    has_vrije = "vrije" in text or "vu amsterdam" in text
    has_institution = has_eth or has_vrije or "university" in text
    has_2022 = "2022" in text
    has_johannes = "wikner" in text or "johannes" in text
    has_kaveh = "razavi" in text or "kaveh" in text

    researcher_ok = has_johannes or has_kaveh or has_institution
    passed = researcher_ok and has_2022
    score = (0.3 if has_institution else 0.0) + (0.3 if has_2022 else 0.0) + (0.2 if has_johannes or has_kaveh else 0.0) + (0.2 if has_eth or has_vrije else 0.0)
    return {
        "check": "researchers",
        "passed": passed,
        "score": min(1.0, score),
        "has_institution": has_institution,
        "has_eth": has_eth,
        "has_2022": has_2022,
        "has_researcher_names": has_johannes or has_kaveh,
        "reason": f"institution={has_institution}, 2022={has_2022}, researcher_names={has_johannes or has_kaveh}",
    }


def validate_github_repo(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a reference implementation repo was found on GitHub."""
    text = extract_final_text(result).lower()
    has_github = "github" in text
    has_retbleed_repo = bool(re.search(r"github\.com/\S*retbleed", text))
    has_repo_desc = bool(re.search(r"(proof.of.concept|poc|reference.impl|evaluation|framework|exploit)", text))
    has_contributors = bool(re.search(r"contributor", text))

    passed = has_github and has_contributors
    score = (0.3 if has_github else 0.0) + (0.3 if has_retbleed_repo else 0.0) + (0.2 if has_repo_desc else 0.0) + (0.2 if has_contributors else 0.0)
    return {
        "check": "github_repo",
        "passed": passed,
        "score": min(1.0, score),
        "has_github": has_github,
        "has_retbleed_repo": has_retbleed_repo,
        "has_repo_desc": has_repo_desc,
        "has_contributors": has_contributors,
        "reason": f"GitHub={has_github}, retbleed_repo={has_retbleed_repo}, desc={has_repo_desc}, contributors={has_contributors}",
    }


async def validate_with_llm(
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm,
    model_name: str,
) -> Dict[str, Any]:
    """LLM validation for Retbleed research quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)
    search_count = observability.get("search", {}).get("count", 0)

    prompt = f"""Validate this security research task about Retbleed:

Task: {task}

Agent Output:
{final_text[:8000]}

Observability:
- Visit actions: {visit_count}
- Search actions: {search_count}

Check:
1. Is Retbleed correctly classified as a transient/speculative execution attack?
2. Is the return instruction / retpoline bypass mechanism explained?
3. Are BOTH Intel and AMD identified as affected?
4. Is the research group (ETH Zurich and/or VU Amsterdam) and year (~2022) mentioned?
5. Was a GitHub repo found with a description and contributor information?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "classification_correct": boolean,
  "mechanism_explained": boolean,
  "vendors_correct": boolean,
  "researchers_identified": boolean,
  "repo_found": boolean
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
        validate_transient_execution,
        validate_affected_cpus,
        validate_researchers,
        validate_github_repo,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
