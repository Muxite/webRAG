"""
Test 031: Compound Multi-Vulnerability + Multi-Repo Comparison
Difficulty: 8/10 (Very Hard)
Category: Multi-Vulnerability Cross-Reference + Repository Comparison

Requires the agent to research multiple transient execution vulnerabilities,
find a distinct PoC/research repo for each, then aggregate and compare
repository metadata (contributors, activity, purpose). This forces parallel
fact-gathering across security writeups and multiple GitHub projects.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "031",
        "test_name": "Compound Vulnerability + Repo Comparison",
        "difficulty_level": "8/10",
        "category": "Multi-Vulnerability Cross-Reference + Repository Comparison",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "Compare two transient execution CPU vulnerabilities: Downfall and Retbleed.\n\n"
        "For EACH vulnerability, research and provide:\n"
        "  (a) The attack category (what kind of transient execution attack)\n"
        "  (b) The affected vendor(s) and CPU microarchitectures\n"
        "  (c) What type of secret data can potentially be leaked\n\n"
        "Then, for EACH vulnerability, find one PoC or research repository on GitHub:\n"
        "  (d) A short description of what the repo does (e.g., 'minimal PoC to leak AES keys', "
        "'benchmark harness to measure mitigation overhead')\n"
        "  (e) Number of contributors and their GitHub usernames\n\n"
        "Finally, compare the two repositories:\n"
        "  (f) Which repo appears more actively maintained or popular (based on stars, "
        "recent commits, issues, or any visible activity indicators)?\n\n"
        "Search the web and visit the relevant pages and GitHub repositories."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Downfall: attack category, affected CPUs, leakable data",
        "Retbleed: attack category, affected CPUs, leakable data",
        "Downfall PoC repo: description, contributors, usernames",
        "Retbleed research repo: description, contributors, usernames",
        "Comparison of repo activity/maintenance",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 3 visit actions (security pages + 2 repos)",
        "At least 2 search actions",
        "Both Downfall and Retbleed correctly described",
        "Both have affected hardware identified",
        "Two distinct GitHub repos found",
        "Contributors listed for both repos",
        "Comparison/activity assessment provided",
    ]


# ── validation functions ──────────────────────────────────────────────────

def validate_visit_and_search(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sufficient search and visit actions for a multi-vuln task."""
    visit_count = observability.get("visit", {}).get("count", 0)
    search_count = observability.get("search", {}).get("count", 0)
    passed = visit_count >= 3 and search_count >= 2
    score = min(1.0, visit_count / 4.0) * 0.5 + min(1.0, search_count / 3.0) * 0.5
    return {
        "check": "visit_and_search",
        "passed": passed,
        "score": score,
        "visit_count": visit_count,
        "search_count": search_count,
        "reason": f"{visit_count} visit(s), {search_count} search(es)",
    }


def validate_downfall_description(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Downfall vulnerability is correctly described."""
    text = extract_final_text(result).lower()
    has_downfall = "downfall" in text
    has_transient = bool(re.search(r"transient.{0,20}execution", text)) or bool(re.search(r"speculative.{0,20}execution", text))
    has_avx = bool(re.search(r"\bavx\b", text)) or "gather" in text
    has_intel = "intel" in text
    has_leak = bool(re.search(r"\b(leak|exfiltrat|extract|steal|read|disclose)\b", text))

    checks = int(has_downfall) + int(has_transient) + int(has_avx) + int(has_intel) + int(has_leak)
    passed = has_downfall and has_intel and (has_transient or has_avx)
    return {
        "check": "downfall_description",
        "passed": passed,
        "score": min(1.0, checks / 4.0),
        "has_downfall": has_downfall,
        "has_transient": has_transient,
        "has_avx": has_avx,
        "has_intel": has_intel,
        "has_leak": has_leak,
        "reason": f"Downfall: name={has_downfall}, transient={has_transient}, avx={has_avx}, intel={has_intel}, leak={has_leak}",
    }


def validate_retbleed_description(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Retbleed vulnerability is correctly described."""
    text = extract_final_text(result).lower()
    has_retbleed = "retbleed" in text
    has_transient = bool(re.search(r"transient.{0,20}execution", text)) or bool(re.search(r"speculative.{0,20}execution", text))
    has_return = "return" in text or "retpoline" in text
    has_amd = "amd" in text
    has_intel = "intel" in text
    has_leak = bool(re.search(r"\b(leak|exfiltrat|extract|steal|read|disclose)\b", text))

    checks = int(has_retbleed) + int(has_transient) + int(has_return) + int(has_amd or has_intel) + int(has_leak)
    passed = has_retbleed and (has_amd or has_intel) and (has_transient or has_return)
    return {
        "check": "retbleed_description",
        "passed": passed,
        "score": min(1.0, checks / 4.0),
        "has_retbleed": has_retbleed,
        "has_transient": has_transient,
        "has_return": has_return,
        "has_amd": has_amd,
        "has_intel": has_intel,
        "has_leak": has_leak,
        "reason": f"Retbleed: name={has_retbleed}, transient={has_transient}, return={has_return}, vendors={has_amd or has_intel}",
    }


def validate_two_repos_found(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate two distinct GitHub repositories were found."""
    text = extract_final_text(result).lower()
    github_urls = re.findall(r"github\.com/[\w\-]+/[\w\-]+", text)
    unique_repos = list(set(github_urls))
    has_contributors = text.count("contributor") >= 2 or text.count("username") >= 2

    passed = len(unique_repos) >= 2 and has_contributors
    score = min(1.0, len(unique_repos) / 2.0) * 0.6 + (0.4 if has_contributors else 0.0)
    return {
        "check": "two_repos_found",
        "passed": passed,
        "score": score,
        "unique_repo_count": len(unique_repos),
        "repos": unique_repos[:5],
        "has_contributors_for_both": has_contributors,
        "reason": f"Found {len(unique_repos)} unique repo(s), contributors_mentioned={has_contributors}",
    }


def validate_repo_comparison(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a comparison between the two repositories was provided."""
    text = extract_final_text(result).lower()
    has_comparison = bool(re.search(r"(compar|more.{0,15}(active|maintained|popular|recent)|less.{0,15}(active|maintained)|star|commit|issue)", text))
    has_stars = bool(re.search(r"\b\d+\s*star", text))
    has_activity = bool(re.search(r"(recent|last|latest).{0,20}(commit|update|release|push)", text))
    has_verdict = bool(re.search(r"(more.{0,10}(active|maintained|popular)|appears.{0,10}(more|less))", text))

    checks = int(has_comparison) + int(has_stars) + int(has_activity) + int(has_verdict)
    passed = has_comparison and checks >= 2
    return {
        "check": "repo_comparison",
        "passed": passed,
        "score": min(1.0, checks / 3.0),
        "has_comparison": has_comparison,
        "has_stars": has_stars,
        "has_activity": has_activity,
        "has_verdict": has_verdict,
        "reason": f"comparison={has_comparison}, stars={has_stars}, activity={has_activity}, verdict={has_verdict}",
    }


async def validate_with_llm(
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm,
    model_name: str,
) -> Dict[str, Any]:
    """LLM validation for compound vulnerability + repo comparison."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)
    search_count = observability.get("search", {}).get("count", 0)

    prompt = f"""Validate this compound security research + repo comparison task:

Task: {task}

Agent Output:
{final_text[:10000]}

Observability:
- Visit actions: {visit_count}
- Search actions: {search_count}

Check:
1. Is Downfall correctly described (transient execution, AVX/GATHER, Intel)?
2. Is Retbleed correctly described (speculative return, Intel/AMD)?
3. Are leakable data types mentioned for each vulnerability?
4. Were TWO distinct GitHub repos found (one per vulnerability)?
5. Are contributors listed for both repos?
6. Is there a meaningful comparison of repo activity/maintenance?
7. Did the agent visit actual pages (not just search results)?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "downfall_correct": boolean,
  "retbleed_correct": boolean,
  "two_repos_found": boolean,
  "contributors_both": boolean,
  "comparison_present": boolean
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
        validate_downfall_description,
        validate_retbleed_description,
        validate_two_repos_found,
        validate_repo_comparison,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
