"""
Test 030: Microarchitectural Fuzzing Framework (Paper -> Repo)
Difficulty: 7/10 (Hard)
Category: Academic Paper to Repository Chain

Requires the agent to chain: understand the concept of microarchitectural security
fuzzing -> find a relevant framework (e.g., Osiris, Transynther, or similar tools
referenced in transient execution literature) -> locate the corresponding open-source
repository -> extract structured metadata (language, contributors, test harnesses).

This tests the agent's ability to pivot from academic/Wikipedia-style knowledge
to practical code artifacts on GitHub.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "030",
        "test_name": "Microarchitectural Fuzzing Framework Discovery",
        "difficulty_level": "7/10",
        "category": "Academic Paper to Repository Chain",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "I'm researching microarchitectural security testing tools, specifically "
        "fuzzing frameworks that can discover transient execution vulnerabilities "
        "or other CPU side-channel bugs.\n\n"
        "1. What problem do microarchitectural fuzzers solve in the context of CPU security? "
        "Why is this harder than normal software fuzzing?\n"
        "2. Name at least two specific microarchitectural fuzzing tools or frameworks "
        "(e.g., from academic papers or the transient execution vulnerability literature). "
        "For each, briefly describe what it does and what classes of bugs it targets.\n"
        "3. Find an open-source implementation of one of these frameworks on GitHub. "
        "Visit the repository and answer:\n"
        "   - What language is the core implementation in?\n"
        "   - How many contributors does the repository list, and what are their usernames?\n"
        "   - Does the repo provide any example configurations or test harnesses?\n\n"
        "Search the web, visit relevant pages and the GitHub repository."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Explanation of microarchitectural fuzzing problem",
        "Why it differs from software fuzzing",
        "At least 2 named fuzzing tools/frameworks",
        "Brief description of each tool's purpose and bug classes",
        "GitHub repo visited for one framework",
        "Core implementation language identified",
        "Contributor count and usernames",
        "Example configs / test harnesses mentioned",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 visit actions executed",
        "At least 1 search action executed",
        "Microarchitectural fuzzing concept explained",
        "At least 2 tools/frameworks named",
        "GitHub repo located and visited",
        "Programming language identified",
        "Contributors listed",
    ]


# ── validation functions ──────────────────────────────────────────────────

def validate_visit_and_search(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate search and visit actions performed."""
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


def validate_fuzzing_concept(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate explanation of microarchitectural fuzzing."""
    text = extract_final_text(result).lower()
    has_microarch = bool(re.search(r"micro.?architect", text))
    has_fuzzing = "fuzz" in text
    has_transient = "transient" in text or "speculative" in text
    has_side_channel = "side.channel" in text.replace("-", ".") or "side channel" in text
    has_hardware = "hardware" in text or "cpu" in text or "processor" in text
    has_difficulty = bool(re.search(r"(harder|difficult|challenge|complex|non.?deterministic|timing)", text))

    concept_ok = has_microarch and has_fuzzing
    checks = int(has_microarch) + int(has_fuzzing) + int(has_transient or has_side_channel) + int(has_hardware) + int(has_difficulty)
    passed = concept_ok and checks >= 3
    return {
        "check": "fuzzing_concept",
        "passed": passed,
        "score": min(1.0, checks / 4.0),
        "has_microarch": has_microarch,
        "has_fuzzing": has_fuzzing,
        "has_transient_or_sidechannel": has_transient or has_side_channel,
        "has_difficulty_explained": has_difficulty,
        "reason": f"concept={concept_ok}, detail_checks={checks}/5",
    }


def validate_tools_named(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate at least 2 named fuzzing tools/frameworks mentioned."""
    text = extract_final_text(result).lower()
    # Known microarchitectural fuzzing/testing tools from the literature
    known_tools = [
        "osiris", "transynther", "revizor", "specfuzz", "spectector",
        "riscover", "speechminer", "cacheout", "medusa", "ridl",
        "kasper", "pandora", "fpvi", "bhi", "inception",
    ]
    found_tools = [t for t in known_tools if t in text]

    # Also check for generic tool-like mentions
    generic_tool_pattern = re.findall(r"\b[A-Z][a-zA-Z]+(?:Fuzz|Test|Scan|Check|Verify)\b", extract_final_text(result))

    total_tools = len(set(found_tools)) + min(1, len(generic_tool_pattern))
    passed = total_tools >= 2
    return {
        "check": "tools_named",
        "passed": passed,
        "score": min(1.0, total_tools / 2.0),
        "known_tools_found": found_tools,
        "tool_count": total_tools,
        "reason": f"Found {total_tools} tool(s): {', '.join(found_tools) if found_tools else 'none known'}",
    }


def validate_github_repo(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a GitHub repository was found and visited."""
    text = extract_final_text(result).lower()
    has_github = "github" in text
    has_github_url = bool(re.search(r"github\.com/\S+", text))
    has_language = bool(re.search(r"\b(python|c\+\+|rust|c |assembly|java|go)\b", text))
    has_contributors = bool(re.search(r"contributor", text))
    has_username = bool(re.search(r"@?\w+[-_]\w+", text))  # GitHub-style usernames

    passed = has_github and has_contributors and has_language
    score = (0.25 if has_github else 0.0) + (0.25 if has_github_url else 0.0) + (0.25 if has_language else 0.0) + (0.25 if has_contributors else 0.0)
    return {
        "check": "github_repo",
        "passed": passed,
        "score": min(1.0, score),
        "has_github": has_github,
        "has_github_url": has_github_url,
        "has_language": has_language,
        "has_contributors": has_contributors,
        "reason": f"GitHub={has_github}, URL={has_github_url}, lang={has_language}, contribs={has_contributors}",
    }


def validate_repo_details(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that repo details (configs, test harnesses) are mentioned."""
    text = extract_final_text(result).lower()
    has_config = bool(re.search(r"(config|configuration|\.yaml|\.json|\.toml|settings)", text))
    has_test = bool(re.search(r"(test.?harness|test.?suite|example|demo|benchmark|sample)", text))
    has_readme = "readme" in text
    has_build = bool(re.search(r"(makefile|cmake|cargo|setup\.py|build|compile)", text))

    checks = int(has_config) + int(has_test) + int(has_readme) + int(has_build)
    passed = checks >= 2
    return {
        "check": "repo_details",
        "passed": passed,
        "score": min(1.0, checks / 3.0),
        "has_config": has_config,
        "has_test_harness": has_test,
        "has_build_info": has_build,
        "reason": f"config={has_config}, test={has_test}, build={has_build}",
    }


async def validate_with_llm(
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm,
    model_name: str,
) -> Dict[str, Any]:
    """LLM validation for microarchitectural fuzzing research quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)

    prompt = f"""Validate this academic-to-repo research task:

Task: {task}

Agent Output:
{final_text[:8000]}

Observability:
- Visit actions: {visit_count}

Check:
1. Is microarchitectural fuzzing clearly explained as a concept?
2. Is the difficulty vs normal software fuzzing addressed?
3. Are at least 2 specific tools/frameworks named with descriptions?
4. Was a GitHub repository actually visited (not just mentioned)?
5. Is the implementation language identified?
6. Are contributors listed with usernames?
7. Are example configs or test harnesses mentioned?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "concept_explained": boolean,
  "tools_count": int,
  "repo_visited": boolean,
  "language_identified": boolean,
  "contributors_listed": boolean
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
        validate_fuzzing_concept,
        validate_tools_named,
        validate_github_repo,
        validate_repo_details,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
