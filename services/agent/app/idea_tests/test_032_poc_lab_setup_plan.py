"""
Test 032: End-to-End PoC Lab Setup from Repository
Difficulty: 8/10 (Very Hard)
Category: Deep Repository Analysis + Experiment Design

Requires the agent to find a prominent PoC repo for a transient execution
vulnerability, deeply read its README and code structure, extract build
requirements and run instructions, then synthesize a practical experimental
plan for testing vendor mitigations. Tests deep repo comprehension and
the ability to produce actionable technical plans.
"""

from typing import Dict, Any, List
import re
import json
from agent.app.idea_test_utils import extract_final_text


def get_test_metadata() -> Dict[str, Any]:
    """Return test metadata."""
    return {
        "test_id": "032",
        "test_name": "PoC Lab Setup + Experiment Plan",
        "difficulty_level": "8/10",
        "category": "Deep Repository Analysis + Experiment Design",
    }


def get_task_statement() -> str:
    """Return task statement."""
    return (
        "I want to set up a lab environment to experiment with a transient execution "
        "CPU vulnerability proof-of-concept. Help me with the Downfall vulnerability "
        "(also known as GDS / Gather Data Sampling).\n\n"
        "1. Summarize the vulnerability and what a PoC for it would demonstrate "
        "(what data can be leaked, under what conditions).\n"
        "2. Find the main proof-of-concept repository for Downfall on GitHub. "
        "Visit the repository and extract:\n"
        "   (a) The main programming language and build system used\n"
        "   (b) Required dependencies or prerequisites (OS, kernel version, CPU requirements)\n"
        "   (c) The high-level steps to compile and run the PoC in a lab environment\n"
        "   (d) Number of contributors and their GitHub usernames\n"
        "3. Based on this PoC, propose a short experimental plan to test the impact of "
        "vendor microcode mitigations. List specific measurements a researcher should "
        "record (e.g., leak rate before/after mitigation, performance overhead, which "
        "CPU generations show different behavior).\n\n"
        "Search the web, visit relevant security pages, and visit the GitHub repository."
    )


def get_required_deliverables() -> List[str]:
    """Return required deliverables."""
    return [
        "Vulnerability summary (what data leaks, conditions)",
        "GitHub PoC repo found and visited",
        "Main language and build system identified",
        "Dependencies / prerequisites listed",
        "Compile and run steps extracted",
        "Contributor count and usernames",
        "Experimental plan for testing mitigations",
        "Specific measurements to record",
    ]


def get_success_criteria() -> List[str]:
    """Return success criteria."""
    return [
        "At least 2 visit actions (security page + GitHub repo)",
        "At least 1 search action",
        "Downfall/GDS vulnerability summarized",
        "GitHub repo located and visited",
        "Build instructions extracted",
        "Contributors listed",
        "Experiment plan with measurements proposed",
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


def validate_vulnerability_summary(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Downfall/GDS vulnerability is properly summarized."""
    text = extract_final_text(result).lower()
    has_downfall = "downfall" in text
    has_gds = "gather data sampling" in text or "gds" in text
    has_avx = bool(re.search(r"\bavx\b", text)) or "gather" in text
    has_leak = bool(re.search(r"\b(leak|exfiltrat|extract|steal|read|disclose)\b", text))
    has_data_type = bool(re.search(r"\b(aes|key|password|secret|cryptographic|register|buffer)\b", text))
    has_transient = bool(re.search(r"transient.{0,20}execution", text)) or bool(re.search(r"speculative", text))

    checks = int(has_downfall or has_gds) + int(has_avx) + int(has_leak) + int(has_data_type) + int(has_transient)
    passed = (has_downfall or has_gds) and has_leak and checks >= 3
    return {
        "check": "vulnerability_summary",
        "passed": passed,
        "score": min(1.0, checks / 4.0),
        "has_downfall_or_gds": has_downfall or has_gds,
        "has_avx": has_avx,
        "has_leak": has_leak,
        "has_data_type": has_data_type,
        "has_transient": has_transient,
        "reason": f"name={has_downfall or has_gds}, avx={has_avx}, leak={has_leak}, data={has_data_type}",
    }


def validate_repo_found(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate GitHub PoC repository was found and visited."""
    text = extract_final_text(result).lower()
    has_github = "github" in text
    has_downfall_repo = bool(re.search(r"github\.com/\S*downfall", text))
    has_contributors = bool(re.search(r"contributor", text))

    passed = has_github and (has_downfall_repo or has_contributors)
    score = (0.3 if has_github else 0.0) + (0.4 if has_downfall_repo else 0.0) + (0.3 if has_contributors else 0.0)
    return {
        "check": "repo_found",
        "passed": passed,
        "score": min(1.0, score),
        "has_github": has_github,
        "has_downfall_repo": has_downfall_repo,
        "has_contributors": has_contributors,
        "reason": f"GitHub={has_github}, downfall_repo={has_downfall_repo}, contribs={has_contributors}",
    }


def validate_build_info(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate build/compile information extracted from the repo."""
    text = extract_final_text(result).lower()
    has_language = bool(re.search(r"\b(c\+\+|c |assembly|asm|python|makefile)\b", text))
    has_build_system = bool(re.search(r"\b(make|cmake|gcc|g\+\+|clang|compile|build)\b", text))
    has_dependency = bool(re.search(r"\b(depend|prerequisite|require|linux|kernel|install)\b", text))
    has_cpu_req = bool(re.search(r"\b(intel|skylake|cpu|processor|core)\b", text))
    has_steps = bool(re.search(r"\b(step|clone|run|execute|cd |\.\/|make |cmake )\b", text))

    checks = int(has_language) + int(has_build_system) + int(has_dependency) + int(has_cpu_req) + int(has_steps)
    passed = has_build_system and has_steps and checks >= 3
    return {
        "check": "build_info",
        "passed": passed,
        "score": min(1.0, checks / 4.0),
        "has_language": has_language,
        "has_build_system": has_build_system,
        "has_dependency": has_dependency,
        "has_cpu_requirement": has_cpu_req,
        "has_steps": has_steps,
        "reason": f"lang={has_language}, build={has_build_system}, deps={has_dependency}, cpu={has_cpu_req}, steps={has_steps}",
    }


def validate_experiment_plan(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the proposed experimental plan for testing mitigations."""
    text = extract_final_text(result).lower()
    has_mitigation = bool(re.search(r"\b(mitigation|microcode|patch|update|fix)\b", text))
    has_before_after = bool(re.search(r"\b(before|after|with.{0,10}without|enable|disable)\b", text))
    has_measurement = bool(re.search(r"\b(measur|record|log|metric|rate|throughput|latency|overhead)\b", text))
    has_leak_rate = bool(re.search(r"\b(leak.{0,10}rate|bytes?.{0,10}(per|\/)\s*(second|sec|s\b)|bandwidth)\b", text))
    has_performance = bool(re.search(r"\b(performance|overhead|slowdown|benchmark|throughput)\b", text))
    has_cpu_gen = bool(re.search(r"\b(generation|skylake|ice.?lake|tiger|cpu.{0,10}(model|version|gen))\b", text))
    has_plan_structure = bool(re.search(r"\b(step|phase|experiment|trial|test.{0,5}\d|procedure)\b", text))

    checks = int(has_mitigation) + int(has_before_after) + int(has_measurement) + int(has_leak_rate) + int(has_performance) + int(has_cpu_gen) + int(has_plan_structure)
    passed = has_mitigation and has_measurement and checks >= 4
    return {
        "check": "experiment_plan",
        "passed": passed,
        "score": min(1.0, checks / 5.0),
        "has_mitigation": has_mitigation,
        "has_before_after": has_before_after,
        "has_measurement": has_measurement,
        "has_leak_rate": has_leak_rate,
        "has_performance": has_performance,
        "has_cpu_generations": has_cpu_gen,
        "has_plan_structure": has_plan_structure,
        "reason": f"mitigation={has_mitigation}, before/after={has_before_after}, measure={has_measurement}, leak_rate={has_leak_rate}, perf={has_performance}, cpus={has_cpu_gen}",
    }


async def validate_with_llm(
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm,
    model_name: str,
) -> Dict[str, Any]:
    """LLM validation for PoC lab setup and experiment plan quality."""
    final_text = extract_final_text(result)
    task = get_task_statement()
    visit_count = observability.get("visit", {}).get("count", 0)
    search_count = observability.get("search", {}).get("count", 0)

    prompt = f"""Validate this end-to-end PoC lab setup and experiment design task:

Task: {task}

Agent Output:
{final_text[:10000]}

Observability:
- Visit actions: {visit_count}
- Search actions: {search_count}

Check:
1. Is the Downfall/GDS vulnerability clearly summarized (data leakage, conditions)?
2. Was a GitHub PoC repo actually found and visited (not just mentioned)?
3. Is the main language and build system identified?
4. Are dependencies/prerequisites listed (OS, kernel, CPU requirements)?
5. Are compile/run steps provided?
6. Are contributors listed with usernames?
7. Is there a practical experimental plan for testing mitigations?
8. Does the plan include specific measurements (leak rate, performance overhead, CPU generations)?
9. Is the plan actionable enough that a researcher could follow it?

Return JSON:
{{
  "passed": boolean,
  "score": float (0.0-1.0),
  "reasons": [string],
  "vuln_summary_quality": string,
  "repo_analysis_quality": string,
  "build_info_complete": boolean,
  "experiment_plan_actionable": boolean,
  "measurements_specific": boolean
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
        validate_vulnerability_summary,
        validate_repo_found,
        validate_build_info,
        validate_experiment_plan,
    ]


def get_llm_validation_function() -> callable:
    """Return LLM validation function."""
    return validate_with_llm
