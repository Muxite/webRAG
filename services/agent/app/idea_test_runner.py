"""
Idea Test Runner - Parallel execution with multi-model support.

Environment Variables:
- IDEA_TEST_MODE: "default" or "benchmark" (benchmark defaults: 3 models, top 8 tests, 3 runs, concurrency 3)
- IDEA_TEST_TOP_N: Number of top-priority tests to run (0 = all)
- IDEA_TEST_IDS: Comma-separated list of specific test IDs to run (e.g., "019,012,014,025") - overrides IDEA_TEST_TOP_N
- IDEA_TEST_RUNS: Repeats per test/model pair
- IDEA_TEST_CONCURRENCY: Max parallel task executions
- IDEA_TEST_MODELS: Comma-separated execution models (default: MODEL_NAME or gpt-5-mini)
- IDEA_TEST_EXECUTION_VARIANTS: Execution styles to run ("graph", "sequential", or comma-separated list)
- IDEA_TEST_LOG_LEVEL: Logging level (default: INFO)

Legacy aliases still supported:
- IDEA_TEST_PRIORITY -> IDEA_TEST_TOP_N
- IDEA_TEST_REPEATS -> IDEA_TEST_RUNS
- IDEA_TEST_MAX_PARALLEL -> IDEA_TEST_CONCURRENCY
- IDEA_TEST_BENCHMARK_MODE -> IDEA_TEST_MODE=benchmark
- IDEA_TEST_BENCHMARK_THIRD_COUNT -> IDEA_TEST_TOP_N

Validation always uses gpt-5-mini regardless of execution model.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import logging

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.connector_config import ConnectorConfig
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.runner import run_complete_test, discover_test_modules
from agent.app.testing.config import (
    MODEL_CANDIDATES,
    normalize_model_name,
    load_models_from_env,
    VALIDATION_MODEL,
    extract_test_id,
    filter_test_files_by_priority,
)
from agent.app.testing.utils import summarize_observability
from agent.app.idea_graph_analyzer import add_graph_visualization
from agent.app.testing.report import TestReportGenerator, Verbosity


async def preflight_check_llm(connector_llm: ConnectorLLM, model_name: str) -> bool:
    """
    Pre-flight check to verify LLM calls work for a model.
    :param connector_llm: LLM connector.
    :param model_name: Model name to test.
    :return: True if model works, False otherwise.
    """
    messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Reply with exactly OK"},
    ]
    payload_candidates = [
        {"max_completion_tokens": 256, "temperature": 1},
        {"max_tokens": 256, "temperature": 1},
        {"max_completion_tokens": 512},
        {"max_tokens": 512},
        {},
    ]
    original_model = connector_llm.get_model()
    last_error = "unknown"
    connector_llm.set_model(model_name)
    try:
        for payload in payload_candidates:
            try:
                response = await asyncio.wait_for(
                    connector_llm.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **payload,
                    ),
                    timeout=20,
                )
                choices = getattr(response, "choices", None) or []
                if not choices:
                    continue
                choice0 = choices[0]
                finish_reason = getattr(choice0, "finish_reason", None)
                content = getattr(getattr(choice0, "message", None), "content", None)
                profile = {
                    "temperature": payload.get("temperature") if "temperature" in payload else None,
                    "use_max_completion_tokens": "max_completion_tokens" in payload or "max_tokens" in payload,
                }
                connector_llm.set_model_profile(model_name, profile)
                content_len = len(content.strip()) if isinstance(content, str) else 0
                logging.info(
                    f"[PASSED] Pre-flight check for {model_name} "
                    f"(finish_reason={finish_reason}, content_len={content_len}, payload={payload})"
                )
                return True
            except Exception as exc:
                last_error = str(exc)
                continue
        logging.warning(f"[FAILED] Pre-flight check for {model_name}: {last_error}")
        return False
    except Exception as exc:
        logging.error(f"[FAILED] Pre-flight check for {model_name}: {exc}")
        return False
    finally:
        connector_llm.set_model(original_model)


TEST_PRIORITY_ORDER = [
    "026",  # Deterministic Page Facts (1/10) - Priority 1 (baseline easy)
    "002",  # Basic Fact Retrieval (1/10) - Priority 2 (baseline easy)
    "025",  # Wikipedia Link Chain Game (2/10) - Priority 3 (easy link-following)
    "019",  # Explicit Visit Requirement (3/10) - Priority 4
    "033",  # Dual-Niche Comparison (5/10) - Priority 5 (branching: 2-way)
    "034",  # Laundry-List Multi-Topic Extraction (6/10) - Priority 6 (branching: 6-way fan-out)
    "035",  # Cross-Domain Niche Synthesis (7/10) - Priority 7 (branching: 3-way diverge-converge)
    "036",  # Adversarial Compare-and-Contrast (8/10) - Priority 8 (branching: 4-way + essay)
    "037",  # Five-Topic Convergence Challenge (9/10) - Priority 9 (branching: 5-way convergence)
    "038",  # Eight-Source Fact Matrix (9/10) - Priority 10 (branching: 8-way parallel extraction)
    "039",  # Multi-Branch Claim Verification (10/10) - Priority 11 (4-branch search+visit+verdict)
    "014",  # Deep Link Exploration (5/10) - Priority 12
    "020",  # GitHub Repository Analysis (4/10) - Priority 11
    "009",  # Deep Research Synthesis (9/10) - Priority 12
    "012",  # Wikipedia Link Collection (3/10) - Priority 13
    "001",  # Conflicting Information (3/10) - Priority 14
    "021",  # News Article Extraction (3/10) - Priority 15
    "004",  # Technical Specification (4/10) - Priority 16
    "013",  # Wikipedia Exploration (4/10) - Priority 17
    "022",  # Technical Documentation (5/10) - Priority 18
    "005",  # Social Media Analysis (5/10) - Priority 19
    "006",  # Obscure Historical Event (6/10) - Priority 20
    "007",  # Multi-Domain Synthesis (7/10) - Priority 21
    "008",  # Complex Data Analysis (8/10) - Priority 22
    "010",  # Extreme Synthesis (10/10) - Priority 23
    "011",  # Advanced Reasoning (9/10) - Priority 24
    "015",  # Multi-Page Synthesis (6/10) - Priority 25
    "016",  # Topic Connection Exploration (6/10) - Priority 26
    "017",  # Recursive Link Analysis (7/10) - Priority 27
    "018",  # Cross-Domain Exploration (7/10) - Priority 28
    "023",  # Sequential Data Gathering (6/10) - Priority 29
    "024",  # Research Document Analysis (7/10) - Priority 30
]


def _is_enabled(value: str) -> bool:
    """
    Parse common truthy env values.
    :param value: Raw environment value.
    :return: True if enabled.
    """
    return value.strip().lower() in ("1", "true", "yes", "on")


def _unique_models(models: List[str]) -> List[str]:
    """
    Preserve order while removing duplicates/empties.
    :param models: Candidate model names.
    :return: Unique model names in original order.
    """
    unique: List[str] = []
    seen = set()
    for model_name in models:
        normalized = normalize_model_name(str(model_name or "").strip())
        if not normalized or normalized in seen:
            continue
        unique.append(normalized)
        seen.add(normalized)
    return unique


def _env_int(primary_key: str, fallback_keys: List[str], default_value: int) -> int:
    """
    Read integer from env using primary key with optional legacy fallbacks.
    :param primary_key: Preferred env var.
    :param fallback_keys: Legacy env vars in precedence order.
    :param default_value: Default when unset/invalid.
    :return: Parsed integer value.
    """
    raw = os.environ.get(primary_key, "").strip()
    if not raw:
        for key in fallback_keys:
            legacy = os.environ.get(key, "").strip()
            if legacy:
                raw = legacy
                break
    if raw:
        compact = raw.replace(",", "").replace("_", "").strip()
        signless = compact[1:] if compact.startswith(("+", "-")) else compact
        if signless.isdigit():
            return int(compact)
    return default_value


def _parse_execution_variants(raw: str) -> List[str]:
    """
    Parse execution variants from env.
    :param raw: Raw env string.
    :return: Ordered variant list.
    """
    text = (raw or "").strip().lower()
    if not text:
        return ["graph"]
    normalized = text.replace(";", ",").replace("\n", ",")
    parts = [item.strip() for item in normalized.split(",") if item.strip()]
    aliases = {
        "graph": "graph",
        "dag": "graph",
        "parallel": "graph",
        "sequential": "sequential",
        "chain": "sequential",
        "cot": "sequential",
    }
    out: List[str] = []
    seen = set()
    for part in parts:
        variant = aliases.get(part, "")
        if not variant or variant in seen:
            continue
        out.append(variant)
        seen.add(variant)
    return out or ["graph"]


def _variant_settings(base_settings: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """
    Build settings for execution variant.
    :param base_settings: Base settings.
    :param variant: Variant key.
    :return: Variant-specific settings.
    """
    settings = dict(base_settings)
    if variant == "sequential":
        settings["allow_execute_all_children"] = False
        settings["best_first_global"] = False
        settings["sequential_prune_siblings"] = True

        settings["max_total_nodes"] = min(int(settings.get("max_total_nodes", 500)), 80)
        settings["max_observation_chars"] = min(int(settings.get("max_observation_chars", 100000)), 40000)
        settings["max_links_per_visit"] = min(int(settings.get("max_links_per_visit", 20)), 5)

        settings["got_dynamic_beam_enabled"] = False

        system_prompt = str(settings.get("expansion_system_prompt", "")).strip()
        if system_prompt:
            settings["expansion_system_prompt"] = (
                system_prompt
                + "\n\nSEQUENTIAL MODE: Propose multiple candidates. Only the highest-scoring one will be selected."
            )
    return settings


def _difficulty_value(metadata: Dict[str, Any]) -> int:
    """
    Parse integer difficulty from metadata.
    :param metadata: Test metadata.
    :return: Difficulty in range 1..10, default 5.
    """
    raw = str(metadata.get("difficulty_level", "")).strip()
    if "/" in raw:
        raw = raw.split("/", 1)[0].strip()
    try:
        value = int(raw)
    except Exception:
        value = 5
    return max(1, min(10, value))


def _is_visit_focused(test_module: IdeaTestModule) -> bool:
    """
    Determine whether a test strongly exercises visit behavior.
    :param test_module: Loaded test module wrapper.
    :return: True when visit behavior is explicit.
    """
    metadata = test_module.metadata or {}
    category = str(metadata.get("category", "")).lower()
    statement = str(test_module.get_task_statement() or "").lower()
    criteria = " ".join(str(item).lower() for item in (test_module.get_success_criteria() or []))
    text = f"{category} {statement} {criteria}"
    return "visit" in text or "must visit" in text or "url" in text


def select_benchmark_test_files(test_files: List[Path], target_count: int = 8) -> List[Path]:
    """
    Select a balanced benchmark subset with strong priority and difficulty spread.
    :param test_files: Available test files.
    :param target_count: Target number of tests.
    :return: Selected tests in priority order.
    """
    if not test_files:
        return []
    test_id_to_file = {extract_test_id(f): f for f in test_files}
    ordered = [test_id_to_file[test_id] for test_id in TEST_PRIORITY_ORDER if test_id in test_id_to_file]
    for test_file in test_files:
        if test_file not in ordered:
            ordered.append(test_file)
    target = max(1, min(target_count, len(ordered)))
    modules: Dict[Path, IdeaTestModule] = {}
    for test_file in ordered:
        modules[test_file] = IdeaTestModule(test_file)
    visit_candidates = [f for f in ordered if _is_visit_focused(modules[f])]
    selected: List[Path] = []
    for candidate in visit_candidates[: min(2, target)]:
        if candidate not in selected:
            selected.append(candidate)
    bucket_targets: Dict[str, int]
    if target == 8:
        bucket_targets = {"low": 3, "mid": 3, "high": 2}
    else:
        low = max(1, round(target * 0.34))
        mid = max(1, round(target * 0.33))
        high = max(1, target - low - mid)
        bucket_targets = {"low": low, "mid": mid, "high": high}
    def bucket_name(value: int) -> str:
        if value <= 4:
            return "low"
        if value <= 7:
            return "mid"
        return "high"
    bucket_counts = {"low": 0, "mid": 0, "high": 0}
    for item in selected:
        b = bucket_name(_difficulty_value(modules[item].metadata))
        bucket_counts[b] += 1
    for candidate in ordered:
        if len(selected) >= target:
            break
        if candidate in selected:
            continue
        b = bucket_name(_difficulty_value(modules[candidate].metadata))
        if bucket_counts[b] < bucket_targets[b]:
            selected.append(candidate)
            bucket_counts[b] += 1
    for candidate in ordered:
        if len(selected) >= target:
            break
        if candidate not in selected:
            selected.append(candidate)
    selected_ids = [extract_test_id(f) for f in selected]
    logging.info(f"Benchmark mode: selected {len(selected)} tests: {selected_ids}")
    logging.info(
        "Benchmark mix: "
        f"low={bucket_counts['low']}, mid={bucket_counts['mid']}, high={bucket_counts['high']}, "
        f"visit_focused={sum(1 for f in selected if _is_visit_focused(modules[f]))}"
    )
    return selected


def extract_test_id(test_file: Path) -> str:
    """
    Extract test ID from test file name.
    :param test_file: Test file path.
    :return: Test ID string (e.g., "001", "020").
    """
    stem = test_file.stem
    if stem.startswith("test_"):
        remaining = stem.replace("test_", "", 1)
        parts = remaining.split("_", 1)
        return parts[0] if parts else ""
    return ""


def filter_test_files_by_priority(test_files: List[Path], priority_count: int = 0, specific_ids: Optional[List[str]] = None) -> List[Path]:
    """
    Filter test files by priority order or specific test IDs.
    :param test_files: List of test file paths.
    :param priority_count: Number of priority tests to run (0 = all, ignored if specific_ids provided).
    :param specific_ids: Optional list of specific test IDs to run (e.g., ["019", "012"]).
    :return: Filtered list of test files in priority order.
    """
    test_id_to_file = {extract_test_id(f): f for f in test_files}
    
    if specific_ids:
        filtered = []
        for test_id in specific_ids:
            test_id = test_id.strip()
            if test_id in test_id_to_file:
                filtered.append(test_id_to_file[test_id])
            else:
                logging.warning(f"Test ID {test_id} not found, skipping")
        logging.info(f"Specific test mode: Running {len(filtered)} specified tests: {[extract_test_id(f) for f in filtered]}")
        return filtered
    
    if priority_count == 0:
        ordered = []
        for test_id in TEST_PRIORITY_ORDER:
            if test_id in test_id_to_file:
                ordered.append(test_id_to_file[test_id])
        for test_file in test_files:
            test_id = extract_test_id(test_file)
            if test_id not in TEST_PRIORITY_ORDER:
                ordered.append(test_file)
        logging.info(f"Running all {len(ordered)} tests in priority order")
        return ordered
    
    filtered = []
    for test_id in TEST_PRIORITY_ORDER[:priority_count]:
        if test_id in test_id_to_file:
            filtered.append(test_id_to_file[test_id])
    
    logging.info(f"Priority mode: Running top {len(filtered)} priority tests: {[extract_test_id(f) for f in filtered]}")
    return filtered


async def run_single_test(
    test_file: Path,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    idea_settings: Dict[str, Any],
    run_id: str,
    results_dir: Path,
    validation_model: str,
    execution_variant: str = "graph",
    repeat_index: int = 1,
    total_repeats: int = 1,
) -> Dict[str, Any]:
    """
    Run a single test for a single model.
    :return: Test result dict or None if failed.
    """
    test_id = extract_test_id(test_file)
    normalized = normalize_model_name(model_name)
    
    try:
        test_module = IdeaTestModule(test_file)
        metadata = test_module.metadata
        
        variant_specific_settings = _variant_settings(idea_settings, execution_variant)
        result = await run_complete_test(
            test_module=test_module,
            model_name=normalized,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            idea_settings=variant_specific_settings,
            run_stamp=run_id,
            summarize_observability_func=summarize_observability,
            validation_model=validation_model,
        )
        
        result["execution_variant"] = execution_variant
        
        result = add_graph_visualization(result)
        
        out_path = results_dir / f"{run_id}_{test_id}_{normalized}_{execution_variant}_r{repeat_index}.json"
        
        report_verbosity = _env_int("IDEA_TEST_REPORT_VERBOSITY", [], 1)
        reporter = TestReportGenerator(verbosity=report_verbosity)
        telemetry_raw = result.get("execution", {}).get("telemetry_raw")
        report = reporter.generate(result, telemetry_data=telemetry_raw)
        reporter.print_console(report)
        if report_verbosity >= Verbosity.STANDARD:
            reporter.save(report, out_path)
        
        if report_verbosity < Verbosity.FULL:
            execution = result.get("execution", {})
            execution.pop("telemetry_raw", None)
        
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        
        validation = result.get("validation", {})
        passed = validation.get("overall_passed", False)
        score = validation.get("overall_score", 0.0)
        duration = result.get("execution", {}).get("duration_seconds", 0)
        observability = result.get("execution", {}).get("observability", {})
        
        logging.info(
            f"[{test_id}] {normalized} [{execution_variant}] [run {repeat_index}/{total_repeats}]: "
            f"{'PASSED' if passed else 'FAILED'} (score: {score:.2f}, {duration:.1f}s)"
        )
        
        return result
        
    except Exception as exc:
        logging.error(f"[{test_id}] {normalized} failed: {exc}", exc_info=True)
        return {
            "test_metadata": {"test_id": test_id},
            "model": normalized,
            "execution_variant": execution_variant,
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


def _print_run_summary(all_results: List[Dict[str, Any]], run_id: str, priority_count: int) -> None:
    """
    Print a consolidated summary table of all test results to the console.

    :param all_results: List of all test result dicts.
    :param run_id: Run identifier.
    :param priority_count: Number of priority tests (0=all).
    """
    sep = "=" * 80
    logging.info(f"\n{sep}")
    logging.info(f"  RUN SUMMARY: {run_id}")
    logging.info(sep)
    logging.info(f"  {'Test':>6}  {'Model':>14}  {'Variant':>10}  {'Pass':>5}  {'Score':>6}  {'Visits':>6}  {'Nodes':>6}  {'Time':>7}")
    logging.info(f"  {'------':>6}  {'--------------':>14}  {'----------':>10}  {'-----':>5}  {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}")

    total_passed = 0
    total_tests = 0
    total_score = 0.0

    for r in all_results:
        meta = r.get("test_metadata", {})
        test_id = meta.get("test_id", "???")
        model = r.get("model", "?")
        variant = r.get("execution_variant", "?")
        val = r.get("validation", {})
        passed = val.get("overall_passed", False)
        score = val.get("overall_score", 0.0)
        exec_data = r.get("execution", {})
        obs = exec_data.get("observability", {})
        visit_count = obs.get("visit", {}).get("count", 0)
        graph = exec_data.get("graph", {})
        node_count = len(graph.get("nodes", {}))
        duration = exec_data.get("duration_seconds", 0)

        icon = "  OK" if passed else "FAIL"
        logging.info(
            f"  {test_id:>6}  {model:>14}  {variant:>10}  {icon:>5}  {score:>6.2f}  {visit_count:>6}  {node_count:>6}  {duration:>6.1f}s"
        )
        total_tests += 1
        total_score += score
        if passed:
            total_passed += 1

    avg_score = total_score / total_tests if total_tests > 0 else 0.0
    logging.info(sep)
    logging.info(f"  Passed: {total_passed}/{total_tests}  Avg Score: {avg_score:.2f}")
    logging.info(f"  Priority mode: {priority_count} tests (0 = all)")

    from collections import defaultdict
    mv_scores: Dict[str, list] = defaultdict(list)
    mv_passed: Dict[str, int] = defaultdict(int)
    mv_count: Dict[str, int] = defaultdict(int)
    for r in all_results:
        model = r.get("model", "?")
        variant = r.get("execution_variant", "?")
        key = f"{model} [{variant}]"
        val = r.get("validation", {})
        score = val.get("overall_score", 0.0)
        mv_scores[key].append(score)
        mv_count[key] += 1
        if val.get("overall_passed", False):
            mv_passed[key] += 1

    if mv_count:
        logging.info("")
        logging.info("  Per Model+Variant Breakdown:")
        logging.info(f"  {'System':>30}  {'Runs':>4}  {'Pass%':>6}  {'Avg':>6}  {'Med':>6}  {'Min':>6}  {'Max':>6}")
        logging.info(f"  {'-'*30:>30}  {'----':>4}  {'-----':>6}  {'-----':>6}  {'-----':>6}  {'-----':>6}  {'-----':>6}")
        ranked = sorted(mv_count.keys(), key=lambda k: sum(mv_scores[k]) / len(mv_scores[k]), reverse=True)
        for key in ranked:
            n = mv_count[key]
            sc = mv_scores[key]
            import numpy as np
            avg = np.mean(sc)
            med = np.median(sc)
            mn = np.min(sc)
            mx = np.max(sc)
            pr = mv_passed[key] / n if n > 0 else 0
            logging.info(f"  {key:>30}  {n:>4}  {pr:>5.1%}  {avg:>6.3f}  {med:>6.3f}  {mn:>6.3f}  {mx:>6.3f}")

        graph_keys = [k for k in ranked if "[graph]" in k]
        seq_keys = [k for k in ranked if "[sequential]" in k]
        if graph_keys:
            best_g = graph_keys[0]
            g_avg = sum(mv_scores[best_g]) / len(mv_scores[best_g])
            logging.info(f"\n  >>> BEST GRAPH: {best_g} (avg {g_avg:.3f})")
        if seq_keys:
            best_s = seq_keys[0]
            s_avg = sum(mv_scores[best_s]) / len(mv_scores[best_s])
            logging.info(f"  >>> BEST SEQUENTIAL: {best_s} (avg {s_avg:.3f})")
        if graph_keys and seq_keys:
            g_avg = sum(mv_scores[graph_keys[0]]) / len(mv_scores[graph_keys[0]])
            s_avg = sum(mv_scores[seq_keys[0]]) / len(mv_scores[seq_keys[0]])
            winner = graph_keys[0] if g_avg >= s_avg else seq_keys[0]
            logging.info(f"  >>> OVERALL BEST: {winner}")

    aggregated_timings: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        obs = r.get("execution", {}).get("observability", {})
        for name, stats in obs.get("timings", {}).items():
            if name not in aggregated_timings:
                aggregated_timings[name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "min_duration": float("inf"),
                    "max_duration": 0.0,
                }
            agg = aggregated_timings[name]
            agg["count"] += stats.get("count", 0)
            agg["total_duration"] += stats.get("total_duration", 0)
            agg["min_duration"] = min(agg["min_duration"], stats.get("min_duration", float("inf")))
            agg["max_duration"] = max(agg["max_duration"], stats.get("max_duration", 0))
    for agg in aggregated_timings.values():
        if agg["min_duration"] == float("inf"):
            agg["min_duration"] = 0.0
        agg["avg_duration"] = round(agg["total_duration"] / agg["count"], 4) if agg["count"] > 0 else 0.0
        agg["total_duration"] = round(agg["total_duration"], 4)
        agg["min_duration"] = round(agg["min_duration"], 4)
        agg["max_duration"] = round(agg["max_duration"], 4)

    if aggregated_timings:
        from agent.app.testing.report import _render_timing_bar_chart
        logging.info("")
        logging.info("  Aggregate Connector Timings Across All Tests:")
        for name, stats in sorted(aggregated_timings.items()):
            logging.info(
                f"    {name:>16}: avg={stats['avg_duration']:.3f}s  "
                f"min={stats['min_duration']:.3f}s  max={stats['max_duration']:.3f}s  "
                f"count={stats['count']}  total={stats['total_duration']:.2f}s"
            )
        chart_lines = _render_timing_bar_chart(aggregated_timings, title="Avg Duration (all tests)")
        for line in chart_lines:
            logging.info(line)

    logging.info(sep)


async def main() -> None:
    """Run idea test suite with parallel execution."""
    mode_env = os.environ.get("IDEA_TEST_MODE", "").strip().lower()
    benchmark_mode = mode_env == "benchmark" or _is_enabled(os.environ.get("IDEA_TEST_BENCHMARK_MODE", ""))
    
    log_level = os.environ.get("IDEA_TEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    
    models = _unique_models(load_models_from_env())
    validation_model = normalize_model_name(
        os.environ.get("IDEA_TEST_VALIDATION_MODEL", "").strip() or VALIDATION_MODEL
    )
    if benchmark_mode:
        if len(models) < 3:
            models = _unique_models(MODEL_CANDIDATES[:3])
        if len(models) > 3:
            models = models[:3]
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    default_top_n = 8 if benchmark_mode else 0
    default_runs = 3 if benchmark_mode else 1
    default_concurrency = 3 if benchmark_mode else 4
    priority_count = _env_int("IDEA_TEST_TOP_N", ["IDEA_TEST_PRIORITY", "IDEA_TEST_BENCHMARK_THIRD_COUNT"], default_top_n)
    repeats = max(1, _env_int("IDEA_TEST_RUNS", ["IDEA_TEST_REPEATS"], default_runs))
    max_parallel = max(1, _env_int("IDEA_TEST_CONCURRENCY", ["IDEA_TEST_MAX_PARALLEL"], default_concurrency))
    specific_test_ids = None
    test_ids_env = os.environ.get("IDEA_TEST_IDS", "").strip()
    if test_ids_env:
        specific_test_ids = [tid.strip() for tid in test_ids_env.split(",") if tid.strip()]
        logging.info(f"Filtering to specific test IDs: {specific_test_ids}")
    
    execution_variants_env = os.environ.get("IDEA_TEST_EXECUTION_VARIANTS", "graph")
    execution_variants = _parse_execution_variants(execution_variants_env)
    
    if benchmark_mode:
        max_parallel = min(max_parallel, 3)
    
    all_test_files = discover_test_modules()
    if benchmark_mode:
        test_files = select_benchmark_test_files(all_test_files, target_count=priority_count)
    else:
        test_files = filter_test_files_by_priority(all_test_files, priority_count=priority_count, specific_ids=specific_test_ids)
    
    if not test_files:
        logging.error("No test files found after filtering")
        return
    
    logging.info(f"Running {len(test_files)} test(s) (total available: {len(all_test_files)})")
    logging.info(f"Mode: {'benchmark' if benchmark_mode else 'default'}")
    logging.info(f"Models: {', '.join(models)}")
    logging.info(f"Validation model: {validation_model}")
    logging.info(f"Execution variants: {', '.join(execution_variants)} ({len(execution_variants)} variant(s))")
    if specific_test_ids:
        logging.info(f"Test ID filter active: {specific_test_ids}")
    logging.info(f"Top N tests: {priority_count} (0 means all)")
    logging.info(f"Max parallel: {max_parallel}")
    logging.info(f"Repeats: {repeats}")
    if benchmark_mode:
        logging.info("Benchmark mode enabled: balanced difficulty subset with explicit visit coverage")
    
    config = ConnectorConfig()
    idea_settings = load_idea_dag_settings()
    idea_settings["log_dag_ascii"] = False
    idea_settings["log_dag_step_interval"] = 0
    idea_settings["allowed_actions"] = ["search", "visit", "save", "think", "merge"]
    if benchmark_mode:
        visit_prompt_suffix = (
            " Benchmark rule: when external evidence is needed, run search then visit the best URLs. "
            "Visit pages to extract concrete details and cite visited URLs in the final answer."
        )
        os.environ["IDEA_TEST_MANDATE_SUFFIX"] = visit_prompt_suffix
    
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)
    
    async with connector_search, connector_http, connector_llm:
        await connector_search.init_search_api()
        await connector_chroma.init_chroma()
        
        logging.info("Warming up ChromaDB to pre-install embedding models...")
        try:
            warmup_collection_name = "warmup_test"
            warmup_collection = await connector_chroma.get_or_create_collection(warmup_collection_name)
            if warmup_collection:
                await connector_chroma.add_to_chroma(
                    collection=warmup_collection_name,
                    ids=["warmup_1"],
                    metadatas=[{"type": "warmup"}],
                    documents=["ChromaDB warmup document to trigger model installation"],
                )
                await connector_chroma.query_chroma(
                    collection=warmup_collection_name,
                    query_texts=["warmup query"],
                    n_results=1,
                )
                logging.info("ChromaDB warmup completed - embedding models ready")
            else:
                logging.warning("ChromaDB warmup collection creation failed, but continuing")
        except Exception as warmup_exc:
            logging.warning(f"ChromaDB warmup failed (non-fatal): {warmup_exc}")
        
        execution_models_requested = _unique_models(models)
        preflight_models = list(execution_models_requested)
        if validation_model not in preflight_models:
            preflight_models.append(validation_model)
        logging.info(f"Pre-flight target models: {', '.join(preflight_models)}")
        preflight_passed: Dict[str, bool] = {}
        for model_name in preflight_models:
            ok = await preflight_check_llm(connector_llm, model_name)
            preflight_passed[model_name] = ok
            if not ok:
                logging.warning(f"Skipping {model_name} - pre-flight check failed")
        execution_models = [model for model in execution_models_requested if preflight_passed.get(model, False)]
        if not execution_models:
            logging.error("No valid execution models after pre-flight checks. Aborting.")
            return
        if not preflight_passed.get(validation_model, False):
            logging.error(f"Validation model {validation_model} failed pre-flight check. Aborting.")
            return
        valid_models = _unique_models(execution_models + [validation_model])
        logging.info(f"Valid models: {', '.join(valid_models)}")
        
        all_results = []
        results_dir = Path(__file__).resolve().parent.parent / "idea_test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        test_tasks: List[Tuple[Path, str, str, int]] = []
        for test_file in test_files:
            for model_name in execution_models:
                for execution_variant in execution_variants:
                    for repeat_index in range(1, repeats + 1):
                        test_tasks.append((test_file, model_name, execution_variant, repeat_index))
        
        logging.info(f"Total test tasks: {len(test_tasks)}")
        
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def run_with_semaphore(task):
            test_file, model_name, execution_variant, repeat_index = task
            queue_wait_start = time.perf_counter()
            async with semaphore:
                queue_wait_end = time.perf_counter()
                queue_wait_seconds = max(0.0, queue_wait_end - queue_wait_start)
                result = await run_single_test(
                    test_file=test_file,
                    model_name=model_name,
                    connector_llm=connector_llm,
                    connector_search=connector_search,
                    connector_http=connector_http,
                    connector_chroma=connector_chroma,
                    idea_settings=idea_settings,
                    run_id=run_id,
                    results_dir=results_dir,
                    validation_model=validation_model,
                    execution_variant=execution_variant,
                    repeat_index=repeat_index,
                    total_repeats=repeats,
                )
                if result and isinstance(result, dict):
                    execution_data = result.get("execution", {})
                    if isinstance(execution_data, dict):
                        execution_data["queue_wait_seconds"] = round(queue_wait_seconds, 2)
                return result
        
        logging.info("Starting parallel test execution...")
        results = await asyncio.gather(*[run_with_semaphore(task) for task in test_tasks], return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Test task failed with exception: {result}")
                continue
            if result:
                all_results.append(result)
        
        summary_path = results_dir / f"{run_id}_summary.json"
        summary = {
            "run_id": run_id,
            "benchmark_mode": benchmark_mode,
            "models": valid_models,
            "validation_model": validation_model,
            "execution_variants": execution_variants,
            "tests_run": len(test_files),
            "total_tests_available": len(all_test_files),
            "priority_count": priority_count,
            "repeats": repeats,
            "max_parallel": max_parallel,
            "test_ids": [extract_test_id(f) for f in test_files],
            "results": all_results,
            "timestamp": datetime.utcnow().isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        
        _print_run_summary(all_results, run_id, priority_count)


if __name__ == "__main__":
    asyncio.run(main())
