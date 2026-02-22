"""
Idea Test Runner - Parallel execution with multi-model support.

Environment Variables:
- IDEA_TEST_MODELS: Comma-separated models (default: MODEL_NAME or gpt-5-mini)
- IDEA_TEST_LOG_LEVEL: Logging level (default: INFO)
- IDEA_TEST_PRIORITY: Number of priority tests (0 = all, N = top N)
- IDEA_TEST_MAX_PARALLEL: Max parallel executions (default: 4)
- IDEA_TEST_VISIT_ONLY: Set to "1", "true", or "yes" to run only visit test

Validation always uses gpt-5-mini regardless of execution model.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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
    normalize_model_name,
    load_models_from_env,
    VALIDATION_MODEL,
    extract_test_id,
    filter_test_files_by_priority,
)
from agent.app.testing.utils import summarize_observability


async def preflight_check_llm(connector_llm: ConnectorLLM, model_name: str) -> bool:
    """
    Pre-flight check to verify LLM calls work for a model.
    :param connector_llm: LLM connector.
    :param model_name: Model name to test.
    :return: True if model works, False otherwise.
    """
    """
    Summarize observability metrics from telemetry.
    :param result: Test result payload.
    :param telemetry: Telemetry session.
    :return: Observability summary.
    """
    output = result.get("output", {})
    final_text = ""
    if isinstance(output, dict):
        final_deliverable = output.get("final_deliverable", "")
        if isinstance(final_deliverable, dict):
            final_text = json.dumps(final_deliverable, ensure_ascii=True)
        elif isinstance(final_deliverable, str):
            final_text = final_deliverable
        elif isinstance(final_deliverable, list):
            final_text = json.dumps(final_deliverable, ensure_ascii=True)
        else:
            final_text = str(final_deliverable)
    
    final_chars = count_chars(final_text)
    final_words = count_words(final_text)
    
    llm_prompt_chars = 0
    llm_prompt_words = 0
    llm_completion_chars = 0
    llm_completion_words = 0
    llm_prompt_tokens = 0
    llm_completion_tokens = 0
    llm_calls = 0
    
    for entry in telemetry.events:
        if entry.get("event") != "connector_io":
            continue
        payload = entry.get("payload") or {}
        if payload.get("connector") != "ConnectorLLM":
            continue
        io_payload = payload.get("payload") or {}
        llm_prompt_chars += int(io_payload.get("prompt_chars", 0))
        llm_prompt_words += int(io_payload.get("prompt_words", 0))
        llm_completion_chars += int(io_payload.get("completion_chars", 0))
        llm_completion_words += int(io_payload.get("completion_words", 0))
        llm_calls += 1
    
    for usage in telemetry.llm_usage:
        usage_payload = usage.get("usage") or {}
        llm_prompt_tokens += int(usage_payload.get("prompt_tokens", 0))
        llm_completion_tokens += int(usage_payload.get("completion_tokens", 0))
    
    chroma_store_chars = 0
    chroma_store_words = 0
    chroma_store_count = 0
    for entry in telemetry.chroma_stored:
        docs = entry.get("documents") or []
        chroma_store_count += len(docs)
        for doc in docs:
            chroma_store_chars += count_chars(doc)
            chroma_store_words += count_words(doc)
    
    chroma_retrieve_chars = 0
    chroma_retrieve_words = 0
    chroma_retrieve_count = 0
    for entry in telemetry.chroma_retrieved:
        docs = entry.get("documents") or []
        chroma_retrieve_count += len(docs)
        for doc in docs:
            chroma_retrieve_chars += count_chars(doc)
            chroma_retrieve_words += count_words(doc)
    
    search_count = 0
    search_chars = 0
    search_words = 0
    visit_count = 0
    visit_chars = 0
    visit_words = 0
    
    for entry in telemetry.documents_seen:
        source = entry.get("source")
        document = entry.get("document") or {}
        if source == "search":
            search_count += 1
            text = " ".join(
                str(value) for value in [document.get("title"), document.get("url"), document.get("description")] if value
            )
            search_chars += count_chars(text)
            search_words += count_words(text)
        elif source == "visit":
            visit_count += 1
            content = document.get("content") or ""
            visit_chars += count_chars(content)
            visit_words += count_words(content)
    
    timings_summary = {}
    for timing in telemetry.timings:
        name = timing.get("name", "unknown")
        if name not in timings_summary:
            timings_summary[name] = {
                "count": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "error_count": 0,
            }
        timings_summary[name]["count"] += 1
        timings_summary[name]["total_duration"] += timing.get("duration", 0.0)
        if timing.get("success"):
            timings_summary[name]["success_count"] += 1
        else:
            timings_summary[name]["error_count"] += 1
    
    return {
        "final_output": {
            "chars": final_chars,
            "words": final_words,
            "kilobytes": round(final_chars / 1024, 2),
        },
        "llm": {
            "calls": llm_calls,
            "prompt": {
                "chars": llm_prompt_chars,
                "words": llm_prompt_words,
                "kilobytes": round(llm_prompt_chars / 1024, 2),
                "tokens": llm_prompt_tokens,
            },
            "completion": {
                "chars": llm_completion_chars,
                "words": llm_completion_words,
                "kilobytes": round(llm_completion_chars / 1024, 2),
                "tokens": llm_completion_tokens,
            },
            "total_tokens": llm_prompt_tokens + llm_completion_tokens,
        },
        "chroma": {
            "store": {
                "count": chroma_store_count,
                "chars": chroma_store_chars,
                "words": chroma_store_words,
                "kilobytes": round(chroma_store_chars / 1024, 2),
            },
            "retrieve": {
                "count": chroma_retrieve_count,
                "chars": chroma_retrieve_chars,
                "words": chroma_retrieve_words,
                "kilobytes": round(chroma_retrieve_chars / 1024, 2),
            },
        },
        "search": {
            "count": search_count,
            "chars": search_chars,
            "words": search_words,
            "kilobytes": round(search_chars / 1024, 2),
        },
        "visit": {
            "count": visit_count,
            "chars": visit_chars,
            "words": visit_words,
            "kilobytes": round(visit_chars / 1024, 2),
        },
        "timings": timings_summary,
        "events_count": len(telemetry.events),
    }


TEST_PRIORITY_ORDER = [
    "001",  # Conflicting Information (3/10) - Priority 1
    "002",  # Basic Fact Retrieval (1/10) - Priority 2
    "003",  # Multi Query Search (2/10) - Priority 3
    "019",  # Explicit Visit Requirement (3/10) - Priority 4
    "021",  # News Article Extraction (3/10) - Priority 5
    "004",  # Technical Specification (4/10) - Priority 6
    "020",  # GitHub Repository Analysis (4/10) - Priority 7
    "012",  # Wikipedia Link Collection (3/10) - Priority 8
    "013",  # Wikipedia Exploration (4/10) - Priority 9
    "022",  # Technical Documentation (5/10) - Priority 10
    "005",  # Social Media Analysis (5/10) - Priority 11
    "006",  # Obscure Historical Event (6/10) - Priority 12
    "007",  # Multi-Domain Synthesis (7/10) - Priority 13
    "008",  # Complex Data Analysis (8/10) - Priority 14
    "009",  # Deep Research Synthesis (9/10) - Priority 15
    "010",  # Extreme Synthesis (10/10) - Priority 16
    "011",  # Advanced Reasoning (9/10) - Priority 17
    "014",  # Deep Link Exploration (5/10) - Priority 18
    "015",  # Multi-Page Synthesis (6/10) - Priority 19
    "016",  # Topic Connection Exploration (6/10) - Priority 20
    "017",  # Recursive Link Analysis (7/10) - Priority 21
    "018",  # Cross-Domain Exploration (7/10) - Priority 22
    "023",  # Sequential Data Gathering (6/10) - Priority 23
    "024",  # Research Document Analysis (7/10) - Priority 24
]


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


def filter_test_files_by_priority(test_files: List[Path], priority_count: int = 0) -> List[Path]:
    """
    Filter test files by priority order.
    :param test_files: List of test file paths.
    :param priority_count: Number of priority tests to run (0 = all).
    :return: Filtered list of test files in priority order.
    """
    test_id_to_file = {extract_test_id(f): f for f in test_files}
    
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


async def preflight_check_llm(connector_llm: ConnectorLLM, model_name: str) -> bool:
    """
    Pre-flight check to verify LLM calls work for a model.
    :param connector_llm: LLM connector.
    :param model_name: Model name to test.
    :return: True if model works, False otherwise.
    """
    try:
        original_model = connector_llm.get_model()
        connector_llm.set_model(model_name)
        try:
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Say 'OK'"}],
                "max_tokens": 5,
            }
            response = await connector_llm.query_llm(test_payload, timeout_seconds=10)
            if response and response.get("content"):
                logging.info(f"[PASSED] Pre-flight check for {model_name}")
                return True
            else:
                logging.warning(f"[FAILED] Pre-flight check for {model_name}: No response content")
                return False
        finally:
            connector_llm.set_model(original_model)
    except Exception as exc:
        logging.error(f"[FAILED] Pre-flight check for {model_name}: {exc}")
        return False


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
        
        result = await run_complete_test(
            test_module=test_module,
            model_name=normalized,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            idea_settings=idea_settings,
            run_stamp=run_id,
            summarize_observability_func=summarize_observability,
            validation_model=VALIDATION_MODEL,
        )
        
        out_path = results_dir / f"{run_id}_{test_id}_{normalized}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        
        validation = result.get("validation", {})
        passed = validation.get("overall_passed", False)
        score = validation.get("overall_score", 0.0)
        duration = result.get("execution", {}).get("duration_seconds", 0)
        observability = result.get("execution", {}).get("observability", {})
        
        logging.info(f"[{test_id}] {normalized}: {'PASSED' if passed else 'FAILED'} (score: {score:.2f}, {duration:.1f}s)")
        
        return result
        
    except Exception as exc:
        logging.error(f"[{test_id}] {normalized} failed: {exc}", exc_info=True)
        return {
            "test_metadata": {"test_id": test_id},
            "model": normalized,
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def main() -> None:
    """Run idea test suite with parallel execution."""
    visit_test_mode = os.environ.get("IDEA_TEST_VISIT_ONLY", "").strip().lower() in ("1", "true", "yes")
    
    if visit_test_mode:
        from agent.app.visit_test import main as visit_test_main
        success = await visit_test_main()
        exit(0 if success else 1)
        return
    
    log_level = os.environ.get("IDEA_TEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    
    models = load_models_from_env()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    priority_env = os.environ.get("IDEA_TEST_PRIORITY", "").strip()
    priority_count = int(priority_env) if priority_env and priority_env.isdigit() else 0
    
    max_parallel_env = os.environ.get("IDEA_TEST_MAX_PARALLEL", "4").strip()
    max_parallel = int(max_parallel_env) if max_parallel_env.isdigit() else 4
    
    all_test_files = discover_test_modules()
    test_files = filter_test_files_by_priority(all_test_files, priority_count=priority_count)
    
    if not test_files:
        logging.error("No test files found after filtering")
        return
    
    logging.info(f"Running {len(test_files)} test(s) (total available: {len(all_test_files)})")
    logging.info(f"Models: {', '.join(models)}")
    logging.info(f"Validation model: {VALIDATION_MODEL}")
    logging.info(f"Max parallel: {max_parallel}")
    
    config = ConnectorConfig()
    idea_settings = load_idea_dag_settings()
    idea_settings["log_dag_ascii"] = False
    idea_settings["log_dag_step_interval"] = 0
    idea_settings["allowed_actions"] = ["search", "visit", "save"]
    
    connector_llm = ConnectorLLM(config)
    connector_search = ConnectorSearch(config)
    connector_http = ConnectorHttp(config)
    connector_chroma = ConnectorChroma(config)
    
    async with connector_search, connector_http, connector_llm:
        await connector_search.init_search_api()
        await connector_chroma.init_chroma()
        
        logging.info("Running pre-flight checks for all models...")
        valid_models = []
        for model_name in models:
            normalized = normalize_model_name(model_name)
            if await preflight_check_llm(connector_llm, normalized):
                valid_models.append(normalized)
            else:
                logging.warning(f"Skipping {normalized} - pre-flight check failed")
        
        if not valid_models:
            logging.error("No valid models after pre-flight checks. Aborting.")
            return
        
        if VALIDATION_MODEL not in valid_models:
            logging.info(f"Adding validation model {VALIDATION_MODEL} to valid models")
            if await preflight_check_llm(connector_llm, VALIDATION_MODEL):
                if VALIDATION_MODEL not in valid_models:
                    valid_models.append(VALIDATION_MODEL)
            else:
                logging.error(f"Validation model {VALIDATION_MODEL} failed pre-flight check. Aborting.")
                return
        
        logging.info(f"Valid models: {', '.join(valid_models)}")
        
        all_results = []
        results_dir = Path(__file__).resolve().parent / "idea_test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        test_tasks = []
        for test_file in test_files:
            for model_name in valid_models:
                if model_name == VALIDATION_MODEL:
                    continue
                test_tasks.append((test_file, model_name))
        
        logging.info(f"Total test tasks: {len(test_tasks)}")
        
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def run_with_semaphore(task):
            test_file, model_name = task
            async with semaphore:
                return await run_single_test(
                    test_file=test_file,
                    model_name=model_name,
                    connector_llm=connector_llm,
                    connector_search=connector_search,
                    connector_http=connector_http,
                    connector_chroma=connector_chroma,
                    idea_settings=idea_settings,
                    run_id=run_id,
                    results_dir=results_dir,
                )
        
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
            "models": valid_models,
            "validation_model": VALIDATION_MODEL,
            "tests_run": len(test_files),
            "total_tests_available": len(all_test_files),
            "priority_count": priority_count,
            "max_parallel": max_parallel,
            "test_ids": [extract_test_id(f) for f in test_files],
            "results": all_results,
            "timestamp": datetime.utcnow().isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info(f"\n{'='*70}")
        logging.info(f"Summary saved: {summary_path.name}")
        logging.info(f"Priority mode: {priority_count} tests (0 = all)")
        logging.info(f"Total results: {len(all_results)}")
        logging.info(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
