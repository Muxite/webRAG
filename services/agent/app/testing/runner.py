"""
Test runner with parallel execution support.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import logging

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_browser import ConnectorBrowser
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.execution import run_test_execution, run_baseline_execution
from agent.app.testing.execution_sequential import run_sequential_execution
from agent.app.testing.execution_compiled import run_compiled_execution
from agent.app.testing.execution_compiled_code import run_compiled_code_execution
from agent.app.testing.validation import ValidationRunner
from agent.app.testing import json_telemetry as _json_telemetry

BASELINE_VARIANTS = ("parametric", "naive_rag", "minimal")
# Single-pass agent comparators that have their own runner (not the GoT engine).
LINEAR_AGENT_VARIANTS = ("sequential_react",)
# Cheap-model agents that execute an expensive-model-authored offline plan (no runtime planning).
COMPILED_AGENT_VARIANTS = ("graph_compiled",)
# Same, in the CODE domain: leaves act on a sandbox workdir instead of the web.
COMPILED_CODE_AGENT_VARIANTS = ("graph_compiled_code",)

_logger = logging.getLogger(__name__)

VALIDATION_MODEL = "gpt-5-mini"

_logger = logging.getLogger(__name__)


def discover_test_modules() -> List[Path]:
    """
    Discover all test Python files.
    :return: List of test file paths, sorted by test ID.
    """
    tests_dir = Path(__file__).resolve().parent.parent / "idea_tests"
    if not tests_dir.exists():
        return []
    return sorted(tests_dir.glob("test_*.py"))


async def run_complete_test(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    idea_settings: Dict[str, Any],
    run_stamp: str,
    summarize_observability_func,
    validation_model: str = VALIDATION_MODEL,
    execution_variant: str = "graph",
    connector_browser: Optional[ConnectorBrowser] = None,
) -> Dict[str, Any]:
    """
    Run complete test: execution + validation.
    :param test_module: Test module wrapper.
    :param model_name: Model name for execution.
    :param connector_llm: LLM connector.
    :param connector_search: Search connector.
    :param connector_http: HTTP connector.
    :param connector_chroma: ChromaDB connector.
    :param idea_settings: DAG settings.
    :param run_stamp: Run timestamp.
    :param summarize_observability_func: Function to summarize observability.
    :param validation_model: Model name for validation.
    :param execution_variant: graph / sequential_react / graph_compiled / graph_compiled_code
        (agents) or parametric / naive_rag / minimal (baseline).
    :param connector_browser: Optional headless-Chrome fallback connector. Passed to EVERY
        execution variant uniformly (F18) so no arm is structurally handicapped relative to
        another just because it happened to hit a bot-blocked site.
    :return: Complete test result.
    """
    _json_telemetry.set_task(test_module.metadata.get("test_id"))
    if execution_variant in LINEAR_AGENT_VARIANTS:
        execution_result = await run_sequential_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            run_stamp=run_stamp,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in COMPILED_AGENT_VARIANTS:
        execution_result = await run_compiled_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            run_stamp=run_stamp,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in COMPILED_CODE_AGENT_VARIANTS:
        execution_result = await run_compiled_code_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            run_stamp=run_stamp,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in BASELINE_VARIANTS:
        execution_result = await run_baseline_execution(
            test_module=test_module,
            model_name=model_name,
            variant=execution_variant,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            run_stamp=run_stamp,
            summarize_observability_func=summarize_observability_func,
        )
    else:
        execution_result = await run_test_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            summarize_observability_func=summarize_observability_func,
        )

    validation_runner = test_module.validation_runner
    validation_runner.validation_model = validation_model

    result = {
        "output": execution_result.get("output", {}),
        "graph": execution_result.get("graph", {}),
    }
    observability = execution_result.get("observability", {})

    validation_result = await validation_runner.run(
        result=result,
        observability=observability,
        connector_llm=connector_llm,
    )

    # F17: surface an infra-failure quarantine flag at the top of the result so downstream
    # scoring/analysis can exclude a cell poisoned by a 402/422/429/5xx/transport failure
    # instead of silently counting it as a genuine 0 (see testing/utils.summarize_observability
    # for the classification). The score itself is left untouched — this only tags it.
    infra_block = observability.get("infra") if isinstance(observability, dict) else None
    infra_failed = bool(infra_block.get("failed")) if isinstance(infra_block, dict) else False

    return {
        "test_metadata": test_module.metadata,
        "model": model_name,
        "validation_model": validation_model,
        "execution": execution_result,
        "validation": validation_result,
        "infra_failed": infra_failed,
        "timestamp": datetime.utcnow().isoformat(),
    }
