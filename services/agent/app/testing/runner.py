"""
Test runner with parallel execution support.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import logging

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.execution import run_test_execution
from agent.app.testing.validation import ValidationRunner

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
    :return: Complete test result.
    """
    execution_result = await run_test_execution(
        test_module=test_module,
        model_name=model_name,
        connector_llm=connector_llm,
        connector_search=connector_search,
        connector_http=connector_http,
        connector_chroma=connector_chroma,
        idea_settings=idea_settings,
        run_stamp=run_stamp,
        summarize_observability_func=summarize_observability_func,
    )
    
    validation_runner = test_module.validation_runner
    validation_runner.validation_model = validation_model
    
    result = {"output": execution_result.get("output", {})}
    observability = execution_result.get("observability", {})
    
    validation_result = await validation_runner.run(
        result=result,
        observability=observability,
        connector_llm=connector_llm,
    )
    
    return {
        "test_metadata": test_module.metadata,
        "model": model_name,
        "validation_model": validation_model,
        "execution": execution_result,
        "validation": validation_result,
        "timestamp": datetime.utcnow().isoformat(),
    }
