"""
Abstraction layer for idea test execution and validation.
"""

import asyncio
import importlib.util
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_dag import IdeaDag
from agent.app.idea_finalize import build_final_payload
from agent.app.agent_io import AgentIO
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder
from agent.app.testing.validation import ValidationRunner, FunctionValidationCheck, LLMValidationCheck

_logger = logging.getLogger(__name__)


class IdeaTestModule:
    """
    Wrapper for test module with validation functions.
    """
    def __init__(self, module_path: Path):
        """
        Load test module from file.
        :param module_path: Path to test Python file.
        """
        self.path = module_path
        self.module = None
        self.metadata = {}
        self.validation_runner = ValidationRunner()
        self._load_module()
    
    def _load_module(self):
        """Load test module dynamically."""
        spec = importlib.util.spec_from_file_location("test_module", self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load test module: {self.path}")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.metadata = self.module.get_test_metadata()
        
        validation_functions = self.module.get_validation_functions()
        for func in validation_functions:
            self.validation_runner.add_function_check(func)
        
        llm_func = self.module.get_llm_validation_function()
        if llm_func:
            self.validation_runner.add_llm_check(llm_func)
    
    def get_task_statement(self) -> str:
        """Get task statement from module."""
        return self.module.get_task_statement()
    
    def get_required_deliverables(self) -> List[str]:
        """Get required deliverables from module."""
        return self.module.get_required_deliverables()
    
    def get_success_criteria(self) -> List[str]:
        """Get success criteria from module."""
        return self.module.get_success_criteria()


async def run_test_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    idea_settings: Dict[str, Any],
    run_stamp: str,
    summarize_observability_func,
) -> Dict[str, Any]:
    """
    Execute agent for a test.
    :param test_module: Test module wrapper.
    :param model_name: Model name.
    :param connector_llm: LLM connector.
    :param connector_search: Search connector.
    :param connector_http: HTTP connector.
    :param connector_chroma: ChromaDB connector.
    :param idea_settings: DAG settings.
    :param run_stamp: Run timestamp.
    :param summarize_observability_func: Function to summarize observability.
    :return: Execution result with observability.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_{run_stamp}"
    
    results_dir = Path(__file__).resolve().parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}.jsonl"
    tracer = TraceRecorder(trace_path)
    
    telemetry = TelemetrySession(
        enabled=True,
        mandate=test_module.get_task_statement(),
        correlation_id=correlation_id,
        trace_path=trace_path,
    )
    
    agent_io = AgentIO(
        connector_llm=connector_llm,
        connector_search=connector_search,
        connector_http=connector_http,
        connector_chroma=connector_chroma,
        telemetry=telemetry,
        collection_name=f"idea_test_{test_id}_{run_stamp}",
    )
    
    engine = IdeaDagEngine(
        io=agent_io,
        settings=idea_settings,
        model_name=model_name,
    )
    
    mandate = test_module.get_task_statement()
    graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate})
    current_id = graph.root_id()
    
    started = time.time()
    max_steps = int(os.environ.get("IDEA_TEST_MAX_STEPS", "50"))
    
    for step_num in range(max_steps):
        try:
            result_id = await engine.step(graph, current_id, step_num)
            if result_id is None:
                break
            current_id = result_id
            node = graph.get_node(current_id)
            if node and node.status.value == "done":
                break
        except Exception as exc:
            _logger.error(f"Step {step_num} failed: {exc}")
            break
    
    ended = time.time()
    
    final_node = graph.get_node(current_id)
    if final_node:
        output = await build_final_payload(
            io=engine.io,
            settings=idea_settings,
            graph=graph,
            mandate=mandate,
            model_name=model_name,
        )
    else:
        output = {}
    
    telemetry.finish(success=output.get("success", False))
    tracer.close()
    
    observability = summarize_observability_func({"output": output}, telemetry)
    
    try:
        if trace_path.exists():
            trace_path.unlink()
    except Exception as exc:
        _logger.warning(f"Failed to delete trace file {trace_path}: {exc}")
    
    return {
        "output": output,
        "graph": graph.to_dict() if hasattr(graph, "to_dict") else None,
        "observability": observability,
        "duration_seconds": round(ended - started, 2),
        "telemetry": {
            "correlation_id": correlation_id,
            "trace_file": str(trace_path),
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
    }




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
    validation_model: str = "gpt-5-mini",
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
    :param validation_model: Model name for validation (default: gpt-5-mini).
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


def discover_test_modules() -> List[Path]:
    """
    Discover all test Python files.
    :return: List of test file paths, sorted by test ID.
    """
    tests_dir = Path(__file__).resolve().parent / "idea_tests"
    if not tests_dir.exists():
        return []
    return sorted(tests_dir.glob("test_*.py"))
