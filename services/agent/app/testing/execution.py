"""
Test execution engine.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

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
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability

_logger = logging.getLogger(__name__)


async def run_test_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    idea_settings: Dict[str, Any],
    run_stamp: str,
    summarize_observability_func=summarize_observability,
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
    
    report_verbosity = int(os.environ.get("IDEA_TEST_REPORT_VERBOSITY", "1"))
    if report_verbosity >= 3:
        connector_llm.set_full_capture(True)
        connector_search.set_full_capture(True)
        connector_http.set_full_capture(True)
        connector_chroma.set_full_capture(True)
    
    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
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
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"
    
    # Initialize memory manager and mandate context (mirrors engine.run())
    import hashlib
    namespace = f"idea_dag:{hashlib.sha256(mandate.encode('utf-8')).hexdigest()[:10]}"
    engine.settings["memo_namespace"] = namespace
    engine._current_mandate = mandate
    from agent.app.idea_memory import MemoryManager
    engine._memory_manager = MemoryManager(
        connector_chroma=connector_chroma,
        namespace=namespace,
    )
    
    graph = IdeaDag(root_title=mandate, root_details={"mandate": mandate, "memo_namespace": namespace})
    current_id = graph.root_id()
    
    started = time.perf_counter()
    max_steps = int(os.environ.get("IDEA_TEST_MAX_STEPS", "50"))
    
    for step_num in range(max_steps):
        try:
            result_id = await engine.step(graph, current_id, step_num)
            if result_id is None:
                _logger.warning(f"Step {step_num} returned None, stopping execution")
                # Emergency: if root has no children after step 0, expansion completely failed
                if step_num == 0:
                    root = graph.get_node(graph.root_id())
                    if root and not root.children:
                        _logger.error("ROOT EXPANSION FAILED on step 0 - no children created")
                break
            current_id = result_id
            node = graph.get_node(current_id)
            if node and node.status.value == "done" and current_id == graph.root_id():
                # Only break if the ROOT node is done (all work complete)
                break
        except Exception as exc:
            _logger.error(f"Step {step_num} failed: {exc}", exc_info=True)
            break
    
    final_node = graph.get_node(current_id)
    if final_node:
        output = await build_final_payload(
            io=engine.io,
            settings=idea_settings,
            graph=graph,
            mandate=mandate,
            model_name=model_name,
            memory_manager=engine._memory_manager,
        )
    else:
        output = {}
    
    telemetry.finish(success=output.get("success", False))
    tracer.close()
    
    if report_verbosity >= 3:
        connector_llm.set_full_capture(False)
        connector_search.set_full_capture(False)
        connector_http.set_full_capture(False)
        connector_chroma.set_full_capture(False)
    
    observability = summarize_observability_func({"output": output}, telemetry)
    telemetry_summary = telemetry.summary()
    
    ended = time.perf_counter()
    
    try:
        if trace_path.exists():
            trace_path.unlink()
    except Exception as exc:
        _logger.warning(f"Failed to delete trace file {trace_path}: {exc}")
    
    return {
        "output": output,
        "graph": graph.to_dict() if hasattr(graph, "to_dict") else None,
        "observability": observability,
        "duration_seconds": round(max(0.0, ended - started), 2),
        "telemetry": {
            "correlation_id": correlation_id,
            "trace_file": str(trace_path),
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
        "telemetry_raw": telemetry_summary,
    }
