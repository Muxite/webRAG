"""
Test execution engine.
"""

import logging
import os
import re
import time
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
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability

_logger = logging.getLogger(__name__)

_URL_RE = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')


def _empty_graph() -> Dict[str, Any]:
    """Return an empty graph shape so report/validation code never crashes on baselines."""
    return {"nodes": {}, "edges": []}


async def run_baseline_execution(
    test_module: IdeaTestModule,
    model_name: str,
    variant: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    summarize_observability_func=summarize_observability,
) -> Dict[str, Any]:
    """
    Run a no-graph baseline: ``parametric`` (single completion, no tools) or
    ``naive_rag`` (one fixed round of search+visit stuffed into one synthesis call).

    Returns the same shape as :func:`run_test_execution` so validation, cost
    instrumentation and reporting are identical across variants.

    :param test_module: Test module wrapper.
    :param model_name: Execution model name.
    :param variant: ``parametric`` or ``naive_rag``.
    :param connector_llm: LLM connector.
    :param connector_search: Search connector.
    :param connector_http: HTTP connector.
    :param connector_chroma: ChromaDB connector (unused; kept for signature parity).
    :param run_stamp: Run timestamp.
    :param summarize_observability_func: Observability summarizer.
    :return: Execution result with observability.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_{variant}_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}_{variant}.jsonl"
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    telemetry = TelemetrySession(
        enabled=True,
        mandate=mandate,
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

    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    deliverable = ""
    try:
        if variant == "naive_rag":
            deliverable = await _run_naive_rag(agent_io, mandate, model_name, max_tokens)
        else:
            deliverable = await _run_parametric(agent_io, mandate, model_name, max_tokens)
    except Exception as exc:
        _logger.error(f"Baseline ({variant}) failed: {exc}", exc_info=True)
        deliverable = ""

    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": f"baseline:{variant}",
    }
    telemetry.finish(success=output["success"])
    tracer.close()

    observability = summarize_observability_func({"output": output}, telemetry, model_name)
    telemetry_summary = telemetry.summary()
    ended = time.perf_counter()

    try:
        if trace_path.exists():
            trace_path.unlink()
    except Exception as exc:
        _logger.warning(f"Failed to delete trace file {trace_path}: {exc}")

    return {
        "output": output,
        "graph": _empty_graph(),
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


async def _run_parametric(agent_io: AgentIO, mandate: str, model_name: str, max_tokens: int) -> str:
    """Single completion with no tools — isolates raw model intelligence."""
    messages = [
        {"role": "system", "content": "You are a careful research assistant. Answer the task as completely and accurately as possible. If you cite sources, include their URLs."},
        {"role": "user", "content": mandate},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


async def _run_naive_rag(agent_io: AgentIO, mandate: str, model_name: str, max_tokens: int) -> str:
    """One fixed round of retrieval (no graph): search + visit -> single synthesis call."""
    search_k = int(os.environ.get("IDEA_TEST_NAIVE_RAG_SEARCH_K", "5"))
    visit_n = int(os.environ.get("IDEA_TEST_NAIVE_RAG_VISIT_N", "3"))
    per_page_chars = int(os.environ.get("IDEA_TEST_NAIVE_RAG_PAGE_CHARS", "8000"))
    total_chars = int(os.environ.get("IDEA_TEST_NAIVE_RAG_TOTAL_CHARS", "30000"))

    # URLs explicitly named in the mandate take priority (mirrors engine mandate enforcement).
    urls: List[str] = []
    for u in _URL_RE.findall(mandate):
        cleaned = u.rstrip('.,);]')
        if cleaned not in urls:
            urls.append(cleaned)

    if len(urls) < visit_n:
        try:
            results = await agent_io.search(mandate, count=search_k, timeout_seconds=20) or []
            for item in results:
                u = (item.get("url") or "").strip()
                if u and u not in urls:
                    urls.append(u)
                if len(urls) >= max(visit_n, search_k):
                    break
        except Exception as exc:
            _logger.warning(f"naive_rag search failed: {exc}")

    sources: List[str] = []
    budget = total_chars
    for u in urls[:visit_n]:
        if budget <= 0:
            break
        try:
            content = await agent_io.visit(u, timeout_seconds=25)
        except Exception as exc:
            _logger.warning(f"naive_rag visit failed for {u}: {exc}")
            continue
        snippet = (content or "")[:min(per_page_chars, budget)]
        budget -= len(snippet)
        sources.append(f"SOURCE URL: {u}\n{snippet}")

    context = "\n\n---\n\n".join(sources) if sources else "(no sources retrieved)"
    messages = [
        {"role": "system", "content": "You are a research assistant. Answer the task using ONLY the provided sources. Cite the source URLs you used. If the sources are insufficient, say so explicitly rather than guessing."},
        {"role": "user", "content": f"TASK:\n{mandate}\n\nSOURCES:\n{context}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


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
    # Effort tiers set settings["max_steps"]; fall back to env then default.
    max_steps = int(idea_settings.get("max_steps") or os.environ.get("IDEA_TEST_MAX_STEPS", "50"))
    
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
    
    observability = summarize_observability_func({"output": output}, telemetry, model_name)
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
