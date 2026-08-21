"""
LangGraph off-the-shelf agent — the external comparison arm (DAG v2 relaunch item 4).

Mirrors ``execution_sequential.py``'s scaffolding (trace file, telemetry session, correlation id,
timing, cleanup) exactly, but delegates the actual decision loop to ``LangGraphSolver``
(``agent/app/langgraph_solver.py``) instead of this repo's own ReAct prompting/loop code — a
genuinely third-party orchestration strategy against the SAME search/visit tools every native arm
uses. Wired as the ``langgraph_react`` variant.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.langgraph_solver import LangGraphSolver
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing.execution import _empty_graph

_logger = logging.getLogger(__name__)


async def run_offtheshelf_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    summarize_observability_func=summarize_observability,
    connector_browser=None,
    idea_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the mandate through `LangGraphSolver`; same return shape as `run_sequential_execution`.

    :param connector_browser: Optional headless-Chrome fallback connector, wired uniformly with
        every other variant.
    :param idea_settings: The run's typed-settings dict. Unused by this arm today (LangGraph's
        own loop has no equivalent of this repo's retry/grounding/reasoning-effort levers) but
        accepted for signature parity with every other `run_*_execution`.
    """
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_langgraph_react_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}_langgraph_react.jsonl"
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id, trace_path=trace_path)

    max_steps = int(os.environ.get("IDEA_TEST_LANGGRAPH_MAX_STEPS", "25"))
    # Same gate the native arm uses for connector full capture (`execution.py`), so one env var
    # turns on raw prompt/completion capture for every arm in a run.
    report_verbosity = int(os.environ.get("IDEA_TEST_REPORT_VERBOSITY", "1"))
    solver = LangGraphSolver(
        connector_llm=connector_llm,
        connector_search=connector_search,
        connector_http=connector_http,
        connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        model_name=model_name,
        collection_name=f"idea_test_{test_id}_{run_stamp}",
        full_capture=report_verbosity >= 3,
        # Opt-in, default off: pass even a NATURAL termination through the solver's synthesis
        # pass (see `LangGraphSolver.__init__`). Awaiting a live A/B before it becomes default.
        always_synthesize=os.environ.get("IDEA_TEST_LANGGRAPH_ALWAYS_SYNTHESIZE", "") in ("1", "true", "True"),
    )

    started = time.perf_counter()
    solver_result: Dict[str, Any] = {"final_deliverable": "", "success": False, "observability": {}}
    try:
        solver_result = await solver.solve(mandate, max_steps=max_steps, settings=idea_settings, telemetry=telemetry)
    except Exception as exc:
        # The solver already swallows in-run failures (and still reports their token cost); this
        # only catches a construction-time failure, e.g. a missing API key.
        _logger.error(f"LangGraph solver failed: {exc}", exc_info=True)

    output = {
        "final_deliverable": solver_result.get("final_deliverable", "") or "",
        "success": bool(solver_result.get("success")),
        "goal_achieved": None,
        "action_summary": "langgraph_react",
    }
    warning = solver_result.get("warning")
    if warning:
        output["warning"] = warning
    telemetry.finish(success=output["success"])
    tracer.close()

    # Always recompute through the INJECTED summarizer (not the solver's own module-level one),
    # so this arm honors a caller-supplied observability function exactly like every other
    # `run_*_execution` does — the solver's copy is for standalone `Solver` consumers.
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
