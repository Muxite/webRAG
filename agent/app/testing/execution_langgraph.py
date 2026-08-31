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
from typing import Dict, Any, List, Optional

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.langgraph_solver import LangGraphSolver
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import (
    TraceRecorder,
    build_trace_path,
    llm_io_capture_enabled,
    sanitize_path_component,
    traces_retained,
)
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing.execution import _empty_graph

_logger = logging.getLogger(__name__)


def pages_from_telemetry(telemetry: Any, max_chars: int) -> List[Dict[str, Any]]:
    """Freeze every VISITED page this run read into the ``store_page`` shape.

    Why this exists. Measured over the stored corpus, 36 of 40 sampled ``langgraph_react`` cells
    had ``observability.visit.count > 0`` and ZERO carried recoverable page text: the full body is
    recorded on ``telemetry.documents_seen`` (``source="visit"``, ``document={"url", "content"}``
    -- the CLEANED page text, not a preview), but ``testing/utils.slim_telemetry_raw`` strips
    ``documents_seen`` before the result JSON is written, this arm emitted no ``output["pages"]``,
    and it never populates ``result["graph"]``. The archive therefore recorded THAT this arm
    visited and never WHAT it read, which makes claim-grounding, an arm-symmetric verdict and a
    risk-coverage curve impossible for it rather than merely awkward.

    ``evidence_loop`` and ``sequential_react_extract`` already freeze their pages exactly this
    way, so this puts the off-the-shelf arm on the same contract and nothing downstream needs a
    special case: ``idea_test_utils.visited_evidence`` reads ``output["pages"]`` directly.

    Search hits are deliberately NOT frozen -- a result snippet is not a page the agent read.

    :param telemetry: the run's ``TelemetrySession`` (or None).
    :param max_chars: cap on the STORED window; the content hash still covers the whole text.
    :returns: ``store_page``-shaped dicts, one per distinct visited URL, in visit order.
    :raises: nothing.
    """
    from agent.app.testing.execution_evidence_loop import store_page

    pages: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in (getattr(telemetry, "documents_seen", None) or []):
        if not isinstance(entry, dict) or entry.get("source") != "visit":
            continue
        document = entry.get("document") or {}
        url = str(document.get("url") or "").strip()
        text = str(document.get("content") or "")
        if not url or not text.strip() or url in seen:
            continue
        seen.add(url)
        pages.append(store_page(f"p{len(pages) + 1}", url, text, max_chars))
    return pages


async def run_offtheshelf_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    cell_tag: str = "",
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
    trace_path = build_trace_path(results_dir, run_stamp, test_id, model_name, "langgraph_react", cell_tag)
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
    # Context-budget knobs, mirroring `execution_sequential`'s IDEA_TEST_SEQ_* pair. Without them
    # the solver's defaults were unreachable from a run's environment, so this arm could not join
    # a context-budget sweep that every other arm takes part in.
    search_k = int(os.environ.get("IDEA_TEST_LANGGRAPH_SEARCH_K", "6"))
    page_chars = int(os.environ.get("IDEA_TEST_LANGGRAPH_PAGE_CHARS", "6000"))
    solver = LangGraphSolver(
        connector_llm=connector_llm,
        connector_search=connector_search,
        connector_http=connector_http,
        connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        model_name=model_name,
        collection_name=f"idea_test_{test_id}_{run_stamp}",
        search_k=search_k,
        page_chars=page_chars,
        full_capture=llm_io_capture_enabled(report_verbosity),
        # Opt-in, default off: pass even a NATURAL termination through the solver's synthesis
        # pass (see `LangGraphSolver.__init__`). Awaiting a live A/B before it becomes default.
        always_synthesize=os.environ.get("IDEA_TEST_LANGGRAPH_ALWAYS_SYNTHESIZE", "") in ("1", "true", "True"),
        # DEFAULT ON as of 2026-08-23: refuse to finalize while a named candidate was never
        # visited (see `LangGraphSolver.__init__`). Live-confirmed twice (single-rep spot check +
        # a paired 2-rep A/B, n=12: +0.227 mean score, t=2.56, W/T/L 7/5/0, never lost a paired
        # cell) — see docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md. Opt OUT with
        # IDEA_TEST_LANGGRAPH_CANDIDATE_COVERAGE_GATE=0.
        candidate_coverage_gate=os.environ.get("IDEA_TEST_LANGGRAPH_CANDIDATE_COVERAGE_GATE", "1") not in ("0", "false", "False"),
        # DEFAULT ON as of 2026-08-23: bound what the model sees per turn (see
        # `LangGraphSolver.__init__` / `_trim_for_model`). Live-confirmed via a paired 2-rep A/B
        # (n=12, both conditions with candidate_coverage_gate=1): +0.216 mean score, t=2.23,
        # W/T/L 6/3/3 — see docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md. Opt OUT with
        # IDEA_TEST_LANGGRAPH_CONTEXT_TRIM=0.
        context_trim=os.environ.get("IDEA_TEST_LANGGRAPH_CONTEXT_TRIM", "1") not in ("0", "false", "False"),
        # Opt-in, default off: nudge toward a different approach after a run of non-progress tool
        # results (see `LangGraphSolver.__init__`). Awaiting a live A/B before it becomes default.
        stall_recovery_gate=os.environ.get("IDEA_TEST_LANGGRAPH_STALL_RECOVERY_GATE", "") in ("1", "true", "True"),
        # Opt-in, default off: imitate sequential_react's explicit finish(answer) action (see
        # `LangGraphSolver.__init__`). Awaiting a live A/B before it becomes default.
        require_finish_tool=os.environ.get("IDEA_TEST_LANGGRAPH_REQUIRE_FINISH_TOOL", "") in ("1", "true", "True"),
        # DEFAULT ON (arm fairness): a model with no tool-calling endpoint fails
        # `create_react_agent` in ~0.1s and scores a genuine 0.0, while the native engine runs it
        # fine over text/JSON — so this arm's roster was narrower than the comparison implied.
        # The path actually taken is reported as `output.tool_transport`. Opt OUT with
        # IDEA_TEST_LANGGRAPH_TOOL_EMULATION=0 to reproduce a pre-shim measurement.
        tool_call_emulation=os.environ.get("IDEA_TEST_LANGGRAPH_TOOL_EMULATION", "1") not in ("0", "false", "False"),
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
        # "native" | "emulated" | "unknown" (the solver never got far enough to route). Analysis
        # MUST stratify on this: an emulated cell is a different transport, not a different model.
        "tool_transport": solver_result.get("tool_transport") or "unknown",
        # The pages this arm actually read, frozen so a stored cell can be re-audited offline.
        # Without this the archive records THAT it visited and never WHAT it read -- see
        # `pages_from_telemetry`. Same shape and same purpose as `evidence_loop`'s `pages`.
        "pages": pages_from_telemetry(
            telemetry, int(os.environ.get("IDEA_TEST_PERSIST_PAGE_CHARS", "6000"))),
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

    if not traces_retained():
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
