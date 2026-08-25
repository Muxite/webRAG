"""Deterministic evidence-queue executor — a HARNESS STUB, not the DAG v3 architecture.

Phase 0's ``evidence_queue_deterministic`` arm
(docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md section 3). The plan's Phase B describes a typed,
reducer-owned queue whose jobs are minted deterministically and whose LLM is only a selector.
None of that is here, on purpose: this module exists so the harness plumbing for such a variant
(dispatch, telemetry, trace file, result JSON shape) is proven end-to-end BEFORE any of that
machinery is written, and so a later real implementation can replace the body of one function
instead of also debugging its wiring.

What it actually does, and nothing more:

1. enumerate the mandate's named requirements with the SAME function the coverage gate and the
   task ledger use (``candidate_coverage.extract_named_candidates``) -- no new parser, no LLM;
2. for each requirement, in mandate order: one SEARCH, one VISIT of the first result, one
   EXTRACT (a single bounded LLM read of that page against that requirement);
3. one final synthesis over the extracts.

Explicitly ABSENT (deliberate, see the plan's build order): typed ``Requirement``/``WorkItem``
records, illegal-transition enforcement, job eligibility/priority scheduling, constrained
selection, re-queueing, and any notion of contradiction. There is no scheduler at all -- the
"queue" is a for-loop over a list, which is the honest description of this arm.

It is fine, and expected, for this arm to score weakly. Judge it on whether it dispatches, runs
and reports like every other variant.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.agent_io import AgentIO
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.idea_policies.candidate_coverage import extract_named_candidates
from agent.app.telemetry import TelemetrySession
from agent.app.testing.execution import _empty_graph
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.trace_recorder import TraceRecorder, build_trace_path, traces_retained

_logger = logging.getLogger(__name__)

#: Requirements this stub will work through. A hard stop, not a scheduling decision: the loop has
#: no way to prioritize, so an over-long roster would just spend the budget in mandate order.
_MAX_REQUIREMENTS = 8
#: Page text handed to one EXTRACT call.
_PAGE_CHARS = 6000
#: Search results requested per requirement (only the first usable URL is ever visited).
_SEARCH_K = 5

_EXTRACT_SYSTEM = (
    "Extract ONLY what the page below states about the requirement. Quote exact values. If the "
    "page does not state it, reply exactly: NOT ON THIS PAGE. Never answer from memory."
)
_SYNTHESIS_SYSTEM = (
    "Write the FINAL answer using ONLY the extracts below. Address every part the task asks for; "
    "quote each value and cite the source URL it came from. If a required fact is missing from "
    "the extracts, say so explicitly rather than guessing."
)


def _requirements(mandate: str) -> List[str]:
    """The mandate's enumerated requirements, or a single whole-mandate requirement.

    Falls back to one requirement rather than to zero, so an un-enumerated mandate still produces
    a search/visit/extract cycle instead of an empty run that would look like an infra failure.
    """
    named = [name for name in extract_named_candidates(mandate or "") if name.strip()]
    if named:
        return named[:_MAX_REQUIREMENTS]
    return [(mandate or "").strip()[:200]] if (mandate or "").strip() else []


def _first_url(results: Any) -> str:
    if not isinstance(results, list):
        return ""
    for item in results:
        if isinstance(item, dict) and str(item.get("url", "")).strip():
            return str(item["url"]).strip()
    return ""


async def _run_evidence_queue(agent_io: AgentIO, mandate: str, model_name: str,
                              max_tokens: int) -> str:
    """One SEARCH + VISIT + EXTRACT per requirement, then one synthesis. No scheduler."""
    extracts: List[str] = []
    for requirement in _requirements(mandate):
        query = f"{requirement} {mandate}".strip()[:200] if len(requirement) < 12 else requirement
        try:
            results = await agent_io.search(query, count=_SEARCH_K, timeout_seconds=20)
        except Exception as exc:  # noqa: BLE001. One dead requirement must not end the run.
            _logger.warning(f"[EVIDENCE-QUEUE] search failed for {requirement!r}: {exc}")
            continue
        url = _first_url(results)
        if not url:
            continue
        try:
            content = await agent_io.visit(url, timeout_seconds=30)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"[EVIDENCE-QUEUE] visit failed for {url}: {exc}")
            continue
        content = (content or "")[:_PAGE_CHARS]
        if not content.strip():
            continue
        payload = agent_io.build_llm_payload(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": (
                    f"TASK:\n{mandate}\n\nREQUIREMENT: {requirement}\n\n"
                    f"SOURCE {url}\n{content}"
                )},
            ],
            json_mode=False, model_name=model_name, temperature=0.0, max_tokens=1000,
        )
        extracted = await agent_io.query_llm(payload, model_name=model_name)
        extracts.append(f"REQUIREMENT: {requirement}\nSOURCE {url}\n{extracted or '(nothing extracted)'}")

    payload = agent_io.build_llm_payload(
        messages=[
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": (
                f"TASK:\n{mandate}\n\nEXTRACTS:\n" + ("\n\n".join(extracts) or "(none)")
            )},
        ],
        json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens,
    )
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


async def run_evidence_queue_execution(
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
    """Run the deterministic evidence-queue stub; same return shape as every other variant.

    Signature mirrors :func:`testing.execution_sequential.run_sequential_execution` exactly, so
    ``runner.run_complete_test`` dispatches it with the same keyword set.

    :param idea_settings: Accepted for signature parity and currently UNREAD -- this stub has no
        knobs, and pretending otherwise would suggest it can be tuned when it cannot.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_evidence_queue_deterministic_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = build_trace_path(
        results_dir, run_stamp, test_id, model_name, "evidence_queue_deterministic", cell_tag
    )
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    telemetry = TelemetrySession(
        enabled=True, mandate=mandate, correlation_id=correlation_id, trace_path=trace_path
    )
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    deliverable = ""
    try:
        deliverable = await _run_evidence_queue(agent_io, mandate, model_name, max_tokens)
    except Exception as exc:  # noqa: BLE001. Same failure contract as the sibling variants.
        _logger.error(f"Evidence queue failed: {exc}", exc_info=True)

    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": "evidence_queue_deterministic",
    }
    telemetry.finish(success=output["success"])
    tracer.close()

    observability = summarize_observability_func({"output": output}, telemetry, model_name)
    telemetry_summary = telemetry.summary()
    ended = time.perf_counter()

    if not traces_retained():
        try:
            if trace_path.exists():
                trace_path.unlink()
        except Exception as exc:  # noqa: BLE001
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
