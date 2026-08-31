"""
Sequential ReAct + per-hop typed extraction — the evidence-matched linear comparator.

``evidence_loop`` spends an EXTRA LLM call on every successful visit to read typed records off the
page. Comparing it against ``sequential_react`` therefore measures that extra compute as much as
it measures the ledger. This arm closes that gap: it is ``execution_sequential.py``'s loop, whose
helpers it imports rather than restates, plus exactly the same per-hop extraction call and the
same frozen pages — and nothing else. No ledger, no row minting, no verdict, no gating, no
table-first finalization; the deliverable is produced exactly the way the sequential control
produces it.

So ``sequential_react`` vs ``sequential_react_extract`` isolates the COST of extraction, and
``sequential_react_extract`` vs ``evidence_loop`` isolates the LEDGER with evidence-gathering held
equal. Wired as the ``sequential_react_extract`` variant.
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.agent_io import AgentIO
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.sandbox_tool_surface import PARITY_ACTIONS, run_sandbox_action
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder, build_trace_path, traces_retained
from agent.app.testing.execution import _empty_graph
from agent.app.testing.execution_evidence_loop import Extraction, Ledger, extract_from_page, store_page
from agent.app.testing.execution_sequential import (
    SequentialContextCap,
    ToolRetry,
    _EMPTY_PAGE,
    _build_history,
    _call_search_with_retry,
    _call_tool_with_retry,
    _fmt_search,
    _system_prompt,
    _verify_claim,
)
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing import json_telemetry as _json_telemetry

_logger = logging.getLogger(__name__)


@dataclass
class SequentialExtractResult:
    """What one run of :func:`_run_react_extract` produced.

    ``extractions`` and ``pages`` are the products of the added extraction call only — they are
    reported for auditing and never read back into the loop, which is what keeps this arm one
    single step away from the sequential control.
    """

    deliverable: str
    extractions: List[Extraction] = dataclass_field(default_factory=list)
    pages: List[Dict[str, Any]] = dataclass_field(default_factory=list)


async def _run_react_extract(agent_io: AgentIO, mandate: str, model_name: str, max_steps: int,
                             max_tokens: int, retry: Optional[ToolRetry] = None,
                             context_cap: Optional[SequentialContextCap] = None
                             ) -> SequentialExtractResult:
    """Run the sequential control's flat ReAct loop, extracting typed records after each visit.

    Every decision, observation, nudge and budget below is the control's, by construction: the
    prompt, history window, retry wrapper and search formatter are IMPORTED from
    ``execution_sequential``, and the env knobs read are the same ``IDEA_TEST_SEQ_*`` keys, so a
    benchmark cannot hand the two arms different tool budgets. The single addition is the
    :func:`extract_from_page` call on a visit that returned text, whose records are collected in a
    ROW-LESS ledger used purely as an append-only sink — ``Ledger.find`` matches nothing without
    rows, so no row state, verdict or gating can arise from it.

    :param agent_io: the run's IO facade (search / visit / query_llm / build_llm_payload).
    :param mandate: the task statement.
    :param model_name: executor model.
    :param max_steps: hard step budget for the flat loop.
    :param max_tokens: cap for the forced final synthesis.
    :param retry: the arm's bounded tool-retry policy; default off, as in the control.
    :param context_cap: optional per-turn context cap; default uncapped, as in the control.
    :returns: a :class:`SequentialExtractResult` with the deliverable, the extraction records in
        emission order, and every visited page frozen for offline re-audit.
    :raises: nothing from the tool surface — a failed search/visit becomes an observation, a
        malformed decision becomes an empty decision and a failed extraction yields no records.
    """
    retry = retry or ToolRetry()
    context_cap = context_cap or SequentialContextCap()
    page_chars = int(os.environ.get("IDEA_TEST_SEQ_PAGE_CHARS", "6000"))
    search_k = int(os.environ.get("IDEA_TEST_SEQ_SEARCH_K", "6"))
    dedup_search = os.environ.get("IDEA_TEST_SEQ_DEDUP_SEARCH", "1") not in ("0", "false", "False")
    step_max_tokens = int(os.environ.get("IDEA_TEST_SEQ_STEP_MAX_TOKENS", "4096"))
    store_chars = int(os.environ.get("IDEA_TEST_SEQ_EXTRACT_STORE_CHARS", str(page_chars)))
    sandbox = getattr(agent_io, "connector_sandbox", None)
    sink = Ledger(rows=[])              # append-only record sink; no rows -> no ledger semantics
    scratchpad: List[str] = []          # in-context working memory
    evidence: List[str] = []            # visited-page text, for verify and forced synthesis
    pages: List[Dict[str, Any]] = []    # frozen pages, for offline quote re-audit
    seen_queries: set = set()           # normalized queries already searched (breadth-loop guard)
    last_answer = ""

    for step in range(max_steps):
        history = _build_history(scratchpad, context_cap)
        messages = [
            {"role": "system", "content": _system_prompt(sandbox is not None)},
            {"role": "user", "content": f"TASK:\n{mandate}\n\nSCRATCHPAD (your prior steps):\n{history}\n\nReturn the next step as JSON."},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name, temperature=0.1, max_tokens=step_max_tokens)
        raw = await agent_io.query_llm(payload, model_name=model_name)
        try:
            decision = json.loads(raw or "{}")
            _parsed_ok = True
        except (json.JSONDecodeError, TypeError):
            decision = {}
            _parsed_ok = False
        _json_telemetry.record(model_name, raw, True, _parsed_ok, phase="sequential_react_extract")
        if isinstance(decision, list):
            decision = next((item for item in decision if isinstance(item, dict)), {})
        if not isinstance(decision, dict):
            decision = {}
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args")
        if not isinstance(args, dict):
            args = {}
        thought = str(decision.get("thought", ""))[:300]

        if action == "finish" or step == max_steps - 1:
            last_answer = str(args.get("answer", "")) or last_answer
            if last_answer:
                return SequentialExtractResult(last_answer, sink.extractions, pages)
            messages = [
                {"role": "system", "content": (
                    "Synthesize the FINAL answer using ONLY the gathered evidence. Address every part "
                    "the task asks for; for each fact quote the exact value from the page and cite the "
                    "source URL it came from. Do not add facts that are not in the evidence — if a "
                    "required fact is missing, say so explicitly rather than guessing."
                )},
                {"role": "user", "content": f"TASK:\n{mandate}\n\nEVIDENCE:\n{chr(10).join(evidence)[:12000] or '(none)'}"},
            ]
            payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens)
            answer = (await agent_io.query_llm(payload, model_name=model_name)) or ""
            return SequentialExtractResult(answer, sink.extractions, pages)

        if action == "search":
            query = str(args.get("query", ""))
            norm = re.sub(r"\s+", " ", query).strip().lower()
            if dedup_search and norm and norm in seen_queries:
                obs = (f"ALREADY SEARCHED '{query[:80]}'. Its results are in your scratchpad above — "
                       "VISIT one of those result URLs to read it, or FINISH if you have enough. "
                       "Do not repeat a search you have already run.")
            else:
                if norm:
                    seen_queries.add(norm)
                results, error, _used_query = await _call_search_with_retry(
                    lambda q: agent_io.search(q, count=search_k, timeout_seconds=20),
                    query, lambda r: not r, retry,
                )
                obs = f"SEARCH ERROR: {error}" if error is not None else _fmt_search(results or [], search_k)
        elif action == "visit":
            url = str(args.get("url", "")).strip()
            content, error = await _call_tool_with_retry(
                lambda: agent_io.visit(url, timeout_seconds=30),
                lambda c: not (c or "").strip() or (c or "").strip() == _EMPTY_PAGE, retry,
            )
            if error is not None:
                obs = f"VISIT ERROR for {url}: {error}"
            else:
                content = (content or "")[:page_chars]
                evidence.append(f"SOURCE {url}\n{content}")
                if content.strip():
                    page_id = f"p{len(pages) + 1}"
                    pages.append(store_page(page_id, url, content, store_chars))
                    await extract_from_page(agent_io, model_name, mandate, sink,
                                            page_id=page_id, page_url=url, page_text=content)
                obs = f"PAGE {url}:\n{content}"
        elif action == "verify":
            claim = str(args.get("claim", ""))
            verdict = await _verify_claim(agent_io, claim, "\n\n".join(evidence), model_name)
            obs = f"VERIFY '{claim[:80]}': {verdict}"
        elif action in PARITY_ACTIONS:
            obs = await run_sandbox_action(sandbox, action, args)
        else:
            available = "search/visit/verify/finish"
            if sandbox is not None:
                available += "/" + "/".join(PARITY_ACTIONS)
            obs = f"INVALID ACTION. Use {available}."

        scratchpad.append(f"STEP {step+1}: thought={thought}\naction={action} args={json.dumps(args)[:200]}\nobservation={obs[:context_cap.observation_chars()]}")

    return SequentialExtractResult(last_answer, sink.extractions, pages)


async def run_sequential_extract_execution(
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
    """Run the evidence-matched sequential arm; same result shape as every other variant.

    ``output`` carries the standard ``final_deliverable`` / ``success`` / ``action_summary`` keys
    plus the products of the extraction call (``extractions``, ``pages``) and this cell's BUDGET
    (``total_tokens``, ``duration_seconds``), so an arm comparison can never report "better"
    without also reporting what it spent. Deliberately absent: any ledger, row, verdict or quote
    tally — those are the mechanism under test in ``evidence_loop`` and must not leak into its
    control.

    :param test_module: the task under test.
    :param model_name: executor model.
    :param connector_llm: LLM connector (its model is set from ``model_name``).
    :param connector_search: search connector.
    :param connector_http: HTTP fetch connector.
    :param connector_chroma: vector-store connector.
    :param run_stamp: the run's timestamp, used in the correlation id and trace path.
    :param cell_tag: disambiguating suffix shared with the result JSON's filename.
    :param summarize_observability_func: observability summarizer.
    :param connector_browser: optional headless-Chrome fallback, wired like every other arm.
    :param idea_settings: the run's typed-settings dict; read for exactly the same tool-retry and
        context-cap keys the sequential control reads, so the two arms stay matched.
    :returns: the standard execution-result dict.
    :raises: nothing — a crashing loop is reported as an unsuccessful, well-formed result.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_sequential_react_extract_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = build_trace_path(results_dir, run_stamp, test_id, model_name,
                                  "sequential_react_extract", cell_tag)
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id,
                                 trace_path=trace_path)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    max_steps = int(os.environ.get("IDEA_TEST_SEQUENTIAL_MAX_STEPS", "25"))
    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    retry = ToolRetry.from_settings(idea_settings)
    context_cap = SequentialContextCap.from_settings(idea_settings)
    started = time.perf_counter()
    result: Optional[SequentialExtractResult] = None
    try:
        result = await _run_react_extract(agent_io, mandate, model_name, max_steps, max_tokens,
                                          retry=retry, context_cap=context_cap)
    except Exception as exc:  # noqa: BLE001. Same failure contract as the sibling variants.
        _logger.error(f"Sequential ReAct + extract failed: {exc}", exc_info=True)

    deliverable = result.deliverable if result else ""
    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": "sequential_react_extract",
        "extractions": [record.as_dict() for record in (result.extractions if result else [])],
        "pages": result.pages if result else [],
    }
    telemetry.finish(success=output["success"])
    tracer.close()

    observability = summarize_observability_func({"output": output}, telemetry, model_name)
    telemetry_summary = telemetry.summary()
    ended = time.perf_counter()
    duration_seconds = round(max(0.0, ended - started), 2)
    # Budget, restated on the cell itself: the analysis scripts read ``output`` first, and an arm
    # that buys its score with an extra call per visit must say so in the same place.
    output["total_tokens"] = int(((observability or {}).get("llm") or {}).get("total_tokens", 0))
    output["duration_seconds"] = duration_seconds

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
        "duration_seconds": duration_seconds,
        "telemetry": {
            "correlation_id": correlation_id,
            "trace_file": str(trace_path),
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
        "telemetry_raw": telemetry_summary,
    }
