"""
Sequential ReAct agent — the strong linear comparator for graph-vs-sequential.

A classic think -> act -> observe loop with the SAME toolset as the Graph-of-Thoughts
agent (search, visit, verify) and an in-context scratchpad as working memory — but NO
GoT planning/parallelism/beam. Holding the toolset fixed means a graph-vs-sequential gap
is attributable to the graph STRUCTURE, not to richer tools.

It reuses ``AgentIO`` (search/visit/query_llm/build_llm_payload) and returns the same
result shape as ``run_baseline_execution`` so cost instrumentation, validation and the
analysis scripts treat it identically. Wired as the ``sequential_react`` variant.
"""
import json
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
from agent.app.agent_io import AgentIO
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing.execution import _empty_graph

_logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a web-research agent solving a TASK with tools. Work ONE step at a time: "
    "think, then call exactly one tool. Tools:\n"
    "- search(query): web search; returns titles+URLs+snippets.\n"
    "- visit(url): read a page's full text. Use EXACT URLs from search results.\n"
    "- verify(claim): cross-check a claim against the pages you have already read.\n"
    "- finish(answer): output the final answer. Cite the source URLs you used.\n"
    "Strategy: break a multi-part task into its sub-facts and resolve EACH one by searching and "
    "then visiting its authoritative page (e.g. Wikipedia) before you finish — do not stop after "
    "the first fact, and never answer any part from memory. Read each value directly off the page "
    "and cite the exact source URL it came from.\n"
    "Each step, return ONLY JSON: {\"thought\": \"...\", \"action\": \"search|visit|verify|finish\", "
    "\"args\": {\"query|url|claim|answer\": \"...\"}}."
)


def _fmt_search(results: List[Dict[str, str]], k: int) -> str:
    lines = []
    for i, item in enumerate((results or [])[:k], 1):
        lines.append(f"{i}. {item.get('title','')} — {item.get('url','')}\n   {item.get('description','')}")
    return "SEARCH RESULTS:\n" + ("\n".join(lines) if lines else "(none)")


async def _verify_claim(agent_io: AgentIO, claim: str, evidence: str, model_name: str) -> str:
    messages = [
        {"role": "system", "content": "Judge whether the CLAIM is supported by the EVIDENCE (text from the pages already visited). Reply in one line: TRUE / PARTIALLY_TRUE / FALSE / UNVERIFIABLE, then the supporting-or-contradicting source URL and a brief reason."},
        {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence[:8000] or '(no evidence gathered yet)'}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.0, max_tokens=300)
    return (await agent_io.query_llm(payload, model_name=model_name)) or "UNVERIFIABLE"


async def _run_react(agent_io: AgentIO, mandate: str, model_name: str, max_steps: int, max_tokens: int) -> str:
    page_chars = int(os.environ.get("IDEA_TEST_SEQ_PAGE_CHARS", "6000"))
    search_k = int(os.environ.get("IDEA_TEST_SEQ_SEARCH_K", "6"))
    dedup_search = os.environ.get("IDEA_TEST_SEQ_DEDUP_SEARCH", "1") not in ("0", "false", "False")
    scratchpad: List[str] = []          # in-context working memory
    evidence: List[str] = []            # visited-page text, for verify
    seen_queries: set = set()           # normalized queries already searched (breadth-loop guard)
    last_answer = ""

    for step in range(max_steps):
        history = "\n\n".join(scratchpad[-12:]) if scratchpad else "(no actions yet)"
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"TASK:\n{mandate}\n\nSCRATCHPAD (your prior steps):\n{history}\n\nReturn the next step as JSON."},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name, temperature=0.1, max_tokens=1024)
        raw = await agent_io.query_llm(payload, model_name=model_name)
        try:
            decision = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            decision = {}
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args") or {}
        thought = str(decision.get("thought", ""))[:300]

        if action == "finish" or step == max_steps - 1:
            last_answer = str(args.get("answer", "")) or last_answer
            if last_answer:
                return last_answer
            # forced final synthesis if the model never produced an answer
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
            return (await agent_io.query_llm(payload, model_name=model_name)) or ""

        if action == "search":
            query = str(args.get("query", ""))
            norm = re.sub(r"\s+", " ", query).strip().lower()
            if dedup_search and norm and norm in seen_queries:
                # Breadth failure mode: re-running the same search instead of synthesizing (a 6-way
                # fan-out cell looped through the same entities, burning ~2.8x the calls). Don't spend
                # another search on a query already issued — nudge toward visit/finish. Distinct
                # per-entity queries are unaffected, so legitimate fan-out exploration is preserved.
                obs = (f"ALREADY SEARCHED '{query[:80]}'. Its results are in your scratchpad above — "
                       "VISIT one of those result URLs to read it, or FINISH if you have enough. "
                       "Do not repeat a search you have already run.")
            else:
                if norm:
                    seen_queries.add(norm)
                try:
                    results = await agent_io.search(query, count=search_k, timeout_seconds=20) or []
                    obs = _fmt_search(results, search_k)
                except Exception as exc:  # noqa: BLE001
                    obs = f"SEARCH ERROR: {exc}"
        elif action == "visit":
            url = str(args.get("url", "")).strip()
            try:
                content = (await agent_io.visit(url, timeout_seconds=30) or "")[:page_chars]
                evidence.append(f"SOURCE {url}\n{content}")
                obs = f"PAGE {url}:\n{content}"
            except Exception as exc:  # noqa: BLE001
                obs = f"VISIT ERROR for {url}: {exc}"
        elif action == "verify":
            claim = str(args.get("claim", ""))
            verdict = await _verify_claim(agent_io, claim, "\n\n".join(evidence), model_name)
            obs = f"VERIFY '{claim[:80]}': {verdict}"
        else:
            obs = "INVALID ACTION. Use search/visit/verify/finish."

        scratchpad.append(f"STEP {step+1}: thought={thought}\naction={action} args={json.dumps(args)[:200]}\nobservation={obs[:1500]}")

    return last_answer


async def run_sequential_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    summarize_observability_func=summarize_observability,
) -> Dict[str, Any]:
    """Run the sequential ReAct agent; same return shape as ``run_baseline_execution``."""
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_sequential_react_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}_sequential_react.jsonl"
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id, trace_path=trace_path)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    # Step budget parity with graph_compiled's effective ~24 (6 leaves x 4 steps): a strong linear
    # ReAct agent must be allowed enough turns to resolve every sub-fact of a multi-part task, or the
    # comparison hamstrings it. Kept a clean LINEAR agent (no k-sample voting / shared memory — those
    # are graph_compiled's distinctive features, not handicaps to "fix" here).
    max_steps = int(os.environ.get("IDEA_TEST_SEQUENTIAL_MAX_STEPS", "25"))
    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    deliverable = ""
    try:
        deliverable = await _run_react(agent_io, mandate, model_name, max_steps, max_tokens)
    except Exception as exc:
        _logger.error(f"Sequential ReAct failed: {exc}", exc_info=True)

    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": "sequential_react",
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
