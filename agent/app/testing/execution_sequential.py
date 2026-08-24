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
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, NamedTuple, Optional

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.agent_io import AgentIO
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder
from agent.app.idea_policies.action_constants import is_transient_tool_error
from agent.app.idea_policies.config import ActionConfig
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing.execution import _empty_graph
from agent.app.testing import json_telemetry as _json_telemetry

_logger = logging.getLogger(__name__)

#: Sentinel ``AgentIO.visit`` returns when a page fetched fine but yielded no extractable text.
#: The graph arm's VISIT action flips that same outcome into a retryable tool failure, so this
#: arm treats it as an empty payload too (see :func:`_call_tool_with_retry`).
_EMPTY_PAGE = "[No main content found]"

_SYSTEM = (
    "You are a web-research agent solving a TASK with tools. Work ONE step at a time: "
    "think, then call exactly one tool. Tools:\n"
    "- search(query): web search; returns titles+URLs+snippets. Keep the query SHORT — a few "
    "focused keywords (max ~400 characters / 50 words). Do NOT put whole sentences, your reasoning, "
    "or the full task text into the query; over-long queries are rejected by the search API.\n"
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


class ToolRetry(NamedTuple):
    """Bounded in-place retry policy for this arm's search/visit calls (F16, arm fairness).

    The graph arms already retry a TRANSIENT tool failure at the source
    (``IdeaEngine._maybe_retry_tool_failure``, gated by ``connector_retry_on_failure_enabled``),
    and the benchmark drivers turn that on for every arm (``IDEA_TEST_CONNECTOR_RETRY=1``). This
    linear arm had no equivalent, so one flaky search/visit became a permanent error observation
    for the reference model only — an infra confound in an arm comparison, not a model difference.
    Reading the SAME three settings keys keeps the two arms symmetric; with no settings dict (any
    non-benchmark caller) or the flag off, ``enabled`` is False and behavior is unchanged.
    """

    enabled: bool = False
    max_attempts: int = 0
    backoff_seconds: float = 0.0

    @classmethod
    def from_settings(cls, settings: Optional[Dict[str, Any]]) -> "ToolRetry":
        cfg = ActionConfig.from_settings(settings or {})
        return cls(
            enabled=bool(cfg.connector_retry_on_failure_enabled),
            max_attempts=max(0, int(cfg.connector_retry_max_attempts)),
            backoff_seconds=max(0.0, float(cfg.connector_retry_backoff_seconds)),
        )


async def _call_tool_with_retry(call, is_empty, retry: ToolRetry):
    """Run one tool call, retrying a TRANSIENT failure in place.

    Mirrors the graph engine's retry semantics exactly: retry when the call RAISED a transient
    error (timeout / 429 / 5xx — :func:`is_transient_tool_error`) or "succeeded" with an EMPTY
    payload (no search results / no page text), up to ``max_attempts`` extra attempts with a
    growing backoff. A permanent failure (401/403/404, bot-block) is never retried — re-running
    it only burns budget.

    :param call: Zero-arg coroutine factory for the tool call.
    :param is_empty: Predicate marking a successful-but-empty payload.
    :param retry: The arm's retry policy.
    :returns: ``(result, error)``; ``error`` is the last exception when every attempt raised.
    """
    attempt = 0
    while True:
        try:
            result, error = await call(), None
        except Exception as exc:  # noqa: BLE001
            result, error = None, exc
        transient = is_transient_tool_error(error) if error is not None else is_empty(result)
        if not (retry.enabled and transient and attempt < retry.max_attempts):
            return result, error
        attempt += 1
        _logger.info(f"[TOOL-RETRY] sequential tool failure; retrying "
                     f"(attempt {attempt}/{retry.max_attempts})")
        if retry.backoff_seconds > 0:
            await asyncio.sleep(retry.backoff_seconds * attempt)


#: A quoted phrase in a search query, e.g. `"Erie Canal"`.
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')


def _reformulate_multi_entity_query(query: Optional[str]) -> Optional[str]:
    """OR-join a query's quoted phrases when it bundles 2+ distinct entity names.

    Ported from ``idea_engine.py``'s identically-named function (kept as a local pure-function
    duplicate rather than a cross-module import — no graph dependency either way). Multiple
    quoted names AND together and rarely appear on one page; OR-joining asks for any of them,
    which a per-entity lookup needs. Returns ``None`` (no-op) when fewer than two phrases, or the
    query is already OR-joined.

    :param query: Query string that produced zero results.
    :return: OR-joined reformulation, or ``None`` if there's nothing to reformulate.
    """
    if not query or " OR " in query:
        return None
    phrases = _QUOTED_PHRASE_RE.findall(query)
    if len(phrases) < 2:
        return None
    remainder = _QUOTED_PHRASE_RE.sub("", query)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    or_joined = " OR ".join(f'"{p}"' for p in phrases)
    return f"{or_joined} {remainder}" if remainder else or_joined


async def _call_search_with_retry(search_fn, query: str, is_empty, retry: ToolRetry):
    """Like :func:`_call_tool_with_retry`, specialized for SEARCH: on a transient/empty retry,
    OR-joins a multi-quoted-entity query in place before resending (see
    :func:`_reformulate_multi_entity_query`) — mirrors ``idea_engine.py``'s
    ``_reformulate_search_query_if_multi_entity``. Resending an identical AND-shaped query that
    already returned nothing just fails again; this gives the retry budget a real second attempt
    instead of a wasted repeat. Only engages when ``retry.enabled`` (same gate as every other
    retry behavior — no new flag), and only reformulates when the query is actually multi-entity
    shaped; a normal single-entity query is retried unchanged, exactly as before.

    :param search_fn: ``async (query: str) -> results`` — called fresh each attempt so a
        reformulated query is actually used.
    :param query: The original query.
    :param is_empty: Predicate marking a successful-but-empty payload.
    :param retry: The arm's retry policy.
    :returns: ``(result, error, query_used)`` — the last query actually sent, so a caller can
        report what was searched.
    """
    attempt = 0
    current_query = query
    while True:
        try:
            result, error = await search_fn(current_query), None
        except Exception as exc:  # noqa: BLE001
            result, error = None, exc
        transient = is_transient_tool_error(error) if error is not None else is_empty(result)
        if not (retry.enabled and transient and attempt < retry.max_attempts):
            return result, error, current_query
        attempt += 1
        _logger.info(f"[TOOL-RETRY] search failure; retrying "
                     f"(attempt {attempt}/{retry.max_attempts})")
        if retry.backoff_seconds > 0:
            await asyncio.sleep(retry.backoff_seconds * attempt)
        reformulated = _reformulate_multi_entity_query(current_query)
        if reformulated:
            _logger.info(f"[TOOL-RETRY] reformulating multi-entity query: "
                         f"{current_query!r} -> {reformulated!r}")
            current_query = reformulated


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


async def _run_react(agent_io: AgentIO, mandate: str, model_name: str, max_steps: int,
                     max_tokens: int, retry: Optional[ToolRetry] = None) -> str:
    retry = retry or ToolRetry()  # default: retry OFF -> unchanged behavior
    page_chars = int(os.environ.get("IDEA_TEST_SEQ_PAGE_CHARS", "6000"))
    search_k = int(os.environ.get("IDEA_TEST_SEQ_SEARCH_K", "6"))
    dedup_search = os.environ.get("IDEA_TEST_SEQ_DEDUP_SEARCH", "1") not in ("0", "false", "False")
    # Per-step decision budget. Reasoning models (e.g. gemini-3.1-pro) emit reasoning
    # before the action JSON, so a small cap truncates the JSON mid-object -> garbage
    # decisions. Default to a reasoning-adequate floor (matches the 4096 preflight-JSON
    # convention); overridable via env so the runner can raise it per model.
    step_max_tokens = int(os.environ.get("IDEA_TEST_SEQ_STEP_MAX_TOKENS", "4096"))
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
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name, temperature=0.1, max_tokens=step_max_tokens)
        raw = await agent_io.query_llm(payload, model_name=model_name)
        try:
            decision = json.loads(raw or "{}")
            _parsed_ok = True
        except (json.JSONDecodeError, TypeError):
            decision = {}
            _parsed_ok = False
        _json_telemetry.record(model_name, raw, True, _parsed_ok, phase="sequential_react")
        # Models sometimes wrap the step in a list (e.g. ``[{...}]`` or a list of
        # actions) instead of a bare object; take the first dict and never let a
        # non-dict reach ``.get`` (would crash the whole react run).
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
                obs = f"PAGE {url}:\n{content}"
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
    connector_browser=None,
    idea_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the sequential ReAct agent; same return shape as ``run_baseline_execution``.

    :param connector_browser: Optional headless-Chrome fallback connector (F18). This is the
        strong comparator arm the web-connector audit found most penalized by an unwired
        browser fallback (its free-form queries/URLs hit bot-blocked sites more often), so
        wiring it here uniformly with the other variants removes an infra-driven handicap
        from the arm comparison rather than adding one.
    :param idea_settings: The run's typed-settings dict. This arm has no GoT engine, so it reads
        exactly the three ``connector_retry_*`` keys (F16) — the SAME flag the graph arms use for
        their in-place tool retry, so a benchmark can't hand one arm a retry the other never
        gets. Omitted/None (any non-benchmark caller) keeps the retry off.
    """
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
        connector_browser=connector_browser,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    # Step budget parity with graph_compiled's effective ~24 (6 leaves x 4 steps): a strong linear
    # ReAct agent must be allowed enough turns to resolve every sub-fact of a multi-part task, or the
    # comparison hamstrings it. Kept a clean LINEAR agent (no k-sample voting / shared memory — those
    # are graph_compiled's distinctive features, not handicaps to "fix" here).
    max_steps = int(os.environ.get("IDEA_TEST_SEQUENTIAL_MAX_STEPS", "25"))
    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    retry = ToolRetry.from_settings(idea_settings)
    started = time.perf_counter()
    deliverable = ""
    try:
        deliverable = await _run_react(agent_io, mandate, model_name, max_steps, max_tokens,
                                       retry=retry)
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
