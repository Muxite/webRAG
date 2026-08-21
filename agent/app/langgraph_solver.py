"""``LangGraphSolver`` — the off-the-shelf, publicly-available agent-system comparison arm.

Implements the `Solver` Protocol (`agent/app/solver.py`), anticipated but never built by the
Phase 0 in-tree refactor's docstring: "The Phase 3 comparison harness adds `LangGraphSolver` and
`LangChainSolver` against the same contract." This runs `langgraph.prebuilt.create_react_agent` —
a genuinely third-party orchestration loop, not this repo's own ReAct prompting/decision logic
(unlike the `sequential_react` reference arm, which is this repo's own code) — against the SAME
search/visit tools every native arm uses, so a lift/gap is attributable to the orchestration
strategy, not to a richer or different toolset.

ARM-FAIRNESS INVARIANTS (each mirrors something the native arms already get; without them this
arm would look artificially bad and misrepresent the comparison):

* **Tool retry parity (F16).** The native arms retry a TRANSIENT search/visit failure in place
  (``connector_retry_*`` settings, which the benchmark drivers turn on for every arm). The tools
  here go through the SAME :func:`_call_tool_with_retry` helper with the same policy, so one
  flaky search doesn't become a permanent error observation for this arm only.
* **Forced final synthesis.** ``execution_sequential._run_react`` gives its agent a last-turn
  synthesis from gathered evidence when it runs out of steps. LangGraph instead raises
  ``GraphRecursionError`` and would otherwise return NOTHING — scoring a hard 0 on runs that had
  in fact gathered good evidence. We reproduce the native arm's graceful path. Step exhaustion
  ALSO arrives without an exception, as a canned ``_STEP_EXHAUSTED_TEXT`` apology substituted for
  the model's turn; that is routed through the same synthesis, since otherwise the apology itself
  is a perfectly truthy "answer" and every gathered page is discarded silently.
* **Token accounting survives a crash.** Messages are accumulated via ``astream`` so a run that
  dies mid-way still reports the tokens it really burned. ``ainvoke`` would raise and discard
  them, reporting $0 for a run that spent real money.
* **Infra-failure quarantine (F17).** LangGraph drives its own LLM calls (not ``ConnectorLLM``),
  so a 402/429/5xx storm would produce no infra-classified telemetry timing and the cell would be
  scored as a genuine 0 instead of quarantined. A provider-side failure is recorded explicitly.

Tools wrap ``AgentIO.search``/``AgentIO.visit`` bound to the caller's ``TelemetrySession``, so
``search.count``/``visit.count`` telemetry is recorded identically to every other arm for free.
Token usage is recovered from each response's ``usage_metadata`` and fed into the same
``telemetry.record_llm_usage`` every arm uses, so ``summarize_observability`` produces the
identical observability shape (cost, tokens, search/visit counts) with no bespoke synthesis.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from agent.app.agent_io import AgentIO
from agent.app.connector_llm import ConnectorLLM, is_infra_llm_failure
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.idea_test_utils import count_chars, count_words
from agent.app.solver import SolverResult
from agent.app.testing.execution_sequential import ToolRetry, _EMPTY_PAGE, _call_tool_with_retry
from agent.app.testing.utils import summarize_observability
from services.shared.connector_config import ConnectorConfig

if TYPE_CHECKING:
    from agent.app.telemetry import TelemetrySession

_logger = logging.getLogger(__name__)

#: Mirrors execution_sequential._SYSTEM's guidance, including the search-length limit the native
#: arms are told about (LADDER_PREREGISTRATION's fairness rule: no arm may be handicapped by a
#: 422 storm on over-long queries). Tool CONTRACT wording is deliberately left to LangGraph's own
#: function-calling schema — that is the third-party behavior under test.
_SYSTEM = (
    "You are a web-research agent solving a TASK with tools. Use the search and visit tools to "
    "gather evidence before answering — never answer from memory.\n"
    "Keep each search query SHORT — a few focused keywords (max ~400 characters / 50 words). Do "
    "NOT put whole sentences, your reasoning, or the full task text into the query; over-long "
    "queries are rejected by the search API.\n"
    "Strategy: break a multi-part task into its sub-facts and resolve EACH one by searching and "
    "then visiting its authoritative page (e.g. Wikipedia) before you finish — do not stop after "
    "the first fact. Read each value directly off the page and cite the exact source URL it came "
    "from in your final answer."
)

#: What ``create_react_agent`` puts in the response INSTEAD of raising when it runs out of steps
#: (``langgraph.prebuilt.chat_agent_executor``'s ``call_model``/``acall_model``, verified against
#: langgraph 1.2.9 at lines 689/716). Step exhaustion therefore reaches us as an ordinary, truthy,
#: tool-call-free assistant turn: no ``GraphRecursionError``, so without this constant the canned
#: apology becomes the final deliverable and the forced-synthesis invariant above never fires,
#: silently discarding every page the run had already gathered.
_STEP_EXHAUSTED_TEXT = "Sorry, need more steps to process this request."

#: What the search tool returns when the backend itself is unavailable (``AgentIO.search`` ->
#: ``query_search`` returns ``None`` without raising when its health probe fails, e.g. a 403 on the
#: search key). Distinct from ``"No results."``: an outage is not fixable by rephrasing, and a
#: model told "No results." burns its remaining turns re-querying a dead backend.
_SEARCH_UNAVAILABLE = (
    "SEARCH BACKEND UNAVAILABLE — the search service is down, not your query. Do NOT retry this "
    "or any other search; use the visit tool on a URL you can name, or answer from the evidence "
    "you already have. If you have gathered no evidence, do NOT answer from prior knowledge — "
    "state that the answer cannot be determined from available sources. Cite only URLs you "
    "actually visited with the visit tool in this conversation; never a remembered or invented one."
)

_SYNTHESIS_SYSTEM = (
    "Synthesize the FINAL answer using ONLY the gathered evidence. Address every part the task "
    "asks for; for each fact quote the exact value from the page and cite the source URL it came "
    "from. Do not add facts that are not in the evidence — if a required fact is missing, say so "
    "explicitly rather than guessing."
)


def _make_tools(agent_io: AgentIO, search_k: int, page_chars: int,
                retry: Optional[ToolRetry] = None):
    """Build the search/visit tools bound to ``agent_io``, with native-arm retry parity."""
    retry = retry or ToolRetry()  # default: retry OFF -> unchanged behavior

    @tool
    async def search(query: str) -> str:
        """Web search. Returns titles, URLs, and snippets for the query."""
        results, error = await _call_tool_with_retry(
            lambda: agent_io.search(query, count=search_k, timeout_seconds=20),
            lambda r: not r, retry,
        )
        if error is not None:
            return f"SEARCH ERROR: {error}"
        if results is None:
            return _SEARCH_UNAVAILABLE
        if not results:
            return "No results."
        lines = [
            f"{i}. {r.get('title', '')} — {r.get('url', '')}\n   {r.get('description', '')}"
            for i, r in enumerate(results[:search_k], 1)
        ]
        return "\n".join(lines)

    @tool
    async def visit(url: str) -> str:
        """Fetch a page's text content. Use an exact URL from search results."""
        content, error = await _call_tool_with_retry(
            lambda: agent_io.visit(url, timeout_seconds=30),
            lambda c: not (c or "").strip() or (c or "").strip() == _EMPTY_PAGE, retry,
        )
        if error is not None:
            return f"VISIT ERROR for {url}: {error}"
        return (content or _EMPTY_PAGE)[:page_chars]

    return [search, visit]


def _extract_usage(messages: List[Any]) -> List[Dict[str, Any]]:
    """Pull (prompt_tokens, completion_tokens) per AIMessage turn, in the shape
    ``TelemetrySession.record_llm_usage`` / ``summarize_observability`` already expect
    (mirrors ``ConnectorLLM.query_llm``'s own payload shape)."""
    usages = []
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            um = m.usage_metadata
            usages.append({
                "prompt_tokens": int(um.get("input_tokens", 0) or 0),
                "completion_tokens": int(um.get("output_tokens", 0) or 0),
            })
    return usages


def _msg_text(m: Any) -> str:
    content = getattr(m, "content", "")
    return content if isinstance(content, str) else str(content or "")


def _msg_tool_calls(m: Any) -> List[Dict[str, Any]]:
    """The message's tool calls as plain JSON-able ``{name, args}`` dicts (empty when there are
    none). An ``AIMessage`` that ONLY calls tools has empty ``content``, so a capture keyed on
    content alone records neither the queries searched nor the URLs visited — the run's entire
    activity has to be reconstructed from payload byte counts."""
    calls = getattr(m, "tool_calls", None) or []
    out: List[Dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            out.append({"name": call.get("name", ""), "args": call.get("args", {})})
        else:
            out.append({"name": str(getattr(call, "name", "") or ""),
                        "args": getattr(call, "args", {}) or {}})
    return out


def _msg_role(m: Any) -> str:
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, ToolMessage):
        return "tool"
    return str(getattr(m, "type", "") or "unknown")


def _record_io_parity(telemetry: "TelemetrySession", messages: List[Any],
                      full_capture: bool = False) -> None:
    """Emit the ``connector_io`` events ``observability["llm"]`` is actually derived from.

    ``summarize_observability`` counts ``llm.calls`` (and the prompt/completion char+word stats)
    from telemetry EVENTS whose ``connector == "ConnectorLLM"`` — not from ``llm_usage``. This arm
    drives its own LLM calls through LangGraph and never touches ``ConnectorLLM``, so without this
    it reports ``llm.calls = 0`` and zero prompt/completion chars while its token counts are
    non-zero: level_ladder would show the arm doing no LLM work at all.

    One "in" + one "out" event per assistant turn, matching ``ConnectorLLM.query_llm``'s own
    request/response pair (``connector_llm.py`` records ``direction="in"`` before the call and
    ``direction="out"`` after), so ``llm.calls`` lands on the SAME scale as every other arm.

    ``full_capture`` additionally records the raw ``prompt_text``/``messages``/``completion_text``
    under the SAME payload keys ``ConnectorLLM`` uses when its own full capture is on, so a
    captured run of this arm is diffable against a native-arm run with the same readers. Off by
    default: the counts-only payload is what every existing consumer expects.
    """
    prior: List[str] = []
    prior_msgs: List[Dict[str, Any]] = []
    for m in messages or []:
        text = _msg_text(m)
        tool_calls = _msg_tool_calls(m)
        if isinstance(m, AIMessage):
            prompt_blob = "\n".join(prior)
            in_payload: Dict[str, Any] = {"prompt_chars": count_chars(prompt_blob),
                                          "prompt_words": count_words(prompt_blob)}
            out_payload: Dict[str, Any] = {"completion_chars": count_chars(text),
                                           "completion_words": count_words(text)}
            if full_capture:
                in_payload["prompt_text"] = prompt_blob
                in_payload["messages"] = list(prior_msgs)
                out_payload["completion_text"] = text
                if tool_calls:
                    out_payload["completion_tool_calls"] = tool_calls
            telemetry.record_event("connector_io", {
                "connector": "ConnectorLLM", "direction": "in", "operation": "llm_query",
                "payload": in_payload,
            })
            telemetry.record_event("connector_io", {
                "connector": "ConnectorLLM", "direction": "out", "operation": "llm_query",
                "payload": out_payload,
            })
        prior.append(text)
        prior_entry: Dict[str, Any] = {"role": _msg_role(m), "content": text}
        if tool_calls:
            prior_entry["tool_calls"] = tool_calls
        prior_msgs.append(prior_entry)


def _final_answer(messages: List[Any]) -> str:
    """The last assistant turn that actually carries prose (not a bare tool call)."""
    for m in reversed(messages or []):
        if not isinstance(m, AIMessage) or getattr(m, "tool_calls", None):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if content.strip():
            return content
    return ""


def _evidence_from(messages: List[Any], limit: int = 12000) -> str:
    """Tool outputs gathered so far — the input to the forced final synthesis."""
    chunks = [
        m.content if isinstance(m.content, str) else str(m.content or "")
        for m in (messages or []) if isinstance(m, ToolMessage)
    ]
    return "\n\n".join(c for c in chunks if c.strip())[:limit]


class LangGraphSolver:
    """Wraps `langgraph.prebuilt.create_react_agent` in the `Solver` interface."""

    name = "langgraph_react"

    def __init__(
        self,
        connector_llm: ConnectorLLM,
        connector_search: ConnectorSearch,
        connector_http: ConnectorHttp,
        connector_chroma: ConnectorChroma,
        model_name: str,
        connector_browser: Optional[Any] = None,
        collection_name: str = "langgraph_solver",
        search_k: int = 6,
        page_chars: int = 6000,
        full_capture: bool = False,
        always_synthesize: bool = False,
    ) -> None:
        self._connector_llm = connector_llm
        self._connector_search = connector_search
        self._connector_http = connector_http
        self._connector_chroma = connector_chroma
        self._connector_browser = connector_browser
        self._model_name = model_name
        self._collection_name = collection_name
        self._search_k = search_k
        self._page_chars = page_chars
        self._full_capture = bool(full_capture)
        #: Opt-in (default OFF, behavior unchanged): also run the synthesis pass on a NATURAL
        #: termination. LangGraph ends any turn that carries no tool call, so a "thinking out
        #: loud" turn ("Now I will compute the difference…") is returned verbatim as the
        #: deliverable — ``execution_sequential``'s agent cannot do this because it has an
        #: explicit ``finish(answer)`` action. Left off by default because it also rewrites turns
        #: that already ARE complete answers, which costs an extra call and can paraphrase a good
        #: answer worse; it needs a live A/B before it becomes the default.
        self._always_synthesize = bool(always_synthesize)

    def _build_llm(self) -> ChatOpenAI:
        """Point LangChain's OpenAI-compatible client at whatever provider the run is configured
        for. Fails FAST on a missing key: ``ChatOpenAI`` would otherwise silently fall back to a
        stray ``OPENAI_API_KEY`` env var and send it to OpenRouter, producing a confusing 401
        mid-run instead of an obvious misconfiguration at startup."""
        cfg = ConnectorConfig()
        if not cfg.llm_api_key:
            raise RuntimeError(
                "LangGraph arm: no LLM API key resolved (checked LLM_API_KEY / OPENROUTER_API_KEY "
                "/ OPENAI_API_KEY via ConnectorConfig). Refusing to start a run that would fail "
                "at the first call."
            )
        return ChatOpenAI(
            base_url=cfg.llm_api_url,
            api_key=cfg.llm_api_key,
            model=self._model_name,
            temperature=0.1,
        )

    async def solve(
        self,
        mandate: str,
        *,
        max_steps: int = 25,
        settings: Optional[Dict[str, Any]] = None,
        telemetry: Optional["TelemetrySession"] = None,
        run_id: Optional[str] = None,
    ) -> SolverResult:
        agent_io = AgentIO(
            connector_llm=self._connector_llm,
            connector_search=self._connector_search,
            connector_http=self._connector_http,
            connector_chroma=self._connector_chroma,
            connector_browser=self._connector_browser,
            telemetry=telemetry,
            collection_name=self._collection_name,
        )
        # F16 parity: same three connector_retry_* keys the graph and sequential arms read.
        retry = ToolRetry.from_settings(settings)
        tools = _make_tools(agent_io, self._search_k, self._page_chars, retry)
        llm = self._build_llm()
        graph = create_react_agent(llm, tools, prompt=_SYSTEM)

        started = time.perf_counter()
        messages: List[Any] = []
        run_error: Optional[BaseException] = None
        recursion_hit = False

        # astream (not ainvoke) so a mid-run crash still leaves us the messages — and therefore
        # the token usage — accumulated so far. ainvoke would raise and discard the whole state,
        # reporting $0 for a run that really did spend money.
        try:
            async for state in graph.astream(
                {"messages": [HumanMessage(content=mandate)]},
                config={"recursion_limit": max(4, int(max_steps) * 2)},
                stream_mode="values",
            ):
                if isinstance(state, dict) and state.get("messages"):
                    messages = state["messages"]
        except GraphRecursionError as exc:
            recursion_hit = True
            run_error = exc
            _logger.warning(f"LangGraph hit its step budget ({max_steps} steps): {exc}")
        except Exception as exc:  # noqa: BLE001
            run_error = exc
            _logger.error(f"LangGraph run failed: {exc}", exc_info=True)

        usages = _extract_usage(messages)
        final_text = _final_answer(messages)

        # Step exhaustion WITHOUT an exception: create_react_agent swaps the model's tool-calling
        # turn for a canned apology (_STEP_EXHAUSTED_TEXT) once remaining_steps runs low. Treat it
        # as the GraphRecursionError case it really is — otherwise the apology is truthy, the
        # synthesis below is skipped, and the run's evidence is thrown away with no warning.
        if final_text.strip() == _STEP_EXHAUSTED_TEXT:
            recursion_hit = True
            final_text = ""
            _logger.warning(
                f"LangGraph exhausted its step budget ({max_steps} steps) without raising; "
                "synthesizing from gathered evidence instead of returning its canned apology."
            )

        # Native-arm parity: out of steps (or stopped on a tool call) but evidence in hand ->
        # synthesize a best answer rather than scoring a hard 0 on a run that did the work.
        if not final_text or self._always_synthesize:
            evidence = _evidence_from(messages)
            if evidence:
                draft = f"\n\nDRAFT ANSWER (the agent's own last turn — may be incomplete):\n{final_text}" if final_text else ""
                try:
                    synth = await llm.ainvoke([
                        SystemMessage(content=_SYNTHESIS_SYSTEM),
                        HumanMessage(content=f"TASK:\n{mandate}\n\nEVIDENCE:\n{evidence}{draft}"),
                    ])
                    synth_text = synth.content if isinstance(synth.content, str) else str(synth.content or "")
                    # Only replace on a non-empty synthesis: an empty one must never turn a real
                    # last-turn answer into a hard 0.
                    final_text = synth_text if synth_text.strip() else final_text
                    usages.extend(_extract_usage([synth]))
                except Exception as exc:  # noqa: BLE001
                    run_error = run_error or exc
                    _logger.error(f"LangGraph forced synthesis failed: {exc}", exc_info=True)

        if telemetry is not None:
            for usage in usages:
                telemetry.record_llm_usage({
                    "model": self._model_name,
                    "usage": usage,
                    "duration": 0.0,
                })
            _record_io_parity(telemetry, messages, full_capture=self._full_capture)
            # F17 parity: a provider-side failure must be quarantinable, not scored as a real 0.
            # A recursion/step-budget stop is a genuine agent outcome, NOT infra — never tagged.
            if run_error is not None and not recursion_hit and is_infra_llm_failure(run_error):
                telemetry.record_timing(
                    name="llm_call", started_at=time.perf_counter(), success=False,
                    payload={"model": self._model_name, "infra_failed": True},
                    error=str(run_error),
                )

        wall_time_s = round(max(0.0, time.perf_counter() - started), 2)
        output = {"final_deliverable": final_text, "success": bool(final_text.strip())}
        observability = (
            summarize_observability({"output": output}, telemetry, self._model_name)
            if telemetry is not None else {"visit": {"count": 0}, "search": {"count": 0}}
        )

        result_out: SolverResult = {
            "final_deliverable": final_text,
            "success": bool(final_text.strip()),
            "observability": observability,
            "wall_time_s": wall_time_s,
            "llm_calls": len(usages),
            "search_calls": int(observability.get("search", {}).get("count", 0) or 0),
            "visit_calls": int(observability.get("visit", {}).get("count", 0) or 0),
        }
        cost_usd = observability.get("cost", {}).get("usd")
        if cost_usd is not None:
            result_out["cost_usd"] = cost_usd
        if recursion_hit:
            result_out["warning"] = f"step budget ({max_steps}) exhausted; answer synthesized from gathered evidence"
        elif run_error is not None:
            result_out["warning"] = f"langgraph run error: {type(run_error).__name__}: {run_error}"
        return result_out
