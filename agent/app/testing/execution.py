"""
Test execution engine.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

from agent.app.observation import clean_operation
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.idea_engine import IdeaDagEngine
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
    connector_browser=None,
) -> Dict[str, Any]:
    """
    Run a no-graph baseline: ``parametric`` (single completion, no tools),
    ``minimal`` (search-only, no crawl) or ``naive_rag`` (one fixed round of
    search+visit stuffed into one synthesis call).

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
    :param connector_browser: Optional headless-Chrome fallback connector (F18 — wired
        uniformly across every execution variant so no arm is handicapped by bot-blocks
        relative to another; None disables it for this run).
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
        connector_browser=connector_browser,
        telemetry=telemetry,
        collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    deliverable = ""
    try:
        if variant == "naive_rag":
            deliverable = await _run_naive_rag(agent_io, mandate, model_name, max_tokens)
        elif variant == "minimal":
            deliverable = await _run_search_only(agent_io, mandate, model_name, max_tokens)
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
    """Single completion with NO tools — the hardness floor / parametric-knowledge control.

    The prompt must NOT invite fabricated citations (the model has no web access), or the floor is
    polluted by hallucinated URLs that the validators then credit. Instead: answer only from
    confident knowledge, say UNKNOWN when unsure, and never invent sources. A low score here is the
    signal we want — it proves the task needs grounding, not recall.
    """
    messages = [
        {"role": "system", "content": (
            "You are answering from memory ONLY — you have NO web access and cannot look anything up. "
            "Answer the task as accurately as you can from knowledge you are confident in. For any "
            "fact you are not sure of, write UNKNOWN rather than guessing. Do NOT invent or cite any "
            "source URLs — you have not visited any pages."
        )},
        {"role": "user", "content": mandate},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


async def _run_search_only(agent_io: AgentIO, mandate: str, model_name: str, max_tokens: int) -> str:
    """Search-only (no crawl): search-result snippets -> single synthesis call.

    The ``minimal`` tooling rung. Isolates the value of *retrieval without reading*:
    the model sees search titles/snippets/URLs but never opens a page, so any lift
    from ``partial``/``full`` is attributable to actually visiting evidence.
    """
    search_k = int(os.environ.get("IDEA_TEST_MINIMAL_SEARCH_K", "8"))
    total_chars = int(os.environ.get("IDEA_TEST_MINIMAL_TOTAL_CHARS", "12000"))

    try:
        results = await agent_io.search(mandate, count=search_k, timeout_seconds=20) or []
    except Exception as exc:
        _logger.warning(f"minimal search failed: {exc}")
        results = []

    snippets: List[str] = []
    budget = total_chars
    for item in results:
        if budget <= 0:
            break
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        snippet = (item.get("description") or item.get("snippet") or "").strip()
        block = f"URL: {url}\nTITLE: {title}\nSNIPPET: {snippet}"[:budget]
        budget -= len(block)
        snippets.append(block)

    context = "\n\n---\n\n".join(snippets) if snippets else "(no search results)"
    messages = [
        {"role": "system", "content": "You are a research assistant with access ONLY to web search result snippets (you cannot open pages). Answer the task using the snippets plus your own knowledge. Cite source URLs from the snippets where relevant. If the snippets are insufficient, say so explicitly rather than guessing."},
        {"role": "user", "content": f"TASK:\n{mandate}\n\nSEARCH RESULTS:\n{context}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


async def _run_naive_rag(agent_io: AgentIO, mandate: str, model_name: str, max_tokens: int) -> str:
    """Iterative naive RAG (no graph) with a HARD, predictable budget.

    Keeps going — visit, hop links, occasionally search — until it can answer or the
    budget is spent, then forces a final synthesis that either answers or explicitly
    declares the evidence INSUFFICIENT (never guesses, never hangs). This is the honest
    "cheap model that keeps trying" floor for the ladder: it can recover from a bad first
    page, unlike a one-shot RAG, but it has none of the graph engine's structured
    re-expansion — so the graph arm's lift over it isolates the value of the token-burner,
    not mere persistence.

    Cost is bounded and predictable — the worst case is a closed formula:
      * <= (MAX_ROUNDS + 1) LLM calls: one decision per round + one final synthesis,
        each with <= TOTAL_CHARS of evidence in and <= max_tokens out;
      * <= MAX_SEARCHES web searches (the expensive tool — kept scarce on purpose);
      * <= MAX_VISITS page fetches (the cheap "burn"; link-hopping lets it keep going
        WITHOUT spending searches, matching the search-is-pricier-than-tokens economics).
    Every knob is an ``IDEA_TEST_NAIVE_RAG_*`` env var so a run's price ceiling is set
    up front. Model-proposed URLs are accepted ONLY if they are known candidate links or
    unvisited search results, so a hallucinated URL can never inflate the visit budget.
    """
    max_rounds = int(os.environ.get("IDEA_TEST_NAIVE_RAG_MAX_ROUNDS", "4"))
    max_searches = int(os.environ.get("IDEA_TEST_NAIVE_RAG_MAX_SEARCHES", "2"))
    max_visits = int(os.environ.get("IDEA_TEST_NAIVE_RAG_MAX_VISITS", "8"))
    visits_per_round = int(os.environ.get("IDEA_TEST_NAIVE_RAG_VISITS_PER_ROUND", "3"))
    search_k = int(os.environ.get("IDEA_TEST_NAIVE_RAG_SEARCH_K", "5"))
    per_page_chars = int(os.environ.get("IDEA_TEST_NAIVE_RAG_PAGE_CHARS", "8000"))
    total_chars = int(os.environ.get("IDEA_TEST_NAIVE_RAG_TOTAL_CHARS", "30000"))
    links_per_page = int(os.environ.get("IDEA_TEST_NAIVE_RAG_LINKS_PER_PAGE", "20"))

    state = {"searches": 0, "visits": 0, "evidence_chars": 0}
    visited: set = set()
    evidence: List[str] = []              # "SOURCE URL: <url>\n<cleaned content>"
    hop_candidates: Dict[str, str] = {}   # unvisited absolute url -> anchor text
    search_pool: List[Dict[str, str]] = []  # unvisited {url,title,desc} from searches
    queue: List[str] = []                 # urls scheduled to visit next

    # Mandate-named URLs take priority (mirrors the engine's mandate enforcement).
    for u in _URL_RE.findall(mandate):
        cu = u.rstrip('.,);]')
        if cu not in queue:
            queue.append(cu)

    async def do_search(query: str) -> None:
        if state["searches"] >= max_searches or not query.strip():
            return
        state["searches"] += 1
        try:
            results = await agent_io.search(query, count=search_k, timeout_seconds=20) or []
        except Exception as exc:
            _logger.warning(f"naive_rag search failed: {exc}")
            return
        pool_urls = {s["url"] for s in search_pool}
        for item in results:
            u = (item.get("url") or "").strip()
            if u and u not in visited and u not in queue and u not in pool_urls:
                search_pool.append({"url": u, "title": item.get("title", ""),
                                    "desc": (item.get("description") or "")[:200]})
                pool_urls.add(u)

    def drain_pool(n: int) -> None:
        while search_pool and n > 0:
            item = search_pool.pop(0)
            if item["url"] not in visited and item["url"] not in queue:
                queue.append(item["url"])
                n -= 1

    async def visit(url: str) -> bool:
        if state["visits"] >= max_visits or url in visited or state["evidence_chars"] >= total_chars:
            return False
        visited.add(url)
        state["visits"] += 1
        content, links = await _naive_fetch_page(agent_io, url, per_page_chars, links_per_page)
        if content:
            room = total_chars - state["evidence_chars"]
            snippet = content[:max(0, room)]
            if snippet:
                evidence.append(f"SOURCE URL: {url}\n{snippet}")
                state["evidence_chars"] += len(snippet)
        for lu, lt in links:
            if lu not in visited and lu not in hop_candidates:
                hop_candidates[lu] = lt
        return True

    # Seed: if the mandate named no URLs, spend one search to get started.
    if not queue:
        await do_search(mandate)
        drain_pool(visits_per_round)

    answer = None
    for round_i in range(1, max_rounds + 1):
        visited_any = False
        picked = 0
        while queue and picked < visits_per_round and state["visits"] < max_visits:
            if await visit(queue.pop(0)):
                visited_any = True
            picked += 1

        budget = {
            "rounds_left": max_rounds - round_i,
            "searches_left": max_searches - state["searches"],
            "visits_left": max_visits - state["visits"],
        }
        decision = await _naive_decide(
            agent_io, mandate, model_name, max_tokens, evidence, hop_candidates, search_pool, budget
        )
        if decision.get("can_answer") and (decision.get("answer") or "").strip():
            answer = decision["answer"].strip()
            break

        # Enqueue only KNOWN urls the model chose (anti-hallucination cost guard).
        known = set(hop_candidates) | {s["url"] for s in search_pool}
        for u in (decision.get("visit_urls") or []):
            u = (u or "").strip()
            if u in known and u not in visited and u not in queue:
                queue.append(u)

        if not queue:
            drain_pool(visits_per_round)
        if not queue and state["searches"] < max_searches:
            await do_search((decision.get("search_query") or "").strip())
            drain_pool(visits_per_round)

        # Nothing visited this round and nothing left to do -> stop early (still bounded).
        if not visited_any and not queue:
            break

    if answer:
        return answer
    return await _naive_final_synthesis(agent_io, mandate, model_name, max_tokens, evidence)


def _parse_page(raw: str, base_url: str, per_page_chars: int, links_per_page: int) -> Tuple[str, List[Tuple[str, str]]]:
    """Pure CPU-bound HTML -> (cleaned text, [(absolute_url, anchor_text)]).

    Run in a thread by ``_naive_fetch_page`` — bs4 parsing blocks the event loop.
    """
    try:
        content = clean_operation(raw)[:per_page_chars]
    except Exception:
        content = (raw or "")[:per_page_chars]
    links: List[Tuple[str, str]] = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        seen: set = set()
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            absolute = urljoin(base_url, href)
            if urlparse(absolute).scheme not in ("http", "https") or absolute in seen:
                continue
            seen.add(absolute)
            text = " ".join((a.get_text(" ", strip=True) or "").split())[:80]
            links.append((absolute, text))
            if len(links) >= links_per_page:
                break
    except Exception:
        pass
    return content, links


async def _naive_fetch_page(
    agent_io: AgentIO, url: str, per_page_chars: int, links_per_page: int
) -> Tuple[str, List[Tuple[str, str]]]:
    """Fetch a page ONCE and derive both cleaned content and hop-candidate links.

    Uses ``fetch_url`` (raw HTML) rather than ``visit`` because ``visit`` returns cleaned
    text with the hrefs stripped — we need the raw markup to extract links to hop to. The
    HTTP cost is tracked by the connector either way; we re-record the visit for
    observability parity. Parsing is offloaded to a thread (loop-blocking bs4).
    """
    try:
        raw = await agent_io.fetch_url(url, timeout_seconds=25)
    except Exception as exc:
        _logger.warning(f"naive_rag fetch failed for {url}: {exc}")
        return "", []
    if not raw:
        return "", []
    loop = asyncio.get_running_loop()
    content, links = await loop.run_in_executor(
        None, _parse_page, raw, url, per_page_chars, links_per_page
    )
    telemetry = getattr(agent_io, "telemetry", None)
    if telemetry is not None and content:
        try:
            telemetry.record_document_seen(source="visit", document={"url": url, "content": content})
        except Exception:
            pass
    return content, links


def _parse_decision(raw: str) -> Dict[str, Any]:
    """Parse the decision LLM's JSON, tolerating fences/prose around the object."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


async def _naive_decide(
    agent_io: AgentIO,
    mandate: str,
    model_name: str,
    max_tokens: int,
    evidence: List[str],
    hop_candidates: Dict[str, str],
    search_pool: List[Dict[str, str]],
    budget: Dict[str, int],
) -> Dict[str, Any]:
    """One bounded decision step: answer now, or request more evidence (prefer hops)."""
    evidence_text = "\n\n---\n\n".join(evidence) if evidence else "(no evidence gathered yet)"
    cand_lines = [f"- {u}  ({t})" for u, t in list(hop_candidates.items())[:40]]
    pool_lines = [f"- {s['url']}  ({s.get('title', '')})" for s in search_pool[:15]]
    system = (
        "You are an iterative research agent working under a strict, limited budget. Using ONLY "
        "the evidence gathered so far, decide whether you can now answer the task confidently and "
        "with citations. If yes, answer. If not, request MORE evidence — STRONGLY prefer visiting "
        "specific pages by following the candidate links (cheap) over a new web search (expensive "
        "and scarce). Never guess. Respond with JSON ONLY:\n"
        '{"can_answer": boolean, "answer": string, "visit_urls": [string], "search_query": string}\n'
        "Fill \"answer\" (with the cited source URLs) only when can_answer is true. Choose "
        "visit_urls from the candidate links or unvisited search results below. Use search_query "
        "only when no listed link is likely to help — and keep it to a few focused keywords "
        "(max ~400 chars / 50 words), never a full sentence, or the search API will reject it."
    )
    user = (
        f"TASK:\n{mandate}\n\n"
        f"BUDGET LEFT — rounds:{budget['rounds_left']} searches:{budget['searches_left']} "
        f"visits:{budget['visits_left']}\n\n"
        f"EVIDENCE GATHERED:\n{evidence_text}\n\n"
        f"CANDIDATE LINKS (hop targets):\n{chr(10).join(cand_lines) if cand_lines else '(none)'}\n\n"
        f"UNVISITED SEARCH RESULTS:\n{chr(10).join(pool_lines) if pool_lines else '(none)'}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    payload = agent_io.build_llm_payload(
        messages=messages, json_mode=True, model_name=model_name, temperature=0.2, max_tokens=max_tokens
    )
    try:
        raw = await agent_io.query_llm(payload, model_name=model_name)
    except Exception as exc:
        _logger.warning(f"naive_rag decision failed: {exc}")
        return {}
    return _parse_decision(raw or "")


async def _naive_final_synthesis(
    agent_io: AgentIO, mandate: str, model_name: str, max_tokens: int, evidence: List[str]
) -> str:
    """Forced final answer: synthesize from evidence, or explicitly say it's insufficient."""
    context = "\n\n---\n\n".join(evidence) if evidence else "(no sources retrieved)"
    messages = [
        {"role": "system", "content": (
            "You are a research assistant. Answer the task using ONLY the provided sources, and "
            "cite the source URLs you used. If the sources do not contain enough to answer, respond "
            "with a message that STARTS with 'INSUFFICIENT EVIDENCE:' and briefly states what is "
            "missing. Do NOT guess or invent facts or URLs."
        )},
        {"role": "user", "content": f"TASK:\n{mandate}\n\nSOURCES:\n{context}"},
    ]
    payload = agent_io.build_llm_payload(
        messages=messages, json_mode=False, model_name=model_name, temperature=0.3, max_tokens=max_tokens
    )
    try:
        out = await agent_io.query_llm(payload, model_name=model_name)
    except Exception as exc:
        _logger.warning(f"naive_rag final synthesis failed: {exc}")
        out = ""
    return (out or "").strip() or "INSUFFICIENT EVIDENCE: retrieval did not gather usable sources."


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
    connector_browser=None,
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
    :param connector_browser: Optional headless-Chrome fallback connector (F18 — wired
        uniformly across every execution variant; None disables it for this run).
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
        connector_browser=connector_browser,
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

    started = time.perf_counter()
    # Effort tiers set settings["max_steps"]; fall back to env then default.
    max_steps = int(idea_settings.get("max_steps") or os.environ.get("IDEA_TEST_MAX_STEPS", "50"))

    # The benchmark harness delegates the ENTIRE control loop + finalize to the same
    # `IdeaDagEngine.run()` that production callers use, instead of hand-rolling engine setup,
    # the step loop and finalize (which historically let the two implementations drift —
    # "Phase 2 bug #4"). `run()` performs its own memo-namespace / memory-manager / GoT setup,
    # runs the shared `_run_loop`, and returns the fully-annotated final payload (graph,
    # pending_nodes_count, candidate_coverage_*, got_stats, grounding). `fail_soft=True`
    # preserves the harness's long-standing fail-soft contract: a mid-run `step()` exception
    # is logged and the run still finalizes with partial state (production `run()` is
    # fail-fast). No `run_id` is passed, so the checkpointer is never touched.
    # `task_source` is the harness-only argument: it gives the strategy library's read-time leak
    # re-check the task module to screen a retrieved note against (production callers have no
    # benchmark ledger, and pass nothing). `getattr` because the duck-typed stand-ins some tests
    # pass as `test_module` carry only the accessors used above.
    output = await engine.run(mandate, max_steps=max_steps, fail_soft=True,
                              task_source=getattr(test_module, "module", None))
    graph_dict = output.get("graph") if isinstance(output, dict) else None

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
        "graph": graph_dict,
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
