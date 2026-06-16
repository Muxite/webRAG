"""
Compiled-graph agent — the "expensive-model-authored scaffold, cheap-model execution"
comparator. Wired as the ``graph_compiled`` variant.

The Graph-of-Thoughts engine loses to a simple sequential agent on cheap models because
the cheap model has to BUILD the graph at runtime (decompose, plan parallelism, decide how
to aggregate) and builds bad graphs. This variant moves that planning OFF the cheap model:
the *expensive* model (Claude Code, paid offline by subscription) authors a static plan per
task class — a set of independent leaves to fan out plus an aggregation recipe — and the
cheap runtime model only EXECUTES it: gather one fact per leaf (in parallel), then run the
aggregation. The plan is read from the test module's ``get_compiled_plan()``.

Same toolset (search/visit) and the same ``AgentIO`` instrumentation as the graph and
sequential arms, so a graph_compiled-vs-sequential gap is attributable to *who planned the
structure* (paid offline model vs cheap runtime model), not to richer tools. Returns the
same result shape as ``run_sequential_execution``.
"""
import asyncio
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from agent.app import model_costs

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
from agent.app.testing.execution_sequential import _fmt_search
from agent.app.testing.compiled_plan import (
    plan_structure,
    substitute_deps,
    topological_waves,
    validate_plan,
)
from agent.app.testing import scaffold_compiler

_logger = logging.getLogger(__name__)

_LEAF_SYSTEM = (
    "You resolve ONE fact with web tools. Work one step at a time: think, then call exactly "
    "one tool. Tools:\n"
    "- search(query): web search; returns titles+URLs+snippets.\n"
    "- visit(url): read a page's full text. Use an EXACT URL from search results.\n"
    "- finish(answer): output the resolved fact. Include the exact source URL you read it from.\n"
    "Rules: open the authoritative page and read the fact off it before finishing; do not "
    "guess from memory. Each step return ONLY JSON: "
    "{\"thought\": \"...\", \"action\": \"search|visit|finish\", \"args\": {\"query|url|answer\": \"...\"}}."
)


async def _run_leaf(agent_io: AgentIO, instruction: str, expect: str, model_name: str,
                    leaf_steps: int, page_chars: int, search_k: int) -> str:
    """Gather a single leaf fact with a small bounded ReAct loop. Returns the fact text."""
    scratchpad: List[str] = []
    last_evidence = ""
    task = f"{instruction}\n\nReport exactly: {expect}"
    for step in range(leaf_steps):
        history = "\n\n".join(scratchpad[-6:]) if scratchpad else "(no actions yet)"
        messages = [
            {"role": "system", "content": _LEAF_SYSTEM},
            {"role": "user", "content": f"FACT TO RESOLVE:\n{task}\n\nYOUR STEPS SO FAR:\n{history}\n\nReturn the next step as JSON."},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name, temperature=0.1, max_tokens=700)
        raw = await agent_io.query_llm(payload, model_name=model_name)
        try:
            decision = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            decision = {}
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args") or {}

        if action == "finish" or step == leaf_steps - 1:
            answer = str(args.get("answer", "")).strip()
            if answer:
                return answer
            # Forced extraction from whatever page we last read.
            messages = [
                {"role": "system", "content": "Using ONLY the page text, answer the fact and include the source URL. If unknown, say UNKNOWN."},
                {"role": "user", "content": f"FACT:\n{task}\n\nPAGE TEXT:\n{last_evidence[:page_chars] or '(none)'}"},
            ]
            payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.0, max_tokens=300)
            return (await agent_io.query_llm(payload, model_name=model_name)) or "UNKNOWN"

        if action == "search":
            try:
                results = await agent_io.search(str(args.get("query", "")), count=search_k, timeout_seconds=20) or []
                obs = _fmt_search(results, search_k)
            except Exception as exc:  # noqa: BLE001
                obs = f"SEARCH ERROR: {exc}"
        elif action == "visit":
            url = str(args.get("url", "")).strip()
            try:
                content = (await agent_io.visit(url, timeout_seconds=30) or "")[:page_chars]
                last_evidence = f"SOURCE {url}\n{content}"
                obs = f"PAGE {url}:\n{content}"
            except Exception as exc:  # noqa: BLE001
                obs = f"VISIT ERROR for {url}: {exc}"
        else:
            obs = "INVALID ACTION. Use search/visit/finish."

        scratchpad.append(f"STEP {step+1}: action={action} args={json.dumps(args)[:160]}\nobservation={obs[:1200]}")
    return last_evidence[:400] or "UNKNOWN"


# --- Thin leaf: the harness owns the control flow; the LLM only does atomic perception ---------
# The JSON-ReAct leaf above makes the (weak) model choose actions, form JSON, and self-terminate —
# many ways to flake (returns UNKNOWN, bad JSON, stops early). The THIN leaf removes all of that:
# a FIXED pipeline (one search -> pick the wiki page -> one visit -> extract), where the model is
# asked only micro-questions with tiny outputs we can read off directly. Same tools, far less rope.
_THIN_QUERY_SYS = (
    "Output ONLY a short web-search query (a few words) that would find the requested fact. "
    "No quotes, no explanation, no punctuation — just the query."
)
_THIN_EXTRACT_SYS = (
    "Read the PAGE and answer the QUESTION with ONLY the value — a name, a number, or a year — and "
    "nothing else (no sentence, no units unless asked, no source). If the PAGE does not contain it, "
    "output exactly: UNKNOWN."
)


def _votes_for_model(model_name: str) -> int:
    """Price-aware redundancy. A dirt-cheap (usually weaker) model spends its cheapness on MORE
    independent extractions to vote/prune over; a premium model needs only one trustworthy call.
    Driven by the model's output price/Mtok (model_costs); override with IDEA_TEST_COMPILED_VOTES.
    """
    override = os.environ.get("IDEA_TEST_COMPILED_VOTES", "").strip()
    if override.isdigit():
        return max(1, int(override))
    try:
        pricing = model_costs._lookup_pricing(model_name) or {}
        out_price = float(pricing.get("output_per_million") or 0.0)
    except Exception:  # noqa: BLE001
        out_price = 0.0
    if out_price <= 0.0:
        return 3            # unknown price -> a little redundancy is cheap insurance
    if out_price <= 1.0:
        return 5            # dirt cheap -> heavy redundancy
    if out_price <= 5.0:
        return 3
    return 1               # premium -> trust a single call


def _vote_key(ans: str) -> str:
    """Normalized voting key so equivalent answers vote together. Prefer the longest number token
    when present ('1,642 m', '1642 metres', 'Max depth: 1,642' -> '1642'); else a cleaned text key."""
    low = ans.strip().lower()
    nums = re.findall(r"\d[\d,]*", low)
    if nums:
        return max((n.replace(",", "") for n in nums), key=len)
    return re.sub(r"[^a-z0-9]+", " ", low).strip()


async def _thin_extract_once(agent_io: AgentIO, page: str, instruction: str, model_name: str,
                             temperature: float) -> str:
    ep = agent_io.build_llm_payload(
        messages=[{"role": "system", "content": _THIN_EXTRACT_SYS},
                  {"role": "user", "content": f"PAGE:\n{page}\n\nQUESTION: {instruction}"}],
        json_mode=False, model_name=model_name, temperature=temperature, max_tokens=24,
    )
    return (await agent_io.query_llm(ep, model_name=model_name) or "").strip()


async def _vote_extract(agent_io: AgentIO, page: str, instruction: str, model_name: str, k: int) -> str:
    """Run k INDEPENDENT extractions (neutral prompt — no leading answer) and return the majority
    value; '' if every sample is UNKNOWN. The cheap 'make candidate nodes -> prune' step.

    The first sample is ANCHORED at temperature 0 (the deterministic best read); the remaining
    k-1 add mild diversity (temp 0.3) only to surface alternatives. Ties break toward the anchor.
    This keeps clean single-read facts (e.g. an infobox year) stable while still letting redundancy
    rescue genuinely uncertain extractions (e.g. a chain hop) — voting that helps, never hurts.
    """
    if k <= 1:
        a = await _thin_extract_once(agent_io, page, instruction, model_name, 0.0)
        return a if a and not a.upper().startswith("UNKNOWN") else ""
    temps = [0.0] + [0.3] * (k - 1)
    answers = await asyncio.gather(*[
        _thin_extract_once(agent_io, page, instruction, model_name, t) for t in temps
    ])
    cands = [a for a in answers if a and not a.upper().startswith("UNKNOWN")]
    if not cands:
        return ""
    counts = Counter(_vote_key(a) for a in cands)
    top_count = counts.most_common(1)[0][1]
    tied = {key for key, c in counts.items() if c == top_count}
    anchor = answers[0] if (answers[0] and not answers[0].upper().startswith("UNKNOWN")) else ""
    chosen_key = _vote_key(anchor) if anchor and _vote_key(anchor) in tied else counts.most_common(1)[0][0]
    return next(a for a in cands if _vote_key(a) == chosen_key)


async def _run_leaf_thin(agent_io: AgentIO, instruction: str, expect: str, model_name: str,
                         page_chars: int, search_k: int) -> str:
    """Fixed search->pick->visit->vote-extract pipeline of thin prompts.

    The model only answers micro-questions (a ~few-token search query, then value extractions) —
    no JSON, no action-planning. Price-aware k-sample voting (``_votes_for_model``) makes a cheap
    model's noisy extraction reliable via redundancy + majority pruning, and a second candidate page
    is tried (a repeat cycle) if the first yields no consensus. Returns ``"<value> — source:<url>"``.
    """
    # 1) thin search query (tiny output)
    qp = agent_io.build_llm_payload(
        messages=[{"role": "system", "content": _THIN_QUERY_SYS}, {"role": "user", "content": instruction}],
        json_mode=False, model_name=model_name, temperature=0.0, max_tokens=24,
    )
    raw_q = (await agent_io.query_llm(qp, model_name=model_name) or "").strip()
    query = raw_q.splitlines()[0].strip(' "\'')[:200] if raw_q else " ".join(instruction.split()[:12])

    # 2) search
    try:
        results = await agent_io.search(query, count=search_k, timeout_seconds=20) or []
    except Exception as exc:  # noqa: BLE001
        _logger.warning(f"thin leaf search failed: {exc}")
        results = []
    if not results:
        return "UNKNOWN"

    # 3) candidate pages — Wikipedia article(s) first (stable), then the rest (more 'nodes' to try)
    urls = [str(r.get("url", "")) for r in results if str(r.get("url", ""))]
    urls.sort(key=lambda u: 0 if "wikipedia.org/wiki/" in u else 1)
    if not urls:
        return "UNKNOWN"

    # 4/5) try up to 2 candidate pages; vote-extract on each (repeat cycle if no consensus)
    k = _votes_for_model(model_name)
    for url in urls[:2]:
        try:
            page = (await agent_io.visit(url, timeout_seconds=30) or "")[:page_chars]
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"thin leaf visit failed for {url}: {exc}")
            continue
        ans = await _vote_extract(agent_io, page, instruction, model_name, k)
        if ans:
            return f"{ans} — source: {url}"
    return f"UNKNOWN — {urls[0]}"


async def _execute_plan(agent_io: AgentIO, plan: Dict[str, Any], model_name: str, max_tokens: int) -> str:
    """Execute a compiled DAG plan topologically, then run the aggregation call.

    Leaves are grouped into dependency waves (``compiled_plan.topological_waves``): each wave's
    leaves are mutually independent and fan out in parallel (bounded by the concurrency cap);
    later waves run after their upstreams, with each dependent leaf's ``{dep_id}`` placeholders
    substituted with the resolved upstream fact. A plan with no dependencies reduces to a single
    wave — the original pure-parallel fan-out (so test 052 is unchanged). Then aggregate over all
    gathered facts in plan order.
    """
    norm = validate_plan(plan)  # normalizes ids/deps and rejects cycles/missing deps
    leaves: List[Dict[str, Any]] = norm["leaves"]
    aggregation: str = norm["aggregation"]
    if not leaves:
        return ""
    by_id = {leaf["id"]: leaf for leaf in leaves}
    waves = topological_waves(leaves)

    leaf_steps = int(os.environ.get("IDEA_TEST_COMPILED_LEAF_STEPS", "4"))
    page_chars = int(os.environ.get("IDEA_TEST_COMPILED_PAGE_CHARS", "6000"))
    search_k = int(os.environ.get("IDEA_TEST_COMPILED_SEARCH_K", "6"))
    concurrency = max(1, int(os.environ.get("IDEA_TEST_COMPILED_CONCURRENCY", "6")))
    # "react" (default): per-leaf JSON ReAct loop. "thin": fixed micro-prompt pipeline (cheaper,
    # more robust for weak models — the harness owns control flow, the LLM only perceives).
    leaf_mode = os.environ.get("IDEA_TEST_COMPILED_LEAF_MODE", "react").strip().lower()
    sem = asyncio.Semaphore(concurrency)

    results: Dict[str, str] = {}

    async def _guarded(leaf: Dict[str, Any]) -> Tuple[str, str]:
        async with sem:
            # Substitute resolved upstream facts (run in earlier waves) into this instruction.
            dep_results = {dep: results.get(dep, "UNKNOWN") for dep in leaf["depends_on"]}
            instruction = substitute_deps(leaf["instruction"], dep_results)
            try:
                if leaf_mode == "thin":
                    fact = await _run_leaf_thin(
                        agent_io, instruction, leaf["expect"], model_name, page_chars, search_k,
                    )
                else:
                    fact = await _run_leaf(
                        agent_io, instruction, leaf["expect"],
                        model_name, leaf_steps, page_chars, search_k,
                    )
            except Exception as exc:  # noqa: BLE001 — a single bad leaf must not sink the run
                _logger.warning(f"compiled leaf '{leaf['id']}' failed: {exc}")
                fact = "UNKNOWN"
            return leaf["id"], fact

    for wave in waves:
        gathered = await asyncio.gather(*[_guarded(by_id[lid]) for lid in wave])
        for lid, fact in gathered:
            results[lid] = fact

    # Aggregate over every leaf, in declared plan order (stable for the judge/validators).
    # Facts are NUMBERED, not tagged with the leaf id: weak models copy a leading "[leaf_id]" tag
    # verbatim as if it were a citation instead of citing the source URL inside the fact.
    facts_block = "\n".join(
        f"Fact {i}: {results.get(leaf['id'], 'UNKNOWN')}" for i, leaf in enumerate(leaves, 1)
    )
    # Compilers sometimes template {leaf_id} into the aggregation too — fill those in as well so
    # the recipe reads with resolved values, not literal placeholders (facts_block still carries them).
    aggregation = substitute_deps(aggregation, results)

    messages = [
        {"role": "system", "content": (
            "You are an aggregation step. Follow the AGGREGATION INSTRUCTION exactly, using ONLY the "
            "gathered facts. Cite ONLY the http(s) source URLs that appear inside those facts — never "
            "output the 'Fact N' labels or any bracketed internal identifiers as if they were citations."
        )},
        {"role": "user", "content": f"AGGREGATION INSTRUCTION:\n{aggregation}\n\nGATHERED FACTS:\n{facts_block}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name, temperature=0.2, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


async def _resolve_plan(
    test_module: IdeaTestModule,
    mandate: str,
    correlation_id: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    summarize_observability_func,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Select the compiled plan for this run and return ``(plan, plan_meta)``.

    Source is controlled by ``IDEA_TEST_COMPILED_PLAN_SOURCE``:
      * ``hand`` (default): use the test module's ``get_compiled_plan()`` if present; otherwise
        fall back to the offline compiler (every task still gets a scaffold automatically).
      * ``auto``: always use the compiler — so B-auto can be measured even where a hand plan
        exists, isolating "did the compiler reproduce the hand-authored structure".

    The compiler is cache-first: a warm ``compiled_plans/`` cache costs no LLM call. On a cold
    miss it authors the plan with a *separate* telemetry session/AgentIO so the offline authoring
    cost never pollutes the cheap model's runtime dollars; that cost is returned in ``plan_meta``.
    """
    source = os.environ.get("IDEA_TEST_COMPILED_PLAN_SOURCE", "hand").strip().lower()
    force = os.environ.get("IDEA_TEST_COMPILED_FORCE_RECOMPILE", "").strip().lower() in ("1", "true", "yes", "on")
    hand_fn = getattr(test_module.module, "get_compiled_plan", None)
    meta: Dict[str, Any] = {"plan_source": None, "compiler": {}}

    # hand path
    if source != "auto" and callable(hand_fn):
        try:
            plan = hand_fn()
            meta["plan_source"] = "hand"
            meta["plan_structure"] = plan_structure(plan)
            return plan, meta
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"hand get_compiled_plan() failed ({exc}); falling back to compiler")

    # auto / compiler path — cache-first
    author_model = os.environ.get("IDEA_TEST_COMPILED_AUTHOR_MODEL", scaffold_compiler.DEFAULT_AUTHOR_MODEL).strip()
    compile_max_tokens = int(os.environ.get("IDEA_TEST_COMPILED_AUTHOR_MAX_TOKENS", "2048"))
    cached = None if force else scaffold_compiler.load_cached_plan(mandate)
    if cached is not None:
        meta["plan_source"] = "auto"
        meta["compiler"] = {"cache": "hit", "author_model": author_model}
        meta["plan_structure"] = plan_structure(cached)
        return cached, meta

    # Cold miss: author the plan on an isolated telemetry session so its cost is separate.
    compile_telemetry = TelemetrySession(
        enabled=True, mandate=mandate, correlation_id=f"{correlation_id}_compile", trace_path=None,
    )
    compile_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        telemetry=compile_telemetry, collection_name="scaffold_compiler",
    )
    try:
        plan, info = await scaffold_compiler.compile_plan(
            mandate, author_model=author_model, agent_io=compile_io,
            max_tokens=compile_max_tokens, force=force,
        )
    except scaffold_compiler.CompileError as exc:
        _logger.error(f"scaffold compilation failed: {exc}")
        if callable(hand_fn):
            try:
                plan = hand_fn()
                meta["plan_source"] = "hand_fallback"
                meta["plan_structure"] = plan_structure(plan)
                return plan, meta
            except Exception:  # noqa: BLE001
                pass
        return None, meta

    compile_cost = summarize_observability_func({"output": {}}, compile_telemetry, author_model).get("cost", {})
    meta["plan_source"] = "auto"
    info["cost"] = compile_cost
    meta["compiler"] = info
    meta["plan_structure"] = info.get("structure") or plan_structure(plan)
    return plan, meta


async def run_compiled_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    summarize_observability_func=summarize_observability,
) -> Dict[str, Any]:
    """Run the compiled-graph agent; same return shape as ``run_sequential_execution``."""
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_graph_compiled_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}_graph_compiled.jsonl"
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    # Resolve the plan FIRST (hand or compiler). Any compiler authoring runs on its own isolated
    # telemetry; building the runtime AgentIO afterward re-points the shared connectors at the
    # runtime telemetry, so only execution counts toward this run's cost.
    plan, plan_meta = await _resolve_plan(
        test_module, mandate, correlation_id,
        connector_llm, connector_search, connector_http, connector_chroma,
        summarize_observability_func,
    )

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id, trace_path=trace_path)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    deliverable = ""
    if plan is None:
        _logger.error(f"Test {test_id} has no compiled plan (hand or compiled); graph_compiled cannot run.")
    else:
        _logger.info(f"[{test_id}] graph_compiled plan_source={plan_meta.get('plan_source')} "
                     f"structure={plan_meta.get('plan_structure')}")
        try:
            deliverable = await _execute_plan(agent_io, plan, model_name, max_tokens)
        except Exception as exc:
            _logger.error(f"Compiled execution failed: {exc}", exc_info=True)

    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": "graph_compiled",
        "plan_source": plan_meta.get("plan_source"),
        "plan_structure": plan_meta.get("plan_structure"),
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
        # The offline scaffold-authoring cost, reported SEPARATELY from the runtime observability
        # above (which is the cheap model's only on-task spend). Cache hits cost nothing.
        "compiler": plan_meta.get("compiler", {}),
        "plan_source": plan_meta.get("plan_source"),
        "telemetry": {
            "correlation_id": correlation_id,
            "trace_file": str(trace_path),
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
        "telemetry_raw": telemetry_summary,
    }
