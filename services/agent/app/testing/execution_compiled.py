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
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
    sem = asyncio.Semaphore(concurrency)

    results: Dict[str, str] = {}

    async def _guarded(leaf: Dict[str, Any]) -> Tuple[str, str]:
        async with sem:
            # Substitute resolved upstream facts (run in earlier waves) into this instruction.
            dep_results = {dep: results.get(dep, "UNKNOWN") for dep in leaf["depends_on"]}
            instruction = substitute_deps(leaf["instruction"], dep_results)
            try:
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
    facts_block = "\n".join(f"- [{leaf['id']}] {results.get(leaf['id'], 'UNKNOWN')}" for leaf in leaves)
    # Compilers sometimes template {leaf_id} into the aggregation too — fill those in as well so
    # the recipe reads with resolved values, not literal placeholders (facts_block still carries them).
    aggregation = substitute_deps(aggregation, results)

    messages = [
        {"role": "system", "content": "You are an aggregation step. Follow the AGGREGATION INSTRUCTION exactly, using ONLY the gathered facts. Cite the source URLs they contain."},
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
