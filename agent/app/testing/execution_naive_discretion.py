"""
Naive-discretion agent — the honest FLOOR for the "does structure give uplift" thesis.

Every other arm in this harness bakes in engineered structure of some kind:
``sequential_react`` ships a system prompt that already states the winning strategy (decompose
into sub-facts, resolve each, never answer from memory), a budget reverse-engineered to match
``graph_compiled``, a forced-synthesis rescue and a search-dedup guard; ``naive_rag`` is a fixed
one-round pipeline, not an agent loop at all; ``graph``/``graph_compiled`` are the structure being
measured. So "structure vs no structure" had never actually been measured against a no-structure
arm. This variant is that arm: the SAME tools, the SAME action-execution machinery, a MINIMAL
prompt, and a deliberately generous budget — what a model does when nobody tells it how.

What is deliberately absent (each one would collapse the variable under measurement):
  * no decomposition/verification/citation strategy in the prompt — the menu, the JSON shape and
    the turn count, nothing else;
  * no dedup guard — re-running the same search is a real naive-loop failure mode, and this arm
    exists to measure it, not to prevent it;
  * no forced-synthesis rescue — a run whose model never calls ``finish`` ends with
    ``success = False``. That is signal, not a bug.

What it deliberately REUSES (so the floor is honest about tools, not just weak): actions dispatch
through ``IdeaDagEngine._execute_action_guarded`` — the same registry resolution, the same
``allowed_actions`` gate (hence the same ``ToolsConfig`` tool-ablation semantics), the same
tool-failure retry and per-action watchdog a graph run gets. Decomposition/beam/merge/scoring live
in ``expand``/``run``, which this loop simply never calls. Tool DESCRIPTIONS come verbatim from the
shared ``ACTIONS:`` block via ``core_actions_menu_lines`` rather than a hand-written second copy,
so no arm is handicapped by worse-documented tools.

Returns the same result shape as ``run_sequential_execution`` so cost instrumentation, validation
and the analysis scripts treat it identically. Wired as the ``naive_discretion`` variant.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.agent_io import AgentIO
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.expansion import core_actions_menu_lines
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing import json_telemetry as _json_telemetry

_logger = logging.getLogger(__name__)

VARIANT = "naive_discretion"

#: The one loop-level pseudo-action: it is not a tool, it ends the run. Everything else the model
#: names is dispatched through the registry (and gated by ``allowed_actions``).
FINISH_ACTION = "finish"

_FINISH_MENU_LINE = (
    '- finish: details={answer: "<your final answer>"}. Ends the run and returns that answer.'
)

_SYSTEM_TEMPLATE = (
    "You are an agent working on a TASK. These actions are available to you:\n"
    "{menu}\n"
    "{finish_line}\n"
    "You have at most {max_steps} turns. Each turn, return ONLY JSON:\n"
    '{{"thought": "...", "action": "<one action name from the list>", "details": {{...}}}}\n'
    "Choose ONE action per turn. How you use your turns is entirely your own judgement."
)


def _build_action_menu(engine: IdeaDagEngine) -> str:
    """The model-facing action menu for this run's ``allowed_actions``.

    Core actions are described in the shipped prompt's own words; anything an installed tool pack
    added (``ToolsConfig``) describes itself through the registry. The registry's extra-actions
    block wrapper (``build_extra_actions_menu``) is NOT used here: its header/footer are written
    in Graph-of-Thought vocabulary ("subproblem", "child", "no separate visit node is needed"),
    which is both wrong and quietly steering in a flat loop — the bare per-action lines it would
    wrap are identically formatted to the core ones, so they are simply appended.

    :param engine: The constructed engine (its ``allowed_actions`` is already ToolsConfig-derived).
    :returns: Newline-joined menu lines.
    """
    allowed = list(engine.settings.get("allowed_actions") or [])
    lines = core_actions_menu_lines(engine.settings, allowed)
    try:
        lines += engine.actions.menu_lines(allowed)
    except Exception as exc:  # noqa: BLE001 — a menu is an enhancement, never a failure mode
        _logger.warning(f"[{VARIANT}] Could not describe the installed extra actions: {exc}")
    return "\n".join(lines)


def _decision_of(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse one turn's JSON decision, tolerating the shapes models actually emit.

    Purely defensive parsing (a list wrapper, a non-dict body) — no strategy, no rewriting of what
    the model chose. Returns ``None`` when the text was not JSON at all (the loop reports that back
    as an invalid step rather than crashing, and the telemetry stream records it as unparsed).
    """
    try:
        decision = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(decision, list):
        decision = next((item for item in decision if isinstance(item, dict)), {})
    return decision if isinstance(decision, dict) else {}


def _details_of(decision: Dict[str, Any]) -> Dict[str, Any]:
    """The action arguments, under either key models use (``details`` / ``args``)."""
    for key in ("details", "args"):
        value = decision.get(key)
        if isinstance(value, dict):
            return value
    return {}


#: Result keys worth showing the model, in the order they read best. Deliberately generic so an
#: action added by a tool pack (``run_python``'s ``stdout``, ...) renders without special-casing.
_OBSERVATION_KEYS = (
    "url", "content", "thinking_content", "synthesized", "verdict", "confidence", "quote",
    "supporting_url", "contradicting_url", "reasoning", "stdout", "stderr", "output", "count",
)


def _format_search_results(results: Any, limit: int) -> str:
    lines = []
    for index, item in enumerate(results or [], 1):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"{index}. {item.get('title','')} — {item.get('url','')}\n"
            f"   {item.get('description','')}"
        )
    return ("\n".join(lines) or "(no results)")[:limit]


def _observation(result: Optional[Dict[str, Any]], limit: int) -> str:
    """Render an action result as the observation fed back to the model.

    ``None`` means the guarded call timed out or the node carried no action — reported as the
    failure it is, never smoothed over.
    """
    if result is None:
        return "NO RESULT (the action failed, timed out, or was not runnable)."
    if not isinstance(result, dict):
        return str(result)[:limit]
    if not result.get("success"):
        return f"FAILED: {result.get('error') or 'unknown error'}"[:limit]
    parts: List[str] = []
    if result.get("results"):
        parts.append(_format_search_results(result.get("results"), limit))
    for key in _OBSERVATION_KEYS:
        value = result.get(key)
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}: {value}")
    return ("\n".join(parts) or "(action succeeded with no output)")[:limit]


async def _run_discretion(
    engine: IdeaDagEngine,
    graph: IdeaDag,
    mandate: str,
    model_name: str,
    max_steps: int,
) -> str:
    """The loop: ask for one action, run it through the engine's own dispatch, feed back what
    happened, repeat until the model finishes or the turns run out.

    :returns: The model's final answer, or ``""`` when it never called ``finish``.
    """
    obs_chars = int(os.environ.get("IDEA_TEST_NAIVE_OBS_CHARS", "6000"))
    step_max_tokens = int(os.environ.get("IDEA_TEST_NAIVE_STEP_MAX_TOKENS", "4096"))
    system = _SYSTEM_TEMPLATE.format(
        menu=_build_action_menu(engine), finish_line=_FINISH_MENU_LINE, max_steps=max_steps,
    )
    scratchpad: List[str] = []

    for step in range(max_steps):
        history = "\n\n".join(scratchpad[-12:]) if scratchpad else "(no actions yet)"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"TASK:\n{mandate}\n\nSCRATCHPAD (your prior steps):\n{history}\n\n"
                "Return the next step as JSON."
            )},
        ]
        payload = engine.io.build_llm_payload(
            messages=messages, json_mode=True, model_name=model_name,
            temperature=0.1, max_tokens=step_max_tokens,
        )
        raw = await engine.io.query_llm(payload, model_name=model_name)
        parsed = _decision_of(raw)
        _json_telemetry.record(model_name, raw, True, parsed is not None, phase=VARIANT)
        decision = parsed or {}

        action = str(decision.get("action", "")).strip().lower()
        details = _details_of(decision)
        thought = str(decision.get("thought", ""))[:300]

        if action == FINISH_ACTION:
            # No rescue: whatever the model put in `answer` IS the deliverable, empty included.
            return str(details.get("answer", ""))

        if not action:
            obs = "INVALID STEP: no action named. Return JSON with an 'action' from the list."
        else:
            node = graph.add_child(
                graph.root_id(),
                (thought or f"step {step + 1}: {action}")[:200],
                details={DetailKey.ACTION.value: action, **details},
            )
            # The engine's own guarded dispatch: registry resolution, the `allowed_actions` gate
            # (an action this run does not permit silently degrades to THINK, exactly as in a
            # graph run), tool-failure retry and the per-action timeout watchdog.
            result = await engine._execute_action_guarded(graph, graph.root_id(), node.node_id)
            obs = _observation(result, obs_chars)

        scratchpad.append(
            f"STEP {step + 1}: thought={thought}\n"
            f"action={action} details={json.dumps(details, default=str)[:200]}\n"
            f"observation={obs}"
        )

    # Out of turns without a `finish`: an empty deliverable, reported as the failure it is.
    return ""


async def run_naive_discretion_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    idea_settings: Optional[Dict[str, Any]] = None,
    summarize_observability_func=summarize_observability,
    connector_browser=None,
) -> Dict[str, Any]:
    """Run the naive-discretion agent; same return shape as ``run_sequential_execution``.

    :param idea_settings: The run's DAG settings. Passed straight to the engine, which derives
        ``allowed_actions`` and installs the configured tool packs from ``ToolsConfig`` itself —
        so a tool-ablation run is a settings change here, nothing else. ``None`` means the
        shipped defaults (the engine merges them).
    :param connector_browser: Optional headless-Chrome fallback connector (F18), wired uniformly
        with the other variants so no arm is handicapped by bot-blocked sites.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_{VARIANT}_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}_{VARIANT}.jsonl"
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
    engine = IdeaDagEngine(io=agent_io, settings=idea_settings, model_name=model_name)
    graph = IdeaDag(root_title=mandate)

    # Deliberately GENEROUS, and deliberately NOT budget-matched to another arm (unlike
    # `sequential_react`'s 25, which was reverse-engineered from `graph_compiled`'s effective
    # budget). What this arm lacks is engineered structure, not turns — a starved floor would
    # confound "no structure" with "no budget".
    max_steps = int(os.environ.get("IDEA_TEST_NAIVE_MAX_STEPS", "40"))
    started = time.perf_counter()
    deliverable = ""
    try:
        deliverable = await _run_discretion(engine, graph, mandate, model_name, max_steps)
    except Exception as exc:
        _logger.error(f"Naive-discretion run failed: {exc}", exc_info=True)

    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": VARIANT,
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
        # A real graph, unlike the linear/compiled arms' `_empty_graph()`: every action ran as a
        # node under the root, so the reporting/visualization path sees what was actually done.
        "graph": graph.to_dict(),
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
