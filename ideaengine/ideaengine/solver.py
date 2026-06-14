"""
The `Solver` abstraction.

A `Solver` accepts a research mandate and produces a deliverable. The Idea
Engine is one Solver implementation; the Phase 3 comparison harness will add
adapter Solvers for LangGraph and LangChain so all three sit behind the same
interface.

This module deliberately exposes a *narrow* contract built around
`engine.run()`. The test harness in `services/agent/app/testing/` currently
reaches into the engine's `step()` loop directly, so it is not yet plumbed
through `Solver`; that migration is part of Phase 3 of the extraction plan.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

try:
    from typing import NotRequired, TypedDict  # Python 3.11+
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

if TYPE_CHECKING:
    from ideaengine.idea_engine import IdeaDagEngine
    from ideaengine.telemetry import TelemetrySession


class SolverResult(TypedDict, total=False):
    """The shape every Solver returns.

    `final_deliverable`, `success`, and `observability` are required for the
    comparison harness; everything else is optional and filled in when the
    underlying solver can produce it. Non-DAG solvers (e.g. LangChain
    `AgentExecutor`) synthesize `observability` from their intermediate steps.
    """

    final_deliverable: str
    success: bool
    observability: Dict[str, Any]
    graph: NotRequired[Dict[str, Any]]
    token_usage: NotRequired[Dict[str, int]]
    cost_usd: NotRequired[float]
    wall_time_s: NotRequired[float]
    llm_calls: NotRequired[int]
    search_calls: NotRequired[int]
    visit_calls: NotRequired[int]
    goal_achieved: NotRequired[bool]
    has_failures: NotRequired[bool]
    warning: NotRequired[str]


class Solver(Protocol):
    """A mandate-to-deliverable runner.

    Implementations should be stateless across `solve` calls. State (memory,
    checkpoints) lives behind injected collaborators.
    """

    name: str

    async def solve(
        self,
        mandate: str,
        *,
        max_steps: int,
        settings: Dict[str, Any],
        telemetry: Optional["TelemetrySession"] = None,
    ) -> SolverResult: ...


class IdeaEngineSolver:
    """Wraps `IdeaDagEngine.run()` in the `Solver` interface.

    Used by the comparison harness (Phase 3) to drive ideaengine alongside
    LangGraph and LangChain adapters with the same call signature. The wrapper
    does NOT own the engine's collaborators (LLM, search, vector store) —
    those are baked into the engine at construction.
    """

    name = "ideaengine"

    def __init__(self, engine: "IdeaDagEngine") -> None:
        self._engine = engine

    async def solve(
        self,
        mandate: str,
        *,
        max_steps: int = 50,
        settings: Optional[Dict[str, Any]] = None,
        telemetry: Optional["TelemetrySession"] = None,
        run_id: Optional[str] = None,
    ) -> SolverResult:
        if settings:
            # Settings precedence is on the engine: caller-provided overrides
            # win for keys they specify; the rest fall back to the engine's
            # construction-time settings.
            self._engine.settings.update(settings)

        raw = await self._engine.run(mandate, max_steps=max_steps, run_id=run_id)
        return _normalize_engine_result(raw)


def _normalize_engine_result(raw: Dict[str, Any]) -> SolverResult:
    """Map `IdeaDagEngine.run()`'s dict to the `SolverResult` shape."""
    result: SolverResult = {
        "final_deliverable": str(raw.get("final_deliverable") or ""),
        "success": bool(raw.get("success", False)),
        "observability": _observability_from_engine_result(raw),
    }

    if "graph" in raw:
        result["graph"] = raw["graph"]
    if "goal_achieved" in raw:
        result["goal_achieved"] = bool(raw["goal_achieved"])
    if "has_failures" in raw:
        result["has_failures"] = bool(raw["has_failures"])
    if "warning" in raw and raw["warning"]:
        result["warning"] = str(raw["warning"])
    return result


def _observability_from_engine_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize the `observability` dict that test validators consume.

    Validators expect counts like `observability["visit"]["count"]`. We mine
    those from the serialized graph if present; otherwise return an empty
    skeleton that validators can read without crashing.
    """
    obs: Dict[str, Dict[str, Any]] = {
        "visit": {"count": 0},
        "search": {"count": 0},
        "think": {"count": 0},
        "save": {"count": 0},
    }
    graph = raw.get("graph") or {}
    nodes = graph.get("nodes") or {}
    for node in nodes.values() if isinstance(nodes, dict) else nodes:
        if not isinstance(node, dict):
            continue
        details = node.get("details") or {}
        action = details.get("action")
        if action in obs:
            obs[action]["count"] = obs[action]["count"] + 1
    if isinstance(raw.get("got_stats"), dict):
        obs["got_stats"] = raw["got_stats"]
    return obs
