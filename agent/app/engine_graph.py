"""
Minimal LangGraph-style state graph for the GoT engine.

Goals: declarative routing between async functions, with conditional edges
that can branch based on a state dict. Homegrown (no dependency) and kept
small — under ~250 LOC — so we don't undercut the project's efficiency
selling point.

Typical use:

    g = EngineStateGraph()
    g.add_node("expand", expand_fn)
    g.add_node("evaluate", evaluate_fn)
    g.add_conditional_edge(
        "expand",
        router_fn,
        {"ok": "evaluate", "empty": EngineStateGraph.END},
    )
    g.add_edge("evaluate", EngineStateGraph.END)
    g.set_entry("expand")
    runnable = g.compile()
    final_state = await runnable.invoke(initial_state)

Each node function receives the mutable state dict and returns an updated
state dict. Conditional routers receive the state and return a key from
the routes mapping.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict, Union


class GraphPhase(str, Enum):
    """
    Canonical engine phases. Optional — callers can use any string in state["phase"].
    """
    EXPAND = "expand"
    EVALUATE = "evaluate"
    SELECT = "select"
    EXECUTE = "execute"
    MERGE = "merge"
    PRUNE = "prune"
    DONE = "done"


class EngineState(TypedDict, total=False):
    """
    Recommended shape for engine state passed between graph nodes.
    All fields optional; nodes can add their own keys freely.
    """
    current_node_id: Optional[str]
    step_index: int
    phase: str
    pending_count: int
    last_action_result: Optional[Dict[str, Any]]


NodeFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
RouterFn = Callable[[Dict[str, Any]], Union[str, Awaitable[str]]]


class EngineStateGraph:
    """
    Declarative graph of async nodes with optional conditional edges.
    """

    END = "__end__"
    START = "__start__"

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeFn] = {}
        self._edges: Dict[str, str] = {}
        self._conditional: Dict[str, tuple[RouterFn, Dict[str, str]]] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, fn: NodeFn) -> "EngineStateGraph":
        """
        Register an async function as a graph node.

        :param name: Node identifier.
        :param fn: Async callable taking state and returning new state.
        :returns: Self, for chaining.
        """
        if name in (self.END, self.START):
            raise ValueError(f"Reserved node name: {name}")
        if name in self._nodes:
            raise ValueError(f"Duplicate node: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, from_node: str, to_node: str) -> "EngineStateGraph":
        """
        Unconditional edge.

        :param from_node: Source node name.
        :param to_node: Target node name or EngineStateGraph.END.
        :returns: Self, for chaining.
        """
        if from_node in self._conditional:
            raise ValueError(f"Node {from_node} already has a conditional edge")
        self._edges[from_node] = to_node
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        router: RouterFn,
        routes: Dict[str, str],
    ) -> "EngineStateGraph":
        """
        Conditional edge: router returns a key; routes maps key -> next node.

        :param from_node: Source node name.
        :param router: Function (sync or async) taking state, returning route key.
        :param routes: Mapping of route key -> next node name (or END).
        :returns: Self, for chaining.
        """
        if from_node in self._edges:
            raise ValueError(f"Node {from_node} already has an unconditional edge")
        if not routes:
            raise ValueError("routes mapping must be non-empty")
        self._conditional[from_node] = (router, dict(routes))
        return self

    def set_entry(self, node_name: str) -> "EngineStateGraph":
        """
        Set the starting node.

        :param node_name: Name of entry node.
        :returns: Self, for chaining.
        """
        self._entry = node_name
        return self

    def compile(self, max_steps: int = 200) -> "CompiledGraph":
        """
        Validate the graph and return a runnable.

        :param max_steps: Hard cap on hops per invoke() to detect infinite loops.
        :returns: Compiled runnable.
        :raises ValueError: When the graph references undeclared nodes.
        """
        if self._entry is None:
            raise ValueError("Entry node not set")
        if self._entry not in self._nodes:
            raise ValueError(f"Entry references unknown node: {self._entry}")
        for src, dst in self._edges.items():
            if src not in self._nodes:
                raise ValueError(f"Unknown source node: {src}")
            if dst != self.END and dst not in self._nodes:
                raise ValueError(f"Edge {src}->{dst} references unknown node")
        for src, (_, routes) in self._conditional.items():
            if src not in self._nodes:
                raise ValueError(f"Unknown source node: {src}")
            for key, dst in routes.items():
                if dst != self.END and dst not in self._nodes:
                    raise ValueError(f"Conditional {src}[{key}]->{dst} references unknown node")
        for name in self._nodes:
            if name not in self._edges and name not in self._conditional:
                raise ValueError(f"Node {name} has no outgoing edge")
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional=dict(self._conditional),
            entry=self._entry,
            max_steps=max_steps,
        )


class CompiledGraph:
    """
    Runnable form of an EngineStateGraph. Use invoke() to execute.
    """

    def __init__(
        self,
        nodes: Dict[str, NodeFn],
        edges: Dict[str, str],
        conditional: Dict[str, tuple[RouterFn, Dict[str, str]]],
        entry: str,
        max_steps: int,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry = entry
        self._max_steps = max_steps

    async def _resolve_route(self, name: str, state: Dict[str, Any]) -> Optional[str]:
        if name in self._edges:
            return self._edges[name]
        if name in self._conditional:
            router, routes = self._conditional[name]
            outcome = router(state)
            if hasattr(outcome, "__await__"):
                outcome = await outcome  # type: ignore[assignment]
            key = str(outcome)
            if key not in routes:
                raise RuntimeError(f"Router for {name} returned unmapped key {key!r}")
            return routes[key]
        return None

    async def invoke(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the graph starting at the entry node.

        :param state: Initial state dict (mutable, returned at end).
        :returns: Final state dict after END is reached.
        :raises RuntimeError: On exceeded max_steps or routing failures.
        """
        current: Optional[str] = self._entry
        state = dict(state or {})
        hops = 0
        visited_trace: List[str] = []
        while current is not None and current != EngineStateGraph.END:
            if hops >= self._max_steps:
                raise RuntimeError(
                    f"EngineStateGraph exceeded max_steps={self._max_steps}; trace={visited_trace[-10:]}"
                )
            hops += 1
            visited_trace.append(current)
            node_fn = self._nodes[current]
            updated = await node_fn(state)
            if updated is not None:
                state = updated
            current = await self._resolve_route(current, state)
        state["_trace"] = visited_trace
        return state
