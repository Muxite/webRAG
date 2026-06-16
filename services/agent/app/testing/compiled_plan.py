"""
Compiled-plan schema v2 — the DAG.

A compiled plan is the offline-authored scaffold the ``graph_compiled`` arm executes: a set
of leaves (each resolving ONE fact with web tools) plus an aggregation recipe. Schema v1 was
a flat *parallel* leaf set. Schema v2 generalizes it to a **DAG**: a leaf may declare
``depends_on: [ids]`` and template an upstream leaf's resolved fact into its own instruction
via ``{dep_id}`` placeholders. This lets one mechanism cover both task shapes:

  * independent fan-out (no deps)          -> all leaves in one parallel wave   (e.g. test 052)
  * dependent chain (each hop needs prior) -> one leaf per wave, chained         (e.g. test 051)
  * mixed                                  -> parallel waves + dependent tails   (e.g. test 054)

Backward-compatible: a plan whose leaves declare no ``depends_on`` topologically reduces to a
single wave — exactly the v1 parallel behavior — so existing hand-authored plans are unchanged.

This module is pure (no I/O, no LLM): it normalizes, validates (structure + cycle/missing-dep
detection), computes the topological execution waves, and substitutes upstream results into a
dependent leaf's instruction. The executor (``execution_compiled.py``) and the offline compiler
(``scaffold_compiler.py``) both build on it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_DEFAULT_AGGREGATION = "Combine the gathered facts into a final answer; cite source URLs."


class PlanValidationError(ValueError):
    """Raised when a compiled plan is structurally invalid (bad leaf, missing dep, cycle)."""


def normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized copy of ``plan``: every leaf has ``id``/``instruction``/``expect``/
    ``depends_on`` (deps defaulted to ``[]``), and a non-empty ``aggregation``.

    Does NOT validate the dependency graph — call :func:`validate_plan` for that. Normalization
    is forgiving so a slightly-shaped hand or compiler plan still runs (ids are slugged from the
    instruction when absent); validation is where we get strict.
    """
    if not isinstance(plan, dict):
        raise PlanValidationError(f"plan must be a dict, got {type(plan).__name__}")
    raw_leaves = plan.get("leaves") or []
    if not isinstance(raw_leaves, list):
        raise PlanValidationError("plan['leaves'] must be a list")

    leaves: List[Dict[str, Any]] = []
    for idx, leaf in enumerate(raw_leaves):
        if not isinstance(leaf, dict):
            raise PlanValidationError(f"leaf #{idx} must be a dict, got {type(leaf).__name__}")
        instruction = str(leaf.get("instruction", "")).strip()
        leaf_id = str(leaf.get("id", "")).strip() or _slug(instruction) or f"leaf_{idx}"
        deps_raw = leaf.get("depends_on") or []
        if isinstance(deps_raw, str):
            deps_raw = [deps_raw]
        if not isinstance(deps_raw, list):
            raise PlanValidationError(f"leaf '{leaf_id}': depends_on must be a list")
        depends_on = [str(d).strip() for d in deps_raw if str(d).strip()]
        leaves.append({
            "id": leaf_id,
            "instruction": instruction,
            "expect": str(leaf.get("expect", "")).strip(),
            "depends_on": depends_on,
        })

    aggregation = str(plan.get("aggregation") or "").strip() or _DEFAULT_AGGREGATION
    return {"leaves": leaves, "aggregation": aggregation}


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize then strictly validate a plan; return the normalized plan or raise
    :class:`PlanValidationError`.

    Checks: at least one leaf; unique non-empty leaf ids; every ``depends_on`` id exists; no
    leaf depends on itself; the dependency graph is acyclic.
    """
    norm = normalize_plan(plan)
    leaves = norm["leaves"]
    if not leaves:
        raise PlanValidationError("plan has no leaves")

    ids = [leaf["id"] for leaf in leaves]
    seen: set = set()
    for leaf_id in ids:
        if not leaf_id:
            raise PlanValidationError("a leaf has an empty id")
        if leaf_id in seen:
            raise PlanValidationError(f"duplicate leaf id '{leaf_id}'")
        seen.add(leaf_id)

    id_set = set(ids)
    for leaf in leaves:
        for dep in leaf["depends_on"]:
            if dep == leaf["id"]:
                raise PlanValidationError(f"leaf '{leaf['id']}' depends on itself")
            if dep not in id_set:
                raise PlanValidationError(f"leaf '{leaf['id']}' depends on unknown leaf '{dep}'")

    # Cycle detection falls out of computing the waves (Kahn's algorithm).
    topological_waves(leaves)
    return norm


def topological_waves(leaves: List[Dict[str, Any]]) -> List[List[str]]:
    """Group leaf ids into ordered execution **waves** (Kahn's algorithm).

    Wave 0 holds every leaf with no dependencies; wave k holds leaves whose deps are all
    satisfied by waves < k. Leaves within a wave are mutually independent and run in parallel.
    Within a wave, ids preserve the plan's declared order (stable, so output/diffs are
    deterministic). Raises :class:`PlanValidationError` if the graph has a cycle.
    """
    indegree: Dict[str, int] = {leaf["id"]: 0 for leaf in leaves}
    dependents: Dict[str, List[str]] = {leaf["id"]: [] for leaf in leaves}
    order = [leaf["id"] for leaf in leaves]  # declared order, for stable wave membership
    for leaf in leaves:
        # de-dupe deps so a leaf listing the same dep twice doesn't inflate indegree
        for dep in dict.fromkeys(leaf["depends_on"]):
            indegree[leaf["id"]] += 1
            dependents[dep].append(leaf["id"])

    waves: List[List[str]] = []
    remaining = dict(indegree)
    placed: set = set()
    while len(placed) < len(order):
        wave = [lid for lid in order if lid not in placed and remaining[lid] == 0]
        if not wave:
            unresolved = [lid for lid in order if lid not in placed]
            raise PlanValidationError(f"dependency cycle among leaves: {unresolved}")
        for lid in wave:
            placed.add(lid)
            for child in dependents[lid]:
                remaining[child] -= 1
        waves.append(wave)
    return waves


def substitute_deps(instruction: str, dep_results: Dict[str, str]) -> str:
    """Substitute resolved upstream facts into a dependent leaf's instruction.

    Only ``{dep_id}`` placeholders whose ``dep_id`` is a key in ``dep_results`` are replaced
    (one-level substitution). Any other braces in the instruction are left untouched, so we
    never choke on stray ``{`` characters the way ``str.format`` would.
    """
    if not dep_results:
        return instruction
    out = instruction
    for dep_id, value in dep_results.items():
        out = out.replace("{" + dep_id + "}", str(value))
    return out


def plan_structure(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, answer-free structural summary of a plan — for logging and auto-vs-hand diffing.

    Returns leaf count, edge count, the wave shape (ids per wave) and the sorted edge list, so a
    compiler-authored plan can be compared to a hand-authored one without dumping instructions.
    """
    norm = normalize_plan(plan)
    leaves = norm["leaves"]
    edges = sorted(
        (dep, leaf["id"]) for leaf in leaves for dep in dict.fromkeys(leaf["depends_on"])
    )
    try:
        waves = topological_waves(leaves)
    except PlanValidationError:
        waves = []
    return {
        "leaf_count": len(leaves),
        "edge_count": len(edges),
        "waves": waves,
        "wave_widths": [len(w) for w in waves],
        "edges": [f"{a}->{b}" for a, b in edges],
        "is_dag_chain": bool(waves) and all(len(w) == 1 for w in waves) and len(waves) > 1,
        "is_pure_fanout": len(waves) == 1,
    }


def _slug(text: str) -> str:
    """Lowercase kebab/underscore slug from free text (first ~6 words), for fallback leaf ids."""
    words = re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split()
    return "_".join(words[:6]).strip("_")
