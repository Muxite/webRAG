"""Env-gated contract-outcome log — what a leaf PROMISED vs what it delivered.

No-op unless ``IDEA_TEST_CONTRACT_LOG`` is truthy. When enabled, three JSONL event
types are appended to one file (``{run_id}_contract_log.jsonl``):

* ``contract_made`` — a leaf was created carrying an ``expect`` contract;
* ``contract_resolved`` — that leaf completed, and ``contract_satisfaction``'s
  deterministic check has (or has not) a verdict on it;
* ``plan_library_search`` — one retrieval attempt against the plan library.

The ``expect`` contract is checked once and discarded today
(``idea_policies/contract_satisfaction.py`` feeds the re-expansion gate and nothing
else), so there is no persistent record of WHICH contracts fail — exactly the corpus
needed to author targeted plan-library content later. Every event carries the node's
``ancestor_ids`` so multilevel blame ("this parent's contract failed because THIS
child's did") is reconstructed OFFLINE by a later analysis pass, with no real-time
blame propagation in the engine.

Sibling of :mod:`json_telemetry` and deliberately built the same way: pure
observability, so it is env-gated rather than wired into the JSON-settings/typed-config
chain reserved for behavior-changing knobs, and it shares that module's ``set_task``
so one call in the runner tags both streams. Best-effort by construction: any error in
here is swallowed, and every ``record_*`` returns on its first line when the flag is
off, so telemetry can never perturb a run or a unit test.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent.app.idea_policies.base import DetailKey
from agent.app.testing import json_telemetry as _json_telemetry

# testing/ -> app/ -> agent/ ; results live at agent/idea_test_results/
_RES = Path(__file__).resolve().parents[2] / "idea_test_results"

# How many retrieval matches to keep per ``plan_library_search`` row. The decision only
# ever looks at the top few; an unbounded list would bloat the log for no analysis value.
_MAX_MATCHES = 10

# Re-exported so a caller that stamps the task id once (``runner.run_complete_test``)
# tags this stream too — same global, never a second copy that can drift.
set_task = _json_telemetry.set_task


def enabled() -> bool:
    return os.environ.get("IDEA_TEST_CONTRACT_LOG", "") not in ("", "0", "false", "False")


def _run_id() -> str:
    return os.environ.get("IDEA_TEST_RUN_ID", "adhoc")


def _path() -> Path:
    override = os.environ.get("IDEA_TEST_CONTRACT_LOG_PATH")
    if override:
        return Path(override)
    return _RES / f"{_run_id()}_contract_log.jsonl"


def _event(name: str) -> Dict[str, Any]:
    return {
        "event": name,
        "t": round(time.time(), 3),
        "run_id": _run_id(),
        "task_id": _json_telemetry.current_task(),
    }


def _write(entry: Dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _lineage(graph: Any, node: Any) -> Dict[str, Any]:
    """The node's id plus its ancestry, nearest-first, up to the root.

    ``graph.path_to_root`` starts AT the node (dropped here — the event already
    identifies it) and walks the first parent, which is how every other consumer reads a
    merge node's lineage. ``parent_ids`` keeps the extra parents of a multi-parent merge
    node, since a merge's failure can trace to any of its branches.
    """
    node_id = getattr(node, "node_id", None)
    parents = list(getattr(node, "parent_ids", None) or [])
    parent_id = getattr(node, "parent_id", None)
    if not parents and parent_id:
        parents = [parent_id]
    return {
        "node_id": node_id,
        "parent_id": parents[0] if parents else None,
        "parent_ids": parents,
        "ancestor_ids": [n.node_id for n in graph.path_to_root(node_id)][1:],
    }


def _contract_fields(node: Any) -> Dict[str, Any]:
    details = getattr(node, "details", {}) or {}
    return {
        "expect": details.get(DetailKey.EXPECT.value),
        "goal": (
            details.get(DetailKey.GOAL.value)
            or details.get(DetailKey.ORIGINAL_GOAL.value)
            or getattr(node, "title", "")
        ),
        "action": details.get(DetailKey.ACTION.value),
    }


def record_made(
    graph: Any,
    node: Any,
    *,
    origin: str = "organic",
    template_id: Optional[str] = None,
) -> None:
    """A leaf was created WITH a contract (``origin`` says who authored it)."""
    if not enabled():
        return
    try:
        entry = _event("contract_made")
        entry.update(_lineage(graph, node))
        entry.update(_contract_fields(node))
        entry["origin"] = origin
        entry["template_id"] = template_id
        _write(entry)
    except Exception:  # noqa: BLE001 — telemetry must never affect a run
        pass


def record_resolved(
    graph: Any,
    node: Any,
    verdict: Any,
    *,
    step_index: Optional[int] = None,
    origin: str = "organic",
    template_id: Optional[str] = None,
) -> None:
    """A leaf completed — log what ``contract_satisfaction`` made of it.

    ``verdict`` is a ``ContractSatisfaction``, or ``None`` for "no verdict" (the leaf
    retrieved nothing checkable, or its contract text yields neither a measurable datum
    nor an identity token). The no-verdict case is logged rather than dropped: a
    contract nothing can check is itself a finding, and one row per completion keeps the
    offline join with ``contract_made`` unambiguous.
    """
    if not enabled():
        return
    try:
        entry = _event("contract_resolved")
        entry.update(_lineage(graph, node))
        entry.update(_contract_fields(node))
        applicable = bool(getattr(verdict, "applicable", False))
        entry.update({
            "step_index": step_index,
            "origin": origin,
            "template_id": template_id,
            "applicable": applicable,
            "satisfied": bool(getattr(verdict, "satisfied", False)) if applicable else None,
            "missing": list(getattr(verdict, "missing", None) or []),
            "reason": getattr(verdict, "reason", "") or ("no verdict" if verdict is None else ""),
        })
        _write(entry)
    except Exception:  # noqa: BLE001 — telemetry must never affect a run
        pass


def _normalize_matches(matches: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """``[{template_id, similarity}, ...]`` from dicts or ``(id, similarity)`` pairs."""
    out: List[Dict[str, Any]] = []
    for match in list(matches or [])[:_MAX_MATCHES]:
        if isinstance(match, dict):
            similarity = match.get("similarity")
            if similarity is None:
                similarity = match.get("score")
            out.append({
                "template_id": match.get("template_id"),
                "similarity": similarity,
                "archetype": match.get("archetype"),
            })
        elif isinstance(match, (tuple, list)) and len(match) >= 2:
            out.append({"template_id": match[0], "similarity": match[1], "archetype": None})
        else:
            out.append({"template_id": match, "similarity": None, "archetype": None})
    return out


def record_search(
    *,
    mode: str,
    query: str,
    matches: Optional[Sequence[Any]] = None,
    adopted: bool = False,
    adopted_template_id: Optional[str] = None,
    threshold: Optional[float] = None,
    node_id: Optional[str] = None,
    decision: Optional[str] = None,
) -> None:
    """One plan-library retrieval attempt.

    ``mode`` is the call site (``"auto_pre_expansion"`` / ``"on_demand"``). Logged
    whether or not anything was adopted — a retrieval that correctly found nothing is as
    informative as a hit when the library's coverage is being audited offline.
    """
    if not enabled():
        return
    try:
        entry = _event("plan_library_search")
        entry.update({
            "mode": mode,
            "node_id": node_id,
            "query": query,
            "matches": _normalize_matches(matches),
            "adopted": bool(adopted),
            "adopted_template_id": adopted_template_id,
            "threshold": threshold,
            "decision": decision,
        })
        _write(entry)
    except Exception:  # noqa: BLE001 — telemetry must never affect a run
        pass
