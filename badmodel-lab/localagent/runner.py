"""Run a task through the orchestrator and compute the per-run metrics the ablation
study needs. Writes one JSONL trace line per run (the truth); analyze_agent.py rolls
these up. Latency is wall-clock (the ≤50× budget axis is a ratio vs a cheap-API run).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .actions import ActionRegistry
from .loop import RunResult, run_task
from .narrator import Narrator
from .state import AgentState
from .tools.base import ToolContext


def _action_tool_map(registry: ActionRegistry) -> Dict[str, str]:
    return {sp.name: sp.tool for sp in registry.all()}


def run_task_once(task, registry: ActionRegistry, tools, llm, *, workdir: Path, memory,
                  max_steps: int = 30, narrator: Optional[Narrator] = None, finalize_n: int = 1):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if task.setup:
        task.setup(workdir, memory)
    ctx = ToolContext(workdir=workdir, memory=memory)
    state = AgentState(task_goal=task.goal, enabled_groups=set(task.groups))
    # Pre-resolve known top-level files as entities so a weak model can act on them by id
    # (F1) immediately, instead of having to discover them via LIST_DIR first.
    for p in sorted(workdir.iterdir()):
        if p.is_file():
            state.add_entity("file", p.name, str(p.resolve()))
    t0 = time.time()
    res = run_task(state, registry, tools, llm, ctx, narrator=narrator, max_steps=max_steps,
                   finalize_n=finalize_n, finish_gate=getattr(task, "finish_gate", None))
    latency = round(time.time() - t0, 3)
    return res, ctx, latency


def metrics_for(task, res: RunResult, ctx: ToolContext, registry: ActionRegistry,
                *, model: str, rep: int, latency_s: float) -> Dict[str, Any]:
    steps = res.steps
    n = len(steps) or 1
    a2t = _action_tool_map(registry)
    valid = sum(1 for s in steps if s.action and not s.errors) / n
    first_pass = sum(1 for s in steps if s.action and not s.errors and s.repairs == 0) / n
    tool_sel = None
    if task.expected_tool:
        tool_sel = any(a2t.get(s.action) == task.expected_tool and s.ok for s in steps)
    verdicts = task.validate(res, ctx)
    return {
        "task_id": task.id, "model": model, "rep": rep,
        "finished": res.state.finished,
        "success": all(v.passed for v in verdicts),
        "verdicts": [{"passed": v.passed, "reason": v.reason} for v in verdicts],
        "n_steps": len(steps), "n_calls": res.state.budget.used_calls,
        "valid_action_rate": round(valid, 3),
        "first_pass_arg_validity": round(first_pass, 3),
        "tool_selection_ok": tool_sel,
        # tool-layer containment holds by construction (confine()); the OS-level
        # invariant is validated by the sandbox in P1. No path-escape here == True.
        "containment_ok": True,
        "latency_s": latency_s,
        "final_answer": (res.final_answer or "")[:300],
        "actions": [s.action for s in steps],
    }


def write_trace(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_traces(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out
