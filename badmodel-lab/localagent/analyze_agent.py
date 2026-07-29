"""Ablation analyzer — rolls up the JSONL run traces into the capability-floor +
reliability metrics. Same honesty discipline as badmodel-lab/analyze.py: success is
reported with a 95% Wilson lower bound (a point estimate over few reps over-states).

Usage:  ./.venv/bin/python badmodel-lab/localagent/analyze_agent.py results/agent_traces.jsonl
        [--latency-baseline results/nano_traces.jsonl]   # per-task cheap-API denominator
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .runner import read_traces

LATENCY_BAR = 50.0    # local run is "bad" if > 50x the cheap-API baseline (per task)


def wilson_lo(k: int, n: int, z: float = 1.95996) -> float:
    if not n:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (c - m) / d


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.mean(xs) if xs else float("nan")


def _baseline_latency(rows) -> dict:
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r.get("latency_s"))
    return {t: _mean(v) for t, v in by_task.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", help="agent run traces JSONL")
    ap.add_argument("--latency-baseline", default=None, help="cheap-API traces for the 50x ratio")
    args = ap.parse_args()

    rows = read_traces(Path(args.traces))
    if not rows:
        print("no traces found.")
        return 1
    base = _baseline_latency(read_traces(Path(args.latency_baseline))) if args.latency_baseline else {}

    grp = defaultdict(list)
    for r in rows:
        grp[(r["model"], r["task_id"])].append(r)

    print("=" * 108)
    print("CAPABILITY-FLOOR + RELIABILITY  (succ=task success; Lo=95% Wilson lower; "
          "valid=valid-action; 1st=first-pass args; tool=tool-selection)")
    print(f"{'model':<15}{'task':<16}{'n':>3}{'succ%':>6}{'Lo':>6}{'valid%':>7}{'1st%':>6}"
          f"{'tool%':>6}{'steps':>6}{'lat_s':>7}{'xAPI':>6}")
    floor = defaultdict(dict)
    for (model, task), rs in sorted(grp.items()):
        n = len(rs)
        k = sum(1 for r in rs if r.get("success"))
        lo = wilson_lo(k, n)
        valid = _mean([r.get("valid_action_rate") for r in rs])
        first = _mean([r.get("first_pass_arg_validity") for r in rs])
        toolv = [r.get("tool_selection_ok") for r in rs if r.get("tool_selection_ok") is not None]
        tool = (sum(1 for t in toolv if t) / len(toolv)) if toolv else float("nan")
        steps = _mean([r.get("n_steps") for r in rs])
        lat = _mean([r.get("latency_s") for r in rs])
        ratio = (lat / base[task]) if base.get(task) else float("nan")
        floor[task][model] = (k / n, lo)
        xapi = f"{ratio:.0f}x" + ("!" if (ratio == ratio and ratio > LATENCY_BAR) else "")
        print(f"{model:<15}{task:<16}{n:>3}{100*k/n:>6.0f}{lo:>6.2f}{100*valid:>7.0f}"
              f"{100*first:>6.0f}{(f'{100*tool:.0f}' if tool==tool else '-'):>6}"
              f"{steps:>6.1f}{lat:>7.2f}{(xapi if ratio==ratio else '-'):>6}")

    # containment must be zero violations, ever
    violations = [r for r in rows if not r.get("containment_ok", True)]
    print("\n" + "=" * 108)
    print(f"CONTAINMENT VIOLATIONS: {len(violations)} (MUST be 0)")

    print("\nCAPABILITY FLOOR per task (smallest model whose Wilson lower bound clears 0.75):")
    for task, models in sorted(floor.items()):
        confirmed = [m for m, (p, lo) in models.items() if lo >= 0.75]
        print(f"  {task:<16} -> {', '.join(confirmed) if confirmed else '(none confirmed at this n)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
