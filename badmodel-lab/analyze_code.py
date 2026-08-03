#!/usr/bin/env python3
"""Sibling to analyze.py, for the coding-agent benchmark (badmodel-lab/codebench/) instead
of the QA benchmark. Reads codebench/results/runs.jsonl (score_and_record.py's row schema —
see that script's module docstring for why it's a sibling schema, not a literal extension
of cells.jsonl) and reports the badmodel-vs-aider comparison per (model, task_category).

Usage: ./.venv/bin/python badmodel-lab/analyze_code.py [--results-file PATH]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_RESULTS_FILE = Path(__file__).resolve().parent / "codebench" / "results" / "runs.jsonl"


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def fmt(x, nd=3, pref=""):
    if x is None:
        return "  n/a"
    if isinstance(x, float):
        return f"{pref}{x:.{nd}f}"
    return f"{pref}{x}"


def print_agent_kind_comparison(rows: list[dict]) -> None:
    """Per (model, task_category): badmodel vs aider mean score / keystone-pass rate /
    mean tests_passed-of-total, plus the delta. This is the report the whole benchmark
    exists to produce — does the compiled-scaffold agent beat the conventional baseline on
    the SAME weak model?"""
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["model"], r["task_category"])][r["agent_kind"]].append(r)

    print("=" * 96)
    print(f"{'model':<16} {'category':<6} {'agent':<9} {'n':>3}  {'score':>8}  {'keystone':>9}  {'tests':>12}")
    print("-" * 96)
    for (model, category), by_agent in sorted(groups.items()):
        for agent_kind in ("badmodel", "aider"):
            cell_rows = by_agent.get(agent_kind, [])
            if not cell_rows:
                continue
            n = len(cell_rows)
            mean_score = _mean([r.get("score") for r in cell_rows])
            keystone_rate = _mean([r.get("keystone_pass") for r in cell_rows])
            tp = sum(r.get("tests_passed") or 0 for r in cell_rows)
            tt = sum(r.get("tests_total") or 0 for r in cell_rows)
            tests_str = f"{tp}/{tt}" if tt else "  n/a"
            print(f"{model:<16} {category:<6} {agent_kind:<9} {n:>3}  "
                  f"{fmt(mean_score):>8}  {fmt(keystone_rate):>9}  {tests_str:>12}")

        badmodel_scores = [r.get("score") for r in by_agent.get("badmodel", [])]
        aider_scores = [r.get("score") for r in by_agent.get("aider", [])]
        bm_mean, ai_mean = _mean(badmodel_scores), _mean(aider_scores)
        if bm_mean is not None and ai_mean is not None:
            delta = bm_mean - ai_mean
            sign = "+" if delta >= 0 else ""
            print(f"{'':<16} {'':<6} {'delta':<9} {'':>3}  {sign}{delta:.3f} "
                  f"(badmodel {'beats' if delta > 0 else 'trails' if delta < 0 else 'ties'} aider)")
        print("-" * 96)


def print_unscored_warning(rows: list[dict]) -> None:
    unscored = [r for r in rows if r.get("score") is None]
    if unscored:
        by_reason = defaultdict(int)
        for r in unscored:
            by_reason[r.get("task_category", "unknown")] += 1
        print(f"\n{len(unscored)} row(s) recorded without a score (breakdown by category): "
              f"{dict(by_reason)}")
        if by_reason.get("soft"):
            print("  soft-task judging isn't wired yet (see score_and_record.py's docstring) "
                  "— these need code_rubric.py before they'll carry a score.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    args = ap.parse_args()

    rows = load_runs(args.results_file)
    if not rows:
        print(f"no rows in {args.results_file} yet")
        return 0

    print(f"loaded {len(rows)} rows from {args.results_file}\n")
    print_agent_kind_comparison(rows)
    print_unscored_warning(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
