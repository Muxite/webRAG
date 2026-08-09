#!/usr/bin/env python3
"""
Level-ladder report for the layered agentic-web benchmark.

Groups per-cell results by taxonomy LEVEL (micro -> integration -> navigation -> graph)
and by execution variant, and prints the five metrics that matter at every level:
success rate, page visits, tool calls, token cost, time, and groundedness. This realizes
the "ladder from local choices to web-scale pathfinding" view.

Levels come from each test's get_test_metadata()["level"]; tests without a level (the
legacy 001-039 comparison barrage) are bucketed as "legacy". Weight (short/long) is also
reported so short and long tasks can be read separately.

Usage:
    python3 scripts/level_ladder.py --run-id 20260614_153511
    python3 scripts/level_ladder.py --since 20260614_1500
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common  # noqa: E402  (path insert must precede this import)

# Local (self-hosted) models have no meaningful $ price. Reuse the one shared predicate
# (agent.app.model_costs.is_local_row) rather than re-deriving "unpriced" locally, but
# degrade gracefully so this script keeps working standalone without
# PYTHONPATH=.:services:agent (same convention as recovery_curve.py's plot_style import).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.app.model_costs import is_local_row as _is_local_row_impl
except Exception:  # noqa: BLE001
    _is_local_row_impl = None

LEVEL_ORDER = ["micro", "integration", "navigation", "graph", "legacy"]


def _is_local(row: Dict[str, Any]) -> bool:
    if _is_local_row_impl is not None:
        return _is_local_row_impl(row)
    origin = row.get("origin")
    if origin is not None:
        return origin == "local"
    return row.get("usd") is None


def _results_dir() -> Path:
    return bench_common.results_dir()


def _load_row(p: Path) -> Optional[Dict[str, Any]]:
    """Thin adapter over bench_common's canonical row (same fields, level_ladder's names)."""
    return bench_common.load_row(p)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _fmt_usd(rows: List[Dict[str, Any]]) -> str:
    """Mean $ over priced (non-local) rows in a group.

    "local" if every row in the group is a local/free model (so it isn't shown as a
    misleading $0.00000), "n/a" if no row has a price for another reason (e.g. an
    unrecognized paid model with a pricing-table gap).
    """
    priced = [x["usd"] for x in rows if x.get("usd") is not None and not _is_local(x)]
    if priced:
        return f"{_mean(priced):.5f}"
    if rows and all(_is_local(x) for x in rows):
        return "local"
    return "n/a"


def _ci95(xs: List[float]) -> float:
    """95% CI half-width of the mean (normal approx, 1.96*SEM). 0 for n<2."""
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * stdev(xs) / math.sqrt(n)


def _cohens_d(a: List[float], b: List[float]) -> float:
    """Standardized effect size of mean(a)-mean(b). 0 when either group has n<2."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa, sb = stdev(a), stdev(b)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / max(1, na + nb - 2))
    return (mean(a) - mean(b)) / pooled if pooled else 0.0


def _significance_block(groups: Dict[tuple, List[Dict[str, Any]]], levels_present: List[str]) -> None:
    """Per-level head-to-head: graph_compiled vs each baseline. Reports Δmean, Cohen's d, and a
    conservative CI-disjoint verdict (the 95% CIs don't overlap => 'sig'). With REPEATS=3 the
    CIs are wide, so 'sig' here is a deliberately strict bar — far better than comparing bare
    means. Use a real paired test (more repeats + scipy) for a publication number."""
    baselines = ("sequential_react", "graph", "naive_rag", "parametric")
    print("\n=== Significance: graph_compiled vs baselines (per level) ===")
    print("CI-disjoint = 95% CIs don't overlap (strict). d = Cohen's effect size.\n")
    hdr = f"{'level':<12}{'vs':<18}{'Δscore':>9}{'cohen_d':>9}{'gc_n':>6}{'base_n':>7}{'verdict':>10}"
    print(hdr)
    print("-" * len(hdr))
    for lv in levels_present:
        gc = [x["score"] for x in groups.get((lv, "graph_compiled"), [])]
        if not gc:
            continue
        gc_m, gc_ci = _mean(gc), _ci95(gc)
        for base in baselines:
            b = [x["score"] for x in groups.get((lv, base), [])]
            if not b:
                continue
            b_m, b_ci = _mean(b), _ci95(b)
            disjoint = (gc_m - gc_ci > b_m + b_ci) or (b_m - b_ci > gc_m + gc_ci)
            verdict = "sig" if disjoint else "ns"
            print(f"{lv:<12}{base:<18}{gc_m - b_m:>+9.3f}{_cohens_d(gc, b):>9.2f}"
                  f"{len(gc):>6}{len(b):>7}{verdict:>10}")
        print()


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Level-ladder benchmark report")
    ap.add_argument("--run-id", dest="run_id", default="", help="Only files from this exact run_id prefix")
    ap.add_argument("--since", default="", help="Only files with name prefix >= this")
    args = ap.parse_args(argv)

    rd = _results_dir()
    # Empty run_ids => no run-id scoping (matches every file, filtered only by --since),
    # preserving level_ladder's long-standing "analyze everything unless told otherwise"
    # default. Pass --run-id barrage24b explicitly to scope to the final dataset.
    run_ids = [args.run_id] if args.run_id else []
    files = bench_common.discover_files(run_ids, since=args.since)
    rows = [r for r in (_load_row(p) for p in files) if r]
    if not rows:
        print(f"No result rows found (dir={rd}, run_id={args.run_id!r}, since={args.since!r})", file=sys.stderr)
        return 2

    # group by (level, variant)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["variant"])].append(r)

    print(f"Loaded {len(rows)} rows from {len(files)} files\n")
    hdr = (f"{'level':<12}{'variant':<18}{'tooling':<9}{'n':>3}{'success(±ci95)':>18}"
           f"{'visits':>8}{'tools':>7}{'usd':>10}{'secs':>8}{'grounded':>10}")
    print(hdr)
    print("-" * len(hdr))
    levels_present = [lv for lv in LEVEL_ORDER if any(k[0] == lv for k in groups)]
    for lv in levels_present:
        for variant in ("parametric", "minimal", "naive_rag", "sequential_react", "sequential", "graph", "graph_compiled"):
            g = groups.get((lv, variant))
            if not g:
                continue
            scores = [x["score"] for x in g]
            success = f"{_mean(scores):.3f}±{_ci95(scores):.3f}"
            print(f"{lv:<12}{variant:<18}{g[0].get('tooling', '-'):<9}{len(g):>3}"
                  f"{success:>18}"
                  f"{_mean([x['visits'] for x in g]):>8.1f}"
                  f"{_mean([x['tool_calls'] for x in g]):>7.1f}"
                  f"{_fmt_usd(g):>10}"
                  f"{_mean([x['secs'] for x in g]):>8.1f}"
                  f"{_mean([x['grounded'] for x in g]):>10.3f}")
        print()

    _significance_block(groups, levels_present)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
