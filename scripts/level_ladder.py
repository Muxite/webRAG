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
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

LEVEL_ORDER = ["micro", "integration", "navigation", "graph", "legacy"]


def _results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "services" / "agent" / "idea_test_results"


def _load_row(p: Path) -> Optional[Dict[str, Any]]:
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    meta = j.get("test_metadata", {}) or {}
    ex = j.get("execution", {}) or {}
    val = j.get("validation", {}) or {}
    obs = ex.get("observability", {}) or {}
    fn = p.name
    # Prefer the explicit field on the result; fall back to filename heuristics for
    # legacy results that predate the field.
    variant = j.get("execution_variant") or (
        "minimal" if "_minimal" in fn else "graph_compiled" if "_graph_compiled" in fn
        else "graph" if "_graph" in fn
        else "naive_rag" if "naive_rag" in fn else "parametric" if "parametric" in fn
        else "sequential" if "_sequential" in fn else "graph")
    _v2t = {"minimal": "minimal", "naive_rag": "partial", "graph": "full",
            "sequential_react": "sequential", "graph_compiled": "compiled"}
    tooling = j.get("tooling_profile") or _v2t.get(variant, variant)
    score = val.get("overall_score")
    if score is None:
        return None
    # groundedness: prefer the engine's real grounding verdict (visited-page evidence);
    # fall back to a validation-check proxy for legacy results without it.
    grounded = None
    gflag = (obs.get("grounding", {}) or {}).get("grounded")
    if gflag is not None:
        grounded = 1.0 if gflag else 0.0
    else:
        grounded = 0.0
        for c in val.get("grep_validations", []) or []:
            name = str(c.get("check", "")).lower()
            if any(t in name for t in ("citation", "grounding", "source", "adjacency", "url")):
                grounded = max(grounded, float(c.get("score", 0.0)))
    return {
        "level": (meta.get("level") or "legacy"),
        "weight": (meta.get("weight") or "-"),
        "test_id": meta.get("test_id"),
        "variant": variant,
        "tooling": tooling,
        "score": float(score),
        "visits": int(obs.get("visit", {}).get("count", 0) or 0),
        "tool_calls": int(obs.get("llm", {}).get("calls", 0) or 0)
                      + int(obs.get("search", {}).get("count", 0) or 0)
                      + int(obs.get("visit", {}).get("count", 0) or 0),
        "usd": float((obs.get("cost", {}) or {}).get("usd") or 0.0),
        "secs": float(ex.get("duration_seconds") or 0.0),
        "grounded": grounded,
    }


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Level-ladder benchmark report")
    ap.add_argument("--run-id", dest="run_id", default="", help="Only files from this exact run_id prefix")
    ap.add_argument("--since", default="", help="Only files with name prefix >= this")
    args = ap.parse_args(argv)

    rd = _results_dir()
    run_prefix = f"{args.run_id}_" if args.run_id else ""
    files = sorted(
        p for p in rd.glob("*_r*.json")
        if "_report_" not in p.name
        and (not run_prefix or p.name.startswith(run_prefix))
        and (not args.since or p.name >= args.since)
    )
    rows = [r for r in (_load_row(p) for p in files) if r]
    if not rows:
        print(f"No result rows found (dir={rd}, run_id={args.run_id!r}, since={args.since!r})", file=sys.stderr)
        return 2

    # group by (level, variant)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["variant"])].append(r)

    print(f"Loaded {len(rows)} rows from {len(files)} files\n")
    hdr = f"{'level':<12}{'variant':<18}{'tooling':<9}{'n':>3}{'success':>9}{'visits':>8}{'tools':>7}{'usd':>10}{'secs':>8}{'grounded':>10}"
    print(hdr)
    print("-" * len(hdr))
    levels_present = [lv for lv in LEVEL_ORDER if any(k[0] == lv for k in groups)]
    for lv in levels_present:
        for variant in ("parametric", "minimal", "naive_rag", "sequential_react", "sequential", "graph", "graph_compiled"):
            g = groups.get((lv, variant))
            if not g:
                continue
            print(f"{lv:<12}{variant:<18}{g[0].get('tooling', '-'):<9}{len(g):>3}"
                  f"{_mean([x['score'] for x in g]):>9.3f}"
                  f"{_mean([x['visits'] for x in g]):>8.1f}"
                  f"{_mean([x['tool_calls'] for x in g]):>7.1f}"
                  f"{_mean([x['usd'] for x in g]):>10.5f}"
                  f"{_mean([x['secs'] for x in g]):>8.1f}"
                  f"{_mean([x['grounded'] for x in g]):>10.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
