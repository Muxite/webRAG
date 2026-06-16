#!/usr/bin/env python3
"""
Per-cell summary of cross-shape benchmark results, by run-id prefix(es).

Reads ``services/agent/idea_test_results/<prefix>_*.json`` and prints, per model, a
test x variant table of mean score (and a matching mean-USD table). ``graph_compiled`` cells
are split into ``compiled:hand`` vs ``compiled:auto`` from each run's recorded ``plan_source``,
so B-hand and B-auto are distinguishable even when they share the variant name.

Usage::

    ./.venv/bin/python scripts/gate_report.py --run-id gate2_flash --prefixes xshape_20260615_161408
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

_RES = Path(__file__).resolve().parent.parent / "services" / "agent" / "idea_test_results"


def _label(d: dict) -> str:
    v = d.get("execution_variant", "?")
    if v == "graph_compiled":
        return "compiled:" + (d.get("execution", {}).get("output", {}).get("plan_source") or "?")
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="primary run-id filename prefix")
    ap.add_argument("--prefixes", default="", help="comma-separated extra run-id prefixes to merge")
    args = ap.parse_args()

    prefixes = [args.run_id] + [p.strip() for p in args.prefixes.split(",") if p.strip()]
    files: list[str] = []
    for p in prefixes:
        files += [f for f in glob.glob(str(_RES / f"{p}_*.json")) if not f.endswith("_summary.json")]

    cells: dict[tuple, list] = defaultdict(list)
    for f in sorted(set(files)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tid = d.get("test_metadata", {}).get("test_id", "?")
        model = d.get("model", "?")
        obs = d.get("execution", {}).get("observability", {})
        cells[(model, tid, _label(d))].append((
            d.get("validation", {}).get("overall_score"),
            obs.get("visit", {}).get("count"),
            obs.get("cost", {}).get("usd"),
        ))

    if not cells:
        print(f"no result files for prefixes={prefixes}")
        return 1

    models = sorted({k[0] for k in cells})
    tests = sorted({k[1] for k in cells})
    labels = sorted({k[2] for k in cells})

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None

    for model in models:
        print(f"\n=== {model} ===")
        print("SCORE (mean, n)")
        print("test  " + "".join(f"{l:>18}" for l in labels))
        for t in tests:
            row = f"{t:<6}"
            for l in labels:
                vals = cells.get((model, t, l))
                if not vals:
                    row += f"{'-':>18}"
                else:
                    m = _mean([v[0] for v in vals])
                    row += f"{(f'{m:.2f}' if m is not None else 'na'):>13}(n{len(vals)})"
            print(row)
        print("RUNTIME USD (mean)")
        print("test  " + "".join(f"{l:>18}" for l in labels))
        for t in tests:
            row = f"{t:<6}"
            for l in labels:
                vals = cells.get((model, t, l))
                if not vals:
                    row += f"{'-':>18}"
                else:
                    m = _mean([v[2] for v in vals])
                    row += f"{(f'${m:.4f}' if m is not None else 'na'):>18}"
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
