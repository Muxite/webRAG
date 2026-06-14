"""
Recovery-curve & Pareto analysis for the cost-recovery benchmark.

Reads per-run result JSON files (the ``*_r*.json`` files written by
idea_test_runner) and answers the headline question: can a cheap model + webRAG
reach a premium model's quality at a fraction of the dollar cost?

It aggregates by (model, variant, effort_tier), then:
- writes a CSV of every aggregated point (quality vs USD vs context size),
- prints the Pareto frontier (best quality per dollar), and
- renders a recovery-curve plot: quality vs realized USD per cheap model, with two
  horizontal reference lines (premium-raw and premium+webRAG).

Usage:
  python3 scripts/recovery_curve.py                       # all result files
  python3 scripts/recovery_curve.py --since 20260614      # files with prefix >= 20260614
  python3 scripts/recovery_curve.py --reference-models google/gemini-3.1-pro,openai/gpt-5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REFERENCE = ["google/gemini-3.1-pro-preview", "openai/gpt-5", "google/gemini-2.5-pro"]
ENGINE_VARIANTS = {"graph", "sequential"}
BASELINE_VARIANTS = {"parametric", "naive_rag"}


def _results_dir() -> Path:
    for cand in (Path("services/agent/idea_test_results"), Path("agent/idea_test_results")):
        if cand.is_dir():
            return cand
    return Path("services/agent/idea_test_results")


def _load_row(path: Path) -> Optional[Dict[str, Any]]:
    """Extract the fields we need from one result file, or None if unusable."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    execution = d.get("execution") or {}
    obs = execution.get("observability") or {}
    cost = obs.get("cost") or {}
    validation = d.get("validation") or {}
    meta = d.get("test_metadata") or {}
    score = validation.get("overall_score")
    usd = cost.get("usd")
    if score is None or usd is None:
        return None
    return {
        "model": d.get("model"),
        "variant": d.get("execution_variant"),
        "tier": int(d.get("effort_tier") or 0),
        "test_id": meta.get("test_id"),
        "score": float(score),
        "usd": float(usd),
        "cost_estimated": bool(cost.get("estimated")),
        "total_tokens": int((obs.get("llm") or {}).get("total_tokens") or 0),
        "visit_chars": int((obs.get("visit") or {}).get("chars") or 0),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean score / cost / context per (model, variant, tier)."""
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["variant"], r["tier"])].append(r)
    out: List[Dict[str, Any]] = []
    for (model, variant, tier), bucket in groups.items():
        out.append({
            "model": model,
            "variant": variant,
            "tier": tier,
            "n": len(bucket),
            "score": round(mean(r["score"] for r in bucket), 4),
            "usd": round(mean(r["usd"] for r in bucket), 6),
            "visit_chars": int(mean(r["visit_chars"] for r in bucket)),
            "estimated": any(r["cost_estimated"] for r in bucket),
        })
    out.sort(key=lambda a: (a["model"], a["variant"], a["tier"]))
    return out


def _pareto(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Points where no other point has >= score at <= cost (upper-left frontier)."""
    frontier: List[Dict[str, Any]] = []
    for p in points:
        dominated = any(
            q is not p and q["usd"] <= p["usd"] and q["score"] >= p["score"]
            and (q["usd"] < p["usd"] or q["score"] > p["score"])
            for q in points
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(key=lambda a: a["usd"])
    return frontier


def _reference_lines(agg: List[Dict[str, Any]], reference_models: List[str]) -> Dict[str, Dict[str, Any]]:
    """Best premium raw baseline and best premium+webRAG point, as reference bars."""
    lines: Dict[str, Dict[str, Any]] = {}
    refs = [a for a in agg if a["model"] in reference_models]
    raw = [a for a in refs if a["variant"] in BASELINE_VARIANTS]
    graph = [a for a in refs if a["variant"] in ENGINE_VARIANTS]
    if raw:
        best = max(raw, key=lambda a: a["score"])
        lines["premium_raw"] = best
    if graph:
        best = max(graph, key=lambda a: a["score"])
        lines["premium_webrag"] = best
    return lines


def _write_csv(agg: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "variant", "tier", "n", "score", "usd", "visit_chars", "estimated"])
        w.writeheader()
        for row in agg:
            w.writerow(row)


def _plot(agg: List[Dict[str, Any]], lines: Dict[str, Dict[str, Any]], reference_models: List[str], out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] matplotlib unavailable: {exc}", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(11, 7))
    cheap = [a for a in agg if a["model"] not in reference_models]
    models = sorted({a["model"] for a in cheap})
    cmap = plt.get_cmap("tab10")

    for idx, model in enumerate(models):
        color = cmap(idx % 10)
        graph_pts = sorted([a for a in cheap if a["model"] == model and a["variant"] in ENGINE_VARIANTS], key=lambda a: a["usd"])
        if graph_pts:
            ax.plot([p["usd"] for p in graph_pts], [p["score"] for p in graph_pts], "-o", color=color, label=f"{model} (webRAG)")
        base_pts = [a for a in cheap if a["model"] == model and a["variant"] in BASELINE_VARIANTS]
        if base_pts:
            ax.scatter([p["usd"] for p in base_pts], [p["score"] for p in base_pts], marker="x", color=color, s=60, label=f"{model} (baseline)")

    if "premium_raw" in lines:
        ln = lines["premium_raw"]
        ax.axhline(ln["score"], linestyle="--", color="gray", linewidth=1.2, label=f"premium-raw ({ln['model']}) = {ln['score']:.2f}")
    if "premium_webrag" in lines:
        ln = lines["premium_webrag"]
        ax.axhline(ln["score"], linestyle="-.", color="black", linewidth=1.2, label=f"premium+webRAG ({ln['model']}) = {ln['score']:.2f}")

    frontier = _pareto(agg)
    if frontier:
        ax.plot([p["usd"] for p in frontier], [p["score"] for p in frontier], ":", color="red", linewidth=1.0, alpha=0.7, label="Pareto frontier")

    ax.set_xlabel("Realized cost (USD, mean per run)")
    ax.set_ylabel("Quality (mean validation score)")
    ax.set_title("Cost-recovery curve: cheap model + webRAG vs premium reference")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Wrote plot -> {out}")
    return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Recovery-curve & Pareto analysis")
    parser.add_argument("--since", default="", help="Only files with name prefix >= this")
    parser.add_argument("--files", nargs="*", help="Explicit result files")
    parser.add_argument("--reference-models", default=",".join(DEFAULT_REFERENCE))
    parser.add_argument("--tests", default="", help="Comma-separated test_ids to include (default all)")
    parser.add_argument("--out", default="", help="Plot PNG output path")
    parser.add_argument("--csv", default="", help="CSV output path")
    args = parser.parse_args(argv)

    results_dir = _results_dir()
    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = sorted(
            p for p in results_dir.glob("*_r*.json")
            if "_report_" not in p.name and (not args.since or p.name >= args.since)
        )
    if not files:
        print(f"No result files found (dir={results_dir}, since={args.since!r})", file=sys.stderr)
        return 2

    rows = [r for r in (_load_row(p) for p in files) if r and r["model"] and r["variant"]]
    test_filter = {t.strip() for t in args.tests.split(",") if t.strip()}
    if test_filter:
        rows = [r for r in rows if r["test_id"] in test_filter]
    print(f"Loaded {len(rows)} priced rows from {len(files)} files in {results_dir}"
          + (f" (tests={sorted(test_filter)})" if test_filter else ""))
    if not rows:
        print("No rows had both a score and a USD cost — did you run with cost instrumentation?", file=sys.stderr)
        return 2

    reference_models = [m.strip() for m in args.reference_models.split(",") if m.strip()]
    agg = _aggregate(rows)
    lines = _reference_lines(agg, reference_models)

    csv_path = Path(args.csv) if args.csv else results_dir / "recovery_curve.csv"
    _write_csv(agg, csv_path)
    print(f"Wrote CSV  -> {csv_path}")

    print("\n=== Aggregated points (model / variant / tier) ===")
    print(f"{'Model':<34} {'Variant':<11} {'Tier':>5} {'n':>3} {'Score':>7} {'USD':>10} {'CtxChars':>9}")
    for a in agg:
        est = "*" if a["estimated"] else " "
        print(f"{a['model']:<34} {a['variant']:<11} {a['tier']:>5} {a['n']:>3} {a['score']:>7.3f} {a['usd']:>9.5f}{est} {a['visit_chars']:>9}")

    print("\n=== Pareto frontier (quality per dollar) ===")
    for a in _pareto(agg):
        print(f"  {a['model']:<34} {a['variant']:<11} tier={a['tier']:<4} score={a['score']:.3f} usd={a['usd']:.5f}")

    for key, ln in lines.items():
        print(f"\nReference [{key}]: {ln['model']} {ln['variant']} score={ln['score']:.3f} usd={ln['usd']:.5f}")
        crossers = [
            a for a in agg
            if a["model"] not in reference_models and a["variant"] in ENGINE_VARIANTS
            and a["score"] >= ln["score"] and a["usd"] < ln["usd"]
        ]
        for c in sorted(crossers, key=lambda a: a["usd"]):
            pct = 100.0 * c["usd"] / ln["usd"] if ln["usd"] else 0.0
            print(f"    CROSSES: {c['model']} tier={c['tier']} score={c['score']:.3f} at {pct:.0f}% of reference cost")

    out_path = Path(args.out) if args.out else results_dir / "recovery_curve.png"
    _plot(agg, lines, reference_models, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
