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
import sys
from collections import defaultdict
from pathlib import Path
from math import sqrt
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common  # noqa: E402  (path insert must precede this import)

DEFAULT_REFERENCE = ["google/gemini-3.1-pro-preview", "openai/gpt-5", "google/gemini-2.5-pro"]
ENGINE_VARIANTS = {"graph", "sequential", "sequential_react", "graph_compiled"}
BASELINE_VARIANTS = {"parametric", "naive_rag", "minimal"}


def _results_dir() -> Path:
    return bench_common.results_dir()


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean score / cost / context per (model, variant, tier)."""
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["variant"], r["tier"])].append(r)
    out: List[Dict[str, Any]] = []
    for (model, variant, tier), bucket in groups.items():
        scores = [r["score"] for r in bucket]
        n = len(scores)
        # Sample stdev (n>=2) of the per-run score, and the 95% CI half-width of the mean
        # (normal approx, 1.96*sem). These turn a higher REPEATS into reportable error bars;
        # n<2 -> 0.0 (a single run has no spread).
        std = round(stdev(scores), 4) if n >= 2 else 0.0
        ci95 = round(1.96 * std / sqrt(n), 4) if n >= 2 else 0.0
        out.append({
            "model": model,
            "variant": variant,
            "tooling": bucket[0].get("tooling", bench_common._VARIANT_TO_TOOLING.get(variant, variant)),
            "tier": tier,
            "n": n,
            "score": round(mean(scores), 4),
            "score_std": std,
            "score_ci95": ci95,
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
        w = csv.DictWriter(fh, fieldnames=["model", "variant", "tooling", "tier", "n", "score", "score_std", "score_ci95", "usd", "visit_chars", "estimated"])
        w.writeheader()
        for row in agg:
            w.writerow(row)


MIN_SIDE_PX = 1920  # exported plots must be square and at least this on a side


def _plot(agg: List[Dict[str, Any]], lines: Dict[str, Dict[str, Any]], reference_models: List[str],
          out: Path, side_px: int = MIN_SIDE_PX, dpi: int = 160) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] matplotlib unavailable: {exc}", file=sys.stderr)
        return False

    # House style (dataviz skill): fixed categorical palette + fonts that scale with the
    # canvas so this plot reads the same at 1920px or the gallery's 3840px 4K standard.
    # Falls back to a plain tab10 + fixed fonts if the agent package isn't on the path
    # (this script must keep working standalone, without PYTHONPATH=services:services/agent).
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "agent"))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services"))
        from agent.app.testing import plot_style
        # Axes fill the frame; a tight legend band sits directly under the x-axis label.
        fig, ax, fs = plot_style.square_fig(side_px, dpi, margins=(0.13, 0.965, 0.935, 0.185))
        palette = plot_style.CATEGORICAL
        mk = plot_style.mark_sizes(side_px)
    except Exception:
        plot_style = None
        side_px = max(MIN_SIDE_PX, int(side_px))
        inches = side_px / float(dpi)
        fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
        fig.subplots_adjust(left=0.13, right=0.965, top=0.935, bottom=0.185)
        fs = {"title": 19, "label": 16, "tick": 13, "legend": 12, "annot": 12}
        palette = [plt.get_cmap("tab10")(i) for i in range(10)]
        mk = {"line": 2.4, "marker": 8, "scatter": 120, "edge": 1.4, "cap": 5, "eline": 1.4}

    def _short(model: str) -> str:  # drop the provider prefix so the legend fits
        return model.split("/")[-1]

    side_px = max(MIN_SIDE_PX, int(side_px))
    cheap = [a for a in agg if a["model"] not in reference_models]
    models = sorted({a["model"] for a in cheap})
    # Colours pulled from the well-separated interior of the magma family.
    model_color = {m: palette[(i * 3) % len(palette)] for i, m in enumerate(models)}

    for model in models:
        color = model_color[model]
        graph_pts = sorted([a for a in cheap if a["model"] == model and a["variant"] in ENGINE_VARIANTS],
                           key=lambda a: a["usd"])
        if graph_pts:
            xs = [p["usd"] for p in graph_pts]
            ys = [p["score"] for p in graph_pts]
            yerr = [p.get("score_ci95", 0.0) for p in graph_pts]
            ax.errorbar(xs, ys, yerr=yerr, fmt="-o", color=color, lw=mk["line"], ms=mk["marker"],
                        capsize=mk["cap"], elinewidth=mk["eline"], markeredgecolor="white",
                        markeredgewidth=mk["edge"], label=f"{_short(model)} (webRAG)", zorder=5)
            best = max(graph_pts, key=lambda a: a["score"])  # annotate each model's best point
            ax.annotate(f"{best['score']:.2f}", (best["usd"], best["score"]),
                        textcoords="offset points", xytext=(0, 22), ha="center",
                        fontsize=fs["annot"], color=color, fontweight="bold", zorder=6)
        base_pts = [a for a in cheap if a["model"] == model and a["variant"] in BASELINE_VARIANTS]
        if base_pts:
            ax.scatter([p["usd"] for p in base_pts], [p["score"] for p in base_pts],
                       marker="X", color=color, s=mk["scatter"], linewidths=mk["edge"],
                       edgecolors="white", label=f"{_short(model)} (baseline)", zorder=4)

    def _tiny(model: str) -> str:  # ultra-short model tag so reference labels stay narrow
        s = _short(model)
        return s.replace("-preview", "").replace("gemini-3.1-pro", "gemini-pro")

    if "premium_raw" in lines:
        ln = lines["premium_raw"]
        ax.axhline(ln["score"], linestyle="--", color="#8a8880", linewidth=mk["eline"] * 1.5,
                   label=f"premium raw ({_tiny(ln['model'])}) = {ln['score']:.2f}")
    if "premium_webrag" in lines:
        ln = lines["premium_webrag"]
        ax.axhline(ln["score"], linestyle="-.", color="#0b0b0b", linewidth=mk["eline"] * 1.5,
                   label=f"premium+webRAG ({_tiny(ln['model'])}) = {ln['score']:.2f}")

    frontier = _pareto(agg)
    if frontier:
        ax.plot([p["usd"] for p in frontier], [p["score"] for p in frontier], ":",
                color="#b5179e", linewidth=mk["line"], alpha=0.85, label="Pareto frontier", zorder=3)

    ax.set_xlabel("Realized cost (USD, mean per run, log scale)", fontsize=fs["label"])
    ax.set_ylabel("Quality (mean validation score)", fontsize=fs["label"])
    ax.set_title("Cost recovery: cheap + webRAG vs premium",
                 fontsize=fs["title"] * 0.9, fontweight="bold", pad=18)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=fs["tick"])
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.margins(y=0.06)
    # Legend below the axes in two columns so a busy roster stays inside the frame.
    ax.legend(fontsize=fs["legend"] * 0.86, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=True, borderaxespad=0.0, columnspacing=1.4, handletextpad=0.5)
    if plot_style is not None:
        plot_style.savefig_square(fig, out, dpi=dpi)
    else:
        fig.savefig(out, dpi=dpi)
    print(f"Wrote plot -> {out} ({side_px}x{side_px}px)")
    return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Recovery-curve & Pareto analysis")
    parser.add_argument("--since", default="", help="Only files with name prefix >= this")
    parser.add_argument("--run-id", dest="run_id", default="",
                        help="Only files from this exact run_id (the YYYYMMDD_HHMMSS prefix). "
                             "Preferred over --since: isolates one run with no cross-run pollution.")
    parser.add_argument("--files", nargs="*", help="Explicit result files")
    parser.add_argument("--reference-models", default=",".join(DEFAULT_REFERENCE))
    parser.add_argument("--tests", default="", help="Comma-separated test_ids to include (default all)")
    parser.add_argument("--out", default="", help="Plot PNG output path")
    parser.add_argument("--csv", default="", help="CSV output path")
    parser.add_argument("--size", type=int, default=MIN_SIDE_PX,
                        help=f"square plot side in px (clamped to >= {MIN_SIDE_PX})")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    results_dir = _results_dir()
    # Empty run_ids list => no run-id scoping (matches every file, filtered only by
    # --since / --files) -- this preserves recovery_curve's long-standing "analyze
    # everything unless told otherwise" default. Pass --run-id barrage24b explicitly
    # (or use render_gallery.py, which defaults to it) to scope to the final dataset.
    run_ids = [args.run_id] if args.run_id else []
    files = bench_common.discover_files(run_ids, since=args.since, files=args.files)
    if not files:
        print(f"No result files found (dir={results_dir}, run_id={args.run_id!r}, since={args.since!r})",
              file=sys.stderr)
        return 2

    rows = [r for r in (bench_common.load_row(p) for p in files)
            if r and r["model"] and r["variant"] and r.get("usd") is not None]
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
    print(f"{'Model':<34} {'Variant':<11} {'Tier':>5} {'n':>3} {'Score':>7} {'±Std':>6} {'±CI95':>7} {'USD':>10} {'CtxChars':>9}")
    for a in agg:
        est = "*" if a["estimated"] else " "
        print(f"{a['model']:<34} {a['variant']:<11} {a['tier']:>5} {a['n']:>3} {a['score']:>7.3f} {a['score_std']:>6.3f} {a['score_ci95']:>7.3f} {a['usd']:>9.5f}{est} {a['visit_chars']:>9}")

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
    _plot(agg, lines, reference_models, out_path, side_px=args.size, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
