#!/usr/bin/env python3
"""
Render the barrage24b LinkedIn-post gallery: every square 4K PNG (Pareto, heatmap,
scatter x2, trend-by-level, DAG examples) plus the combined-stats CSVs they're built
from, all from the ``barrage24b`` cost-recovery benchmark (already on disk -- $0, no
live model calls).

Usage:
    PYTHONPATH=.:services:agent python3 scripts/render_gallery.py
    PYTHONPATH=.:services:agent python3 scripts/render_gallery.py --size 3840 \
        --extra-run-ids 073fix_validate,080fix_validate   # once those exist

Everything lands in agent/idea_test_results/barrage24b_gallery/.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "services"))
sys.path.insert(0, str(_HERE.parent / "agent"))
sys.path.insert(0, str(_HERE.parent))

import bench_common  # noqa: E402
import recovery_curve  # noqa: E402
from agent.app.testing import plot_style  # noqa: E402
from agent.app.testing.dag_visualizer import render_plan_dag  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANT_ORDER = ["graph_compiled", "sequential_react", "graph", "naive_rag", "parametric", "minimal"]
LEVEL_ORDER = ["micro", "integration", "navigation", "graph", "legacy"]
PLANS_DIR = _HERE.parent / "agent" / "compiled_plans"
# Representative cached compiled plans (same roster render_dag_examples.py uses) --
# a pure fan-out and a mixed dependent+independent DAG, the two shapes that best show
# off the compiled scaffold's structure.
DAG_EXAMPLES = {
    # The widest pure fan-out in the plan cache: 16 independent leaves (8 Wikipedia topics ×
    # 2 facts) in a single wave, 0 dependency edges, all rolling up into one Markdown-table
    # aggregation. Pairs with the mixed DAG below to show the two structural extremes.
    "dag_example_fanout": ("da5f8b81fb1fbd42.json",
                           "Pure fan-out DAG — 8 topics × 2 facts = 16 leaves (test 038)",
                           "Extract two facts from each of 8 Wikipedia pages into a fact matrix"),
    # A genuinely complex compiled plan: 12 leaves across two topological waves with 6
    # data-dependency edges (each birth-year leaf depends on its author leaf). This is the
    # kind of multi-wave DAG the compiled scaffold authors offline -- far more representative
    # than a single-wave fan-out.
    "dag_example_mixed": ("bcef73fe20dd1509.json",
                          "Two-wave dependency DAG — 6 authors → 6 birth-years (argmin, test 052)",
                          "Which of 6 novels has the earliest-born author?"),
}


def _short_model(model: str) -> str:
    return (model or "").split("/")[-1]


def _tiny_model(model: str) -> str:
    """Ultra-short model tag for cramped legends (keeps 9-series rosters inside the frame)."""
    return _short_model(model).replace("gemini-3.1-pro-preview", "gemini-pro")


def _ci95(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    return 1.96 * stdev(xs) / sqrt(n)


def _variant_sort_key(v: str) -> int:
    return VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER)


def _combined_stats(rows: List[Dict[str, Any]], out_dir: Path) -> Tuple[Path, Path]:
    """Write the 'final combined-stats dataset': every raw row, plus a
    (model, variant, tier) aggregate -- the single artifact this gallery (and anyone
    downstream) can point at instead of re-deriving from the 810+ individual JSONs."""
    raw_path = out_dir / "combined_stats_raw.csv"
    fields = ["path", "test_id", "level", "weight", "model", "variant", "tooling", "tier",
              "score", "usd", "cost_estimated", "total_tokens", "visit_chars", "visits",
              "llm_calls", "search_calls", "visit_calls", "tool_calls", "secs", "grounded"]
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["test_id"] or "", r["model"] or "", r["variant"] or "")):
            w.writerow({k: r.get(k) for k in fields})

    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["variant"], r["tier"])].append(r)
    agg_path = out_dir / "combined_stats_agg.csv"
    agg_fields = ["model", "variant", "tooling", "tier", "n", "score_mean", "score_ci95",
                  "usd_mean", "visits_mean", "tokens_mean", "grounded_mean"]
    with agg_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=agg_fields)
        w.writeheader()
        for (model, variant, tier), bucket in sorted(groups.items()):
            scores = [b["score"] for b in bucket]
            usds = [b["usd"] for b in bucket if b["usd"] is not None]
            w.writerow({
                "model": model, "variant": variant, "tooling": bucket[0].get("tooling"), "tier": tier,
                "n": len(bucket), "score_mean": round(mean(scores), 4), "score_ci95": round(_ci95(scores), 4),
                "usd_mean": round(mean(usds), 6) if usds else "",
                "visits_mean": round(mean(b["visits"] for b in bucket), 2),
                "tokens_mean": round(mean(b["total_tokens"] for b in bucket), 1),
                "grounded_mean": round(mean(b["grounded"] for b in bucket), 3),
            })
    print(f"Wrote combined stats -> {raw_path} ({len(rows)} rows), {agg_path} ({len(groups)} groups)")
    return raw_path, agg_path


def _agg_by(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple, Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, bucket in groups.items():
        scores = [b["score"] for b in bucket]
        usds = [b["usd"] for b in bucket if b["usd"] is not None]
        out[k] = {
            "n": len(bucket), "score": mean(scores), "score_ci95": _ci95(scores),
            "usd": mean(usds) if usds else None,
            "visits": mean(b["visits"] for b in bucket),
        }
    return out


def plot_heatmap(rows: List[Dict[str, Any]], out_path: Path, side_px: int, dpi: int) -> None:
    """test_id (rows) x (model, variant) columns -- mean score on the magma ramp.

    Continuous 0..1 magnitude -> magma (dark = low, bright yellow = high/best). Rows are
    sorted by the mean graph_compiled score across models (descending) so the strongest
    tasks for the compiled scaffold sit at the top and the hardest sink to the bottom --
    the vertical gradient itself tells the story. Direct value labels stay on every cell,
    white on the dark low end and near-black on the bright high end.
    """
    by_test_col = _agg_by(rows, ("test_id", "model", "variant"))
    cols = sorted({(r["model"], r["variant"]) for r in rows},
                  key=lambda c: (_short_model(c[0]), _variant_sort_key(c[1])))
    col_labels = [f"{_short_model(m)}\n{v}" for m, v in cols]

    # Sort rows by mean compiled score (across whatever models ran graph_compiled), desc.
    def _compiled_key(tid: str) -> float:
        vals = [c["score"] for (t, m, v), c in by_test_col.items()
                if t == tid and v == "graph_compiled"]
        return mean(vals) if vals else -1.0

    test_ids = sorted({r["test_id"] for r in rows}, key=lambda t: (-_compiled_key(t), t))

    mat = np.full((len(test_ids), len(cols)), np.nan)
    for i, tid in enumerate(test_ids):
        for j, (m, v) in enumerate(cols):
            cell = by_test_col.get((tid, m, v))
            if cell:
                mat[i, j] = cell["score"]

    fig, ax, fs = plot_style.square_fig(side_px, dpi, margins=(0.135, 0.9, 0.9, 0.185), style=False)
    cmap = plot_style.sequential_cmap()
    cmap.set_bad(color="#d8d6cf")
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=fs["tick"] * 0.72,
                       color=plot_style.INK_PRIMARY)
    ax.set_yticks(range(len(test_ids)))
    ax.set_yticklabels(test_ids, fontsize=fs["tick"] * 0.62, color=plot_style.INK_SECONDARY)
    ax.set_xlabel("Model · execution variant", fontsize=fs["label"], color=plot_style.INK_PRIMARY)
    ax.set_ylabel("Test ID  (sorted by graph_compiled score ↓)", fontsize=fs["label"] * 0.9,
                  color=plot_style.INK_PRIMARY)
    ax.set_title("barrage24b — score by test × model · variant",
                 fontsize=fs["title"] * 0.92, fontweight="bold", pad=18, color=plot_style.INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(test_ids), 1), minor=True)
    ax.grid(which="minor", color=plot_style.SURFACE, linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)

    # Direct value labels; readable on both ends of the magma ramp.
    for i in range(len(test_ids)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "#111111" if v >= 0.62 else "#f5f5f5"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=fs["annot"] * 0.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean validation score", fontsize=fs["label"] * 0.85, color=plot_style.INK_PRIMARY)
    cbar.ax.tick_params(labelsize=fs["tick"] * 0.8, colors=plot_style.INK_SECONDARY)

    plot_style.savefig_square(fig, out_path, dpi=dpi)
    print(f"Wrote plot -> {out_path} ({side_px}x{side_px}px)")


def _scatter(rows: List[Dict[str, Any]], x_key: str, x_label: str, title: str,
             out_path: Path, side_px: int, dpi: int, log_x: bool = False) -> None:
    """cost/visits vs score -- color by variant (fixed categorical order), marker by model.
    Point-level (per model, variant, test_id) so the spread across the 27+ tests is visible,
    not just the per-(model,variant) mean the Pareto plot already shows."""
    by_test_col = _agg_by(rows, ("test_id", "model", "variant"))
    variants = sorted({r["variant"] for r in rows}, key=_variant_sort_key)
    models = sorted({r["model"] for r in rows}, key=_short_model)
    markers = ["o", "s", "^", "D", "P", "X"]
    mk = plot_style.mark_sizes(side_px)

    fig, ax, fs = plot_style.square_fig(side_px, dpi, margins=(0.115, 0.97, 0.93, 0.205))

    for vi, variant in enumerate(variants):
        color = plot_style.categorical_color(vi)
        for mi, model in enumerate(models):
            xs, ys = [], []
            for (tid, m, v), cell in by_test_col.items():
                if m == model and v == variant and cell.get(x_key) is not None:
                    xs.append(cell[x_key])
                    ys.append(cell["score"])
            if not xs:
                continue
            ax.scatter(xs, ys, color=color, marker=markers[mi % len(markers)], s=mk["scatter"],
                       alpha=0.85, edgecolors="white", linewidths=mk["edge"],
                       label=f"{variant} · {_tiny_model(model)}")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=fs["label"])
    ax.set_ylabel("Validation score (per test, mean of R repeats)", fontsize=fs["label"] * 0.92)
    ax.set_title(title, fontsize=fs["title"], fontweight="bold", pad=18)
    ax.set_ylim(-0.04, 1.06)
    ax.tick_params(axis="both", labelsize=fs["tick"])
    ncol = 3 if len(variants) * len(models) >= 8 else 2
    ax.legend(fontsize=fs["legend"] * 0.72, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=ncol, frameon=True, borderaxespad=0.0, columnspacing=1.0, handletextpad=0.4)
    plot_style.savefig_square(fig, out_path, dpi=dpi)
    print(f"Wrote plot -> {out_path} ({side_px}x{side_px}px)")


def plot_trend_by_level(rows: List[Dict[str, Any]], out_path: Path, side_px: int, dpi: int) -> None:
    """Per-level success rate (±CI95) bars, grouped by variant -- the PNG counterpart of
    level_ladder.py's text table."""
    by_level_variant = _agg_by(rows, ("level", "variant"))
    levels = [lv for lv in LEVEL_ORDER if any(k[0] == lv for k in by_level_variant)]
    variants = sorted({r["variant"] for r in rows}, key=_variant_sort_key)
    mk = plot_style.mark_sizes(side_px)

    fig, ax, fs = plot_style.square_fig(side_px, dpi, margins=(0.115, 0.97, 0.915, 0.165))
    n_v = len(variants)
    width = 0.82 / n_v
    x = np.arange(len(levels))

    for vi, variant in enumerate(variants):
        color = plot_style.categorical_color(vi)
        means, errs = [], []
        for lv in levels:
            cell = by_level_variant.get((lv, variant))
            means.append(cell["score"] if cell else 0.0)
            errs.append(cell["score_ci95"] if cell else 0.0)
        offset = (vi - (n_v - 1) / 2) * width
        ax.bar(x + offset, means, width * 0.9, yerr=errs, capsize=mk["cap"] * 0.7, color=color,
               edgecolor="white", linewidth=mk["edge"], label=variant,
               error_kw={"elinewidth": mk["eline"], "ecolor": plot_style.INK_SECONDARY})

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=fs["tick"])
    ax.set_ylabel("Success (mean validation score, ±95% CI)", fontsize=fs["label"] * 0.9)
    ax.set_title("barrage24b — success by level",
                 fontsize=fs["title"] * 0.96, fontweight="bold", pad=18)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="both", labelsize=fs["tick"])
    ax.legend(fontsize=fs["legend"] * 0.82, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              ncol=min(len(variants), 5), frameon=True, borderaxespad=0.0, columnspacing=1.1,
              handletextpad=0.4)
    plot_style.savefig_square(fig, out_path, dpi=dpi)
    print(f"Wrote plot -> {out_path} ({side_px}x{side_px}px)")


def _work_agg(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple, Dict[str, float]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return {k: {
        "llm": mean(b["llm_calls"] for b in bucket),
        "search": mean(b["search_calls"] for b in bucket),
        "visit": mean(b["visit_calls"] for b in bucket),
        "n": len(bucket),
    } for k, bucket in groups.items()}


def plot_work_by_variant(rows: List[Dict[str, Any]], out_path: Path, side_px: int, dpi: int) -> None:
    """How much work does each execution strategy actually do? Mean LLM inference calls,
    web searches, and page visits per run, grouped by execution variant. Single axis
    (all three channels are operation counts, so they share a scale honestly -- no dual
    axis). A caption band under the plot carries the takeaway, using the square canvas as
    a wide-plot-plus-text layout rather than a cramped 1:1 axes."""
    by_variant = _work_agg(rows, ("variant",))
    variants = sorted(by_variant.keys(), key=lambda k: _variant_sort_key(k[0]))
    _short_variant = {"graph_compiled": "compiled", "sequential_react": "sequential"}
    labels = [_short_variant.get(k[0], k[0]) for k in variants]
    mk = plot_style.mark_sizes(side_px)
    fs = plot_style.font_sizes(side_px)

    side_px = max(plot_style.MIN_SIDE_PX, int(side_px))
    inches = side_px / float(dpi)
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    fig.set_facecolor(plot_style.SURFACE)
    # Wide bar plot on top (~68%), caption panel on the bottom (~24%).
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 0.28], left=0.1, right=0.97,
                          top=0.9, bottom=0.05, hspace=0.28)
    ax = fig.add_subplot(gs[0])
    cap = fig.add_subplot(gs[1])
    plot_style.style_axes(ax)

    channels = [("llm", "LLM inference calls"), ("search", "Web searches"), ("visit", "Page visits")]
    ch_colors = [plot_style.categorical_color(0), plot_style.categorical_color(3),
                 plot_style.categorical_color(6)]
    x = np.arange(len(variants))
    width = 0.82 / len(channels)
    for ci, (key, name) in enumerate(channels):
        vals = [by_variant[v][key] for v in variants]
        offset = (ci - (len(channels) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, color=ch_colors[ci], edgecolor="white",
                      linewidth=mk["edge"], label=name)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + max(vals) * 0.01 + 0.2, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=fs["annot"] * 0.62,
                    color=plot_style.INK_SECONDARY, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs["tick"] * 0.82)
    ax.set_ylabel("Mean count per run", fontsize=fs["label"])
    ax.set_title("barrage24b — work per execution strategy",
                 fontsize=fs["title"] * 0.9, fontweight="bold", pad=18, color=plot_style.INK_PRIMARY)
    ax.tick_params(axis="both", labelsize=fs["tick"])
    ax.margins(y=0.18)
    ax.legend(fontsize=fs["legend"], loc="upper right", frameon=True, ncol=1)

    # Caption panel: a plain-language takeaway (averaged across all 3 models).
    cap.axis("off")
    comp = by_variant.get(("graph_compiled",), {})
    seq = by_variant.get(("sequential_react",), {})
    lines = [
        "Reading the bars",
        f"• graph_compiled: {comp.get('llm', 0):.0f} LLM calls · {comp.get('search', 0):.0f} searches · "
        f"{comp.get('visit', 0):.0f} visits. The offline-authored DAG fixes the",
        "   plan, so the online executor spends calls filling in leaves, not re-deciding what to do.",
        f"• sequential_react: {seq.get('llm', 0):.0f} LLM calls · {seq.get('visit', 0):.0f} visits — the ReAct "
        "loop pays to re-plan every step.",
        "• naive_rag / parametric baselines barely browse — which is exactly why their scores collapse.",
        "Counts are means across all 3 models × 38 tests (R=3 repeats each).",
    ]
    cap.text(0.0, 0.98, "\n".join(lines), ha="left", va="top", fontsize=fs["annot"] * 0.64,
             color=plot_style.INK_SECONDARY, linespacing=1.55,
             transform=cap.transAxes, family="sans-serif")

    plot_style.savefig_square(fig, out_path, dpi=dpi)
    print(f"Wrote plot -> {out_path} ({side_px}x{side_px}px)")


def plot_score_hist(rows: List[Dict[str, Any]], out_path: Path, side_px: int, dpi: int) -> None:
    """Distribution of per-run validation scores for the two engine strategies vs the
    pooled baselines -- small-multiple histograms so the 'compiled/sequential pile up at
    1.0, baselines spread low' shape is legible at a glance."""
    groups = {
        "graph_compiled": [r["score"] for r in rows if r["variant"] == "graph_compiled"],
        "sequential_react": [r["score"] for r in rows if r["variant"] == "sequential_react"],
        "baselines (graph · naive_rag · parametric)":
            [r["score"] for r in rows if r["variant"] in ("graph", "naive_rag", "parametric")],
    }
    fs = plot_style.font_sizes(side_px)
    mk = plot_style.mark_sizes(side_px)
    colors = [plot_style.categorical_color(1), plot_style.categorical_color(4),
              plot_style.categorical_color(6)]

    side_px = max(plot_style.MIN_SIDE_PX, int(side_px))
    inches = side_px / float(dpi)
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    fig.set_facecolor(plot_style.SURFACE)
    gs = fig.add_gridspec(3, 1, left=0.13, right=0.965, top=0.865, bottom=0.075, hspace=0.34)
    bins = np.linspace(0, 1, 11)
    axes = []
    for i, ((name, vals), color) in enumerate(zip(groups.items(), colors)):
        ax = fig.add_subplot(gs[i])
        plot_style.style_axes(ax)
        ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=mk["edge"])
        m = mean(vals) if vals else 0.0
        ax.axvline(m, color=plot_style.INK_PRIMARY, linestyle="--", linewidth=mk["eline"])
        ax.text(m, ax.get_ylim()[1] * 0.86, f" mean {m:.2f}", ha="left", va="top",
                fontsize=fs["annot"] * 0.7, color=plot_style.INK_PRIMARY, fontweight="bold")
        ax.set_title(f"{name}   (n={len(vals)})", fontsize=fs["label"] * 0.82,
                     fontweight="bold", loc="left", color=plot_style.INK_PRIMARY, pad=8)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=fs["tick"] * 0.82)
        if i < 2:
            ax.set_xticklabels([])
        axes.append(ax)
    axes[-1].set_xlabel("Per-run validation score", fontsize=fs["label"])
    axes[1].set_ylabel("Number of runs", fontsize=fs["label"])
    fig.suptitle("barrage24b — score distribution by strategy",
                 fontsize=fs["title"] * 0.92, fontweight="bold", color=plot_style.INK_PRIMARY, y=0.95)
    plot_style.savefig_square(fig, out_path, dpi=dpi)
    print(f"Wrote plot -> {out_path} ({side_px}x{side_px}px)")


def render_dag_examples(out_dir: Path, side_px: int, dpi: int) -> List[Path]:
    written = []
    for name, (fn, title, mandate) in DAG_EXAMPLES.items():
        plan_path = PLANS_DIR / fn
        if not plan_path.exists():
            print(f"[dag skipped] {plan_path} missing", file=sys.stderr)
            continue
        import json
        plan = json.loads(plan_path.read_text())
        out = out_dir / f"{name}.png"
        render_plan_dag(plan, title=title, mandate=mandate, side_px=side_px, dpi=dpi, out_path=str(out))
        written.append(out)
        print(f"Wrote plot -> {out} ({side_px}x{side_px}px)")
    return written


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Render the barrage24b gallery (Pareto, heatmap, "
                                              "scatter, trend, DAG examples) as square 4K PNGs.")
    ap.add_argument("--run-id", default="barrage24b", help="Final dataset run-id prefix")
    ap.add_argument("--extra-run-ids", default="", help="Comma-separated additional run-id "
                    "prefixes to fold in (e.g. later validation batches once they exist)")
    ap.add_argument("--out-dir", default="", help="Gallery output dir "
                    "(default: <results_dir>/<run-id>_gallery)")
    ap.add_argument("--size", type=int, default=plot_style.DEFAULT_SIDE_PX)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--reference-models", default=",".join(recovery_curve.DEFAULT_REFERENCE))
    args = ap.parse_args(argv)

    extra = [r.strip() for r in args.extra_run_ids.split(",") if r.strip()]
    rows = bench_common.load_rows(run_ids=[args.run_id], extra_run_ids=extra)
    if not rows:
        print(f"No rows found for run_id={args.run_id!r} (+{extra})", file=sys.stderr)
        return 2
    print(f"Loaded {len(rows)} rows for the combined dataset "
          f"(run_id={args.run_id!r}, extra_run_ids={extra})")

    out_dir = Path(args.out_dir) if args.out_dir else bench_common.results_dir() / f"{args.run_id}_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    _combined_stats(rows, out_dir)

    priced_rows = [r for r in rows if r.get("usd") is not None]
    reference_models = [m.strip() for m in args.reference_models.split(",") if m.strip()]
    agg = recovery_curve._aggregate(priced_rows)
    lines = recovery_curve._reference_lines(agg, reference_models)
    recovery_curve._write_csv(agg, out_dir / "recovery_curve.csv")
    recovery_curve._plot(agg, lines, reference_models, out_dir / "pareto.png",
                          side_px=args.size, dpi=args.dpi)

    plot_heatmap(rows, out_dir / "heatmap.png", args.size, args.dpi)
    _scatter(priced_rows, "usd", "Realized cost (USD, mean per test, log scale)",
             "barrage24b — cost vs score (per test)", out_dir / "scatter_cost_vs_score.png",
             args.size, args.dpi, log_x=True)
    _scatter(rows, "visits", "Page visits (mean per test)",
             "barrage24b — visits vs score (per test)", out_dir / "scatter_visits_vs_score.png",
             args.size, args.dpi)
    plot_trend_by_level(rows, out_dir / "trend_by_level.png", args.size, args.dpi)
    plot_work_by_variant(rows, out_dir / "work_by_variant.png", args.size, args.dpi)
    plot_score_hist(rows, out_dir / "score_histogram.png", args.size, args.dpi)
    render_dag_examples(out_dir, args.size, args.dpi)

    print(f"\nGallery written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
