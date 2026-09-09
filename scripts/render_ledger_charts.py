#!/usr/bin/env python3
"""render_ledger_charts.py: the four presentation figures for the Euglena Ledger.

    PYTHONPATH=.:services:agent python3 scripts/render_ledger_charts.py \\
        --cell <cell>.json --prefix mint04 -o gallery/

Four square PNGs in the house style (:mod:`agent.app.testing.plot_style`) on the frozen light
paper surface, so they drop straight into docs, a README or a post:

  ``risk_coverage.png``   the L1 KPI -- when the compiler declines to answer, is what it does
                          answer right more often? Campaign-level; needs a run prefix.
  ``provenance_flow.png`` the funnel from a figure printed in the answer down to a figure a
                          reader could actually check, with the loss at each stage drawn.
  ``run_timeline.png``    every step of one run on a time axis, with the evidence plane -- which
                          costs no time -- banded off the clock rather than pinned to it.
  ``answer_audit.png``    every number in the answer, ranked by how uniquely it is backed.

The three run-level figures are projections of the same storyboard the deck draws, so a figure
and a frame can never disagree about what happened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_ROOT, os.path.join(_ROOT, "services"), os.path.join(_ROOT, "agent"),
              os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

from agent.app import ledger_trace  # noqa: E402
from agent.app.testing import ledger_story, plot_style  # noqa: E402

#: Timeline lanes, top to bottom, in the order a run reads.
LANES = ["search", "visit", "http", "llm", "chroma"]

#: Reserved roles, taken from the house categorical ramp so every figure here and the rest of the
#: gallery stay one system. Never reassigned between figures.
C_EVIDENCE = plot_style.categorical_color(3)   # a located span / a timed step
C_DERIVED = plot_style.categorical_color(1)    # a value computed from spans
C_STAGE = [plot_style.categorical_color(i) for i in (1, 2, 3, 4, 5)]

DPI = 160


def _chars_per_line(side_px: int, fontsize: float) -> int:
    """How many characters fit across this canvas at this point size.

    Long captions were being clipped by the figure edge because nothing wrapped them. Wrapping
    needs the real geometry, not a guess: the figure is ``side_px/DPI`` inches wide, a point is
    1/72 inch, and a proportional glyph averages about 0.55 em.
    """
    inches = side_px / float(DPI)
    return max(28, int(inches * 72 / (0.55 * max(fontsize, 1.0))))


def _frame(fig, fs, side_px: int, title: str, subtitle: str = "", caption: str = "") -> None:
    """Title, subtitle and footnote placed in the margins reserved for them.

    Everything is positioned in FIGURE coordinates and wrapped to the canvas. The earlier version
    put the title on the axes and the subtitle just above it, which collided on every figure, and
    left captions unwrapped so they ran off the right edge.
    """
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=fs["title"],
             fontweight="bold", color=plot_style.INK_PRIMARY)
    if subtitle:
        size = fs["subtitle"] * 0.74
        fig.text(0.5, 0.925, textwrap.fill(subtitle, _chars_per_line(side_px, size)),
                 ha="center", va="top", fontsize=size, color=plot_style.INK_MUTED,
                 linespacing=1.4)
    if caption:
        size = fs["annot"] * 0.8
        fig.text(0.5, 0.075, textwrap.fill(caption, _chars_per_line(side_px, size)),
                 ha="center", va="top", fontsize=size, color=plot_style.INK_MUTED,
                 linespacing=1.45)


# =============================================================================== 1. risk-coverage

def risk_coverage(cells: Sequence[Dict[str, Any]], out_path: str, side_px: int) -> Optional[str]:
    """The selective-prediction curve, with the accuracy anchor printed beside it.

    Two of this repo's frozen KPI reading rules are enforced here rather than assumed: the
    accuracy anchor (``validation.overall_score``) is printed on the same figure, so a KPI gain
    bought with real accuracy is visible in one glance; and a cell that cannot produce the signal
    is reported as UNKNOWN rather than dropped, which would let the denominator silently shrink
    to the cells that happened to work.
    """
    import ledger_risk_coverage as LRC

    usable = [c for c in cells if not c.get("infra_failed") and c.get("score") is not None]
    unknown = len(cells) - len(usable)
    if not usable:
        return None
    curve = [r for r in LRC.graded_curve(usable) if r["coverage"] and r["risk"] is not None]
    if len(curve) < 2:
        return None

    fig, ax, fs = plot_style.square_fig(side_px, dpi=DPI, margins=(0.14, 0.95, 0.83, 0.20))
    ms = plot_style.mark_sizes(side_px)
    xs = [r["coverage"] * 100 for r in curve]
    ys = [r["risk"] * 100 for r in curve]
    ax.plot(xs, ys, "-o", color=C_EVIDENCE, linewidth=ms["line"], markersize=ms["marker"],
            markeredgecolor="white", markeredgewidth=ms["edge"], zorder=3)
    # Four of the six thresholds sit within two coverage points of each other, so no offset
    # scheme separates their labels -- they are genuinely almost the same point. Annotate only
    # the two the eye needs (answer-everything, and the best operating point) and give the exact
    # numbers as a table, which is more precise than a crowded label anyway.
    # graded_curve sorts ASCENDING by threshold, so curve[0] is the answer-everything point at
    # full coverage and curve[-1] is the strictest one. Anchor each label on the side that keeps
    # it inside the axes.
    best = min(curve, key=lambda r: (r["risk"], -r["coverage"]))
    for row, ha, dx in ((curve[0], "right", -ms["marker"]), (best, "left", ms["marker"])):
        x, y = row["coverage"] * 100, row["risk"] * 100
        ax.annotate(f"≥{row['threshold']} clauses · n={row['n_accepted']}", (x, y),
                    textcoords="offset points", xytext=(dx, ms["marker"] * 1.2),
                    ha=ha, va="bottom", fontsize=fs["annot"] * 0.78,
                    color=plot_style.INK_PRIMARY, fontweight="bold")

    table = ["clauses  coverage   risk    n"]
    for row in sorted(curve, key=lambda r: -r["threshold"]):
        table.append(f"  ≥{row['threshold']}      {row['coverage'] * 100:5.1f}%   "
                     f"{row['risk'] * 100:5.1f}%  {row['n_accepted']:3d}")
    ax.text(0.40, 0.52, "\n".join(table), transform=ax.transAxes, va="top", ha="left",
            fontsize=fs["annot"] * 0.72, color=plot_style.INK_SECONDARY,
            family="monospace", linespacing=1.5)

    base = [r for r in curve if r["threshold"] == 0]
    if base:
        ax.axhline(base[0]["risk"] * 100, color=plot_style.BASELINE, linestyle="--",
                   linewidth=ms["grid"] * 1.6, zorder=1)
        ax.text(ax.get_xlim()[0], base[0]["risk"] * 100, " answer everything",
                fontsize=fs["annot"] * 0.78, color=plot_style.INK_MUTED, va="bottom")

    ax.set_xlabel("coverage — % of runs accepted", fontsize=fs["label"] * 0.9)
    ax.set_ylabel("risk — % wrong among accepted", fontsize=fs["label"] * 0.9)
    ax.tick_params(labelsize=fs["tick"])
    anchor = sum(c["score"] for c in usable) / len(usable)
    _frame(fig, fs, side_px, "Declining to answer buys accuracy",
           "risk against coverage as the certify threshold moves · down and left is better",
           f"Accuracy anchor: mean overall_score {anchor:.3f} over {len(usable)} cells."
           + (f" {unknown} cells report UNKNOWN and are not imputed as zero." if unknown else "")
           + " Each certify clause is an independent mechanical check; the threshold is how "
             "many a run must pass to be accepted.")
    plot_style.savefig_square(fig, out_path)
    return out_path


# ============================================================================ 2. provenance flow

def provenance_flow(story: ledger_story.Storyboard, out_path: str, side_px: int) -> Optional[str]:
    """The funnel from a printed figure to a checkable one.

    Every stage counts the SAME thing -- a number printed in the final answer -- so the narrowing
    is real attrition. An earlier version put "pages fetched" at the front, which made the chart
    widen from 5 to 26 and invited the reader to compare a page against a span: different units,
    so the flow between them meant nothing. The evidence base is stated in the caption instead,
    where it is context rather than a stage.
    """
    audit = story.of_kind(ledger_story.KIND_AUDIT)
    if not audit:
        return None
    numbers = audit[0].detail.get("numbers") or []
    if not numbers:
        return None
    non_trivial = [n for n in numbers if not n["trivial"]]
    backed = [n for n in non_trivial if n["status"] in ("backed", "derived")]
    unique = [n for n in backed if n["ambiguity"] <= 1]

    stages = [
        ("printed in\nthe answer", len(numbers)),
        ("carries real\ninformation", len(non_trivial)),
        ("backed by a span\nor a derivation", len(backed)),
        ("backed by exactly\none value", len(unique)),
    ]
    top = max(v for _, v in stages) or 1

    fig, ax, fs = plot_style.square_fig(side_px, dpi=DPI, margins=(0.06, 0.96, 0.83, 0.22),
                                        style=False)
    ax.set_facecolor(plot_style.SURFACE)
    fig.set_facecolor(plot_style.SURFACE)
    ax.set_xlim(-0.55, len(stages) - 0.45)
    ax.set_ylim(-0.30, 1.22)
    ax.axis("off")

    bar_w = 0.26
    heights = [v / top for _, v in stages]
    for i, ((label, value), h) in enumerate(zip(stages, heights)):
        y0 = (1 - h) / 2
        ax.add_patch(Rectangle((i - bar_w / 2, y0), bar_w, h, facecolor=C_STAGE[i],
                               edgecolor="none", zorder=3))
        ax.text(i, y0 + h + 0.04, f"{value}", ha="center", va="bottom",
                fontsize=fs["title"] * 0.85, fontweight="bold", color=plot_style.INK_PRIMARY)
        ax.text(i, -0.05, label, ha="center", va="top", fontsize=fs["label"] * 0.72,
                color=plot_style.INK_SECONDARY, linespacing=1.4)
        if i + 1 < len(stages):
            h2 = heights[i + 1]
            y1 = (1 - h2) / 2
            ax.add_patch(Polygon([(i + bar_w / 2, y0), (i + bar_w / 2, y0 + h),
                                  (i + 1 - bar_w / 2, y1 + h2), (i + 1 - bar_w / 2, y1)],
                                 closed=True, facecolor=C_STAGE[i], alpha=0.20,
                                 edgecolor="none", zorder=2))
            lost = stages[i][1] - stages[i + 1][1]
            if lost > 0:
                ax.text(i + 0.5, max(y0 + h, y1 + h2) + 0.03, f"−{lost}", ha="center", va="bottom",
                        fontsize=fs["annot"] * 0.9, color=plot_style.STATUS_CRITICAL,
                        fontweight="bold")

    counts = story.beats[-1].ledger_after.counts
    _frame(fig, fs, side_px, "From printed to checkable",
           f"task {story.task_id} · {story.model} · every number in the final answer",
           f"Evidence base: {counts['pages']} pages read, {counts['sources']} spans located "
           f"({counts['verified']} verified against the page text), {counts['derived']} values "
           f"derived in code. A figure matching several evidence values backs none of them "
           f"uniquely, so it drops at the last stage.")
    plot_style.savefig_square(fig, out_path)
    return out_path


# ================================================================================ 3. run timeline

def run_timeline(cell: Dict[str, Any], story: ledger_story.Storyboard, out_path: str,
                 side_px: int) -> str:
    """Every step of one run on a time axis, with the evidence plane banded off the clock.

    Derivations and refusals cost no time, so they have no interval. Giving them ``t=0`` would put
    the run's most consequential steps at the origin and imply they happened first. They get a
    shaded band with no time meaning instead, labelled as such, so a reader cannot mistake their
    left-to-right order for a chronology.
    """
    trace = ledger_trace.project_cell(cell)
    timed = [n for n in trace.nodes if n.t_start is not None and n.t_end is not None
             and n.kind != ledger_trace.KIND_RUN]
    plane = [n for n in trace.nodes if n.t_start is None and n.kind != ledger_trace.KIND_RUN]
    lanes = [k for k in LANES if any(n.kind == k for n in timed)]
    span = max((n.t_end for n in timed), default=1.0) or 1.0

    fig, ax, fs = plot_style.square_fig(side_px, dpi=DPI, margins=(0.27, 0.95, 0.83, 0.22))
    ms = plot_style.mark_sizes(side_px)
    rows = lanes + (["evidence plane\n(no clock)"] if plane else [])
    ypos = {name: len(rows) - 1 - i for i, name in enumerate(rows)}

    for node in timed:
        ax.add_patch(Rectangle((node.t_start, ypos[node.kind] - 0.30),
                               max(node.duration, span * 0.004), 0.60,
                               facecolor=C_EVIDENCE if node.status == "ok"
                               else plot_style.dark_status_color(node.status),
                               edgecolor="white", linewidth=ms["edge"] * 0.5, zorder=3))
    if plane:
        y = ypos["evidence plane\n(no clock)"]
        ax.add_patch(Rectangle((-span * 0.02, y - 0.5), span * 1.04, 1.0,
                               facecolor=plot_style.PAGE, edgecolor=plot_style.GRIDLINE,
                               linewidth=ms["grid"] * 1.4, linestyle="--", zorder=1))
        for i, node in enumerate(plane):
            x = span * (0.02 + 0.94 * (i + 0.5) / len(plane))
            ax.plot([x], [y], marker="D", markersize=ms["marker"] * 0.8,
                    color=plot_style.STATUS_SERIOUS if node.status == "refused" else C_DERIVED,
                    markeredgecolor="white", markeredgewidth=ms["edge"] * 0.6, zorder=4)

    ax.set_yticks([ypos[name] for name in rows])
    ax.set_yticklabels(rows, fontsize=fs["tick"] * 0.85)
    ax.set_ylim(-0.85, len(rows) - 0.25)
    ax.set_xlim(-span * 0.03, span * 1.03)
    ax.set_xlabel("seconds from the start of the run", fontsize=fs["label"] * 0.9)
    ax.tick_params(labelsize=fs["tick"])
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=C_EVIDENCE, label="timed step"),
                       Line2D([], [], marker="D", linestyle="none", color=C_DERIVED,
                              markersize=ms["marker"] * 0.7, label="derivation"),
                       Line2D([], [], marker="D", linestyle="none", color=plot_style.STATUS_SERIOUS,
                              markersize=ms["marker"] * 0.7, label="refusal")],
              loc="lower right", fontsize=fs["legend"] * 0.8, frameon=False, ncol=3)

    kinds = Counter(n.kind for n in trace.nodes if n.kind != ledger_trace.KIND_RUN)
    _frame(fig, fs, side_px, f"One run, end to end — {span:.1f}s",
           f"task {story.task_id} · {story.model} · " +
           " · ".join(f"{v} {k}" for k, v in sorted(kinds.items())),
           "The bottom band has no time meaning. Derivations and refusals happen on the evidence "
           "plane and cost no measurable time, so they are spaced evenly rather than placed on "
           "the clock.")
    plot_style.savefig_square(fig, out_path)
    return out_path


# =============================================================================== 4. answer audit

def answer_audit(story: ledger_story.Storyboard, out_path: str, side_px: int) -> Optional[str]:
    """Every number in the answer, ranked by how uniquely the evidence backs it."""
    audit = story.of_kind(ledger_story.KIND_AUDIT)
    if not audit:
        return None
    rows = [n for n in (audit[0].detail.get("numbers") or []) if not n["trivial"]]
    if not rows:
        return None
    order = {"unbacked": 0, "derived": 1, "backed": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 0), -r["ambiguity"]))

    fig, ax, fs = plot_style.square_fig(side_px, dpi=DPI, margins=(0.22, 0.95, 0.83, 0.22))
    ms = plot_style.mark_sizes(side_px)
    colors = {"backed": plot_style.STATUS_GOOD, "derived": C_DERIVED,
              "unbacked": plot_style.STATUS_CRITICAL}
    widths = [1.0 if r["ambiguity"] <= 1 else 1.0 / r["ambiguity"] for r in rows]
    ys = list(range(len(rows)))
    ax.barh(ys, widths, height=0.72,
            color=[colors.get(r["status"], plot_style.INK_MUTED) for r in rows],
            edgecolor="white", linewidth=ms["edge"], zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['text']} {r['unit']}".strip() for r in rows],
                       fontsize=fs["tick"] * 0.60)
    ax.set_xlim(0, 1.30)
    ax.set_xlabel("backing strength  =  1 / matching evidence values", fontsize=fs["label"] * 0.82)
    ax.tick_params(labelsize=fs["tick"])
    for y, row, w in zip(ys, rows, widths):
        if row["ambiguity"] > 1:
            ax.text(w + 0.02, y, f"matches {row['ambiguity']}", va="center",
                    fontsize=fs["annot"] * 0.62, color=plot_style.STATUS_WARNING)
    present = {r["status"] for r in rows}
    keys = [("backed", plot_style.STATUS_GOOD, "read off a page"),
            ("derived", C_DERIVED, "derived in code"),
            ("unbacked", plot_style.STATUS_CRITICAL, "unbacked")]
    ax.legend(handles=[Patch(facecolor=c, label=l) for k, c, l in keys if k in present],
              loc="lower right", fontsize=fs["legend"] * 0.8, frameon=False)

    blockers = audit[0].detail.get("blockers", {})
    named = "  ·  ".join(f"{v} {k}" for k, v in blockers.items() if v) or "no blocking condition"
    _frame(fig, fs, side_px, "Every figure in the answer, audited",
           f"task {story.task_id} · {story.model} · answer_supported = "
           f"{audit[0].detail.get('answer_supported')}",
           f"A full bar means exactly one evidence value matches that figure. A short bar means "
           f"several do, so it backs none of them uniquely. Blocking conditions: {named}.")
    plot_style.savefig_square(fig, out_path)
    return out_path


# ========================================================================================= main

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell", required=True, help="stored cell for the three run-level figures")
    parser.add_argument("--prefix", default="", help="run prefix for the campaign risk-coverage curve")
    parser.add_argument("-o", "--out-dir", default="ledger_gallery")
    parser.add_argument("--size", type=int, default=plot_style.MIN_SIDE_PX)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cell, "r", encoding="utf-8") as handle:
        cell = json.load(handle)
    story = ledger_story.build(cell, source_file=args.cell)

    made = [
        provenance_flow(story, os.path.join(args.out_dir, "provenance_flow.png"), args.size),
        run_timeline(cell, story, os.path.join(args.out_dir, "run_timeline.png"), args.size),
        answer_audit(story, os.path.join(args.out_dir, "answer_audit.png"), args.size),
    ]

    if args.prefix:
        import ledger_risk_coverage as LRC
        cells = []
        for path in LRC.discover_cell_files(Path(args.cell).parent, args.prefix):
            raw = LRC.load_cell(path)
            if raw is not None:
                cells.append(LRC.classify_cell(path, raw))
        print(f"risk-coverage over {len(cells)} cells with prefix {args.prefix!r}")
        made.append(risk_coverage(cells, os.path.join(args.out_dir, "risk_coverage.png"), args.size))

    for path in made:
        print(("  wrote " + path) if path else "  skipped (no data for this figure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
