#!/usr/bin/env python3
"""Render the ladder03 result as a small set of PNGs.

Built on this repo's house chart system (``agent/app/testing/plot_style``): its validated
categorical palette in fixed order, its square canvas, its recessive chrome. No plot here calls
``plt.subplots`` directly, so a resize never needs per-plot font tuning.

What is deliberately NOT plotted: a mean-score ranking between arms or configurations. The frozen
power block sets ``arm_ranking_claim_permitted: false`` at this n, and per-arm orderings on this
suite invert between task subsets. Score appears once, explicitly labelled as an accuracy guard.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/ladder_plots.py --out-dir /tmp/plots
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell_mechanism import RESULTS, facts, load_cells  # noqa: E402
from agent.app.testing import plot_style as ps  # noqa: E402
import module_ab as ma  # noqa: E402

#: Ascending capability — the x-axis of every plot here. tinyllama is absent by decision: it emits
#: no parseable action under any prompt shape tried, which is a boundary, and averaging a floor
#: artifact into a curve is worse than an absent point.
MODELS: List[str] = ["qwen2.5:0.5b", "qwen2.5:1.5b", "gemma2:2b", "llama3.2:3b",
                     "phi3:mini", "qwen2.5:7b", "qwen2.5:14b"]
HOLDOUT = {213, 217, 221}

#: Five OUTCOME stages, worst to best, each with its own hue — never cycled. This is a different
#: axis from ``cell_mechanism.classify``, which answers "what went wrong" and tests the
#: retrieval PATH (url invention) before the OUTCOME (keystone), so a cell that guessed a URL,
#: read it and solved comes back ``URL_INVENTED_OK`` rather than ``SOLVED``. That is the right
#: answer for a failure taxonomy and the wrong one for "how far did this model get", so the plot
#: resolves the outcome itself. URL invention is reported separately in the caption, not as a
#: competing stage.
STAGES: List[Tuple[str, str, int]] = [
    ("never_acted", "never emitted an action", 0),
    ("never_read", "acted, never read a page", 1),
    ("read_ignored", "read a page, ignored it", 3),
    ("read_fabricated", "read a page, wrong value", 4),
    ("solved", "solved", 5),
    ("infra", "infra failed", 7),
]
STAGE_LABEL = {k: lbl for k, lbl, _ in STAGES}
STAGE_HUE = {k: h for k, _, h in STAGES}


def outcome(f: dict) -> str:
    """How far a cell got, resolved outcome-first (unlike the failure taxonomy)."""
    if f["infra_failed"]:
        return "infra"
    if not f["exec_search"] and not f["exec_visit"]:
        return "never_acted"
    if not f["visit_200"]:
        return "never_read"
    keystone = next((c for k, c in f["checks"].items() if k.startswith("keystone")), None)
    if keystone and keystone.get("passed"):
        return "solved"
    coverage = f["checks"].get("coverage")
    return "read_fabricated" if (coverage and coverage.get("score")) else "read_ignored"


def rid(model: str, state: str, host: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower())
    return f"ladder03_{slug}_{state}_{'lg' if host == 'langgraph_react' else 'sq'}"


def short(model: str) -> str:
    return model.replace("qwen2.5:", "qwen ").replace("llama3.2:", "llama ").replace(":", " ")


def mechanism_counts(host: str, results_dir: str) -> Dict[str, Counter]:
    out: Dict[str, Counter] = defaultdict(Counter)
    suffix = "_off_lg" if host == "langgraph_react" else "_off_sq"
    for meta, cell in load_cells(["ladder03_"], results_dir):
        if not meta["run"].endswith(suffix) or int(meta["task"]) in HOLDOUT:
            continue
        out[meta["model"]][outcome(facts(cell))] += 1
    return out


def module_pairs(host: str, results_dir: str) -> Dict[str, Dict[str, Optional[dict]]]:
    """Per model, the structural summary with the module off and on (None when absent)."""
    pairs: Dict[str, Dict[str, Optional[dict]]] = {}
    for model in MODELS:
        row: Dict[str, Optional[dict]] = {}
        for state in ("off", "derive"):
            cells = {t: c for t, c in ma.load_run(rid(model, state, host),
                                                  __import__("pathlib").Path(results_dir)).items()
                     if t not in HOLDOUT}
            if not cells:
                row[state] = None
                continue
            rows = [ma.structural(c) for c in cells.values()]
            scores = [(c.get("validation") or {}).get("overall_score") for c in cells.values()]
            scores = [s for s in scores if s is not None]
            row[state] = {
                "n": len(cells),
                "with_graph": sum(1 for r in rows if r["has_graph"]),
                "derived": sum(r["derived"] for r in rows),
                "invalid": sum(r["invalid"] for r in rows),
                "score": (sum(scores) / len(scores)) if scores else None,
            }
        pairs[model] = row
    return pairs


def _legend(ax, fs, handles, labels, ncol=3, anchor_y=-0.11):
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor_y),
              ncol=ncol, frameon=False, fontsize=fs["legend"], handlelength=1.2,
              columnspacing=1.4, labelcolor=ps.INK_SECONDARY)


def plot_mechanisms(counts: Dict[str, Counter], host: str, out: str, side: int) -> Optional[str]:
    present = [m for m in MODELS if counts.get(m)]
    if not present:
        return None
    fig, ax, fs = ps.square_fig(side, margins=(0.22, 0.96, 0.84, 0.30))
    seen: List[str] = []
    ys = range(len(present))
    for i, model in enumerate(present):
        left = 0.0
        total = sum(counts[model].values()) or 1
        for mech, _lbl, _hue in STAGES:
            v = counts[model].get(mech, 0)
            if not v:
                continue
            frac = 100.0 * v / total
            ax.barh(i, frac, left=left, height=0.62,
                    color=ps.categorical_color(STAGE_HUE[mech]),
                    edgecolor=ps.SURFACE, linewidth=2.0 * ps.font_scale(side))
            if frac >= 12:      # selective direct labels only — never one per segment
                ax.text(left + frac / 2, i, f"{v}", ha="center", va="center",
                        fontsize=fs["tick"], color=ps.SURFACE)
            left += frac
            if mech not in seen:
                seen.append(mech)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([short(m) for m in present], fontsize=fs["tick"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of cells (%)", fontsize=fs["label"], color=ps.INK_SECONDARY)
    host_short = "LangGraph host" if "lang" in host else "ReAct host"
    ax.set_title(f"How far each model got\n{host_short}, module off",
                 fontsize=fs["title"], color=ps.INK_PRIMARY, loc="left", pad=10)
    from matplotlib.patches import Patch
    ordered = [k for k, _, _ in STAGES if k in seen]
    _legend(ax, fs,
            [Patch(facecolor=ps.categorical_color(STAGE_HUE[m])) for m in ordered],
            [STAGE_LABEL[m] for m in ordered], ncol=2)
    path = os.path.join(out, f"01_mechanisms_{'lg' if 'lang' in host else 'sq'}.png")
    ps.savefig_square(fig, path)
    return path


def plot_grouped(pairs, key, title, ylabel, out, side, fname, as_rate=False):
    """Grouped bars: module off vs +derive, one pair per model."""
    present = [m for m in MODELS
               if pairs.get(m, {}).get("off") or pairs.get(m, {}).get("derive")]
    if not present:
        return None

    def val(model, state):
        row = (pairs.get(model) or {}).get(state)
        if not row:
            return None
        v = row.get(key)
        if v is None:
            return None
        return (100.0 * v / row["n"]) if as_rate else float(v)

    # A module comparison needs BOTH conditions. With only one present the chart advertises a
    # series that has no marks and the y-axis auto-scales around zero, which is worse than no
    # chart: it looks like a measured zero rather than an unrun condition.
    if all(val(m, "derive") is None for m in present):
        return None
    if all(val(m, "off") is None for m in present):
        return None
    fig, ax, fs = ps.square_fig(side, margins=(0.15, 0.96, 0.84, 0.36))
    w = 0.36
    for j, (state, label) in enumerate((("off", "host alone"), ("derive", "host + derive"))):
        xs, ys = [], []
        for i, model in enumerate(present):
            v = val(model, state)
            if v is None:
                continue
            xs.append(i + (j - 0.5) * w)
            ys.append(v)
        ax.bar(xs, ys, width=w * 0.92, label=label,
               color=ps.categorical_color(0 if j == 0 else 3),
               edgecolor=ps.SURFACE, linewidth=2.0 * ps.font_scale(side))
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.0f}" if not as_rate else f"{y:.0f}%", ha="center", va="bottom",
                    fontsize=fs["tick"], color=ps.INK_SECONDARY)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([short(m) for m in present], fontsize=fs["tick"], rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=fs["label"], color=ps.INK_SECONDARY)
    ax.set_title(title, fontsize=fs["title"], color=ps.INK_PRIMARY, loc="left", pad=14)
    h, l = ax.get_legend_handles_labels()
    _legend(ax, fs, h, l, ncol=2, anchor_y=-0.26)   # clears the rotated model labels
    path = os.path.join(out, fname)
    ps.savefig_square(fig, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--side-px", type=int, default=1600)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    made: List[str] = []
    for host in ("langgraph_react", "sequential_react"):
        p = plot_mechanisms(mechanism_counts(host, a.results_dir), host, a.out_dir, a.side_px)
        if p:
            made.append(p)
        pairs = module_pairs(host, a.results_dir)
        tag = "lg" if host == "langgraph_react" else "sq"
        for key, title, ylabel, fname, rate in (
            ("with_graph", "Cells with a machine-checkable derivation\n" + ("LangGraph host" if "lang" in host else "ReAct host"),
             "% of cells", f"02_derivation_available_{tag}.png", True),
            ("derived", "Machine-recomputed values\n" + ("LangGraph host" if "lang" in host else "ReAct host"),
             "count", f"03_derived_values_{tag}.png", False),
            ("score", "Accuracy guard — NOT a ranking\n" + ("LangGraph host" if "lang" in host else "ReAct host"),
             "mean overall_score", f"04_score_guard_{tag}.png", False),
        ):
            p = plot_grouped(pairs, key, title, ylabel, a.out_dir, a.side_px, fname, rate)
            if p:
                made.append(p)

    for p in made:
        print(f"  wrote {p}")
    if not made:
        print("no plottable data yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
