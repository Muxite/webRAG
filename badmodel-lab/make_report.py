#!/usr/bin/env python3
"""Bad-model lab report generator — LinkedIn figures from results/cells_long.csv.

Implements CHART_SPEC.md: validated identity/ordinal palettes, the 0.75 feasibility
bar, ✓/✕ status overlay, magma for magnitude only. Each figure is a panel function
rendered as both a 1×1 hero and a 3×2 grid, in light and dark. $0 — no model calls.

  ./.venv/bin/python badmodel-lab/make_report.py            # both modes -> gallery/
  ./.venv/bin/python badmodel-lab/make_report.py --mode dark
"""
from __future__ import annotations

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
RES = REPO / "agent" / "idea_test_results"
GALLERY = LAB / "gallery"
BAR = 0.75  # feasibility bar (repo overall_passed line)

# ---- model / mitigation metadata -------------------------------------------------
PARAMS = {"tinyllama": 1.1, "qwen2.5:0.5b": 0.5, "llama3.2:1b": 1.0, "qwen2.5:1.5b": 1.5,
          "gemma2:2b": 2.0, "llama3.2:3b": 3.0, "phi3:mini": 3.8, "qwen2.5:7b": 7.0}
MIT = {"m0_baseline": (0, "react\nbaseline"), "m1_thin": (1, "thin\nleaf"),
       "m2_thin_votes": (2, "thin\n+votes"), "m3_react_bigtokens": (3, "react\n+budget"),
       "m4_thin_lean": (1, "thin\nlean")}
SUBJECT_ORDER = ["qwen2.5:0.5b", "tinyllama", "llama3.2:1b", "qwen2.5:1.5b",
                 "gemma2:2b", "llama3.2:3b", "phi3:mini"]  # tiny -> small, ceilings excluded
PARSE_CLASSES = ["valid_json", "fenced_json", "malformed_json", "truncated_json",
                 "prose", "refusal", "empty"]

# ---- palette (validated in CHART_SPEC §1) ----------------------------------------
CAT = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
       "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
MODEL_RAMP = {"light": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
              "dark":  ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]}
STATUS = {"good": "#0ca30c", "crit": "#d03b3b"}
SURF = {"light": "#ffffff", "dark": "#1a1a19"}
# ink: primary, secondary, muted, grid
INK = {"light": ("#0b0b0b", "#3f3d3a", "#7a7873", "#e4e2dc"),
       "dark":  ("#f0efec", "#c9c6c0", "#8f8d87", "#3a3a38")}


def ramp(mode, frac):
    cm = LinearSegmentedColormap.from_list("m", MODEL_RAMP[mode])
    return cm(float(np.clip(frac, 0, 1)))


def style(mode):
    p, s, m, g = INK[mode]
    plt.rcParams.update({
        "figure.facecolor": SURF[mode], "axes.facecolor": SURF[mode],
        "savefig.facecolor": SURF[mode], "text.color": p, "axes.labelcolor": p,
        "axes.edgecolor": m, "xtick.color": s, "ytick.color": s,
        "grid.color": g, "font.size": 15, "axes.titlecolor": p,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ---- data ------------------------------------------------------------------------
def load_cells():
    """Aggregate the per-run cells_long.csv to one row per (model, profile, tier)."""
    df = pd.read_csv(LAB / "results" / "cells_long.csv")
    for c in ("score", "keystone_pass", "usd", "visits", "latency_s"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    g = df.groupby(["model", "profile", "tier"], dropna=False)
    cells = g.agg(keystone_mean=("keystone_pass", "mean"),
                  score_mean=("score", "mean"),
                  n=("keystone_pass", "count"),
                  usd_mean=("usd", "mean"),
                  visits_mean=("visits", "mean")).reset_index()
    cells["mit_rank"] = cells["profile"].map(lambda p: MIT.get(p, (9, p))[0])
    cells["params"] = cells["model"].map(lambda m: PARAMS.get(m, np.nan))
    cells["clears"] = cells["keystone_mean"] >= BAR
    return cells


def load_parse():
    """Per (model, profile) parse-class share, joined via cells.jsonl run_id->profile."""
    cellmap = {}
    cj = LAB / "results" / "cells.jsonl"
    if cj.exists():
        for line in cj.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line); cellmap[r["run_id"]] = r
                except Exception:
                    pass
    counts = defaultdict(lambda: defaultdict(int))
    for f in glob.glob(str(RES / "bml__*_json_telemetry.jsonl")):
        rid = Path(f).name.replace("_json_telemetry.jsonl", "")
        cell = cellmap.get(rid, {})
        key = (cell.get("model", "?"), cell.get("profile", "?"))
        for line in Path(f).read_text().splitlines():
            if line.strip():
                try:
                    counts[key][json.loads(line).get("class", "?")] += 1
                except Exception:
                    pass
    return counts


# ---- panel functions: draw_<fig>(ax, data, *, hero, mode) ------------------------
def draw_lift(ax, cells, *, hero, mode, models=None, highlight=None):
    """FIG 1 — mitigation-lift ladder: keystone vs mitigation rung, one line per model."""
    _, s, muted, _ = INK[mode]
    models = models or [m for m in SUBJECT_ORDER if m in set(cells["model"])]
    ranks = {m: i for i, m in enumerate(SUBJECT_ORDER)}
    pooled = defaultdict(list)
    for m in models:
        sub = cells[(cells.model == m) & (cells.tier.isin(["micro", "reachable"]))]
        sub = sub.groupby("mit_rank")["keystone_mean"].mean().sort_index()
        if sub.empty:
            continue
        for r, v in sub.items():
            pooled[r].append(v)
        is_hi = (highlight == m) or (hero and highlight is None)
        col = ramp(mode, ranks.get(m, 0) / max(1, len(SUBJECT_ORDER) - 1))
        ax.plot(sub.index, sub.values, "-o", color=col, lw=3 if is_hi else 1.6,
                alpha=1.0 if is_hi else 0.35, ms=8 if is_hi else 5, zorder=3 if is_hi else 2)
        if is_hi and not hero:
            ax.annotate(m, (sub.index[-1], sub.values[-1]), color=col, fontsize=11,
                        xytext=(4, 0), textcoords="offset points", va="center", fontweight="bold")
    if pooled and hero:
        xs = sorted(pooled); ys = [np.mean(pooled[r]) for r in xs]
        ax.plot(xs, ys, "-", color=CAT[mode][0], lw=4, zorder=5, label="pooled mean")
    ax.axhline(BAR, ls="--", color=muted, lw=1.5)
    ax.set_ylim(-0.03, 1.03)
    ranks_present = sorted(cells["mit_rank"].unique())
    ax.set_xticks(ranks_present)
    ax.set_xticklabels([MIT_LABEL(r) for r in ranks_present], fontsize=10)
    ax.set_ylabel("keystone pass-rate")


def MIT_LABEL(rank):
    for k, (r, lbl) in MIT.items():
        if r == rank:
            return lbl
    return str(rank)


def draw_parse(ax, counts, *, hero, mode, models=None):
    """FIG 3 — parse-failure composition at the react baseline (m0), stacked share."""
    models = models or [m for m in SUBJECT_ORDER if (m, "m0_baseline") in counts]
    # sort by valid_json share ascending (who needs the ladder most)
    def valid_frac(m):
        c = counts[(m, "m0_baseline")]; t = sum(c.values()) or 1
        return c.get("valid_json", 0) / t
    models = sorted(models, key=valid_frac)
    y = np.arange(len(models))
    left = np.zeros(len(models))
    for i, cls in enumerate(PARSE_CLASSES):
        vals = []
        for m in models:
            c = counts[(m, "m0_baseline")]; t = sum(c.values()) or 1
            vals.append(c.get(cls, 0) / t)
        vals = np.array(vals)
        ax.barh(y, vals, left=left, color=CAT[mode][i], height=0.72,
                label=cls, edgecolor=SURF[mode], linewidth=1.2)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels(models, fontsize=11)
    ax.set_xlim(0, 1); ax.set_xlabel("share of JSON-mode decisions")
    ax.set_title("react baseline (m0)", fontsize=12)


def draw_frontier(ax, cells, *, hero, mode, tier=None, task=None):
    """FIG 5 — feasibility frontier heatmap: model x mitigation, magma fill + ✓/✕."""
    _, s, muted, grid = INK[mode]
    sub = cells.copy()
    if tier:
        sub = sub[sub.tier == tier]
    models = [m for m in (SUBJECT_ORDER + ["qwen2.5:7b"]) if m in set(sub["model"])]
    mits = sorted(sub["mit_rank"].unique())
    M = np.full((len(models), len(mits)), np.nan)
    clear = np.zeros_like(M, dtype=bool)
    for i, mo in enumerate(models):
        for j, mr in enumerate(mits):
            cell = sub[(sub.model == mo) & (sub.mit_rank == mr)]
            if not cell.empty:
                M[i, j] = cell["keystone_mean"].mean()
                clear[i, j] = cell["keystone_mean"].mean() >= BAR
    ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto", origin="lower")
    for i in range(len(models)):
        for j in range(len(mits)):
            if np.isnan(M[i, j]):
                continue
            ok = clear[i, j]
            ax.text(j, i, "✓" if ok else "✕", ha="center", va="center",
                    color=STATUS["good"] if ok else "#ffffff",
                    fontsize=16 if hero else 12, fontweight="bold")
            ax.text(j, i - 0.30, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="#ffffff" if M[i, j] < 0.5 else "#111111", fontsize=8)
    ax.set_xticks(range(len(mits))); ax.set_xticklabels([MIT_LABEL(r) for r in mits], fontsize=10)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=10)
    ax.set_title(task or (tier or "all tiers"), fontsize=12)


def draw_recovery(ax, cells, *, hero, mode):
    """FIG 4 — merged recovery curve: keystone vs model capability, line per task tier."""
    _, s, muted, _ = INK[mode]
    tiers = [t for t in ["micro", "reachable", "hard"] if t in set(cells["tier"])]
    for i, t in enumerate(tiers):
        sub = cells[cells.tier == t]
        # best-mitigation per model
        best = sub.loc[sub.groupby("model")["keystone_mean"].idxmax()]
        best = best.dropna(subset=["params"]).sort_values("params")
        m0 = sub[sub.mit_rank == 0].dropna(subset=["params"]).sort_values("params")
        col = CAT[mode][i]
        ax.plot(best["params"], best["keystone_mean"], "-o", color=col, lw=3, ms=8,
                label=f"{t} (best mit.)", zorder=3)
        if not m0.empty:
            ax.plot(m0["params"], m0["keystone_mean"], "--", color=col, lw=1.4, alpha=0.5, zorder=2)
    ax.axhline(BAR, ls="--", color=muted, lw=1.5)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3, 7]); ax.set_xticklabels(["0.5", "1", "2", "3", "7B"])
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("model size (B params, log)")
    ax.set_ylabel("keystone pass-rate")


# ---- renderers -------------------------------------------------------------------
def hero(fn, mode, **kw):
    fig, ax = plt.subplots(figsize=(11, 11), dpi=170)
    fn(ax, hero=True, mode=mode, **kw)
    ax.grid(axis="y", lw=0.6, alpha=0.5)
    h, l = ax.get_legend_handles_labels()
    if l:
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9, ncol=1)
    fig.tight_layout(pad=2.2)
    return fig


def grid(fn, mode, panels):
    """panels: list of (title, kwargs) — up to 6 in a 3x2."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6), dpi=150)
    for ax, (title, kw) in zip(axes.flat, panels):
        fn(ax, hero=False, mode=mode, **kw)
        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", lw=0.5, alpha=0.4)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    # one figure legend
    h, l = axes.flat[0].get_legend_handles_labels()
    if l:
        fig.legend(h, l, loc="lower center", ncol=min(7, len(l)), fontsize=10,
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=2.4, rect=(0, 0.04, 1, 1))
    return fig


def save(fig, name, mode):
    out = GALLERY / f"{name}_{mode}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build(mode):
    style(mode)
    cells = load_cells()
    counts = load_parse()
    subj = [m for m in SUBJECT_ORDER if m in set(cells["model"])]
    tiers = [t for t in ["micro", "reachable"] if t in set(cells["tier"])]
    made = []

    # FIG 1 — lift
    made.append(save(hero(lambda ax, **k: draw_lift(ax, cells, **k), mode), "fig1_lift_hero", mode))
    made.append(save(grid(lambda ax, **k: draw_lift(ax, cells, **k), mode,
                          [(m, {"models": [m], "highlight": m}) for m in subj[:6]]),
                     "fig1_lift_grid", mode))

    # FIG 3 — parse health
    if counts:
        made.append(save(hero(lambda ax, **k: draw_parse(ax, counts, **k), mode), "fig3_parse_hero", mode))

    # FIG 4 — recovery
    made.append(save(hero(lambda ax, **k: draw_recovery(ax, cells, **k), mode), "fig4_recovery_hero", mode))

    # FIG 5 — frontier
    hero_tier = "reachable" if "reachable" in tiers else "micro"
    made.append(save(hero(lambda ax, **k: draw_frontier(ax, cells, tier=hero_tier, **k), mode),
                     "fig5_frontier_hero", mode))
    made.append(save(grid(lambda ax, **k: draw_frontier(ax, cells, **k), mode,
                          [(t, {"tier": t}) for t in tiers]), "fig5_frontier_grid", mode))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["light", "dark", "both"], default="both")
    args = ap.parse_args()
    modes = ["light", "dark"] if args.mode == "both" else [args.mode]
    all_out = []
    for mode in modes:
        all_out += build(mode)
    print(f"wrote {len(all_out)} figures to {GALLERY}/:")
    for p in all_out:
        print("  ", p.name)


if __name__ == "__main__":
    raise SystemExit(main())
