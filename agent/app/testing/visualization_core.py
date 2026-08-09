"""
Core visualization: 4 information-dense images ranked by importance.

Image 1 (P1): Executive Summary — leaderboard, scores, pass rates, KPIs
Image 2 (P2): Test × Model Heatmap — per-test scores, difficulty, pass rates
Image 3 (P3): Efficiency Dashboard — cost, time, tokens, graph-vs-sequential
Image 4 (P4): Actions & Structure — searches/visits, graph shape, validation
"""

from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .visualization_helpers import (
    _system_label,
    _get_system_colors,
    _format_tokens,
    _extract_graph_metrics,
)


def _collect_per_system(results: List[Dict[str, Any]]):
    from agent.app.model_costs import estimate_cost, estimate_cost_from_total

    systems = sorted(set(_system_label(r) for r in results))
    data = {s: {"scores": [], "passed": [], "durations": [], "tokens": [],
                "costs": [], "visits": [], "searches": [], "nodes": [],
                "depth": [], "branching": [],
                "search_kb": [], "visit_kb": []} for s in systems}

    for r in results:
        label = _system_label(r)
        score = r.get("validation", {}).get("overall_score", 0.0)
        passed = r.get("validation", {}).get("overall_passed", False)
        exe = r.get("execution", {})
        obs = exe.get("observability", {})
        llm = obs.get("llm", {})
        model = str(r.get("model", "unknown"))
        inp = llm.get("prompt", {}).get("tokens", 0)
        out = llm.get("completion", {}).get("tokens", 0)
        total_tok = llm.get("total_tokens", 0)
        cost = (estimate_cost(model, inp, out) if (inp or out)
                else estimate_cost_from_total(model, total_tok)) or 0.0

        gm = _extract_graph_metrics(r)

        d = data[label]
        d["scores"].append(score)
        d["passed"].append(passed)
        d["durations"].append(exe.get("duration_seconds", 0))
        d["tokens"].append(total_tok)
        d["costs"].append(cost)
        d["visits"].append(obs.get("visit", {}).get("count", 0))
        d["searches"].append(obs.get("search", {}).get("count", 0))
        d["nodes"].append(gm["total_nodes"])
        d["depth"].append(gm["max_depth"])
        d["branching"].append(gm["avg_branching"])
        d["search_kb"].append(obs.get("search", {}).get("kilobytes", 0))
        d["visit_kb"].append(obs.get("visit", {}).get("kilobytes", 0))

    return systems, data


def _variant_colors(systems: List[str]) -> Dict[str, str]:
    """
    Return a color dict where graph variants use blue tones and
    sequential variants use red/orange tones so the gap is immediately visible.
    """
    blues = ["#2B7BBA", "#7BBFEA"]
    reds = ["#D94F3B", "#F4A582"]
    graph_sys = sorted(s for s in systems if "[graph]" in s)
    seq_sys = sorted(s for s in systems if "[sequential]" in s)
    colors = {}
    for i, s in enumerate(graph_sys):
        colors[s] = blues[i % len(blues)]
    for i, s in enumerate(seq_sys):
        colors[s] = reds[i % len(reds)]
    for s in systems:
        if s not in colors:
            colors[s] = "#888888"
    return colors


def _test_system_matrix(results: List[Dict[str, Any]]):
    systems = sorted(set(_system_label(r) for r in results))
    test_ids = sorted(set(r.get("test_metadata", {}).get("test_id", "?") for r in results))
    matrix = {}
    passed_matrix = {}
    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "?")
        label = _system_label(r)
        score = r.get("validation", {}).get("overall_score", 0.0)
        ok = r.get("validation", {}).get("overall_passed", False)
        matrix[(tid, label)] = score
        passed_matrix[(tid, label)] = ok
    return test_ids, systems, matrix, passed_matrix


# ──────────────────────────────────────────────────────────────────────
# IMAGE 1 — Executive Summary (highest importance)
# ──────────────────────────────────────────────────────────────────────

def plot_core_p1_executive(results: List[Dict[str, Any]], output_dir: Path):
    """
    Priority-1 image: score heatmap, score box-plot, data volume chart,
    graph-vs-sequential score gap.
    :param results: Test result dicts.
    :param output_dir: Output directory.
    """
    from agent.app.model_costs import format_cost

    systems, data = _collect_per_system(results)
    colors = _variant_colors(systems)
    test_ids, systems_hm, matrix, passed_matrix = _test_system_matrix(results)

    fig = plt.figure(figsize=(30, 18), facecolor="white")
    # Optimized layout: heatmap left, graph vs sequential (most important) top right, leaderboard bottom right
    gs = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.20,
                           left=0.04, right=0.97, top=0.98, bottom=0.04,
                           width_ratios=[2.2, 1.2, 1], height_ratios=[1.2, 1])

    ax_hm = fig.add_subplot(gs[0, 0])
    n_tests = len(test_ids)
    n_sys = len(systems_hm)
    grid = np.full((n_tests, n_sys), np.nan)
    for i, tid in enumerate(test_ids):
        for j, s in enumerate(systems_hm):
            if (tid, s) in matrix:
                grid[i, j] = matrix[(tid, s)]

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#f0f0f0")
    im = ax_hm.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_hm.set_xticks(range(n_sys))
    ax_hm.set_xticklabels([s.replace(" [", "\n[") for s in systems_hm], fontsize=20, fontweight="bold")
    ax_hm.set_yticks(range(n_tests))
    test_names = {}
    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "?")
        name = r.get("test_metadata", {}).get("test_name", tid)
        test_names[tid] = name
    ax_hm.set_yticklabels([f"{tid}" for tid in test_ids], fontsize=18, fontweight="bold")
    ax_hm.text(0.99, 1.01, "(green=1.0, red=0.0)", transform=ax_hm.transAxes,
               fontsize=16, color="gray", ha="right", va="bottom", fontweight="bold")

    for i, tid in enumerate(test_ids):
        for j, s in enumerate(systems_hm):
            if (tid, s) in matrix:
                val = matrix[(tid, s)]
                ok = passed_matrix.get((tid, s), False)
                mark = "✓" if ok else "✗"
                txt_color = "white" if val < 0.4 or val > 0.85 else "black"
                ax_hm.text(j, i, f"{val:.2f}\n{mark}",
                            ha="center", va="center", fontsize=14, color=txt_color,
                            fontweight="bold")

    cb = fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02)
    cb.set_label("Score", fontsize=18, fontweight="bold")
    cb.ax.tick_params(labelsize=16)

    # Graph vs Sequential - TOP RIGHT (most important, frontloaded)
    ax_gap = fig.add_subplot(gs[0, 1:])
    ax_gap.text(0.99, 0.99, "(higher is better)", transform=ax_gap.transAxes,
                fontsize=16, color="gray", ha="right", va="top", fontweight="bold")
    
    graph_sys = sorted(s for s in systems if "[graph]" in s)
    seq_sys = sorted(s for s in systems if "[sequential]" in s)
    models = sorted(set(s.split("[")[0].strip() for s in systems))
    
    if graph_sys and seq_sys:
        x_gap = np.arange(len(models))
        w_gap = 0.35
        g_scores = []
        s_scores = []
        for m in models:
            gl = f"{m} [graph]"
            sl = f"{m} [sequential]"
            g_scores.append(np.mean(data[gl]["scores"]) if gl in data else 0)
            s_scores.append(np.mean(data[sl]["scores"]) if sl in data else 0)
        
        b_g = ax_gap.bar(x_gap - w_gap / 2, g_scores, w_gap,
                          color="#2B7BBA", edgecolor="black", linewidth=1.5, label="Graph", alpha=0.85)
        b_s = ax_gap.bar(x_gap + w_gap / 2, s_scores, w_gap,
                          color="#D94F3B", edgecolor="black", linewidth=1.5, label="Sequential", alpha=0.85)
        
        for i, (gv, sv) in enumerate(zip(g_scores, s_scores)):
            ax_gap.text(x_gap[i] - w_gap / 2, gv + 0.02, f"{gv:.3f}",
                         ha="center", fontsize=18, fontweight="bold", color="#2B7BBA")
            ax_gap.text(x_gap[i] + w_gap / 2, sv + 0.02, f"{sv:.3f}",
                         ha="center", fontsize=18, fontweight="bold", color="#D94F3B")
            if gv > sv:
                pct = (gv - sv) / sv * 100 if sv > 0 else 0
                ax_gap.annotate(f"+{pct:.0f}%", xy=(x_gap[i], max(gv, sv) + 0.06),
                                 fontsize=22, fontweight="bold", ha="center", color="#1a5c2a")
        
        ax_gap.set_xticks(x_gap)
        ax_gap.set_xticklabels(models, fontsize=20, fontweight="bold")
        ax_gap.set_ylabel("Avg Score", fontsize=20, fontweight="bold")
        ax_gap.set_ylim(0, 1.2)
        ax_gap.axhline(0.75, color="gray", linestyle="--", alpha=0.4, linewidth=2.0)
        ax_gap.legend(fontsize=16, loc="upper left")
        ax_gap.tick_params(labelsize=18)
        ax_gap.grid(axis="y", alpha=0.3)
    else:
        ax_gap.text(0.5, 0.5, "Single variant only", ha="center", va="center",
                     fontsize=20, color="gray", transform=ax_gap.transAxes, fontweight="bold")

    ax_tok = fig.add_subplot(gs[1, 0])
    ax_tok.text(0.99, 0.99, "(lower tokens better, higher activity better)",
                transform=ax_tok.transAxes, fontsize=16, color="gray", ha="right", va="top", fontweight="bold")
    x = np.arange(len(systems))
    bw = 0.22
    avg_tokens_k = [np.mean(data[s]["tokens"]) / 1000 for s in systems]
    avg_searches = [np.mean(data[s]["searches"]) for s in systems]
    avg_visits = [np.mean(data[s]["visits"]) for s in systems]

    sys_colors = [colors[s] for s in systems]
    ax_tok.bar(x - bw, avg_tokens_k, bw, color=sys_colors,
               edgecolor="black", linewidth=0.8, label="Tokens (k)")
    for i, val in enumerate(avg_tokens_k):
        ax_tok.text(x[i] - bw, val + 0.3, f"{val:.1f}k", ha="center", fontsize=16, fontweight="bold")
    ax_tok.set_ylabel("Avg Tokens (thousands)", fontsize=18, color="#333333", fontweight="bold")
    ax_tok.set_xticks(x)
    ax_tok.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=16, fontweight="bold")
    ax_tok.tick_params(axis="y", labelsize=16)
    ax_tok.tick_params(axis="x", labelsize=16)
    ax_tok.grid(axis="y", alpha=0.3)

    ax_ct = ax_tok.twinx()
    ax_ct.bar(x, avg_searches, bw, color=sys_colors, edgecolor="black",
              linewidth=0.8, alpha=0.6, hatch="//", label="Searches")
    ax_ct.bar(x + bw, avg_visits, bw, color=sys_colors, edgecolor="black",
              linewidth=0.8, alpha=0.6, hatch="..", label="Visits")
    for i, (sc, vc) in enumerate(zip(avg_searches, avg_visits)):
        ax_ct.text(x[i], sc + 0.1, f"{sc:.1f}", ha="center", fontsize=16, fontweight="bold")
        ax_ct.text(x[i] + bw, vc + 0.1, f"{vc:.1f}", ha="center", fontsize=16, fontweight="bold")
    ax_ct.set_ylabel("Avg Count", fontsize=18, color="#666666", fontweight="bold")
    ax_ct.tick_params(axis="y", labelsize=16)

    lines_tok, labels_tok = ax_tok.get_legend_handles_labels()
    lines_ct, labels_ct = ax_ct.get_legend_handles_labels()
    ax_tok.legend(lines_tok + lines_ct, labels_tok + labels_ct, fontsize=14, loc="upper left", ncol=2)

    # Overall Leaderboard - BOTTOM RIGHT (all systems ranked, same bar chart style)
    ax_leader = fig.add_subplot(gs[1, 1:])
    ax_leader.text(0.99, 0.99, "(higher is better)", transform=ax_leader.transAxes,
                   fontsize=16, color="gray", ha="right", va="top", fontweight="bold")
    
    # Calculate average scores for all systems and sort
    system_avgs = [(s, np.mean(data[s]["scores"]), np.mean([1 if p else 0 for p in data[s]["passed"]])) 
                    for s in systems]
    system_avgs.sort(key=lambda x: (x[1], x[2]), reverse=True)  # Sort by score, then pass rate
    
    sorted_systems = [s[0] for s in system_avgs]
    sorted_scores = [s[1] for s in system_avgs]
    sorted_pass_rates = [s[2] for s in system_avgs]
    
    x_leader = np.arange(len(sorted_systems))
    w_leader = 0.6
    bars = ax_leader.bar(x_leader, sorted_scores, w_leader,
                         color=[colors[s] for s in sorted_systems],
                         edgecolor="black", linewidth=1.5, alpha=0.85)
    
    # Add score labels on bars
    for i, (bar, score, pr) in enumerate(zip(bars, sorted_scores, sorted_pass_rates)):
        ax_leader.text(bar.get_x() + bar.get_width() / 2, score + 0.02,
                       f"{score:.3f}", ha="center", fontsize=16, fontweight="bold",
                       color=colors[sorted_systems[i]])
        # Add pass rate below
        ax_leader.text(bar.get_x() + bar.get_width() / 2, -0.05,
                       f"{pr*100:.0f}%", ha="center", fontsize=14, fontweight="bold",
                       color="gray", va="top")
    
    ax_leader.set_xticks(x_leader)
    ax_leader.set_xticklabels([s.replace(" [", "\n[") for s in sorted_systems], 
                               fontsize=16, fontweight="bold", rotation=0)
    ax_leader.set_ylabel("Avg Score", fontsize=20, fontweight="bold")
    ax_leader.set_ylim(-0.15, 1.2)
    ax_leader.axhline(0.75, color="gray", linestyle="--", alpha=0.4, linewidth=2.0)
    ax_leader.tick_params(labelsize=16)
    ax_leader.grid(axis="y", alpha=0.3)

    plt.savefig(output_dir / "core_p1_executive.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# IMAGE 2 — Score Heatmap & Test Difficulty (high importance)
# ──────────────────────────────────────────────────────────────────────

def plot_core_p2_heatmap(results: List[Dict[str, Any]], output_dir: Path):
    """
    Priority-2 image: test × system score heatmap with difficulty sidebar.
    :param results: Test result dicts.
    :param output_dir: Output directory.
    """
    test_ids, systems, matrix, passed_matrix = _test_system_matrix(results)

    fig = plt.figure(figsize=(30, 18), facecolor="white")
    # Optimized: larger heatmap, compact sidebars, tighter spacing
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1], width_ratios=[3, 1, 1],
                           hspace=0.20, wspace=0.15,
                           left=0.04, right=0.98, top=0.98, bottom=0.04)

    # --- Main: Heatmap ---
    ax_hm = fig.add_subplot(gs[0, 0])
    n_tests = len(test_ids)
    n_sys = len(systems)
    grid = np.full((n_tests, n_sys), np.nan)
    for i, tid in enumerate(test_ids):
        for j, s in enumerate(systems):
            if (tid, s) in matrix:
                grid[i, j] = matrix[(tid, s)]

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#f0f0f0")
    im = ax_hm.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_hm.set_xticks(range(n_sys))
    ax_hm.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=14)
    ax_hm.set_yticks(range(n_tests))
    test_names = {}
    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "?")
        name = r.get("test_metadata", {}).get("test_name", tid)
        test_names[tid] = name
    ax_hm.set_yticklabels([f"{tid} -- {test_names.get(tid, tid)[:30]}" for tid in test_ids], fontsize=11)
    ax_hm.text(0.99, 1.01, "(higher is better)", transform=ax_hm.transAxes,
               fontsize=11, color="gray", ha="right", va="bottom")

    for i, tid in enumerate(test_ids):
        for j, s in enumerate(systems):
            if (tid, s) in matrix:
                val = matrix[(tid, s)]
                ok = passed_matrix.get((tid, s), False)
                mark = "v" if ok else "x"
                txt_color = "white" if val < 0.4 or val > 0.85 else "black"
                ax_hm.text(j, i, f"{val:.2f}\n{mark}",
                            ha="center", va="center", fontsize=11, color=txt_color,
                            fontweight="bold")

    cb = fig.colorbar(im, ax=ax_hm, fraction=0.03, pad=0.02)
    cb.set_label("Score", fontsize=14)
    cb.ax.tick_params(labelsize=14)

    # --- Right sidebar: Test difficulty ---
    ax_diff = fig.add_subplot(gs[0, 2])
    test_avg = []
    for tid in test_ids:
        scores = [matrix[(tid, s)] for s in systems if (tid, s) in matrix]
        test_avg.append(np.mean(scores) if scores else 0)

    bar_colors = [plt.cm.RdYlGn(v) for v in test_avg]
    ax_diff.barh(range(n_tests), test_avg, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax_diff.set_yticks(range(n_tests))
    ax_diff.set_yticklabels(test_ids, fontsize=14)
    ax_diff.set_xlim(0, 1.15)
    ax_diff.text(0.99, 0.99, "(higher is better)", transform=ax_diff.transAxes,
                 fontsize=11, color="gray", ha="right", va="top")
    ax_diff.tick_params(labelsize=14)
    for i, v in enumerate(test_avg):
        ax_diff.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=14, fontweight="bold")
    ax_diff.axvline(0.75, color="red", linestyle="--", alpha=0.4)
    ax_diff.grid(axis="x", alpha=0.3)

    # --- Bottom-left: Per-test pass rates grouped by system (spans 2 columns) ---
    ax_pass = fig.add_subplot(gs[1, :2])
    test_pass_rates = defaultdict(dict)
    for r in results:
        tid = r.get("test_metadata", {}).get("test_id", "?")
        label = _system_label(r)
        ok = r.get("validation", {}).get("overall_passed", False)
        test_pass_rates[tid][label] = 1 if ok else 0

    x = np.arange(n_tests)
    width = 0.8 / max(n_sys, 1)
    sys_colors = _variant_colors(systems)
    for j, s in enumerate(systems):
        vals = [test_pass_rates.get(tid, {}).get(s, 0) for tid in test_ids]
        ax_pass.bar(x + j * width - (n_sys - 1) * width / 2, vals,
                     width=width, color=sys_colors[s], edgecolor="black", linewidth=0.5,
                     label=s)
    ax_pass.set_xticks(x)
    ax_pass.set_xticklabels(test_ids, fontsize=14)
    ax_pass.set_ylabel("Pass (1) / Fail (0)", fontsize=16)
    ax_pass.text(0.99, 0.99, "(higher is better)", transform=ax_pass.transAxes,
                 fontsize=11, color="gray", ha="right", va="top")
    ax_pass.tick_params(labelsize=12)
    ax_pass.set_ylim(-0.1, 1.4)
    ax_pass.legend(fontsize=14, ncol=min(n_sys, 4), loc="upper right")
    ax_pass.grid(axis="y", alpha=0.3)

    # --- Bottom-right: difficulty ranking as text ---
    ax_rank = fig.add_subplot(gs[1, 2])
    ax_rank.axis("off")
    ranked = sorted(zip(test_ids, test_avg), key=lambda t: t[1])
    for i, (tid, avg) in enumerate(ranked):
        y = 0.95 - i * (0.85 / max(len(ranked), 1))
        marker = "[OK]" if avg >= 0.75 else "[~~]" if avg >= 0.5 else "[!!]"
        color = "#70AD47" if avg >= 0.75 else "#FFC000" if avg >= 0.5 else "#FF4444"
        ax_rank.text(0.05, y, f"{marker} {tid}: {avg:.2f}", fontsize=14,
                      color=color, fontweight="bold",
                      transform=ax_rank.transAxes, va="top")

    plt.savefig(output_dir / "core_p2_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# IMAGE 3 — Efficiency Dashboard (medium importance)
# ──────────────────────────────────────────────────────────────────────

def plot_core_p3_efficiency(results: List[Dict[str, Any]], output_dir: Path):
    """
    Priority-3 image: cost, time, tokens, graph-vs-sequential comparison.
    :param results: Test result dicts.
    :param output_dir: Output directory.
    """
    from agent.app.model_costs import format_cost

    systems, data = _collect_per_system(results)
    colors = _variant_colors(systems)

    fig = plt.figure(figsize=(30, 18), facecolor="white")
    # Optimized: tighter spacing, better use of space
    gs = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.20,
                           left=0.04, right=0.98, top=0.98, bottom=0.04,
                           width_ratios=[1, 1, 1.2], height_ratios=[1, 1])

    # --- Top-left: Cost per test by system ---
    ax_cost = fig.add_subplot(gs[0, 0])
    avg_costs = [np.mean(data[s]["costs"]) for s in systems]
    bars = ax_cost.bar(range(len(systems)), avg_costs,
                        color=[colors[s] for s in systems], edgecolor="black", linewidth=0.8)
    for i, (bar, c) in enumerate(zip(bars, avg_costs)):
        ax_cost.text(bar.get_x() + bar.get_width() / 2, c + max(avg_costs) * 0.03,
                      format_cost(c), ha="center", fontsize=14, fontweight="bold")
    ax_cost.set_xticks(range(len(systems)))
    ax_cost.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=14)
    ax_cost.set_ylabel("Avg Cost (USD)", fontsize=16)
    ax_cost.text(0.99, 0.99, "(lower is better)", transform=ax_cost.transAxes,
                 fontsize=11, color="gray", ha="right", va="top")
    ax_cost.tick_params(labelsize=14)
    ax_cost.grid(axis="y", alpha=0.3)

    # --- Top-middle: Time vs Score scatter ---
    ax_ts = fig.add_subplot(gs[0, 1])
    for s in systems:
        ax_ts.scatter(data[s]["durations"], data[s]["scores"],
                       color=colors[s], s=120, edgecolors="black", linewidth=0.6,
                       label=s, alpha=0.8, zorder=3)
    ax_ts.set_xlabel("Duration (s)", fontsize=16)
    ax_ts.set_ylabel("Score", fontsize=16)
    ax_ts.text(0.99, 0.99, "(higher score, lower time better)", transform=ax_ts.transAxes,
               fontsize=11, color="gray", ha="right", va="top")
    ax_ts.set_ylim(-0.05, 1.15)
    ax_ts.axhline(0.75, color="red", linestyle="--", alpha=0.4, label="Pass (0.75)")
    ax_ts.legend(fontsize=14, loc="lower right")
    ax_ts.tick_params(labelsize=14)
    ax_ts.grid(alpha=0.3)

    # --- Top-right: Avg tokens breakdown ---
    ax_tok = fig.add_subplot(gs[0, 2])
    avg_tokens = [np.mean(data[s]["tokens"]) for s in systems]
    bars = ax_tok.bar(range(len(systems)), avg_tokens,
                       color=[colors[s] for s in systems], edgecolor="black", linewidth=0.8)
    for i, (bar, t) in enumerate(zip(bars, avg_tokens)):
        ax_tok.text(bar.get_x() + bar.get_width() / 2, t + max(avg_tokens) * 0.03,
                     _format_tokens(t), ha="center", fontsize=14, fontweight="bold")
    ax_tok.set_xticks(range(len(systems)))
    ax_tok.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=14)
    ax_tok.set_ylabel("Avg Tokens", fontsize=16)
    ax_tok.text(0.99, 0.99, "(lower is better)", transform=ax_tok.transAxes,
                fontsize=11, color="gray", ha="right", va="top")
    ax_tok.tick_params(labelsize=14)
    ax_tok.grid(axis="y", alpha=0.3)

    # --- Bottom: Graph vs Sequential distribution (spans all 3 columns) ---
    ax_gvs = fig.add_subplot(gs[1, :3])
    ax_gvs.text(0.99, 0.99, f"(showing all {sum(len(data[s]['scores']) for s in systems)} test runs)",
                transform=ax_gvs.transAxes, fontsize=16, color="gray", ha="right", va="top", fontweight="bold")

    graph_sys = [s for s in systems if "[graph]" in s]
    seq_sys = [s for s in systems if "[sequential]" in s]

    if graph_sys and seq_sys:
        graph_scores_all = []
        seq_scores_all = []
        
        for s in graph_sys:
            graph_scores_all.extend(data[s]["scores"])
        for s in seq_sys:
            seq_scores_all.extend(data[s]["scores"])

        g_score = np.mean(graph_scores_all)
        s_score = np.mean(seq_scores_all)
        
        parts = ax_gvs.violinplot([seq_scores_all, graph_scores_all], positions=[0, 1], 
                                  widths=0.6, showmeans=True, showmedians=True)
        
        for pc, color in zip(parts['bodies'], ["#D94F3B", "#2B7BBA"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1.5)
        
        parts['cmeans'].set_colors(["#D94F3B", "#2B7BBA"])
        parts['cmeans'].set_linewidths(3)
        parts['cmedians'].set_colors(["#8B0000", "#1a5c2a"])
        parts['cmedians'].set_linewidths(2.5)
        
        ax_gvs.scatter([0] * len(seq_scores_all), seq_scores_all, 
                       s=25, alpha=0.4, color="#D94F3B", edgecolors="black", linewidth=0.3, zorder=3)
        ax_gvs.scatter([1] * len(graph_scores_all), graph_scores_all,
                       s=25, alpha=0.4, color="#2B7BBA", edgecolors="black", linewidth=0.3, zorder=3)
        
        ax_gvs.axhline(0.75, color="gray", linestyle="--", alpha=0.5, linewidth=2, label="Pass threshold")
        ax_gvs.axhline(g_score, color="#2B7BBA", linestyle=":", alpha=0.7, linewidth=2.5, 
                       label=f"Graph avg: {g_score:.3f}")
        ax_gvs.axhline(s_score, color="#D94F3B", linestyle=":", alpha=0.7, linewidth=2.5,
                       label=f"Seq avg: {s_score:.3f}")
        
        ax_gvs.set_xticks([0, 1])
        ax_gvs.set_xticklabels(["Sequential", "Graph"], fontsize=20, fontweight="bold")
        ax_gvs.set_ylabel("Score", fontsize=22, fontweight="bold")
        ax_gvs.set_ylim(-0.05, 1.15)
        ax_gvs.grid(axis="y", alpha=0.3)
        ax_gvs.legend(fontsize=16, loc="upper left")
        ax_gvs.tick_params(labelsize=18)
        
        delta = g_score - s_score
        delta_pct = (delta / s_score * 100) if s_score > 0 else 0
        ax_gvs.text(0.5, 0.05, f"Graph advantage: +{delta:.3f} (+{delta_pct:.1f}%)",
                    transform=ax_gvs.transAxes, fontsize=20, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#E3F2FD", edgecolor="#2196F3", linewidth=2.5),
                    va="bottom", ha="center")
    else:
        ax_gvs.text(0.5, 0.5, "Single variant only\n(no comparison available)",
                     ha="center", va="center", fontsize=20, color="gray", fontweight="bold",
                     transform=ax_gvs.transAxes)

    plt.savefig(output_dir / "core_p3_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# IMAGE 4 — Actions & Structure (detail importance)
# ──────────────────────────────────────────────────────────────────────

def plot_core_p4_details(results: List[Dict[str, Any]], output_dir: Path):
    """
    Priority-4 image: action counts, graph structure, validation checks, timing.
    :param results: Test result dicts.
    :param output_dir: Output directory.
    """
    systems, data = _collect_per_system(results)
    colors = _variant_colors(systems)

    fig = plt.figure(figsize=(30, 18), facecolor="white")
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.07, right=0.97, top=0.96, bottom=0.06)

    # --- Top-left: Avg searches & visits stacked bars ---
    ax_act = fig.add_subplot(gs[0, 0])
    avg_searches = [np.mean(data[s]["searches"]) for s in systems]
    avg_visits = [np.mean(data[s]["visits"]) for s in systems]
    x = np.arange(len(systems))
    w = 0.45
    b1 = ax_act.bar(x, avg_searches, w, color="#4472C4", edgecolor="black", linewidth=0.6, label="Searches")
    b2 = ax_act.bar(x, avg_visits, w, bottom=avg_searches, color="#ED7D31",
                     edgecolor="black", linewidth=0.6, label="Visits")
    for i, (s_val, v_val) in enumerate(zip(avg_searches, avg_visits)):
        total = s_val + v_val
        ax_act.text(i, total + 0.15, f"{total:.1f}", ha="center", fontsize=14, fontweight="bold")
    ax_act.set_xticks(x)
    ax_act.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=14)
    ax_act.set_ylabel("Avg Actions per Test", fontsize=16)
    ax_act.text(0.99, 0.99, "(higher is better)", transform=ax_act.transAxes,
                fontsize=11, color="gray", ha="right", va="top")
    ax_act.legend(fontsize=14)
    ax_act.tick_params(labelsize=14)
    ax_act.grid(axis="y", alpha=0.3)

    # --- Top-right: Graph structure (depth, branching, nodes) ---
    ax_gs = fig.add_subplot(gs[0, 1])
    avg_nodes = [np.mean(data[s]["nodes"]) for s in systems]
    avg_depth = [np.mean(data[s]["depth"]) for s in systems]
    avg_branch = [np.mean(data[s]["branching"]) for s in systems]

    bar_w = 0.25
    x_gs = np.arange(len(systems))
    ax_gs.bar(x_gs - bar_w, avg_nodes, bar_w, color="#4472C4", edgecolor="black", linewidth=0.6, label="Nodes")
    ax_gs.bar(x_gs, avg_depth, bar_w, color="#ED7D31", edgecolor="black", linewidth=0.6, label="Depth")
    ax_gs.bar(x_gs + bar_w, avg_branch, bar_w, color="#70AD47", edgecolor="black", linewidth=0.6, label="Branching")

    for i, (n, d, b) in enumerate(zip(avg_nodes, avg_depth, avg_branch)):
        ax_gs.text(i - bar_w, n + 0.15, f"{n:.1f}", ha="center", fontsize=14, fontweight="bold")
        ax_gs.text(i, d + 0.15, f"{d:.1f}", ha="center", fontsize=14, fontweight="bold")
        ax_gs.text(i + bar_w, b + 0.15, f"{b:.1f}", ha="center", fontsize=14, fontweight="bold")

    ax_gs.set_xticks(x_gs)
    ax_gs.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=14)
    ax_gs.set_ylabel("Count", fontsize=16)
    ax_gs.text(0.99, 0.99, "(informational)", transform=ax_gs.transAxes,
               fontsize=11, color="gray", ha="right", va="top")
    ax_gs.legend(fontsize=14)
    ax_gs.tick_params(labelsize=14)
    ax_gs.grid(axis="y", alpha=0.3)

    # --- Bottom-left: Validation check success rates ---
    ax_val = fig.add_subplot(gs[1, 0])
    check_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        validation = r.get("validation", {})
        for check in validation.get("grep_validations", []):
            name = check.get("check", "unknown")
            check_stats[name]["total"] += 1
            if check.get("passed", False):
                check_stats[name]["passed"] += 1

    if check_stats:
        check_names = sorted(check_stats.keys(), key=lambda n: check_stats[n]["passed"] / max(check_stats[n]["total"], 1))
        rates = [check_stats[n]["passed"] / max(check_stats[n]["total"], 1) for n in check_names]
        bar_colors = ["#70AD47" if r >= 0.75 else "#FFC000" if r >= 0.5 else "#FF4444" for r in rates]
        ax_val.barh(range(len(check_names)), rates, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax_val.set_yticks(range(len(check_names)))
        short_names = [n[:25] + "..." if len(n) > 25 else n for n in check_names]
        ax_val.set_yticklabels(short_names, fontsize=14)
        ax_val.set_xlim(0, 1.15)
        for i, r in enumerate(rates):
            ax_val.text(r + 0.02, i, f"{r:.0%}", va="center", fontsize=14, fontweight="bold")
        ax_val.axvline(0.75, color="red", linestyle="--", alpha=0.4)
    else:
        ax_val.text(0.5, 0.5, "No validation checks", ha="center", va="center",
                     fontsize=14, color="gray", transform=ax_val.transAxes)
    ax_val.set_xlabel("Success Rate", fontsize=16)
    ax_val.text(0.99, 0.99, "(higher is better)", transform=ax_val.transAxes,
                fontsize=11, color="gray", ha="right", va="top")
    ax_val.tick_params(labelsize=14)
    ax_val.grid(axis="x", alpha=0.3)

    # --- Bottom-right: Per-system duration breakdown ---
    ax_dur = fig.add_subplot(gs[1, 1])
    avg_dur = [np.mean(data[s]["durations"]) for s in systems]
    bars = ax_dur.bar(range(len(systems)), avg_dur,
                       color=[colors[s] for s in systems], edgecolor="black", linewidth=0.8)
    for i, (bar, d) in enumerate(zip(bars, avg_dur)):
        ax_dur.text(bar.get_x() + bar.get_width() / 2, d + max(avg_dur) * 0.03,
                     f"{d:.0f}s", ha="center", fontsize=14, fontweight="bold")
    ax_dur.set_xticks(range(len(systems)))
    ax_dur.set_xticklabels([s.replace(" [", "\n[") for s in systems], fontsize=14)
    ax_dur.set_ylabel("Avg Duration (s)", fontsize=16)
    ax_dur.text(0.99, 0.99, "(lower is better)", transform=ax_dur.transAxes,
                fontsize=11, color="gray", ha="right", va="top")
    ax_dur.tick_params(labelsize=14)
    ax_dur.grid(axis="y", alpha=0.3)

    plt.savefig(output_dir / "core_p4_details.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def _filter_valid(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop results with unknown/empty model or zero-duration (broken runs).
    :param results: Raw results.
    :returns: Filtered results.
    """
    return [r for r in results
            if str(r.get("model", "unknown")) != "unknown"
            and r.get("execution", {}).get("duration_seconds", 0) > 0]


def generate_core_plots(results: List[Dict[str, Any]], output_dir: Path):
    """
    Generate the 4 core summary images.
    :param results: Test result dicts.
    :param output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _filter_valid(results)
    if not results:
        print("[SKIP] No valid results after filtering unknown/broken runs")
        return

    plot_core_p1_executive(results, output_dir)
    print("[OK] core_p1_executive.png — Executive Summary")

    plot_core_p2_heatmap(results, output_dir)
    print("[OK] core_p2_heatmap.png — Test × Model Heatmap")

    plot_core_p3_efficiency(results, output_dir)
    print("[OK] core_p3_efficiency.png — Efficiency Dashboard")

    plot_core_p4_details(results, output_dir)
    print("[OK] core_p4_details.png — Actions & Structure")

    print(f"\n4 core images saved to: {output_dir}")
