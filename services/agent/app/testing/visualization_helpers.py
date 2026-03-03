"""
Helper functions for visualization.

Shared utilities used by visualization_plots, visualization_summary,
and the test runner summary output.
"""

from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Label / formatting helpers
# ---------------------------------------------------------------------------

def _system_label(result: Dict) -> str:
    """Build label like 'gpt-5-mini [graph]' for a result dict."""
    model = str(result.get("model", "unknown"))
    variant = str(result.get("execution_variant", "graph"))
    return f"{model} [{variant}]"


def _format_tokens(value: float) -> str:
    """Format token count: 1234 -> '1.2k', 950 -> '950'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return f"{int(value)}"


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _get_difficulty_colormap():
    """Orange (easy) -> blue (medium) -> reddish-purple (hard)."""
    colors = [(1.0, 0.5, 0.0), (0.0, 0.4, 0.8), (0.6, 0.2, 0.5)]
    return mcolors.LinearSegmentedColormap.from_list("difficulty", colors, N=256)


def _get_system_colors(models: List[str], colormap_name: str = "Set3") -> Dict[str, tuple]:
    """
    Assign colors to systems. Sequential variants get a darker shade
    of the same base-model color as the graph variant.
    """
    model_to_base = {}
    for label in models:
        base = label.split("[")[0].strip() if "[" in label else label
        model_to_base[label] = base

    unique_bases = sorted(set(model_to_base.values()))
    cmap = plt.cm.get_cmap(colormap_name)
    n = max(len(unique_bases), 1)
    base_colors = {b: cmap(i / n) for i, b in enumerate(unique_bases)}

    out: Dict[str, tuple] = {}
    for label in models:
        bc = base_colors[model_to_base[label]]
        if "[sequential]" in label:
            rgb = np.array(bc[:3]) * 0.4
            out[label] = tuple(rgb) + (bc[3],)
        else:
            out[label] = bc
    return out


# ---------------------------------------------------------------------------
# Graph structure analysis
# ---------------------------------------------------------------------------

def _extract_graph_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute depth and branching metrics from the graph stored in a result.

    Returns dict with:
      total_nodes, max_depth, avg_depth, leaf_count, internal_count,
      max_branching, avg_branching (non-leaf only), action_counts.
    All values default to 0/empty when graph data is missing.
    """
    nodes = result.get("execution", {}).get("graph", {}).get("nodes", {})
    if not nodes:
        return {
            "total_nodes": 0, "max_depth": 0, "avg_depth": 0.0,
            "leaf_count": 0, "internal_count": 0,
            "max_branching": 0, "avg_branching": 0.0,
            "action_counts": {},
        }

    # Depth of each node (root = 0).
    depths: Dict[str, int] = {}
    for nid in nodes:
        if nid in depths:
            continue
        chain: List[str] = [nid]
        pid = nodes[nid].get("parent_id")
        while pid and pid in nodes and pid not in depths:
            chain.append(pid)
            pid = nodes[pid].get("parent_id")
        base = depths[pid] + 1 if pid and pid in depths else 0
        for i, cid in enumerate(reversed(chain)):
            depths[cid] = base + i

    # Branching factor of internal nodes (nodes with children).
    branching = [len(n.get("children", [])) for n in nodes.values() if n.get("children")]
    leaf_count = sum(1 for n in nodes.values() if not n.get("children"))

    # Action type counts.
    action_counts: Counter = Counter()
    for n in nodes.values():
        details = n.get("details", {})
        action = details.get("action") or details.get("_action")
        if action:
            action_counts[str(action)] += 1

    return {
        "total_nodes": len(nodes),
        "max_depth": max(depths.values()) if depths else 0,
        "avg_depth": sum(depths.values()) / len(depths) if depths else 0.0,
        "leaf_count": leaf_count,
        "internal_count": len(nodes) - leaf_count,
        "max_branching": max(branching) if branching else 0,
        "avg_branching": sum(branching) / len(branching) if branching else 0.0,
        "action_counts": dict(action_counts),
    }
