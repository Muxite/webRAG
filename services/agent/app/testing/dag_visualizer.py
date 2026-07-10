"""
DAG visualizer — draw the *whole* graph as a high-resolution, square image.

This renders two related structures the rest of the system only ever exposed as ASCII or
text before:

  * **compiled plans** (the ``graph_compiled`` scaffold): leaves + ``depends_on`` edges +
    an aggregation step. Laid out by topological wave so the pay-once-offline DAG that is
    the project's headline mechanism is finally *visible*.
  * **runtime reasoning graphs** (``IdeaDag.to_dict()``): nodes + parent/child edges, laid
    out by longest-path rank from the root.

Design constraints (set by the product owner):
  * output is **square** (1:1 aspect) by default,
  * at a **configurable** resolution that is **>= 1920 px** on a side,
  * dependency-free beyond matplotlib (no graphviz / networkx), so it runs anywhere the
    benchmark already runs.

The layout is a deterministic layered ("Sugiyama-lite") placement: assign every node a
rank (wave for plans, longest-path depth for runtime graphs), spread ranks top-to-bottom,
order nodes within a rank by the barycenter of their parents to cut edge crossings, then
draw rounded node boxes and arrowed edges. Pure geometry — no I/O except the final
``savefig`` — so it is unit-testable headless (Agg backend).
"""
from __future__ import annotations

import os
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import logging as _logging

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_logging.getLogger("matplotlib.font_manager").setLevel(_logging.WARNING)  # silence font spam

from agent.app.testing.compiled_plan import normalize_plan, topological_waves

# --- resolution policy -------------------------------------------------------------------
MIN_SIDE_PX = 1920           # hard floor on the exported image's shorter side
DEFAULT_SIDE_PX = 1920       # default square side
DEFAULT_DPI = 200            # figsize = side_px / dpi (kept modest so fonts stay crisp)

# --- palette (wave-indexed, color-blind-friendly-ish, dark text on light fills) ----------
_WAVE_FILLS = [
    "#cfe3ff", "#cfeede", "#ffe6c7", "#ead8ff", "#ffd6dd",
    "#d4f0f0", "#f3e6c0", "#dfe7d5", "#f6d8ec", "#d8dcf0",
]
_ROOT_FILL = "#2f3b52"       # mandate node (dark, light text)
_AGG_FILL = "#1f6f54"        # aggregation node (accent, light text)
_FAIL_FILL = "#f3b6b6"       # failed leaf
_DONE_EDGE = "#1f6f54"
_EDGE_DEP = "#3a3a3a"        # solid: a data dependency (depends_on)
_EDGE_ROLLUP = "#9aa0a8"     # light: a fact rolling up into aggregation


class _Node:
    __slots__ = ("id", "label", "sub", "kind", "rank", "x", "y", "status", "score")

    def __init__(self, nid: str, label: str, sub: str = "", kind: str = "leaf",
                 status: Optional[str] = None, score: Optional[float] = None) -> None:
        self.id = nid
        self.label = label
        self.sub = sub
        self.kind = kind          # "root" | "leaf" | "agg"
        self.status = status      # "done" | "failed" | None
        self.score = score
        self.rank = 0
        self.x = 0.0
        self.y = 0.0


def _rank_by_barycenter(nodes: Dict[str, _Node], edges: Sequence[Tuple[str, str]],
                        ranks: List[List[str]]) -> None:
    """Place every node's (x, y); reorder within each rank by parent barycenter.

    y descends with rank (rank 0 at top). x is spread evenly across the unit width. Two
    barycenter sweeps (down then up) are enough to untangle the small DAGs we draw.
    """
    n_ranks = max(1, len(ranks))
    parents: Dict[str, List[str]] = {nid: [] for nid in nodes}
    children: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for a, b in edges:
        if a in nodes and b in nodes:
            parents[b].append(a)
            children[a].append(b)

    order = [list(r) for r in ranks]

    def _assign_x() -> None:
        for r, layer in enumerate(order):
            width = max(1, len(layer))
            for i, nid in enumerate(layer):
                nodes[nid].x = (i + 0.5) / width
                nodes[nid].rank = r
                nodes[nid].y = 1.0 - (r + 0.5) / n_ranks

    _assign_x()
    for _ in range(2):
        for r in range(1, len(order)):              # downward: order by parents' x
            order[r].sort(key=lambda nid: (
                sum(nodes[p].x for p in parents[nid]) / len(parents[nid]) if parents[nid] else nodes[nid].x))
            _assign_x()
        for r in range(len(order) - 2, -1, -1):     # upward: order by children's x
            order[r].sort(key=lambda nid: (
                sum(nodes[c].x for c in children[nid]) / len(children[nid]) if children[nid] else nodes[nid].x))
            _assign_x()


# A single rank wider than this many boxes is reflowed onto multiple stacked rows so each
# box keeps a legible width. Chosen to echo the mixed DAG's ~6-per-row waves; a 16-leaf pure
# fan-out becomes two balanced rows of 8 rather than sixteen unreadable slivers.
MAX_ROW_WIDTH = 8


def _reflow_wide_ranks(ranks: List[List[str]]) -> List[List[str]]:
    """Split any rank wider than ``MAX_ROW_WIDTH`` into balanced stacked sub-rows.

    A wide single wave (e.g. a 16-way fan-out) is unreadable as one row of slivers, so it is
    laid out as a grid: N balanced rows, each <= ``MAX_ROW_WIDTH`` boxes. Narrow ranks pass
    through untouched, so multi-wave dependency DAGs are unaffected.
    """
    out: List[List[str]] = []
    for layer in ranks:
        if len(layer) <= MAX_ROW_WIDTH:
            out.append(list(layer))
            continue
        rows = (len(layer) + MAX_ROW_WIDTH - 1) // MAX_ROW_WIDTH
        per = (len(layer) + rows - 1) // rows        # balanced chunk size
        for i in range(0, len(layer), per):
            out.append(list(layer[i:i + per]))
    return out


def _wrap(text: str, width_chars: int, max_lines: int) -> str:
    if not text:
        return ""
    lines = textwrap.wrap(text, width=max(6, width_chars))
    if len(lines) > max_lines:
        lines = lines[: max_lines]
        lines[-1] = lines[-1].rstrip(".,; ") + "…"
    return "\n".join(lines)


# --- rendered-width text fitting ---------------------------------------------------------
# The old sizing used a fixed chars-per-point heuristic that under-estimated real glyph
# widths (badly for bold text), so long titles and long node ids overflowed their box and
# even the canvas at 4K. These helpers measure the *actual* rendered pixel width against the
# available box/figure width, so text is always wrapped/shrunk/truncated to genuinely fit.

# Prefer breaking after whitespace, underscores, hyphens and slashes (keeps node ids like
# ``author_pride_and_prejudice`` breaking at the underscores instead of mid-word).
_BREAK_RE = re.compile(r"[^\s_\-/]+[\s_\-/]*")


def _measure(fig, renderer, s: str, fontsize: float, bold: bool) -> float:
    """Rendered width of ``s`` in figure pixels (== side_px units for our square canvas)."""
    if not s:
        return 0.0
    t = fig.text(0.0, 0.0, s, fontsize=fontsize, fontweight="bold" if bold else "normal")
    try:
        return float(t.get_window_extent(renderer).width)
    finally:
        t.remove()


def _char_split(s: str, max_px: float, w: Callable[[str], float]) -> List[str]:
    """Hard-split a single overlong token to fit ``max_px``, char by char."""
    out: List[str] = []
    cur = ""
    for ch in s:
        cand = cur + ch
        if not cur or w(cand) <= max_px:
            cur = cand
        else:
            out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def _wrap_lines(fig, renderer, text: str, max_px: float, fontsize: float,
                bold: bool) -> List[str]:
    """Wrap ``text`` so every line's rendered width is <= ``max_px`` (unbounded line count)."""
    text = " ".join(str(text).split())          # collapse whitespace/newlines
    if not text:
        return []

    def w(s: str) -> float:
        return _measure(fig, renderer, s, fontsize, bold)

    lines: List[str] = []
    cur = ""
    for tok in (_BREAK_RE.findall(text) or [text]):
        cand = cur + tok
        if not cur or w(cand) <= max_px:
            cur = cand
        else:
            lines.append(cur.rstrip())
            cur = tok
    if cur.strip():
        lines.append(cur.rstrip())
    # A single token wider than the box still overflows -> hard char-split those lines.
    out: List[str] = []
    for ln in lines:
        if len(ln) <= 1 or w(ln) <= max_px:
            out.append(ln)
        else:
            out.extend(_char_split(ln, max_px, w))
    return out


def _ellipsize(s: str, max_px: float, w: Callable[[str], float]) -> str:
    """Trim ``s`` from the end and append an ellipsis so it fits ``max_px``."""
    s = s.rstrip(" .,;:")
    if w(s + "…") <= max_px:
        return s + "…"
    while s and w(s + "…") > max_px:
        s = s[:-1]
    return (s + "…") if s else "…"


def _fit_box_text(fig, renderer, text: str, max_px: float, fontsize: float, bold: bool,
                  max_lines: int) -> str:
    """Wrap ``text`` to fit ``max_px`` per line at a fixed font size, capping at ``max_lines``
    with an ellipsis on the last line (used for node ids and box bodies)."""
    lines = _wrap_lines(fig, renderer, text, max_px, fontsize, bold)
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _ellipsize(lines[-1], max_px,
                               lambda s: _measure(fig, renderer, s, fontsize, bold))
    return "\n".join(lines)


def _fit_title(fig, renderer, text: str, max_px: float, fontsize: float, *,
               max_lines: int = 2, min_fs: float = 16.0) -> Tuple[str, float]:
    """Shrink (and, if needed, wrap onto <= ``max_lines`` lines) a title/subtitle so it fits
    ``max_px`` in full — no truncation until the font floor is genuinely exhausted."""
    text = " ".join(str(text).split())
    if not text:
        return "", fontsize
    fs = float(fontsize)
    while fs >= min_fs:
        lines = _wrap_lines(fig, renderer, text, max_px, fs, True)
        if len(lines) <= max_lines:
            return "\n".join(lines), fs
        fs -= 1.0
    # Font floor reached: wrap at min_fs and ellipsize the overflow as a last resort.
    return _fit_box_text(fig, renderer, text, max_px, min_fs, True, max_lines), min_fs


def render_layered_dag(
    nodes: Dict[str, _Node],
    edges: Sequence[Tuple[str, str]],
    ranks: List[List[str]],
    *,
    title: str = "",
    subtitle: str = "",
    side_px: int = DEFAULT_SIDE_PX,
    dpi: int = DEFAULT_DPI,
    out_path: str,
    rollup_edges: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    """Draw a layered DAG to a square PNG and return ``out_path``.

    ``edges`` are solid data dependencies; ``rollup_edges`` (optional) are drawn light to
    show facts feeding the aggregation node. ``side_px`` is clamped up to ``MIN_SIDE_PX``.
    """
    side_px = max(MIN_SIDE_PX, int(side_px))
    inches = side_px / float(dpi)
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ranks = _reflow_wide_ranks(ranks)              # wrap over-wide waves into stacked rows
    _rank_by_barycenter(nodes, list(edges) + list(rollup_edges or []), ranks)

    n_ranks = max(1, len(ranks))
    width_by_rank = [max(1, len(r)) for r in ranks]
    px_per_pt = dpi / 72.0

    # Box width is per-rank (a lone node in its rank gets a wide box, capped so a chain is a
    # centered column rather than full-width bars); height is uniform so edge endpoints line up.
    def _box_w(rank: int) -> float:
        return min(0.46, (1.0 / width_by_rank[rank]) * 0.86)

    box_h = min(0.15, (1.0 / n_ranks) * 0.62)
    bh_px = box_h * side_px
    # Font is driven by vertical room (clamped); wrap width then follows from the box's
    # pixels. The gallery exports a 3840px square viewed small (LinkedIn/slide), so the type
    # is scaled up well past the old caps -- a title lands ~48pt, node labels ~30pt at 4K.
    title_fs = max(28.0, bh_px * 0.20 / px_per_pt)

    # Measure against the real renderer so long titles/ids never spill past the canvas.
    renderer = fig.canvas.get_renderer()
    header_frac = 0.0                              # vertical room the header actually consumes
    if title:
        title_text, title_fs = _fit_title(fig, renderer, title, side_px * 0.94, title_fs,
                                           max_lines=2)
        n_tlines = title_text.count("\n") + 1
        fig.text(0.5, 0.988, title_text, ha="center", va="top", fontsize=title_fs,
                 fontweight="bold", color="#1b1b1b", linespacing=1.04)
        header_frac += n_tlines * (title_fs * px_per_pt / side_px) * 1.06
    if subtitle:
        sub_fs = title_fs * 0.58
        sub_text, sub_fs = _fit_title(fig, renderer, subtitle, side_px * 0.94, sub_fs,
                                      max_lines=1)
        sub_y = 0.984 - header_frac - 0.006
        fig.text(0.5, sub_y, sub_text, ha="center", va="top", fontsize=sub_fs,
                 color="#555555")
        header_frac += (sub_fs * px_per_pt / side_px) * 1.4
    top_pad = min(0.16, 0.02 + header_frac) if (title or subtitle) else 0.0
    for nd in nodes.values():          # squeeze down so the header never collides with rank 0
        nd.y = nd.y * (1.0 - top_pad)

    def _xy(nid: str) -> Tuple[float, float]:
        return nodes[nid].x, nodes[nid].y

    # edges first (under the boxes); line/arrow widths scale with the canvas.
    esc = max(1.0, side_px / 1920.0)
    for a, b in rollup_edges or []:
        if a in nodes and b in nodes:
            _draw_edge(ax, _xy(a), _xy(b), box_h, color=_EDGE_ROLLUP, lw=1.4 * esc,
                       alpha=0.5, arrow=False, scale=esc)
    for a, b in edges:
        if a in nodes and b in nodes:
            _draw_edge(ax, _xy(a), _xy(b), box_h, color=_EDGE_DEP, lw=2.2 * esc,
                       alpha=0.9, arrow=True, scale=esc)

    # nodes
    for nd in nodes.values():
        fill, txt_color, edge_color, edge_lw = _style(nd)
        x, y = nd.x, nd.y
        box_w = _box_w(nd.rank)
        bw_px = box_w * side_px
        # Base node font is kept deliberately modest: the boxes carry long snake_case ids
        # (``great_wall_length``) and multi-line expect text, so an oversized base font forced
        # almost everything into ``author_pr…`` ellipsis. Smaller base -> most ids show in full
        # while staying legible when the 4K square is viewed ~1000-1400px wide.
        fs = min(max(11.0, bh_px * 0.108 / px_per_pt), 27.0)
        id_fs, sub_fs = fs * 1.06, fs * 0.84
        # Text must fit *inside* the box: measure against the box's inner width in px.
        inner_px = bw_px * 0.88
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle=f"round,pad=0.004,rounding_size={min(box_w, box_h) * 0.16:.4f}",
            linewidth=edge_lw, edgecolor=edge_color, facecolor=fill, zorder=3,
        )
        ax.add_patch(box)
        label = _fit_box_text(fig, renderer, nd.label, inner_px, id_fs, True, 2)
        sub = _fit_box_text(fig, renderer, nd.sub, inner_px, sub_fs, False, 3)
        if sub:
            ax.text(x, y + box_h * 0.24, label, ha="center", va="center",
                    fontsize=id_fs, fontweight="bold", color=txt_color, zorder=4,
                    linespacing=1.05)
            ax.text(x, y - box_h * 0.16, sub, ha="center", va="center",
                    fontsize=sub_fs, color=txt_color, zorder=4, linespacing=1.1)
        else:
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=id_fs, fontweight="bold", color=txt_color, zorder=4,
                    linespacing=1.05)
        if nd.score is not None:
            ax.text(x + box_w * 0.47, y + box_h * 0.47, f"{nd.score:.2f}",
                    ha="right", va="top", fontsize=sub_fs * 0.95, color="#0d3a2b",
                    fontweight="bold", zorder=5)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def _style(nd: _Node) -> Tuple[str, str, str, float]:
    if nd.kind == "root":
        return _ROOT_FILL, "#ffffff", "#11151f", 2.0
    if nd.kind == "agg":
        return _AGG_FILL, "#ffffff", "#0d3a2b", 2.0
    if nd.status == "failed":
        return _FAIL_FILL, "#3a1212", "#a33", 2.0
    fill = _WAVE_FILLS[nd.rank % len(_WAVE_FILLS)]
    edge = _DONE_EDGE if nd.status == "done" else "#5a6472"
    return fill, "#16202b", edge, 2.2 if nd.status == "done" else 1.6


def _draw_edge(ax, p_from: Tuple[float, float], p_to: Tuple[float, float], box_h: float,
               *, color: str, lw: float, alpha: float, arrow: bool, scale: float = 1.0) -> None:
    """Arrow from the bottom of the upper box to the top of the lower box."""
    x0, y0 = p_from
    x1, y1 = p_to
    start = (x0, y0 - box_h / 2)
    end = (x1, y1 + box_h / 2)
    style = "-|>" if arrow else "-"
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=(14 * scale) if arrow else 0,
        connectionstyle="arc3,rad=0.04", color=color, lw=lw, alpha=alpha, zorder=1,
    )
    ax.add_patch(patch)


# --- adapters ----------------------------------------------------------------------------

def render_plan_dag(
    plan: Dict[str, Any],
    *,
    title: Optional[str] = None,
    mandate: Optional[str] = None,
    side_px: int = DEFAULT_SIDE_PX,
    dpi: int = DEFAULT_DPI,
    out_path: str,
    leaf_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Render a compiled plan (schema v2) as a wave-layered DAG.

    Rank 0 is the mandate; the next ranks are the topological waves; the final rank is the
    aggregation step. ``depends_on`` edges are solid; every leaf also rolls up (light edge)
    into aggregation. ``leaf_results`` (optional, from a run) colors leaves done/failed and
    prints per-leaf scores.
    """
    norm = normalize_plan(plan)
    leaves = norm["leaves"]
    waves = topological_waves(leaves)
    by_id = {lf["id"]: lf for lf in leaves}

    nodes: Dict[str, _Node] = {}
    nodes["__root__"] = _Node("__root__", "MANDATE",
                              mandate or "research task", kind="root")
    for lf in leaves:
        res = (leaf_results or {}).get(lf["id"]) or {}
        nodes[lf["id"]] = _Node(
            lf["id"], lf["id"], lf.get("expect", ""), kind="leaf",
            status=res.get("status"), score=res.get("score"),
        )
    nodes["__agg__"] = _Node("__agg__", "AGGREGATE", norm["aggregation"], kind="agg")

    edges: List[Tuple[str, str]] = []
    rollup: List[Tuple[str, str]] = []
    for lf in leaves:
        if lf["depends_on"]:
            for dep in lf["depends_on"]:
                if dep in by_id:
                    edges.append((dep, lf["id"]))
        else:
            edges.append(("__root__", lf["id"]))     # wave-0 leaves hang off the mandate
        rollup.append((lf["id"], "__agg__"))

    ranks: List[List[str]] = [["__root__"]] + [list(w) for w in waves] + [["__agg__"]]

    dep_edge_count = sum(len(lf["depends_on"]) for lf in leaves)  # real deps, not root->leaf
    shape = f"{len(leaves)} leaves · {len(waves)} wave(s) {[len(w) for w in waves]} · {dep_edge_count} dep edge(s)"
    return render_layered_dag(
        nodes, edges, ranks,
        title=title or "Compiled scaffold DAG",
        subtitle=shape,
        side_px=side_px, dpi=dpi, out_path=out_path, rollup_edges=rollup,
    )


def render_runtime_dag(
    graph: Dict[str, Any],
    *,
    title: Optional[str] = None,
    side_px: int = DEFAULT_SIDE_PX,
    dpi: int = DEFAULT_DPI,
    out_path: str,
) -> str:
    """Render a runtime ``IdeaDag.to_dict()`` graph by longest-path rank from the root."""
    raw_nodes = (graph or {}).get("nodes") or {}
    if not isinstance(raw_nodes, dict):
        raw_nodes = {n.get("id"): n for n in raw_nodes if isinstance(n, dict)}

    edges: List[Tuple[str, str]] = []
    for nid, n in raw_nodes.items():
        for child in (n.get("children") or []):
            if child in raw_nodes:
                edges.append((nid, child))

    rank = _longest_path_rank(raw_nodes, edges)
    max_rank = max(rank.values(), default=0)
    ranks: List[List[str]] = [[] for _ in range(max_rank + 1)]
    for nid in raw_nodes:
        ranks[rank[nid]].append(nid)

    nodes: Dict[str, _Node] = {}
    for nid, n in raw_nodes.items():
        details = n.get("details") or {}
        action = details.get("action") or n.get("action") or ""
        nodes[nid] = _Node(
            nid, (n.get("title") or nid)[:60],
            (f"[{action}]" if action else ""),
            kind="leaf",
            status=n.get("status"),
            score=n.get("score"),
        )
    return render_layered_dag(
        nodes, edges, ranks,
        title=title or "Runtime reasoning DAG",
        subtitle=f"{len(raw_nodes)} nodes · {len(edges)} edges · depth {max_rank}",
        side_px=side_px, dpi=dpi, out_path=out_path,
    )


def render_plan_structure(
    structure: Dict[str, Any],
    *,
    title: Optional[str] = None,
    mandate: Optional[str] = None,
    aggregation: Optional[str] = None,
    side_px: int = DEFAULT_SIDE_PX,
    dpi: int = DEFAULT_DPI,
    out_path: str,
) -> str:
    """Render a compiled DAG from a saved ``plan_structure`` (waves + ``a->b`` edges only).

    Compiled benchmark runs persist the *topology* (``plan_structure``) but not the leaf
    instructions, so this is the renderer the run-artifact CLI uses. Leaf ids carry no
    ``expect`` text here — the value is the shape (waves, dependencies, roll-up).
    """
    waves: List[List[str]] = [list(w) for w in (structure.get("waves") or [])]
    edge_strs: List[str] = list(structure.get("edges") or [])
    dep_edges: List[Tuple[str, str]] = []
    for e in edge_strs:
        if "->" in e:
            a, b = e.split("->", 1)
            dep_edges.append((a.strip(), b.strip()))
    has_dep_target = {b for _, b in dep_edges}

    nodes: Dict[str, _Node] = {"__root__": _Node("__root__", "MANDATE",
                               mandate or "research task", kind="root")}
    edges: List[Tuple[str, str]] = []
    rollup: List[Tuple[str, str]] = []
    for w_idx, wave in enumerate(waves):
        for lid in wave:
            nodes[lid] = _Node(lid, lid, "", kind="leaf")
            if lid not in has_dep_target:        # no upstream dependency -> hangs off mandate
                edges.append(("__root__", lid))
            rollup.append((lid, "__agg__"))
    edges.extend((a, b) for a, b in dep_edges if a in nodes and b in nodes)
    nodes["__agg__"] = _Node("__agg__", "AGGREGATE",
                             aggregation or "combine facts → final answer", kind="agg")

    ranks: List[List[str]] = [["__root__"]] + waves + [["__agg__"]]
    shape = (f"{structure.get('leaf_count', sum(len(w) for w in waves))} leaves · "
             f"{len(waves)} wave(s) {structure.get('wave_widths', [len(w) for w in waves])} · "
             f"{len(dep_edges)} dep edge(s)")
    return render_layered_dag(
        nodes, edges, ranks,
        title=title or "Compiled scaffold DAG", subtitle=shape,
        side_px=side_px, dpi=dpi, out_path=out_path, rollup_edges=rollup,
    )


def render_result_file(path: str, *, out_path: str, side_px: int = DEFAULT_SIDE_PX,
                       dpi: int = DEFAULT_DPI) -> str:
    """Render whatever DAG a saved artifact contains: a compiled plan, a benchmark result
    (compiled ``plan_structure`` or native runtime ``graph``). Auto-detected."""
    import json
    data = json.load(open(path))

    if isinstance(data, dict) and data.get("leaves"):           # a bare compiled plan file
        return render_plan_dag(data, side_px=side_px, dpi=dpi, out_path=out_path)

    ex = data.get("execution") or {}
    out = ex.get("output") or {}
    meta = data.get("test_metadata") or {}
    mandate = meta.get("statement") or meta.get("task") or meta.get("name")
    model = data.get("model", "")
    variant = data.get("execution_variant", "")

    structure = out.get("plan_structure")
    if structure and structure.get("waves"):
        title = f"{variant or 'graph_compiled'} · {meta.get('id', '')} · {model}".strip(" ·")
        return render_plan_structure(structure, title=title, mandate=mandate,
                                     side_px=side_px, dpi=dpi, out_path=out_path)

    graph = ex.get("graph") or data.get("graph")
    if graph and (graph.get("nodes")):
        title = f"{variant or 'graph'} · {meta.get('id', '')} · {model}".strip(" ·")
        return render_runtime_dag(graph, title=title, side_px=side_px, dpi=dpi, out_path=out_path)

    raise ValueError(f"{path}: no compiled plan_structure and no runtime graph to render")


def _longest_path_rank(raw_nodes: Dict[str, Any], edges: Sequence[Tuple[str, str]]) -> Dict[str, int]:
    """Rank = longest path from any root. Stable for DAGs; tolerant of cycles (caps depth)."""
    children: Dict[str, List[str]] = {nid: [] for nid in raw_nodes}
    indeg: Dict[str, int] = {nid: 0 for nid in raw_nodes}
    for a, b in edges:
        children[a].append(b)
        indeg[b] += 1
    rank = {nid: 0 for nid in raw_nodes}
    # Kahn-style longest path; if a cycle leaves nodes unprocessed, they keep rank 0.
    from collections import deque
    q = deque([nid for nid, d in indeg.items() if d == 0])
    seen = 0
    while q:
        nid = q.popleft()
        seen += 1
        for c in children[nid]:
            if rank[c] < rank[nid] + 1:
                rank[c] = rank[nid] + 1
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return rank


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI: render a DAG from a saved artifact (compiled plan, or a benchmark result JSON).

        python -m agent.app.testing.dag_visualizer RESULT.json -o dag.png --size 1920
    """
    import argparse
    p = argparse.ArgumentParser(description="Draw the whole DAG (compiled plan or runtime graph).")
    p.add_argument("input", help="a compiled-plan JSON, or a benchmark result JSON")
    p.add_argument("-o", "--out", default=None, help="output PNG (default: <input>.dag.png)")
    p.add_argument("--size", type=int, default=DEFAULT_SIDE_PX,
                   help=f"square side in px (clamped to >= {MIN_SIDE_PX})")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = p.parse_args(argv)
    out = args.out or (os.path.splitext(args.input)[0] + ".dag.png")
    path = render_result_file(args.input, out_path=out, side_px=args.size, dpi=args.dpi)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
