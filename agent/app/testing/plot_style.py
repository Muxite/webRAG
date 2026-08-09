"""
Shared matplotlib style for the benchmark gallery (barrage24b and successors).

Implements this repo's house data-viz system (see the ``dataviz`` skill) for static,
square, resolution-independent PNGs:

  * the validated categorical palette (8 hues, fixed order -- never cycled),
  * a one-hue sequential ramp for continuous 0..1 magnitudes (e.g. a score heatmap),
  * a single frozen light chart surface + recessive chrome (gridlines, spines, ticks),
  * :func:`square_fig`, which returns a figure sized for an exact 1:1 export at any
    ``side_px`` with font sizes pre-scaled to that canvas -- the same math
    ``recovery_curve.py`` already used for its Pareto plot, factored out so every new
    plot (heatmap, scatter, trend bars) looks like one consistent system.

Every plot module in this package should build its figure through this helper rather
than calling ``plt.subplots`` directly, so a resize (1920 -> 3840) never requires
per-plot font tuning.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- surface & ink (light mode; the gallery targets a single frozen surface) --------------
SURFACE = "#ffffff"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#3f3d3a"
INK_MUTED = "#7a7873"
GRIDLINE = "#e4e2dc"
BASELINE = "#b8b6ac"

# --- categorical: the MAGMA family (this deliverable's house palette). Distinct, well
# separated steps sampled out of matplotlib's magma colormap, avoiding the near-black and
# near-white extremes (poor contrast on a white surface). Fixed order -- never cycle past
# the sampled roster; a 9th series folds into "Other" or small multiples. ------------------
def _magma_categorical(n: int = 8) -> List[str]:
    from matplotlib import colormaps
    import numpy as _np
    cmap = colormaps["magma"]
    xs = _np.linspace(0.14, 0.86, n)
    return [matplotlib.colors.to_hex(cmap(x)) for x in xs]


CATEGORICAL = _magma_categorical(8)

# --- sequential ramp: magma (dark purple -> bright yellow) for continuous magnitude only
# (e.g. a 0..1 score heatmap): higher = brighter = "hotter/better". Never for identity. ----
SEQUENTIAL_NAME = "magma"

# --- status: fixed, reserved meaning -- never reused for series identity -------------------
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

DEFAULT_SIDE_PX = 3840   # the gallery's square-4K standard
MIN_SIDE_PX = 1920       # hard floor shared with dag_visualizer.py / recovery_curve.py


def categorical_color(idx: int) -> str:
    """Fixed-order categorical slot. Assign in sequence; never cycle past 8 series --
    fold a 9th into 'Other' or small multiples instead."""
    return CATEGORICAL[idx % len(CATEGORICAL)]


def sequential_cmap(name: str = SEQUENTIAL_NAME):
    """The house sequential colormap (magma) for continuous-magnitude heatmaps."""
    from matplotlib import colormaps
    return colormaps[name].copy()


def font_scale(side_px: int) -> float:
    """Scale factor relative to the 4K reference canvas (1.0 at 3840px)."""
    return max(side_px, MIN_SIDE_PX) / float(DEFAULT_SIDE_PX)


def font_sizes(side_px: int) -> Dict[str, float]:
    """Font sizes pre-scaled for this canvas.

    The gallery exports a 3840px square but is *viewed* at roughly 1080-1400px (LinkedIn
    embed / slide). So the type is scaled up ~2.5-3x versus what would look right on a
    native 1080p plot -- at 3840px a title is ~80pt, axis labels ~56pt, ticks ~44pt --
    otherwise everything reads tiny and thin once the image is displayed at normal size.
    """
    scale = font_scale(side_px)
    return {
        "title": max(34.0, 66 * scale),
        "subtitle": max(24.0, 48 * scale),
        "label": max(28.0, 54 * scale),
        "tick": max(22.0, 43 * scale),
        "legend": max(20.0, 39 * scale),
        "annot": max(22.0, 44 * scale),
    }


def mark_sizes(side_px: int) -> Dict[str, float]:
    """Line widths / marker sizes / edge widths pre-scaled to the same canvas as the fonts,
    so data marks read as bold, distinct shapes (not thin hairlines / tiny dots) when the
    4K export is viewed small."""
    scale = font_scale(side_px)
    return {
        "line": max(3.0, 6.5 * scale),        # data-line width (pt)
        "marker": max(11.0, 22 * scale),      # scatter/line marker diameter (pt, for `ms`)
        "scatter": max(120.0, 620 * scale),   # scatter area (pt^2, for `s`)
        "edge": max(1.4, 2.6 * scale),        # mark edge / bar edge line (pt)
        "cap": max(6.0, 13 * scale),          # errorbar cap size
        "eline": max(1.6, 3.2 * scale),       # errorbar line width
        "grid": max(0.8, 1.6 * scale),        # gridline width
    }


def style_axes(ax) -> None:
    """House chrome: light surface, muted gridlines/spines, recessive ticks/ink."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, linestyle=":", linewidth=2.0, color=GRIDLINE, alpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(2.0)
    ax.tick_params(colors=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)


def square_fig(side_px: int = DEFAULT_SIDE_PX, dpi: int = 160, *,
               margins: Tuple[float, float, float, float] = (0.11, 0.97, 0.92, 0.24),
               style: bool = True):
    """A square figure sized for an exact 1:1 export at ``side_px``.

    ``margins`` = (left, right, top, bottom) fractions passed to ``subplots_adjust``;
    the default reserves a bottom strip for a below-axes legend. Returns
    ``(fig, ax, fs)`` where ``fs`` is :func:`font_sizes` for this canvas -- use it for
    every title/label/tick/legend call so the plot scales cleanly from 1920 to 3840px.
    """
    side_px = max(MIN_SIDE_PX, int(side_px))
    inches = side_px / float(dpi)
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    left, right, top, bottom = margins
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    if style:
        style_axes(ax)
    return fig, ax, font_sizes(side_px)


def savefig_square(fig, out_path, dpi: int = 160) -> None:
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
