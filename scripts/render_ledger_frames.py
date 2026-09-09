#!/usr/bin/env python3
"""render_ledger_frames.py: export every frame of a ledger deck as a numbered PNG.

For slides, a README, a thumbnail sheet, or the frames of a screen recording::

    PYTHONPATH=.:services:agent python3 scripts/render_ledger_frames.py <cell>.json -o frames/

The frames are screenshots of the deck itself, driven headlessly. That is a deliberate choice:
the alternative -- reimplementing the frame grammar a second time in matplotlib -- would be a
second layout engine to keep in step with the first, and every future change to the deck would
silently stop being true of the exported frames. Here there is one layout, so an exported frame
is by construction the frame a viewer sees.

Matplotlib keeps the work it is actually better at: the run-level and campaign-level charts in
``scripts/render_ledger_charts.py``.

Also writes ``contact_sheet.png`` (every frame tiled, for picking the ones worth using) and the
storyboard JSON beside the frames.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_ROOT, os.path.join(_ROOT, "services"), os.path.join(_ROOT, "agent")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.app.testing import ledger_story  # noqa: E402
import render_ledger_deck  # noqa: E402


def _serve(directory: str):
    """A throwaway localhost server.

    Chromium refuses ``file://`` under the sandbox this repo's tooling runs in, and a deck is a
    single self-contained file, so the cheapest correct answer is to serve the one directory it
    lives in for the length of the export.
    """
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def export(cell_path: str, out_dir: str, width: int, height: int, scale: int,
           only: List[int] | None = None) -> List[str]:
    from playwright.sync_api import sync_playwright

    story = ledger_story.build_file(cell_path)
    os.makedirs(out_dir, exist_ok=True)
    deck_path = os.path.join(out_dir, "deck.html")
    with open(deck_path, "w", encoding="utf-8") as handle:
        handle.write(render_ledger_deck.build_html(story, f"Ledger Run {story.task_id or ''}".strip()))
    with open(os.path.join(out_dir, "storyboard.json"), "w", encoding="utf-8") as handle:
        json.dump(story.as_dict(), handle, indent=1, ensure_ascii=False)

    server, port = _serve(out_dir)
    written: List[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height},
                                    device_scale_factor=scale)
            page.goto(f"http://127.0.0.1:{port}/deck.html")
            page.wait_for_function("() => typeof window.gotoBeat === 'function'")
            wanted = only if only else list(range(len(story)))
            for index in wanted:
                beat = story.beats[index]
                page.evaluate("i => window.gotoBeat(i)", index)
                # The rail re-renders synchronously, but a webfont/layout pass can trail it.
                page.wait_for_timeout(60)
                name = f"{index:03d}_{beat.kind}.png"
                path = os.path.join(out_dir, name)
                page.screenshot(path=path)
                written.append(path)
            browser.close()
    finally:
        server.shutdown()
    return written


def contact_sheet(frames: List[str], out_path: str, columns: int = 6) -> None:
    """Every frame tiled on one sheet -- how you pick the four worth putting in a post."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from agent.app.testing import plot_style

    rows = max(1, math.ceil(len(frames) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.6, rows * 1.75), dpi=170)
    fig.patch.set_facecolor(plot_style.DARK_GROUND)
    for ax, frame in zip(getattr(axes, "flat", [axes]), frames):
        ax.imshow(mpimg.imread(frame))
        ax.set_title(os.path.basename(frame).split("_", 1)[-1].replace(".png", ""),
                     color=plot_style.DARK_INK_MUTED, fontsize=5, pad=2)
    for ax in getattr(axes, "flat", [axes]):
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(plot_style.DARK_HAIRLINE)
    for ax in list(getattr(axes, "flat", [axes]))[len(frames):]:
        ax.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, facecolor=plot_style.DARK_GROUND)
    plt.close(fig)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cell")
    parser.add_argument("-o", "--out-dir", default="ledger_frames")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--scale", type=int, default=2, help="device pixel ratio (2 = retina)")
    parser.add_argument("--beats", default="", help="comma-separated beat indexes; default all")
    parser.add_argument("--no-sheet", action="store_true")
    args = parser.parse_args(argv)

    only = [int(x) for x in args.beats.split(",") if x.strip()] if args.beats else None
    frames = export(args.cell, args.out_dir, args.width, args.height, args.scale, only)
    print(f"{len(frames)} frames -> {args.out_dir}")
    if not args.no_sheet and frames:
        sheet = os.path.join(args.out_dir, "contact_sheet.png")
        contact_sheet(frames, sheet)
        print(f"contact sheet -> {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
