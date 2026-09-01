"""Search-behaviour metrics for a stored result cell. Reports only -- never gates, never scores.

Why this exists. On the `ledgernum22` campaign, high-scoring `evidence_loop` cells averaged 2.60
search queries and low-scoring ones 5.75: bad searching is real, and it is expensive. Nothing in
the repo measured it. Queries were being persisted all along, in
``execution.telemetry_raw.timings[].payload.query``, and never read back.

The obvious metric is a trap, so this module deliberately does not use it. Across 66 cells there
were ZERO exact query repeats (the loop's ``ALREADY SEARCHED`` dedup catches those) but 106
near-duplicate pairs by string similarity -- and three of the first five flagged were
``"Eiffel Tower original construction cost historical record"`` ->
``"Statue of Liberty ... historical record"`` -> ``"Great Pyramid ... historical record"``: one
template applied to different entities, which is exactly correct behaviour on a comparison task.
A similarity penalty would punish good breadth.

So the signal here is YIELD-CONDITIONED, not textual:

* a requery issued when the previous search produced no successful visit is **adaptation** --
  the agent looked, got nothing, and reformulated;
* a requery issued when the previous search already produced a usable visit is only **thrash**
  if it never goes on to open a NEW page. A requery that opens a new page is **productive**,
  which is what the entity-varying template case above actually is.

The real waste this exposes, from task 227's trace: search -> visit a page -> search -> search ->
re-visit THE SAME page. The loop dedups searches but has no visit dedup at all, so the re-visit is
unguarded and the reworded searches slip past the exact-match dedup.

Nothing here adjusts a score. The repo has been bitten by an inverted gate before -- the roster
gate blocked 46 of 48 eligible cells including two scoring 1.00 -- so this signal must be shown to
track ground truth before anything is allowed to act on it.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/query_quality.py --run-ids ledgernum22
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "agent", "idea_test_results")


@dataclass(frozen=True)
class Event:
    """One search or visit, in the order the run performed it."""

    kind: str
    text: str
    success: bool


def search_events(cell: Dict[str, Any]) -> List[Event]:
    """Every search and visit in a stored cell, in execution order.

    :param cell: a parsed result-cell dict.
    :returns: ordered :class:`Event` list; empty when the cell carries no timings.
    :raises: nothing -- a malformed cell yields no events rather than an error.
    """
    timings = (((cell or {}).get("execution") or {}).get("telemetry_raw") or {}).get("timings")
    events: List[Event] = []
    for entry in timings or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if name == "search":
            events.append(Event("search", str(payload.get("query") or ""), bool(entry.get("success"))))
        elif name == "visit":
            events.append(Event("visit", str(payload.get("url") or ""), bool(entry.get("success"))))
    return events


def query_quality(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Search-behaviour metrics for one cell.

    The run is split into SEGMENTS: each search, plus everything it produced before the next
    search. A requery is then judged on two independent axes --

    * its CONTEXT, from the previous segment: did the previous search yield a successful visit?
      Requerying after a dry search is adaptation; requerying after a yield is at least suspicious.
    * its OUTCOME, from its own segment: did it open a page not already read?

    ``thrash`` is the intersection -- issued after a yield AND opening nothing new. That
    intersection is what keeps entity-varying template reuse out of the penalty box: those
    requeries also follow a yield, but each opens a new page, so they count as ``productive``.

    :param cell: a parsed result-cell dict.
    :returns: counts plus the cell's arm/task/score for grouping. No pass/fail, by design.
    :raises: nothing.
    """
    events = search_events(cell)
    seen_urls: set = set()
    visits = redundant = 0
    query_texts: List[str] = []
    # Per search, in order: (yielded_a_visit, opened_a_new_page).
    segments: List[List[bool]] = []

    for event in events:
        if event.kind == "search":
            query_texts.append(event.text)
            segments.append([False, False])
            continue
        visits += 1
        if not event.success:
            continue
        if event.text in seen_urls:
            redundant += 1
        else:
            seen_urls.add(event.text)
            if segments:
                segments[-1][1] = True
        if segments:
            segments[-1][0] = True

    after_dry = after_yield = productive = thrash = 0
    for index in range(1, len(segments)):
        context_yielded = segments[index - 1][0]
        opened_new = segments[index][1]
        if context_yielded:
            after_yield += 1
            if opened_new:
                productive += 1
            else:
                thrash += 1
        else:
            after_dry += 1
            if opened_new:
                productive += 1

    return {
        "arm": cell.get("execution_variant"),
        "task": str(((cell or {}).get("test_metadata") or {}).get("id") or ""),
        "score": ((cell or {}).get("validation") or {}).get("overall_score"),
        "queries": len(segments),
        "distinct_queries": len({q.strip().lower() for q in query_texts if q.strip()}),
        "requery_after_dry": after_dry,
        "requery_after_yield": after_yield,
        "productive_requeries": productive,
        "thrash_requeries": thrash,
        "visits": visits,
        "distinct_urls": len(seen_urls),
        "redundant_visits": redundant,
    }

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-cell metrics by arm.

    :param rows: :func:`query_quality` outputs.
    :returns: ``{arm: {...}}``; empty for no rows.
    :raises: nothing.
    """
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row.get("arm") or ""), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for arm, group in by_arm.items():
        def total(key: str) -> int:
            return sum(int(r.get(key) or 0) for r in group)
        out[arm] = {
            "cells": len(group),
            "mean_queries": statistics.mean([r["queries"] for r in group]),
            "mean_visits": statistics.mean([r["visits"] for r in group]),
            "requery_after_dry": total("requery_after_dry"),
            "requery_after_yield": total("requery_after_yield"),
            "productive_requeries": total("productive_requeries"),
            "thrash_requeries": total("thrash_requeries"),
            "redundant_visits": total("redundant_visits"),
        }
    return out


def load_cells(run_ids: List[str], results_dir: str = RESULTS_DIR) -> List[Dict[str, Any]]:
    """Parsed cells whose filename starts with any of ``run_ids``."""
    cells = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_r*.json"))):
        name = os.path.basename(path)
        if name.endswith("_summary.json") or "_report_" in name:
            continue
        if run_ids and not any(name.startswith(f"{rid}_") for rid in run_ids):
            continue
        try:
            cells.append(json.load(open(path)))
        except Exception:  # noqa: BLE001 -- an unreadable cell is skipped, never fatal
            continue
    return cells


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-ids", default="", help="comma-separated run-id prefixes")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--by-score", action="store_true",
                        help="split each arm into high (>=0.75) and low (<=0.25) scoring cells")
    args = parser.parse_args(argv)
    ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    rows = [query_quality(c) for c in load_cells(ids, args.results_dir)]
    if not rows:
        print("no cells matched")
        return 1
    print(f"{'arm':26} {'cells':>5} {'q/cell':>7} {'v/cell':>7} {'dry':>5} {'yield':>6} "
          f"{'prod':>5} {'thrash':>7} {'redup':>6}")
    for arm, s in sorted(summarize(rows).items()):
        print(f"{arm:26} {s['cells']:5} {s['mean_queries']:7.2f} {s['mean_visits']:7.2f} "
              f"{s['requery_after_dry']:5} {s['requery_after_yield']:6} "
              f"{s['productive_requeries']:5} {s['thrash_requeries']:7} {s['redundant_visits']:6}")
    if args.by_score:
        print("\n-- split by cell score (does the signal track ground truth?) --")
        for arm in sorted({r["arm"] for r in rows if r["arm"]}):
            for label, keep in (("high >=0.75", lambda s: (s or 0) >= 0.75),
                                ("low  <=0.25", lambda s: (s or 0) <= 0.25)):
                group = [r for r in rows if r["arm"] == arm and keep(r["score"])]
                if not group:
                    continue
                print(f"  {arm:26} {label}  n={len(group):3} "
                      f"q/cell={statistics.mean([r['queries'] for r in group]):5.2f} "
                      f"thrash={sum(r['thrash_requeries'] for r in group):3} "
                      f"redup={sum(r['redundant_visits'] for r in group):3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
