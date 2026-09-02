#!/usr/bin/env python3
"""Host vs host + module: what attaching a Ledger module makes TRUE, with score as the guard.

Why this is not `compare_arms.py`. That script compares two ARMS and leads with a paired score
delta, which is the right shape for "is A better than B". It is the wrong shape here, for a reason
this project measured rather than assumed: with decoding deterministic (`LLM_SEED` fixed) and one
mechanism flipped, 12 of 16 tasks came back byte-identical while 2 swung by ~0.6
(`docs/LEDGER_FINAL01_TUNING_RESULT.md`). A mean score over that distribution is an average of
mostly-zeros and a couple of large cancelling swings, and per-arm KPI orderings on this suite
INVERT between task subsets. So a score ranking between two configurations is not resolvable at
this n and this script refuses to lead with one.

What it reports instead are STRUCTURAL outcomes -- categorical facts about what the host can now
do, which do not depend on resolving a small mean difference:

  derivation_available   cells where the host produced >=1 machine-verified derived value
  derived_nodes          how many such values, and how many were marked invalid
  fabrication_rate       invalid / derived -- UNKNOWN, never 0.0, where no graph exists at all
  refusals               operand-not-on-page and unit-mismatch refusals, i.e. fabrications averted
  replayable             cells whose stored artifact re-verifies offline with no page drift

and then `validation.overall_score` as the GUARD, where a null result is the win: auditability at
no accuracy cost is the claim. A score DROP is a finding; a score gain is not claimable.

Usage:
  python3 scripts/module_ab.py --off ledgerhost_off --on ledgerhost_on
  python3 scripts/module_ab.py --off A --on B --tasks 210,211 --json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.app.testing.evidence_graph import reverify_graph  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "agent" / "idea_test_results"


def load_run(run_id: str, results_dir: Path = RESULTS) -> Dict[int, Dict[str, Any]]:
    """Every scorable cell of ``run_id``, keyed by task id. Infra failures are dropped."""
    cells: Dict[int, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob(f"{run_id}_*.json")):
        try:
            cell = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(cell, dict) or "execution" not in cell or cell.get("infra_failed"):
            continue
        meta = cell.get("test_metadata") or {}
        try:
            cells[int(meta.get("test_id"))] = cell
        except (TypeError, ValueError):
            continue
    return cells


def structural(cell: Dict[str, Any]) -> Dict[str, Any]:
    """The categorical facts about one cell: what did this run actually make checkable?

    ``fabrication_rate`` is None -- never 0.0 -- when the cell carries no evidence graph at all.
    An absent graph is not evidence of a 0% fabrication rate; it is evidence that nobody can
    compute one, which is the whole difference the module makes.
    """
    graph = (cell.get("execution", {}).get("output") or {}).get("evidence_graph")
    if not isinstance(graph, dict) or not graph.get("nodes"):
        return {"has_graph": False, "derived": 0, "invalid": 0, "fabrication_rate": None,
                "refusals": 0, "replayable": None}
    nodes = graph.get("nodes") or []
    derived = [n for n in nodes if n.get("kind") == "derived"]
    invalid = [n for n in derived if n.get("derivation_valid") is False]
    try:
        report = reverify_graph(graph)
        counts = report.get("counts") or {}
        replayable = counts.get("failed", 0) == 0 and counts.get("page_drift", 0) == 0
    except Exception:  # noqa: BLE001 -- an unreplayable artifact is a result, not a crash
        replayable = False
    return {
        "has_graph": True,
        "derived": len(derived),
        "invalid": len(invalid),
        "fabrication_rate": (len(invalid) / len(derived)) if derived else None,
        # DISTINCT operands refused, not rejection RECORDS. The toolkit tries every registered
        # page before giving up, so one unlocatable operand emits one rejection per page and a
        # raw record count overstates by the page count -- it read 176 for a run with about five
        # genuinely refused operands. Counting distinct values keeps this column meaning "how many
        # things did the module decline to treat as evidence".
        "refusals": len({r.get("value") for r in (graph.get("rejections") or [])
                         if isinstance(r, dict)})
        + sum((graph.get("refusal_counts") or {}).values()),
        "replayable": replayable,
    }


def summarize(cells: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Structural totals plus the accuracy anchor for one configuration."""
    facts = {t: structural(c) for t, c in cells.items()}
    scores = [c["validation"]["overall_score"] for c in cells.values()
              if isinstance(c.get("validation"), dict)]
    derived = sum(f["derived"] for f in facts.values())
    invalid = sum(f["invalid"] for f in facts.values())
    replayed = [f["replayable"] for f in facts.values() if f["replayable"] is not None]
    return {
        "cells": len(cells),
        "derivation_available": sum(1 for f in facts.values() if f["derived"] > 0),
        "derived_nodes": derived,
        "invalid_nodes": invalid,
        "fabrication_rate": (invalid / derived) if derived else None,
        "refusals": sum(f["refusals"] for f in facts.values()),
        "replayable_cells": sum(1 for r in replayed if r),
        "replay_checked": len(replayed),
        "score_mean": statistics.mean(scores) if scores else None,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def report(off_id: str, on_id: str, off: Dict[str, Any], on: Dict[str, Any],
           paired: List[float]) -> str:
    """The human-readable table. Structural facts first, score last and labelled as a guard."""
    lines = [
        f"HOST vs HOST + MODULE   off={off_id}   on={on_id}",
        "",
        "STRUCTURAL OUTCOMES (categorical -- what the host can now do)",
        f"{'':30s} {'off':>12s} {'on':>12s}",
        f"{'  cells':30s} {off['cells']:>12d} {on['cells']:>12d}",
        f"{'  cells with a derivation':30s} {off['derivation_available']:>12d} "
        f"{on['derivation_available']:>12d}",
        f"{'  machine-computed values':30s} {off['derived_nodes']:>12d} {on['derived_nodes']:>12d}",
        f"{'  of those, invalid':30s} {off['invalid_nodes']:>12d} {on['invalid_nodes']:>12d}",
        f"{'  fabricated-arith rate':30s} {_fmt(off['fabrication_rate']):>12s} "
        f"{_fmt(on['fabrication_rate']):>12s}",
        f"{'  refusals (averted)':30s} {off['refusals']:>12d} {on['refusals']:>12d}",
        f"{'  replayable cells':30s} "
        f"{str(off['replayable_cells'])+'/'+str(off['replay_checked']):>12s} "
        f"{str(on['replayable_cells'])+'/'+str(on['replay_checked']):>12s}",
        "",
        "ACCURACY GUARD (a null result is the win; a drop is a finding)",
        f"{'  overall_score':30s} {_fmt(off['score_mean']):>12s} {_fmt(on['score_mean']):>12s}",
    ]
    if paired:
        p, n = bench_stats.signflip_p(paired)
        _, mean, ci, _dz = bench_stats.paired_stats(paired)
        moved = [d for d in paired if abs(d) > 1e-9]
        lines += [
            f"{'  paired delta':30s} {mean:+12.4f}  +/-{ci:.4f}  p={p:.3f}  n={n}",
            f"  tasks that moved at all: {len(moved)}/{len(paired)}"
            + (f"   magnitudes {sorted(round(abs(d), 3) for d in moved)}" if moved else ""),
            "",
            "  NOTE: a mean over mostly-unchanged tasks with a few large swings is not a ranking.",
            "  This suite's measured trajectory-chaos floor makes a score ORDERING between two",
            "  configurations unresolvable at this n; read the structural rows above instead.",
        ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off", required=True, help="run_id with the module OFF (the host alone)")
    parser.add_argument("--on", required=True, help="run_id with the module ON")
    parser.add_argument("--tasks", help="comma-separated task ids to restrict to")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    off_cells = load_run(args.off, results_dir)
    on_cells = load_run(args.on, results_dir)
    if args.tasks:
        keep = {int(t) for t in args.tasks.split(",") if t.strip()}
        off_cells = {t: c for t, c in off_cells.items() if t in keep}
        on_cells = {t: c for t, c in on_cells.items() if t in keep}

    shared = sorted(set(off_cells) & set(on_cells))
    paired = [on_cells[t]["validation"]["overall_score"]
              - off_cells[t]["validation"]["overall_score"] for t in shared]
    off, on = summarize(off_cells), summarize(on_cells)

    if args.json:
        print(json.dumps({"off": off, "on": on, "paired_tasks": len(shared)}, indent=2))
    else:
        print(report(args.off, args.on, off, on, paired))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
