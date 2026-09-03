#!/usr/bin/env python3
"""Byte-identity between two run prefixes: did the same configuration produce the same run?

Two uses, both load-bearing:

* **Determinism.** With ``LLM_SEED`` fixed, a rerun of an identical configuration must come back
  byte-identical. That is what makes ``reps=1`` defensible. It was NOT true of ``langgraph_react``
  before commit ``adeeffa5`` -- that arm built its own ``ChatOpenAI`` and never received the seed,
  so every "seeded" A/B on it was actually sampling. Measured then: 0/3 identical. After: 15/15.
* **Drift during a long sweep.** A campaign whose cells duplicate an earlier run's configuration
  self-checks for free. A mismatch part-way through means the ENVIRONMENT moved -- ollama
  restarted or re-pulled a model, or a visited page changed, since visits are live HTTP even under
  corpus replay and only *search* is frozen. It is not, by itself, evidence of a code bug.

Compares ``final_deliverable``, ``overall_score`` and turn count on cells paired by
``(model, task)``. Exits non-zero if any pair differs.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/run_diff.py probe3_ probe3b_
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/run_diff.py probe3_ ladder03_ --tasks 210,211,212
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_mechanism import RESULTS, load_cells  # noqa: E402


def gather(prefix: str, results_dir: str, tasks: Optional[Sequence[str]] = None,
           run_suffix: Optional[str] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Cells under ``prefix``, keyed by ``(model, task)``.

    :param run_suffix: keep only runs whose id ends with this (e.g. ``_off_lg`` to select one
        condition out of a sweep that writes many run_ids under a shared prefix).
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for meta, cell in load_cells([prefix], results_dir):
        if tasks and meta["task"] not in tasks:
            continue
        if run_suffix and not meta["run"].endswith(run_suffix):
            continue
        ex = cell.get("execution") or {}
        out[(meta["model"], meta["task"])] = {
            "answer": (ex.get("output") or {}).get("final_deliverable") or "",
            "score": (cell.get("validation") or {}).get("overall_score"),
            "turns": len(((ex.get("telemetry_raw") or {}).get("llm_usage")) or []),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="run-id prefix holding the known-good cells")
    ap.add_argument("candidate", help="run-id prefix to check against it")
    ap.add_argument("--tasks", help="comma-separated task ids to restrict to")
    ap.add_argument("--reference-suffix", help="keep only reference runs ending with this")
    ap.add_argument("--candidate-suffix", help="keep only candidate runs ending with this")
    ap.add_argument("--results-dir", default=RESULTS)
    a = ap.parse_args()

    tasks = [t.strip() for t in a.tasks.split(",")] if a.tasks else None
    ref = gather(a.reference, a.results_dir, tasks, a.reference_suffix)
    new = gather(a.candidate, a.results_dir, tasks, a.candidate_suffix)
    keys = sorted(set(ref) & set(new))
    if not keys:
        print(f"no cells pair between {a.reference!r} and {a.candidate!r} yet "
              f"({len(ref)} vs {len(new)} loaded)")
        return 0

    bad = [k for k in keys
           if ref[k]["answer"] != new[k]["answer"] or ref[k]["score"] != new[k]["score"]]
    print(f"{len(keys) - len(bad)}/{len(keys)} paired cells identical "
          f"({a.reference} -> {a.candidate})\n")
    for k in keys:
        r, n = ref[k], new[k]
        print(f"  {'ok  ' if k not in bad else 'DIFF'} {k[0]} {k[1]}  "
              f"score {r['score']} -> {n['score']}  turns {r['turns']} -> {n['turns']}")
    if not bad:
        return 0
    print(f"\n{len(bad)} differ. Check in order: (1) did ollama restart or re-pull a model; "
          "(2) did a visited page change -- visits are live HTTP even under corpus replay; "
          "(3) only then suspect code.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
