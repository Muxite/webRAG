#!/usr/bin/env python3
"""Classify stored cells into ONE failure mechanism each, so a floor can be read.

A bare score says a cell failed; it does not say how, and on this suite "0.000" covers at least
five unrelated causes. Grouping cells by mechanism is what separates a model that cannot emit a
tool call from one that retrieves correctly and then fabricates the arithmetic — the second is
the Ledger's target, the first is a capability boundary and belongs in no average.

Reads ONLY ``execution.telemetry_raw`` and ``validation.grep_validations``. It never reads
``observability.*``: ``search.count`` counted result DOCUMENTS rather than calls before commit
0fa6e733 (6x at search_k=6), ``llm.calls`` runs ~1.4-2x the real turn count, and ``visit.count``
counts stored documents, so 25 failed visits read as 0.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/cell_mechanism.py ladder03_ probe3_
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, Sequence, Tuple

RESULTS = "agent/idea_test_results"

#: Cell filenames are ``{run_id}_{task}_{model}_{variant}[_t<n>][_cfg<hex>]_r{rep}.json``. The
#: anchor on ``_r<N>.json`` matters: verbosity>=3 also writes ``..._r1_report_v3.json``, a
#: different schema entirely, and ``*_summary.json`` re-embeds cells (a prior phase inflated a
#: headline 1.7x by counting them).
CELL_RE = re.compile(r"^(?P<run>.+?)_(?P<task>\d{2,3})_(?P<model>.+?)_"
                     r"(?P<variant>[a-z_]+?)(?:_t\d+)?(?:_cfg[0-9a-f]+)?_r(?P<rep>\d+)\.json$")


def load_cells(prefixes: Sequence[str], results_dir: str = RESULTS
               ) -> Iterator[Tuple[Dict[str, str], Dict[str, Any]]]:
    """Yield ``(meta, cell)`` for each result file matching any prefix, deduped by identity."""
    seen = set()
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json") or name.endswith("_summary.json"):
            continue
        if not any(name.startswith(p) for p in prefixes):
            continue
        m = CELL_RE.match(name)
        if not m:
            continue
        key = (m["run"], m["task"], m["model"], m["variant"], m["rep"])
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(os.path.join(results_dir, name)) as fh:
                cell = json.load(fh)
        except (OSError, ValueError):
            continue
        yield dict(zip(("run", "task", "model", "variant", "rep"), key), path=name), cell


def facts(cell: Dict[str, Any]) -> Dict[str, Any]:
    """The raw quantities the classifier is allowed to use.

    Note the nesting: the payload lives under ``execution``, and a visit's HTTP status is at
    ``payload.status`` inside a timing entry. Reading either at the top level returns nothing
    silently and prints as a plausible finding.
    """
    ex = cell.get("execution") or {}
    tr = ex.get("telemetry_raw") or {}
    out = ex.get("output") or {}
    tim = tr.get("timings") or []
    emu = [t for t in tim if t.get("name") == "tool_call_emulation"]
    visits = [t for t in tim if t.get("name") == "visit"]
    return {
        "turns": len(tr.get("llm_usage") or []),
        "exec_search": sum(1 for t in tim if t.get("name") == "search"),
        "exec_visit": len(visits),
        "visit_200": sum(1 for t in visits
                         if (t.get("payload") or {}).get("status") == 200),
        "emu": len(emu),
        "emu_named": sum(1 for t in emu if (t.get("payload") or {}).get("action")),
        "claimed_ok": sum(1 for t in emu
                          if t.get("success") and (t.get("payload") or {}).get("action")),
        "transport": out.get("tool_transport"),
        "checks": {g.get("check"): g for g in
                   ((cell.get("validation") or {}).get("grep_validations") or [])},
        "score": (cell.get("validation") or {}).get("overall_score"),
        "infra_failed": cell.get("infra_failed"),
    }


def classify(f: Dict[str, Any]) -> str:
    """Exactly one mechanism per cell. Order matters — earlier rules are more specific."""
    if f["infra_failed"]:
        return "INFRA_FAILED"
    if f["emu"] and not f["emu_named"] and not f["exec_search"] and not f["exec_visit"]:
        return "NO_ACTION"            # emitted turns, none carried a parseable action
    if f["claimed_ok"] and not f["exec_search"] and not f["exec_visit"]:
        return "TOOL_NEVER_EXECUTED"  # parsed and flagged successful, but nothing ran
    if not f["exec_search"] and not f["exec_visit"]:
        return "NO_TOOL_CALL"
    if f["exec_visit"] and not f["exec_search"]:
        return "URL_INVENTED_OK" if f["visit_200"] else "URL_INVENTED_404"
    if f["exec_search"] and not f["exec_visit"]:
        return "NO_VISIT"
    if f["exec_visit"] and not f["visit_200"]:
        return "VISIT_ALL_FAILED"
    keystone = next((c for k, c in f["checks"].items() if k.startswith("keystone")), None)
    if keystone and keystone.get("passed"):
        return "SOLVED"
    coverage = f["checks"].get("coverage")
    if coverage and coverage.get("score"):
        return "READ_THEN_FABRICATE"  # operands gathered, derived value wrong
    return "READ_THEN_IGNORE"         # page read, operands never reached the answer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefixes", nargs="+", help="run-id prefixes, e.g. ladder03_")
    ap.add_argument("--results-dir", default=RESULTS)
    a = ap.parse_args()

    table: Dict[str, Counter] = defaultdict(Counter)
    for meta, cell in load_cells(a.prefixes, a.results_dir):
        table[meta["model"]][classify(facts(cell))] += 1
    if not table:
        print("no cells matched")
        return

    mechs = sorted({m for row in table.values() for m in row})
    w = max(len(m) for m in mechs) + 2
    print(f"{'model':16}" + "".join(f"{m:>{w}}" for m in mechs) + f"{'total':>8}")
    for model in sorted(table, key=lambda k: -sum(table[k].values())):
        row = table[model]
        print(f"{model:16}" + "".join(f"{row.get(m, 0) or '.':>{w}}" for m in mechs)
              + f"{sum(row.values()):>8d}")


if __name__ == "__main__":
    main()
