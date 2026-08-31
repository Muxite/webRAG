#!/usr/bin/env python3
"""
reverify.py: CLI entry point over the two offline audit functions that were previously
test-only, with no way to run them against a real saved result outside pytest:

  * ``execution_evidence_loop.reverify_cell`` -- re-checks every stored extraction's quote
    against the PERSISTED page text.
  * ``evidence_graph.reverify_graph`` -- re-derives each evidence-graph node's verification
    from the stored artifact alone, and flags a page whose text no longer hashes to what was
    recorded at capture time (page drift).

Both are fully offline: no model, no network, no GPU -- they only read back what a run already
persisted into its result JSON. ``reverify_graph`` has a real target as of the evidence loop
wiring the derivation layer in: every cell's ``output`` now carries an ``evidence_graph`` key
(``ledger.graph.to_dict()``), so a cell produced by that (or a newer) run has an artifact to
audit; older cells will not, and this CLI says so rather than failing.

CLI usage:
    python scripts/reverify.py <cell.json> [--json] [--strict]

``<cell.json>`` is a saved result-cell -- the full cell dict, its ``execution`` node, or its
``output`` node all resolve to the same payload (mirrors ``reverify_cell``'s own docstring).
``--strict`` exits 1 when either report found a genuine failure (never for "unverifiable" --
a missing page or truncated window is not a failure, it is an audit gap).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.app.testing.execution_evidence_loop import reverify_cell
from agent.app.testing.evidence_graph import reverify_graph

GRAPH_STATUS_OK = "ok"
GRAPH_STATUS_MISSING = "no_evidence_graph_artifact"


def resolve_output(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a saved cell dict down to its ``output`` node.

    Mirrors ``execution_evidence_loop._cell_output``'s resolution rule (kept as its own copy
    here rather than importing that private helper): the full cell dict, its ``execution`` node,
    and its ``output`` node all resolve to the same payload.

    :param cell: A result-cell dict in any of the three shapes above.
    :returns: The innermost ``output`` dict, or ``{}`` if none of the shapes matched.
    """
    node = cell if isinstance(cell, dict) else {}
    for key in ("execution", "output"):
        if isinstance(node.get(key), dict):
            node = node[key]
    return node


def load_cell(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a saved result-cell JSON file.

    :param path: Path to the ``.json`` file.
    :returns: The parsed dict.
    :raises ValueError: if the file does not contain a JSON object.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level, got {type(data).__name__}")
    return data


def run_reverify(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Run both audit functions against one saved cell.

    :param cell: A result-cell dict (any of the three shapes ``resolve_output`` understands).
    :returns: ``{"cell": <reverify_cell report>, "graph": <reverify_graph report or None>,
        "graph_status": GRAPH_STATUS_OK | GRAPH_STATUS_MISSING}``.
    """
    cell_report = reverify_cell(cell)
    output = resolve_output(cell)
    artifact = output.get("evidence_graph")
    if isinstance(artifact, dict) and artifact:
        graph_report: Optional[Dict[str, Any]] = reverify_graph(artifact)
        graph_status = GRAPH_STATUS_OK
    else:
        graph_report = None
        graph_status = GRAPH_STATUS_MISSING
    return {"cell": cell_report, "graph": graph_report, "graph_status": graph_status}


def has_genuine_failure(report: Dict[str, Any]) -> bool:
    """True when either sub-report found a real (not merely unverifiable) mismatch.

    :param report: Result of :func:`run_reverify`.
    :returns: True if ``cell["counts"]["failed"] > 0`` or (when present)
        ``graph["counts"]["failed"] > 0``.
    """
    if report["cell"]["counts"].get("failed", 0) > 0:
        return True
    graph = report.get("graph")
    if graph and graph["counts"].get("failed", 0) > 0:
        return True
    return False


def format_report(report: Dict[str, Any]) -> str:
    """Render a human-readable audit summary.

    :param report: Result of :func:`run_reverify`.
    :returns: Report text.
    """
    lines = ["=== reverify report ===", ""]
    cell = report["cell"]
    lines.append(f"-- reverify_cell (quote verification) -- pages={cell['pages']}")
    lines.append(f"   counts: {cell['counts']}")
    for row in cell["extractions"]:
        if row["quote_verified"] is not True:
            lines.append(
                f"   FAIL/UNCHECKED page={row['page_id']} reason={row['quote_fail_reason']} "
                f"quote={row['quote']!r}"
            )
    lines.append("")
    if report["graph_status"] == GRAPH_STATUS_MISSING:
        lines.append(
            "-- reverify_graph -- SKIPPED: no evidence_graph artifact on this cell "
            "(pre-derivation-layer cell, or the run had nothing to derive)."
        )
    else:
        graph = report["graph"]
        lines.append(f"-- reverify_graph (derivation provenance) -- pages={graph['pages']}")
        lines.append(f"   counts: {graph['counts']}")
        for row in graph["nodes"]:
            if row["verified"] is not True or row["drifted"]:
                lines.append(
                    f"   FAIL/UNCHECKED node={row['node_id']} kind={row['kind']} "
                    f"reason={row['fail_reason']} drifted={row['drifted']} value={row['value']!r}"
                )
    lines.append("")
    lines.append(f"genuine failures found: {has_genuine_failure(report)}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-check a saved result cell's quotes and evidence-graph provenance, offline."
    )
    parser.add_argument("cell_path", help="Path to a saved result-cell .json file")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the text report.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 if either audit found a genuine failure (not merely unverifiable).",
    )
    args = parser.parse_args(argv)

    try:
        cell = load_cell(args.cell_path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"reverify: {e}", file=sys.stderr)
        return 1

    report = run_reverify(cell)

    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(format_report(report))

    if args.strict and has_genuine_failure(report):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
