#!/usr/bin/env python3
"""ledger_trace_query.py: query and draw the ledger trace of a stored run.

The trace (:mod:`agent.app.ledger_trace`) is a projection of a per-cell result JSON into typed,
parented, time-stamped nodes that cross-reference the evidence graph. This is the tool that
proves it is actually queryable and actually drawable -- if a question here needs a ``grep``
over free text, the schema is wrong.

Examples::

    # what ran, in one table
    python scripts/ledger_trace_query.py agent/idea_test_results/<cell>.json

    # every derivation the Ledger refused, with its operands resolved to real values
    python scripts/ledger_trace_query.py <cell>.json --kind derive --status refused --resolve

    # every visit that did not return 2xx
    python scripts/ledger_trace_query.py <cell>.json --kind visit --status error

    # every step that touched one page, and everything done with one evidence node
    python scripts/ledger_trace_query.py <cell>.json --page-id p1
    python scripts/ledger_trace_query.py <cell>.json --evidence 9583ec011e08e5a3d8ce2b71198d1dc7

    # a mermaid graph, ready to paste into a ```mermaid block (artifacts render these natively)
    python scripts/ledger_trace_query.py <cell>.json --format mermaid > trace.mmd

    # what the trace costs against the encodings it subsumes
    python scripts/ledger_trace_query.py <cell>.json --size

Filters compose (they AND together) and apply to every output format, so ``--format mermaid
--kind derive`` draws just the derivation plane.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ``ledger_trace`` reuses ``evidence_store.canonicalize_url`` rather than reimplementing URL
# canonicalization, and that module's package pulls in the connectors, which import from
# ``shared`` -- so this script needs the same repo path set as the test suite
# (``PYTHONPATH=.:services:agent``) rather than just the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_ROOT, os.path.join(_ROOT, "services"), os.path.join(_ROOT, "agent")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.app.ledger_trace import (  # noqa: E402
    LedgerTrace,
    TraceNode,
    evidence_for,
    project_cell,
    to_mermaid,
)


def load_cell(path: str) -> Dict[str, Any]:
    """The parsed per-cell result JSON at ``path``."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def evidence_graph_of(cell: Dict[str, Any]) -> Dict[str, Any]:
    """The cell's evidence graph artifact, or an empty one."""
    execution = cell.get("execution") if isinstance(cell.get("execution"), dict) else {}
    output = execution.get("output") if isinstance(execution.get("output"), dict) else {}
    graph = output.get("evidence_graph")
    return graph if isinstance(graph, dict) else {}


def _interval(node: TraceNode) -> str:
    """``[t_start-t_end]`` when there is one, ``-`` when there is not. Never a fake zero."""
    if node.t_start is None or node.t_end is None:
        return "-"
    return f"{node.t_start:8.3f}-{node.t_end:8.3f}"


def _detail_text(node: TraceNode, width: int = 70) -> str:
    return " ".join(f"{k}={v}" for k, v in node.detail.items())[:width]


def format_table(trace: LedgerTrace, nodes: List[TraceNode]) -> str:
    """One line per node: id, parent, kind, status, interval, detail, cross-references."""
    lines = [
        f"{'id':<7} {'parent':<7} {'kind':<9} {'status':<8} {'interval':<19} detail",
        "-" * 110,
    ]
    for node in nodes:
        refs = ""
        if node.page_id:
            refs += f" page={node.page_id}"
        if node.evidence_in:
            refs += " in=" + ",".join(e[:8] for e in node.evidence_in)
        if node.evidence_out:
            refs += " out=" + ",".join(e[:8] for e in node.evidence_out)
        lines.append(
            f"{node.id:<7} {node.parent or '-':<7} {node.kind:<9} {node.status:<8} "
            f"{_interval(node):<19} {_detail_text(node)}{refs}"
        )
    lines.append("")
    lines.append(f"{len(nodes)} of {len(trace.nodes)} nodes  {json.dumps(trace.counts())}")
    return "\n".join(lines)


def format_resolved(nodes: List[TraceNode], graph: Dict[str, Any]) -> str:
    """Each node followed into the evidence graph: what it consumed, produced, and read.

    This is the "follow a node to its evidence" half. The trace stores only ids; the values and
    page spans come from the graph at read time, which is why neither artifact duplicates the
    other.
    """
    blocks = []
    for node in nodes:
        resolved = evidence_for(node, graph)
        lines = [f"{node.id}  {node.kind}/{node.status}  {_detail_text(node, 90)}"]
        for label, key in (("consumed", "inputs"), ("produced", "outputs")):
            for item in resolved[key]:
                verified = item.get("verified")
                mark = {True: "verified", False: "UNVERIFIED", None: "unchecked"}.get(verified, str(verified))
                lines.append(
                    f"    {label:<8} {str(item.get('id'))[:12]}  {str(item.get('value'))[:40]!r}"
                    f"  page={item.get('page_id') or '-'}  {mark}"
                )
        for key, label in (("missing_inputs", "consumed"), ("missing_outputs", "produced")):
            for eid in resolved[key]:
                lines.append(f"    {label:<8} {eid[:12]}  <NOT A NODE IN THE GRAPH>")
        for page in resolved["pages"]:
            lines.append(f"    page     {page.get('page_id')}  {page.get('url')}  chars={page.get('chars')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(no nodes matched)"


def format_size(cell: Dict[str, Any], trace: LedgerTrace) -> str:
    """The trace's byte cost against the encodings of the same facts already in the cell."""
    execution = cell.get("execution") if isinstance(cell.get("execution"), dict) else {}
    observability = execution.get("observability") if isinstance(execution.get("observability"), dict) else {}
    telemetry_raw = execution.get("telemetry_raw") if isinstance(execution.get("telemetry_raw"), dict) else {}

    def size(value: Any) -> int:
        return len(json.dumps(value, separators=(",", ":"))) if value else 0

    rows = [
        ("telemetry_raw.timings", size(telemetry_raw.get("timings"))),
        ("observability.timings_per_call", size(observability.get("timings_per_call"))),
    ]
    subsumed = sum(n for _, n in rows)
    trace_bytes = size(trace.as_dict())
    lines = ["encoding                          bytes", "-" * 42]
    lines.extend(f"{name:<33} {count:>7}" for name, count in rows)
    lines.append(f"{'= subsumed today':<33} {subsumed:>7}")
    lines.append(f"{'ledger_trace':<33} {trace_bytes:>7}")
    if subsumed:
        lines.append(f"{'reduction':<33} {100.0 * (1 - trace_bytes / subsumed):>6.1f}%")
    output = execution.get("output") if isinstance(execution.get("output"), dict) else {}
    pages = output.get("pages")
    if pages and pages == (output.get("evidence_graph") or {}).get("pages"):
        lines.append("")
        lines.append(
            f"note: execution.output.pages is byte-identical to output.evidence_graph.pages "
            f"({size(pages)} bytes stored twice)"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query and draw a run's ledger trace.")
    parser.add_argument("cell_path", help="Path to an agent/idea_test_results/*.json cell")
    parser.add_argument("--kind", help="Filter: node kind (run/llm/search/visit/http/chroma/derive/decision/other)")
    parser.add_argument("--status", help="Filter: node status (ok/error/empty/refused/invalid/unknown)")
    parser.add_argument("--page-id", dest="page_id", help="Filter: nodes touching this page id")
    parser.add_argument("--evidence", help="Filter: nodes consuming OR producing this evidence node id")
    parser.add_argument("--node", help="Filter: one node by trace id")
    parser.add_argument(
        "--format", choices=("table", "json", "mermaid"), default="table",
        help="table (default), json (renderer-ready), or mermaid (paste into a ```mermaid block)",
    )
    parser.add_argument("--resolve", action="store_true", help="Follow each matched node into the evidence graph")
    parser.add_argument("--size", action="store_true", help="Report the trace's bytes against what it subsumes")
    args = parser.parse_args(argv)

    cell = load_cell(args.cell_path)
    trace = project_cell(cell)
    graph = evidence_graph_of(cell)

    nodes = trace.query(
        kind=args.kind, status=args.status, page_id=args.page_id, evidence_id=args.evidence
    )
    if args.node:
        nodes = [n for n in nodes if n.id == args.node]

    if args.size:
        print(format_size(cell, trace))
        return 0
    if args.resolve:
        print(format_resolved(nodes, graph))
        return 0
    if args.format == "json":
        json.dump({"schema": trace.as_dict()["schema"], "counts": trace.counts(),
                   "nodes": [n.as_dict() for n in nodes]}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.format == "mermaid":
        print(to_mermaid(LedgerTrace(nodes), graph))
        return 0
    print(format_table(trace, nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
