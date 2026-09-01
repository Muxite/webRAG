#!/usr/bin/env python3
"""Precision/recall at each pipeline edge: retrieved -> extracted -> verified -> stated.

Follows RAGChecker's decomposition (arXiv 2408.08067, verified in
``docs/research/EVIDENCE_COMPILER_CITATION_LEDGER_2026-08-31.md`` -- its claim-level mechanism is
in the BODY, section 3.2/3.3, not the abstract): "a text-to-claim extractor that decomposes a
given text T into a set of claims, and a claim-entailment checker" (SS3.2), separated into
"Diagnostic Retriever Metrics ... Diagnostic Generator Metrics" (SS3.3) so a score gap can be
localized to retrieval or generation instead of guessed at. RAGChecker's OWN checkers are LLM
calls (a caveat the doc raises explicitly, given this repo's merge-AUC history with LLM judges);
this module borrows the DECOMPOSITION, not the judge -- every edge here is computed from the
mechanically-verified fields ``execution_evidence_loop.py`` already produces
(``Extraction.value_verified`` / ``quote_verified``, the evidence graph's ``derivation_valid``),
never a fresh model call.

Four stages, each a claim that survived (or didn't) the stage before it:

  retrieved  -- a page the arm actually fetched (``output["pages"]``)
  extracted  -- a typed ``(entity, field, value)`` record the per-hop extraction call read off
                one of those pages (``output["extractions"]``)
  verified   -- an extracted record whose VALUE was mechanically located on the page it cites
                (``Extraction.value_verified is True`` -- the stronger of the two checks per
                ``evidence_graph.py``'s own measurement: value verification catches 3.0x more of
                a corpus than quote verification alone)
  stated     -- a verified value that the final answer text actually asserts

This is only computable for arms that emit typed extraction records at all: ``evidence_loop`` and
``sequential_react_extract``. ``langgraph_react`` and plain ``sequential_react`` have no
retrieved->extracted edge to measure (no ``extractions`` field), exactly as ``arm_verdict.py``
documents for its own, cruder text/evidence grounding check -- this module does not paper over
that gap either.

Also computes the FABRICATED-ARITHMETIC RATE: the fraction of the run's DERIVED evidence-graph
nodes (Lane A, commit 1464afa8 / ``evidence_graph.py``) whose value disagreed with the model's own
recomputation. A derived node's value is ALWAYS the Python recomputation; a model's disagreeing
proposal is recorded and marks the node invalid (``derivation_valid=False``) rather than silently
overwritten -- see ``evidence_graph.py``'s module docstring. This rate is now testable against
real data: 65 of the 66 ``evidence_loop`` cells in the ``ledgernum22r3`` campaign carry a
populated ``evidence_graph``, together holding 248 source nodes and 57 derived nodes, with zero
derivations marked invalid -- so on this data the measured fabrication rate is 0.0 wherever a
graph is present, not an untestable ``None``. ``None`` is still returned, and must stay that way,
for any cell without a graph at all (an arm that doesn't build one, or a crashed run) -- an
absent graph is not evidence of a 0% rate, only of nothing to measure.

Usage:
  python3 scripts/claim_metrics.py                                  # pipeline edges, all arms
  python3 scripts/claim_metrics.py --variant evidence_loop --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common  # noqa: E402
import bench_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.app.idea_test_utils import extract_final_text  # noqa: E402
from agent.app.testing.arm_verdict import _claims  # noqa: E402

# Arms whose output contract carries typed extraction records at all.
EXTRACTION_VARIANTS = {"evidence_loop", "sequential_react_extract"}


def pipeline_edges(output: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieved/extracted/verified/stated counts and the precision/recall between adjacent
    stages, for ONE cell's ``output`` dict. ``None`` when ``output`` has no ``extractions`` field
    at all (an arm outside :data:`EXTRACTION_VARIANTS`, or a crashed run)."""
    if not isinstance(output, dict) or "extractions" not in output:
        return None

    pages = output.get("pages") or []
    retrieved_page_ids = {str(p.get("page_id")) for p in pages if isinstance(p, dict) and p.get("page_id")}
    retrieved = len(pages)

    extractions = [e for e in (output.get("extractions") or [])
                  if isinstance(e, dict) and str(e.get("value") or "").strip()]
    extracted = len(extractions)
    extracted_page_ids = {str(e.get("page_id")) for e in extractions if e.get("page_id")}
    # Sanity check baked into the metric itself: an extraction citing a page_id that was never
    # retrieved would be a real bug (fabricated provenance), not a rounding error.
    orphaned = extracted_page_ids - retrieved_page_ids

    verified_records = [e for e in extractions if e.get("value_verified") is True]
    verified = len(verified_records)
    quote_verified_records = [e for e in extractions if e.get("quote_verified") is True]

    final_text = extract_final_text({"output": output}).lower()
    stated_records = [e for e in verified_records if str(e.get("value", "")).strip().lower() in final_text]
    stated = len(stated_records)

    # What the final answer itself asserts as checkable claims (arm_verdict's own claim
    # extractor, reused rather than reimplemented). NOTE what the counter below actually tests:
    # EXACT string equality between a claim token and an extraction's whole `value`. Real values
    # carry units and parentheticals ("6,300 km (3,900 mi)") while `_claims` yields bare tokens
    # ("6300"), and on this suite the answer states DERIVED values while extractions hold the RAW
    # operands -- so the two sets are largely disjoint BY CONSTRUCTION. It was previously named
    # `stated_claim_precision`, which read as "how much of the answer is supported" and reported
    # 0.040 for evidence_loop; the arm-blind auditor measures 0.224 unsupported for the same arm.
    # It is a verbatim-restatement counter, not a support metric, and is named that way now.
    stated_claims = _claims(extract_final_text({"output": output}))
    stated_claims_backed = sum(
        1 for claim in stated_claims
        if any(str(r.get("value", "")).strip().lower() == claim.lower() for r in verified_records)
    )

    return {
        "retrieved": retrieved,
        "extracted": extracted,
        "verified": verified,
        "quote_verified": len(quote_verified_records),
        "stated": stated,
        "orphaned_extractions": len(orphaned),
        "retrieved_to_extracted_recall": (len(extracted_page_ids & retrieved_page_ids) / retrieved
                                          if retrieved else None),
        "retrieved_to_extracted_precision": (len(extracted_page_ids & retrieved_page_ids) / len(extracted_page_ids)
                                             if extracted_page_ids else None),
        "extracted_to_verified_rate": (verified / extracted if extracted else None),
        "verified_to_stated_recall": (stated / verified if verified else None),
        "stated_claim_count": len(stated_claims),
        "stated_claim_verbatim_extraction_rate": (stated_claims_backed / len(stated_claims) if stated_claims else None),
    }


def _ratio(numer_key: str, denom_key: str, edges: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Micro-averaged ratio across cells: sum of numerators over sum of denominators."""
    numer = sum(e[numer_key] for e in edges)
    denom = sum(e[denom_key] for e in edges)
    return numer / denom if denom else None


def aggregate_pipeline_edges(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Micro-averaged pipeline-edge stats over every cell that has extraction records.

    :param cells: rows with an ``"output"`` key (a stored cell's ``execution.output``).
    """
    edges = [pipeline_edges(c["output"]) for c in cells]
    edges = [e for e in edges if e is not None]
    n = len(edges)
    if n == 0:
        return {"n": 0}
    retrieved_extracted = [e["retrieved_to_extracted_recall"] for e in edges
                           if e["retrieved_to_extracted_recall"] is not None]
    extracted_verified = [e["extracted_to_verified_rate"] for e in edges
                          if e["extracted_to_verified_rate"] is not None]
    verified_stated = [e["verified_to_stated_recall"] for e in edges
                       if e["verified_to_stated_recall"] is not None]
    stated_verbatim = [e["stated_claim_verbatim_extraction_rate"] for e in edges
                        if e["stated_claim_verbatim_extraction_rate"] is not None]
    return {
        "n": n,
        "total_retrieved": sum(e["retrieved"] for e in edges),
        "total_extracted": sum(e["extracted"] for e in edges),
        "total_verified": sum(e["verified"] for e in edges),
        "total_stated": sum(e["stated"] for e in edges),
        "total_orphaned_extractions": sum(e["orphaned_extractions"] for e in edges),
        "retrieved_to_extracted_recall_mean": bench_stats.mean(retrieved_extracted),
        "retrieved_to_extracted_recall_ci95": bench_stats.ci95(retrieved_extracted),
        "extracted_to_verified_rate_mean": bench_stats.mean(extracted_verified),
        "extracted_to_verified_rate_ci95": bench_stats.ci95(extracted_verified),
        "verified_to_stated_recall_mean": bench_stats.mean(verified_stated),
        "verified_to_stated_recall_ci95": bench_stats.ci95(verified_stated),
        "stated_claim_verbatim_extraction_rate_mean": bench_stats.mean(stated_verbatim),
        "stated_claim_verbatim_extraction_rate_ci95": bench_stats.ci95(stated_verbatim),
    }


def derivation_fabrication_rate(output: Dict[str, Any]) -> Optional[float]:
    """Fraction of DERIVED evidence-graph nodes whose recomputation disagreed with any value the
    model proposed for it (``derivation_valid is False``). ``None`` when this cell carries no
    ``evidence_graph`` or the graph has no derived nodes -- there is nothing to fabricate an
    arithmetic claim FROM, not a 0% fabrication rate."""
    graph = output.get("evidence_graph") if isinstance(output, dict) else None
    if not isinstance(graph, dict):
        return None
    nodes = graph.get("nodes") or []
    derived = [n for n in nodes if isinstance(n, dict) and n.get("kind") == "derived"]
    if not derived:
        return None
    fabricated = sum(1 for n in derived if n.get("derivation_valid") is False)
    return fabricated / len(derived)


def load_cells(run_ids: Optional[Sequence[str]] = None, *,
               extra_run_ids: Optional[Sequence[str]] = None, since: str = "",
               files: Optional[Sequence[str]] = None, tests: Optional[Sequence[str]] = None,
               variants: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every stored cell's ``output`` plus identifying metadata, scoped like
    :func:`risk_coverage.load_cells` (same ``bench_common.discover_files`` path scoping)."""
    ids = list(bench_common.DEFAULT_RUN_IDS) if run_ids is None else list(run_ids)
    if extra_run_ids:
        ids = ids + list(extra_run_ids)
    paths = bench_common.discover_files(ids, since=since, files=files)
    vset = {v.strip() for v in variants if v and v.strip()} if variants else None
    tset = {t.strip() for t in tests if t and t.strip()} if tests else None

    cells: List[Dict[str, Any]] = []
    for path in paths:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        variant = d.get("execution_variant")
        if not variant or (vset and variant not in vset):
            continue
        meta = d.get("test_metadata") or {}
        test_id = meta.get("test_id")
        if tset and test_id not in tset:
            continue
        ex = d.get("execution") or {}
        if not isinstance(ex, dict):
            continue
        output = ex.get("output")
        if not isinstance(output, dict):
            continue
        cells.append({"path": str(path), "test_id": test_id, "model": d.get("model"),
                      "variant": variant, "output": output})
    return cells


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-ids", default=None)
    parser.add_argument("--variant", default=",".join(sorted(EXTRACTION_VARIANTS)))
    parser.add_argument("--tests", default=None)
    parser.add_argument("--since", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    run_ids = args.run_ids.split(",") if args.run_ids is not None else None
    if args.run_ids == "":
        run_ids = []
    variants = args.variant.split(",") if args.variant else None
    tests = args.tests.split(",") if args.tests else None

    cells = load_cells(run_ids=run_ids, since=args.since, tests=tests, variants=variants)
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_variant.setdefault(cell["variant"], []).append(cell)

    report = {}
    fab_rates = []
    for variant, vcells in by_variant.items():
        report[variant] = aggregate_pipeline_edges(vcells)
        for cell in vcells:
            rate = derivation_fabrication_rate(cell["output"])
            if rate is not None:
                fab_rates.append(rate)
    fabrication_summary = {
        "n_cells_with_evidence_graph": len(fab_rates),
        "mean": bench_stats.mean(fab_rates) if fab_rates else None,
    }

    if args.json:
        print(json.dumps({"pipeline_edges": report, "fabrication": fabrication_summary}, indent=2))
    else:
        for variant, agg in report.items():
            print(f"== {variant} (n={agg['n']}) ==")
            if agg["n"] == 0:
                print("  no cells with extraction records")
                continue
            print(f"  retrieved={agg['total_retrieved']} extracted={agg['total_extracted']} "
                 f"verified={agg['total_verified']} stated={agg['total_stated']} "
                 f"orphaned_extractions={agg['total_orphaned_extractions']}")
            print(f"  retrieved->extracted recall: {agg['retrieved_to_extracted_recall_mean']:.3f} "
                 f"+/- {agg['retrieved_to_extracted_recall_ci95']:.3f}")
            print(f"  extracted->verified rate:    {agg['extracted_to_verified_rate_mean']:.3f} "
                 f"+/- {agg['extracted_to_verified_rate_ci95']:.3f}")
            print(f"  verified->stated recall:     {agg['verified_to_stated_recall_mean']:.3f} "
                 f"+/- {agg['verified_to_stated_recall_ci95']:.3f}")
            print(f"  verbatim-extraction rate:    {agg['stated_claim_verbatim_extraction_rate_mean']:.3f} "
                 f"+/- {agg['stated_claim_verbatim_extraction_rate_ci95']:.3f}")
        print(f"== fabricated-arithmetic rate: n_cells_with_evidence_graph="
             f"{fabrication_summary['n_cells_with_evidence_graph']} "
             f"mean={fabrication_summary['mean']} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
