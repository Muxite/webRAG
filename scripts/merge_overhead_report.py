#!/usr/bin/env python3
"""Merge-node overhead report: how many of DAG v2's action calls are merge synthesis calls?

Context: `SimpleMergePolicy` / `MergeLeafAction` fold synthesis + goal-eval into one LLM call per
>=2-child ancestor node, and `enable_recursive_merge` (default True) cascades that up the whole
subtree on the way back to root. ENGINE_DESIGN_REVIEW.md flags this as a cost source but never
actually counts it — merge calls land in the same `decisions.by_stage.action` bucket as
search/visit/verify, so they're invisible as their own line in any existing report. This counts
them directly from the raw result JSON, per run and pooled, so "merge overhead" stops being a
guess.

Only `graph`-arm results carry a `execution.graph.nodes` tree with merge nodes in it; the
`langgraph_react` / `sequential_react` arms don't run the merge policy at all and are skipped.

Two independent extraction paths are cross-checked against each other for every file:
  1. `execution.graph.nodes[*].details.action == "merge"` — the node itself.
  2. `execution.observability.decisions.trace[*]` entries with `stage == "action"` and
     `chosen == "merge"` — the decision log kept alongside `by_stage.action`, the denominator.
A mismatch between the two is reported loudly rather than silently trusted.

Per-node LLM cost is NOT recoverable from these result files: `execution.observability.llm` /
`.cost` are run-level aggregates only, and a merge node's `details.action_result` carries the
synthesized answer but no token/usage figures. This script reports merge-node COUNT (a hard
number) and labels cost/token attribution as unavailable rather than estimating it.
"""
import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOB = os.path.join(REPO, "agent", "idea_test_results",
                             "capability_spectrum_v2_20260820_*.json")


def variant_of(basename):
    # runner convention: `..._<graph|langgraph_react|sequential_react>_cfg<hash>_r<rep>.json`
    for v in ("langgraph_react", "sequential_react", "graph"):
        if f"_{v}_" in basename:
            return v
    return None


def load_one(path):
    base = os.path.basename(path)
    if base.endswith("_summary.json") or "_report_" in base:
        return None
    variant = variant_of(base)
    if variant is None:
        return None
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"!! failed to parse {base}: {e}", file=sys.stderr)
        return None

    execu = d.get("execution") or {}
    graph = execu.get("graph") or {}
    nodes = graph.get("nodes") or {}
    if variant != "graph" or not nodes:
        # langgraph_react / sequential_react have no merge policy at all; skip gracefully
        return {
            "path": base, "variant": variant, "model": d.get("model", ""),
            "has_graph": False, "merge_nodes": 0, "total_nodes": 0,
            "action_total": None, "trace_merge": None, "mismatch": False,
            "usd": None, "prompt_tok": None, "completion_tok": None, "llm_calls": None,
        }

    merge_node_count = sum(
        1 for n in nodes.values() if (n.get("details") or {}).get("action") == "merge"
    )

    obs = execu.get("observability") or {}
    decisions = obs.get("decisions") or {}
    by_stage = decisions.get("by_stage") or {}
    action_total = by_stage.get("action")
    trace = decisions.get("trace") or []
    trace_merge_count = sum(
        1 for t in trace if t.get("stage") == "action" and t.get("chosen") == "merge"
    )

    cost = obs.get("cost") or {}
    llm = obs.get("llm") or {}

    return {
        "path": base,
        "variant": variant,
        "model": d.get("model", ""),
        "has_graph": True,
        "merge_nodes": merge_node_count,
        "total_nodes": len(nodes),
        "action_total": action_total,
        "trace_merge": trace_merge_count,
        "mismatch": merge_node_count != trace_merge_count,
        "usd": cost.get("usd"),
        "prompt_tok": llm.get("prompt", {}).get("tokens"),
        "completion_tok": llm.get("completion", {}).get("tokens"),
        "llm_calls": llm.get("calls"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glob_pattern", nargs="?", default=DEFAULT_GLOB,
                     help="glob of result JSON files (default: capability_spectrum_v2_20260820_*)")
    ap.add_argument("--verbose", action="store_true",
                     help="print every matching file, not just graph-arm ones")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob_pattern))
    if not paths:
        print(f"no files matched: {args.glob_pattern}", file=sys.stderr)
        return 1

    rows = []
    skipped_no_variant = 0
    for p in paths:
        r = load_one(p)
        if r is None:
            skipped_no_variant += 1
            continue
        rows.append(r)

    graph_rows = [r for r in rows if r["has_graph"]]
    other_rows = [r for r in rows if not r["has_graph"]]

    print(f"{'='*100}\nMERGE-NODE OVERHEAD  ({args.glob_pattern})\n{'='*100}")
    print(f"{len(paths)} files matched, {skipped_no_variant} skipped (no recognizable arm suffix), "
          f"{len(other_rows)} skipped (non-graph arm, no merge policy), "
          f"{len(graph_rows)} graph-arm files analyzed\n")

    if not graph_rows:
        print("no graph-arm result files found — nothing to report")
        return 0

    mismatches = [r for r in graph_rows if r["mismatch"]]
    if mismatches:
        print(f"!! {len(mismatches)} files where node-count and decision-trace merge counts "
              f"DISAGREE (reported below per-row, both numbers shown)")
    else:
        print("cross-check OK on every graph-arm file: node-scan merge count == "
              "decision-trace merge count")
    print()

    print(f"{'file':70s}{'merge/action':>14s}{'total nodes':>13s}{'usd':>10s}")
    tot_merge = tot_action = 0
    cost_known_merge_runs = 0
    for r in graph_rows:
        at = r["action_total"]
        frac = f"{r['merge_nodes']}/{at}" if at is not None else f"{r['merge_nodes']}/?"
        if r["mismatch"]:
            frac += f" (trace={r['trace_merge']}!)"
        usd = f"${r['usd']:.4f}" if r["usd"] is not None else "—"
        print(f"{r['path'][:69]:70s}{frac:>14s}{r['total_nodes']:>13d}{usd:>10s}")
        tot_merge += r["merge_nodes"]
        if at is not None:
            tot_action += at
        if r["merge_nodes"] and r["usd"] is not None:
            cost_known_merge_runs += 1

    print(f"\n{'='*100}\nPOOLED\n{'='*100}")
    print(f"total merge nodes across all graph-arm files: {tot_merge}")
    print(f"total action calls (decisions.by_stage.action, summed): {tot_action}")
    if tot_action:
        print(f"merge share of all action calls: {tot_merge}/{tot_action} = "
              f"{100.0*tot_merge/tot_action:.1f}%")
    files_with_merge = sum(1 for r in graph_rows if r["merge_nodes"] > 0)
    print(f"graph-arm files with >=1 merge node: {files_with_merge}/{len(graph_rows)}")

    print(f"\n{'='*100}\nBY MODEL\n{'='*100}")
    by_model = collections.defaultdict(list)
    for r in graph_rows:
        by_model[r["model"]].append(r)
    print(f"{'model':32s}{'files':>7s}{'merge nodes':>13s}{'action calls':>14s}{'merge %':>9s}")
    for model, rs in sorted(by_model.items()):
        m = sum(r["merge_nodes"] for r in rs)
        a = sum(r["action_total"] or 0 for r in rs)
        pct = f"{100.0*m/a:.1f}%" if a else "—"
        print(f"{model:32s}{len(rs):>7d}{m:>13d}{a:>14d}{pct:>9s}")

    print(f"\n{'='*100}\nCOST / TOKEN ATTRIBUTION\n{'='*100}")
    print("Per-node LLM cost is NOT recoverable from these result files: "
          "execution.observability.llm/.cost are run-level aggregates only, and a merge node's "
          "details.action_result carries the synthesized answer with no token/usage fields.")
    print(f"Runs with >=1 merge node where a run-level usd figure exists: "
          f"{cost_known_merge_runs}/{files_with_merge}")
    tot_usd = sum(r["usd"] for r in graph_rows if r["usd"] is not None)
    tot_calls = sum(r["llm_calls"] for r in graph_rows if r["llm_calls"] is not None)
    if tot_calls:
        print(f"run-level average: ${tot_usd/tot_calls:.5f}/LLM-call across all graph-arm runs "
              f"(NOT merge-specific — merge calls typically carry a larger prompt than search/visit "
              f"since they concatenate all child results, so this likely understates a merge call's "
              f"true share of run cost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
