#!/usr/bin/env python3
"""Forensic extraction over the dagbase_20260824 four-way baseline result JSONs.

Answers (see docs/handoffs/ for the write-up):
  A. Does the native DAG (graph/sequential) actually fan out & execute on wide-breadth tasks?
  B. Does it ever execute siblings concurrently?
  C. Do the flat arms (seqreact/langgraph) loop / exhaust their step budget?
  D. Where does the DAG lose score on breadth: not-visiting, visiting-not-extracting, or
     extracting-but-not-reporting?

Reads agent/idea_test_results/dagbase_20260824_{graph,sequential,seqreact,langgraph}_*.json
(no LLM calls, no benchmark execution -- pure static analysis of existing artifacts).

Writes docs/handoffs/data/dagbase_20260824_mechanism.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

RESULTS_DIR = "agent/idea_test_results"
OUT_CSV = "docs/handoffs/data/dagbase_20260824_mechanism.csv"

FNAME_RE = re.compile(
    r"dagbase_20260824_(?P<arm>graph|sequential|seqreact|langgraph)_"
    r"(?P<task>\d+)_(?P<model>[^_]+)_(?P<variant>.+?)_cfg[0-9a-f]+_r(?P<rep>\d+)\.json$"
)

WIDE_BREADTH_TASKS = {"152", "156"}
NARROW_BREADTH_TASKS = {"154", "155"}


def node_depth(nodes: dict, node_id: str, cache: dict) -> int:
    if node_id in cache:
        return cache[node_id]
    n = nodes.get(node_id)
    if not n or not n.get("parent_id") or n["parent_id"] not in nodes:
        cache[node_id] = 0
        return 0
    d = 1 + node_depth(nodes, n["parent_id"], cache)
    cache[node_id] = d
    return d


def analyze_native(execution: dict) -> dict:
    """graph / sequential arms: real DAG with execution.graph.nodes"""
    g = execution.get("graph") or {}
    nodes = g.get("nodes") or {}
    root_id = g.get("root_id")
    out = {}
    out["node_count"] = len(nodes)

    # fan-out: max number of children under any single node
    max_fanout = 0
    for n in nodes.values():
        children = n.get("children") or []
        max_fanout = max(max_fanout, len(children))
    out["max_fanout_created"] = max_fanout
    out["root_children_count"] = len((nodes.get(root_id) or {}).get("children") or []) if root_id else None

    # depth
    depth_cache = {}
    depths = [node_depth(nodes, nid, depth_cache) for nid in nodes]
    out["max_depth"] = max(depths) if depths else 0

    # status counts
    status_counts = Counter(n.get("status") for n in nodes.values())
    out["status_done"] = status_counts.get("done", 0)
    out["status_skipped"] = status_counts.get("skipped", 0)
    out["status_failed"] = status_counts.get("failed", 0)
    out["status_pending"] = status_counts.get("pending", 0)
    out["status_other"] = sum(
        v for k, v in status_counts.items() if k not in ("done", "skipped", "failed", "pending")
    )

    # action-type breakdown, and executed vs created per type
    action_created = Counter()
    action_done = Counter()
    search_terms = set()
    visited_urls = set()
    for n in nodes.values():
        det = n.get("details") or {}
        action = det.get("action")
        if action is None:
            continue  # root node
        action_created[action] += 1
        if n.get("status") == "done":
            action_done[action] += 1
        if action == "search" and det.get("query"):
            search_terms.add(det["query"].strip().lower())
        if action == "visit":
            url = det.get("optional_url") or (det.get("action_result") or {}).get("url")
            if url:
                visited_urls.add(url.strip())

    out["search_nodes_created"] = action_created.get("search", 0)
    out["search_nodes_done"] = action_done.get("search", 0)
    out["visit_nodes_created"] = action_created.get("visit", 0)
    out["visit_nodes_done"] = action_done.get("visit", 0)
    out["merge_nodes_created"] = action_created.get("merge", 0)
    out["merge_nodes_done"] = action_done.get("merge", 0)
    out["distinct_search_terms"] = len(search_terms)
    out["distinct_visited_urls"] = len(visited_urls)

    # termination signal
    output = execution.get("output") or {}
    out["pending_nodes_count"] = output.get("pending_nodes_count")
    out["grounding_replans"] = output.get("grounding_replans")
    out["success"] = output.get("success")
    out["goal_achieved"] = output.get("goal_achieved")
    out["missing_requirements_count"] = len(output.get("missing_requirements") or [])
    out["parallel_leaves_total"] = (output.get("got_stats") or {}).get("parallel_leaves_total")

    return out


def analyze_flat(execution: dict) -> dict:
    """seqreact / langgraph arms: flat loop, empty execution.graph"""
    out = {}
    output = execution.get("output") or {}
    out["success"] = output.get("success")
    out["goal_achieved"] = output.get("goal_achieved")
    # NOTE: no per-step / per-query trace is recoverable for these arms in this
    # artifact set -- the referenced telemetry .jsonl trace files
    # (execution.telemetry.trace_file) do not exist on disk for this run, so
    # loop/repeated-action detection at the query/URL-text level is NOT POSSIBLE
    # from these JSONs. Only aggregate counts below are recoverable.
    return out


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "dagbase_20260824_*.json")))
    rows = []
    unmatched = []
    for fp in files:
        fname = os.path.basename(fp)
        if fname.endswith("_summary.json"):
            continue  # per-arm aggregate summary file, not a per-cell result
        m = FNAME_RE.search(fname)
        if not m:
            unmatched.append(fname)
            continue
        with open(fp) as f:
            d = json.load(f)

        arm = m.group("arm")
        task = m.group("task")
        rep = m.group("rep")

        execution = d.get("execution") or {}
        validation = d.get("validation") or {}
        obs = execution.get("observability") or {}

        row = {
            "file": fname,
            "task_id": task,
            "arm": arm,
            "rep": rep,
            "model": d.get("model"),
            "score": validation.get("overall_score"),
            "checks_passed": validation.get("checks_passed"),
            "total_checks": validation.get("total_checks"),
            "infra_failed": d.get("infra_failed"),
            "duration_seconds": execution.get("duration_seconds"),
            "prompt_tokens": (obs.get("cost") or {}).get("prompt_tokens"),
            "completion_tokens": (obs.get("cost") or {}).get("completion_tokens"),
            "total_tokens": (obs.get("llm") or {}).get("total_tokens"),
            "llm_calls": (obs.get("llm") or {}).get("calls"),
            "visit_count_obs": (obs.get("visit") or {}).get("count"),
            "search_count_obs": (obs.get("search") or {}).get("count"),
            "trace_file_exists": os.path.exists((execution.get("telemetry") or {}).get("trace_file", "")),
        }

        if arm in ("graph", "sequential"):
            row.update(analyze_native(execution))
        else:
            row.update(analyze_flat(execution))

        rows.append(row)

    # union of all keys, stable order (common cols first)
    common = [
        "file", "task_id", "arm", "rep", "model", "score", "checks_passed", "total_checks",
        "infra_failed", "duration_seconds", "prompt_tokens", "completion_tokens", "total_tokens",
        "llm_calls", "visit_count_obs", "search_count_obs", "trace_file_exists",
    ]
    extra_keys = []
    for r in rows:
        for k in r:
            if k not in common and k not in extra_keys:
                extra_keys.append(k)
    fieldnames = common + sorted(extra_keys)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} files did not match filename pattern:")
        for u in unmatched:
            print("  ", u)

    # ---- quick console summary for the wide-breadth tasks (152/156) ----
    print("\n=== WIDE BREADTH (152, 156): fan-out created vs executed, graph & sequential ===")
    for r in rows:
        if r["task_id"] in WIDE_BREADTH_TASKS and r["arm"] in ("graph", "sequential"):
            print(
                f"  {r['arm']:10s} task={r['task_id']} rep={r['rep']} "
                f"root_children={r.get('root_children_count')} "
                f"search(created/done)={r.get('search_nodes_created')}/{r.get('search_nodes_done')} "
                f"visit(created/done)={r.get('visit_nodes_created')}/{r.get('visit_nodes_done')} "
                f"merge(created/done)={r.get('merge_nodes_created')}/{r.get('merge_nodes_done')} "
                f"distinct_urls={r.get('distinct_visited_urls')} score={r['score']:.3f}"
            )

    print("\n=== WIDE BREADTH (152, 156): flat arms (seqreact/langgraph) visit/search counts ===")
    for r in rows:
        if r["task_id"] in WIDE_BREADTH_TASKS and r["arm"] in ("seqreact", "langgraph"):
            print(
                f"  {r['arm']:10s} task={r['task_id']} rep={r['rep']} "
                f"llm_calls={r['llm_calls']} visit_obs={r['visit_count_obs']} "
                f"search_obs={r['search_count_obs']} score={r['score']:.3f} "
                f"trace_file_exists={r['trace_file_exists']}"
            )


if __name__ == "__main__":
    main()
