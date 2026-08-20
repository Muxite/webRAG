#!/usr/bin/env python3
"""Offline ($0) similarity histogram for memory retrieval — calibrates E3's floor.

`memory_retrieval_similarity_floor` shipped default-OFF precisely because nobody knew where the
corpus sits: a floor above the typical similarity guts retrieval, one below it is inert
(ASSUMPTION_AUDIT.md E3). The `similarity` field the floor reads is computed at query time and
never serialized, so no stored result JSON carries the distribution. This script reconstructs it
without spending anything:

* the QUERIES are rebuilt from stored run JSONs exactly as ``idea_engine._expand_or_execute``
  builds them (node title + justification[:100] + parent goal[:100] + mandate[:100], then
  ``retrieve_memories_split``'s node-context suffix);
* the CORPUS is the run's own ``mem_*`` collection, still resident in the HTTP chroma, addressed
  through the same ``MemoryManager`` the engine uses, so the numbers come from the shipped code
  path rather than a re-implementation of it;
* embedding is chroma's bundled local MiniLM, so no LLM/API call is made.

Caveat worth carrying into any conclusion: a persistent collection accumulates across every run
that shared a mandate, so it is at least as large as it was at query time. More neighbours can
only push the *k* nearest closer, so every number here is an UPPER bound on the live similarity,
and a floor picked from it is correspondingly conservative.

Usage (from repo root):
    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/histogram_memory_similarity.py \
        --results 'agent/idea_test_results/*.json' --max-runs 40
"""
import argparse
import asyncio
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "services"), os.path.join(REPO, "agent")]

from shared.connector_config import ConnectorConfig  # noqa: E402
from agent.app.connector_chroma import ConnectorChroma  # noqa: E402
from agent.app.idea_memory import MemoryManager  # noqa: E402

# Candidate floors the report scores. 0.3/0.5 are the two values E3 pre-registered; the rest
# bracket them so the answer is not forced to be one of the two guesses.
CANDIDATE_FLOORS = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
BUCKET = 0.05


def memo_namespace(mandate):
    """Mirror of ``IdeaEngine._memo_namespace``."""
    return f"idea_dag:{hashlib.sha256(mandate.encode('utf-8')).hexdigest()[:10]}"


def collection_name(namespace):
    """Mirror of ``MemoryManager.__init__``'s collection naming."""
    return f"mem_{hashlib.sha256(namespace.encode('utf-8')).hexdigest()[:12]}"


def load_run(path):
    """(mandate, [query, ...]) for one result JSON, or None if it carries no usable graph."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception:
        return None
    graph = (data.get("execution") or {}).get("graph") or {}
    nodes = graph.get("nodes") or {}
    root_id = graph.get("root_id")
    root = nodes.get(root_id) or {}
    mandate = (root.get("details") or {}).get("mandate") or ""
    if not mandate or not nodes:
        return None
    queries = []
    for node_id, node in nodes.items():
        if node_id == root_id:
            continue
        details = node.get("details") or {}
        title = node.get("title") or ""
        if not title:
            continue
        # idea_engine._expand_or_execute's query assembly, then retrieve_memories_split's
        # node-context suffix (title again, plus action, plus any action error).
        parts = [title]
        justification = details.get("justification") or ""
        if justification:
            parts.append(str(justification)[:100])
        parent_goal = details.get("parent_goal") or ""
        if parent_goal:
            parts.append(str(parent_goal)[:100])
        parts.append(mandate[:100])
        context = [title]
        action = details.get("action")
        if action:
            context.append(f"action: {action}")
        error = details.get("action_error")
        if error:
            context.append(f"error: {error}")
        queries.append(" ".join(parts) + " " + " ".join(context))
    return (mandate, queries) if queries else None


async def sample_run(chroma, mandate, queries, n_internal, n_observations, per_run_cap):
    """Similarities for one run's queries against its own memory collection."""
    namespace = memo_namespace(mandate)
    manager = MemoryManager(connector_chroma=chroma, namespace=namespace)
    rows = []
    for q_index, query in enumerate(queries[:per_run_cap]):
        split = await manager.retrieve_memories_split(
            query=query,
            node_context=None,  # already folded into `query` above
            n_internal=n_internal,
            n_observations=n_observations,
        )
        for kind in ("internal_thoughts", "observations"):
            for rank, memory in enumerate(split[kind]):
                rows.append({
                    # (namespace, query index, kind) identifies one retrieval CALL, which is the
                    # unit a floor actually acts on: "how many rows does it cut" understates the
                    # risk if the cuts concentrate on a few calls it empties entirely.
                    "query": f"{namespace}#{q_index}#{kind}",
                    "kind": kind,
                    "rank": rank,
                    "similarity": float(memory.get("similarity") or 0.0),
                })
    return rows


def floor_effects(rows, floors=CANDIDATE_FLOORS):
    """Per-floor (row drop rate, emptied-call rate) — the two ways a floor can bite."""
    sims = [r["similarity"] for r in rows]
    calls = defaultdict(list)
    for r in rows:
        calls[r.get("query")].append(r["similarity"])
    out = []
    for floor in floors:
        dropped = sum(1 for s in sims if s < floor)
        emptied = sum(1 for vals in calls.values() if all(s < floor for s in vals))
        out.append({
            "floor": floor,
            "row_drop_rate": dropped / len(sims) if sims else 0.0,
            "emptied_call_rate": emptied / len(calls) if calls else 0.0,
        })
    return out


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def report(rows):
    sims = [r["similarity"] for r in rows]
    print(f"\nretrieved rows: {len(sims)}")
    if not sims:
        return
    print("percentiles: " + "  ".join(
        f"p{int(q * 100)}={percentile(sims, q):.3f}" for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
    print(f"min={min(sims):.3f} mean={sum(sims) / len(sims):.3f} max={max(sims):.3f}")

    print("\nhistogram (bucket width %.2f):" % BUCKET)
    buckets = Counter(int(s // BUCKET) for s in sims)
    for b in sorted(buckets):
        lo = b * BUCKET
        n = buckets[b]
        print(f"  [{lo:.2f},{lo + BUCKET:.2f})  {n:6d}  {'#' * max(1, round(60 * n / len(sims)))}")

    print("\nby rank (does the k-th neighbour fall off a cliff?):")
    by_rank = defaultdict(list)
    for r in rows:
        by_rank[r["rank"]].append(r["similarity"])
    for rank in sorted(by_rank):
        vals = by_rank[rank]
        print(f"  rank {rank}: n={len(vals):5d} mean={sum(vals) / len(vals):.3f} "
              f"p10={percentile(vals, 0.1):.3f} median={percentile(vals, 0.5):.3f}")

    print("\nby memory type:")
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["similarity"])
    for kind in sorted(by_kind):
        vals = by_kind[kind]
        print(f"  {kind:18s} n={len(vals):5d} mean={sum(vals) / len(vals):.3f} "
              f"p10={percentile(vals, 0.1):.3f} median={percentile(vals, 0.5):.3f}")

    print("\nper candidate floor: rows cut, and retrieval calls left with NOTHING:")
    for eff in floor_effects(rows):
        print(f"  floor {eff['floor']:.2f}: rows cut {100.0 * eff['row_drop_rate']:5.1f}%   "
              f"calls emptied {100.0 * eff['emptied_call_rate']:5.1f}%")


async def main_async(args):
    paths = [p for p in sorted(glob.glob(args.results), key=os.path.getmtime, reverse=True)
             if not (p.endswith("_summary.json") or "_report_" in os.path.basename(p))]
    chroma = ConnectorChroma(ConnectorConfig())
    live = set()
    try:
        collections = await chroma.list_collections()
        live = {c for c in collections}
    except Exception as exc:
        print(f"could not list collections ({exc}); will attempt every run anyway")

    rows, used, skipped = [], 0, 0
    for path in paths:
        if used >= args.max_runs:
            break
        loaded = load_run(path)
        if not loaded:
            continue
        mandate, queries = loaded
        name = collection_name(memo_namespace(mandate))
        if live and name not in live:
            skipped += 1
            continue
        got = await sample_run(chroma, mandate, queries, args.n_internal,
                               args.n_observations, args.queries_per_run)
        if not got:
            skipped += 1
            continue
        rows.extend(got)
        used += 1
        print(f"[{used:3d}] {os.path.basename(path)[:60]:60s} {name} "
              f"queries={min(len(queries), args.queries_per_run):3d} rows={len(got):4d}")

    print(f"\nruns sampled: {used}; runs skipped (no live collection / no rows): {skipped}")
    report(rows)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh)
        print(f"\nraw rows -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{REPO}/agent/idea_test_results/*.json")
    ap.add_argument("--max-runs", type=int, default=30)
    ap.add_argument("--queries-per-run", type=int, default=12)
    ap.add_argument("--n-internal", type=int, default=5)
    ap.add_argument("--n-observations", type=int, default=5)
    ap.add_argument("--out", default="")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
