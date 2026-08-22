#!/usr/bin/env python3
"""Why do ``identify_prune_candidates`` and ``should_backtrack`` never fire? $0, offline.

ASSUMPTION_AUDIT.md's dead-calibration table records the symptom -- 0 nodes pruned and 0
backtracks across 261 runs -- without a cause. Two candidate causes are testable against the
recorded corpus alone:

* **Prune.** The default-ON adaptive branch sets ``threshold = mean - prune_stddev_factor *
  stddev`` over every scored non-root node, and the flat ``got_prune_score_threshold`` (0.15)
  only when fewer than 5 nodes are scored. The candidate set the threshold is then applied to
  is exactly the nodes with **no ``action_result``** -- the same not-yet-executed
  subpopulation ``evaluate_batch`` floors at ``no_action_result_base_score`` (0.4) and clips
  at ``evaluation_no_action_result_score_cap`` (0.5) (T1-3a, T1-5). If the pending scores live
  in a narrow band anchored on those two constants, both thresholds sit under the whole band
  by construction.
* **Backtrack.** ``should_backtrack`` walks ``path_to_root`` from the current node, counts the
  leading run of nodes scoring below ``backtrack_low_score_threshold``, and fires at
  ``backtrack_dead_end_threshold`` (5). That count is bounded above by the node's depth, so
  the question is simply how deep recorded graphs get.

Both are measured here over the same result-JSON corpus and node-extraction pattern as
``scripts/analyze_beam_spread_contamination.py``. Analysis only: reads result JSON and prints.

**Result (2026-08-22, ASSUMPTION_AUDIT.md T1-6).** The prune hypothesis is REFUTED on both
counts and the audit's "0 nodes pruned" line is stale: ``_got_pruned`` is stamped on 108 nodes
across 75 runs, at scores 0.0-0.4, and only 0.2% of the pending population carries a capped
score. The backtrack finding is CONFIRMED: maximum node depth in the corpus is 3 against a
threshold of 5.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/analyze_prune_backtrack_deadzone.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_DEFAULT = REPO_ROOT / "agent" / "idea_test_results"
SETTINGS_PATH = REPO_ROOT / "agent" / "app" / "idea_dag_settings.json"

FALLBACK_STDDEV_FACTOR = 1.0
FALLBACK_FLAT_THRESHOLD = 0.15
FALLBACK_MIN_NODES = 6
FALLBACK_LOW_SCORE = 0.3
FALLBACK_DEAD_END = 5

TERMINAL_STATUSES = {"done", "failed", "skipped"}


@dataclass(frozen=True)
class Params:
    stddev_factor: float = FALLBACK_STDDEV_FACTOR
    flat_threshold: float = FALLBACK_FLAT_THRESHOLD
    min_nodes: int = FALLBACK_MIN_NODES
    low_score: float = FALLBACK_LOW_SCORE
    dead_end: int = FALLBACK_DEAD_END

    @classmethod
    def from_settings_file(cls, path: Path = SETTINGS_PATH) -> "Params":
        try:
            settings = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return cls()
        return cls(
            # Absent from the settings file on purpose (see config.py); the typed default rules.
            stddev_factor=float(settings.get("got_prune_stddev_factor", FALLBACK_STDDEV_FACTOR)),
            flat_threshold=float(settings.get("got_prune_score_threshold", FALLBACK_FLAT_THRESHOLD)),
            min_nodes=int(settings.get("got_prune_min_nodes_before_prune", FALLBACK_MIN_NODES)),
            low_score=float(settings.get("got_backtrack_low_score_threshold", FALLBACK_LOW_SCORE)),
            dead_end=int(settings.get("got_backtrack_dead_end_threshold", FALLBACK_DEAD_END)),
        )


@dataclass(frozen=True)
class Node:
    node_id: str
    parent_id: Optional[str]
    score: Optional[float]
    raw_score: Optional[float]
    capped: bool
    status: str
    has_result: bool
    depth: int
    pruned: bool

    @property
    def counterfactual(self) -> Optional[float]:
        if self.raw_score is not None:
            return float(self.raw_score)
        return self.score


def _num(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def nodes_from_result(payload: Dict[str, Any]) -> List[Node]:
    graph = ((payload.get("execution") or {}).get("graph") or {})
    raw_nodes = graph.get("nodes")
    if not isinstance(raw_nodes, dict):
        return []
    root_id = graph.get("root_id")

    depth: Dict[str, int] = {}

    def node_depth(node_id: str, guard: int = 0) -> int:
        if node_id in depth:
            return depth[node_id]
        node = raw_nodes.get(node_id)
        if not isinstance(node, dict) or guard > 200:
            return 0
        parent = node.get("parent_id") or (node.get("parent_ids") or [None])[0]
        value = 0 if not parent or parent == node_id else node_depth(parent, guard + 1) + 1
        depth[node_id] = value
        return value

    out: List[Node] = []
    for node_id, node in raw_nodes.items():
        if not isinstance(node, dict):
            continue
        if node_id == root_id or not (node.get("parent_id") or node.get("parent_ids")):
            continue
        details = node.get("details") or {}
        evaluation = details.get("evaluation") if isinstance(details, dict) else None
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        parent = node.get("parent_id") or (node.get("parent_ids") or [None])[0]
        out.append(
            Node(
                node_id=node_id,
                parent_id=parent,
                score=_num(node.get("score")),
                raw_score=_num(evaluation.get("raw_score")),
                capped=bool(evaluation.get("capped")),
                status=str(node.get("status") or "").lower(),
                has_result=bool(
                    isinstance(details, dict) and details.get("action_result") is not None
                ),
                depth=node_depth(node_id),
                pruned=bool(isinstance(details, dict) and details.get("_got_pruned")),
            )
        )
    return out


def load_runs(results_dir: Path, pattern: str) -> List[Tuple[str, List[Node]]]:
    runs: List[Tuple[str, List[Node]]] = []
    for path in sorted(results_dir.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        nodes = nodes_from_result(payload)
        if nodes:
            runs.append((str(path), nodes))
    return runs


def _threshold(scores: Sequence[float], params: Params) -> Tuple[float, str]:
    """``identify_prune_candidates``' branch, verbatim."""
    if len(scores) >= 5:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        return max(0.0, mean - params.stddev_factor * variance ** 0.5), "adaptive"
    return params.flat_threshold, "flat"


def analyse(runs: Sequence[Tuple[str, List[Node]]], params: Params) -> Dict[str, Any]:
    pending_scores: List[float] = []
    pending_cf: List[float] = []
    capped_pending = 0
    rows: List[Dict[str, Any]] = []
    depths: List[int] = []
    max_low_runs: List[int] = []
    pending_status: Counter = Counter()
    under_threshold_pending: List[str] = []
    #: The decisive ground truth: nodes ``prune_nodes`` actually stamped, live.
    observed_prunes: List[Optional[float]] = []
    runs_with_prune = 0

    for _source, nodes in runs:
        depths.extend(n.depth for n in nodes)
        stamped = [n for n in nodes if n.pruned]
        if stamped:
            runs_with_prune += 1
            observed_prunes.extend(n.score for n in stamped)
        for node in nodes:
            if node.score is None or node.has_result:
                continue
            pending_scores.append(node.score)
            cf = node.counterfactual
            if cf is not None:
                pending_cf.append(cf)
            if node.capped:
                capped_pending += 1

        if len(nodes) + 1 < params.min_nodes:
            continue
        scored = [n.score for n in nodes if n.score is not None]
        if not scored:
            continue
        threshold, branch = _threshold(scored, params)
        cf_scored = [n.counterfactual for n in nodes if n.counterfactual is not None]
        cf_threshold, _ = _threshold(cf_scored, params) if cf_scored else (threshold, branch)

        pending_here = [n for n in nodes if n.score is not None and not n.has_result]
        for node in pending_here:
            pending_status.update([node.status])
            if node.score is not None and node.score < threshold:
                under_threshold_pending.append(node.status)
        eligible = [n for n in pending_here if n.status not in TERMINAL_STATUSES]
        eligible_scores = [n.score for n in eligible if n.score is not None]
        rows.append(
            {
                "branch": branch,
                "threshold": threshold,
                "cf_threshold": cf_threshold,
                "min_scored": min(scored),
                "eligible": len(eligible),
                "eligible_min": min(eligible_scores) if eligible_scores else None,
                "pruned": sum(1 for s in eligible_scores if s < threshold),
                "pruned_flat": sum(1 for s in eligible_scores if s < params.flat_threshold),
                "below_threshold_any": sum(1 for s in scored if s < threshold),
                "margin": min(scored) - threshold,
            }
        )

        # Backtrack: the leading low-score run ``should_backtrack`` would count at each node,
        # walking ``path_to_root`` exactly as the engine does (root carries no score, so the
        # walk always terminates there at the latest).
        by_id = {n.node_id: n for n in nodes}
        best = 0
        for node in nodes:
            count = 0
            cursor: Optional[Node] = node
            guard = 0
            while cursor is not None and guard < 200:
                guard += 1
                if cursor.score is not None and cursor.score < params.low_score:
                    count += 1
                else:
                    break
                cursor = by_id.get(cursor.parent_id) if cursor.parent_id else None
            best = max(best, count)
        max_low_runs.append(best)

    return {
        "params": vars(params),
        "runs": len(runs),
        "pending": {
            "n": len(pending_scores),
            "capped": capped_pending,
            "min": min(pending_scores) if pending_scores else None,
            "p05": _pct(pending_scores, 0.05),
            "median": statistics.median(pending_scores) if pending_scores else None,
            "max": max(pending_scores) if pending_scores else None,
            "hist": dict(sorted(Counter(round(s, 2) for s in pending_scores).items())),
            "cf_min": min(pending_cf) if pending_cf else None,
            "cf_p05": _pct(pending_cf, 0.05),
        },
        "prune_rows": rows,
        "observed_prunes": {
            "nodes": len(observed_prunes),
            "runs": runs_with_prune,
            "score_hist": dict(sorted(Counter(observed_prunes).items(), key=lambda kv: -kv[1])),
        },
        "pending_status": dict(pending_status.most_common()),
        "under_threshold_pending": dict(Counter(under_threshold_pending).most_common()),
        "depth": {
            "n": len(depths),
            "max": max(depths) if depths else None,
            "hist": dict(sorted(Counter(depths).items())),
        },
        "max_low_run": {
            "n": len(max_low_runs),
            "max": max(max_low_runs) if max_low_runs else None,
            "hist": dict(sorted(Counter(max_low_runs).items())),
        },
    }


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * q)))]


def render(report: Dict[str, Any]) -> None:
    print(f"=== corpus: {report['runs']} runs ===")
    pending = report["pending"]
    print("\n=== pending (no action_result) scored nodes -- the prune-eligible population ===")
    print(
        f"  n={pending['n']}  capped={pending['capped']} "
        f"({pending['capped'] / max(1, pending['n']):.1%})"
    )
    print(
        f"  score min={pending['min']} p05={pending['p05']} "
        f"median={pending['median']} max={pending['max']}"
    )
    top = dict(list(sorted(pending["hist"].items(), key=lambda kv: -kv[1]))[:8])
    print(f"  most common scores: {top}")
    print(f"  raw_score (uncapped) min={pending['cf_min']} p05={pending['cf_p05']}")

    obs = report["observed_prunes"]
    print(
        f"\n=== nodes the live pruner ACTUALLY stamped (_got_pruned) ===\n"
        f"  {obs['nodes']} nodes across {obs['runs']} / {report['runs']} runs "
        f"({obs['runs'] / max(1, report['runs']):.1%})\n"
        f"  their scores: {obs['score_hist']}"
    )

    rows = report["prune_rows"]
    if rows:
        adaptive = [r for r in rows if r["branch"] == "adaptive"]
        print(f"\n=== reconstructed prune thresholds ({len(rows)} runs past min_nodes) ===")
        print(f"  adaptive branch used in {len(adaptive)}/{len(rows)} runs")
        thr = [r["threshold"] for r in rows]
        print(
            f"  threshold min={min(thr):.3f} mean={sum(thr) / len(thr):.3f} max={max(thr):.3f}"
        )
        cthr = [r["cf_threshold"] for r in rows]
        print(
            f"  threshold on raw_score min={min(cthr):.3f} "
            f"mean={sum(cthr) / len(cthr):.3f} max={max(cthr):.3f}"
        )
        margins = [r["margin"] for r in rows]
        print(
            f"  (lowest scored node - threshold) min={min(margins):.3f} "
            f"mean={sum(margins) / len(margins):.3f}; negative means something WOULD be under it"
        )
        print(f"  runs where some scored node is under the threshold: "
              f"{sum(1 for r in rows if r['below_threshold_any'])}")
        print(f"  runs with >=1 prune-ELIGIBLE node at end of run: "
              f"{sum(1 for r in rows if r['eligible'])}")
        print(f"  nodes pruned (adaptive): {sum(r['pruned'] for r in rows)}; "
              f"flat 0.15: {sum(r['pruned_flat'] for r in rows)}")
        print(f"  pending scored nodes by status: {report['pending_status']}")
        print(f"  ... of those, UNDER the threshold: {report['under_threshold_pending']}")

    depth = report["depth"]
    print("\n=== node depth (bounds should_backtrack's consecutive_low count) ===")
    print(f"  n={depth['n']}  max={depth['max']}  histogram={depth['hist']}")
    runs_ = report["max_low_run"]
    print(
        f"  best per-run leading low-score run (self only): max={runs_['max']} "
        f"histogram={runs_['hist']}"
    )
    print(f"  dead_end_threshold={report['params']['dead_end']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--pattern", default="**/*.json")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    params = Params.from_settings_file()
    runs = load_runs(args.results_dir, args.pattern)
    if not runs:
        print(f"no scored runs under {args.results_dir}", file=sys.stderr)
        return 1
    report = analyse(runs, params)
    render(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
