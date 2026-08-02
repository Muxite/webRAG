#!/usr/bin/env python3
"""Is ``node.score`` — the evaluation judge's number — worth backtracking on? $0, offline.

``got_operations.should_backtrack``/``find_backtrack_target`` abandon a chain by walking
``node.score`` along ``graph.path_to_root(current_id)``: count consecutive nodes below
``got_backtrack_low_score_threshold`` (0.3), trigger at ``got_backtrack_dead_end_threshold``
(5). That score comes from ``idea_policies/evaluation.py`` — a DIFFERENT LLM judge from
``judge_step_confidence``, whose weakness is already quantified in
``app/CONFIDENCE_JUDGE_MISCALIBRATION.md``. The evaluation score's predictive power had never
been measured in either direction. This script measures it, on recorded runs only.

Methodology is deliberately copied from ``analyze_confidence_judge_miscalibration.py`` — same
regular-roster filter, same ``validation.overall_score >= 0.75`` label, and the AUC/interval
maths is *imported* from that script rather than reimplemented, so the two docs' numbers are
computed identically and can be quoted side by side.

Two deviations from that script, both forced by the corpus and both material to read before
quoting anything:

1. **Different corpus.** The flat top-level ``idea_test_results/*.json`` population the
   confidence analysis used (354 July-2026 trajectories) is no longer on disk — only empty
   per-run directories remain, and the confidence script now reports "no usable trajectories".
   The surviving regular-roster runs with a recorded graph live in per-run *subdirectories*,
   so this script globs recursively (``--pattern``). They are older and are NOT the same runs.
2. **Different unit.** The confidence analysis' unit is a judged step from an observability
   trace; here it is a graph node carrying a numeric ``score``, reconstructed from
   ``execution.graph`` (the exact structure ``IdeaDag.to_dict`` writes).

What it reports:

* **availability** — how many roster runs record a graph at all, how many nodes in them carry
  a numeric score, broken down by ``execution_variant`` (only the variants that run the
  native GoT loop evaluate anything) and by action kind.
* **score distribution** — the value vocabulary, and how much of it sits exactly on the two
  penalty constants ``evaluation_no_action_result_base_score`` (0.4) and
  ``..._score_cap`` (0.5). ``evaluate``/``evaluate_batch`` are invoked from
  ``idea_engine`` *before* the candidate's action runs, so every scored node is
  "action but no action_result" and takes that penalty path; the batch prompt separately
  instructs "Nodes with actions but no action_result score <=0.2". Both are re-derived from
  the recorded scores here rather than trusted from the source.
* **backtrack reachability** — the path-to-root walk ``should_backtrack`` performs, replayed
  per node, with a sweep over ``(low_score_threshold, dead_end_threshold)``. This answers
  "could the mechanism ever have fired?" separately from "would it have been right?".
* **predictiveness** — run-level AUC of node-score aggregations (mean/min/max/first/last, the
  running-min along the deepest node's path to root — literally the statistic
  ``should_backtrack`` reads — and executed-nodes-only mean) against the eventual label, with
  the same free LLM-free structural baselines the confidence analysis used as its bar.
* **by model and by variant**, plus a node-level split by status (executed vs the siblings
  the selector dropped), which is the closest thing this corpus has to
  "on the answer path vs a pruned branch".
* **rationale availability** — whether there is any judge prose to quote at all.

Findings are written up in ``services/agent/app/EVALUATION_SCORE_PREDICTIVE_POWER.md``. This
script is analysis only: it reads result JSON and prints; it writes no artifact the engine
reads and changes no behaviour.

Usage::

    PYTHONPATH=services:services/agent ./.venv/bin/python \\
      scripts/analyze_evaluation_score_predictive_power.py

    ... --samples 8              # high-scoring nodes inside failed runs to print
    ... --json-out report.json   # also dump the whole report as JSON
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "services", _ROOT / "services" / "agent", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent.app.idea_policies.confidence_early_exit import LABEL_THRESHOLD  # noqa: E402
from analyze_confidence_judge_miscalibration import (  # noqa: E402
    auc_with_ci,
    roc_auc,
    summarize,
)
from calibrate_confidence_early_exit import DEFAULT_RESULTS_DIR, is_regular_roster  # noqa: E402

#: ``got_backtrack_low_score_threshold`` / ``got_backtrack_dead_end_threshold`` JSON defaults,
#: so every reachability number describes the rule the engine actually ships with.
BACKTRACK_LOW_SCORE = 0.3
BACKTRACK_DEAD_END = 5
#: Threshold/limit grids for the sweep: how much would either knob have to move before
#: ``should_backtrack`` ever returns True on a recorded run?
LOW_SCORE_GRID: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
DEAD_END_GRID: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: ``evaluation_no_action_result_base_score`` / ``evaluation_no_action_result_score_cap``
#: (``EvaluationConfig`` defaults). Every scored node is evaluated BEFORE its action runs, so
#: both constants are on the live path for every node; the distribution section measures how
#: much of the corpus lands exactly on them.
PENALTY_BASE_SCORE = 0.4
PENALTY_SCORE_CAP = 0.5
#: The ceiling the batch evaluation prompt instructs for unexecuted work ("Nodes with actions
#: but no action_result score <=0.2"), which is every candidate the judge is ever shown.
PROMPT_UNEXECUTED_CEILING = 0.2
#: Node status recorded for a candidate that was evaluated and then actually executed, versus
#: one the selector dropped. The closest proxy this corpus has for "on the answer path".
EXECUTED_STATUS = "done"
DROPPED_STATUS = "skipped"

#: Recorded ``execution_variant`` of the native Graph-of-Thoughts loop — the only place
#: ``should_backtrack`` is reachable. Other variants are reported but never headline.
GOT_VARIANT = "graph"


@dataclass(frozen=True)
class EvaluatedNode:
    """One node of a recorded ``execution.graph``."""

    node_id: str
    parent_ids: Tuple[str, ...]
    score: Optional[float]
    status: str
    action: str
    title: str = ""
    has_rationale: bool = False

    @property
    def scored(self) -> bool:
        return self.score is not None

    def is_low(self, threshold: float) -> bool:
        """``should_backtrack``'s own test: an unscored node is NOT low, it ends the walk."""
        return self.score is not None and self.score < threshold


@dataclass(frozen=True)
class EvaluatedRun:
    """One recorded trajectory's graph plus the eventual label."""

    source: str
    model: str
    variant: str
    score: float
    root_id: Optional[str]
    nodes: Dict[str, EvaluatedNode] = field(default_factory=dict)
    order: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> int:
        return 1 if self.score >= LABEL_THRESHOLD else 0

    @property
    def scored_nodes(self) -> List[EvaluatedNode]:
        """In graph-creation order (``IdeaDag.to_dict`` preserves insertion order)."""
        return [self.nodes[nid] for nid in self.order if self.nodes[nid].scored]

    @property
    def scores(self) -> List[float]:
        return [float(n.score) for n in self.scored_nodes]

    def path_to_root(self, node_id: str) -> List[EvaluatedNode]:
        """Replica of ``IdeaDag.path_to_root``: first parent only, cycle-guarded, node first."""
        path: List[EvaluatedNode] = []
        seen: set = set()
        current = self.nodes.get(node_id)
        while current is not None and current.node_id not in seen:
            path.append(current)
            seen.add(current.node_id)
            if not current.parent_ids:
                break
            current = self.nodes.get(current.parent_ids[0])
        return path

    def consecutive_low(self, node_id: str, threshold: float = BACKTRACK_LOW_SCORE) -> int:
        """Replica of ``should_backtrack``'s counter: walk up, stop at the first non-low node."""
        count = 0
        for node in self.path_to_root(node_id):
            if node.is_low(threshold):
                count += 1
            else:
                break
        return count

    def max_consecutive_low(self, threshold: float = BACKTRACK_LOW_SCORE) -> int:
        """The best case for the mechanism: the deepest low chain anywhere in this graph."""
        if not self.nodes:
            return 0
        return max(self.consecutive_low(nid, threshold) for nid in self.nodes)

    def would_backtrack(
        self, threshold: float = BACKTRACK_LOW_SCORE, dead_end: int = BACKTRACK_DEAD_END
    ) -> bool:
        return self.max_consecutive_low(threshold) >= dead_end

    def depth(self, node_id: str) -> int:
        """Length of the path ``should_backtrack`` gets to walk from this node (node included)."""
        return len(self.path_to_root(node_id))

    def deepest_scored(self) -> Optional[EvaluatedNode]:
        """The scored node with the longest path to root; ties broken by creation order."""
        scored = self.scored_nodes
        if not scored:
            return None
        return max(scored, key=lambda n: (self.depth(n.node_id), self.order.index(n.node_id)))


# --- run statistics -----------------------------------------------------------------------


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _path_running_min(run: EvaluatedRun) -> Optional[float]:
    """The statistic ``should_backtrack`` actually reads: the minimum score along the deepest
    node's path to root. Unscored ancestors are skipped rather than counted as 0."""
    node = run.deepest_scored()
    if node is None:
        return None
    along = [n.score for n in run.path_to_root(node.node_id) if n.score is not None]
    return min(along) if along else None


def _executed_mean(run: EvaluatedRun) -> Optional[float]:
    return _mean([n.score for n in run.scored_nodes if n.status == EXECUTED_STATUS])


#: Run-level aggregations of ``node.score``. ``None`` means "this run cannot supply the
#: statistic" and the run is dropped from that row only.
RUN_STATISTICS: Dict[str, Callable[[EvaluatedRun], Optional[float]]] = {
    "mean": lambda r: _mean(r.scores),
    "min": lambda r: min(r.scores) if r.scores else None,
    "max": lambda r: max(r.scores) if r.scores else None,
    "first": lambda r: r.scores[0] if r.scores else None,
    "last": lambda r: r.scores[-1] if r.scores else None,
    "path_running_min": _path_running_min,
    "executed_mean": _executed_mean,
}

#: LLM-free graph statistics, scored on the same task — the bar the judge has to clear to be
#: worth its calls. Descriptive, not proposed levers (same caveat as the confidence analysis).
FREE_BASELINES: Dict[str, Callable[[EvaluatedRun], Optional[float]]] = {
    "n_scored_nodes": lambda r: float(len(r.scores)),
    "n_nodes": lambda r: float(len(r.nodes)),
    "done_fraction": lambda r: (
        sum(1 for n in r.nodes.values() if n.status == EXECUTED_STATUS) / len(r.nodes)
        if r.nodes
        else None
    ),
    "failed_fraction": lambda r: (
        sum(1 for n in r.nodes.values() if n.status == "failed") / len(r.nodes)
        if r.nodes
        else None
    ),
}


# --- loading -------------------------------------------------------------------------------


def node_from_payload(node: Dict[str, Any]) -> Optional[EvaluatedNode]:
    node_id = node.get("node_id")
    if not node_id:
        return None
    parents = node.get("parent_ids")
    if not parents:
        parents = [] if node.get("parent_id") is None else [node["parent_id"]]
    details = node.get("details") or {}
    evaluation = details.get("evaluation")
    score = node.get("score")
    return EvaluatedNode(
        node_id=str(node_id),
        parent_ids=tuple(str(p) for p in parents),
        score=float(score) if isinstance(score, (int, float)) else None,
        status=str(node.get("status") or "unknown"),
        action=str(details.get("action") or "none"),
        title=str(node.get("title") or ""),
        has_rationale=bool(isinstance(evaluation, dict) and evaluation.get("rationale")),
    )


def run_from_result(payload: Dict[str, Any], source: str) -> Optional[EvaluatedRun]:
    """One result JSON -> one :class:`EvaluatedRun`, or ``None`` when it is not usable.

    Requires a recorded graph with at least one node and a numeric
    ``validation.overall_score`` (the same label rule the confidence analysis and the A6
    calibration driver use). Runs whose nodes are all unscored are kept on purpose — the
    availability section is about exactly those.
    """
    graph = ((payload.get("execution") or {}).get("graph")) or {}
    raw_nodes = graph.get("nodes") or {}
    iterable = raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes
    nodes: Dict[str, EvaluatedNode] = {}
    order: List[str] = []
    for entry in iterable:
        if not isinstance(entry, dict):
            continue
        node = node_from_payload(entry)
        if node is None:
            continue
        nodes[node.node_id] = node
        order.append(node.node_id)
    if not nodes:
        return None
    score = (payload.get("validation") or {}).get("overall_score")
    if not isinstance(score, (int, float)):
        return None
    return EvaluatedRun(
        source=source,
        model=str(payload.get("model") or "unknown"),
        variant=str(payload.get("execution_variant") or "unknown"),
        score=float(score),
        root_id=graph.get("root_id"),
        nodes=nodes,
        order=tuple(order),
    )


def load_runs(results_dir: Path, pattern: str = "**/*.json") -> List[EvaluatedRun]:
    """Every regular-roster result under ``results_dir`` that records a graph and a label.

    Recursive by default: unlike the confidence analysis' corpus, the runs that still carry a
    recorded graph live in per-run subdirectories (see the module docstring).
    """
    runs: List[EvaluatedRun] = []
    root = Path(results_dir)
    for path in sorted(glob.glob(str(root / pattern), recursive=True)):
        if not is_regular_roster(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:  # noqa: BLE001 — a truncated/aborted run file is just skipped
            continue
        if not isinstance(payload, dict):
            continue
        try:
            source = str(Path(path).relative_to(root))
        except ValueError:
            source = os.path.basename(path)
        run = run_from_result(payload, source)
        if run is not None:
            runs.append(run)
    return runs


# --- the individual analyses ---------------------------------------------------------------


def availability(runs: Sequence[EvaluatedRun]) -> Dict[str, Any]:
    """Step 0 in the artifact: is the score populated often enough to analyse at all?"""
    scored_runs = [r for r in runs if r.scores]
    by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in sorted({r.variant for r in runs}):
        subset = [r for r in runs if r.variant == variant]
        nodes = sum(len(r.nodes) for r in subset)
        scored = sum(len(r.scores) for r in subset)
        by_variant[variant] = {
            "runs": len(subset),
            "runs_with_scored_node": sum(1 for r in subset if r.scores),
            "nodes": nodes,
            "scored_nodes": scored,
            "scored_node_fraction": (scored / nodes) if nodes else 0.0,
        }
    by_action: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        for node in run.nodes.values():
            row = by_action.setdefault(node.action, {"nodes": 0, "scored_nodes": 0})
            row["nodes"] += 1
            row["scored_nodes"] += 1 if node.scored else 0
    for row in by_action.values():
        row["scored_node_fraction"] = row["scored_nodes"] / row["nodes"] if row["nodes"] else 0.0
    return {
        "runs_with_graph_and_label": len(runs),
        "runs_with_scored_node": len(scored_runs),
        "nodes": sum(len(r.nodes) for r in runs),
        "scored_nodes": sum(len(r.scores) for r in runs),
        "scored_nodes_per_run": summarize([float(len(r.scores)) for r in scored_runs]),
        "by_variant": dict(sorted(by_variant.items(), key=lambda kv: -kv[1]["scored_nodes"])),
        "by_action": dict(sorted(by_action.items(), key=lambda kv: -kv[1]["scored_nodes"])),
    }


def score_distribution(runs: Sequence[EvaluatedRun]) -> Dict[str, Any]:
    """The value vocabulary, and how much of it is a constant the engine or prompt dictated."""
    scores = [s for r in runs for s in r.scores]
    if not scores:
        return {"n": 0}
    scored_runs = [r for r in runs if r.scores]
    histogram: Dict[float, int] = {}
    for value in scores:
        key = round(value, 3)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "n": len(scores),
        "distinct_values": len(histogram),
        "histogram": dict(sorted(histogram.items())),
        "summary": summarize(scores),
        "max": max(scores),
        "min": min(scores),
        "above_cap_fraction": sum(1 for s in scores if s > PENALTY_SCORE_CAP) / len(scores),
        "at_cap_fraction": sum(1 for s in scores if s == PENALTY_SCORE_CAP) / len(scores),
        "at_base_score_fraction": sum(1 for s in scores if s == PENALTY_BASE_SCORE) / len(scores),
        "at_penalty_constant_fraction": (
            sum(1 for s in scores if s in (PENALTY_BASE_SCORE, PENALTY_SCORE_CAP)) / len(scores)
        ),
        "at_or_below_prompt_ceiling_fraction": (
            sum(1 for s in scores if s <= PROMPT_UNEXECUTED_CEILING) / len(scores)
        ),
        "below_backtrack_threshold_fraction": (
            sum(1 for s in scores if s < BACKTRACK_LOW_SCORE) / len(scores)
        ),
        "runs_with_constant_score": (
            sum(1 for r in scored_runs if len(set(r.scores)) == 1) / len(scored_runs)
        ),
        "within_run_spread": summarize(
            [max(r.scores) - min(r.scores) for r in scored_runs]
        ),
    }


def backtrack_reachability(runs: Sequence[EvaluatedRun]) -> Dict[str, Any]:
    """Replay ``should_backtrack``'s walk on every recorded node, then sweep both knobs.

    Separates "the signal is wrong" from "the walk never had anything to walk on": the
    counter can only reach N if some node sits N levels deep with every ancestor scored low.
    """
    depths: Dict[int, int] = {}
    scored_depths: Dict[int, int] = {}
    for run in runs:
        for node_id, node in run.nodes.items():
            depth = run.depth(node_id)
            depths[depth] = depths.get(depth, 0) + 1
            if node.scored:
                scored_depths[depth] = scored_depths.get(depth, 0) + 1
    max_low: Dict[int, int] = {}
    for run in runs:
        value = run.max_consecutive_low(BACKTRACK_LOW_SCORE)
        max_low[value] = max_low.get(value, 0) + 1
    sweep: Dict[str, Dict[str, Any]] = {}
    for low in LOW_SCORE_GRID:
        for dead_end in DEAD_END_GRID:
            firing = [r for r in runs if r.would_backtrack(low, dead_end)]
            row: Dict[str, Any] = {
                "runs_firing": len(firing),
                "run_fraction": (len(firing) / len(runs)) if runs else 0.0,
            }
            if firing:
                row["fire_pass_rate"] = statistics.mean([r.label for r in firing])
                rest = [r for r in runs if not r.would_backtrack(low, dead_end)]
                row["no_fire_pass_rate"] = statistics.mean([r.label for r in rest]) if rest else None
            sweep[f"low<{low}_dead_end>={dead_end}"] = row
    return {
        "shipped_rule": f"low<{BACKTRACK_LOW_SCORE} dead_end>={BACKTRACK_DEAD_END}",
        "max_path_length": max(depths) if depths else 0,
        "path_length_histogram": dict(sorted(depths.items())),
        "scored_path_length_histogram": dict(sorted(scored_depths.items())),
        "max_consecutive_low_histogram": dict(sorted(max_low.items())),
        "runs_firing_shipped_rule": sum(
            1 for r in runs if r.would_backtrack(BACKTRACK_LOW_SCORE, BACKTRACK_DEAD_END)
        ),
        "sweep": sweep,
    }


def run_level_auc(runs: Sequence[EvaluatedRun]) -> Dict[str, Any]:
    """AUC of each run-level statistic — one sample per trajectory, so intervals are honest.

    ``inverted_auc`` restates a sub-0.5 AUC as a *failure* predictor so the direction of any
    signal is readable at a glance (the confidence analysis uses the same convention).
    """
    report: Dict[str, Any] = {}
    for name, fn in list(RUN_STATISTICS.items()) + list(FREE_BASELINES.items()):
        values: List[float] = []
        labels: List[int] = []
        for run in runs:
            value = fn(run)
            if value is None:
                continue
            values.append(float(value))
            labels.append(run.label)
        row = auc_with_ci(values, labels)
        row["free"] = name in FREE_BASELINES
        row["inverted_auc"] = None if row["auc"] is None else 1.0 - row["auc"]
        report[name] = row
    return report


def node_level_auc(runs: Sequence[EvaluatedRun]) -> Dict[str, Any]:
    """Node-score vs the run's eventual label, pooled and split by status and action kind.

    Intervals here are optimistic (nodes in one run share a label); the run-level table is the
    one to argue from. The status split is this corpus' closest proxy for "was the node on the
    path the answer came from, or on a branch the selector dropped".
    """
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    by_status: Dict[str, List[Tuple[float, int]]] = {}
    by_action: Dict[str, List[Tuple[float, int]]] = {}
    for run in runs:
        for node in run.scored_nodes:
            value = float(node.score)
            pooled_scores.append(value)
            pooled_labels.append(run.label)
            by_status.setdefault(node.status, []).append((value, run.label))
            by_action.setdefault(node.action, []).append((value, run.label))

    def block(rows: Sequence[Tuple[float, int]]) -> Dict[str, Any]:
        values = [v for v, _ in rows]
        labels = [l for _, l in rows]
        row = auc_with_ci(values, labels)
        row["mean_score"] = statistics.mean(values) if values else None
        row["below_backtrack_threshold_fraction"] = (
            sum(1 for v in values if v < BACKTRACK_LOW_SCORE) / len(values) if values else None
        )
        return row

    executed = [v for v, _ in by_status.get(EXECUTED_STATUS, [])]
    dropped = [v for v, _ in by_status.get(DROPPED_STATUS, [])]
    selection_auc = roc_auc(
        executed + dropped, [1] * len(executed) + [0] * len(dropped)
    )
    return {
        "pooled": block(list(zip(pooled_scores, pooled_labels))),
        "by_status": {
            status: block(rows) for status, rows in sorted(by_status.items(), key=lambda kv: -len(kv[1]))
        },
        "by_action": {
            action: block(rows)
            for action, rows in sorted(by_action.items(), key=lambda kv: -len(kv[1]))
            if len(rows) >= 5
        },
        # Does the score even drive which sibling ran? (Selection reads it, so a value near
        # 0.5 means the score was not what decided — e.g. the parallel path skips evaluation.)
        "executed_vs_dropped_auc": selection_auc,
        "executed_mean": _mean(executed),
        "dropped_mean": _mean(dropped),
    }


def by_group(runs: Sequence[EvaluatedRun], key: Callable[[EvaluatedRun], str]) -> Dict[str, Any]:
    """Per-model / per-variant view of the headline statistics plus the free baseline."""
    report: Dict[str, Any] = {}
    for name in sorted({key(r) for r in runs}):
        subset = [r for r in runs if key(r) == name]
        scored = [r for r in subset if r.scores]
        if not scored:
            continue
        labels = [r.label for r in scored]
        report[name] = {
            "runs": len(scored),
            "base_rate": statistics.mean(labels),
            "scored_nodes": sum(len(r.scores) for r in scored),
            "mean_auc": auc_with_ci([statistics.mean(r.scores) for r in scored], labels),
            "min_auc": auc_with_ci([min(r.scores) for r in scored], labels),
            "n_scored_nodes_auc": auc_with_ci([float(len(r.scores)) for r in scored], labels),
            "max_consecutive_low": max(r.max_consecutive_low() for r in scored),
        }
    return report


def rationale_availability(runs: Sequence[EvaluatedRun]) -> Dict[str, Any]:
    """Is there any judge prose to quote? ``LlmBatchEvaluationPolicy`` records only the number.

    The confidence analysis could mine 1683 ``reason`` strings; this section reports whether
    the same is possible here at all before anyone tries.
    """
    scored = [n for r in runs for n in r.scored_nodes]
    return {
        "scored_nodes": len(scored),
        "with_rationale": sum(1 for n in scored if n.has_rationale),
        "with_rationale_fraction": (
            sum(1 for n in scored if n.has_rationale) / len(scored) if scored else 0.0
        ),
    }


def high_score_failures(
    runs: Sequence[EvaluatedRun], limit: int = 8
) -> List[Dict[str, Any]]:
    """The top-scored node of trajectories that eventually failed, one per source run."""
    picked: List[Dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: r.source):
        if run.label or not run.scores:
            continue
        best = max(run.scored_nodes, key=lambda n: float(n.score))
        picked.append(
            {
                "source": run.source,
                "model": run.model,
                "variant": run.variant,
                "overall_score": run.score,
                "node_score": best.score,
                "status": best.status,
                "action": best.action,
                "title": best.title,
                "has_rationale": best.has_rationale,
            }
        )
        if len(picked) >= limit:
            break
    return picked


def build_report(runs: Sequence[EvaluatedRun], samples: int = 8) -> Dict[str, Any]:
    scored_runs = [r for r in runs if r.scores]
    got_runs = [r for r in scored_runs if r.variant == GOT_VARIANT]
    return {
        "corpus": {
            "runs_scanned": len(runs),
            "runs_with_score": len(scored_runs),
            "got_variant_runs": len(got_runs),
            "scored_nodes": sum(len(r.scores) for r in scored_runs),
            "base_rate": statistics.mean([r.label for r in scored_runs]) if scored_runs else None,
            "got_base_rate": statistics.mean([r.label for r in got_runs]) if got_runs else None,
            "models": sorted({r.model for r in scored_runs}),
            "label_rule": f"validation.overall_score >= {LABEL_THRESHOLD}",
        },
        "availability": availability(runs),
        "score_distribution": score_distribution(scored_runs),
        "score_distribution_got": score_distribution(got_runs),
        "backtrack_reachability": backtrack_reachability(scored_runs),
        "backtrack_reachability_got": backtrack_reachability(got_runs),
        "run_level_auc": run_level_auc(scored_runs),
        "run_level_auc_got": run_level_auc(got_runs),
        "node_level_auc": node_level_auc(scored_runs),
        "node_level_auc_got": node_level_auc(got_runs),
        "by_model": by_group(scored_runs, lambda r: r.model),
        "by_model_got": by_group(got_runs, lambda r: r.model),
        "by_variant": by_group(scored_runs, lambda r: r.variant),
        "rationale_availability": rationale_availability(scored_runs),
        "high_score_failures": high_score_failures(scored_runs, limit=samples),
    }


# --- rendering ------------------------------------------------------------------------------


def _auc(row: Optional[Dict[str, Any]]) -> str:
    if not row or row.get("auc") is None:
        return "  --  "
    ci = row.get("ci95")
    tail = f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
    return f"{row['auc']:.3f}{tail}"


def render(report: Dict[str, Any]) -> None:
    corpus = report["corpus"]
    print(
        f"corpus: {corpus['runs_with_score']}/{corpus['runs_scanned']} roster runs carry a node "
        f"score ({corpus['scored_nodes']} scored nodes), base rate "
        f"{corpus['base_rate']:.3f} ({corpus['label_rule']})"
    )
    print(
        f"  of those, {corpus['got_variant_runs']} are execution_variant={GOT_VARIANT} "
        f"(where should_backtrack lives), base rate "
        f"{(corpus['got_base_rate'] if corpus['got_base_rate'] is not None else float('nan')):.3f}"
    )
    print(f"models: {', '.join(corpus['models'])}")

    avail = report["availability"]
    print("\n== availability (does evaluation run at all?) ==")
    print(
        f"  {avail['scored_nodes']}/{avail['nodes']} recorded nodes carry a numeric score; "
        f"{avail['scored_nodes_per_run']['n']} runs supply one "
        f"(mean {avail['scored_nodes_per_run']['mean']:.2f} scored nodes/run)"
    )
    for variant, row in avail["by_variant"].items():
        print(
            f"  variant {variant:18s} runs={row['runs']:4d} with-score={row['runs_with_scored_node']:4d} "
            f"nodes={row['nodes']:5d} scored={row['scored_nodes']:5d} ({row['scored_node_fraction']:.3f})"
        )
    for action, row in avail["by_action"].items():
        print(
            f"  action  {action:18s} nodes={row['nodes']:5d} scored={row['scored_nodes']:5d} "
            f"({row['scored_node_fraction']:.3f})"
        )

    for name, key in (("all variants", "score_distribution"), (f"{GOT_VARIANT} only", "score_distribution_got")):
        dist = report[key]
        if not dist.get("n"):
            continue
        print(f"\n== score distribution ({name}) ==")
        print(
            f"  n={dist['n']} distinct={dist['distinct_values']} range=[{dist['min']:.2f}, "
            f"{dist['max']:.2f}] mean={dist['summary']['mean']:.3f} median={dist['summary']['median']:.3f}"
        )
        print(
            f"  above the {PENALTY_SCORE_CAP} penalty cap: {dist['above_cap_fraction']:.3f} | "
            f"exactly on a penalty constant ({PENALTY_BASE_SCORE}/{PENALTY_SCORE_CAP}): "
            f"{dist['at_penalty_constant_fraction']:.3f} | <= the prompt's "
            f"{PROMPT_UNEXECUTED_CEILING} ceiling: {dist['at_or_below_prompt_ceiling_fraction']:.3f}"
        )
        print(
            f"  below the {BACKTRACK_LOW_SCORE} backtrack threshold: "
            f"{dist['below_backtrack_threshold_fraction']:.3f} | runs whose scores are all "
            f"identical: {dist['runs_with_constant_score']:.3f} | within-run spread mean "
            f"{dist['within_run_spread']['mean']:.3f}"
        )
        print(f"  histogram: {dist['histogram']}")

    for name, key in (("all variants", "backtrack_reachability"), (f"{GOT_VARIANT} only", "backtrack_reachability_got")):
        reach = report[key]
        print(f"\n== backtrack reachability ({name}); shipped rule {reach['shipped_rule']} ==")
        print(
            f"  longest path_to_root anywhere: {reach['max_path_length']} nodes "
            f"(histogram {reach['path_length_histogram']}); scored nodes sit at "
            f"{reach['scored_path_length_histogram']}"
        )
        print(
            f"  per-run best consecutive-low chain: {reach['max_consecutive_low_histogram']} -> "
            f"{reach['runs_firing_shipped_rule']} runs would ever fire the shipped rule"
        )
        for label, row in reach["sweep"].items():
            if not row["runs_firing"]:
                continue
            print(
                f"    {label:26s} fires on {row['runs_firing']:4d} runs "
                f"({row['run_fraction']:.3f}); pass rate fired={row['fire_pass_rate']:.3f} "
                f"vs not-fired="
                f"{(row['no_fire_pass_rate'] if row['no_fire_pass_rate'] is not None else float('nan')):.3f}"
            )

    for name, key in (("all variants", "run_level_auc"), (f"{GOT_VARIANT} only", "run_level_auc_got")):
        print(f"\n== run-level AUC ({name}; one sample per trajectory) ==")
        for statistic, row in report[key].items():
            tag = "free" if row["free"] else "judge"
            inverted = row.get("inverted_auc")
            tail = "" if inverted is None else f"  (as a FAILURE predictor: {inverted:.3f})"
            print(f"  {statistic:22s} {tag:5s} n={row['n']:4d} AUC={_auc(row)}{tail}")

    for name, key in (("all variants", "node_level_auc"), (f"{GOT_VARIANT} only", "node_level_auc_got")):
        block = report[key]
        print(f"\n== node-level AUC ({name}; intervals optimistic, nodes share a run label) ==")
        pooled = block["pooled"]
        print(f"  pooled  n={pooled['n']:5d} mean={pooled['mean_score']:.3f} AUC={_auc(pooled)}")
        for status, row in block["by_status"].items():
            print(
                f"  status {status:10s} n={row['n']:5d} mean={row['mean_score']:.3f} "
                f"below-threshold={row['below_backtrack_threshold_fraction']:.3f} AUC={_auc(row)}"
            )
        for action, row in block["by_action"].items():
            print(
                f"  action {action:10s} n={row['n']:5d} mean={row['mean_score']:.3f} "
                f"below-threshold={row['below_backtrack_threshold_fraction']:.3f} AUC={_auc(row)}"
            )
        sel = block["executed_vs_dropped_auc"]
        print(
            f"  score as a predictor of which sibling actually RAN: "
            f"{('  --  ' if sel is None else format(sel, '.3f'))} "
            f"(executed mean {(block['executed_mean'] or float('nan')):.3f} vs dropped "
            f"{(block['dropped_mean'] or float('nan')):.3f})"
        )

    for name, key in (("all variants", "by_model"), (f"{GOT_VARIANT} only", "by_model_got")):
        print(f"\n== by model ({name}) ==")
        for model, row in report[key].items():
            print(
                f"  {model:30s} runs={row['runs']:4d} base={row['base_rate']:.3f} "
                f"mean={_auc(row['mean_auc'])} min={_auc(row['min_auc'])} "
                f"free-n={_auc(row['n_scored_nodes_auc'])} max-low-chain={row['max_consecutive_low']}"
            )

    print("\n== by execution variant ==")
    for variant, row in report["by_variant"].items():
        print(
            f"  {variant:30s} runs={row['runs']:4d} base={row['base_rate']:.3f} "
            f"mean={_auc(row['mean_auc'])} min={_auc(row['min_auc'])} "
            f"free-n={_auc(row['n_scored_nodes_auc'])} max-low-chain={row['max_consecutive_low']}"
        )

    rat = report["rationale_availability"]
    print("\n== is there any judge prose to mine? ==")
    print(
        f"  {rat['with_rationale']}/{rat['scored_nodes']} scored nodes recorded a rationale "
        f"({rat['with_rationale_fraction']:.3f})"
    )

    print("\n== the top-scored node inside eventually-FAILED trajectories ==")
    for case in report["high_score_failures"]:
        print(
            f"  --- {case['source']} run={case['overall_score']:.2f} node_score={case['node_score']:.2f} "
            f"status={case['status']} action={case['action']}"
        )
        print(f"      {case['title'][:140]}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--pattern",
        default="**/*.json",
        help="glob under --results-dir; recursive because the runs that still record a graph "
        "live in per-run subdirectories",
    )
    parser.add_argument("--samples", type=int, default=8, help="failed-run nodes to print")
    parser.add_argument("--json-out", default=None, help="also write the full report as JSON")
    args = parser.parse_args(argv)

    runs = load_runs(Path(args.results_dir), args.pattern)
    if not any(r.scores for r in runs):
        print(
            f"no regular-roster run under {args.results_dir} records a scored graph node",
            file=sys.stderr,
        )
        return 1
    report = build_report(runs, samples=args.samples)
    render(report)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=False, default=str)
            handle.write("\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
