#!/usr/bin/env python3
"""Does the capped-score mass distort ``compute_dynamic_beam_width``? $0, offline.

``GoTOperations.compute_dynamic_beam_width`` (``got_operations.py``) is default-ON
(``got_dynamic_beam_enabled`` and ``got_adaptive_policies`` both default true), so its
adaptive branch runs on every live run today. It pools ``node.score`` for **every** scored
non-root node anywhere in the graph so far, takes a p25/p75 spread, and maps that spread onto
``[beam_min, beam_max]``: a wide spread means "uncertain, widen the beam", a narrow one means
"converged, narrow it".

ASSUMPTION_AUDIT.md T1-3a establishes that those scores are largely not judge opinions.
Every candidate the judge sees is pending (``_expand_or_execute`` filters executed children
out), so ``evaluate_batch`` clips each score at ``evaluation_no_action_result_score_cap``
(0.5) or, for a candidate the judge omitted, substitutes ``no_action_result_base_score``
(0.4). 43.3% of recorded candidates sit exactly on the cap. A pool that clusters on a
constant has a small measured spread **by construction**, which the beam formula cannot
distinguish from genuine model convergence.

Commit 9b710b91 started recording, next to each node's ``score``, the judge's pre-cap opinion
as ``raw_score`` (``None`` when the judge was never called at all) and a ``capped`` flag. That
makes the counterfactual answerable offline: recompute the same beam formula over
``raw_score`` and see whether the number the engine consumes actually changes.

Corpus: result files under ``agent/idea_test_results`` that contain ``raw_score`` at all --
i.e. runs recorded after 9b710b91. Earlier runs are silently excluded, since for them the
counterfactual pool is identical to the actual one and would only dilute the rates.

Two pools are reported, because the recorded graph is END-OF-RUN state:

* **final** -- every scored non-root node in the finished graph. This is exactly what the
  last ``compute_dynamic_beam_width`` call of the run saw, so it is measured, not inferred.
* **prefix** -- the beam recomputed at each expansion point, approximating the pool by the
  depth-first prefix of the graph before that parent's children. Node creation order is not
  recorded, so this is an approximation and is labelled as one; it exists to show whether the
  contamination is present early (when the beam still has something to decide) or only at the
  end.

The substitution rule under test matches the proposed
``got_beam_spread_uses_raw_score_enabled``: use ``raw_score`` when it is a number, else fall
back to ``node.score`` (the base-score shortcut records ``raw_score: None`` and must degrade
gracefully). Under shipped settings every ``evaluation_weight_*`` is 1.0, so ``score`` is
``min(raw_score, cap)`` and the two are directly comparable; a run with non-unit weights would
need the weight reapplied and is reported separately as a caveat count.

Analysis only: reads result JSON and prints. Writes nothing the engine reads.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/analyze_beam_spread_contamination.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_DEFAULT = REPO_ROOT / "agent" / "idea_test_results"
SETTINGS_PATH = REPO_ROOT / "agent" / "app" / "idea_dag_settings.json"

#: ``GotConfig`` defaults, used when the settings file cannot be read.
FALLBACK_BEAM_MIN = 2
FALLBACK_BEAM_MAX = 5
FALLBACK_TARGET_SPREAD = 0.4


@dataclass(frozen=True)
class BeamParams:
    beam_min: int = FALLBACK_BEAM_MIN
    beam_max: int = FALLBACK_BEAM_MAX
    target_spread: float = FALLBACK_TARGET_SPREAD

    @classmethod
    def from_settings_file(cls, path: Path = SETTINGS_PATH) -> "BeamParams":
        try:
            settings = json.loads(path.read_text())
        except Exception:  # noqa: BLE001. Fall back to the dataclass defaults
            return cls()
        return cls(
            beam_min=int(settings.get("got_beam_min", FALLBACK_BEAM_MIN)),
            beam_max=int(settings.get("got_beam_max", FALLBACK_BEAM_MAX)),
            # Absent from the settings file on purpose (see config.py); the typed default rules.
            target_spread=float(settings.get("got_beam_target_spread", FALLBACK_TARGET_SPREAD)),
        )


def beam_from_scores(scores: Sequence[float], params: BeamParams) -> Tuple[int, float]:
    """The adaptive branch of ``compute_dynamic_beam_width``, verbatim, plus the spread.

    Kept as a transcription rather than an import so the analysis states the formula it is
    measuring; ``beam_width_raw_score_test.py`` pins the two together.
    """
    if not scores:
        return params.beam_max, float("nan")
    if len(scores) < 4:
        return -1, float("nan")  # legacy avg branch; excluded from the comparison
    ordered = sorted(scores)
    p25 = ordered[max(0, int(len(ordered) * 0.25) - 1)]
    p75 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))]
    spread = max(0.0, p75 - p25)
    ratio = min(1.0, spread / params.target_spread) if params.target_spread > 0 else 0.0
    beam = params.beam_min + int(round(ratio * (params.beam_max - params.beam_min)))
    return max(params.beam_min, min(params.beam_max, beam)), spread


@dataclass(frozen=True)
class ScoredNode:
    node_id: str
    score: float
    raw_score: Optional[float]
    capped: bool

    @property
    def counterfactual(self) -> float:
        """What the proposed flag would feed the spread: raw opinion, else the recorded score."""
        return self.score if self.raw_score is None else float(self.raw_score)

    @property
    def is_placeholder(self) -> bool:
        """Score the engine wrote over (or instead of) a judge opinion."""
        return self.capped or self.raw_score is None


def dfs_order(nodes: Dict[str, Any], root_id: Optional[str]) -> List[str]:
    """``IdeaDag.iter_depth_first`` order over the recorded node dict."""
    if not root_id or root_id not in nodes:
        return list(nodes)
    out: List[str] = []
    seen: set[str] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in nodes:
            continue
        seen.add(node_id)
        out.append(node_id)
        children = nodes[node_id].get("children") or []
        stack.extend(reversed(list(children)))
    out.extend(n for n in nodes if n not in seen)
    return out


def scored_nodes_from_result(payload: Dict[str, Any]) -> Tuple[List[ScoredNode], List[ScoredNode]]:
    """``(nodes in DFS order, nodes with an evaluation record)``.

    The first list is every node the beam pool would draw on; the second is the subset the
    counterfactual can actually speak for.
    """
    graph = ((payload.get("execution") or {}).get("graph") or {})
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return [], []
    pooled: List[ScoredNode] = []
    for node_id in dfs_order(nodes, graph.get("root_id")):
        node = nodes.get(node_id)
        if not isinstance(node, dict):
            continue
        score = node.get("score")
        # The exact guard in compute_dynamic_beam_width.
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if not node.get("parent_id"):
            continue
        evaluation = (node.get("details") or {}).get("evaluation") or {}
        raw = evaluation.get("raw_score")
        pooled.append(
            ScoredNode(
                node_id=node_id,
                score=float(score),
                raw_score=float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None,
                capped=bool(evaluation.get("capped")),
            )
        )
    with_record = [n for n in pooled if n.raw_score is not None or n.capped]
    return pooled, with_record


def load_runs(results_dir: Path, pattern: str) -> List[Tuple[str, List[ScoredNode]]]:
    runs: List[Tuple[str, List[ScoredNode]]] = []
    for path in sorted(results_dir.glob(pattern)):
        try:
            text = path.read_text()
        except Exception:  # noqa: BLE001
            continue
        if '"raw_score"' not in text:  # pre-9b710b91 run: no counterfactual to compute
            continue
        try:
            payload = json.loads(text)
        except Exception:  # noqa: BLE001
            continue
        pooled, _ = scored_nodes_from_result(payload)
        if pooled:
            runs.append((str(path), pooled))
    return runs


def analyse(runs: Sequence[Tuple[str, List[ScoredNode]]], params: BeamParams) -> Dict[str, Any]:
    all_nodes = [n for _, pool in runs for n in pool]
    final_rows: List[Dict[str, Any]] = []
    prefix_rows: List[Dict[str, Any]] = []
    for source, pool in runs:
        actual, spread_a = beam_from_scores([n.score for n in pool], params)
        counter, spread_c = beam_from_scores([n.counterfactual for n in pool], params)
        if actual >= 0:
            final_rows.append(
                {
                    "source": source,
                    "n": len(pool),
                    "actual": actual,
                    "counterfactual": counter,
                    "spread_actual": spread_a,
                    "spread_counterfactual": spread_c,
                    "placeholder_frac": sum(1 for n in pool if n.is_placeholder) / len(pool),
                }
            )
        # Prefix pools: every point at which a beam width would have been computed.
        for cut in range(1, len(pool)):
            window = pool[:cut]
            a, sa = beam_from_scores([n.score for n in window], params)
            c, sc = beam_from_scores([n.counterfactual for n in window], params)
            if a < 0:
                continue
            prefix_rows.append(
                {
                    "n": cut,
                    "actual": a,
                    "counterfactual": c,
                    "spread_actual": sa,
                    "spread_counterfactual": sc,
                    "placeholder_frac": sum(1 for n in window if n.is_placeholder) / cut,
                }
            )

    def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0}
        changed = [r for r in rows if r["actual"] != r["counterfactual"]]
        return {
            "n": len(rows),
            "mean_placeholder_frac": sum(r["placeholder_frac"] for r in rows) / len(rows),
            "mean_spread_actual": sum(r["spread_actual"] for r in rows) / len(rows),
            "mean_spread_counterfactual": sum(r["spread_counterfactual"] for r in rows) / len(rows),
            "mean_beam_actual": sum(r["actual"] for r in rows) / len(rows),
            "mean_beam_counterfactual": sum(r["counterfactual"] for r in rows) / len(rows),
            "beam_actual_hist": dict(sorted(Counter(r["actual"] for r in rows).items())),
            "beam_counterfactual_hist": dict(sorted(Counter(r["counterfactual"] for r in rows).items())),
            "changed": len(changed),
            "changed_frac": len(changed) / len(rows),
            "widened": sum(1 for r in changed if r["counterfactual"] > r["actual"]),
            "narrowed": sum(1 for r in changed if r["counterfactual"] < r["actual"]),
            "at_beam_min_actual": sum(1 for r in rows if r["actual"] == params.beam_min) / len(rows),
            "at_beam_min_counterfactual": sum(
                1 for r in rows if r["counterfactual"] == params.beam_min
            ) / len(rows),
        }

    return {
        "params": vars(params),
        "runs": len(runs),
        "pool_nodes": len(all_nodes),
        "census": {
            "capped": sum(1 for n in all_nodes if n.capped),
            "raw_missing": sum(1 for n in all_nodes if n.raw_score is None),
            "placeholder": sum(1 for n in all_nodes if n.is_placeholder),
            "raw_equals_score": sum(1 for n in all_nodes if n.raw_score == n.score),
            "raw_above_score": sum(
                1 for n in all_nodes if n.raw_score is not None and n.raw_score > n.score
            ),
        },
        "final": summarise(final_rows),
        "prefix": summarise(prefix_rows),
    }


def render(report: Dict[str, Any]) -> None:
    print("=== corpus (post-9b710b91 runs only) ===")
    print(f"{report['runs']} runs / {report['pool_nodes']} pooled scored nodes")
    census = report["census"]
    total = max(1, report["pool_nodes"])
    print(
        f"  capped by the engine: {census['capped']} ({census['capped'] / total:.1%})\n"
        f"  no judge opinion at all (raw_score None): {census['raw_missing']} "
        f"({census['raw_missing'] / total:.1%})\n"
        f"  placeholder mass (either): {census['placeholder']} ({census['placeholder'] / total:.1%})\n"
        f"  raw_score > recorded score: {census['raw_above_score']} "
        f"({census['raw_above_score'] / total:.1%})"
    )
    for label in ("final", "prefix"):
        row = report[label]
        if not row.get("n"):
            print(f"\n=== {label}: no adaptive-branch pools ===")
            continue
        note = "measured (end-of-run pool)" if label == "final" else "approximated (DFS prefix)"
        print(f"\n=== {label} pools -- {note} ===")
        print(f"  pools: {row['n']}  mean placeholder fraction {row['mean_placeholder_frac']:.1%}")
        print(
            f"  spread   actual {row['mean_spread_actual']:.4f} -> "
            f"raw_score {row['mean_spread_counterfactual']:.4f}"
        )
        print(
            f"  beam     actual {row['mean_beam_actual']:.3f} -> "
            f"raw_score {row['mean_beam_counterfactual']:.3f}"
        )
        print(f"  beam histogram actual        {row['beam_actual_hist']}")
        print(f"  beam histogram counterfactual {row['beam_counterfactual_hist']}")
        print(
            f"  beam CHANGES in {row['changed']} / {row['n']} pools ({row['changed_frac']:.1%}); "
            f"{row['widened']} wider, {row['narrowed']} narrower"
        )
        print(
            f"  pinned at beam_min: {row['at_beam_min_actual']:.1%} actual -> "
            f"{row['at_beam_min_counterfactual']:.1%} with raw_score"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--pattern", default="**/*.json")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    params = BeamParams.from_settings_file()
    runs = load_runs(args.results_dir, args.pattern)
    if not runs:
        print(f"no post-9b710b91 scored runs under {args.results_dir}", file=sys.stderr)
        return 1
    report = analyse(runs, params)
    render(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
