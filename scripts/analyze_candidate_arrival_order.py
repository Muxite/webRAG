#!/usr/bin/env python3
"""Does LLM candidate output ORDER carry quality signal? $0, offline.

ASSUMPTION_AUDIT.md T1-3 records that the engine's "beam" truncates in ARRIVAL order:
``idea_engine`` does ``candidates[:max_branching]`` before any score exists, so whichever
candidates the model happened to emit first are the ones that survive. The embedded
assumption -- never tested -- is that **LLM output order correlates with candidate
quality**. Experiment E2 was scoped as a live shuffle A/B; this script answers the same
question for $0 from recorded runs, which is the cheaper discriminating test and the one
that should run before any live spend on ``engine.beam_after_evaluation``.

Unit of analysis: a **sibling batch** -- one parent's ``children`` list, which
``IdeaDag.expand`` builds in candidate order, so the child's index IS its arrival position.
Each child's ``score`` is what ``evaluation.evaluate_batch`` later wrote. Both survive into
``execution.graph`` in the result JSON, so arrival position and eventual score are both
recoverable without re-running anything.

Two confounds the raw correlation must be cleared of, because both are properties of the
scorer rather than of the ordering:

1. **Action composition.** Position 0 is disproportionately ``search`` and position 1
   disproportionately ``visit`` (planners open with a search), and the judge does not score
   the two kinds alike. A raw position-vs-score correlation partly measures that.
2. **Execution state.** ``evaluate_batch`` caps a candidate at
   ``evaluation_no_action_result_score_cap`` when it has an action but no ``action_result``
   yet, and the rate of "already executed" falls with arrival position. That alone would
   manufacture an earlier-is-better gradient.

A third caveat cannot be cleared, only bounded: a re-expanded parent
(``_got_reexpand_count``) appends a SECOND batch of children to the same list, so positions
past the first batch's width belong to a different LLM call. The recorded corpus puts under
2% of batches above the ``max_branching`` width, so this cannot move the headline, but it is
why the tail positions are reported with their own counts rather than folded in.

So the headline number is the correlation computed on scores **residualised within
(action, executed) cells**, alongside the raw one. Reported next to it is the statistic the
truncation decision actually consumes: how often the arrival-first candidate is the unique
top-scorer, against the 1/k chance baseline.

Significance is a within-batch permutation test (shuffle arrival positions, keep scores),
which is exact under the null "position is exchangeable" and needs no distributional
assumption about a mean of per-batch rank correlations.

Analysis only: reads result JSON and prints. Writes nothing the engine reads.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/analyze_candidate_arrival_order.py
    ./.venv/bin/python scripts/analyze_candidate_arrival_order.py --json-out /tmp/e2.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR_DEFAULT = REPO_ROOT / "agent" / "idea_test_results"

#: Permutation seed. Fixed so a re-run of the same corpus reprints the same p-values.
DEFAULT_SEED = 20260820
DEFAULT_PERMUTATIONS = 2000


# --- pure statistics -----------------------------------------------------------------------


def average_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks with ties averaged.

    Tie handling is not cosmetic here: a majority of recorded sibling batches score every
    child identically, and integer ranking would invent an ordering the judge never
    expressed.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho, or ``None`` when either side is constant (rho undefined, not zero)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    """``(mean, lo, hi)`` for a normal-approximation 95% interval on the mean."""
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    m = sum(values) / n
    if n < 2:
        return m, m, m
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    se = sd / math.sqrt(n)
    return m, m - 1.96 * se, m + 1.96 * se


def first_is_unique_top(scores: Sequence[float]) -> Optional[bool]:
    """Is the arrival-first candidate the single highest-scored one?

    ``None`` when the batch does not discriminate (no unique top), which is the case the
    caller must exclude rather than count as a miss: with every score equal, keeping the
    first costs nothing and a score-ordered beam would gain nothing.
    """
    if len(scores) < 2:
        return None
    top = max(scores)
    if sum(1 for s in scores if s == top) != 1:
        return None
    return scores[0] == top


# --- corpus --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    position: int
    score: float
    action: Optional[str]
    executed: bool


@dataclass(frozen=True)
class Batch:
    """One parent's scored children, in arrival order."""

    source: str
    parent_id: str
    model: Optional[str]
    candidates: Tuple[Candidate, ...]

    @property
    def scores(self) -> List[float]:
        return [c.score for c in self.candidates]

    @property
    def discriminates(self) -> bool:
        return len(set(self.scores)) > 1


def batches_from_result(payload: Dict[str, Any], source: str) -> List[Batch]:
    """Every scored sibling batch in one result JSON.

    Reads ``execution.graph`` -- the structure ``IdeaDag.to_dict`` writes -- rather than the
    duplicate under ``execution.output.graph``, and keeps only children carrying a numeric
    ``score``, since an unscored child contributes nothing to a position-vs-score question.
    """
    if not isinstance(payload, dict):
        return []
    nodes = ((payload.get("execution") or {}).get("graph") or {}).get("nodes")
    if not isinstance(nodes, dict):
        return []
    model = payload.get("model")
    out: List[Batch] = []
    for parent_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        found: List[Candidate] = []
        for position, child_id in enumerate(node.get("children") or []):
            child = nodes.get(child_id)
            if not isinstance(child, dict):
                continue
            score = child.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            details = child.get("details") or {}
            found.append(
                Candidate(
                    position=position,
                    score=float(score),
                    action=details.get("action"),
                    executed=details.get("action_result") is not None,
                )
            )
        if len(found) >= 2:
            out.append(Batch(source, parent_id, model, tuple(found)))
    return out


def load_batches(results_dir: Path, pattern: str = "**/*.json") -> List[Batch]:
    out: List[Batch] = []
    for path in sorted(results_dir.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001. A corrupt or non-result JSON is simply skipped
            continue
        out.extend(batches_from_result(payload, str(path)))
    return out


# --- analysis ------------------------------------------------------------------------------


def residualise(batches: Sequence[Batch]) -> Dict[Tuple[Optional[str], bool], float]:
    """Corpus-wide mean score per ``(action, executed)`` cell.

    Subtracting this is what separates "the model puts its best idea first" from "the model
    puts its *search* first and the judge likes searches".
    """
    cells: Dict[Tuple[Optional[str], bool], List[float]] = defaultdict(list)
    for batch in batches:
        for cand in batch.candidates:
            cells[(cand.action, cand.executed)].append(cand.score)
    return {key: sum(vals) / len(vals) for key, vals in cells.items()}


def order_correlation(
    batches: Sequence[Batch],
    values: Sequence[Sequence[float]],
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Mean per-batch Spearman(position, value), with a within-batch permutation p."""
    pairs = [
        ([c.position for c in b.candidates], list(v))
        for b, v in zip(batches, values)
        if len(set(v)) > 1
    ]
    rhos = [r for r in (spearman(x, y) for x, y in pairs) if r is not None]
    if not rhos:
        return {"batches": 0, "rho": None}
    mean, lo, hi = mean_ci(rhos)
    rng = random.Random(seed)
    at_least = 0
    for _ in range(permutations):
        total, count = 0.0, 0
        for xs, ys in pairs:
            shuffled = list(xs)
            rng.shuffle(shuffled)
            r = spearman(shuffled, ys)
            if r is not None:
                total += r
                count += 1
        if count and abs(total / count) >= abs(mean):
            at_least += 1
    return {
        "batches": len(rhos),
        "rho": mean,
        "ci": [lo, hi],
        "p": (at_least + 1) / (permutations + 1),
    }


def first_wins(batches: Sequence[Batch], values: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """How often arrival-first is the unique top-scorer, versus the 1/k chance baseline.

    This is the decision-relevant statistic: it is exactly the question
    ``candidates[:max_branching]`` answers when it keeps the head of the list.

    Batches with no unique top are dropped rather than counted as misses -- with the maximum
    shared, keeping the first costs nothing and a score-ordered beam would gain nothing, so
    they belong in the "cannot discriminate" census above and not in this rate. Conditioning
    on a unique top also makes 1/k the exact null: under a random permutation of arrival
    positions, exactly one position holds the maximum.
    """
    usable = [(b, list(v)) for b, v in zip(batches, values) if first_is_unique_top(list(v)) is not None]
    if not usable:
        return {"batches": 0}
    hits = sum(1 for _, v in usable if v[0] == max(v))
    chance = sum(1.0 / len(v) for _, v in usable) / len(usable)
    rate = hits / len(usable)
    # Binomial-ish normal approximation against the (heterogeneous-k) chance baseline.
    var = sum((1.0 / len(v)) * (1 - 1.0 / len(v)) for _, v in usable)
    z = (hits - sum(1.0 / len(v) for _, v in usable)) / math.sqrt(var) if var > 0 else 0.0
    return {
        "batches": len(usable),
        "rate": rate,
        "chance": chance,
        "edge": rate - chance,
        "z": z,
    }


def build_report(
    batches: Sequence[Batch], permutations: int = DEFAULT_PERMUTATIONS, seed: int = DEFAULT_SEED
) -> Dict[str, Any]:
    cells = residualise(batches)
    raw = [b.scores for b in batches]
    resid = [
        [c.score - cells[(c.action, c.executed)] for c in b.candidates] for b in batches
    ]
    discriminating = [b for b in batches if b.discriminates]
    by_position: Dict[int, List[float]] = defaultdict(list)
    by_position_resid: Dict[int, List[float]] = defaultdict(list)
    executed_by_position: Dict[int, List[int]] = defaultdict(list)
    action_by_position: Dict[int, Counter] = defaultdict(Counter)
    by_action: Dict[Optional[str], List[float]] = defaultdict(list)
    for batch, res in zip(batches, resid):
        for cand, r in zip(batch.candidates, res):
            by_position[cand.position].append(cand.score)
            by_position_resid[cand.position].append(r)
            executed_by_position[cand.position].append(1 if cand.executed else 0)
            action_by_position[cand.position][cand.action] += 1
            by_action[cand.action].append(cand.score)
    return {
        "corpus": {
            "batches": len(batches),
            "candidates": sum(len(b.candidates) for b in batches),
            "files": len({b.source for b in batches}),
            "discriminating": len(discriminating),
            "flat": len(batches) - len(discriminating),
            "batch_sizes": dict(sorted(Counter(len(b.candidates) for b in batches).items())),
        },
        "raw": order_correlation(batches, raw, permutations, seed),
        "residualised": order_correlation(batches, resid, permutations, seed),
        "first_wins": first_wins(batches, raw),
        "first_wins_residualised": first_wins(batches, resid),
        "by_position": {
            p: {
                "n": len(by_position[p]),
                "mean": sum(by_position[p]) / len(by_position[p]),
                "resid": sum(by_position_resid[p]) / len(by_position_resid[p]),
                "executed_rate": sum(executed_by_position[p]) / len(executed_by_position[p]),
                "actions": dict(action_by_position[p].most_common(3)),
            }
            for p in sorted(by_position)
        },
        "by_action": {
            str(a): {"n": len(v), "mean": sum(v) / len(v)}
            for a, v in sorted(by_action.items(), key=lambda kv: -len(kv[1]))
        },
    }


def render(report: Dict[str, Any]) -> None:
    corpus = report["corpus"]
    print("=== corpus ===")
    print(
        f"{corpus['batches']} sibling batches / {corpus['candidates']} scored candidates "
        f"across {corpus['files']} result files"
    )
    flat_pct = corpus["flat"] / corpus["batches"] if corpus["batches"] else 0.0
    print(
        f"batches where every child scored the SAME: {corpus['flat']} ({flat_pct:.1%}) "
        f"-- no beam, score-ordered or not, can distinguish these"
    )
    print(f"batch sizes: {corpus['batch_sizes']}")

    def line(label: str, row: Dict[str, Any]) -> None:
        if not row.get("batches"):
            print(f"{label:34s} (no discriminating batches)")
            return
        lo, hi = row["ci"]
        print(
            f"{label:34s} rho={row['rho']:+.4f} [{lo:+.4f},{hi:+.4f}] "
            f"n={row['batches']:4d} perm p={row['p']:.4f}"
        )

    print("\n=== Spearman(arrival position, score) ===")
    line("raw score", report["raw"])
    line("residualised (action x executed)", report["residualised"])

    print("\n=== the statistic truncation consumes ===")
    for label, key in (("raw score", "first_wins"), ("residualised", "first_wins_residualised")):
        fw = report[key]
        if not fw.get("batches"):
            continue
        print(
            f"{label:14s} arrival-first is the UNIQUE top in {fw['rate']:.3f} of "
            f"{fw['batches']:4d} batches; chance {fw['chance']:.3f}; "
            f"edge {fw['edge']:+.3f} (z={fw['z']:+.2f})"
        )

    print("\n=== by arrival position ===")
    for pos, row in report["by_position"].items():
        if row["n"] < 10:
            continue
        print(
            f"  pos {pos}: n={row['n']:4d} mean={row['mean']:.4f} resid={row['resid']:+.4f} "
            f"executed={row['executed_rate']:.3f} {row['actions']}"
        )

    print("\n=== by action (the composition confound) ===")
    for action, row in report["by_action"].items():
        if row["n"] < 10:
            continue
        print(f"  {action:10s} n={row['n']:4d} mean={row['mean']:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--pattern", default="**/*.json")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    batches = load_batches(args.results_dir, args.pattern)
    if not batches:
        print(f"no scored sibling batches under {args.results_dir}", file=sys.stderr)
        return 1
    report = build_report(batches, args.permutations, args.seed)
    render(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
