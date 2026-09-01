#!/usr/bin/env python3
"""Risk-coverage analysis over arm-symmetric verdicts.

Selective prediction, applied to the benchmark suite: a system that can DECLINE to answer is
worth more than a system that always answers and is sometimes wrong, provided its decline signal
actually tracks correctness. That signal, here, is
:func:`agent.app.testing.arm_verdict.derive_verdict` -- the one verdict function computable for
every execution variant, not just ``evidence_loop``.

For a confidence tier ``tau`` (ANSWER=2, PARTIAL=1, ABSTAIN=0):

  coverage(tau)           = fraction of cells whose verdict rank >= tau (i.e. the fraction of
                             the suite the system is willing to stand behind at that tier)
  selective_accuracy(tau) = fraction of THOSE covered cells that are correct (score >= threshold)
  risk(tau)                = 1 - selective_accuracy(tau)

This is the differentiator the ledger pivot is for: ``evidence_loop`` is the only arm with a
NATIVE verdict, so before Part 1 (``arm_verdict.py``) a risk-coverage curve could only ever be
plotted for one arm -- which proves nothing about whether structure beats a baseline, only that
the one arm with a verdict field has a verdict field. ``derive_verdict`` gives every arm a verdict
computed the identical way, so this curve can be plotted -- and compared -- for baselines too.

Usage:
  python3 scripts/risk_coverage.py                         # all arms, barrage24b (default scope)
  python3 scripts/risk_coverage.py --run-ids '' --variant evidence_loop,sequential_react_extract
  python3 scripts/risk_coverage.py --threshold 0.5 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.app.testing.arm_verdict import (  # noqa: E402
    VERDICT_ANSWER,
    VERDICT_PARTIAL,
    VERDICT_ABSTAIN,
    derive_verdict,
    derive_verdict_graded,
)

# --rule old|graded: "old" is the literal-match rule (derive_verdict), "graded" delegates claim
# support to claim_audit.audit and can credit a RECOMPUTABLE derived claim. Default stays "old"
# until the coordinator reviews the graded rule's live numbers and flips it.
_RULES = {"old": derive_verdict, "graded": derive_verdict_graded}
DEFAULT_RULE = "old"

DEFAULT_CORRECT_THRESHOLD = 1.0

# ANSWER is the most confident tier, ABSTAIN the least. Coverage at rank r counts every cell
# whose verdict rank is >= r, so rank 0 ("ALL", including ABSTAIN) is always full coverage.
_CONFIDENCE_RANK = {VERDICT_ANSWER: 2, VERDICT_PARTIAL: 1, VERDICT_ABSTAIN: 0}
_RANK_LABEL = {2: "ANSWER_ONLY", 1: "ANSWER_OR_PARTIAL", 0: "ALL"}


def is_correct(score: Optional[float], threshold: float = DEFAULT_CORRECT_THRESHOLD) -> bool:
    """A cell counts as correct iff its validated score meets ``threshold``."""
    return score is not None and score >= threshold


def load_cells(run_ids: Optional[Sequence[str]] = None, *,
               extra_run_ids: Optional[Sequence[str]] = None, since: str = "",
               files: Optional[Sequence[str]] = None, tests: Optional[Sequence[str]] = None,
               variants: Optional[Sequence[str]] = None,
               rule: str = DEFAULT_RULE) -> List[Dict[str, Any]]:
    """Every stored cell, flattened to ``{test_id, model, variant, score, verdict}``.

    Reuses :func:`bench_common.discover_files` for path/run-id scoping (the one place that logic
    lives) but reads the raw JSON itself -- ``bench_common.load_row`` flattens away ``execution``
    entirely, and both verdict rules need ``output`` / ``graph`` / ``observability``.

    :param rule: ``"old"`` (:func:`arm_verdict.derive_verdict`, literal-match) or ``"graded"``
        (:func:`arm_verdict.derive_verdict_graded`, delegates to the arm-blind claim auditor).
    """
    try:
        derive = _RULES[rule]
    except KeyError:
        raise ValueError(f"unknown --rule {rule!r}; choose one of {sorted(_RULES)}") from None
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
        val = d.get("validation") or {}
        score = val.get("overall_score")
        if score is None:
            continue
        ex = d.get("execution") or {}
        if not isinstance(ex, dict):
            continue
        result = {"output": ex.get("output") or {}, "graph": ex.get("graph") or {}}
        observability = ex.get("observability") if isinstance(ex.get("observability"), dict) else None
        verdict = derive(result, observability)
        cells.append({
            "path": str(path), "test_id": test_id, "model": d.get("model"),
            "variant": variant, "score": float(score), "verdict": verdict,
        })
    return cells


def coverage_point(cells: Sequence[Dict[str, Any]], min_rank: int,
                   threshold: float = DEFAULT_CORRECT_THRESHOLD) -> Dict[str, Any]:
    """One point on the risk-coverage curve: cells whose verdict rank >= ``min_rank``."""
    total = len(cells)
    covered = [c for c in cells if _CONFIDENCE_RANK.get(c.get("verdict"), 0) >= min_rank]
    n = len(covered)
    correct = sum(1 for c in covered if is_correct(c.get("score"), threshold))
    coverage = n / total if total else 0.0
    selective_accuracy = correct / n if n else float("nan")
    risk = 1.0 - selective_accuracy if n else float("nan")
    return {
        "tau": _RANK_LABEL[min_rank], "min_rank": min_rank, "n": n, "total": total,
        "coverage": coverage, "selective_accuracy": selective_accuracy, "risk": risk,
    }


def risk_coverage_curve(cells: Sequence[Dict[str, Any]],
                        threshold: float = DEFAULT_CORRECT_THRESHOLD) -> List[Dict[str, Any]]:
    """The full curve, most-confident tier first: ANSWER-only, ANSWER-or-PARTIAL, ALL."""
    return [coverage_point(cells, rank, threshold) for rank in (2, 1, 0)]


def curves_by_variant(cells: Sequence[Dict[str, Any]],
                      threshold: float = DEFAULT_CORRECT_THRESHOLD) -> Dict[str, List[Dict[str, Any]]]:
    """One risk-coverage curve per ``variant`` present in ``cells``."""
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_variant.setdefault(cell["variant"], []).append(cell)
    return {variant: risk_coverage_curve(vcells, threshold)
            for variant, vcells in by_variant.items()}


def _print_table(curves: Dict[str, List[Dict[str, Any]]]) -> None:
    print(f"{'variant':<28}{'tau':<20}{'n/total':<12}{'coverage':<12}{'sel_acc':<12}{'risk':<10}")
    for variant in sorted(curves):
        for point in curves[variant]:
            sel_acc = point["selective_accuracy"]
            risk = point["risk"]
            sel_acc_s = "nan" if sel_acc != sel_acc else f"{sel_acc:.3f}"
            risk_s = "nan" if risk != risk else f"{risk:.3f}"
            print(f"{variant:<28}{point['tau']:<20}{point['n']}/{point['total']:<10}"
                 f"{point['coverage']:<12.3f}{sel_acc_s:<12}{risk_s:<10}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-ids", default=None,
                        help="comma-separated run-id prefixes ('' = every result file)")
    parser.add_argument("--variant", default=None, help="comma-separated execution_variant filter")
    parser.add_argument("--tests", default=None, help="comma-separated test_id filter")
    parser.add_argument("--since", default="")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CORRECT_THRESHOLD,
                        help="score >= threshold counts as correct (default: 1.0)")
    parser.add_argument("--rule", choices=sorted(_RULES), default=DEFAULT_RULE,
                        help="verdict rule: 'old' (literal-match) or 'graded' (arm-blind claim "
                             "auditor, credits recomputable derivations) (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    run_ids = args.run_ids.split(",") if args.run_ids is not None else None
    if args.run_ids == "":
        run_ids = []
    variants = args.variant.split(",") if args.variant else None
    tests = args.tests.split(",") if args.tests else None

    cells = load_cells(run_ids=run_ids, since=args.since, tests=tests, variants=variants,
                       rule=args.rule)
    curves = curves_by_variant(cells, args.threshold)

    if args.json:
        print(json.dumps(curves, indent=2))
    else:
        _print_table(curves)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
