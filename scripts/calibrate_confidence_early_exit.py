#!/usr/bin/env python3
"""
Derive the calibrated high-confidence early-exit rule (A6) from recorded benchmark runs. $0.

Reads every ``services/agent/idea_test_results/*.json`` that (a) comes from the REGULAR model
roster, (b) carries a non-empty ``execution.observability.step_confidence.trace``, and (c) has
a ``validation.overall_score``; turns each into one E-valuator-style calibration example
``(confidence-sequence, eventual-label)``; splits it deterministically into a fit set and a
never-fitted holdout; fits per-timestep stop thresholds at each rung of
``TARGET_STOP_PRECISION_LADDER``; and writes the versioned artifact the engine reads:

    services/agent/app/confidence_early_exit_calibration.json

The statistics live in ``idea_policies/confidence_early_exit.py`` (shared with the engine, so
the rule that is measured here is byte-for-byte the rule that runs); this file is only the
data plumbing and the selection policy.

Selection policy (deliberately boring, all constants named in the shared module):
  1. Try the ladder rungs from strictest to loosest.
  2. A rung counts as certified only if some statistic yields a rule whose FIT-SET coverage is
     at least ``MIN_STOP_COVERAGE`` (a rule that never fires saves nothing).
  3. Among statistics certifying a rung, take the highest fit-set coverage; ties break by the
     declaration order of ``STATISTICS``.
  4. Ship the first rung that certifies, record whether it is the preferred rung, and report
     the holdout numbers for it. The holdout is measured ONCE, at the end, and never steers
     the choice.

Usage::

    PYTHONPATH=services:services/agent ./.venv/bin/python \\
      scripts/calibrate_confidence_early_exit.py

    ... --dry-run            # print the report, write nothing
    ... --target 0.75        # force one rung instead of walking the ladder
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "services", _ROOT / "services" / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent.app.idea_policies.confidence_early_exit import (  # noqa: E402
    ARTIFACT_VERSION,
    CALIBRATION_PATH,
    CALIBRATION_SPLIT,
    CERTIFICATION_DELTA,
    EXCLUDED_MARKERS,
    LABEL_THRESHOLD,
    MAX_STEP_STOP_FRACTION,
    MAX_TIMESTEP,
    MIN_STOP_COVERAGE,
    MIN_TIMESTEP,
    MODEL_ROSTER,
    PREFERRED_TARGET_STOP_PRECISION,
    STATISTICS,
    TARGET_STOP_PRECISION_LADDER,
    EarlyExitRule,
    LabelledTrajectory,
    best_certifiable_precision,
    evaluate_rule,
    fit_thresholds,
    prefix_statistic,
)

DEFAULT_RESULTS_DIR = _ROOT / "services" / "agent" / "idea_test_results"


def is_regular_roster(filename: str) -> bool:
    """Regular-roster runs only — badmodel-lab's tiny local models are excluded on purpose."""
    name = os.path.basename(filename)
    if any(marker in name for marker in EXCLUDED_MARKERS):
        return False
    return any(marker in name for marker in MODEL_ROSTER)


def trajectory_from_result(payload: Dict[str, Any], source: str) -> Optional[LabelledTrajectory]:
    """One result JSON -> one ``(confidence-sequence, eventual-label)`` example, or ``None``."""
    execution = payload.get("execution") or {}
    trace = (((execution.get("observability") or {}).get("step_confidence") or {}).get("trace")) or []
    if not trace:
        return None
    score = (payload.get("validation") or {}).get("overall_score")
    if not isinstance(score, (int, float)):
        return None
    confidences = tuple(
        float(entry["confidence"])
        for entry in trace
        if isinstance(entry, dict) and isinstance(entry.get("confidence"), (int, float))
    )
    if not confidences:
        return None
    return LabelledTrajectory(
        confidences=confidences,
        label=1 if float(score) >= LABEL_THRESHOLD else 0,
        source=os.path.basename(source),
    )


def load_trajectories(results_dir: Path) -> List[LabelledTrajectory]:
    trajectories: List[LabelledTrajectory] = []
    for path in sorted(glob.glob(str(Path(results_dir) / "*.json"))):
        if not is_regular_roster(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:  # noqa: BLE001 — a truncated/aborted run file is just skipped
            continue
        if not isinstance(payload, dict):
            continue
        trajectory = trajectory_from_result(payload, path)
        if trajectory is not None:
            trajectories.append(trajectory)
    return trajectories


def split_trajectories(
    trajectories: List[LabelledTrajectory], fraction: float = CALIBRATION_SPLIT
) -> Tuple[List[LabelledTrajectory], List[LabelledTrajectory]]:
    """Deterministic fit/holdout split, keyed on the source filename.

    Hash-keyed rather than random so re-running the script on the same corpus reproduces the
    same artifact exactly, and so adding new runs cannot reshuffle old ones across the split.
    """
    fit: List[LabelledTrajectory] = []
    holdout: List[LabelledTrajectory] = []
    for trajectory in trajectories:
        digest = hashlib.sha256(trajectory.source.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        (fit if bucket < fraction else holdout).append(trajectory)
    return fit, holdout


def fit_best_statistic(
    fit_set: List[LabelledTrajectory], target: float
) -> Optional[Tuple[str, Dict[int, float], Dict[str, Any]]]:
    """Best (statistic, thresholds) for one ladder rung, judged on the fit set alone."""
    best: Optional[Tuple[str, Dict[int, float], Dict[str, Any]]] = None
    for statistic in STATISTICS:
        thresholds = fit_thresholds(fit_set, statistic, target)
        if not thresholds:
            continue
        rule = EarlyExitRule(statistic, thresholds, target)
        metrics = evaluate_rule(fit_set, rule)
        if metrics["coverage"] < MIN_STOP_COVERAGE:
            continue
        if best is None or metrics["coverage"] > best[2]["coverage"]:
            best = (statistic, thresholds, metrics)
    return best


def signal_ceiling(fit_set: List[LabelledTrajectory]) -> Dict[str, Any]:
    """How high a stop precision the confidence signal could certify at all, per timestep.

    Reported unconditionally so a "no rung certified" artifact still says by how much the
    corpus falls short (and a future, larger corpus can be compared against it).
    """
    tested = max(1, MAX_TIMESTEP - MIN_TIMESTEP + 1)
    per_step_delta = CERTIFICATION_DELTA / tested
    per_statistic: Dict[str, Any] = {}
    ceiling = 0.0
    for statistic in STATISTICS:
        rows: Dict[str, Any] = {}
        for timestep in range(MIN_TIMESTEP, MAX_TIMESTEP + 1):
            samples = [
                (prefix_statistic(tr.confidences[:timestep], statistic), tr.label)
                for tr in fit_set
                if len(tr.confidences) >= timestep
            ]
            if not samples:
                continue
            bound, tau = best_certifiable_precision(samples, per_step_delta)
            rows[str(timestep)] = {
                "n_reaching": len(samples),
                "base_rate": sum(label for _, label in samples) / len(samples),
                "best_certified_precision": round(bound, 4),
                "at_threshold": tau,
            }
            ceiling = max(ceiling, bound)
        per_statistic[statistic] = rows
    return {"best_certified_precision_any": round(ceiling, 4), "per_statistic": per_statistic}


def build_artifact(
    trajectories: List[LabelledTrajectory],
    targets: Tuple[float, ...],
) -> Dict[str, Any]:
    fit_set, holdout = split_trajectories(trajectories)
    ladder: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    for target in targets:
        best = fit_best_statistic(fit_set, target)
        rung: Dict[str, Any] = {"target_stop_precision": target, "certified": best is not None}
        if best is not None:
            statistic, thresholds, fit_metrics = best
            rule = EarlyExitRule(statistic, thresholds, target)
            rung.update(
                statistic=statistic,
                thresholds={str(k): v for k, v in sorted(thresholds.items())},
                fit=fit_metrics,
                holdout=evaluate_rule(holdout, rule),
            )
            if selected is None:
                selected = dict(rung)
        ladder.append(rung)

    artifact_base_rate = (
        (sum(t.label for t in trajectories) / len(trajectories)) if trajectories else 0.0
    )
    artifact: Dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "method": (
            "per-timestep threshold on a scalar confidence-prefix statistic, certified by an "
            "exact one-sided Clopper-Pearson lower bound (Bonferroni-corrected across "
            "timesteps) on the stop-set's eventual-pass rate; fitted sequential-consistently "
            "on a deterministic 70% split, measured on the untouched 30% holdout. Simplified "
            "from E-valuator (arXiv 2512.03109) for a 354-trajectory corpus - see "
            "idea_policies/confidence_early_exit.py."
        ),
        "label_rule": f"validation.overall_score >= {LABEL_THRESHOLD}",
        "model_roster": list(MODEL_ROSTER),
        "excluded_markers": list(EXCLUDED_MARKERS),
        "certification_delta": CERTIFICATION_DELTA,
        "min_stop_coverage": MIN_STOP_COVERAGE,
        "max_step_stop_fraction": MAX_STEP_STOP_FRACTION,
        "min_timestep": MIN_TIMESTEP,
        "max_timestep": MAX_TIMESTEP,
        "calibration_split": CALIBRATION_SPLIT,
        "preferred_target_stop_precision": PREFERRED_TARGET_STOP_PRECISION,
        "n_trajectories": len(trajectories),
        "n_fit": len(fit_set),
        "n_holdout": len(holdout),
        "base_rate": artifact_base_rate,
        "signal_ceiling": signal_ceiling(fit_set),
        "ladder": ladder,
    }
    if selected is None:
        artifact.update(
            statistic=None,
            thresholds={},
            target_stop_precision=None,
            preferred_target_met=False,
            note=(
                "No ladder rung certified a non-vacuous rule on this corpus: the step-confidence "
                "judge is not predictive enough of eventual success. The highest stop precision "
                "any admissible threshold could certify is signal_ceiling.best_certified_precision_any "
                f"(vs. a {artifact_base_rate:.3f} base rate). The engine therefore never exits early "
                "even with native_confidence_early_exit_enabled on. Re-run this script when the "
                "corpus grows or the confidence judge improves; no code change is needed for a "
                "certified rule to go live."
            ),
        )
    else:
        artifact.update(
            statistic=selected["statistic"],
            thresholds=selected["thresholds"],
            target_stop_precision=selected["target_stop_precision"],
            preferred_target_met=selected["target_stop_precision"] >= PREFERRED_TARGET_STOP_PRECISION,
            fit=selected["fit"],
            holdout=selected["holdout"],
        )
    return artifact


def _fmt(metrics: Dict[str, Any]) -> str:
    return (
        f"n={metrics['n']} stops={metrics['stops']} coverage={metrics['coverage']:.3f} "
        f"stop_precision={metrics['stop_precision']:.3f} "
        f"false_stop_rate={metrics['false_stop_rate']:.3f} "
        f"steps_saved={metrics['judged_steps_saved_fraction']:.3f}"
    )


def report(artifact: Dict[str, Any]) -> None:
    print(
        f"corpus: n={artifact['n_trajectories']} (fit={artifact['n_fit']} "
        f"holdout={artifact['n_holdout']}) base_rate={artifact['base_rate']:.3f}"
    )
    print("ladder:")
    for rung in artifact["ladder"]:
        target = rung["target_stop_precision"]
        if not rung["certified"]:
            print(f"  target={target:.2f}  NOT CERTIFIED (no statistic reaches the coverage floor)")
            continue
        print(
            f"  target={target:.2f}  statistic={rung['statistic']} thresholds={rung['thresholds']}"
        )
        print(f"      fit    : {_fmt(rung['fit'])}")
        print(f"      holdout: {_fmt(rung['holdout'])}")
    ceiling = artifact["signal_ceiling"]["best_certified_precision_any"]
    print(
        f"signal ceiling: the best stop precision ANY admissible threshold certifies is "
        f"{ceiling:.3f} (base rate {artifact['base_rate']:.3f})"
    )
    if artifact.get("statistic") is None:
        print("SELECTED: none — the artifact ships no thresholds, the engine never exits early.")
        return
    print(
        f"SELECTED: statistic={artifact['statistic']} target={artifact['target_stop_precision']:.2f} "
        f"preferred_target_met={artifact['preferred_target_met']} thresholds={artifact['thresholds']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--out", default=str(CALIBRATION_PATH))
    parser.add_argument("--target", type=float, default=None, help="force one ladder rung")
    parser.add_argument("--dry-run", action="store_true", help="print the report, write nothing")
    args = parser.parse_args(argv)

    trajectories = load_trajectories(Path(args.results_dir))
    if not trajectories:
        print(f"no usable trajectories under {args.results_dir}", file=sys.stderr)
        return 1
    targets = (float(args.target),) if args.target is not None else TARGET_STOP_PRECISION_LADDER
    artifact = build_artifact(trajectories, targets)
    report(artifact)
    if args.dry_run:
        print("(dry run — nothing written)")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
