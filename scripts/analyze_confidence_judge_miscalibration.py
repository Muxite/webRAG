#!/usr/bin/env python3
"""
Characterise *why* the step-confidence judge barely predicts eventual success. $0, offline.

A6's calibration (``scripts/calibrate_confidence_early_exit.py``) found the confidence prefix
statistic is worth AUC ~0.58 at step 1 and decays to chance by step 5, so no stopping rule
certified. That answers "how much signal" but not "where it goes". This script answers the
second question on the *same* 354-trajectory population — it reuses ``is_regular_roster`` and
the label rule from the calibration driver rather than re-deriving a corpus, so every number
here is comparable to the artifact's.

What it reports (all from recorded runs, no LLM calls):

* **overconfidence** — step-confidence distribution on eventually-FAILED vs eventually-PASSED
  trajectories.
* **judge blindness by action kind** — ``judge_step_confidence`` reads only
  ``result["content"] / result["content_full"] / result["results"]``. ``visit`` and ``search``
  populate those; ``merge`` (``synthesized``/``raw_response``), ``think`` (``thinking_content``),
  ``verify`` (``verdict``/``quote``/``reasoning``) and ``save`` (``count``) do NOT. The judge
  therefore scores those kinds off an empty payload. This section measures how often that
  happens and what it does to the signal; the *mechanism* is re-derived from the recorded
  ``execution.graph`` action results rather than trusted from the source, so the claim stands
  on data.
* **predictiveness by kind and by model**, with Hanley-McNeil intervals, computed over ALL
  judged steps and again over content-bearing kinds only — the ablation that separates "the
  judge is a bad judge" from "the trace is polluted by steps the judge could not see".
* **the reason text** — whether its length or its hedging vocabulary predicts the outcome
  better than the number does.
* **already-recorded alternative signals** — ``verify``'s own evidence-grounded ``confidence``
  and ``merge``'s ``goal_achieved`` flag are computed on every run and thrown away; are they
  better predictors than the judge? (Recorded so the negative result is not re-investigated.)
* **a qualitative sample** of high-confidence steps inside failed trajectories, printed
  verbatim so the failure mode can be read rather than inferred.

Findings are written up in ``agent/app/CONFIDENCE_JUDGE_MISCALIBRATION.md``. This
script is analysis only: it reads result JSON and prints; it writes no artifact the engine
reads and changes no behaviour.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/analyze_confidence_judge_miscalibration.py

    ... --samples 10             # how many high-confidence-but-failed steps to print
    ... --json-out report.json   # also dump the whole report as JSON
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "services", _ROOT / "agent", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent.app.idea_policies.confidence_early_exit import (  # noqa: E402
    LABEL_THRESHOLD,
    MAX_TIMESTEP,
    STATISTICS,
    prefix_statistic,
)
from calibrate_confidence_early_exit import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    is_regular_roster,
)

#: Action kinds whose result payload lands in a field ``judge_step_confidence`` actually reads
#: (``content``/``content_full``/``results``). Everything else is judged blind — see the module
#: docstring for the field-by-field mapping.
CONTENT_BEARING_KINDS: Tuple[str, ...] = ("visit", "search")
#: A confidence at or below this is treated as degenerate ("the judge scored nothing"), used to
#: separate the judge's graded opinion from its empty-payload floor.
DEGENERATE_CONFIDENCE = 0.05
#: What counts as the judge vouching for a step (the lift and the qualitative sample use it).
HIGH_CONFIDENCE = 0.8
#: A6's own re-expansion trigger (``got_step_confidence_reexpand_threshold`` JSON default), so the
#: exposure numbers below describe the rule A1 actually ships with.
A1_REEXPAND_THRESHOLD = 0.5
#: Run-level split used by the original barrage census, reproduced here for comparability.
CENSUS_SPLIT = 0.6
#: The census slice whose anti-calibration claim is quoted in ``contract_satisfaction.py``.
CENSUS_SLICE_MODEL_MARKER = "nano"
CENSUS_SLICE_SOURCE_MARKER = "good_adaptive"

#: The judge's own words when it was handed an empty payload. Matched on the logged ``reason``
#: so blindness is counted from what the judge SAID, not only from the score it emitted.
BLIND_REASON_RE = re.compile(
    r"resolved[_ ]content is empty|no (?:step )?(?:output|content|resolved content)|"
    r"nothing to (?:verify|assess|evaluate)|content is empty|no output|empty output",
    re.IGNORECASE,
)
#: Hedging vocabulary — the cheap test for "is the prose more honest than the number".
HEDGE_RE = re.compile(
    r"\b(?:appears? to|seems? to|likely|probabl\w+|may |might |could |uncertain\w*|unclear|"
    r"unable to|cannot|can't|not (?:fully|directly|entirely))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JudgedStep:
    """One entry of ``execution.observability.step_confidence.trace``."""

    kind: Optional[str]
    confidence: float
    reason: str = ""
    step: int = 0
    node_id: str = ""

    @property
    def degenerate(self) -> bool:
        return self.confidence <= DEGENERATE_CONFIDENCE

    @property
    def blind_reason(self) -> bool:
        return bool(BLIND_REASON_RE.search(self.reason or ""))


@dataclass(frozen=True)
class ActionRecord:
    """One successful action result from the recorded graph — the judge's *input* side.

    ``judge_visible`` is the whole mechanism claim in one boolean: did this result put
    anything into the three fields ``judge_step_confidence`` reads?
    """

    action: str
    judge_visible: bool
    self_confidence: Optional[float] = None
    goal_achieved: Optional[bool] = None


@dataclass(frozen=True)
class JudgedRun:
    """One recorded trajectory: its judged steps plus the eventual grep-computed label."""

    source: str
    model: str
    score: float
    steps: Tuple[JudgedStep, ...] = field(default_factory=tuple)
    actions: Tuple[ActionRecord, ...] = field(default_factory=tuple)

    @property
    def label(self) -> int:
        return 1 if self.score >= LABEL_THRESHOLD else 0

    def confidences(self, kinds: Optional[Sequence[str]] = None) -> List[float]:
        return [s.confidence for s in self.steps if kinds is None or s.kind in kinds]


# --- statistics ------------------------------------------------------------------------


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Mann-Whitney AUC (ties count a half), or ``None`` when one class is missing."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def auc_stderr(auc: float, n_pos: int, n_neg: int) -> Optional[float]:
    """Hanley-McNeil standard error of an AUC estimate.

    Assumes independent samples. Step-level AUCs below share one label across a run's steps,
    so their intervals are optimistic; run-level AUCs are the ones to quote.
    """
    if n_pos <= 0 or n_neg <= 0:
        return None
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def auc_with_ci(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    """``{auc, n, n_pos, n_neg, stderr, ci95}`` — the reporting unit used everywhere below."""
    n_pos = sum(1 for l in labels if l)
    n_neg = len(labels) - n_pos
    auc = roc_auc(scores, labels)
    out: Dict[str, Any] = {"n": len(labels), "n_pos": n_pos, "n_neg": n_neg, "auc": auc}
    if auc is None:
        out["stderr"] = None
        out["ci95"] = None
        return out
    se = auc_stderr(auc, n_pos, n_neg)
    out["stderr"] = se
    out["ci95"] = None if se is None else (auc - 1.96 * se, auc + 1.96 * se)
    return out


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


# --- loading (same population as the A6 calibration driver) -----------------------------


def action_records(payload: Dict[str, Any]) -> Tuple[ActionRecord, ...]:
    """Every SUCCESSFUL action result in the recorded graph, tagged with judge visibility.

    Mirrors ``judge_step_confidence``'s own extraction (``content`` -> ``content_full`` ->
    ``results``) so "the judge saw nothing" is measured, not assumed. ``verify``'s own
    ``confidence`` and ``merge``'s ``goal_achieved`` ride along for the alternative-signal probe.
    """
    nodes = ((payload.get("execution") or {}).get("graph") or {}).get("nodes") or {}
    iterable = nodes.values() if isinstance(nodes, dict) else nodes
    records: List[ActionRecord] = []
    for node in iterable:
        if not isinstance(node, dict):
            continue
        result = (node.get("details") or {}).get("action_result")
        if not isinstance(result, dict) or not result.get("success"):
            continue
        visible = bool(result.get("content") or result.get("content_full") or result.get("results"))
        confidence = result.get("confidence")
        records.append(
            ActionRecord(
                action=str(result.get("action") or "unknown"),
                judge_visible=visible,
                self_confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                goal_achieved=(
                    bool(result.get("goal_achieved"))
                    if result.get("goal_achieved") is not None
                    else None
                ),
            )
        )
    return tuple(records)


def run_from_result(payload: Dict[str, Any], source: str) -> Optional[JudgedRun]:
    """One result JSON -> one :class:`JudgedRun`, or ``None`` when it is not usable.

    Filtering matches ``calibrate_confidence_early_exit.trajectory_from_result`` exactly (a
    non-empty trace, a numeric ``validation.overall_score``, numeric confidences) so this
    analysis and the calibration artifact describe the same corpus.
    """
    execution = payload.get("execution") or {}
    trace = (((execution.get("observability") or {}).get("step_confidence") or {}).get("trace")) or []
    if not trace:
        return None
    score = (payload.get("validation") or {}).get("overall_score")
    if not isinstance(score, (int, float)):
        return None
    steps = tuple(
        JudgedStep(
            kind=entry.get("kind"),
            confidence=float(entry["confidence"]),
            reason=str(entry.get("reason") or ""),
            step=int(entry.get("step") or 0),
            node_id=str(entry.get("node_id") or ""),
        )
        for entry in trace
        if isinstance(entry, dict) and isinstance(entry.get("confidence"), (int, float))
    )
    if not steps:
        return None
    return JudgedRun(
        source=os.path.basename(source),
        model=str(payload.get("model") or "unknown"),
        score=float(score),
        steps=steps,
        actions=action_records(payload),
    )


def load_runs(results_dir: Path) -> List[JudgedRun]:
    runs: List[JudgedRun] = []
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
        run = run_from_result(payload, path)
        if run is not None:
            runs.append(run)
    return runs


# --- the individual analyses -----------------------------------------------------------


def overconfidence(runs: Sequence[JudgedRun], kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Is the judge systematically more generous than the outcome warrants?"""
    passed = [c for r in runs if r.label for c in r.confidences(kinds)]
    failed = [c for r in runs if not r.label for c in r.confidences(kinds)]
    scores = [c for r in runs for c in r.confidences(kinds)]
    labels = [r.label for r in runs for _ in r.confidences(kinds)]
    out = {
        "passed_steps": summarize(passed),
        "failed_steps": summarize(failed),
        "step_auc": auc_with_ci(scores, labels),
    }
    if passed and failed:
        out["mean_gap"] = statistics.mean(passed) - statistics.mean(failed)
    else:
        out["mean_gap"] = None
    return out


def by_kind(runs: Sequence[JudgedRun], min_n: int = 5) -> Dict[str, Any]:
    """Per-action-kind means, blindness rate and predictiveness."""
    buckets: Dict[str, List[Tuple[JudgedStep, int]]] = {}
    for run in runs:
        for step in run.steps:
            buckets.setdefault(step.kind or "unknown", []).append((step, run.label))
    report: Dict[str, Any] = {}
    for kind, rows in buckets.items():
        if len(rows) < min_n:
            continue
        confs = [s.confidence for s, _ in rows]
        labels = [l for _, l in rows]
        graded = [(s, l) for s, l in rows if not s.degenerate]
        confident = [(s, l) for s, l in rows if s.confidence >= HIGH_CONFIDENCE]
        base_rate = statistics.mean(labels)
        report[kind] = {
            "n": len(rows),
            "content_bearing": kind in CONTENT_BEARING_KINDS,
            "mean_confidence": statistics.mean(confs),
            "degenerate_fraction": sum(1 for s, _ in rows if s.degenerate) / len(rows),
            "blind_reason_fraction": sum(1 for s, _ in rows if s.blind_reason) / len(rows),
            "mean_confidence_passed": (
                statistics.mean([s.confidence for s, l in rows if l]) if any(labels) else None
            ),
            "mean_confidence_failed": (
                statistics.mean([s.confidence for s, l in rows if not l])
                if not all(labels)
                else None
            ),
            "auc": auc_with_ci(confs, labels),
            "auc_graded_only": auc_with_ci(
                [s.confidence for s, _ in graded], [l for _, l in graded]
            ),
            "auc_degenerate_indicator": auc_with_ci(
                [0.0 if s.degenerate else 1.0 for s, _ in rows], labels
            ),
            # What a caller actually gets from a high score: the eventual pass rate of the
            # steps the judge was confident about, against this kind's own base rate.
            "high_confidence_fraction": len(confident) / len(rows),
            "high_confidence_pass_rate": (
                statistics.mean([l for _, l in confident]) if confident else None
            ),
            "high_confidence_lift": (
                statistics.mean([l for _, l in confident]) - base_rate if confident else None
            ),
            "base_rate": base_rate,
        }
    return dict(sorted(report.items(), key=lambda kv: -kv[1]["n"]))


def by_model(runs: Sequence[JudgedRun]) -> Dict[str, Any]:
    """Per-model predictiveness, with and without the blind kinds.

    E-valuator's own caution is that calibration is per-model; this tests whether the weakness
    is really a bad-model average or something the models share.
    """
    report: Dict[str, Any] = {}
    for model in sorted({r.model for r in runs}):
        subset = [r for r in runs if r.model == model]
        step_scores = [c for r in subset for c in r.confidences()]
        step_labels = [r.label for r in subset for _ in r.confidences()]
        run_labels = [r.label for r in subset]
        mean_all = [statistics.mean(r.confidences()) for r in subset]
        with_content = [r for r in subset if r.confidences(CONTENT_BEARING_KINDS)]
        mean_content = [
            statistics.mean(r.confidences(CONTENT_BEARING_KINDS)) for r in with_content
        ]
        report[model] = {
            "runs": len(subset),
            "base_rate": statistics.mean(run_labels) if run_labels else None,
            "step_auc": auc_with_ci(step_scores, step_labels),
            "run_mean_auc": auc_with_ci(mean_all, run_labels),
            "run_mean_content_auc": auc_with_ci(mean_content, [r.label for r in with_content]),
        }
    return report


def payload_visibility(runs: Sequence[JudgedRun]) -> Dict[str, Any]:
    """Per action kind: how often the result carried anything the judge can actually read.

    This is the mechanism behind the blindness numbers, measured from recorded runs.
    """
    buckets: Dict[str, List[ActionRecord]] = {}
    for run in runs:
        for record in run.actions:
            buckets.setdefault(record.action, []).append(record)
    return {
        action: {
            "n_results": len(records),
            "judge_visible_fraction": sum(1 for r in records if r.judge_visible) / len(records),
        }
        for action, records in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    }


def alternative_signals(runs: Sequence[JudgedRun]) -> Dict[str, Any]:
    """Do the signals the engine already computes beat the judge? (Recorded negative result.)"""
    verify_steps = [
        (r.self_confidence, run.label)
        for run in runs
        for r in run.actions
        if r.action == "verify" and r.self_confidence is not None
    ]
    verify_runs = [
        (statistics.mean([r.self_confidence for r in run.actions if r.action == "verify" and r.self_confidence is not None]), run.label)
        for run in runs
        if any(r.action == "verify" and r.self_confidence is not None for r in run.actions)
    ]
    merge_steps = [
        (1.0 if r.goal_achieved else 0.0, run.label)
        for run in runs
        for r in run.actions
        if r.action == "merge" and r.goal_achieved is not None
    ]
    merge_runs = [
        (statistics.mean([1.0 if r.goal_achieved else 0.0 for r in run.actions if r.action == "merge" and r.goal_achieved is not None]), run.label)
        for run in runs
        if any(r.action == "merge" and r.goal_achieved is not None for r in run.actions)
    ]
    return {
        "verify_self_confidence_step": auc_with_ci([v for v, _ in verify_steps], [l for _, l in verify_steps]),
        "verify_self_confidence_run": auc_with_ci([v for v, _ in verify_runs], [l for _, l in verify_runs]),
        "merge_goal_achieved_step": auc_with_ci([v for v, _ in merge_steps], [l for _, l in merge_steps]),
        "merge_goal_achieved_run": auc_with_ci([v for v, _ in merge_runs], [l for _, l in merge_runs]),
    }


def structural_baselines(runs: Sequence[JudgedRun]) -> Dict[str, Any]:
    """Free, LLM-free graph statistics scored on the same task as the judge.

    The yardstick the judge has to beat to be worth its calls: how many leaves were judged at
    all, and what fraction of them were content-bearing. Both are *descriptive* — an early-exit
    rule that acts on them changes them, so they are a benchmark, not a proposed lever.
    """
    def block(subset: Sequence[JudgedRun]) -> Dict[str, Any]:
        labels = [r.label for r in subset]
        return {
            "runs": len(subset),
            "n_judged_steps": auc_with_ci([float(len(r.steps)) for r in subset], labels),
            "content_step_fraction": auc_with_ci(
                [
                    sum(1 for s in r.steps if s.kind in CONTENT_BEARING_KINDS) / len(r.steps)
                    for r in subset
                ],
                labels,
            ),
            "run_mean_confidence": auc_with_ci(
                [statistics.mean(r.confidences()) for r in subset], labels
            ),
        }

    return {
        "pooled": block(runs),
        "by_model": {
            model: block([r for r in runs if r.model == model])
            for model in sorted({r.model for r in runs})
        },
    }


def run_level_auc(runs: Sequence[JudgedRun], kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """AUC of each whole-run confidence statistic — one sample per trajectory, so the
    intervals are honest (unlike the step-level ones, which share labels within a run)."""
    subset = [r for r in runs if r.confidences(kinds)]
    labels = [r.label for r in subset]
    return {
        statistic: auc_with_ci(
            [prefix_statistic(r.confidences(kinds), statistic) for r in subset], labels
        )
        for statistic in STATISTICS
    }


def prefix_auc_table(
    runs: Sequence[JudgedRun],
    statistic: str,
    kinds: Optional[Sequence[str]] = None,
    max_timestep: int = MAX_TIMESTEP,
) -> Dict[int, Dict[str, Any]]:
    """AUC of the A6 prefix statistic at each timestep — the decay curve, ablatable by kind."""
    table: Dict[int, Dict[str, Any]] = {}
    for timestep in range(1, max_timestep + 1):
        values: List[float] = []
        labels: List[int] = []
        for run in runs:
            seq = run.confidences(kinds)
            if len(seq) >= timestep:
                values.append(prefix_statistic(seq[:timestep], statistic))
                labels.append(run.label)
        if not values:
            continue
        table[timestep] = auc_with_ci(values, labels)
    return table


def reason_text_signal(runs: Sequence[JudgedRun], kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Does the prose carry signal the number throws away?

    An AUC below 0.5 means the feature predicts FAILURE; ``inverted_auc`` restates it as a
    failure predictor so the two directions are comparable at a glance.
    """
    lengths: List[float] = []
    hedges: List[float] = []
    confs: List[float] = []
    labels: List[int] = []
    for run in runs:
        for step in run.steps:
            if kinds is not None and step.kind not in kinds:
                continue
            lengths.append(float(len(step.reason)))
            hedges.append(float(len(HEDGE_RE.findall(step.reason))))
            confs.append(step.confidence)
            labels.append(run.label)
    out = {
        "confidence": auc_with_ci(confs, labels),
        "reason_length": auc_with_ci(lengths, labels),
        "hedge_count": auc_with_ci(hedges, labels),
    }
    for row in out.values():
        row["inverted_auc"] = None if row["auc"] is None else 1.0 - row["auc"]
    return out


def trigger_exposure(
    runs: Sequence[JudgedRun], threshold: float = A1_REEXPAND_THRESHOLD
) -> Dict[str, Any]:
    """What A1's ``confidence < threshold`` re-expansion trigger actually fires on."""
    steps = [s for r in runs for s in r.steps]
    below = [s for s in steps if s.confidence < threshold]
    kinds: Dict[str, int] = {}
    for step in below:
        kinds[step.kind or "unknown"] = kinds.get(step.kind or "unknown", 0) + 1
    return {
        "threshold": threshold,
        "n_steps": len(steps),
        "n_below": len(below),
        "below_fraction_of_steps": (len(below) / len(steps)) if steps else 0.0,
        "blind_kind_fraction_of_below": (
            sum(1 for s in below if s.kind not in CONTENT_BEARING_KINDS) / len(below)
        )
        if below
        else 0.0,
        "blind_reason_fraction_of_below": (
            sum(1 for s in below if s.blind_reason) / len(below)
        )
        if below
        else 0.0,
        "kind_mix": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
    }


def census_split(
    runs: Sequence[JudgedRun],
    threshold: float = CENSUS_SPLIT,
    kinds: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run-level "high mean confidence vs low" split — the barrage census's own statistic."""
    high: List[JudgedRun] = []
    low: List[JudgedRun] = []
    for run in runs:
        seq = run.confidences(kinds)
        if not seq:
            continue
        (high if statistics.mean(seq) >= threshold else low).append(run)
    def side(group: List[JudgedRun]) -> Dict[str, Any]:
        return {
            "n": len(group),
            "mean_score": statistics.mean([r.score for r in group]) if group else None,
            "pass_rate": statistics.mean([r.label for r in group]) if group else None,
        }
    return {"threshold": threshold, "high_confidence": side(high), "low_confidence": side(low)}


def high_confidence_failures(
    runs: Sequence[JudgedRun], minimum: float = HIGH_CONFIDENCE, limit: int = 10
) -> List[Dict[str, Any]]:
    """High-scoring steps inside trajectories that eventually failed, one per source run."""
    picked: List[Dict[str, Any]] = []
    for run in sorted(runs, key=lambda r: r.source):
        if run.label:
            continue
        best = max(
            (s for s in run.steps if s.confidence >= minimum), key=lambda s: s.confidence, default=None
        )
        if best is None:
            continue
        picked.append(
            {
                "source": run.source,
                "model": run.model,
                "overall_score": run.score,
                "kind": best.kind,
                "confidence": best.confidence,
                "reason": best.reason,
            }
        )
        if len(picked) >= limit:
            break
    return picked


def blindness_summary(runs: Sequence[JudgedRun]) -> Dict[str, Any]:
    steps = [s for r in runs for s in r.steps]
    blind = [s for s in steps if s.kind not in CONTENT_BEARING_KINDS]
    return {
        "n_steps": len(steps),
        "n_blind_kind": len(blind),
        "blind_kind_fraction": (len(blind) / len(steps)) if steps else 0.0,
        "blind_kind_degenerate_fraction": (
            sum(1 for s in blind if s.degenerate) / len(blind)
        )
        if blind
        else 0.0,
        "blind_kind_blind_reason_fraction": (
            sum(1 for s in blind if s.blind_reason) / len(blind)
        )
        if blind
        else 0.0,
        "runs_with_blind_step": (
            sum(1 for r in runs if any(s.kind not in CONTENT_BEARING_KINDS for s in r.steps)) / len(runs)
        )
        if runs
        else 0.0,
        "runs_blind_within_first_five": (
            sum(
                1
                for r in runs
                if any(s.kind not in CONTENT_BEARING_KINDS for s in r.steps[:5])
            )
            / len(runs)
        )
        if runs
        else 0.0,
    }


def build_report(runs: Sequence[JudgedRun], samples: int = 10) -> Dict[str, Any]:
    census_slice = [
        r
        for r in runs
        if CENSUS_SLICE_MODEL_MARKER in r.model and CENSUS_SLICE_SOURCE_MARKER in r.source
    ]
    return {
        "corpus": {
            "n_runs": len(runs),
            "n_steps": sum(len(r.steps) for r in runs),
            "base_rate": statistics.mean([r.label for r in runs]) if runs else None,
            "models": sorted({r.model for r in runs}),
            "label_rule": f"validation.overall_score >= {LABEL_THRESHOLD}",
        },
        "overconfidence_all_kinds": overconfidence(runs),
        "overconfidence_content_kinds": overconfidence(runs, CONTENT_BEARING_KINDS),
        "run_level_auc_all_kinds": run_level_auc(runs),
        "run_level_auc_content_kinds": run_level_auc(runs, CONTENT_BEARING_KINDS),
        "blindness": blindness_summary(runs),
        "payload_visibility": payload_visibility(runs),
        "by_kind": by_kind(runs),
        "by_model": by_model(runs),
        "structural_baselines": structural_baselines(runs),
        "prefix_auc": {
            statistic: {
                "all_kinds": prefix_auc_table(runs, statistic),
                "content_kinds": prefix_auc_table(runs, statistic, CONTENT_BEARING_KINDS),
            }
            for statistic in STATISTICS
        },
        "reason_text_all_kinds": reason_text_signal(runs),
        "reason_text_content_kinds": reason_text_signal(runs, CONTENT_BEARING_KINDS),
        "alternative_signals": alternative_signals(runs),
        "a1_trigger_exposure": trigger_exposure(runs),
        "census_split_all_kinds": census_split(runs),
        "census_split_content_kinds": census_split(runs, kinds=CONTENT_BEARING_KINDS),
        "census_slice": {
            "selector": f"model~{CENSUS_SLICE_MODEL_MARKER} and source~{CENSUS_SLICE_SOURCE_MARKER}",
            "n_runs": len(census_slice),
            "all_kinds": census_split(census_slice),
            "content_kinds": census_split(census_slice, kinds=CONTENT_BEARING_KINDS),
        },
        "high_confidence_failures": high_confidence_failures(runs, limit=samples),
    }


# --- rendering --------------------------------------------------------------------------


def _auc(row: Optional[Dict[str, Any]]) -> str:
    if not row or row.get("auc") is None:
        return "  --  "
    ci = row.get("ci95")
    tail = f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
    return f"{row['auc']:.3f}{tail}"


def render(report: Dict[str, Any]) -> None:
    corpus = report["corpus"]
    print(
        f"corpus: {corpus['n_runs']} trajectories / {corpus['n_steps']} judged steps, "
        f"base rate {corpus['base_rate']:.3f} ({corpus['label_rule']})"
    )
    print(f"models: {', '.join(corpus['models'])}")

    print("\n== overconfidence (step confidence vs eventual outcome) ==")
    for name, key in (("all kinds", "overconfidence_all_kinds"), ("content-bearing", "overconfidence_content_kinds")):
        row = report[key]
        print(
            f"  {name:15s} passed-run steps mean={row['passed_steps']['mean']:.3f} "
            f"median={row['passed_steps']['median']:.3f} (n={row['passed_steps']['n']}) | "
            f"failed-run steps mean={row['failed_steps']['mean']:.3f} "
            f"median={row['failed_steps']['median']:.3f} (n={row['failed_steps']['n']}) | "
            f"gap={row['mean_gap']:+.3f} step-AUC={_auc(row['step_auc'])}"
        )

    print("\n== run-level AUC (one sample per trajectory — the honest interval) ==")
    for name, key in (("all kinds", "run_level_auc_all_kinds"), ("content-bearing", "run_level_auc_content_kinds")):
        row = report[key]
        cells = " ".join(f"{statistic}={_auc(row[statistic])}" for statistic in row)
        print(f"  {name:15s} {cells}")

    print("\n== payload visibility (did the judge receive anything at all?) ==")
    for action, row in report["payload_visibility"].items():
        print(
            f"  {action:22s} successful results={row['n_results']:5d} "
            f"judge-visible={row['judge_visible_fraction']:.3f}"
        )

    blind = report["blindness"]
    print("\n== judge blindness (kinds whose payload the judge never receives) ==")
    print(
        f"  {blind['n_blind_kind']}/{blind['n_steps']} steps ({blind['blind_kind_fraction']:.3f}) are "
        f"merge/think/verify/save; {blind['blind_kind_degenerate_fraction']:.3f} of those score "
        f"<={DEGENERATE_CONFIDENCE}, {blind['blind_kind_blind_reason_fraction']:.3f} say so in the reason text"
    )
    print(
        f"  {blind['runs_with_blind_step']:.3f} of runs contain one; "
        f"{blind['runs_blind_within_first_five']:.3f} contain one in the first five judged steps"
    )

    print("\n== by action kind ==")
    print(
        f"  {'kind':9s} {'n':>5s} {'mean':>6s} {'degen':>6s} {'blindtxt':>8s} {'pass_mu':>7s} "
        f"{'fail_mu':>7s} {'AUC':>18s} {'AUC(graded)':>18s} {'hi>=0.8':>8s} {'lift':>7s}"
    )
    for kind, row in report["by_kind"].items():
        lift = row["high_confidence_lift"]
        print(
            f"  {kind:9s} {row['n']:5d} {row['mean_confidence']:6.3f} {row['degenerate_fraction']:6.3f} "
            f"{row['blind_reason_fraction']:8.3f} "
            f"{(row['mean_confidence_passed'] or 0):7.3f} {(row['mean_confidence_failed'] or 0):7.3f} "
            f"{_auc(row['auc']):>18s} {_auc(row['auc_graded_only']):>18s} "
            f"{row['high_confidence_fraction']:8.3f} {('  --  ' if lift is None else format(lift, '+7.3f'))}"
        )

    print("\n== by model (does the weakness live in one model?) ==")
    for model, row in report["by_model"].items():
        print(
            f"  {model:28s} runs={row['runs']:4d} base={row['base_rate']:.3f} "
            f"step={_auc(row['step_auc'])} run-mean={_auc(row['run_mean_auc'])} "
            f"run-mean(content)={_auc(row['run_mean_content_auc'])}"
        )

    print("\n== free structural baselines the judge has to beat (descriptive, not levers) ==")
    baselines = report["structural_baselines"]
    for name, row in [("pooled", baselines["pooled"])] + list(baselines["by_model"].items()):
        print(
            f"  {name:28s} runs={row['runs']:4d} n_judged_steps={_auc(row['n_judged_steps'])} "
            f"content_fraction={_auc(row['content_step_fraction'])} "
            f"run_mean_confidence={_auc(row['run_mean_confidence'])}"
        )

    print("\n== prefix-statistic AUC by timestep (A6's own statistic; ablated by kind) ==")
    for statistic, tables in report["prefix_auc"].items():
        for name, table in tables.items():
            cells = " ".join(
                f"t{t}:{(row['auc'] if row['auc'] is not None else float('nan')):.3f}({row['n']})"
                for t, row in sorted(table.items())
            )
            print(f"  {statistic:13s} {name:14s} {cells}")

    print("\n== reason text as a predictor (AUC>0.5 predicts PASS, <0.5 predicts FAIL) ==")
    for name, key in (("all kinds", "reason_text_all_kinds"), ("content-bearing", "reason_text_content_kinds")):
        row = report[key]
        print(
            f"  {name:15s} confidence={_auc(row['confidence'])} "
            f"reason_length={_auc(row['reason_length'])} hedge_count={_auc(row['hedge_count'])}"
        )

    print("\n== signals the engine already computes and discards ==")
    for name, row in report["alternative_signals"].items():
        print(f"  {name:34s} {_auc(row)} (n={row['n']})")

    trig = report["a1_trigger_exposure"]
    print(f"\n== A1 exposure (confidence < {trig['threshold']} drives re-expansion) ==")
    print(
        f"  {trig['n_below']}/{trig['n_steps']} judged steps ({trig['below_fraction_of_steps']:.3f}) are below the "
        f"threshold; {trig['blind_kind_fraction_of_below']:.3f} of those are blind kinds, "
        f"{trig['blind_reason_fraction_of_below']:.3f} say 'nothing to judge' outright"
    )
    print(f"  kind mix: {trig['kind_mix']}")

    print("\n== run-level census split (mean confidence >= 0.6 vs below) ==")
    for name, key in (("all kinds", "census_split_all_kinds"), ("content-bearing", "census_split_content_kinds")):
        row = report[key]
        hi, lo = row["high_confidence"], row["low_confidence"]
        print(
            f"  {name:15s} high n={hi['n']:4d} score={hi['mean_score']:.3f} pass={hi['pass_rate']:.3f} | "
            f"low n={lo['n']:4d} score={lo['mean_score']:.3f} pass={lo['pass_rate']:.3f}"
        )
    slice_row = report["census_slice"]
    for name in ("all_kinds", "content_kinds"):
        row = slice_row[name]
        hi, lo = row["high_confidence"], row["low_confidence"]
        if not hi["n"] or not lo["n"]:
            continue
        print(
            f"  slice({slice_row['selector']}, n={slice_row['n_runs']}) {name:12s} "
            f"high n={hi['n']:3d} score={hi['mean_score']:.3f} | low n={lo['n']:3d} score={lo['mean_score']:.3f}"
        )

    print("\n== high confidence (>=0.8) inside eventually-FAILED trajectories ==")
    for case in report["high_confidence_failures"]:
        print(
            f"  --- {case['source']} score={case['overall_score']:.2f} kind={case['kind']} "
            f"conf={case['confidence']:.2f}"
        )
        print(f"      {case['reason'][:400]}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--samples", type=int, default=10, help="high-confidence failures to print")
    parser.add_argument("--json-out", default=None, help="also write the full report as JSON")
    args = parser.parse_args(argv)

    runs = load_runs(Path(args.results_dir))
    if not runs:
        print(f"no usable trajectories under {args.results_dir}", file=sys.stderr)
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
