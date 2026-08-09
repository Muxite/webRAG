"""
Calibrated high-confidence early exit (A6) — the pure math, no I/O.

The step-confidence judge (``got_step_confidence_judge_enabled``) only ever *adds*
compute today: a low score drives re-expansion (A1). Nothing short-circuits an easy,
high-confidence mandate, which is where most of the compute-optimal literature's
efficiency actually comes from. This module supplies the missing half: a stopping rule
that says "the trajectory so far looks good enough — stop expanding and finalize".

Why a calibrated rule and not a hand-picked threshold
-----------------------------------------------------
A hand-picked "stop when confidence > 0.9" is unfalsifiable and, on this project's own
data, wrong (the judge is known to be anti-calibrated — see ``got_contract_reexpand_enabled``'s
F33 note). The rule here is *derived* from held-out ``(confidence-sequence, eventual-label)``
pairs with a quantified false-stop rate, following E-valuator (arXiv 2512.03109; verified
against the primary text in ``RESEARCH_NOTES.md``): one calibration example is a whole
trajectory ``(S₁..S_T, Y)``, never a per-step label.

The documented simplification vs. E-valuator
--------------------------------------------
E-valuator fits a *separate logistic-regression classifier per timestep*, converts it to a
density ratio via Bayes' rule, and sets the stop threshold from an order statistic of a
disjoint threshold-calibration split (a distribution-free PAC procedure). That needs n in
the high hundreds *per split*; this repo has 354 usable regular-roster trajectories total.
So:

* **The classifier is replaced by one scalar prefix statistic** (``running_min`` /
  ``running_mean`` / ``last`` — see ``STATISTICS``), chosen on the calibration split only.
  With ~250 calibration trajectories a per-timestep logistic fit would be fitting noise.
* **The PAC order statistic is replaced by an exact one-sided Clopper–Pearson lower bound**
  on the stop-set precision. Accept a threshold τ at timestep t only if, with confidence
  ``1 - CERTIFICATION_DELTA``, the true probability that a trajectory stopped there
  eventually succeeds is at least the target. This is the same guarantee shape (a
  distribution-free, finite-sample bound on the false-stop rate) computed the way small
  samples allow.
* **Multiplicity is handled by Bonferroni** over the timesteps tested, so the family-wise
  confidence is still ``1 - CERTIFICATION_DELTA``.
* **Fitting is sequential-consistent**: the threshold for timestep t is certified only on
  the trajectories that reached t *without already having been stopped* — exactly the
  conditional distribution the rule faces online, so the per-timestep certificates compose.
* **When nothing certifies, nothing stops.** ``fit_thresholds`` returns no threshold for a
  timestep it cannot certify, which reproduces E-valuator's own degenerate case (``c_α = ∞``,
  never reject, zero power) rather than shipping an uncertified guess.

Beyond ``MAX_TIMESTEP`` the ``MAX_TIMESTEP`` rule is reused unchanged — E-valuator's
"ratio held constant beyond ``T_max``" convention.

The engine reads the fitted rule from the versioned artifact
``app/confidence_early_exit_calibration.json`` (written by
``scripts/calibrate_confidence_early_exit.py``); ``GoTOperations.should_exit_early``
applies it. Nothing here does I/O except :func:`load_rule`.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_logger = logging.getLogger(__name__)


# --- named calibration constants (the knobs of the statistical procedure) -------------
#: The pass bar the suite itself uses (``idea_tests/README.md``); the trajectory label is
#: ``validation.overall_score >= LABEL_THRESHOLD``.
LABEL_THRESHOLD = 0.75
#: The false-stop rate we would *like* to bound: stopping is allowed only where at least
#: this fraction of comparable calibration trajectories eventually passed. Kept at the
#: literature-aligned 0.90 even though this corpus cannot certify it (see
#: ``TARGET_STOP_PRECISION_LADDER``) so the shortfall is visible in the artifact rather
#: than hidden in a conveniently-lowered constant.
PREFERRED_TARGET_STOP_PRECISION = 0.90
#: Descending rungs tried by the calibration driver. The artifact ships the highest rung the
#: corpus can actually certify with non-vacuous coverage and records whether the preferred
#: rung was met. A rung is a *bound on the false-stop rate*: target 0.65 == false-stop rate
#: <= 0.35 with confidence ``1 - CERTIFICATION_DELTA``.
TARGET_STOP_PRECISION_LADDER = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
#: Family-wise error level of the Clopper-Pearson certificates (0.05 -> 95% confidence).
CERTIFICATION_DELTA = 0.05
#: A rule that fires on fewer than this fraction of calibration trajectories saves no real
#: compute; the driver treats such a rung as not certified.
MIN_STOP_COVERAGE = 0.05
#: Selectivity guard. A threshold that fires on more than this fraction of the trajectories
#: reaching a timestep is not a *high-confidence* rule, it is an unconditional stop that
#: happens to inherit the base rate — which is exactly what a loose target would certify if
#: the confidence signal carried no information (the degenerate ``tau = 0`` solution).
MAX_STEP_STOP_FRACTION = 0.5
#: Longest judged prefix modelled; beyond it the ``MAX_TIMESTEP`` rule is reused.
MAX_TIMESTEP = 8
#: Never stop before this many judged steps, regardless of what the data says — one lucky
#: high score on the first leaf is not a trajectory.
MIN_TIMESTEP = 2
#: Fraction of trajectories used to FIT; the remainder is held out and only ever measured.
CALIBRATION_SPLIT = 0.7
#: Candidate prefix statistics. Declared up front (a 3-way choice, resolved on the
#: calibration split) so the selection is transparent rather than a data-dredged feature.
STATISTICS: Tuple[str, ...] = ("running_min", "running_mean", "last")
#: Model-name markers of the "regular" roster the rule is calibrated for.
MODEL_ROSTER: Tuple[str, ...] = ("gpt-4.1-nano", "gpt-5-mini", "deepseek", "sonnet", "gemini")
#: Markers of runs deliberately EXCLUDED (badmodel-lab tiny local models have a very
#: different confidence/failure distribution and would miscalibrate the regular engine).
EXCLUDED_MARKERS: Tuple[str, ...] = ("bmladapt", "qwen", "localagent", "ollama")

#: Artifact filename, resolved next to the engine package (``app/``).
CALIBRATION_FILENAME = "confidence_early_exit_calibration.json"
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / CALIBRATION_FILENAME
#: Bumped whenever the artifact's shape changes; a rule with an unknown version is ignored.
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class LabelledTrajectory:
    """One calibration example: a whole confidence sequence plus its eventual label.

    Trajectory-level, never per-step — this is E-valuator's ``(S, Y)``.
    """

    confidences: Tuple[float, ...]
    label: int
    source: str = ""


@dataclass(frozen=True)
class EarlyExitDecision:
    """Why the rule did (or did not) stop, so the engine can log an honest reason."""

    stop: bool
    reason: str
    timestep: int = 0
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass(frozen=True)
class EarlyExitRule:
    """A fitted stopping rule: one threshold per judged-step count."""

    statistic: str
    thresholds: Mapping[int, float]
    target_stop_precision: float
    min_timestep: int = MIN_TIMESTEP
    max_timestep: int = MAX_TIMESTEP

    def threshold_for(self, timestep: int) -> Optional[float]:
        """Threshold at ``timestep``; beyond ``max_timestep`` the last modelled rule holds."""
        if timestep < self.min_timestep:
            return None
        return self.thresholds.get(min(int(timestep), self.max_timestep))

    def decide(self, confidences: Sequence[float], margin: float = 0.0) -> EarlyExitDecision:
        """Apply the rule to the judged-confidence prefix observed so far."""
        seq = [float(c) for c in confidences if isinstance(c, (int, float))]
        timestep = len(seq)
        if timestep < self.min_timestep:
            return EarlyExitDecision(
                False, f"only {timestep} judged step(s) < min_timestep {self.min_timestep}", timestep
            )
        threshold = self.threshold_for(timestep)
        if threshold is None:
            return EarlyExitDecision(False, f"no certified threshold at timestep {timestep}", timestep)
        value = prefix_statistic(seq, self.statistic)
        bar = threshold + float(margin)
        if value >= bar:
            return EarlyExitDecision(
                True,
                (
                    f"{self.statistic}={value:.3f} >= {threshold:.3f}+{float(margin):.3f} at "
                    f"timestep {timestep} (certified stop precision >= {self.target_stop_precision:.2f})"
                ),
                timestep,
                value,
                bar,
            )
        return EarlyExitDecision(
            False, f"{self.statistic}={value:.3f} < {bar:.3f} at timestep {timestep}", timestep, value, bar
        )


def prefix_statistic(confidences: Sequence[float], statistic: str) -> float:
    """Collapse a confidence prefix to the one scalar the rule thresholds.

    ``running_min`` = "no step so far was distrusted", ``running_mean`` = "the trajectory
    reads well on average", ``last`` = "the most recent step read well".
    """
    seq = [float(c) for c in confidences]
    if not seq:
        raise ValueError("prefix_statistic needs at least one confidence")
    if statistic == "running_min":
        return min(seq)
    if statistic == "running_mean":
        return sum(seq) / len(seq)
    if statistic == "last":
        return seq[-1]
    raise ValueError(f"unknown statistic {statistic!r} (expected one of {STATISTICS})")


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binomial_tail_ge(successes: int, trials: int, p: float) -> float:
    """``P(X >= successes)`` for ``X ~ Binomial(trials, p)``, computed in log space."""
    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return sum(math.exp(_log_binom_pmf(k, trials, p)) for k in range(successes, trials + 1))


def clopper_pearson_lower(successes: int, trials: int, delta: float) -> float:
    """Exact one-sided Clopper-Pearson lower bound on a binomial proportion.

    The largest ``p`` with ``P(X >= successes | trials, p) <= delta`` — i.e. with confidence
    ``1 - delta`` the true success probability is at least the returned value. This is the
    small-sample stand-in for E-valuator's PAC order-statistic threshold: same
    distribution-free finite-sample guarantee, computed exactly instead of from calibration
    maxima we do not have enough of.
    """
    if trials <= 0 or successes <= 0:
        return 0.0
    successes = min(int(successes), int(trials))
    if successes >= trials:
        # Closed form: P(X >= n) = p**n = delta.
        return float(delta) ** (1.0 / trials)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if binomial_tail_ge(successes, trials, mid) > delta:
            hi = mid
        else:
            lo = mid
    return lo


def certify_threshold(
    samples: Sequence[Tuple[float, int]],
    target: float,
    delta: float,
    max_stop_fraction: float = MAX_STEP_STOP_FRACTION,
) -> Optional[float]:
    """Smallest threshold whose stop-set is certified at ``target`` precision, else ``None``.

    ``samples`` are ``(statistic_value, label)`` pairs. Candidate thresholds are the values
    actually observed (every other threshold induces an identical stop-set). Smallest wins
    because it maximises coverage — the Clopper-Pearson bound, not the grid, is what keeps a
    thinly-supported threshold out. ``max_stop_fraction`` rejects the degenerate wide
    thresholds (notably ``tau`` at the bottom of the range, which stops everything and merely
    inherits the base rate); see :data:`MAX_STEP_STOP_FRACTION`.
    """
    if not samples:
        return None
    limit = max_stop_fraction * len(samples)
    for tau in sorted({value for value, _ in samples}):
        stopped = [label for value, label in samples if value >= tau]
        if not stopped or len(stopped) > limit:
            continue
        if clopper_pearson_lower(sum(stopped), len(stopped), delta) >= target:
            return float(tau)
    return None


def best_certifiable_precision(
    samples: Sequence[Tuple[float, int]],
    delta: float,
    max_stop_fraction: float = MAX_STEP_STOP_FRACTION,
) -> Tuple[float, Optional[float]]:
    """Highest stop precision any admissible threshold could certify, and that threshold.

    The diagnostic that turns "no rung certified" from an opaque verdict into a number: it
    reports the actual ceiling of this confidence signal at this timestep, so the artifact
    records *how far short* the corpus falls rather than merely that it does.
    """
    if not samples:
        return 0.0, None
    limit = max_stop_fraction * len(samples)
    best = (0.0, None)
    for tau in sorted({value for value, _ in samples}):
        stopped = [label for value, label in samples if value >= tau]
        if not stopped or len(stopped) > limit:
            continue
        bound = clopper_pearson_lower(sum(stopped), len(stopped), delta)
        if bound > best[0]:
            best = (bound, float(tau))
    return best


def fit_thresholds(
    trajectories: Sequence[LabelledTrajectory],
    statistic: str,
    target: float,
    delta: float = CERTIFICATION_DELTA,
    min_timestep: int = MIN_TIMESTEP,
    max_timestep: int = MAX_TIMESTEP,
) -> Dict[int, float]:
    """Derive per-timestep thresholds, sequential-consistently and Bonferroni-corrected.

    Timesteps are visited in order; a trajectory already stopped by an earlier timestep's
    threshold is removed before the next one is certified, so each certificate is conditioned
    on exactly what the online rule sees ("reached t, not yet stopped"). ``delta`` is split
    evenly across the timesteps tested, keeping the family-wise confidence at ``1 - delta``.
    Timesteps that certify nothing are simply absent from the result (never stop there).
    """
    tested = max(1, int(max_timestep) - int(min_timestep) + 1)
    per_step_delta = float(delta) / tested
    remaining = list(trajectories)
    thresholds: Dict[int, float] = {}
    for timestep in range(int(min_timestep), int(max_timestep) + 1):
        samples = [
            (prefix_statistic(tr.confidences[:timestep], statistic), tr.label)
            for tr in remaining
            if len(tr.confidences) >= timestep
        ]
        tau = certify_threshold(samples, target, per_step_delta)
        if tau is None:
            continue
        thresholds[timestep] = tau
        remaining = [
            tr
            for tr in remaining
            if not (
                len(tr.confidences) >= timestep
                and prefix_statistic(tr.confidences[:timestep], statistic) >= tau
            )
        ]
    return thresholds


def evaluate_rule(
    trajectories: Sequence[LabelledTrajectory],
    rule: EarlyExitRule,
    margin: float = 0.0,
) -> Dict[str, Any]:
    """Replay ``rule`` over whole trajectories and report what it would actually have done.

    This — not the per-timestep fit — is the number to quote: it measures the realised
    false-stop rate of the composed sequential rule on whatever split it is given.
    """
    total = len(trajectories)
    stops = 0
    false_stops = 0
    saved_steps = 0
    total_steps = 0
    stop_timesteps: List[int] = []
    for tr in trajectories:
        total_steps += len(tr.confidences)
        for timestep in range(1, len(tr.confidences) + 1):
            decision = rule.decide(tr.confidences[:timestep], margin=margin)
            if not decision.stop:
                continue
            stops += 1
            stop_timesteps.append(timestep)
            saved_steps += len(tr.confidences) - timestep
            if not tr.label:
                false_stops += 1
            break
    return {
        "n": total,
        "base_rate": (sum(t.label for t in trajectories) / total) if total else 0.0,
        "stops": stops,
        "coverage": (stops / total) if total else 0.0,
        "false_stops": false_stops,
        "false_stop_rate": (false_stops / stops) if stops else 0.0,
        "stop_precision": ((stops - false_stops) / stops) if stops else 0.0,
        "judged_steps_saved": saved_steps,
        "judged_steps_saved_fraction": (saved_steps / total_steps) if total_steps else 0.0,
        "mean_stop_timestep": (sum(stop_timesteps) / len(stop_timesteps)) if stop_timesteps else 0.0,
    }


def rule_from_artifact(payload: Mapping[str, Any]) -> Optional[EarlyExitRule]:
    """Rebuild an :class:`EarlyExitRule` from a calibration artifact mapping.

    Returns ``None`` for a missing/unknown-version/thresholdless artifact so a bad or absent
    file can only ever mean "never stop early", never a crash or an uncertified stop.
    """
    if not isinstance(payload, Mapping):
        return None
    if int(payload.get("version", 0)) != ARTIFACT_VERSION:
        return None
    statistic = payload.get("statistic")
    if statistic not in STATISTICS:
        return None
    raw = payload.get("thresholds") or {}
    if not isinstance(raw, Mapping) or not raw:
        return None
    thresholds: Dict[int, float] = {}
    for key, value in raw.items():
        try:
            thresholds[int(key)] = float(value)
        except (TypeError, ValueError):
            return None
    if not thresholds:
        return None
    return EarlyExitRule(
        statistic=str(statistic),
        thresholds=thresholds,
        target_stop_precision=float(payload.get("target_stop_precision", PREFERRED_TARGET_STOP_PRECISION)),
        min_timestep=int(payload.get("min_timestep", MIN_TIMESTEP)),
        max_timestep=int(payload.get("max_timestep", MAX_TIMESTEP)),
    )


_RULE_CACHE: Dict[str, Optional[EarlyExitRule]] = {}


def load_rule(path: Optional[Path] = None) -> Optional[EarlyExitRule]:
    """Load (and memoise) the shipped calibration artifact. Never raises."""
    target = Path(path) if path is not None else CALIBRATION_PATH
    key = str(target)
    if key in _RULE_CACHE:
        return _RULE_CACHE[key]
    rule: Optional[EarlyExitRule] = None
    try:
        with open(target, "r", encoding="utf-8") as handle:
            rule = rule_from_artifact(json.load(handle))
    except FileNotFoundError:
        _logger.warning(f"[EARLY-EXIT] no calibration artifact at {target}; early exit disabled")
    except Exception as exc:  # noqa: BLE001 — a broken artifact must never break a run
        _logger.warning(f"[EARLY-EXIT] unreadable calibration artifact {target}: {exc}")
    _RULE_CACHE[key] = rule
    return rule


def clear_rule_cache() -> None:
    """Drop the memoised artifact (tests that write their own artifact use this)."""
    _RULE_CACHE.clear()
