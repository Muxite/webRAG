"""Calibration metrics: does a stated confidence mean anything?

``promptbench`` v1 measured discrete accuracy, so it could not speak to the finding
that motivated the cycle. ``CONFIDENCE_JUDGE_MISCALIBRATION.md`` concerns a
*continuous* score correlated against run outcomes; this module supplies the
metrics that make that question askable on constructed items.

THE BAR IS INHERITED, NOT INVENTED
----------------------------------
That document establishes that two **LLM-free** graph statistics outrank every
confidence statistic the shipped judge produces:

    number of judged steps                        AUC 0.655
    fraction of steps that are content-bearing    AUC 0.634
    best confidence statistic (running_mean)      AUC 0.571

So 0.5 is not the bar. ``FREE_BASELINE_AUC`` is, and an arm below it has not earned
its LLM call. Using the document's own number keeps this cycle from grading itself
on an easier curve than the investigation it follows.

WHY THE DECOMPOSITION AND NOT JUST BRIER
----------------------------------------
Brier conflates two different failures. A judge that answers 0.5 to everything on a
balanced set is perfectly *reliable* and carries zero *resolution* -- it has
distinguished nothing while scoring respectably. That is close to what the
miscalibration document actually found (blind on 43% of the trace, near-useless on
the rest), and it is invisible in a single number. Murphy's decomposition splits it:

    Brier = reliability - resolution + uncertainty

Lower reliability is better; higher resolution is better; uncertainty is a property
of the item set, not of the model, and is reported so the other two can be read
against it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# The bar an LLM-produced confidence must clear, from
# CONFIDENCE_JUDGE_MISCALIBRATION.md §1: the best LLM-free structural statistic.
FREE_BASELINE_AUC = 0.655

# C_verbal's mapping, declared in source rather than inferred from the data.
# Inferring it would let the arm be tuned after the fact into whatever ordering
# best fits the outcomes, which is a different (and much weaker) claim.
VERBAL_BANDS: Dict[str, float] = {
    "certain": 0.95,
    "likely": 0.75,
    "unsure": 0.45,
    "guessing": 0.20,
}

DEFAULT_BINS = 10


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> Optional[float]:
    """Mean squared error of the stated probability against the outcome."""
    if not confidences or len(confidences) != len(correct):
        return None
    return sum((c - (1.0 if k else 0.0)) ** 2 for c, k in zip(confidences, correct)) / len(confidences)


def _bin_index(c: float, bins: int) -> int:
    # A confidence of exactly 1.0 belongs in the top bin, not in a bin past the end.
    return min(bins - 1, max(0, int(c * bins)))


def reliability_diagram(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
) -> List[Dict[str, float]]:
    buckets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    for c, k in zip(confidences, correct):
        buckets[_bin_index(c, bins)].append((c, k))
    out = []
    for b in sorted(buckets):
        rows = buckets[b]
        out.append({
            "bin": b,
            "lo": b / bins,
            "hi": (b + 1) / bins,
            "n": len(rows),
            "mean_confidence": sum(r[0] for r in rows) / len(rows),
            "observed_accuracy": sum(1 for r in rows if r[1]) / len(rows),
        })
    return out


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
        ) -> Optional[float]:
    """Expected calibration error: n-weighted |stated - observed| across bins."""
    if not confidences or len(confidences) != len(correct):
        return None
    n = len(confidences)
    return sum(
        (row["n"] / n) * abs(row["mean_confidence"] - row["observed_accuracy"])
        for row in reliability_diagram(confidences, correct, bins)
    )


def auc(confidences: Sequence[float], correct: Sequence[bool]) -> Optional[float]:
    """Probability a correct item is scored above an incorrect one.

    Computed by rank-sum with explicit tie handling: a judge emitting the SAME
    number for everything -- which is what the miscalibration document found on 43%
    of the trace -- must score exactly 0.5 here, not 0.0 or 1.0 depending on sort
    order. Undefined when either class is empty, and ``None`` says so rather than
    returning a number nobody can interpret.
    """
    pos = [c for c, k in zip(confidences, correct) if k]
    neg = [c for c, k in zip(confidences, correct) if not k]
    if not pos or not neg:
        return None
    order = sorted(range(len(confidences)), key=lambda i: confidences[i])
    ranks = [0.0] * len(confidences)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and confidences[order[j + 1]] == confidences[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0                      # average rank, 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    rank_sum = sum(r for r, k in zip(ranks, correct) if k)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def calibration_slope(confidences: Sequence[float], correct: Sequence[bool]) -> Optional[float]:
    """Slope of outcome regressed on stated confidence.

    1.0 is ideal. Below 1 is overconfidence (the score moves further than the
    outcome), above 1 underconfidence, and a NEGATIVE slope is the anti-calibration
    this whole line of investigation is about -- higher stated confidence predicting
    a worse outcome. ``None`` when confidence never varies, because a slope through
    a single x-value is not defined.
    """
    n = len(confidences)
    if n < 2:
        return None
    mean_c = sum(confidences) / n
    var_c = sum((c - mean_c) ** 2 for c in confidences)
    if var_c <= 1e-12:
        return None
    mean_y = sum(1.0 if k else 0.0 for k in correct) / n
    cov = sum((c - mean_c) * ((1.0 if k else 0.0) - mean_y) for c, k in zip(confidences, correct))
    return cov / var_c


def murphy_decomposition(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
) -> Dict[str, Optional[float]]:
    """Brier = reliability - resolution + uncertainty."""
    n = len(confidences)
    if n == 0:
        return {"reliability": None, "resolution": None, "uncertainty": None}
    base = sum(1 for k in correct if k) / n
    rows = reliability_diagram(confidences, correct, bins)
    reliability = sum(
        (r["n"] / n) * (r["mean_confidence"] - r["observed_accuracy"]) ** 2 for r in rows)
    resolution = sum(
        (r["n"] / n) * (r["observed_accuracy"] - base) ** 2 for r in rows)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": base * (1.0 - base),
    }


def summarize_calibration(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS
) -> Dict[str, object]:
    """Every metric for one cell, plus whether it cleared the inherited bar."""
    paired = [(c, k) for c, k in zip(confidences, correct) if c is not None]
    cs = [c for c, _ in paired]
    ks = [k for _, k in paired]
    a = auc(cs, ks)
    out: Dict[str, object] = {
        "n": len(cs),
        "mean_confidence": (sum(cs) / len(cs)) if cs else None,
        "base_rate": (sum(1 for k in ks if k) / len(ks)) if ks else None,
        "brier": brier(cs, ks),
        "ece": ece(cs, ks, bins),
        "auc": a,
        "slope": calibration_slope(cs, ks),
        "beats_free_baseline": (a is not None and a > FREE_BASELINE_AUC),
        "free_baseline_auc": FREE_BASELINE_AUC,
    }
    out.update(murphy_decomposition(cs, ks, bins))
    # A judge emitting one number has no resolution by construction; without this
    # flag its respectable-looking Brier reads as competence. This is the continuous
    # analogue of the degeneracy check that caught six cells in v1.
    out["degenerate"] = bool(cs) and (max(cs) - min(cs)) < 1e-9
    return out


def distinct_confidence_fraction(confidences: Sequence[float]) -> Optional[float]:
    if not confidences:
        return None
    return len(set(confidences)) / len(confidences)


def is_finite_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
