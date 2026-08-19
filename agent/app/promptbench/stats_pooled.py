"""The pooled primary statistic, and why the sign test could not be it.

v1's headline was a sign test over 5 models. Its p-value is quantised to powers of
one half::

    smallest attainable two-sided p with 5 models = 2 x (1/2)^5 = 0.0625

A perfect 5-of-5 sweep therefore *cannot* reach p < 0.05. v1's H1 landed at 0.062,
which is that floor and not a measurement. The design was capped before the first
call was placed, and no amount of extra items would have moved it, because items
were never the unit of the test.

WHAT CHANGED
------------
The unit of analysis becomes the ``(model, item)`` pair. Models are treated as
**fixed effects** -- this estimates the effect across *this roster*, not across
models in general, and the write-up says so rather than implying the larger claim.
The random sampling unit is the **task module**: items from one module share a
topic, a statement and an author, so they are not independent, and the bootstrap
resamples modules rather than items.

EQUAL MODEL WEIGHT, NOT CENTRING
--------------------------------
An early draft of this design said to centre each model's deltas on that model's
own mean before pooling. That is wrong in a way worth recording: centring within
model removes exactly the quantity being estimated, and the pooled mean would be
0.000 by construction, every time, for every arm.

What was actually wanted is that no model dominates through having more usable
cells than another -- transport errors and parse failures make the cell counts
unequal even though the design is balanced. ``model_balanced_mean`` therefore
averages the per-model means, giving each model equal weight regardless of how many
of its cells survived. On a fully balanced set it equals the simple pooled mean; it
diverges only where it should.

The per-model sign test is still computed and still printed, as a **consistency
display**. "Do all models agree on the direction" is a genuinely different question
from "how large is the effect", and it costs nothing to answer.
"""

from __future__ import annotations

import dataclasses
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_ITERS = 10000
DEFAULT_SEED = 20260819


@dataclasses.dataclass(frozen=True)
class DeltaRecord:
    model: str
    cluster: str
    item_id: str
    delta: float


def pooled_delta_records(
    rows: Sequence[Dict[str, object]], arm: str, baseline: str, family: str
) -> List[DeltaRecord]:
    """Paired arm-minus-baseline deltas for every (model, item) with both arms.

    Pairing is on ``item_id`` within a model, matching ``report.paired_deltas``.
    An item present in only one arm contributes nothing, rather than being scored
    against an assumed value.
    """
    by_arm: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    cluster_of: Dict[str, str] = {}
    for r in rows:
        if r.get("family") != family or "error" in r:
            continue
        variant = r.get("variant")
        if variant not in (arm, baseline):
            continue
        item_id = str(r.get("item_id"))
        by_arm[(str(r.get("model")), str(variant))][item_id] = bool(r.get("correct"))
        cluster_of[item_id] = str(r.get("cluster") or "")

    out: List[DeltaRecord] = []
    models = {m for m, _ in by_arm}
    for model in sorted(models):
        a = by_arm.get((model, arm), {})
        b = by_arm.get((model, baseline), {})
        for item_id in sorted(set(a) & set(b)):
            out.append(DeltaRecord(
                model=model,
                cluster=cluster_of.get(item_id, ""),
                item_id=item_id,
                delta=(1.0 if a[item_id] else 0.0) - (1.0 if b[item_id] else 0.0),
            ))
    return out


def model_balanced_mean(records: Sequence[DeltaRecord]) -> Optional[float]:
    """Mean over models of each model's mean delta -- equal weight per model."""
    if not records:
        return None
    per_model: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        per_model[r.model].append(r.delta)
    means = [sum(v) / len(v) for v in per_model.values() if v]
    return sum(means) / len(means) if means else None


def simple_mean(records: Sequence[DeltaRecord]) -> Optional[float]:
    if not records:
        return None
    return sum(r.delta for r in records) / len(records)


def _by_cluster(records: Sequence[DeltaRecord]) -> Dict[str, List[DeltaRecord]]:
    groups: Dict[str, List[DeltaRecord]] = defaultdict(list)
    for r in records:
        groups[r.cluster].append(r)
    return groups


def cluster_bootstrap_ci(
    records: Sequence[DeltaRecord],
    stat: Callable[[Sequence[DeltaRecord]], Optional[float]] = model_balanced_mean,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap interval, resampling whole task modules.

    A module is drawn with all of its rows across all models at once. Resampling
    individual rows would treat two items from the same module as independent
    evidence and shrink the interval toward a confidence nobody earned.
    """
    groups = _by_cluster(records)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(iters):
        sample: List[DeltaRecord] = []
        for _ in keys:
            sample.extend(groups[keys[rng.randrange(len(keys))]])
        value = stat(sample)
        if value is not None:
            draws.append(value)
    if not draws:
        return None
    draws.sort()
    lo = draws[max(0, int((alpha / 2) * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return lo, hi


def cluster_permutation_p(
    records: Sequence[DeltaRecord],
    stat: Callable[[Sequence[DeltaRecord]], Optional[float]] = model_balanced_mean,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> Optional[float]:
    """Two-sided p by flipping the sign of whole clusters.

    Under the null that shape does not matter, a module's deltas are equally likely
    to have come out reversed. Flipping module-at-a-time preserves the within-module
    correlation; flipping item-at-a-time would assume the independence this design
    exists to avoid assuming.

    The observed statistic is included in the null distribution (the ``+1`` terms),
    so the smallest reportable p is ``1/(iters+1)`` rather than 0 -- a permutation
    test cannot certify a p of exactly zero.
    """
    groups = _by_cluster(records)
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    observed = stat(records)
    if observed is None:
        return None
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(iters):
        flipped: List[DeltaRecord] = []
        for key in keys:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            for r in groups[key]:
                flipped.append(dataclasses.replace(r, delta=sign * r.delta))
        value = stat(flipped)
        if value is not None and abs(value) >= abs(observed) - 1e-12:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (iters + 1)


def per_model_directions(records: Sequence[DeltaRecord]) -> Dict[str, float]:
    """Each model's mean delta -- the input to the consistency display."""
    per_model: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        per_model[r.model].append(r.delta)
    return {m: sum(v) / len(v) for m, v in per_model.items() if v}


def pooled_report(
    rows: Sequence[Dict[str, object]],
    arm: str,
    baseline: str,
    family: str,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """The primary estimate for one (family, arm) contrast."""
    records = pooled_delta_records(rows, arm, baseline, family)
    ci = cluster_bootstrap_ci(records, iters=iters, seed=seed)
    directions = per_model_directions(records)
    positive = sum(1 for v in directions.values() if v > 0)
    negative = sum(1 for v in directions.values() if v < 0)
    return {
        "family": family,
        "arm": arm,
        "baseline": baseline,
        "n_pairs": len(records),
        "n_clusters": len({r.cluster for r in records}),
        "n_models": len(directions),
        "mean_delta": model_balanced_mean(records),
        "pooled_mean_delta": simple_mean(records),
        "ci95": ci,
        "ci_excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0)),
        "ci_halfwidth": (ci[1] - ci[0]) / 2.0 if ci else None,
        "permutation_p": cluster_permutation_p(records, iters=iters, seed=seed),
        # Consistency display, not the headline.
        "per_model_delta": directions,
        "direction_pos_neg": f"{positive}/{negative}",
    }
