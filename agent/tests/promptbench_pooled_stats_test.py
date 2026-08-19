"""The pooled primary statistic.

This is the module most able to be wrong in the direction that flatters the
hypothesis, and least able to look wrong while being so. Two failure modes are
pinned explicitly:

* a statistic that returns 0.000 for everything (the within-model *centring* bug
  an early draft of the design specified) reads as a well-powered null result;
* a bootstrap resampling ITEMS rather than task modules narrows every interval,
  turning "not significant" into "significant" with no change to the data.
"""

from __future__ import annotations

import random

import pytest

from agent.app.promptbench.stats_pooled import (
    DeltaRecord,
    cluster_bootstrap_ci,
    cluster_permutation_p,
    model_balanced_mean,
    per_model_directions,
    pooled_delta_records,
    pooled_report,
    simple_mean,
)


def synth(effect: float, n_clusters: int = 20, n_models: int = 5, seed: int = 1):
    """Rows where A2 is lifted over A1 by a known amount."""
    rng = random.Random(seed)
    rows = []
    for c in range(n_clusters):
        for i in range(2):
            item = f"it-{c}-{i}"
            for m in range(n_models):
                base = rng.random() < 0.5
                arm = base or (rng.random() < effect)
                for variant, ok in (("A1", base), ("A2", arm)):
                    rows.append({"family": "verify", "variant": variant, "model": f"m{m}",
                                 "item_id": item, "cluster": f"cl{c}", "correct": ok})
    return rows


# ---------------------------------------------------------------------------
# It must find a real effect, and must not invent one
# ---------------------------------------------------------------------------

def test_a_known_effect_is_recovered_with_an_interval_excluding_zero():
    r = pooled_report(synth(0.30), "A2", "A1", "verify", iters=1500)
    assert r["mean_delta"] > 0.05
    assert r["ci_excludes_zero"] is True
    assert r["permutation_p"] < 0.05


def test_a_zero_effect_yields_an_interval_containing_zero():
    r = pooled_report(synth(0.0), "A2", "A1", "verify", iters=1500)
    assert r["ci_excludes_zero"] is False
    assert r["permutation_p"] > 0.05


def test_the_estimate_is_not_identically_zero():
    """Guards the centring bug: centring each model's deltas on its own mean
    removes exactly the quantity being estimated, so every arm would report
    0.000 with a tight interval -- a null result that looks like a finding."""
    records = pooled_delta_records(synth(0.30), "A2", "A1", "verify")
    assert model_balanced_mean(records) != 0.0
    assert simple_mean(records) != 0.0


def test_the_pooled_test_beats_the_sign_tests_floor_on_the_same_five_models():
    """The reason for the whole change. A 5-model sign test cannot report below
    2*(1/2)^5 = 0.0625, so a perfect sweep still fails at alpha = 0.05."""
    r = pooled_report(synth(0.30), "A2", "A1", "verify", iters=3000)
    assert r["n_models"] == 5
    assert r["permutation_p"] < 0.0625


# ---------------------------------------------------------------------------
# Equal model weight
# ---------------------------------------------------------------------------

def test_equal_model_weight_survives_wildly_unequal_cell_counts():
    records = ([DeltaRecord("big", "c1", f"i{i}", 1.0) for i in range(100)]
               + [DeltaRecord("small", "c2", "j", -1.0)])
    assert simple_mean(records) > 0.9              # one model swamps the pool
    assert model_balanced_mean(records) == pytest.approx(0.0)


def test_on_a_balanced_design_both_estimators_agree():
    records = pooled_delta_records(synth(0.30), "A2", "A1", "verify")
    assert model_balanced_mean(records) == pytest.approx(simple_mean(records), abs=1e-12)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_the_bootstrap_resamples_modules_not_items():
    """Resampling items would treat two items from one module as independent
    evidence and shrink the interval toward unearned confidence. With every item
    in ONE cluster there is nothing to resample, so no interval is defined."""
    records = [DeltaRecord("m0", "only-cluster", f"i{i}", 1.0 if i % 2 else 0.0)
               for i in range(40)]
    assert cluster_bootstrap_ci(records, iters=200) is None
    assert cluster_permutation_p(records, iters=200) is None


def test_identical_clusters_produce_a_degenerate_interval_rather_than_a_narrow_one():
    """Twenty copies of the same module carry one module's worth of information."""
    records = [DeltaRecord("m0", f"cl{c}", f"i{c}", 1.0) for c in range(20)]
    lo, hi = cluster_bootstrap_ci(records, iters=500)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_permutation_p_can_never_be_exactly_zero():
    """The observed statistic is in its own null distribution, so the floor is
    1/(iters+1). A permutation test cannot certify p = 0."""
    records = [DeltaRecord("m0", f"cl{c}", f"i{c}", 1.0) for c in range(30)]
    p = cluster_permutation_p(records, iters=500)
    assert p == pytest.approx(1 / 501)


def test_results_are_reproducible_across_runs():
    a = pooled_report(synth(0.2), "A2", "A1", "verify", iters=400, seed=7)
    b = pooled_report(synth(0.2), "A2", "A1", "verify", iters=400, seed=7)
    assert a["ci95"] == b["ci95"] and a["permutation_p"] == b["permutation_p"]


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_items_present_in_only_one_arm_contribute_nothing():
    rows = [
        {"family": "verify", "variant": "A1", "model": "m", "item_id": "a",
         "cluster": "c", "correct": True},
        {"family": "verify", "variant": "A2", "model": "m", "item_id": "a",
         "cluster": "c", "correct": False},
        {"family": "verify", "variant": "A2", "model": "m", "item_id": "orphan",
         "cluster": "c", "correct": True},
    ]
    records = pooled_delta_records(rows, "A2", "A1", "verify")
    assert [r.item_id for r in records] == ["a"]
    assert records[0].delta == -1.0


def test_transport_errors_are_excluded_from_pairing():
    rows = [
        {"family": "verify", "variant": "A1", "model": "m", "item_id": "a",
         "cluster": "c", "correct": True},
        {"family": "verify", "variant": "A2", "model": "m", "item_id": "a",
         "cluster": "c", "error": "Timeout"},
    ]
    assert pooled_delta_records(rows, "A2", "A1", "verify") == []


def test_rows_from_another_family_never_leak_into_a_contrast():
    rows = synth(0.3) + [
        {"family": "select", "variant": v, "model": "m0", "item_id": "x",
         "cluster": "clX", "correct": True} for v in ("A1", "A2")
    ]
    records = pooled_delta_records(rows, "A2", "A1", "verify")
    assert all(r.cluster != "clX" for r in records)


def test_per_model_directions_feed_the_consistency_display():
    directions = per_model_directions(pooled_delta_records(synth(0.3), "A2", "A1", "verify"))
    assert len(directions) == 5
    assert all(isinstance(v, float) for v in directions.values())
