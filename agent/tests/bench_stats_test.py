"""Known-answer unit tests for scripts/bench_stats.py.

scripts/bench_stats.py is a plain module (not a package), so it is imported by inserting
scripts/ onto sys.path -- mirroring how scripts/adaptive_ab_analyze.py itself imports it.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from bench_stats import (  # noqa: E402
    _f, _t975, _T975, ci95, cohens_d, holm, mean, paired_stats, signflip_p, stdev,
)


# ---- _f ----

def test_f_filters_non_numeric():
    assert _f([1, 2.5, "x", None, True, [1]]) == [1, 2.5, True]


# ---- mean / stdev ----

def test_mean_basic():
    assert mean([1, 2, 3]) == 2.0


def test_mean_empty_is_nan():
    assert math.isnan(mean([]))


def test_mean_ignores_non_numeric():
    assert mean([1, 2, "x", None]) == 1.5


def test_stdev_n1_is_zero():
    assert stdev([5.0]) == 0.0


def test_stdev_zero_variance():
    assert stdev([3.0, 3.0, 3.0]) == 0.0


def test_stdev_known_value():
    # sample stdev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.13809...
    xs = [2, 4, 4, 4, 5, 5, 7, 9]
    assert math.isclose(stdev(xs), 2.1380899352993950, rel_tol=1e-9)


# ---- _t975 / _T975 ----

def test_t975_table_lookup():
    assert _t975(1) == _T975[1] == 12.706
    assert _t975(30) == _T975[30] == 2.042


def test_t975_zero_df():
    assert _t975(0) == 0.0
    assert _t975(-1) == 0.0


def test_t975_beyond_table_uses_fallback():
    assert _t975(31) == 2.0
    assert _t975(100) == 2.0
    assert _t975(101) == 1.96


# ---- ci95 ----

def test_ci95_n1_is_zero():
    assert ci95([5.0]) == 0.0


def test_ci95_zero_variance_is_zero():
    assert ci95([1.0, 1.0, 1.0]) == 0.0


def test_ci95_all_zero_deltas():
    assert ci95([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_ci95_known_value():
    xs = [1.0, 2.0, 3.0]
    sd = stdev(xs)  # 1.0
    expected = _t975(2) * sd / math.sqrt(3)
    assert math.isclose(ci95(xs), expected, rel_tol=1e-12)


# ---- cohens_d ----

def test_cohens_d_identical_samples_is_zero():
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0.0


def test_cohens_d_n1_returns_zero():
    assert cohens_d([1.0], [1, 2, 3]) == 0.0
    assert cohens_d([1, 2, 3], [1.0]) == 0.0


def test_cohens_d_known_value():
    a, b = [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]
    # pooled sd = 1.0 (both groups have sd=1.0), mean diff = 3.0
    assert math.isclose(cohens_d(a, b), 3.0, rel_tol=1e-9)


def test_cohens_d_zero_pooled_sd_is_zero():
    assert cohens_d([2.0, 2.0], [2.0, 2.0]) == 0.0


# ---- paired_stats ----

def test_paired_stats_n0():
    assert paired_stats([]) == (0, 0.0, 0.0, 0.0)


def test_paired_stats_n1():
    n, m, ci, dz = paired_stats([2.0])
    assert n == 1
    assert m == 2.0
    assert ci == 0.0  # n<2 => no CI
    assert dz == 0.0  # sd is 0 for a single point => dz=0 (guarded, no div-by-zero)


def test_paired_stats_all_zero_deltas():
    n, m, ci, dz = paired_stats([0.0, 0.0, 0.0])
    assert n == 3
    assert m == 0.0
    assert ci == 0.0
    assert dz == 0.0


def test_paired_stats_zero_variance_nonzero_mean():
    # constant nonzero deltas: sd=0 => dz guarded to 0.0, ci also 0
    n, m, ci, dz = paired_stats([5.0, 5.0, 5.0])
    assert n == 3
    assert m == 5.0
    assert ci == 0.0
    assert dz == 0.0


def test_paired_stats_known_value():
    xs = [1.0, 2.0, 3.0]
    n, m, ci, dz = paired_stats(xs)
    assert n == 3
    assert m == 2.0
    sd = stdev(xs)
    assert math.isclose(ci, _t975(2) * sd / math.sqrt(3), rel_tol=1e-12)
    assert math.isclose(dz, 2.0 / sd, rel_tol=1e-12)


# ---- signflip_p ----

def test_signflip_p_empty():
    p, n = signflip_p([])
    assert p is None
    assert n == 0


def test_signflip_p_n1_always_significant_minimum():
    # A single nonzero delta: only 2 sign patterns, both give |mean|==obs, so p=1.0 always
    # (can never be "more extreme" than itself in either direction).
    p, n = signflip_p([3.0])
    assert n == 1
    assert p == 1.0


def test_signflip_p_all_zero_deltas():
    p, n = signflip_p([0.0, 0.0, 0.0])
    assert n == 3
    assert p == 1.0  # obs=0, every sign-flip pattern also sums to 0 => all >= obs


def test_signflip_p_exact_enumeration_consistent_sign():
    # All-positive deltas of equal magnitude: only the all-same-sign pattern (2 of 2**n) is as
    # extreme as observed => p = 2/2**n.
    xs = [1.0, 1.0, 1.0, 1.0]
    p, n = signflip_p(xs)
    assert n == 4
    assert math.isclose(p, 2 / 16, rel_tol=1e-9)


def test_signflip_p_uses_monte_carlo_above_18():
    xs = [1.0] * 19
    p, n = signflip_p(xs, iters=5000, seed=1)
    assert n == 19
    assert p is not None and 0.0 <= p <= 1.0


# ---- holm ----

def test_holm_empty():
    assert holm([]) == []


def test_holm_none_treated_as_one():
    adj = holm([None, 0.01])
    assert adj[0] == 1.0
    assert adj[1] == 0.02


def test_holm_single_pvalue_unchanged():
    assert holm([0.03]) == [0.03]


def test_holm_monotone_with_ties():
    # Two equal p-values in a family of 3: Holm multiplies the smaller-or-equal ranks by
    # decreasing factors (m, m-1, ...) and running-maxes to keep the result monotone.
    pvals = [0.01, 0.01, 0.20]
    adj = holm(pvals)
    # rank order (stable sort keeps input order for ties): idx0 (0.01) rank0 -> *3=0.03
    # idx1 (0.01) rank1 -> *2=0.02, but running max with prior (0.03) => 0.03
    # idx2 (0.20) rank2 -> *1=0.20, running max(0.20, 0.03) => 0.20
    assert adj[0] == 0.03
    assert adj[1] == 0.03
    assert adj[2] == 0.20
    # Monotone non-decreasing when read in ascending-input-p order among the tied+later entries.
    assert adj[0] <= adj[2]
    assert adj[1] <= adj[2]


def test_holm_all_significant_capped_at_one():
    adj = holm([0.9, 0.95, 0.99])
    assert all(a == 1.0 for a in adj)


def test_holm_order_preserved_length():
    pvals = [0.5, 0.01, None, 0.2]
    adj = holm(pvals)
    assert len(adj) == 4
