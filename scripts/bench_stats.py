#!/usr/bin/env python3
"""Shared paired-A/B statistics helpers for webRAG benchmark analysis scripts.

Extracted (behavior-preserving) from scripts/adaptive_ab_analyze.py, which had grown the most
complete implementation of these helpers while every dated `scripts/analyze_*_2026082*.py`
script re-implemented its own ~10-line version. This module is the ONE place these functions
should live going forward; new analysis scripts should `from bench_stats import ...` rather than
copy-pasting. The dated historical scripts are left untouched (they are records of past runs).

Functions:
  mean, stdev              -- basic descriptive stats, NaN-safe on empty input
  _t975                    -- two-sided 95% Student-t multiplier by degrees of freedom
  ci95                     -- 95% CI half-width for a sample
  cohens_d                 -- unpaired (pooled-sd) Cohen's d between two samples
  paired_stats             -- (n, mean, ci95, cohen_dz) for a list of paired differences
  signflip_p                -- two-sided paired sign-flip permutation p-value
  holm                     -- Holm-Bonferroni step-down FWER correction, input order preserved
"""
import itertools
import math
import random

# Student-t two-sided .975 quantiles by degrees of freedom (df = n-1). For small n the
# normal z=1.96 understates the CI by ~5-20%; this table is the scipy-free correction.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t975(df):
    """Two-sided 95% t-multiplier for `df` degrees of freedom (→ z=1.96 as df→∞)."""
    if df <= 0:
        return 0.0
    if df in _T975:
        return _T975[df]
    return 2.0 if df <= 100 else 1.96


def _f(xs):
    return [x for x in xs if isinstance(x, (int, float))]


def mean(xs):
    xs = _f(xs)
    # NaN (not a fake 0.0) for an empty set so a HOLE — a cell with no data — prints as
    # `nan` and is visibly a hole, instead of masquerading as a real score/cost of 0.
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = _f(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    xs = _f(xs)
    return _t975(len(xs) - 1) * stdev(xs) / math.sqrt(len(xs)) if len(xs) >= 2 else 0.0


def cohens_d(a, b):
    a, b = _f(a), _f(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * stdev(a) ** 2 + (nb - 1) * stdev(b) ** 2) / (na + nb - 2))
    return (mean(a) - mean(b)) / sp if sp else 0.0


def signflip_p(deltas, iters=200000, seed=12345):
    """Two-sided paired sign-flip (permutation) p-value on paired differences.

    Under H0 (no arm effect) each paired Δ is exchangeable in sign, so we compare the observed
    |mean(Δ)| to the null distribution of |mean(±Δ)|. Exact enumeration for n<=18 (2**n signs),
    Monte-Carlo above. This is the right test for the interleaved paired A/B design and needs no
    scipy / t-distribution. Returns (p_value, n_pairs).
    """
    xs = _f(deltas)
    n = len(xs)
    if n == 0:
        return None, 0
    obs = abs(sum(xs)) / n
    if n <= 18:
        cnt = tot = 0
        for signs in itertools.product((1, -1), repeat=n):
            tot += 1
            if abs(sum(s * d for s, d in zip(signs, xs))) / n >= obs - 1e-12:
                cnt += 1
        return cnt / tot, n
    rng = random.Random(seed)
    cnt = 0
    for _ in range(iters):
        if abs(sum(d if rng.random() < 0.5 else -d for d in xs)) / n >= obs - 1e-12:
            cnt += 1
    return cnt / iters, n


def paired_stats(deltas):
    """(n, mean, ci95, cohen_dz) for a list of paired differences. dz = mean/sd (paired effect)."""
    xs = _f(deltas)
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    m = sum(xs) / n
    sd = stdev(xs)
    ci = _t975(n - 1) * sd / math.sqrt(n) if n >= 2 else 0.0
    dz = m / sd if sd else 0.0
    return n, m, ci, dz


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, in the INPUT order.

    Controls the family-wise error rate across a family of related tests (e.g. the 4
    per-archetype comparisons, or a set of arm pairings) WITHOUT assuming independence —
    scanning k tests and quoting the smallest raw p inflates the false-positive rate (~19%
    under the null for k=4). A result is significant iff its adjusted p < 0.05. ``None``
    p-values are treated as 1.0 (not significant).
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (1.0 if pvals[i] is None else pvals[i]))
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = 1.0 if pvals[idx] is None else pvals[idx]
        running = max(running, min(1.0, (m - rank) * p))  # enforce monotone non-decreasing
        adj[idx] = running
    return adj
