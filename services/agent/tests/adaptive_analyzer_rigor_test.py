"""Regression tests for the analyzer rigor fixes (scripts/adaptive_ab_analyze.py).

Pins the two headline-integrity fixes:
  * paired_deltas(missing="zero") scores a timed-out/missing cell as 0 over the UNION
    grid instead of intersection-dropping it (survivorship inflated the pilot delta ~15%).
  * _rep_key anchors rep(\\d+) to a token boundary so run-ids containing prep/grep/step
    don't false-match a replicate index (which silently mis-pairs / drops data).
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from adaptive_ab_analyze import (  # noqa: E402
    paired_deltas,
    _rep_key,
    holm,
    oaxaca_grounding_split,
    _dollars_per_solved,
)


def _rows(pairs):
    return [{"test_id": t, "rep": (1, 1), "score": s} for t, s in pairs]


def test_missing_zero_uses_full_union_grid_no_survivorship():
    # Adaptive (A) completed tasks 1,2,3; baseline (B) missing task 3 (timeout).
    A = _rows([("1", 0.8), ("2", 0.9), ("3", 1.0)])
    B = _rows([("1", 0.5), ("2", 0.4)])
    drop = paired_deltas(A, B, by="task", missing="drop")
    zero = paired_deltas(A, B, by="task", missing="zero")
    assert [t for t, _ in drop] == ["1", "2"]              # intersection drops task 3
    assert [t for t, _ in zero] == ["1", "2", "3"]         # union keeps it
    # The missing arm is scored 0 for that cell (delta = A(1.0) - 0.0).
    assert dict(zero)["3"] == pytest.approx(1.0)


def test_missing_zero_penalizes_the_arm_that_times_out():
    # The real survivorship case: ADAPTIVE times out on the hard task -> must be scored 0,
    # pulling the honest delta BELOW the intersection-dropped one.
    A = _rows([("1", 0.8), ("2", 0.9)])                    # adaptive missing task 3
    B = _rows([("1", 0.5), ("2", 0.4), ("3", 0.3)])
    drop_mean = sum(d for _, d in paired_deltas(A, B, by="task", missing="drop")) / 2
    zero = paired_deltas(A, B, by="task", missing="zero")
    zero_mean = sum(d for _, d in zero) / len(zero)
    assert dict(zero)["3"] == pytest.approx(-0.3)          # 0.0 - baseline 0.3
    assert zero_mean < drop_mean                           # honest grid is lower (no survivorship)


def test_rep_key_anchor_rejects_false_match():
    # 'prep7' must NOT yield a run-rep; a real '..._rep3' must. (model defaults None.)
    assert _rep_key("run_prep7_x", "f_r2.json") == (None, 2, None)
    assert _rep_key("barrage1_nano_baseline_rep3", "f_r3.json") == (3, 3, None)
    assert _rep_key("x_grep5_y", "f.json") == "f.json"     # no rep/r-suffix -> self-pair fallback


def test_rep_key_model_discriminator_prevents_collision():
    # Same (task,rep) across two models must NOT collide — the model is in the key.
    a = _rep_key("run_rep1", "f_r1.json", model="openai/gpt-4.1-nano")
    b = _rep_key("run_rep1", "f_r1.json", model="deepseek/deepseek-v4-flash")
    assert a != b


def test_holm_step_down_adjustment():
    # sorted 0.01,0.03,0.04 -> (3-0)*.01=.03, (3-1)*.03=.06, (3-2)*.04=.04 then monotone -> .06
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.06)  # the 0.04, pulled up by the monotone step
    assert adj[2] == pytest.approx(0.06)
    # a lone tiny p in a family of 4 is inflated by ~m at the front
    assert holm([0.001, 0.9, 0.9, 0.9])[0] == pytest.approx(0.004)
    assert holm([]) == []
    assert holm([None, 0.2]) == [pytest.approx(1.0), pytest.approx(0.4)]


def _grow(scores_visits):
    return [{"score": s, "visits": v, "usd": 0.01} for s, v in scores_visits]


def test_oaxaca_terms_sum_to_delta_raw_with_empirical_ungrounded():
    # baseline: 2 grounded (0.4,0.6 -> p=.5), 2 ungrounded (0.0,0.2 -> u=.1); g=.5
    B = _grow([(0.4, 1), (0.6, 1), (0.0, 0), (0.2, 0)])
    # adaptive: 3 grounded (0.7,0.8,0.9 -> p=.8), 1 ungrounded (0.1 -> u=.1); g=.75
    A = _grow([(0.7, 1), (0.8, 1), (0.9, 1), (0.1, 0)])
    d = oaxaca_grounding_split(A, B)
    # empirical ungrounded mean is used (NOT the false ==0 assumption)
    assert d["ub"] == pytest.approx(0.1)
    assert d["ua"] == pytest.approx(0.1)
    # the three terms sum EXACTLY to Δraw (additive identity)
    total = d["reasoning"] + d["grounding_rate"] + d["ungrounded_residual"]
    assert total == pytest.approx(d["delta_raw"])
    assert d["reasoning"] == pytest.approx(0.15)         # 0.5*(0.8-0.5)
    assert d["grounding_rate"] == pytest.approx(0.20)    # 0.8*(0.75-0.5)


def test_dollars_per_solved():
    rows = _grow([(0.9, 1), (0.8, 1), (0.3, 0)])         # 2 solved (>=0.75), 3 * $0.01 spend
    assert _dollars_per_solved(rows) == pytest.approx(0.03 / 2)
    none_solved = _grow([(0.1, 0), (0.2, 0)])
    import math
    assert math.isnan(_dollars_per_solved(none_solved))
