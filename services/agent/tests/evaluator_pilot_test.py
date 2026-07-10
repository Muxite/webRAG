"""Offline tests for the E-valuator calibration pilot's pure-stdlib helpers.

Only the dependency-free extraction/summary/split logic is covered here — the
pandas/scikit-learn/e-valuator machinery is intentionally out of the main venv
(mirrors the ConSol precedent) and is exercised in a throwaway pilot venv, not
in the offline suite.
"""
from __future__ import annotations

import math

from agent.app.testing import evaluator_pilot as ep


def _run_json(greps, overall_passed):
    return {"validation": {"grep_validations": greps, "overall_passed": overall_passed}}


def test_extract_run_orders_scores_and_labels():
    data = _run_json(
        [
            {"check": "visit_count", "score": 1.0, "passed": True},
            {"check": "author", "score": 0.0, "passed": False},
            {"check": "citation", "score": 0.5, "passed": False},
        ],
        overall_passed=True,
    )
    scores, names, solved = ep.extract_run(data)
    assert scores == [1.0, 0.0, 0.5]
    assert names == ["visit_count", "author", "citation"]
    assert solved == 1


def test_extract_run_missing_score_defaults_zero_and_fail_label():
    data = _run_json(
        [{"check": "a", "score": None}, {"check": "b"}],
        overall_passed=False,
    )
    scores, names, solved = ep.extract_run(data)
    assert scores == [0.0, 0.0]
    assert solved == 0


def test_extract_run_returns_none_without_grep_sequence():
    assert ep.extract_run({"validation": {}}) is None
    assert ep.extract_run({"validation": {"grep_validations": []}}) is None
    assert ep.extract_run({}) is None


def test_min_calibration_successes_matches_paper_formula():
    # ceil(log delta / log(1 - alpha)); e.g. alpha=0.1, delta=0.1 -> 22
    assert ep.min_calibration_successes(0.1, 0.1) == math.ceil(math.log(0.1) / math.log(0.9))
    assert ep.min_calibration_successes(0.1, 0.1) == 22
    # Tighter budget needs many more successes.
    assert ep.min_calibration_successes(0.09, 0.01) == 49


def test_summarize_counts_success_fail_and_lengths():
    trajs = [
        ep.Trajectory("t1", "050", "m", "v", [1.0, 1.0], 1),
        ep.Trajectory("t2", "051", "m", "v", [0.0, 1.0, 0.0], 0),
        ep.Trajectory("t3", "052", "m", "v", [1.0], 0),
    ]
    s = ep.summarize(trajs)
    assert s == {"n": 3, "success": 1, "fail": 2, "lengths": [1, 2, 3]}


def test_stratified_split_keeps_both_classes_in_each_slice():
    trajs = [ep.Trajectory(f"s{i}", "t", "m", "v", [1.0], 1) for i in range(10)]
    trajs += [ep.Trajectory(f"f{i}", "t", "m", "v", [0.0], 0) for i in range(10)]
    calib, held = ep.stratified_split(trajs, calib_frac=0.5, seed=7)
    assert len(calib) == 10 and len(held) == 10
    for slice_ in (calib, held):
        labels = {t.solved for t in slice_}
        assert labels == {0, 1}
    # Disjoint by trajectory id.
    assert not ({t.traj_id for t in calib} & {t.traj_id for t in held})
