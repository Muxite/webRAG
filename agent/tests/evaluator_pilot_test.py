"""Offline tests for the E-valuator calibration pilot's pure-stdlib helpers.

Only the dependency-free extraction/summary/split logic is covered here — the
pandas/scikit-learn/e-valuator machinery is intentionally out of the main venv
(mirrors the ConSol precedent) and is exercised in a throwaway pilot venv, not
in the offline suite.
"""
from __future__ import annotations

import math
import random

import pytest

from agent.app.testing import evaluator_pilot as ep


def _run_json(greps, overall_passed):
    return {"validation": {"grep_validations": greps, "overall_passed": overall_passed}}


def _conf_json_obs(sequence, overall_passed, kinds=None):
    """Result JSON carrying the decorrelated confidence trace under observability."""
    trace = [
        {"step": i, "node_id": f"n{i}", "kind": (kinds[i] if kinds else "visit"), "confidence": c}
        for i, c in enumerate(sequence)
    ]
    return {
        "execution": {"observability": {"step_confidence": {"sequence": list(sequence), "trace": trace}}},
        "validation": {"overall_passed": overall_passed},
    }


def _conf_json_raw(sequence, overall_passed):
    """Result JSON carrying the confidence trace only under the raw engine output."""
    return {
        "execution": {"output": {"step_confidences": [
            {"step": i, "node_id": f"n{i}", "kind": "search", "confidence": c}
            for i, c in enumerate(sequence)
        ]}},
        "validation": {"overall_passed": overall_passed},
    }


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


def test_extract_run_confidence_reads_observability_sequence():
    data = _conf_json_obs([0.9, 0.2, 0.7], overall_passed=True, kinds=["search", "visit", "visit"])
    scores, kinds, solved = ep.extract_run_confidence(data)
    assert scores == [0.9, 0.2, 0.7]
    assert kinds == ["search", "visit", "visit"]
    assert solved == 1


def test_extract_run_confidence_falls_back_to_raw_output():
    data = _conf_json_raw([0.4, 0.6], overall_passed=False)
    scores, kinds, solved = ep.extract_run_confidence(data)
    assert scores == [0.4, 0.6]
    assert kinds == ["search", "search"]
    assert solved == 0


def test_extract_run_confidence_returns_none_when_absent():
    assert ep.extract_run_confidence({"execution": {}}) is None
    assert ep.extract_run_confidence({}) is None
    # A grep-only run (no confidence trace) yields nothing on the confidence path.
    assert ep.extract_run_confidence(_run_json([{"check": "a", "score": 1.0}], True)) is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        ep.load_trajectories("nope.csv", "m", source="bogus")


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


def _synthetic_decorrelated_trajectories(n=400, seed=1):
    """Trajectories whose per-step scores are a NOISY (partially-informative) function of
    the label — the exact regime the grep substrate could not reach (there the scores
    *determined* the label, forcing FAR=0). Failures trend low-with-noise, successes
    trend high-with-noise, with heavy overlap so no per-step threshold is a clean
    separator. This is what a real per-step confidence judge looks like.
    """
    rng = random.Random(seed)
    trajs = []
    for i in range(n):
        solved = 1 if rng.random() < 0.5 else 0
        base = 0.62 if solved else 0.42  # overlapping means, noisy per step
        scores = [min(1.0, max(0.0, rng.gauss(base, 0.25))) for _ in range(6)]
        trajs.append(ep.Trajectory(f"t{i}", "synthetic", "m", "confidence", scores, solved))
    return trajs


def test_synthetic_decorrelated_pilot_yields_nontrivial_far():
    """End-to-end proof the pilot wiring produces a REAL signal on a decorrelated substrate.

    Requires the pilot-only ``e-valuator``/pandas deps (kept out of the main venv, mirroring
    the ConSol precedent), so it skips in the default offline suite and runs in the throwaway
    pilot venv. The assertion is the whole point of this task: unlike the grep substrate
    (which drove FAR to a trivial 0.000 because the scores computed the label), a decorrelated
    signal must produce a NON-trivial held-out false-alarm rate — bounded by, but not pinned to
    exactly zero at, the target alpha.
    """
    pytest.importorskip("evaluator")
    pytest.importorskip("pandas")

    trajs = _synthetic_decorrelated_trajectories(n=400, seed=1)
    calib, held = ep.stratified_split(trajs, calib_frac=0.5, seed=42)

    from evaluator import EValuator

    alphas = [0.1, 0.2]
    ev = EValuator(mt_variant="PAC", alphas=list(alphas), random_state=42)
    ev.fit(ep.build_frame(calib))
    applied = ev.apply(ep.build_frame(held))

    fars = [ep.false_alarm_rate(applied, a) for a in alphas]
    # Not the degenerate grep result: at least one alpha rejects a real, positive fraction of
    # held-out successes (FAR strictly > 0), while still respecting the PAC bound.
    assert any(r["false_alarm_rate"] > 0.0 for r in fars), (
        f"decorrelated substrate should yield a non-trivial FAR, got {[r['false_alarm_rate'] for r in fars]}"
    )
    for r in fars:
        assert r["false_alarm_rate"] <= r["alpha"] + 0.15, (
            f"FAR {r['false_alarm_rate']} should not blow past target alpha {r['alpha']}"
        )
