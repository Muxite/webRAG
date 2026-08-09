"""Unit tests for the A6 calibrated early-exit statistics and its calibration driver.

Two layers, both pure (no result files read at test time, no network, no LLM):

  * ``idea_policies/confidence_early_exit.py`` — the maths the engine and the driver share:
    the Clopper-Pearson bound, the prefix statistics, threshold certification (including the
    selectivity guard that keeps a degenerate "stop everything" threshold out), the
    sequential-consistent fit, rule replay, and artifact (de)serialisation.
  * ``scripts/calibrate_confidence_early_exit.py`` — the roster filter and the
    result-JSON -> ``(sequence, label)`` conversion, fed synthetic payloads.

Plus a pin on the artifact actually committed to the repo, so a silent regeneration that
started certifying an uncertified rule cannot slip in unnoticed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from agent.app.idea_policies.confidence_early_exit import (  # noqa: E402
    ARTIFACT_VERSION,
    CALIBRATION_PATH,
    CERTIFICATION_DELTA,
    MAX_STEP_STOP_FRACTION,
    MAX_TIMESTEP,
    MIN_TIMESTEP,
    EarlyExitRule,
    LabelledTrajectory,
    best_certifiable_precision,
    binomial_tail_ge,
    certify_threshold,
    clopper_pearson_lower,
    clear_rule_cache,
    evaluate_rule,
    fit_thresholds,
    load_rule,
    prefix_statistic,
    rule_from_artifact,
)


# --------------------------------------------------------------------------------------
# prefix statistics
# --------------------------------------------------------------------------------------


def test_prefix_statistics_collapse_the_sequence_as_documented():
    seq = [0.9, 0.4, 0.8]
    assert prefix_statistic(seq, "running_min") == pytest.approx(0.4)
    assert prefix_statistic(seq, "running_mean") == pytest.approx(0.7)
    assert prefix_statistic(seq, "last") == pytest.approx(0.8)


def test_prefix_statistic_rejects_unknown_statistic_and_empty_prefix():
    with pytest.raises(ValueError):
        prefix_statistic([0.5], "geometric_mean")
    with pytest.raises(ValueError):
        prefix_statistic([], "running_min")


# --------------------------------------------------------------------------------------
# Clopper-Pearson lower bound
# --------------------------------------------------------------------------------------


def test_clopper_pearson_all_successes_matches_the_closed_form():
    # P(X >= n | n, p) = p**n, so the bound solving p**n = delta is delta**(1/n).
    for n in (1, 5, 20):
        assert clopper_pearson_lower(n, n, 0.05) == pytest.approx(0.05 ** (1.0 / n), abs=1e-9)


def test_clopper_pearson_is_the_root_of_the_binomial_tail():
    bound = clopper_pearson_lower(8, 10, 0.05)
    assert binomial_tail_ge(8, 10, bound) == pytest.approx(0.05, abs=1e-6)


def test_clopper_pearson_is_conservative_and_monotone_in_evidence():
    # Same observed proportion, more evidence -> a tighter (higher) lower bound, always
    # below the point estimate.
    small = clopper_pearson_lower(8, 10, 0.05)
    large = clopper_pearson_lower(80, 100, 0.05)
    assert small < large < 0.8
    # No evidence -> no claim.
    assert clopper_pearson_lower(0, 0, 0.05) == 0.0
    assert clopper_pearson_lower(0, 10, 0.05) == 0.0


def test_clopper_pearson_small_sample_cannot_certify_a_high_target():
    """The property the whole design rests on: 3/3 successes prove nothing at 0.90."""
    assert clopper_pearson_lower(3, 3, 0.05) < 0.90
    # E-valuator's own minimum, n >= ceil(log delta / log(1-alpha)), for alpha=0.10:
    needed = math.ceil(math.log(0.05) / math.log(1 - 0.10))
    assert clopper_pearson_lower(needed, needed, 0.05) >= 0.90
    assert clopper_pearson_lower(needed - 1, needed - 1, 0.05) < 0.90


# --------------------------------------------------------------------------------------
# threshold certification
# --------------------------------------------------------------------------------------


def _samples(pairs):
    return [(float(v), int(y)) for v, y in pairs]


def test_certify_threshold_picks_the_smallest_certified_threshold():
    # Everything at/above 0.8 passed (30 of them); everything below is a coin flip.
    pairs = [(0.9, 1)] * 15 + [(0.8, 1)] * 15 + [(0.3, 1)] * 20 + [(0.3, 0)] * 20
    tau = certify_threshold(_samples(pairs), target=0.90, delta=0.05)
    assert tau == pytest.approx(0.8), "the smallest certified threshold maximises coverage"


def test_certify_threshold_returns_none_when_nothing_certifies():
    """E-valuator's ``c_alpha = infinity`` case: no evidence -> never stop."""
    pairs = [(0.95, 1), (0.95, 0)] * 25  # high confidence, coin-flip outcome
    assert certify_threshold(_samples(pairs), target=0.90, delta=0.05) is None
    assert certify_threshold([], target=0.5, delta=0.05) is None


def test_certify_threshold_rejects_a_thinly_supported_perfect_stop_set():
    # Three perfect examples look like precision 1.0 but certify nothing at 0.90.
    pairs = [(0.99, 1)] * 3 + [(0.2, 1)] * 30 + [(0.2, 0)] * 30
    assert certify_threshold(_samples(pairs), target=0.90, delta=0.05) is None


def test_selectivity_guard_rejects_the_degenerate_stop_everything_threshold():
    """A loose target must not certify "stop always" by inheriting the base rate."""
    pairs = [(0.9, 1)] * 60 + [(0.1, 1)] * 5 + [(0.1, 0)] * 35
    # Without the guard the smallest threshold (0.1) stops all 100 at precision 0.65.
    assert certify_threshold(_samples(pairs), target=0.55, delta=0.05, max_stop_fraction=1.0) == (
        pytest.approx(0.1)
    )
    # With it, only the genuinely selective 0.9 threshold (60% > 50% of the set) is... also
    # too wide, so nothing certifies -- the guard is doing real work.
    assert certify_threshold(_samples(pairs), target=0.55, delta=0.05, max_stop_fraction=0.5) is None


def test_best_certifiable_precision_reports_the_ceiling_and_its_threshold():
    pairs = [(0.9, 1)] * 30 + [(0.2, 1)] * 35 + [(0.2, 0)] * 35
    bound, tau = best_certifiable_precision(_samples(pairs), delta=0.05)
    assert tau == pytest.approx(0.9)
    assert 0.8 < bound < 1.0
    assert best_certifiable_precision([], delta=0.05) == (0.0, None)


# --------------------------------------------------------------------------------------
# sequential-consistent fitting
# --------------------------------------------------------------------------------------


def _traj(confidences, label, source=""):
    return LabelledTrajectory(tuple(confidences), label, source)


def test_fit_thresholds_certifies_a_clean_separable_signal():
    # 50 "easy" trajectories: high confidence throughout, all pass.
    # 80 "hard" ones: low confidence, coin flip. (n=50 all-success is the smallest run that
    # clears 0.90 under the Bonferroni-corrected bound: 0.05/7 ** (1/50) = 0.906.)
    easy = [_traj([0.95, 0.95, 0.95, 0.95], 1) for _ in range(50)]
    hard = [_traj([0.2, 0.2, 0.2, 0.2], i % 2) for i in range(80)]
    thresholds = fit_thresholds(easy + hard, "running_min", target=0.90)
    assert thresholds, "a cleanly separable signal must certify at least one timestep"
    assert min(thresholds) == MIN_TIMESTEP, "certification starts at the documented floor"
    assert all(0.2 < tau <= 0.95 for tau in thresholds.values())


def test_fit_thresholds_is_sequential_consistent():
    """A trajectory stopped at t=2 must not be re-counted when certifying t=3.

    Built so the t=2 rule stops every passing trajectory: whatever remains at t=3 is pure
    failure, so t=3 can certify nothing. A non-sequential fit would happily certify t=3
    from the already-stopped winners.
    """
    stopped_early = [_traj([0.95, 0.95, 0.95], 1) for _ in range(50)]
    survivors = [_traj([0.1, 0.1, 0.95], 0) for _ in range(80)]
    thresholds = fit_thresholds(stopped_early + survivors, "running_min", target=0.90)
    assert 2 in thresholds
    assert 3 not in thresholds, "t=3 must be certified only on trajectories not already stopped"


def test_fit_thresholds_returns_nothing_for_a_signal_that_predicts_nothing():
    noise = [_traj([0.95, 0.95, 0.95], i % 2) for i in range(200)]
    assert fit_thresholds(noise, "running_mean", target=0.90) == {}


def test_fit_thresholds_bonferroni_split_is_stricter_than_a_single_test():
    """The family-wise correction must actually bind, not just be documented."""
    # Sized so the pooled evidence certifies at delta=0.05 but not at delta=0.05/7.
    pairs = [_traj([0.9, 0.9], 1) for _ in range(11)] + [_traj([0.1, 0.1], 0) for _ in range(30)]
    single = certify_threshold(
        [(prefix_statistic(t.confidences[:2], "running_min"), t.label) for t in pairs],
        target=0.75,
        delta=CERTIFICATION_DELTA,
    )
    corrected = fit_thresholds(pairs, "running_min", target=0.75)
    assert single is not None
    assert corrected == {}, "Bonferroni across timesteps must reject what a single test accepts"


# --------------------------------------------------------------------------------------
# rule application / replay
# --------------------------------------------------------------------------------------


def test_rule_never_fires_below_the_minimum_timestep():
    rule = EarlyExitRule("running_min", {1: 0.5, 2: 0.5}, 0.9, min_timestep=2)
    assert rule.decide([0.99]).stop is False
    assert rule.decide([0.99, 0.99]).stop is True


def test_rule_holds_the_max_timestep_threshold_beyond_the_horizon():
    rule = EarlyExitRule("last", {MAX_TIMESTEP: 0.7}, 0.9, min_timestep=2, max_timestep=MAX_TIMESTEP)
    long_prefix = [0.9] * (MAX_TIMESTEP + 5)
    assert rule.threshold_for(MAX_TIMESTEP + 5) == pytest.approx(0.7)
    assert rule.decide(long_prefix).stop is True


def test_rule_does_not_fire_at_an_uncertified_timestep():
    rule = EarlyExitRule("last", {4: 0.7}, 0.9, min_timestep=2)
    assert rule.decide([0.99, 0.99]).stop is False
    assert rule.decide([0.99] * 4).stop is True


def test_margin_raises_the_bar_above_the_calibrated_threshold():
    rule = EarlyExitRule("running_min", {2: 0.80}, 0.9, min_timestep=2)
    assert rule.decide([0.82, 0.82], margin=0.0).stop is True
    assert rule.decide([0.82, 0.82], margin=0.05).stop is False
    decision = rule.decide([0.82, 0.82], margin=0.05)
    assert "0.850" in decision.reason or decision.threshold == pytest.approx(0.85)


def test_rule_ignores_non_numeric_entries_in_the_prefix():
    rule = EarlyExitRule("running_min", {2: 0.5}, 0.9, min_timestep=2)
    assert rule.decide([0.9, None, 0.9, "x"]).stop is True  # type: ignore[list-item]


def test_evaluate_rule_reports_realised_stop_precision_and_savings():
    rule = EarlyExitRule("running_min", {2: 0.8}, 0.9, min_timestep=2)
    trajectories = [
        _traj([0.9, 0.9, 0.9, 0.9], 1),  # stops at t=2, correct, 2 steps saved
        _traj([0.9, 0.9, 0.9, 0.9], 0),  # stops at t=2, FALSE stop, 2 steps saved
        _traj([0.1, 0.1, 0.1, 0.1], 1),  # never stops
    ]
    metrics = evaluate_rule(trajectories, rule)
    assert metrics["stops"] == 2
    assert metrics["false_stops"] == 1
    assert metrics["stop_precision"] == pytest.approx(0.5)
    assert metrics["false_stop_rate"] == pytest.approx(0.5)
    assert metrics["judged_steps_saved"] == 4
    assert metrics["mean_stop_timestep"] == pytest.approx(2.0)
    assert metrics["coverage"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------------------
# artifact (de)serialisation — a bad artifact may only ever mean "never stop"
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"version": ARTIFACT_VERSION + 99, "statistic": "last", "thresholds": {"2": 0.8}},
        {"version": ARTIFACT_VERSION, "statistic": "vibes", "thresholds": {"2": 0.8}},
        {"version": ARTIFACT_VERSION, "statistic": "last", "thresholds": {}},
        {"version": ARTIFACT_VERSION, "statistic": "last", "thresholds": {"2": "high"}},
        {"version": ARTIFACT_VERSION, "statistic": None, "thresholds": {}},
    ],
)
def test_rule_from_artifact_fails_closed(payload):
    assert rule_from_artifact(payload) is None


def test_rule_from_artifact_round_trips_a_valid_artifact():
    rule = rule_from_artifact(
        {
            "version": ARTIFACT_VERSION,
            "statistic": "running_mean",
            "thresholds": {"2": 0.7, "3": 0.65},
            "target_stop_precision": 0.85,
            "min_timestep": 2,
            "max_timestep": 6,
        }
    )
    assert rule is not None
    assert rule.statistic == "running_mean"
    assert rule.thresholds == {2: 0.7, 3: 0.65}
    assert rule.max_timestep == 6
    assert rule.decide([0.8, 0.8]).stop is True


def test_load_rule_returns_none_for_a_missing_or_corrupt_file(tmp_path):
    clear_rule_cache()
    assert load_rule(tmp_path / "nope.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_rule(corrupt) is None
    clear_rule_cache()


def test_load_rule_reads_and_memoises_a_written_artifact(tmp_path):
    clear_rule_cache()
    path = tmp_path / "cal.json"
    path.write_text(
        json.dumps(
            {"version": ARTIFACT_VERSION, "statistic": "last", "thresholds": {"2": 0.6}}
        ),
        encoding="utf-8",
    )
    rule = load_rule(path)
    assert rule is not None and rule.decide([0.7, 0.7]).stop is True
    path.unlink()
    assert load_rule(path) is rule, "the artifact is read once per path, not per decision"
    clear_rule_cache()


# --------------------------------------------------------------------------------------
# the artifact actually committed to the repo
# --------------------------------------------------------------------------------------


def test_committed_artifact_is_wellformed_and_documents_its_provenance():
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert payload["version"] == ARTIFACT_VERSION
    assert payload["n_trajectories"] >= 200, "E-valuator's own ablation puts the floor near n=200"
    assert payload["n_fit"] + payload["n_holdout"] == payload["n_trajectories"]
    assert payload["label_rule"] == "validation.overall_score >= 0.75"
    assert "bmladapt" in payload["excluded_markers"], "badmodel-lab runs must stay excluded"
    assert payload["certification_delta"] == CERTIFICATION_DELTA
    assert payload["min_timestep"] == MIN_TIMESTEP and payload["max_timestep"] == MAX_TIMESTEP
    assert payload["max_step_stop_fraction"] == MAX_STEP_STOP_FRACTION
    assert payload["generated_utc"] and payload["method"]
    assert payload["ladder"], "the ladder diagnostics must be recorded, certified or not"


def test_committed_artifact_certifies_nothing_so_the_engine_cannot_stop_early():
    """Pins the honest current state: this corpus does not support a stopping rule.

    The signal ceiling (0.553) barely clears the 0.511 base rate, far below the loosest
    ladder rung (0.65) let alone the preferred 0.90 -- so the shipped rule is empty and
    ``should_exit_early`` can never fire even when the flag is on. If a regeneration ever
    *does* certify a rung, this test must be updated deliberately (and the A6 docs with it),
    never silently.
    """
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert payload["thresholds"] == {}
    assert payload["statistic"] is None
    assert payload["preferred_target_met"] is False
    assert not any(rung["certified"] for rung in payload["ladder"])
    ceiling = payload["signal_ceiling"]["best_certified_precision_any"]
    assert ceiling < min(rung["target_stop_precision"] for rung in payload["ladder"])
    assert rule_from_artifact(payload) is None

    clear_rule_cache()
    assert load_rule() is None, "the shipped artifact must load as 'no rule'"
    clear_rule_cache()


# --------------------------------------------------------------------------------------
# the calibration driver's data plumbing
# --------------------------------------------------------------------------------------


def _driver():
    import calibrate_confidence_early_exit as driver  # noqa: PLC0415 — path set at import time

    return driver


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("suite50_007_gpt-4.1-nano_graph_r1.json", True),
        ("ladder_012_deepseek-v4-flash_graph_r2.json", True),
        ("x_anthropic-claude-sonnet-5_graph_r1.json", True),
        ("x_google-gemini-3.1-pro-preview_graph_r1.json", True),
        ("bmladapt_007_qwen3-4b_graph_r1.json", False),  # badmodel-lab: excluded
        ("localagent_007_gpt-4.1-nano_graph_r1.json", False),  # excluded marker wins
        ("suite50_007_some-other-model_graph_r1.json", False),  # off-roster
    ],
)
def test_roster_filter_selects_the_regular_models_only(filename, expected):
    assert _driver().is_regular_roster(filename) is expected


def test_trajectory_from_result_builds_one_labelled_example():
    payload = {
        "execution": {
            "observability": {
                "step_confidence": {
                    "trace": [
                        {"step": 1, "confidence": 0.4, "kind": "search"},
                        {"step": 2, "confidence": 0.9, "kind": "visit"},
                        {"step": 3, "confidence": None, "kind": "merge"},  # dropped
                    ]
                }
            }
        },
        "validation": {"overall_score": 0.8},
    }
    trajectory = _driver().trajectory_from_result(payload, "suite50_gpt-5-mini_r1.json")
    assert trajectory is not None
    assert trajectory.confidences == (0.4, 0.9)
    assert trajectory.label == 1, "overall_score 0.8 >= the 0.75 suite pass bar"
    assert trajectory.source == "suite50_gpt-5-mini_r1.json"


@pytest.mark.parametrize(
    "score, label", [(0.75, 1), (0.7499, 0), (1.0, 1), (0.0, 0)]
)
def test_label_uses_the_suite_pass_threshold(score, label):
    payload = {
        "execution": {"observability": {"step_confidence": {"trace": [{"confidence": 0.5}]}}},
        "validation": {"overall_score": score},
    }
    assert _driver().trajectory_from_result(payload, "f.json").label == label


@pytest.mark.parametrize(
    "payload",
    [
        {"validation": {"overall_score": 0.9}},  # no trace
        {"execution": {"observability": {"step_confidence": {"trace": []}}},
         "validation": {"overall_score": 0.9}},  # empty trace
        {"execution": {"observability": {"step_confidence": {"trace": [{"confidence": 0.5}]}}}},
        # ^ no validation score
        {"execution": {"observability": {"step_confidence": {"trace": [{"confidence": "x"}]}}},
         "validation": {"overall_score": 0.9}},  # no numeric confidence
    ],
)
def test_trajectory_from_result_skips_unusable_runs(payload):
    assert _driver().trajectory_from_result(payload, "f.json") is None


def test_split_is_deterministic_disjoint_and_covers_the_corpus():
    driver = _driver()
    trajectories = [_traj([0.5], i % 2, f"run_{i}.json") for i in range(400)]
    fit_a, hold_a = driver.split_trajectories(trajectories)
    fit_b, hold_b = driver.split_trajectories(trajectories)
    assert [t.source for t in fit_a] == [t.source for t in fit_b], "re-running reproduces the split"
    assert len(fit_a) + len(hold_a) == 400
    assert not ({t.source for t in fit_a} & {t.source for t in hold_a})
    assert 0.6 < len(fit_a) / 400 < 0.8, "roughly the documented 70/30 split"


def test_build_artifact_reports_an_uncertified_corpus_honestly():
    driver = _driver()
    noise = [_traj([0.95, 0.95, 0.95], i % 2, f"n_{i}.json") for i in range(300)]
    artifact = driver.build_artifact(noise, (0.90, 0.75))
    assert artifact["statistic"] is None
    assert artifact["thresholds"] == {}
    assert artifact["preferred_target_met"] is False
    assert "note" in artifact and "never exits early" in artifact["note"]
    assert [rung["certified"] for rung in artifact["ladder"]] == [False, False]
    assert artifact["signal_ceiling"]["best_certified_precision_any"] >= 0.0


def test_build_artifact_ships_the_strictest_certified_rung():
    driver = _driver()
    easy = [_traj([0.95] * 4, 1, f"e_{i}.json") for i in range(150)]
    hard = [_traj([0.1] * 4, i % 2, f"h_{i}.json") for i in range(250)]
    artifact = driver.build_artifact(easy + hard, (0.99, 0.90, 0.75))
    assert artifact["statistic"] in ("running_min", "running_mean", "last")
    assert artifact["thresholds"], "a separable corpus must certify a rule"
    assert artifact["target_stop_precision"] == 0.90, "0.99 is unreachable; 0.90 is the first rung"
    assert artifact["preferred_target_met"] is True
    assert artifact["holdout"]["n"] > 0, "the holdout must be measured, not skipped"
    assert artifact["holdout"]["stop_precision"] > artifact["holdout"]["base_rate"]
    assert rule_from_artifact(artifact) is not None
