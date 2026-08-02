"""Unit tests for ``scripts/analyze_confidence_judge_miscalibration.py``.

Pure computation only — no result files are read at test time, no network, no LLM. The
script is analysis (it changes no engine behaviour), so what is worth protecting is the
arithmetic the findings in ``app/CONFIDENCE_JUDGE_MISCALIBRATION.md`` are quoted from:

  * the AUC/interval maths,
  * the judge-visibility classification (which action results the judge can actually read),
  * the by-kind / blindness / trigger-exposure aggregation,
  * the kind ablation in the prefix table — the doc's central claim rests on the ablated and
    un-ablated numbers differing, so a filter that silently stopped filtering would matter,
  * population parity with ``calibrate_confidence_early_exit`` (both loaders must accept and
    reject exactly the same result payloads, or the two analyses stop describing one corpus).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from analyze_confidence_judge_miscalibration import (  # noqa: E402
    CONTENT_BEARING_KINDS,
    DEGENERATE_CONFIDENCE,
    ActionRecord,
    JudgedRun,
    JudgedStep,
    alternative_signals,
    auc_stderr,
    auc_with_ci,
    blindness_summary,
    by_kind,
    census_split,
    high_confidence_failures,
    payload_visibility,
    prefix_auc_table,
    reason_text_signal,
    roc_auc,
    run_from_result,
    run_level_auc,
    structural_baselines,
    summarize,
    trigger_exposure,
)
from calibrate_confidence_early_exit import trajectory_from_result  # noqa: E402


def _step(kind: str, confidence: float, reason: str = "") -> JudgedStep:
    return JudgedStep(kind=kind, confidence=confidence, reason=reason)


def _run(score: float, steps, model: str = "m", source: str = "s.json", actions=()) -> JudgedRun:
    return JudgedRun(source=source, model=model, score=score, steps=tuple(steps), actions=tuple(actions))


# --- AUC maths --------------------------------------------------------------------------


def test_roc_auc_perfect_inverted_and_tied():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5


def test_roc_auc_counts_ties_as_half():
    # one pos/neg pair tied, one pair won -> 0.75
    assert roc_auc([0.9, 0.5], [1, 1]) is None  # single class -> undefined
    assert roc_auc([0.9, 0.5, 0.5], [1, 1, 0]) == 0.75


def test_roc_auc_needs_both_classes():
    assert roc_auc([0.1, 0.2], [0, 0]) is None
    assert roc_auc([], []) is None


def test_auc_stderr_shrinks_with_sample_size():
    small = auc_stderr(0.7, 10, 10)
    large = auc_stderr(0.7, 100, 100)
    assert small is not None and large is not None
    assert large < small
    assert auc_stderr(0.7, 0, 10) is None


def test_auc_with_ci_reports_counts_and_interval():
    row = auc_with_ci([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0])
    assert row["n"] == 4 and row["n_pos"] == 2 and row["n_neg"] == 2
    assert row["auc"] == 0.875
    lo, hi = row["ci95"]
    assert lo < row["auc"] < hi

    # Perfect separation is the documented degenerate case: Hanley-McNeil gives SE 0 there.
    perfect = auc_with_ci([0.9, 0.8, 0.2], [1, 1, 0])
    assert perfect["auc"] == 1.0 and perfect["stderr"] == 0.0

    single_class = auc_with_ci([0.4, 0.5], [1, 1])
    assert single_class["auc"] is None and single_class["ci95"] is None


def test_summarize_handles_empty():
    assert summarize([]) == {"n": 0, "mean": None, "median": None}
    row = summarize([0.0, 1.0, 1.0])
    assert row["n"] == 3 and row["median"] == 1.0


# --- step classification ----------------------------------------------------------------


def test_degenerate_boundary_is_inclusive():
    assert _step("merge", DEGENERATE_CONFIDENCE).degenerate
    assert not _step("merge", DEGENERATE_CONFIDENCE + 0.01).degenerate


@pytest.mark.parametrize(
    "reason",
    [
        "resolved_content is empty and resolved_results is null, so there is nothing to verify.",
        "No step output was provided, so there is nothing to assess.",
        "The merged output is empty; content is empty.",
    ],
)
def test_blind_reason_matches_the_judges_empty_payload_phrasing(reason):
    assert _step("merge", 0.0, reason).blind_reason


def test_blind_reason_does_not_match_a_real_judgement():
    reason = "The page clearly states the elevation is 5,793 ft, which matches the sub-task."
    assert not _step("visit", 0.9, reason).blind_reason


# --- loading ----------------------------------------------------------------------------


def _payload(trace, score=1.0, nodes=None):
    return {
        "model": "openai/gpt-4.1-nano",
        "execution": {
            "observability": {"step_confidence": {"trace": trace}},
            "graph": {"nodes": nodes or {}},
        },
        "validation": {"overall_score": score},
    }


def test_run_from_result_extracts_steps_and_label():
    run = run_from_result(
        _payload([{"kind": "visit", "confidence": 0.9, "reason": "ok", "step": 1}]), "/tmp/a.json"
    )
    assert run is not None
    assert run.source == "a.json" and run.label == 1
    assert run.steps[0].kind == "visit" and run.steps[0].confidence == 0.9


def test_run_from_result_rejects_unusable_payloads():
    assert run_from_result(_payload([]), "a.json") is None
    assert run_from_result(_payload([{"kind": "visit", "confidence": 0.9}], score="n/a"), "a.json") is None
    assert run_from_result(_payload([{"kind": "visit", "confidence": None}]), "a.json") is None


def test_loader_population_matches_the_a6_calibration_driver():
    """Both loaders must accept/reject the same payloads, or the corpora diverge."""
    cases = [
        _payload([{"kind": "visit", "confidence": 0.9}]),
        _payload([]),
        _payload([{"kind": "visit", "confidence": 0.9}], score="n/a"),
        _payload([{"kind": "visit", "confidence": None}]),
        _payload([{"kind": "visit", "confidence": 0.4}, {"kind": "merge", "confidence": 0.0}], score=0.3),
    ]
    for payload in cases:
        mine = run_from_result(payload, "a.json")
        theirs = trajectory_from_result(payload, "a.json")
        assert (mine is None) == (theirs is None)
        if mine is not None:
            assert tuple(s.confidence for s in mine.steps) == theirs.confidences
            assert mine.label == theirs.label


def test_action_records_flag_judge_visibility_per_action():
    nodes = {
        "a": {"details": {"action_result": {"action": "visit", "success": True, "content": "text"}}},
        "b": {"details": {"action_result": {"action": "search", "success": True, "results": [{"url": "u"}]}}},
        "c": {"details": {"action_result": {"action": "merge", "success": True, "synthesized": {"x": 1}, "goal_achieved": True}}},
        "d": {"details": {"action_result": {"action": "think", "success": True, "thinking_content": "hm"}}},
        "e": {"details": {"action_result": {"action": "verify", "success": True, "verdict": "TRUE", "confidence": 0.8}}},
        "f": {"details": {"action_result": {"action": "visit", "success": False, "content": "x"}}},
    }
    run = run_from_result(_payload([{"kind": "visit", "confidence": 0.9}], nodes=nodes), "a.json")
    assert run is not None
    visible = {r.action: r.judge_visible for r in run.actions}
    assert visible == {"visit": True, "search": True, "merge": False, "think": False, "verify": False}
    assert len(run.actions) == 5  # the failed visit is not a judged input
    verify = next(r for r in run.actions if r.action == "verify")
    assert verify.self_confidence == 0.8
    merge = next(r for r in run.actions if r.action == "merge")
    assert merge.goal_achieved is True


# --- aggregation ------------------------------------------------------------------------


def _corpus():
    """Two passing and two failing runs with a visit/merge mix and known confidences."""
    return [
        _run(1.0, [_step("visit", 0.9, "clearly states X"), _step("merge", 0.0, "resolved_content is empty")], source="p1.json"),
        _run(0.8, [_step("visit", 0.8, "states X"), _step("merge", 0.0, "no step output")], source="p2.json"),
        _run(0.3, [_step("visit", 0.2, "page does not mention X"), _step("merge", 0.9, "looks plausible")], source="f1.json"),
        _run(0.0, [_step("visit", 0.1, "wrong page"), _step("merge", 0.8, "looks plausible")], source="f2.json"),
    ]


def test_by_kind_separates_content_bearing_from_blind_kinds():
    report = by_kind(_corpus(), min_n=1)
    assert report["visit"]["content_bearing"] is True
    assert report["visit"]["auc"]["auc"] == 1.0
    assert report["merge"]["content_bearing"] is False
    # merge is inverted here exactly as it is in the real corpus
    assert report["merge"]["auc"]["auc"] == 0.0
    assert report["merge"]["degenerate_fraction"] == 0.5
    assert report["merge"]["blind_reason_fraction"] == 0.5


def test_by_kind_reports_the_lift_a_high_score_actually_buys():
    report = by_kind(_corpus(), min_n=1)
    visit = report["visit"]
    assert visit["high_confidence_fraction"] == 0.5
    assert visit["high_confidence_pass_rate"] == 1.0
    assert visit["high_confidence_lift"] == 0.5  # base rate 0.5
    # A confident blind merge is a bad omen here, exactly as in the real corpus.
    assert report["merge"]["high_confidence_lift"] == -0.5
    assert by_kind(_corpus(), min_n=1)["visit"]["base_rate"] == 0.5


def test_by_kind_drops_kinds_below_min_n():
    assert "merge" not in by_kind(_corpus(), min_n=5)


def test_blindness_summary_counts_blind_kinds_and_runs():
    row = blindness_summary(_corpus())
    assert row["n_steps"] == 8 and row["n_blind_kind"] == 4
    assert row["blind_kind_fraction"] == 0.5
    assert row["blind_kind_degenerate_fraction"] == 0.5
    assert row["runs_with_blind_step"] == 1.0
    assert row["runs_blind_within_first_five"] == 1.0


def test_payload_visibility_aggregates_per_action():
    runs = [
        _run(1.0, [_step("visit", 0.9)], actions=[ActionRecord("visit", True), ActionRecord("merge", False)]),
        _run(0.0, [_step("visit", 0.1)], actions=[ActionRecord("visit", True), ActionRecord("merge", False)]),
    ]
    report = payload_visibility(runs)
    assert report["visit"] == {"n_results": 2, "judge_visible_fraction": 1.0}
    assert report["merge"] == {"n_results": 2, "judge_visible_fraction": 0.0}


def test_alternative_signals_score_the_discarded_fields():
    runs = [
        _run(1.0, [_step("verify", 0.0)], actions=[ActionRecord("verify", False, self_confidence=0.9),
                                                   ActionRecord("merge", False, goal_achieved=True)]),
        _run(0.0, [_step("verify", 0.0)], actions=[ActionRecord("verify", False, self_confidence=0.1),
                                                   ActionRecord("merge", False, goal_achieved=False)]),
    ]
    report = alternative_signals(runs)
    assert report["verify_self_confidence_step"]["auc"] == 1.0
    assert report["verify_self_confidence_run"]["auc"] == 1.0
    assert report["merge_goal_achieved_step"]["auc"] == 1.0


def test_prefix_auc_table_ablation_changes_the_answer():
    """The doc's central claim is that filtering the blind kinds changes the curve."""
    all_kinds = prefix_auc_table(_corpus(), "running_min", max_timestep=2)
    content = prefix_auc_table(_corpus(), "running_min", CONTENT_BEARING_KINDS, max_timestep=2)
    assert all_kinds[1]["auc"] == 1.0 and all_kinds[1]["n"] == 4
    # with the blind merge included, every run's running_min collapses toward zero
    assert all_kinds[2]["auc"] == 0.0
    # content-only keeps only the visit step, so timestep 2 has nothing to score
    assert 2 not in content and content[1]["auc"] == 1.0


def test_run_level_auc_uses_one_sample_per_run():
    report = run_level_auc(_corpus())
    assert report["last"]["n"] == 4
    # Means are 0.45/0.40 (passed) vs 0.55/0.45 (failed): the blind merge score inverts the
    # ranking, which is the whole point of quoting the content-only variant alongside.
    assert report["running_mean"]["auc"] == 0.125
    assert run_level_auc(_corpus(), CONTENT_BEARING_KINDS)["running_mean"]["auc"] == 1.0


def test_reason_text_signal_reports_the_inverted_direction():
    report = reason_text_signal(_corpus())
    # Below chance over all kinds (the blind merges), above it once they are excluded.
    assert report["confidence"]["auc"] == pytest.approx(0.375)
    assert reason_text_signal(_corpus(), CONTENT_BEARING_KINDS)["confidence"]["auc"] == 1.0
    for row in report.values():
        if row["auc"] is not None:
            assert row["inverted_auc"] == pytest.approx(1.0 - row["auc"])


def test_trigger_exposure_reports_what_a1_would_fire_on():
    row = trigger_exposure(_corpus(), threshold=0.5)
    assert row["n_steps"] == 8 and row["n_below"] == 4
    assert row["below_fraction_of_steps"] == 0.5
    assert row["kind_mix"] == {"merge": 2, "visit": 2}
    assert row["blind_kind_fraction_of_below"] == 0.5


def test_census_split_partitions_runs_by_mean_confidence():
    row = census_split(_corpus(), threshold=0.5, kinds=CONTENT_BEARING_KINDS)
    assert row["high_confidence"]["n"] == 2 and row["high_confidence"]["pass_rate"] == 1.0
    assert row["low_confidence"]["n"] == 2 and row["low_confidence"]["pass_rate"] == 0.0


def test_census_split_skips_runs_without_the_requested_kinds():
    runs = [_run(1.0, [_step("merge", 0.0)]), _run(0.0, [_step("visit", 0.9)])]
    row = census_split(runs, threshold=0.5, kinds=CONTENT_BEARING_KINDS)
    assert row["high_confidence"]["n"] == 1 and row["low_confidence"]["n"] == 0


def test_high_confidence_failures_picks_the_top_step_of_failed_runs_only():
    cases = high_confidence_failures(_corpus(), minimum=0.8, limit=10)
    assert [c["source"] for c in cases] == ["f1.json", "f2.json"]
    assert cases[0]["kind"] == "merge" and cases[0]["confidence"] == 0.9
    assert high_confidence_failures(_corpus(), minimum=0.8, limit=1)[0]["source"] == "f1.json"
    assert high_confidence_failures(_corpus(), minimum=0.99) == []


def test_structural_baselines_score_llm_free_graph_statistics():
    runs = [
        _run(1.0, [_step("visit", 0.1), _step("visit", 0.1), _step("visit", 0.1)]),
        _run(0.0, [_step("visit", 0.9)]),
    ]
    report = structural_baselines(runs)
    assert report["pooled"]["n_judged_steps"]["auc"] == 1.0
    assert report["pooled"]["run_mean_confidence"]["auc"] == 0.0
    assert set(report["by_model"]) == {"m"}
