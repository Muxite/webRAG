"""Calibration metrics, checked against hand-computed values.

A metric bug here is indistinguishable from a model result: if ECE were computed
wrong the output would still be a plausible-looking number in [0, 1], and the
write-up would report it as a finding about the judge. So every metric is pinned
to a case whose answer is known by arithmetic rather than by running the code.
"""

from __future__ import annotations

import math

import pytest

from agent.app.promptbench.calibration import (
    FREE_BASELINE_AUC,
    VERBAL_BANDS,
    auc,
    brier,
    calibration_slope,
    distinct_confidence_fraction,
    ece,
    is_finite_probability,
    murphy_decomposition,
    reliability_diagram,
    summarize_calibration,
)
from agent.app.promptbench.grade import extract_confidence, grade_confidence

PERFECT_C = [0.0, 0.0, 1.0, 1.0]
PERFECT_K = [False, False, True, True]


# ---------------------------------------------------------------------------
# Known-answer cases
# ---------------------------------------------------------------------------

def test_brier_matches_hand_arithmetic():
    # (0.8-1)^2 + (0.4-0)^2 = 0.04 + 0.16 = 0.20, over 2 = 0.10
    assert brier([0.8, 0.4], [True, False]) == pytest.approx(0.10)


def test_perfect_prediction_scores_zero_on_every_error_metric():
    assert brier(PERFECT_C, PERFECT_K) == pytest.approx(0.0)
    assert ece(PERFECT_C, PERFECT_K) == pytest.approx(0.0)
    assert auc(PERFECT_C, PERFECT_K) == pytest.approx(1.0)
    assert calibration_slope(PERFECT_C, PERFECT_K) == pytest.approx(1.0)


def test_inverted_confidence_is_visible_as_a_negative_slope():
    """The anti-calibration this whole line of investigation is about: higher
    stated confidence predicting a WORSE outcome."""
    inverted = [1.0, 1.0, 0.0, 0.0]
    assert auc(inverted, PERFECT_K) == pytest.approx(0.0)
    assert calibration_slope(inverted, PERFECT_K) < 0


def test_a_constant_confidence_scores_exactly_chance_not_zero_or_one():
    """A judge emitting one number for everything -- 43% of the real trace, per
    CONFIDENCE_JUDGE_MISCALIBRATION.md -- must land on 0.5, not on whatever the
    sort order happened to produce."""
    assert auc([0.5] * 4, PERFECT_K) == pytest.approx(0.5)


def test_a_constant_confidence_is_flagged_degenerate_with_zero_resolution():
    stats = summarize_calibration([0.5] * 4, PERFECT_K)
    assert stats["degenerate"] is True
    assert stats["resolution"] == pytest.approx(0.0)
    assert calibration_slope([0.5] * 4, PERFECT_K) is None


def test_ece_on_a_known_miscalibrated_case():
    # Ten items all stating 0.9, of which 5 are correct: |0.9 - 0.5| = 0.4.
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)


def test_murphy_decomposition_reconstructs_brier():
    """Exact only when the forecast is constant within each bin, which is what the
    decomposition is defined on. Values chosen one per bin so the identity is
    arithmetic rather than approximate."""
    cs = [0.9, 0.8, 0.7, 0.25, 0.15, 0.05]
    ks = [True, True, False, False, True, False]
    d = murphy_decomposition(cs, ks)
    assert (d["reliability"] - d["resolution"] + d["uncertainty"]) == pytest.approx(
        brier(cs, ks), abs=1e-12)


def test_confidence_of_exactly_one_lands_in_the_top_bin():
    rows = reliability_diagram([1.0], [True], bins=10)
    assert rows[0]["bin"] == 9


def test_auc_is_undefined_rather_than_wrong_when_a_class_is_empty():
    assert auc([0.9, 0.8], [True, True]) is None


def test_the_bar_is_the_inherited_free_baseline_not_one_half():
    """0.655 is CONFIDENCE_JUDGE_MISCALIBRATION.md's best LLM-FREE statistic. An
    arm scoring 0.60 beats chance and has still not earned its LLM call."""
    assert FREE_BASELINE_AUC == 0.655
    # AUC 5/9 = 0.556: better than chance, still below the free structural baseline.
    mediocre = summarize_calibration([0.9, 0.6, 0.5, 0.8, 0.7, 0.4],
                                     [True, True, True, False, False, False])
    assert 0.5 < mediocre["auc"] < FREE_BASELINE_AUC
    assert mediocre["beats_free_baseline"] is False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_numeric_confidence_is_read_from_json_in_either_field_order():
    assert extract_confidence('{"answer":"SATISFIES","confidence":0.8,"reason":"x"}') == 0.8
    assert extract_confidence('{"reason":"x","answer":"VIOLATES","confidence":0.25}') == 0.25


def test_verbal_bands_map_through_the_declared_table():
    for band, value in VERBAL_BANDS.items():
        assert extract_confidence('{"answer":"SATISFIES","certainty":"%s"}' % band) == value


def test_verbal_bands_are_monotone_in_the_obvious_direction():
    assert (VERBAL_BANDS["certain"] > VERBAL_BANDS["likely"]
            > VERBAL_BANDS["unsure"] > VERBAL_BANDS["guessing"])


def test_an_out_of_range_number_is_rejected_rather_than_clamped():
    """A model answering 95 for 95% is not stating a probability. Clamping it to
    1.0 would silently convert a formatting failure into maximal confidence -- the
    most damaging direction for a calibration metric to be wrong in."""
    assert extract_confidence('{"answer":"SATISFIES","confidence":95}') is None
    assert extract_confidence('{"answer":"SATISFIES","confidence":-0.2}') is None


def test_a_missing_confidence_is_none_and_never_defaulted_to_a_half():
    assert extract_confidence("SATISFIES") is None
    assert extract_confidence("") is None


def test_booleans_are_not_probabilities():
    assert is_finite_probability(True) is False
    assert is_finite_probability(float("nan")) is False
    assert is_finite_probability(0.5) is True


def test_grade_confidence_grades_the_answer_exactly_like_an_a_arm():
    verdict, conf = grade_confidence(
        '{"answer": "SATISFIES", "confidence": 0.8}', "SATISFIES", ["SATISFIES", "VIOLATES"])
    assert verdict.correct is True and not verdict.parse_failed
    assert conf == 0.8

    verdict, conf = grade_confidence(
        '{"answer": "VIOLATES", "confidence": 0.8}', "SATISFIES", ["SATISFIES", "VIOLATES"])
    assert verdict.correct is False and conf == 0.8


def test_distinct_confidence_fraction_detects_a_one_note_judge():
    assert distinct_confidence_fraction([0.5] * 8) == pytest.approx(0.125)
    assert distinct_confidence_fraction([0.1, 0.2, 0.3]) == pytest.approx(1.0)
