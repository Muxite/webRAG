"""Acceptance test for promptbench/report.py -- aggregation and exclusion rules.

WHY THE EXCLUSION RULES ARE THE POINT
-------------------------------------
A previous cycle in this project shipped a confident number that was an
artifact, and the lesson recorded from it was that the aggregation layer is
where measurement bugs become publishable claims. Three rules exist to stop
that, and they are pre-registered rather than chosen after seeing the data:

  1. Parse failures are NOT incorrect answers. They get their own rate, and a
     row that mostly failed to parse is excluded from accuracy conclusions --
     at that point the accuracy number describes the parser, not the model.
     Folding them into "incorrect" would systematically punish the verbose
     prompt shapes, manufacturing the exact effect the benchmark is testing.

  2. Items drawn from one source task module are not independent. A row is
     UNDERPOWERED if it rests on fewer than 5 clusters, or if dropping any
     single cluster moves accuracy by more than 10 percentage points.

  3. A rate computed over an empty denominator is undefined, and must render
     as None -- not as 0.0, which reads as a perfect score.

Pure functions over plain dicts. No I/O, no network, no LLM.
"""

import pytest

from agent.app.promptbench.report import (
    accuracy_per_1k_completion_tokens,
    apply_exclusions,
    loco_swing_pp,
    paired_deltas,
    summarize,
)


def _row(model="m", family="verify", variant="A1", item_id="i1", cluster="c1",
         correct=True, parse_failed=False, abstained=False, completion_tokens=10):
    return {
        "model": model, "family": family, "variant": variant, "item_id": item_id,
        "cluster": cluster, "correct": correct, "parse_failed": parse_failed,
        "abstained": abstained, "completion_tokens": completion_tokens,
        "prompt_tokens": 100, "cached_prompt_tokens": 0,
    }


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def test_summarize_groups_by_model_family_variant():
    rows = [_row(variant="A0"), _row(variant="A1")]
    out = summarize(rows)
    assert {(s["model"], s["family"], s["variant"]) for s in out} == {
        ("m", "verify", "A0"), ("m", "verify", "A1")}


def test_accuracy_counts_correct_over_all_attempted_cells():
    rows = [_row(item_id=f"i{i}", correct=(i < 3)) for i in range(4)]
    out = summarize(rows)[0]
    assert out["n"] == 4
    assert out["accuracy"] == pytest.approx(0.75)


def test_parse_failures_are_counted_separately_and_are_not_correct():
    rows = [_row(item_id="i1", correct=True),
            _row(item_id="i2", correct=False, parse_failed=True)]
    out = summarize(rows)[0]
    assert out["parse_failure_rate"] == pytest.approx(0.5)
    assert out["accuracy"] == pytest.approx(0.5)


def test_abstentions_are_counted_separately():
    rows = [_row(item_id="i1", correct=False, abstained=True), _row(item_id="i2")]
    assert summarize(rows)[0]["abstention_rate"] == pytest.approx(0.5)


def test_summarize_reports_mean_completion_tokens():
    rows = [_row(item_id="i1", completion_tokens=10), _row(item_id="i2", completion_tokens=30)]
    assert summarize(rows)[0]["mean_completion_tokens"] == pytest.approx(20.0)


def test_summarize_counts_distinct_clusters():
    rows = [_row(item_id="i1", cluster="a"), _row(item_id="i2", cluster="b"),
            _row(item_id="i3", cluster="b")]
    assert summarize(rows)[0]["n_clusters"] == 2


def test_rows_carrying_an_error_key_are_excluded_from_the_denominator():
    """A transport error is not a wrong answer -- it is a missing observation."""
    rows = [_row(item_id="i1", correct=True),
            {"model": "m", "family": "verify", "variant": "A1", "item_id": "i2",
             "cluster": "c1", "error": "TimeoutError: boom"}]
    out = summarize(rows)[0]
    assert out["n"] == 1
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["n_errors"] == 1


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert summarize([]) == []


# --------------------------------------------------------------------------
# accuracy per 1k completion tokens -- the cost axis
# --------------------------------------------------------------------------

def test_accuracy_per_1k_completion_tokens():
    assert accuracy_per_1k_completion_tokens(0.8, 20.0) == pytest.approx(40.0)


def test_accuracy_per_1k_completion_tokens_is_none_when_no_tokens_were_spent():
    """Undefined, not infinite, and certainly not zero."""
    assert accuracy_per_1k_completion_tokens(0.8, 0.0) is None


# --------------------------------------------------------------------------
# leave-one-cluster-out
# --------------------------------------------------------------------------

def test_loco_swing_is_zero_when_every_cluster_agrees():
    rows = [_row(item_id=f"i{i}", cluster=f"c{i}", correct=True) for i in range(6)]
    assert loco_swing_pp(rows) == pytest.approx(0.0)


def test_loco_swing_detects_a_single_cluster_carrying_the_result():
    """Five clusters right, one wrong: dropping the wrong one moves accuracy."""
    rows = [_row(item_id=f"i{i}", cluster=f"c{i}", correct=True) for i in range(5)]
    rows.append(_row(item_id="i5", cluster="c5", correct=False))
    assert loco_swing_pp(rows) > 10.0


def test_loco_swing_is_none_with_fewer_than_two_clusters():
    rows = [_row(item_id="i1", cluster="c1"), _row(item_id="i2", cluster="c1")]
    assert loco_swing_pp(rows) is None


# --------------------------------------------------------------------------
# exclusion rules
# --------------------------------------------------------------------------

def test_row_with_majority_parse_failures_is_excluded():
    rows = [_row(item_id=f"i{i}", cluster=f"c{i}", correct=False, parse_failed=True)
            for i in range(6)]
    rows.append(_row(item_id="i9", cluster="c9", correct=True))
    out = apply_exclusions(summarize(rows))[0]
    assert out["excluded"] is True
    assert "parse" in out["exclusion_reason"].lower()


def test_row_with_too_few_clusters_is_underpowered():
    rows = [_row(item_id=f"i{i}", cluster="c1") for i in range(8)]
    out = apply_exclusions(summarize(rows))[0]
    assert out["excluded"] is True
    assert "UNDERPOWERED" in out["exclusion_reason"]


def test_healthy_row_is_not_excluded():
    rows = [_row(item_id=f"i{i}", cluster=f"c{i}", correct=(i % 2 == 0)) for i in range(12)]
    out = apply_exclusions(summarize(rows))[0]
    assert out["excluded"] is False
    assert out["exclusion_reason"] == ""


def test_exclusion_is_additive_and_never_drops_the_row():
    """An excluded row must still be reportable -- silently disappearing is how
    a selective-reporting bug looks."""
    rows = [_row(item_id=f"i{i}", cluster="c1") for i in range(8)]
    summary = summarize(rows)
    assert len(apply_exclusions(summary)) == len(summary)


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------

def test_paired_deltas_pairs_on_item_id_within_one_model_and_family():
    rows = [
        _row(variant="A1", item_id="i1", correct=False),
        _row(variant="A2", item_id="i1", correct=True),
        _row(variant="A1", item_id="i2", correct=True),
        _row(variant="A2", item_id="i2", correct=True),
    ]
    deltas = paired_deltas(rows, "A2", "A1", model="m", family="verify")
    assert sorted(deltas) == [0.0, 1.0]


def test_paired_deltas_drops_items_missing_from_either_arm():
    """An unpaired item would make the comparison not a paired comparison."""
    rows = [
        _row(variant="A1", item_id="i1", correct=True),
        _row(variant="A2", item_id="i1", correct=True),
        _row(variant="A1", item_id="i2", correct=True),
    ]
    assert len(paired_deltas(rows, "A2", "A1", model="m", family="verify")) == 1


def test_paired_deltas_is_empty_when_arms_do_not_overlap():
    rows = [_row(variant="A1", item_id="i1"), _row(variant="A2", item_id="i2")]
    assert paired_deltas(rows, "A2", "A1", model="m", family="verify") == []
