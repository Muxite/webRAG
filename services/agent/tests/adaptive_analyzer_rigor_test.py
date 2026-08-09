"""Regression tests for the analyzer rigor fixes (scripts/adaptive_ab_analyze.py).

Pins the two headline-integrity fixes:
  * paired_deltas(missing="zero") scores a timed-out/missing cell as 0 over the UNION
    grid instead of intersection-dropping it (survivorship inflated the pilot delta ~15%).
  * _rep_key anchors rep(\\d+) to a token boundary so run-ids containing prep/grep/step
    don't false-match a replicate index (which silently mis-pairs / drops data).
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import adaptive_ab_analyze as ana  # noqa: E402
from adaptive_ab_analyze import (  # noqa: E402
    paired_deltas,
    quarantine_infra,
    _infra_failed,
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


# ------------------------------------------------------- F17: infra-failure quarantine ----------
def _irows(specs):
    """rows as (task, rep, score, infra_failed)."""
    return [{"test_id": t, "rep": (r, r), "score": s, "infra_failed": i} for t, r, s, i in specs]


def test_infra_failed_reads_either_flag():
    # Top-level flag (testing/runner.py) ...
    assert _infra_failed({"infra_failed": True}, {}) is True
    # ... or the observability roll-up it derives from (older result JSONs).
    assert _infra_failed({}, {"infra": {"failed": True, "ops": ["search_query"]}}) is True
    assert _infra_failed({}, {"infra": {"failed": False}}) is False
    assert _infra_failed({}, {}) is False


def test_quarantine_infra_splits_rows_and_excludes_the_poisoned_cell():
    A = _irows([("1", 1, 0.8, False), ("2", 1, 0.0, True)])
    B = _irows([("1", 1, 0.5, False), ("2", 1, 0.4, False)])
    (Ac, Bc), (Ai, Bi), excl = quarantine_infra(A, B)
    assert [r["test_id"] for r in Ac] == ["1"] and [r["test_id"] for r in Ai] == ["2"]
    assert Bi == [] and len(Bc) == 2
    assert excl["rep"] == {("2", (1, 1))}
    assert excl["task"] == {"2"}          # adaptive lost EVERY row for task 2 -> whole task out


def test_infra_cell_is_excluded_not_zero_filled():
    # Without the exclusion, adaptive's 402-poisoned task-2 cell would be zero-filled and read as
    # a -0.4 model regression; the honest result is that the pair is not measured at all.
    A = _irows([("1", 1, 0.8, False), ("2", 1, 0.0, True)])
    B = _irows([("1", 1, 0.5, False), ("2", 1, 0.4, False)])
    (Ac, Bc), _, excl = quarantine_infra(A, B)
    naive = dict(paired_deltas(Ac, Bc, by="task", missing="zero"))
    honest = dict(paired_deltas(Ac, Bc, by="task", missing="zero", exclude=excl))
    assert naive["2"] == pytest.approx(-0.4)   # the bug: an outage scored as a model failure
    assert "2" not in honest                   # quarantined from BOTH arms
    assert honest["1"] == pytest.approx(0.3)   # healthy pairs are untouched


def test_infra_quarantine_keeps_surviving_reps_of_the_same_task():
    # Only the poisoned (task, rep) cell drops; rep 2 of the same task still pairs, so one bad
    # network window doesn't cost the whole task.
    A = _irows([("1", 1, 0.0, True), ("1", 2, 0.9, False)])
    B = _irows([("1", 1, 0.5, False), ("1", 2, 0.4, False)])
    (Ac, Bc), _, excl = quarantine_infra(A, B)
    assert excl["task"] == set()               # task 1 survives via rep 2
    reps = paired_deltas(Ac, Bc, by="rep", missing="zero", exclude=excl)
    assert [k for k, _ in reps] == ["1"]       # only the rep-2 pair remains
    assert reps[0][1] == pytest.approx(0.5)
    # The quarantine is PAIRWISE: baseline's healthy rep-1 partner leaves the task mean too, so
    # the mean is not an unpaired mix (0.9 - mean(0.5, 0.4) = 0.45 would be the sloppy answer).
    assert dict(paired_deltas(Ac, Bc, by="task", missing="zero", exclude=excl))["1"] == \
        pytest.approx(0.5)


def _write_result(tmp_path, run_id, task, score, infra=False):
    body = {
        "test_metadata": {"test_id": task},
        "model": "openai/gpt-4.1-nano",
        "validation": {"overall_score": score},
        "execution": {"observability": {"visit": {"count": 1}, "cost": {"usd": 0.01},
                                        "infra": {"failed": infra,
                                                  "ops": ["search_query"] if infra else []}}},
    }
    if infra:
        body["infra_failed"] = True
    (tmp_path / f"{run_id}_{task}_nano_graph_r1.json").write_text(json.dumps(body))


def test_load_arm_flags_infra_rows_from_the_result_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ana, "RESULTS_DIR", str(tmp_path))
    _write_result(tmp_path, "run1_rep1", "140", 0.0, infra=True)
    _write_result(tmp_path, "run1_rep1", "141", 0.9)
    rows = {r["test_id"]: r for r in ana.load_arm("run1_rep1")}
    assert rows["140"]["infra_failed"] is True
    assert rows["140"]["infra_ops"] == "search_query"
    assert rows["141"]["infra_failed"] is False


def test_main_reports_infra_exclusions_end_to_end(tmp_path, monkeypatch, capsys):
    # Wiring check: a 402-poisoned adaptive cell must be reported and quarantined, NOT folded
    # into the paired grid as a 0 — and the healthy pair must still be scored.
    results, out = tmp_path / "results", tmp_path / "out"
    results.mkdir()
    _write_result(results, "adapt_rep1", "140", 0.0, infra=True)
    _write_result(results, "adapt_rep1", "141", 0.9)
    _write_result(results, "base_rep1", "140", 0.4)
    _write_result(results, "base_rep1", "141", 0.4)
    monkeypatch.setattr(ana, "RESULTS_DIR", str(results))
    monkeypatch.setattr(sys, "argv", ["adaptive_ab_analyze.py", "--adaptive-run-id", "adapt_rep1",
                                      "--baseline-run-id", "base_rep1", "--out", str(out)])
    ana.main()
    printed = capsys.readouterr().out
    assert "infra failures excluded: 1 (adaptive 1, baseline 0)" in printed
    infra_csv = (out / "ab_infra.csv").read_text().splitlines()
    assert infra_csv[0] == "arm,task,rep,score,ops"
    assert infra_csv[1].startswith("adaptive,140,")
    # n=1 pair (task 141 only): task 140 is quarantined, not scored -0.4.
    paired = (out / "ab_paired.csv").read_text()
    assert "OVERALL,task,1,0.5000" in paired


def test_genuine_timeout_is_still_zero_filled():
    # The quarantine must NOT become a survivorship loophole: a MISSING cell with no infra flag
    # (timeout/crash on a hard task) is still a real 0 over the union grid.
    A = _rows([("1", 0.8), ("2", 0.9)])
    B = _rows([("1", 0.5), ("2", 0.4), ("3", 0.3)])
    (Ac, Bc), (Ai, Bi), excl = quarantine_infra(A, B)
    assert Ai == [] and Bi == [] and excl["rep"] == set() and excl["task"] == set()
    assert dict(paired_deltas(Ac, Bc, by="task", missing="zero", exclude=excl))["3"] == \
        pytest.approx(-0.3)


def test_dollars_per_solved():
    rows = _grow([(0.9, 1), (0.8, 1), (0.3, 0)])         # 2 solved (>=0.75), 3 * $0.01 spend
    assert _dollars_per_solved(rows) == pytest.approx(0.03 / 2)
    none_solved = _grow([(0.1, 0), (0.2, 0)])
    import math
    assert math.isnan(_dollars_per_solved(none_solved))
