"""Tests for scripts/compare_arms.py -- the N-way paired benchmark-arm comparison tool.

Exercises the loader, the (task,rep) pairing/rep-key logic (both filename conventions actually
seen in agent/idea_test_results/), the arm-pair stats, and -- most importantly -- the mandatory
sanity block's refusal behavior on synthetic ungrounded/invalid data.
"""
import io
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import compare_arms as ca  # noqa: E402


def _cell(test_id, score=0.5, searches_ok=3, visits=2, infra_failed=False, llm_calls=5,
          prompt_tokens=1000, total_tokens=1200, final_deliverable="answer text", auth_text=""):
    return {
        "test_metadata": {"test_id": str(test_id)},
        "execution": {
            "output": {"final_deliverable": final_deliverable + auth_text},
            "observability": {
                "llm": {"calls": llm_calls, "total_tokens": total_tokens,
                        "prompt": {"tokens": prompt_tokens}},
                "cost": {"prompt_tokens": prompt_tokens},
                "visit": {"count": visits},
                "search": {"count": searches_ok},
                "timings": {"search_query": {"success_count": searches_ok}},
                "infra": {"failed": infra_failed, "ops": []},
            },
        },
        "validation": {"overall_score": score, "llm_validation": None},
        "infra_failed": infra_failed,
    }


def _write_cell(dirpath, run_id, rep, task, suffix_r=1, **kw):
    d = _cell(task, **kw)
    fname = f"{run_id}_rep{rep}_{task}_model_engine_cfgabc_r{suffix_r}.json"
    with open(os.path.join(dirpath, fname), "w") as fh:
        json.dump(d, fh)
    return fname


# ---------------------------------------------------------------------------
# _rep_key
# ---------------------------------------------------------------------------

def test_rep_key_from_filename_rep_marker():
    # rep lives inside the filename (ladder/breadth driver convention)
    k1 = ca._rep_key("myrun", "myrun_rep1_122_model_engine_cfg_r1.json")
    k2 = ca._rep_key("myrun", "myrun_rep2_122_model_engine_cfg_r1.json")
    assert k1 != k2
    assert k1 == (1, 1)
    assert k2 == (2, 1)


def test_rep_key_from_run_id_marker():
    # rep lives inside the run_id itself (native_ab_run.sh convention)
    k = ca._rep_key("honest_adaptive_rep3", "honest_adaptive_rep3_122_r1.json")
    assert k == (3, 1)


def test_rep_key_no_marker_falls_back_to_filename():
    k1 = ca._rep_key("plainrun", "plainrun_122_abc.json")
    k2 = ca._rep_key("plainrun", "plainrun_122_xyz.json")
    assert k1 == "plainrun_122_abc.json"
    assert k1 != k2


def test_rep_key_does_not_false_match_prep_grep_step():
    # token-boundary anchoring: "step1" must not be parsed as a rep marker
    k = ca._rep_key("myrun_step1", "myrun_step1_122_r1.json")
    assert k == (None, 1)  # no rep(\d+) token match, only the trailing _r1 suffix


# ---------------------------------------------------------------------------
# load_arm
# ---------------------------------------------------------------------------

def test_load_arm_reads_cells_and_skips_summary(tmp_path):
    _write_cell(tmp_path, "runA", 1, "100", score=0.8)
    _write_cell(tmp_path, "runA", 2, "100", score=0.6)
    with open(tmp_path / "runA_summary.json", "w") as fh:
        json.dump({"bogus": "should be ignored"}, fh)
    rows, unreadable = ca.load_arm("runA", results_dir=str(tmp_path))
    assert len(rows) == 2
    assert unreadable == []
    assert {r["score"] for r in rows} == {0.8, 0.6}


def test_load_arm_reports_unreadable_json(tmp_path):
    _write_cell(tmp_path, "runA", 1, "100", score=0.8)
    with open(tmp_path / "runA_rep2_100_model_engine_cfgabc_r1.json", "w") as fh:
        fh.write("{not valid json")
    rows, unreadable = ca.load_arm("runA", results_dir=str(tmp_path))
    assert len(rows) == 1
    assert len(unreadable) == 1


# ---------------------------------------------------------------------------
# sanity_check
# ---------------------------------------------------------------------------

def _arm_from_rows(label, rows):
    return (label, rows, [])


def _row(test_id, rep, score=0.5, searches_ok=3, visits=2, infra_failed=False, auth_marker=False):
    return {"file": f"f_{test_id}_{rep}", "test_id": str(test_id), "rep": rep, "score": score,
            "prompt_tokens": 1000, "total_tokens": 1200, "llm_calls": 5,
            "visits": visits, "searches_ok": searches_ok, "infra_failed": infra_failed,
            "auth_marker": auth_marker}


def test_sanity_check_passes_on_healthy_data(capsys):
    rows_a = [_row(100, 1, searches_ok=3, visits=2) for _ in range(5)]
    rows_b = [_row(100, 1, searches_ok=3, visits=2) for _ in range(5)]
    summaries = ca.sanity_check([_arm_from_rows("a", rows_a), _arm_from_rows("b", rows_b)])
    assert len(summaries) == 2
    out = capsys.readouterr().out
    assert "sanity checks passed" in out


def test_sanity_check_refuses_on_ungrounded_run(capsys):
    # every cell has zero successful searches AND zero visits -- a dead search key or
    # equivalent -- must trigger a hard refusal by default.
    rows_a = [_row(100, r, searches_ok=0, visits=0) for r in range(1, 6)]
    rows_b = [_row(100, r, searches_ok=3, visits=2) for r in range(1, 6)]
    try:
        ca.sanity_check([_arm_from_rows("ungrounded", rows_a), _arm_from_rows("healthy", rows_b)])
        assert False, "expected SanityFailure"
    except ca.SanityFailure as e:
        assert "ungrounded" in str(e)
    out = capsys.readouterr().out
    assert "REFUSING TO PRINT COMPARISON RESULTS" in out


def test_sanity_check_override_flag_proceeds_anyway(capsys):
    rows_a = [_row(100, r, searches_ok=0, visits=0) for r in range(1, 6)]
    rows_b = [_row(100, r, searches_ok=3, visits=2) for r in range(1, 6)]
    # should NOT raise when override=True
    summaries = ca.sanity_check(
        [_arm_from_rows("ungrounded", rows_a), _arm_from_rows("healthy", rows_b)], override=True)
    assert len(summaries) == 2
    out = capsys.readouterr().out
    assert "proceeding anyway" in out


def test_sanity_check_refuses_on_empty_arm():
    try:
        ca.sanity_check([_arm_from_rows("empty", []), _arm_from_rows("healthy", [_row(100, 1)])])
        assert False, "expected SanityFailure"
    except ca.SanityFailure as e:
        assert "0 cell files" in str(e)


def test_sanity_check_refuses_on_auth_markers():
    rows_a = [_row(100, r, searches_ok=3, visits=2, auth_marker=True) for r in range(1, 6)]
    rows_b = [_row(100, r, searches_ok=3, visits=2) for r in range(1, 6)]
    try:
        ca.sanity_check([_arm_from_rows("bad_auth", rows_a), _arm_from_rows("healthy", rows_b)])
        assert False, "expected SanityFailure"
    except ca.SanityFailure as e:
        assert "auth" in str(e)


def test_auth_marker_hit_detects_setup_failed():
    d = _cell("100", final_deliverable="Setup failed: search backend unauthorized")
    assert ca._auth_marker_hit(d) is True


def test_auth_marker_hit_false_on_clean_text():
    d = _cell("100", final_deliverable="The bridge was completed in 1932.")
    assert ca._auth_marker_hit(d) is False


# ---------------------------------------------------------------------------
# compare_pair
# ---------------------------------------------------------------------------

def test_compare_pair_known_delta():
    rows_a = [_row(100, 1, score=0.8), _row(100, 2, score=0.6), _row(101, 1, score=1.0)]
    rows_b = [_row(100, 1, score=0.5), _row(100, 2, score=0.5), _row(101, 1, score=0.5)]
    res = ca.compare_pair("a", rows_a, "b", rows_b)
    assert res["n_usable"] == 3
    assert math.isclose(res["mean_a"], (0.8 + 0.6 + 1.0) / 3, rel_tol=1e-9)
    assert math.isclose(res["mean_b"], 0.5, rel_tol=1e-9)
    assert math.isclose(res["score_mean_delta"], res["mean_a"] - res["mean_b"], rel_tol=1e-9)
    assert res["w"] == 3
    assert res["l"] == 0


def test_compare_pair_reports_unpaired_explicitly():
    rows_a = [_row(100, 1, score=0.8), _row(102, 1, score=0.9)]
    rows_b = [_row(100, 1, score=0.5), _row(103, 1, score=0.2)]
    res = ca.compare_pair("a", rows_a, "b", rows_b)
    assert res["n_usable"] == 1
    assert ("102", 1) in res["only_a"]
    assert ("103", 1) in res["only_b"]


def test_compare_pair_drops_infra_failed_cells():
    rows_a = [_row(100, 1, score=0.8, infra_failed=True), _row(101, 1, score=0.9)]
    rows_b = [_row(100, 1, score=0.5), _row(101, 1, score=0.5)]
    res = ca.compare_pair("a", rows_a, "b", rows_b)
    assert res["n_paired_keys"] == 2
    assert res["n_infra_dropped"] == 1
    assert res["n_usable"] == 1


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

def test_main_exits_nonzero_on_ungrounded_run(tmp_path, capsys):
    for r in range(1, 4):
        _write_cell(tmp_path, "bad", r, "100", searches_ok=0, visits=0)
        _write_cell(tmp_path, "good", r, "100", searches_ok=3, visits=2)
    rc = ca.main(["bad:bad", "good:good", "--results-dir", str(tmp_path)])
    assert rc != 0
    out = capsys.readouterr().out
    assert "REFUSING TO PRINT COMPARISON RESULTS" in out
    assert "ARM-PAIR COMPARISONS" not in out


def test_main_prints_results_on_healthy_run(tmp_path, capsys):
    for r in range(1, 4):
        _write_cell(tmp_path, "armA", r, "100", score=0.9, searches_ok=3, visits=2)
        _write_cell(tmp_path, "armB", r, "100", score=0.5, searches_ok=3, visits=2)
    rc = ca.main(["armA:armA", "armB:armB", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ARM-PAIR COMPARISONS" in out
    assert "PER-TASK BREAKDOWN" in out


def test_main_requires_at_least_two_arms():
    try:
        ca.main(["onlyone:onlyone"])
        assert False, "expected SystemExit from argparse"
    except SystemExit as e:
        assert e.code != 0


def test_main_three_way_produces_three_pairs(tmp_path, capsys):
    for r in range(1, 4):
        _write_cell(tmp_path, "armA", r, "100", score=0.9, searches_ok=3, visits=2)
        _write_cell(tmp_path, "armB", r, "100", score=0.5, searches_ok=3, visits=2)
        _write_cell(tmp_path, "armC", r, "100", score=0.7, searches_ok=3, visits=2)
    rc = ca.main(["armA:armA", "armB:armB", "armC:armC", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("---") >= 3 * 2  # each of the 3 pairs opens a "--- X vs Y ---" block


def test_per_shape_breakdown_reports_unmapped_tasks(tmp_path, capsys):
    for r in range(1, 3):
        _write_cell(tmp_path, "armA", r, "100", score=0.9, searches_ok=3, visits=2)
        _write_cell(tmp_path, "armB", r, "100", score=0.5, searches_ok=3, visits=2)
        _write_cell(tmp_path, "armA", r, "999", score=0.4, searches_ok=3, visits=2)
        _write_cell(tmp_path, "armB", r, "999", score=0.3, searches_ok=3, visits=2)
    shapes_path = tmp_path / "shapes.json"
    with open(shapes_path, "w") as fh:
        json.dump({"100": "breadth"}, fh)
    rc = ca.main(["armA:armA", "armB:armB", "--results-dir", str(tmp_path),
                  "--shapes", str(shapes_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PER-SHAPE BREAKDOWN" in out
    assert "no shape mapping" in out
    assert "999" in out
