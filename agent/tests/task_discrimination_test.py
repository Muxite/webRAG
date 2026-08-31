"""Tests for scripts/task_discrimination.py -- the task-discriminative-power ranker.

scripts/task_discrimination.py is a plain module (not a package), imported the same way
agent/tests/compare_arms_test.py and agent/tests/bench_stats_test.py import their siblings: by
inserting scripts/ onto sys.path.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import pytest  # noqa: E402

import task_discrimination as td  # noqa: E402


def _result(task_id, score, model="qwen2.5:7b", variant="graph", infra_failed=False,
            duration_seconds=10.0):
    return {
        "test_metadata": {"test_id": task_id},
        "model": model,
        "execution_variant": variant,
        "validation": {"overall_score": score},
        "infra_failed": infra_failed,
        "execution": {
            "duration_seconds": duration_seconds,
            "observability": {"infra": {"failed": infra_failed, "ops": []}},
        },
    }


def _write(dirpath, name, obj):
    path = dirpath / name
    with open(path, "w") as fh:
        json.dump(obj, fh)
    return path


# ---------------------------------------------------------------------------
# iter_result_files
# ---------------------------------------------------------------------------

def test_iter_result_files_skips_summary_report_ledger(tmp_path):
    _write(tmp_path, "run_100_r1.json", _result("100", 0.5))
    _write(tmp_path, "run_summary.json", {"bogus": True})
    _write(tmp_path, "run_report_x.json", {"bogus": True})
    sub = tmp_path / "_myrun"
    sub.mkdir()
    _write(sub, "ledger.json", {"bogus": True})
    _write(sub, "run_meta.json", {"bogus": True})
    _write(sub, "myrun_100_r1.json", _result("100", 0.6))
    found = {p.name for p in td.iter_result_files(str(tmp_path))}
    assert found == {"run_100_r1.json", "myrun_100_r1.json"}


def test_iter_result_files_skips_cell_logs_dir(tmp_path):
    logs = tmp_path / "_myrun" / "cell_logs"
    logs.mkdir(parents=True)
    _write(logs, "stray.json", {"bogus": True})
    assert td.iter_result_files(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# load_cell
# ---------------------------------------------------------------------------

def test_load_cell_reads_score_and_task(tmp_path):
    p = _write(tmp_path, "a_100_r1.json", _result("100", 0.75, model="m", variant="graph"))
    row = td.load_cell(p)
    assert row["task_id"] == "100"
    assert row["score"] == 0.75
    assert row["model"] == "m"
    assert row["variant"] == "graph"
    assert row["infra_failed"] is False
    assert row["secs"] == 10.0


def test_load_cell_none_on_missing_score(tmp_path):
    d = _result("100", 0.5)
    d["validation"]["overall_score"] = None
    p = _write(tmp_path, "a_100_r1.json", d)
    assert td.load_cell(p) is None


def test_load_cell_none_on_missing_task_id(tmp_path):
    d = _result("100", 0.5)
    d["test_metadata"] = {}
    p = _write(tmp_path, "a_100_r1.json", d)
    assert td.load_cell(p) is None


def test_load_cell_none_on_non_dict_json(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert td.load_cell(p) is None


def test_load_cell_none_on_unreadable_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert td.load_cell(p) is None


def test_load_cell_infra_failed_flag(tmp_path):
    p = _write(tmp_path, "a_100_r1.json", _result("100", 0.5, infra_failed=True))
    row = td.load_cell(p)
    assert row["infra_failed"] is True


# ---------------------------------------------------------------------------
# load_all_cells (infra_failed quarantine)
# ---------------------------------------------------------------------------

def test_load_all_cells_quarantines_infra_failed(tmp_path):
    _write(tmp_path, "a_100_r1.json", _result("100", 0.5, infra_failed=False))
    _write(tmp_path, "a_100_r2.json", _result("100", 0.9, infra_failed=True))
    cells = td.load_all_cells(str(tmp_path))
    assert len(cells) == 1
    assert cells[0]["score"] == 0.5


# ---------------------------------------------------------------------------
# filter_cells
# ---------------------------------------------------------------------------

def _cells():
    return [
        {"task_id": "100", "model": "qwen2.5:7b", "variant": "graph", "score": 0.5,
         "infra_failed": False, "secs": 1.0},
        {"task_id": "100", "model": "gpt-5-mini", "variant": "langgraph_react", "score": 0.9,
         "infra_failed": False, "secs": 2.0},
        {"task_id": "200", "model": "qwen2.5:7b", "variant": "graph", "score": 1.0,
         "infra_failed": False, "secs": 3.0},
    ]


def test_filter_cells_by_task_ids():
    out = td.filter_cells(_cells(), task_ids=["100"])
    assert {c["task_id"] for c in out} == {"100"}


def test_filter_cells_by_model_substring():
    out = td.filter_cells(_cells(), models=["qwen"])
    assert len(out) == 2
    assert all("qwen" in c["model"] for c in out)


def test_filter_cells_by_variant_substring():
    out = td.filter_cells(_cells(), variants=["langgraph"])
    assert len(out) == 1
    assert out[0]["variant"] == "langgraph_react"


def test_filter_cells_no_filters_passes_everything():
    out = td.filter_cells(_cells())
    assert len(out) == 3


# ---------------------------------------------------------------------------
# summarize_task -- floor/ceiling/discriminating classification
# ---------------------------------------------------------------------------

def _c(task_id, score, secs=1.0, model="m", variant="v"):
    return {"task_id": task_id, "model": model, "variant": variant, "score": score,
            "infra_failed": False, "secs": secs}


def test_summarize_task_no_data():
    row = td.summarize_task("100", [], min_n=5)
    assert row["n"] == 0
    assert row["classification"] == "NO_DATA"
    assert row["confidence"] == "LOW"


def test_summarize_task_dead_floor_all_below_threshold():
    cells = [_c("100", 0.0), _c("100", 0.03), _c("100", 0.05)]
    row = td.summarize_task("100", cells, min_n=1)
    assert row["classification"] == "DEAD_FLOOR"
    assert row["floor_rate"] == 1.0


def test_summarize_task_dead_ceiling_all_above_threshold():
    cells = [_c("100", 1.0), _c("100", 0.97), _c("100", 0.95)]
    row = td.summarize_task("100", cells, min_n=1)
    assert row["classification"] == "DEAD_CEILING"
    assert row["ceiling_rate"] == 1.0


def test_summarize_task_discriminating_mixed_scores():
    cells = [_c("100", 0.0), _c("100", 0.5), _c("100", 1.0)]
    row = td.summarize_task("100", cells, min_n=1)
    assert row["classification"] == "DISCRIMINATING"
    assert row["sd"] > 0


def test_summarize_task_floor_ceiling_boundary_values_are_dead_not_mid():
    # exactly-at-threshold scores count toward floor/ceiling, not the "mid" band.
    cells = [_c("100", 0.05), _c("100", 0.05), _c("100", 0.95)]
    row = td.summarize_task("100", cells, min_n=1)
    assert row["floor_rate"] == pytest.approx(2 / 3)
    assert row["ceiling_rate"] == pytest.approx(1 / 3)
    assert row["frac_mid"] == pytest.approx(0.0)
    assert row["classification"] == "DISCRIMINATING"  # not unanimous either direction


def test_summarize_task_low_confidence_below_min_n():
    cells = [_c("100", 0.0), _c("100", 0.02)]
    row = td.summarize_task("100", cells, min_n=5)
    assert row["classification"] == "DEAD_FLOOR"
    assert row["confidence"] == "LOW"


def test_summarize_task_ok_confidence_at_min_n():
    cells = [_c("100", 0.0)] * 5
    row = td.summarize_task("100", cells, min_n=5)
    assert row["confidence"] == "OK"


def test_summarize_task_raises_on_mismatched_task_id():
    cells = [_c("100", 0.5), _c("200", 0.6)]
    with pytest.raises(ValueError):
        td.summarize_task("100", cells, min_n=1)


def test_summarize_task_counts_models_variants_and_wallclock():
    cells = [_c("100", 0.5, secs=2.0, model="a", variant="graph"),
              _c("100", 0.6, secs=3.0, model="b", variant="graph"),
              _c("100", 0.7, secs=4.0, model="a", variant="langgraph_react")]
    row = td.summarize_task("100", cells, min_n=1)
    assert row["n_models"] == 2
    assert row["n_variants"] == 2
    assert row["total_secs"] == pytest.approx(9.0)
    assert row["mean_secs"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# rank_tasks -- ordering + missing-task handling
# ---------------------------------------------------------------------------

def test_rank_tasks_orders_by_score_sd_descending():
    cells = [
        _c("100", 1.0), _c("100", 1.0), _c("100", 1.0),  # sd 0 (dead ceiling)
        _c("200", 0.0), _c("200", 0.5), _c("200", 1.0),  # sd > 0 (discriminating)
    ]
    rows = td.rank_tasks(cells, ["100", "200"], min_n=1)
    assert [r["task_id"] for r in rows] == ["200", "100"]


def test_rank_tasks_includes_no_data_tasks():
    rows = td.rank_tasks([], ["100", "200"], min_n=1)
    assert {r["task_id"] for r in rows} == {"100", "200"}
    assert all(r["classification"] == "NO_DATA" for r in rows)


# ---------------------------------------------------------------------------
# wallclock_summary
# ---------------------------------------------------------------------------

def test_wallclock_summary_splits_dead_vs_discriminating():
    cells_dead = [_c("100", 1.0, secs=10.0), _c("100", 1.0, secs=10.0)]
    cells_disc = [_c("200", 0.0, secs=5.0), _c("200", 1.0, secs=5.0)]
    rows = td.rank_tasks(cells_dead + cells_disc, ["100", "200"], min_n=1)
    wc = td.wallclock_summary(rows)
    assert wc["total_secs"] == pytest.approx(30.0)
    assert wc["dead_secs"] == pytest.approx(20.0)
    assert wc["dead_frac"] == pytest.approx(20.0 / 30.0)


def test_wallclock_summary_zero_total_is_zero_frac():
    rows = td.rank_tasks([], ["100"], min_n=1)
    wc = td.wallclock_summary(rows)
    assert wc["total_secs"] == 0.0
    assert wc["dead_frac"] == 0.0


# ---------------------------------------------------------------------------
# format_table / write_csv -- smoke tests
# ---------------------------------------------------------------------------

def test_format_table_renders_nan_as_na():
    rows = td.rank_tasks([], ["100"], min_n=1)
    out = td.format_table(rows)
    assert "n/a" in out
    assert "NO_DATA" in out


def test_write_csv_round_trips(tmp_path):
    cells = [_c("100", 0.5), _c("100", 0.9)]
    rows = td.rank_tasks(cells, ["100"], min_n=1)
    out_path = tmp_path / "out.csv"
    td.write_csv(rows, str(out_path))
    text = out_path.read_text()
    assert "task_id" in text.splitlines()[0]
    assert "100" in text


# ---------------------------------------------------------------------------
# _task_sets -- reused from adaptive_ladder_run.py, not re-hardcoded
# ---------------------------------------------------------------------------

def test_task_sets_reuses_adaptive_ladder_run_definitions():
    import adaptive_ladder_run as alr
    ts = td._task_sets()
    assert ts is alr.TASK_SETS
    assert "core24" in ts and "suite59" in ts
    assert len(ts["core24"]) == 24
