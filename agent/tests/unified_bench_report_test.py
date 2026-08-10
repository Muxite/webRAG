"""Tests for scripts/unified_bench_report.py — the cross-benchmark (QA + codebench) view.

The load-bearing invariants pinned here, in the order the report can most easily get them
wrong:

  * ``badmodel-lab/results/cells.jsonl`` is a run-LAUNCH ledger (``{run_id, model, place,
    profile, tier, ids, runs, t}``) with no score/cost/duration anywhere in it. It must never
    produce a benchmark row — only attribution joined by run_id onto the real scored rows,
    exactly as ``badmodel-lab/analyze.py`` already does.
  * ``agent/idea_test_results/`` is read through ``scripts/bench_common.py``, the
    canonical shared loader, so this report inherits its scoping (top-level ``*_r*.json``,
    ``_report_*`` excluded by name, ``*_summary.json`` aggregates dropped for having no
    top-level ``validation.overall_score``, legacy timestamped subdirs not recursed) instead
    of becoming a fifth, drifting parser of that directory.
  * The same logical cell reachable from two sources collapses to ONE row (precedence:
    codebench runs.jsonl > idea_test_results > cells attribution), while genuinely distinct
    repeats (r1/r2, badmodel vs aider) stay separate.
"""
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import bench_common  # noqa: E402
import unified_bench_report as ubr  # noqa: E402


# ---------------------------------------------------------------------------------------
# fixtures matching the REAL shape of each file on disk
# ---------------------------------------------------------------------------------------

CELL_LINES = [
    # verbatim shape of badmodel-lab/results/cells.jsonl: attribution, no score fields
    {"run_id": "bml__qwen2.5-7b__m1_thin", "model": "qwen2.5:7b", "place": "local",
     "profile": "m1_thin", "tier": "micro", "ids": "m01,m02,m03", "runs": 3, "t": 1784760660.7},
    {"run_id": "bml__qwen2.5-7b__m1_thin", "model": "qwen2.5:7b", "place": "local",
     "profile": "m1_thin_LATER", "tier": "micro", "ids": "m01", "runs": 1, "t": 1784760999.9},
    {"run_id": "bml__qwen2.5-7b__m1_thin__064", "model": "qwen2.5:7b", "place": "local",
     "profile": "m1_thin", "tier": "064", "ids": "064", "runs": 5, "t": 1785676824.0},
]

RUNS_LINES = [
    # verbatim shape of codebench/results/runs.jsonl
    {"run_id": "calibrate_c33", "model": "qwen2.5:14b", "agent_kind": "badmodel",
     "task_id": "c33", "task_category": "hard", "test_visibility": "hidden",
     "sandbox_exit_code": "0", "duration_s": 265.0, "sandbox_actions_count": 10,
     "tests_passed": 0, "tests_total": 0, "score": 0.0, "keystone_pass": 0,
     "judge_mean": None, "usd": None, "completion_tokens": None},
    {"run_id": "coordinator_batch6", "model": "qwen2.5:14b", "agent_kind": "aider",
     "task_id": "c22", "task_category": "hard", "test_visibility": "hidden",
     "sandbox_exit_code": "0", "duration_s": 115.0, "sandbox_actions_count": None,
     "tests_passed": 13, "tests_total": 16, "score": 0.8125, "keystone_pass": 1,
     "judge_mean": None, "usd": None, "completion_tokens": None},
]


def _result_json(*, test_id="064", model="qwen2.5:7b", variant="graph_compiled",
                 score=0.96, passed=True, usd=None, secs=18.02, level="reachable"):
    """A per-cell result file, same shape idea_test_runner writes."""
    return {
        "test_metadata": {"test_id": test_id, "level": level, "weight": "short"},
        "model": model,
        "timestamp": "2026-08-06T12:33:34.727111",
        "execution_variant": variant,
        "tooling_profile": "compiled",
        "effort_tier": 0,
        "origin": "local",
        "execution": {
            "duration_seconds": secs,
            "output": {"final_deliverable": "The answer is 42 m."},
            "observability": {
                "cost": {"usd": usd},
                "llm": {"calls": 4, "total_tokens": 1234},
                "visit": {"count": 2, "chars": 900},
                "search": {"count": 1},
            },
        },
        "validation": {
            "overall_score": score,
            "overall_passed": passed,
            "grep_validations": [
                {"check": "keystone_length", "passed": True, "score": 1.0, "reason": "ok"},
                {"check": "citation", "passed": True, "score": 1.0, "reason": "url present"},
            ],
        },
    }


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture()
def cells_file(tmp_path):
    p = tmp_path / "cells.jsonl"
    _jsonl(p, CELL_LINES)
    return p


@pytest.fixture()
def results_dir(tmp_path, monkeypatch):
    d = tmp_path / "idea_test_results"
    d.mkdir()
    monkeypatch.setattr(bench_common, "results_dir", lambda: d)
    return d


# ---------------------------------------------------------------------------------------
# reader: cells.jsonl (attribution only)
# ---------------------------------------------------------------------------------------

def test_cells_reader_returns_attribution_only_never_scores(cells_file):
    attr = ubr.read_cells_attribution(cells_file)
    assert set(attr) == {"bml__qwen2.5-7b__m1_thin", "bml__qwen2.5-7b__m1_thin__064"}
    entry = attr["bml__qwen2.5-7b__m1_thin__064"]
    assert entry["place"] == "local"
    assert entry["profile"] == "m1_thin"
    assert entry["cells_tier"] == "064"  # carried verbatim; resolved separately per task id
    # The bug this guards: cells.jsonl has NO score/passed/cost/duration column, so nothing
    # here may pretend to be an independently scorable result.
    for row in attr.values():
        assert not ({"score", "passed", "cost_usd", "duration_s"} & set(row))


def test_cells_reader_last_write_wins_per_run_id(cells_file):
    # matches analyze.py::load_cells' documented convention
    assert ubr.read_cells_attribution(cells_file)["bml__qwen2.5-7b__m1_thin"]["profile"] == "m1_thin_LATER"


def test_cells_reader_missing_file_is_empty_not_an_error(tmp_path):
    assert ubr.read_cells_attribution(tmp_path / "nope.jsonl") == {}


# ---------------------------------------------------------------------------------------
# reader: codebench runs.jsonl
# ---------------------------------------------------------------------------------------

def test_code_reader_normalizes_runs_jsonl(tmp_path):
    p = tmp_path / "runs.jsonl"
    _jsonl(p, RUNS_LINES)
    rows = ubr.read_code_rows(p)
    assert [r["task_id"] for r in rows] == ["c33", "c22"]
    a, b = rows
    assert a["benchmark_type"] == "code" and a["model"] == "qwen2.5:14b"
    assert a["score"] == 0.0 and a["duration_s"] == 265.0
    assert a["cost_usd"] is None          # runs.jsonl's usd is genuinely null; not invented
    assert a["timestamp"] is None         # no timestamp column exists; not inferred
    assert a["source"] == "codebench_runs" and a["source_file"] == str(p)
    assert a["extra"]["agent_kind"] == "badmodel"
    assert a["extra"]["tests_passed"] == 0 and a["extra"]["tests_total"] == 0
    # passed is the binary keystone gate, not a threshold on the partial-credit score
    assert a["passed"] is False
    assert b["passed"] is True and b["score"] == pytest.approx(0.8125)
    assert b["extra"]["keystone_pass"] == 1


def test_code_reader_skips_malformed_lines(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(RUNS_LINES[0]) + "\n{not json\n\n", encoding="utf-8")
    assert len(ubr.read_code_rows(p)) == 1


# ---------------------------------------------------------------------------------------
# reader: idea_test_results (via bench_common) + the cells join
# ---------------------------------------------------------------------------------------

def test_qa_reader_scores_come_from_result_json_attribution_from_cells(results_dir, cells_file):
    name = "bml__qwen2.5-7b__m1_thin__064_064_qwen2.5:7b_graph_compiled_r1.json"
    _write(results_dir / name, _result_json(usd=0.0021))
    rows = ubr.read_qa_rows(run_ids=[], attribution=ubr.read_cells_attribution(cells_file))
    assert len(rows) == 1
    row = rows[0]
    # score/cost/duration: from the result JSON (the only file that has them)
    assert row["score"] == 0.96 and row["passed"] is True
    assert row["cost_usd"] == 0.0021 and row["duration_s"] == 18.02
    assert row["timestamp"] == "2026-08-06T12:33:34.727111"
    assert row["benchmark_type"] == "qa" and row["task_id"] == "064"
    assert row["source"] == "idea_test_results"
    # attribution: from cells.jsonl, joined on the longest run_id prefix
    assert row["run_id"] == "bml__qwen2.5-7b__m1_thin__064"
    assert row["extra"]["place"] == "local" and row["extra"]["profile"] == "m1_thin"
    # extra bag keeps what the aggregate flattens away (barrage analysis needs these)
    assert row["extra"]["variant"] == "graph_compiled"
    assert row["extra"]["level"] == "reachable"
    assert row["extra"]["effort_tier"] == 0
    assert row["extra"]["run_idx"] == 1
    assert row["extra"]["visits"] == 2


def test_qa_reader_task_tier_resolves_from_task_id_not_the_ledgers_tier_column(results_dir, cells_file):
    # cells.jsonl's own `tier` for this run_id is the string "064" (a task id, not a tier).
    # analyze.py's TIER_BY_TASK is authoritative; the raw ledger value is kept as cells_tier.
    name = "bml__qwen2.5-7b__m1_thin__064_064_qwen2.5:7b_graph_compiled_r1.json"
    _write(results_dir / name, _result_json())
    row = ubr.read_qa_rows(run_ids=[], attribution=ubr.read_cells_attribution(cells_file))[0]
    assert row["extra"]["cells_tier"] == "064"
    if ubr._LAB is not None:  # enrichment is optional; the join above is not
        assert row["extra"]["task_tier"] == "reachable"


def test_qa_reader_emits_no_row_for_a_launched_cell_with_no_result_file(results_dir, cells_file):
    # The central bug: a cells.jsonl run_id is a launch record. With no scored result file on
    # disk it must contribute NOTHING — not a row with a fabricated/zero score.
    assert ubr.read_qa_rows(run_ids=[], attribution=ubr.read_cells_attribution(cells_file)) == []


def test_qa_reader_falls_back_to_a_filename_derived_run_id(results_dir):
    _write(results_dir / "barrage1_smoke_baseline_rep1_134_openai-gpt-4.1-nano_graph_r1.json",
           _result_json(test_id="134", model="openai/gpt-4.1-nano", variant="graph", usd=0.005))
    row = ubr.read_qa_rows(run_ids=[], attribution={})[0]
    assert row["run_id"] == "barrage1_smoke_baseline_rep1"
    assert row["extra"].get("profile") is None  # nothing to attribute; not fabricated


def test_qa_reader_uses_bench_commons_exclusion_filters(results_dir, monkeypatch):
    """The directory mixes three incompatible shapes plus legacy subdirs; only real per-cell
    files may become rows, and that filtering must come from bench_common, not a local copy."""
    real = "run7_064_qwen2.5:7b_graph_compiled_r1.json"
    _write(results_dir / real, _result_json())
    # 1) a *_report_v2.json debug dump -> excluded by discover_files' name filter
    _write(results_dir / "run7_064_qwen2.5:7b_graph_compiled_r1_report_v2.json",
           {"debug": "dump", "validation": {"overall_score": 1.0}})
    # 2) a *_summary.json aggregate (real rows nested under "results") -> no top-level
    #    validation.overall_score, so load_row drops it
    _write(results_dir / "run7_reexpand_summary.json",
           {"results": [_result_json(score=1.0)], "summary": {"mean": 1.0}})
    # 3) a legacy timestamped subdirectory -> not recursed
    _write(results_dir / "20260526_003141_002_google" / "gemini-2.5-flash_graph_r1.json",
           _result_json(score=0.1))

    seen = {"discover": 0, "load": 0}
    real_discover, real_load = bench_common.discover_files, bench_common.load_row
    monkeypatch.setattr(bench_common, "discover_files",
                        lambda *a, **k: (seen.__setitem__("discover", seen["discover"] + 1),
                                         real_discover(*a, **k))[1])
    monkeypatch.setattr(bench_common, "load_row",
                        lambda p: (seen.__setitem__("load", seen["load"] + 1), real_load(p))[1])

    rows = ubr.read_qa_rows(run_ids=[], attribution={})
    assert seen["discover"] == 1 and seen["load"] >= 1, "must go through bench_common, not re-glob"
    assert [Path(r["source_file"]).name for r in rows] == [real]


# ---------------------------------------------------------------------------------------
# cross-source overlap / merge precedence
# ---------------------------------------------------------------------------------------

def _row(source, *, score=None, cost=None, dur=None, extra=None, run_idx=1, task="064",
         btype="qa", model="qwen2.5:7b", variant="graph_compiled"):
    return {
        "run_id": "bml__x", "timestamp": None, "benchmark_type": btype, "model": model,
        "task_id": task, "score": score, "passed": None, "cost_usd": cost, "duration_s": dur,
        "origin": None, "source": source, "source_file": f"{source}.file",
        "extra": dict({"variant": variant, "run_idx": run_idx}, **(extra or {})),
    }


def test_merge_attribution_row_never_double_counts_a_scored_row():
    # Same logical cell reachable from cells attribution and from idea_test_results.
    attribution_only = _row("cells_attribution", extra={"place": "local", "profile": "m1_thin"})
    scored = _row("idea_test_results", score=0.96, cost=0.002, dur=18.0)
    merged = ubr.merge_rows([attribution_only, scored])
    assert len(merged) == 1, "one logical cell must not become two rows"
    row = merged[0]
    # idea_test_results is the source of truth for the measured fields ...
    assert row["source"] == "idea_test_results"
    assert (row["score"], row["cost_usd"], row["duration_s"]) == (0.96, 0.002, 18.0)
    # ... while cells contributes only the attribution the winner lacked.
    assert row["extra"]["place"] == "local" and row["extra"]["profile"] == "m1_thin"


def test_merge_precedence_is_order_independent():
    attribution_only = _row("cells_attribution", extra={"profile": "m1_thin"})
    scored = _row("idea_test_results", score=0.96)
    for pair in ([attribution_only, scored], [scored, attribution_only]):
        row = ubr.merge_rows(pair)[0]
        assert row["score"] == 0.96 and row["extra"]["profile"] == "m1_thin"


def test_merge_keeps_distinct_repeats_and_agent_kinds():
    r1 = _row("idea_test_results", score=0.9, run_idx=1)
    r2 = _row("idea_test_results", score=0.5, run_idx=2)
    assert len(ubr.merge_rows([r1, r2])) == 2
    badmodel = _row("codebench_runs", score=0.5, btype="code", variant=None,
                    extra={"agent_kind": "badmodel"})
    aider = _row("codebench_runs", score=0.8, btype="code", variant=None,
                 extra={"agent_kind": "aider"})
    assert len(ubr.merge_rows([badmodel, aider])) == 2


def test_merge_codebench_outranks_the_harness_for_the_same_code_cell():
    # A codebench task also run through the standard harness: the Docker grading pipeline
    # actually executed the tests, so it wins the measured fields.
    harness = _row("idea_test_results", score=1.0, dur=5.0, btype="code", task="c22",
                   variant=None, extra={"agent_kind": None, "level": "code"})
    docker = _row("codebench_runs", score=0.8125, btype="code", task="c22", variant=None,
                  extra={"agent_kind": None, "tests_passed": 13})
    merged = ubr.merge_rows([harness, docker])
    assert len(merged) == 1
    assert merged[0]["score"] == pytest.approx(0.8125)
    assert merged[0]["duration_s"] == 5.0          # filled from the loser's non-null field
    assert merged[0]["extra"]["level"] == "code"   # loser's extras survive


# ---------------------------------------------------------------------------------------
# combined aggregation math
# ---------------------------------------------------------------------------------------

def _agg_row(agg, model, btype):
    return next(a for a in agg if a["model"] == model and a["benchmark_type"] == btype)


def test_aggregate_mean_pass_rate_and_cost_by_model_x_benchmark_type():
    rows = [
        {"model": "m1", "benchmark_type": "qa", "score": 1.0, "passed": True,
         "cost_usd": 0.01, "duration_s": 10.0, "origin": "api"},
        {"model": "m1", "benchmark_type": "qa", "score": 0.5, "passed": False,
         "cost_usd": 0.03, "duration_s": 20.0, "origin": "api"},
        {"model": "m1", "benchmark_type": "qa", "score": 0.0, "passed": False,
         "cost_usd": None, "duration_s": 30.0, "origin": "api"},   # priced-model data gap
        {"model": "m1", "benchmark_type": "code", "score": 0.25, "passed": False,
         "cost_usd": None, "duration_s": 100.0, "origin": "local"},
        {"model": "m2", "benchmark_type": "code", "score": None, "passed": None,
         "cost_usd": None, "duration_s": None, "origin": "local"},  # unscored soft task
    ]
    agg = ubr.aggregate(rows)
    assert [(a["benchmark_type"], a["model"]) for a in agg] == [
        ("code", "m1"), ("code", "m2"), ("qa", "m1")]

    qa = _agg_row(agg, "m1", "qa")
    assert qa["n"] == 3 and qa["n_scored"] == 3
    assert qa["score"] == pytest.approx(0.5)                 # (1.0 + 0.5 + 0.0) / 3
    assert qa["pass_rate"] == pytest.approx(1 / 3, abs=1e-4)   # 1 of 3 pass-judged rows
    assert qa["usd"] == pytest.approx(0.02)                  # mean over PRICED rows only
    assert qa["n_priced"] == 2
    assert qa["duration_s"] == pytest.approx(20.0)

    code = _agg_row(agg, "m1", "code")
    assert code["n"] == 1 and code["score"] == pytest.approx(0.25)
    assert code["usd"] is None and code["usd_note"] == "local"  # never a misleading $0.00

    unscored = _agg_row(agg, "m2", "code")
    assert unscored["n"] == 1 and unscored["n_scored"] == 0
    assert unscored["score"] is None and unscored["pass_rate"] is None


def test_aggregate_counts_a_measured_cost_even_when_the_local_predicate_false_positives(monkeypatch):
    # Legacy result files predate the `origin` stamp, so model_costs.is_local_row falls back to
    # a model-name lookup and can call a paid model "local" when its pricing entry is missing
    # from the static table (a real gap this repo has hit before) or, on a machine where a live
    # OpenRouter pricing cache has since filled that specific gap, simply misclassify some other
    # unpriced model. Either way, a row that RECORDED a dollar cost is a measurement and must
    # still be averaged in — force the false-positive precondition directly rather than relying
    # on a real model name's pricing-cache state at test time, which varies machine to machine.
    monkeypatch.setattr(ubr, "_is_local", lambda row: True)
    rows = [{"model": "openai/gpt-4.1-nano", "benchmark_type": "qa", "score": 0.7,
             "passed": False, "cost_usd": 0.005, "duration_s": 20.0, "origin": None}]
    agg = ubr.aggregate(rows)
    assert agg[0]["usd"] == pytest.approx(0.005) and agg[0]["n_priced"] == 1


def test_aggregate_unpriced_api_row_is_na_not_local():
    # A paid model with a pricing-table gap is a data gap, not "free" (level_ladder's rule).
    agg = ubr.aggregate([{"model": "m1", "benchmark_type": "qa", "score": 1.0, "passed": True,
                          "cost_usd": None, "duration_s": 1.0, "origin": "api"}])
    assert agg[0]["usd"] is None and agg[0]["usd_note"] == "n/a"


# ---------------------------------------------------------------------------------------
# rendering / end-to-end
# ---------------------------------------------------------------------------------------

def test_render_markdown_and_csv_cover_both_benchmark_types(tmp_path):
    rows = ubr.merge_rows([
        _row("idea_test_results", score=1.0, cost=0.01, dur=3.0, extra={"profile": "m1_thin"}),
        _row("codebench_runs", score=0.5, dur=99.0, btype="code", task="c22", variant=None,
             extra={"agent_kind": "badmodel", "tests_passed": 8, "tests_total": 16}),
    ])
    md = ubr.render_markdown(rows, ubr.aggregate(rows), qa_files_scanned=7,
                             attribution_run_ids=3)
    assert "| qwen2.5:7b | qa | 1 |" in md
    assert "| qwen2.5:7b | code | 1 |" in md
    assert "idea_test_results files scanned (bench_common scope): 7" in md
    assert "cells.jsonl run_ids read for attribution: 3 (joined onto 1 row(s))" in md

    csv_text = ubr.render_csv(rows)
    header = csv_text.splitlines()[0].split(",")
    assert header[:len(ubr.COMMON_FIELDS)] == ubr.COMMON_FIELDS
    assert "agent_kind" in header and "extra_json" in header
    assert "badmodel" in csv_text and "m1_thin" in csv_text


def test_main_end_to_end_writes_report_and_csv(results_dir, cells_file, tmp_path, capsys):
    _write(results_dir / "bml__qwen2.5-7b__m1_thin__064_064_qwen2.5:7b_graph_compiled_r1.json",
           _result_json())
    runs = tmp_path / "runs.jsonl"
    _jsonl(runs, RUNS_LINES)
    md, out_csv = tmp_path / "r.md", tmp_path / "r.csv"
    rc = ubr.main(["--cells", str(cells_file), "--code-runs", str(runs),
                   "--md", str(md), "--csv", str(out_csv)])
    assert rc == 0
    report = md.read_text(encoding="utf-8")
    assert "| idea_test_results | 1 |" in report and "| codebench_runs | 2 |" in report
    assert len(out_csv.read_text(encoding="utf-8").strip().splitlines()) == 1 + 3


def test_main_returns_nonzero_when_there_is_nothing_to_report(results_dir, tmp_path):
    assert ubr.main(["--cells", str(tmp_path / "none.jsonl"),
                     "--code-runs", str(tmp_path / "none2.jsonl")]) == 1
