"""Offline unit tests for `scripts/fault_corpus_recovery.py` — the sub-problem harness that
measures `extract_decision`'s recovery rate against the stored fault corpus
(`agent/idea_test_results/*_json_telemetry.jsonl`).

No agent, no model, no GPU — everything here is synthetic records or the on-disk corpus already
committed to the repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from fault_corpus_recovery import (  # noqa: E402
    FAULT_CLASSES, classify_unrecovered_shape, diff_no_regression, find_corpus_files,
    load_fault_records, main, measure_recovery,
)


def _rec(cls, raw_head, task_id="t1", model="m1"):
    return {"class": cls, "raw_head": raw_head, "task_id": task_id, "model": model,
            "parsed_ok": False, "phase": "leaf"}


# --------------------------------------------------------------------------- measure_recovery


def test_measure_recovery_counts_recovered_and_total_per_class():
    records = [
        _rec("prose", "I think the answer is 42."),
        _rec("prose", '{"action": "search"}'),  # a "prose"-classified record that IS valid JSON
        _rec("malformed_json", '{"action": "search",}'),
    ]
    report = measure_recovery(records)
    assert report["per_class"]["prose"] == {"recovered": 1, "total": 2}
    assert report["per_class"]["malformed_json"] == {"recovered": 1, "total": 1}
    assert report["overall"] == {"recovered": 2, "total": 3}


def test_measure_recovery_with_a_custom_extract_fn():
    """The harness must not hardcode `extract_decision` — a caller can pass any callable that
    returns something with a `.value` attribute, which is how a before/after A/B is done without
    monkeypatching the module under test."""
    class _Always:
        value = {"action": "x"}

    records = [_rec("prose", "anything")]
    report = measure_recovery(records, extract_fn=lambda raw: _Always())
    assert report["overall"] == {"recovered": 1, "total": 1}


def test_measure_recovery_tracks_unrecovered_shapes():
    """Decoupled from `extract_decision`'s actual recovery rate (which may improve over time and
    start recovering a kv-line shape) via a stub extractor that never recovers anything — this
    test is only about the harness's shape-classification bookkeeping."""
    class _NeverRecovers:
        value = None

    records = [
        _rec("prose", "action=search\nquery=q"),
        _rec("prose", "I have no idea what to do here."),
    ]
    report = measure_recovery(records, extract_fn=lambda raw: _NeverRecovers())
    assert report["unrecovered_shapes"]["kv_line"] == 1
    assert report["unrecovered_shapes"]["other_prose"] == 1


def test_measure_recovery_records_ids_for_diffing():
    records = [_rec("prose", '{"action": "search"}', task_id="t7", model="gemma2:2b")]
    report = measure_recovery(records)
    assert report["recovered_ids"] == [("prose", "t7", "gemma2:2b", '{"action": "search"}')]
    assert report["unrecovered_ids"] == []


# --------------------------------------------------------------------------- classify_unrecovered_shape


def test_classify_unrecovered_shape_detects_kv_line_form():
    assert classify_unrecovered_shape('action=search args={"query": "q"}') == "kv_line"
    assert classify_unrecovered_shape("action: search\nquery: q") == "kv_line"


def test_classify_unrecovered_shape_detects_action_colon_prefix():
    assert classify_unrecovered_shape("ACTION: visit\nurl: https://x") == "action_colon"


def test_classify_unrecovered_shape_detects_curly_quotes():
    assert classify_unrecovered_shape("“action”: ‘search’") == "curly_quotes"


def test_classify_unrecovered_shape_detects_python_literals():
    assert classify_unrecovered_shape('{"action": "search", "ok": True}') == "python_literal"


def test_classify_unrecovered_shape_falls_back_to_other_prose():
    assert classify_unrecovered_shape("I think the answer is 42.") == "other_prose"


# --------------------------------------------------------------------------- diff_no_regression


def test_diff_no_regression_is_empty_when_nothing_regresses():
    before = measure_recovery([_rec("prose", '{"action": "search"}')])
    after = measure_recovery([_rec("prose", '{"action": "search"}')])
    assert diff_no_regression(before, after) == []


def test_diff_no_regression_flags_a_previously_recovered_fault_that_stopped_recovering():
    records = [_rec("prose", '{"action": "search"}', task_id="t9")]
    before = measure_recovery(records)

    class _NeverRecovers:
        value = None

    after = measure_recovery(records, extract_fn=lambda raw: _NeverRecovers())
    regressed = diff_no_regression(before, after)
    assert regressed == [("prose", "t9", "m1", '{"action": "search"}')]


# --------------------------------------------------------------------------- corpus loading + baseline


def test_find_corpus_files_locates_the_real_committed_corpus():
    files = find_corpus_files()
    assert len(files) > 0, "expected agent/idea_test_results/*_json_telemetry.jsonl to exist"


def test_load_fault_records_only_returns_fault_classes():
    records = load_fault_records()
    assert records, "expected at least one fault record in the committed corpus"
    assert set(r["class"] for r in records) <= set(FAULT_CLASSES)


def test_load_fault_records_only_class_filter():
    records = load_fault_records(only_class="prose")
    assert records
    assert all(r["class"] == "prose" for r in records)


def test_recovery_never_regresses_below_the_pre_repair_baseline():
    """The documented pre-repair-rule baseline (this handoff's brief, independently reproduced
    by this harness before any `extract_decision` change): 703/952 overall, prose 331/500,
    truncated 268/316, malformed 98/121, fenced 6/15. Every later change to `extract_decision`
    must recover AT LEAST this many faults per class — never fewer."""
    records = load_fault_records()
    report = measure_recovery(records)
    assert report["overall"]["recovered"] >= 703
    assert report["per_class"]["prose"]["recovered"] >= 331
    assert report["per_class"]["truncated_json"]["recovered"] >= 268
    assert report["per_class"]["malformed_json"]["recovered"] >= 98
    assert report["per_class"]["fenced_json"]["recovered"] >= 6


def test_the_fault_corpus_is_a_growing_measurement_not_a_fixed_fixture():
    """An absolute corpus SIZE must never be asserted here, and this test says why.

    The original version of this test pinned ``total == 952``, the count measured on 2026-09-02.
    It broke the same day: any run with ``IDEA_TEST_JSON_TELEMETRY=1`` appends new records, and a
    weak-model investigation took the corpus to 1005. Pinning a count from a directory that grows
    whenever anyone collects data makes a green suite depend on NOT doing the thing the corpus
    exists for.

    What is worth pinning is the recovery RATE floor, which survives growth and still fails on a
    real parser regression. Note the rate legitimately MOVES as the corpus grows -- it fell from
    81.3% (952 records) to 77.0% (1005) purely because the newly captured weak-model faults are
    harder than the existing mix, which is a finding about those models rather than about
    ``extract_decision``.
    """
    records = load_fault_records()
    report = measure_recovery(records)

    assert report["overall"]["total"] >= 952, (
        "the corpus should only ever grow; a shrink means telemetry files were deleted")
    # The floor is the pre-repair baseline measured over the original 952 records. A parser change
    # that drops below it is a regression regardless of how much the corpus has since grown.
    assert report["overall"]["recovered"] / report["overall"]["total"] >= 0.74
    for name in ("prose", "truncated_json", "malformed_json", "fenced_json"):
        assert report["per_class"][name]["total"] > 0, f"{name} vanished from the corpus"


# --------------------------------------------------------------------------- CLI


def test_main_json_output_is_parseable(capsys):
    rc = main(["--json", "--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    import json as _json
    payload = _json.loads(out)
    assert "overall" in payload and "per_class" in payload


def test_main_text_output_reports_overall_line(capsys):
    rc = main(["--limit", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("overall:")


def test_main_class_filter_restricts_to_one_class(capsys):
    rc = main(["--json", "--class", "prose"])
    assert rc == 0
    out = capsys.readouterr().out
    import json as _json
    payload = _json.loads(out)
    assert set(payload["per_class"].keys()) <= {"prose"}
