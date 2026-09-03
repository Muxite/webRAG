"""Tests for scripts/kpi_dashboard.py -- the offline validation/verification KPI table.

Fixture-driven: every test builds a minimal cell dict by hand (mirroring
agent/tests/compare_arms_test.py's ``_cell`` convention) rather than depending on any file in
agent/idea_test_results/. Covers each KPI's per-cell extractor, the missing-key/coverage
discipline (a KPI must return None -- never a fabricated 0/False -- when its inputs are absent),
and the local-cell (ollama) null-cost fallback path.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import kpi_dashboard as kd  # noqa: E402


def _cell(test_id="100", model="qwen2.5:7b", variant="graph", visit_count=1, grounded=True,
          score=0.5, pass_rate=0.4, sources=None, unverified_citations=None,
          success=True, final_deliverable="answer text", grounding_gate=None,
          warning=None, grep_validations=None, usd=None, total_tokens=1000, seconds=10.0,
          infra_failed=False, nodes=None, abstention=None, pages=None, visit_urls=None):
    out = {
        "final_deliverable": final_deliverable,
        "success": success,
        "grounded": grounded,
    }
    if sources is not None:
        out["sources"] = sources
    if unverified_citations is not None:
        out["unverified_citations"] = unverified_citations
    if grounding_gate is not None:
        out["grounding_gate"] = grounding_gate
    if warning is not None:
        out["warning"] = warning
    if abstention:
        out.update(abstention)
    if pages is not None:
        out["pages"] = pages
    d = {
        "test_metadata": {"test_id": str(test_id)},
        "model": model,
        "execution_variant": variant,
        "execution": {
            "output": out,
            "graph": {"nodes": nodes or {}},
            "observability": {
                "visit": {"count": visit_count},
                "cost": {"usd": usd},
                "llm": {"total_tokens": total_tokens},
                "infra": {"failed": infra_failed},
            },
            "duration_seconds": seconds,
        },
        "validation": {
            "overall_score": score,
            "pass_rate": pass_rate,
            "grep_validations": grep_validations or [],
            "overall_passed": score is not None and score >= 0.5,
        },
        "infra_failed": infra_failed,
    }
    if visit_urls is not None:
        d["execution"]["telemetry_raw"] = {
            "timings": [{"name": "visit", "payload": {"url": u}} for u in visit_urls]
        }
    return d


def _write(dirpath, run_id, test_id, suffix_r=1, **kw):
    d = _cell(test_id=test_id, **kw)
    fname = f"{run_id}_{kw.get('variant', 'graph')}_{test_id}_model_engine_cfgabc_r{suffix_r}.json"
    with open(os.path.join(dirpath, fname), "w") as fh:
        json.dump(d, fh)
    return fname


# ---------------------------------------------------------------------------
# resolve_task_set
# ---------------------------------------------------------------------------

def test_resolve_task_set_none_when_falsy():
    assert kd.resolve_task_set(None) is None
    assert kd.resolve_task_set("") is None


def test_resolve_task_set_known_alias():
    ids = kd.resolve_task_set("smoke8")
    assert ids == {"122", "125", "128", "130", "134", "138", "140", "144"}


def test_resolve_task_set_literal_comma_list():
    assert kd.resolve_task_set("134,137, 141") == {"134", "137", "141"}


def test_resolve_task_set_invalid_raises():
    try:
        kd.resolve_task_set(",, ")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# load_cells / filter_cells
# ---------------------------------------------------------------------------

def test_load_cells_reads_and_skips_summary(tmp_path):
    _write(tmp_path, "runA", "100")
    _write(tmp_path, "runA", "101", suffix_r=2)
    with open(tmp_path / "runA_summary.json", "w") as fh:
        json.dump({"bogus": True}, fh)
    cells, unreadable = kd.load_cells(["runA"], results_dir=str(tmp_path))
    assert len(cells) == 2
    assert unreadable == []


def test_load_cells_skips_the_verbosity_3_report_render(tmp_path):
    """The verbosity>=3 `_report_v3.json` render matches load_cells' glob but is a different
    schema. Counting it as a cell doubles the denominator with score-less rows, which would move
    every KPI computed downstream — trust_kpi_dashboard reads through this loader."""
    _write(tmp_path, "runA", "100")
    with open(tmp_path / "runA_100_m_v_cfg1_r1_report_v3.json", "w") as fh:
        json.dump({"final_output": "x", "node_table": [], "verbosity_level": 3}, fh)
    cells, unreadable = kd.load_cells(["runA"], results_dir=str(tmp_path))
    assert len(cells) == 1, [c["file"] for c in cells]
    assert unreadable == []


def test_load_cells_reports_unreadable(tmp_path):
    _write(tmp_path, "runA", "100")
    with open(tmp_path / "runA_graph_999_model_engine_cfgabc_r1.json", "w") as fh:
        fh.write("{not valid json")
    cells, unreadable = kd.load_cells(["runA"], results_dir=str(tmp_path))
    assert len(cells) == 1
    assert len(unreadable) == 1


def test_load_cells_requires_prefixes():
    try:
        kd.load_cells([], results_dir=".")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_cells_merges_multiple_prefixes(tmp_path):
    _write(tmp_path, "runA", "100")
    _write(tmp_path, "runB", "200")
    cells, _ = kd.load_cells(["runA", "runB"], results_dir=str(tmp_path))
    assert len(cells) == 2


def test_filter_cells_by_model_variant_tasks():
    cells = [
        {"file": "a", "data": _cell(test_id="1", model="m1", variant="graph")},
        {"file": "b", "data": _cell(test_id="2", model="m2", variant="sequential")},
        {"file": "c", "data": _cell(test_id="3", model="m1", variant="sequential")},
    ]
    assert len(kd.filter_cells(cells, model="m1")) == 2
    assert len(kd.filter_cells(cells, variants=["sequential"])) == 2
    assert len(kd.filter_cells(cells, task_ids={"1", "3"})) == 2
    assert len(kd.filter_cells(cells, model="m1", variants=["graph"])) == 1


# ---------------------------------------------------------------------------
# K1 -- availability / never-started
# ---------------------------------------------------------------------------

def test_k1_available_true():
    d = _cell(visit_count=2, grounded=True)
    assert kd.k1_available(d) is True


def test_k1_available_false_zero_visits():
    d = _cell(visit_count=0, grounded=True)
    assert kd.k1_available(d) is False


def test_k1_available_none_when_grounded_missing():
    d = _cell(visit_count=1)
    del d["execution"]["output"]["grounded"]
    assert kd.k1_available(d) is None


def test_k1_available_none_when_visit_count_missing():
    d = _cell(visit_count=1)
    del d["execution"]["observability"]["visit"]["count"]
    assert kd.k1_available(d) is None


def test_k1_never_started_detects_tool_support_warning():
    d = _cell(warning="Error 400: does not support tools")
    assert kd.k1_never_started(d) is True


def test_k1_never_started_false_on_other_warning():
    d = _cell(warning="step budget (25) exhausted; answer synthesized")
    assert kd.k1_never_started(d) is False


def test_k1_never_started_false_when_absent():
    d = _cell()
    assert kd.k1_never_started(d) is False


# ---------------------------------------------------------------------------
# K2 -- keystone pass rate
# ---------------------------------------------------------------------------

def test_k2_keystone_all_pass():
    d = _cell(visit_count=1, grep_validations=[
        {"check": "keystone_length", "passed": True},
        {"check": "citations", "passed": True},
    ])
    assert kd.k2_keystone(d) is True


def test_k2_keystone_one_fails():
    d = _cell(visit_count=1, grep_validations=[
        {"check": "keystone_length", "passed": True},
        {"check": "keystone_count", "passed": False},
    ])
    assert kd.k2_keystone(d) is False


def test_k2_keystone_none_when_no_keystone_check():
    d = _cell(visit_count=1, grep_validations=[{"check": "citations", "passed": True}])
    assert kd.k2_keystone(d) is None


def test_k2_keystone_none_when_zero_visits():
    d = _cell(visit_count=0, grep_validations=[{"check": "keystone_length", "passed": True}])
    assert kd.k2_keystone(d) is None


# ---------------------------------------------------------------------------
# K3 -- citation validity
# ---------------------------------------------------------------------------

def test_k3_citation_validity_all_visited():
    d = _cell(sources=[{"url": "http://a"}, {"url": "http://b"}], unverified_citations=[])
    assert kd.k3_citation_validity(d) == 1.0


def test_k3_citation_validity_partial():
    d = _cell(sources=[{"url": "http://a"}], unverified_citations=["http://b"])
    assert math.isclose(kd.k3_citation_validity(d), 0.5)


def test_k3_citation_validity_none_when_no_citations():
    d = _cell(sources=[], unverified_citations=[])
    assert kd.k3_citation_validity(d) is None


def test_k3_citation_validity_none_when_fields_absent():
    d = _cell()
    assert kd.k3_citation_validity(d) is None


# ---------------------------------------------------------------------------
# K4 -- fabrication rate
# ---------------------------------------------------------------------------

def test_k4_fabrication_true():
    d = _cell(visit_count=0, success=True, final_deliverable="a confident but ungrounded answer")
    assert kd.k4_fabrication(d) is True


def test_k4_fabrication_false_when_visited():
    d = _cell(visit_count=1, success=True, final_deliverable="grounded answer")
    assert kd.k4_fabrication(d) is False


def test_k4_fabrication_false_when_deliverable_empty():
    d = _cell(visit_count=0, success=True, final_deliverable="")
    assert kd.k4_fabrication(d) is False


def test_k4_fabrication_none_when_success_missing():
    d = _cell(visit_count=0)
    del d["execution"]["output"]["success"]
    assert kd.k4_fabrication(d) is None


def test_k4_refused_ungrounded_true():
    d = _cell(grounding_gate="refused-ungrounded")
    assert kd.k4_refused_ungrounded(d) is True


def test_k4_refused_ungrounded_false_when_absent():
    d = _cell()
    assert kd.k4_refused_ungrounded(d) is False


# ---------------------------------------------------------------------------
# K5 -- abstention cross-tab
# ---------------------------------------------------------------------------

def test_k5_abstention_row_none_when_absent():
    d = _cell()
    assert kd.k5_abstention_row(d) is None


def test_k5_abstention_row_present():
    d = _cell(abstention={"finalization_status": "blocked", "answer_contract": "partial"})
    row = kd.k5_abstention_row(d)
    assert row["finalization_status"] == "blocked"
    assert row["answer_contract"] == "partial"
    assert row["deliverable_complete"] is None
    assert "overall_passed" in row


# ---------------------------------------------------------------------------
# K7 -- cost per grounded answer (incl. local-cell null-cost fallback)
# ---------------------------------------------------------------------------

def test_k7_cost_fields_paid_model():
    d = _cell(usd=0.05, total_tokens=2000, seconds=12.0, visit_count=1, grounded=True)
    fields = kd.k7_cost_fields(d)
    assert fields["usd"] == 0.05
    assert fields["grounded"] is True


def test_k7_cost_fields_local_model_null_usd():
    d = _cell(usd=None, total_tokens=2000, seconds=12.0, visit_count=1, grounded=True)
    fields = kd.k7_cost_fields(d)
    assert fields["usd"] is None
    assert fields["total_tokens"] == 2000


def test_aggregate_group_local_cells_use_token_and_seconds_fallback():
    cells = [{"file": f"f{i}", "data": _cell(test_id=str(i), usd=None, total_tokens=1000 * (i + 1),
                                             seconds=10.0 * (i + 1), visit_count=1, grounded=True)}
             for i in range(3)]
    g = kd.aggregate_group(cells)
    assert g["k7_usd_is_fallback"] is True
    assert g["k7_usd_n"] == 0
    assert g["k7_tokens_n"] == 3
    assert math.isclose(g["k7_tokens_per_grounded"], (1000 + 2000 + 3000) / 3)
    assert g["k7_secs_n"] == 3


def test_aggregate_group_paid_cells_use_usd_directly():
    cells = [{"file": f"f{i}", "data": _cell(test_id=str(i), usd=0.10 * (i + 1),
                                             visit_count=1, grounded=True)}
             for i in range(2)]
    g = kd.aggregate_group(cells)
    assert g["k7_usd_is_fallback"] is False
    assert g["k7_usd_n"] == 2
    assert math.isclose(g["k7_usd_per_grounded"], (0.10 + 0.20) / 2)


# ---------------------------------------------------------------------------
# marker_counts
# ---------------------------------------------------------------------------

def test_marker_counts_detects_firing():
    nodes = {
        "n1": {"details": {"note": "goal_achieved_numeric_unverified applied here"}},
        "n2": {"details": {"note": "chain_closure resolved, candidate_roster used"}},
        "n3": {"details": None},
    }
    d = _cell(nodes=nodes)
    counts = kd.marker_counts(d)
    assert counts["goal_achieved_numeric_unverified"] == 1
    assert counts["chain_closure"] == 1
    assert counts["candidate_roster"] == 1
    assert counts["race_value_disagreement"] == 0
    assert counts["goal_achieved_snippet_only"] == 0


def test_marker_counts_zero_when_no_nodes():
    d = _cell(nodes={})
    counts = kd.marker_counts(d)
    assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# aggregate_group -- coverage discipline (mixed present/missing fields)
# ---------------------------------------------------------------------------

def test_aggregate_group_reports_partial_k1_coverage():
    cells = [
        {"file": "a", "data": _cell(test_id="1", visit_count=1, grounded=True)},
        {"file": "b", "data": _cell(test_id="2", visit_count=1, grounded=True)},
    ]
    # third cell has no `grounded` key at all (e.g. a langgraph_react-style cell)
    d3 = _cell(test_id="3")
    del d3["execution"]["output"]["grounded"]
    cells.append({"file": "c", "data": d3})
    g = kd.aggregate_group(cells)
    assert g["total"] == 3
    assert g["k1_n"] == 2
    assert g["k1_true"] == 2
    assert g["k1_availability_rate"] == 1.0


def test_aggregate_group_k3_coverage_only_over_cells_with_citations():
    cells = [
        {"file": "a", "data": _cell(test_id="1", sources=[{"url": "http://a"}],
                                    unverified_citations=[])},
        {"file": "b", "data": _cell(test_id="2")},  # no sources/unverified fields at all
    ]
    g = kd.aggregate_group(cells)
    assert g["k3_n"] == 1
    assert g["k3_citation_validity_mean"] == 1.0


def test_aggregate_group_k5_zero_coverage_when_never_written():
    cells = [{"file": "a", "data": _cell(test_id="1")}, {"file": "b", "data": _cell(test_id="2")}]
    g = kd.aggregate_group(cells)
    assert g["k5_rows_n"] == 0


# ---------------------------------------------------------------------------
# Arm-blind auditor fallback (Ledger arms: evidence_loop/langgraph_react/sequential_react_extract
# never write execution.output.grounded or sources/unverified_citations/abstention fields -- these
# tests exercise the claim_audit.audit() fallback path added for those cells, while every test
# above stays unmodified proof that cells carrying the graph-engine fields are untouched).
# ---------------------------------------------------------------------------

def test_k1_available_audit_fallback_true_when_claim_on_page():
    d = _cell(grounded=None, visit_count=1,
              final_deliverable="The height is 1234 meters.",
              pages=[{"url": "http://a", "text": "Confirmed: height is 1234 meters."}],
              visit_urls=["http://a"])
    assert kd.k1_available(d) is True


def test_k1_available_audit_fallback_false_when_claim_unsupported():
    d = _cell(grounded=None, visit_count=1,
              final_deliverable="The height is 9999 meters.",
              pages=[{"url": "http://a", "text": "Unrelated content, no matching figure."}],
              visit_urls=["http://a"])
    assert kd.k1_available(d) is False


def test_k1_available_audit_fallback_none_when_nothing_checkable():
    # final_deliverable has no numeric/quoted claim -- the auditor has nothing to check, so this
    # must read UNKNOWN (None), never a fabricated False. Also proves the pre-existing
    # test_k1_available_none_when_grounded_missing case is exactly this shape.
    d = _cell(grounded=None, visit_count=1, final_deliverable="answer text")
    assert kd.k1_available(d) is None


def test_aggregate_group_k1_counts_audit_fallback_cells():
    cells = [
        {"file": "a", "data": _cell(test_id="1", grounded=None, visit_count=1,
                                     final_deliverable="Value is 4242 units.",
                                     pages=[{"url": "http://a", "text": "Value is 4242 units."}],
                                     visit_urls=["http://a"])},
        {"file": "b", "data": _cell(test_id="2", grounded=None, visit_count=1,
                                     final_deliverable="Value is 8181 units.",
                                     pages=[{"url": "http://b", "text": "no matching figure here"}],
                                     visit_urls=["http://b"])},
    ]
    g = kd.aggregate_group(cells)
    assert g["k1_n"] == 2
    assert g["k1_true"] == 1
    assert g["k1_fallback_n"] == 2


def test_k3_citation_validity_audit_fallback_full_support():
    d = _cell(final_deliverable="Area is 4321 km2.",
              pages=[{"url": "http://a", "text": "Area is 4321 km2 according to the record."}],
              visit_urls=["http://a"])
    val = kd.k3_citation_validity(d)
    assert val == 1.0


def test_k3_citation_validity_audit_fallback_partial_support():
    d = _cell(final_deliverable="Values are 1111 and 2222.",
              pages=[{"url": "http://a", "text": "Confirmed value 1111 in source."}],
              visit_urls=["http://a"])
    val = kd.k3_citation_validity(d)
    assert math.isclose(val, 0.5)


def test_k3_citation_validity_audit_fallback_only_when_native_fields_absent():
    # native sources/unverified_citations present -- fallback must NOT override them.
    d = _cell(sources=[{"url": "http://a"}], unverified_citations=[],
              final_deliverable="Value is 1234 units.",
              pages=[{"url": "http://b", "text": "no matching figure"}],
              visit_urls=["http://b"])
    assert kd.k3_citation_validity(d) == 1.0


def test_k5_abstention_row_audit_fallback_when_native_absent():
    d = _cell(final_deliverable="Total is 5555 units.",
              pages=[{"url": "http://a", "text": "Total is 5555 units, confirmed."}],
              visit_urls=["http://a"])
    row = kd.k5_abstention_row(d)
    assert row is not None
    assert row["source"] == "audit_fallback"
    assert row["audit_supported"] is True


def test_k5_abstention_row_native_still_marked_native():
    d = _cell(abstention={"finalization_status": "blocked", "answer_contract": "partial"})
    row = kd.k5_abstention_row(d)
    assert row["source"] == "native"


def test_k5_abstention_row_audit_fallback_none_when_nothing_checkable():
    d = _cell(final_deliverable="answer text")
    assert kd.k5_abstention_row(d) is None


def test_aggregate_group_k5_counts_fallback_rows():
    cells = [
        {"file": "a", "data": _cell(test_id="1", final_deliverable="Total is 5555 units.",
                                     pages=[{"url": "http://a", "text": "Total is 5555 units."}],
                                     visit_urls=["http://a"])},
        {"file": "b", "data": _cell(test_id="2")},
    ]
    g = kd.aggregate_group(cells)
    assert g["k5_rows_n"] == 1
    assert g["k5_fallback_n"] == 1
    assert g["k5_native_n"] == 0


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

def test_main_runs_and_writes_csv(tmp_path, capsys):
    for i, variant in enumerate(["graph", "graph"]):
        _write(tmp_path, "runX", str(100 + i), variant=variant, visit_count=1, grounded=True)
    csv_path = tmp_path / "out.csv"
    rc = kd.main(["--run-id", "runX", "--results-dir", str(tmp_path), "--csv", str(csv_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "KPI DASHBOARD" in out
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "qwen2.5:7b" in content
    assert "graph" in content


def test_main_requires_at_least_one_run_id(tmp_path):
    try:
        kd.main(["--results-dir", str(tmp_path)])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code != 0


def test_main_drops_infra_failed_cells(tmp_path, capsys):
    _write(tmp_path, "runY", "100", infra_failed=True)
    _write(tmp_path, "runY", "101", suffix_r=2, infra_failed=False)
    rc = kd.main(["--run-id", "runY", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 infra_failed cells dropped" in out


def test_main_task_set_filter(tmp_path, capsys):
    _write(tmp_path, "runZ", "100")
    _write(tmp_path, "runZ", "200", suffix_r=2)
    rc = kd.main(["--run-id", "runZ", "--results-dir", str(tmp_path), "--task-set", "100"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 cells after filters" in out
