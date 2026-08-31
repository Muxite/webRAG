"""Tests for scripts/coverage_report.py -- distinct-URL coverage / duplication-factor report.

Fixture-driven: builds minimal synthetic result JSONs (graph shapes, missing fields, infra
failures) and asserts the extraction/aggregation/A-B-pairing contracts documented in each
function's docstring.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import coverage_report as cr  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _visit_node(node_id, url, success=True, has_url_key=True):
    ar = {"action": "visit", "success": success}
    if has_url_key:
        ar["url"] = url
    return {"node_id": node_id, "details": {"action": "visit", "action_result": ar}}


def _non_visit_node(node_id):
    return {"node_id": node_id, "details": {"action": "finalize", "action_result": {}}}


def _graph(nodes):
    return {"root_id": "root", "nodes": nodes}


def _cell(test_id, graph=None, visits=None, score=0.5, infra_failed=False,
          grep_validations=None, coverage_ratio=None, variant="graph", model="m"):
    output = {"success": True}
    if coverage_ratio is not None:
        output["coverage_ratio"] = coverage_ratio
    execution = {"output": output, "observability": {"visit": {"count": visits}}}
    if graph is not None:
        execution["graph"] = graph
    return {
        "test_metadata": {"test_id": str(test_id)},
        "execution": execution,
        "validation": {"overall_score": score, "grep_validations": grep_validations or []},
        "infra_failed": infra_failed,
        "execution_variant": variant,
        "model": model,
    }


def _write(dirpath, run_id, rep, task, **kw):
    d = _cell(task, **kw)
    fname = f"{run_id}_rep{rep}_{task}_model_engine_cfgabc_r1.json"
    with open(os.path.join(dirpath, fname), "w") as fh:
        json.dump(d, fh)
    return fname


# ---------------------------------------------------------------------------
# normalize_url / extract_visits -- distinct-URL counting
# ---------------------------------------------------------------------------

def test_normalize_url_collapses_fragment_slash_case():
    assert cr.normalize_url("HTTPS://Example.com/Page#section") == "https://example.com/page"
    assert cr.normalize_url("https://example.com/page/") == "https://example.com/page"
    assert cr.normalize_url("https://example.com/page") == "https://example.com/page"


def test_extract_visits_dedups_fragment_slash_case_variants():
    nodes = {
        "a": _visit_node("a", "https://ex.com/x"),
        "b": _visit_node("b", "https://EX.com/x/"),
        "c": _visit_node("c", "https://ex.com/x#frag"),
        "d": _visit_node("d", "https://ex.com/y"),
    }
    result = cr.extract_visits({"graph": _graph(nodes)})
    assert result is not None
    n_visits, urls = result
    assert n_visits == 4
    assert urls == {"https://ex.com/x", "https://ex.com/y"}


def test_extract_visits_skips_failed_and_non_visit_nodes():
    nodes = {
        "a": _visit_node("a", "https://ex.com/x", success=True),
        "b": _visit_node("b", "https://ex.com/z", success=False),
        "c": _non_visit_node("c"),
    }
    n_visits, urls = cr.extract_visits({"graph": _graph(nodes)})
    assert n_visits == 1
    assert urls == {"https://ex.com/x"}


# ---------------------------------------------------------------------------
# missing graph / missing url key -> None, not 0
# ---------------------------------------------------------------------------

def test_extract_visits_none_when_graph_missing():
    assert cr.extract_visits({}) is None
    assert cr.extract_visits({"graph": None}) is None


def test_extract_visits_none_when_nodes_missing_or_malformed():
    assert cr.extract_visits({"graph": {"root_id": "r"}}) is None
    assert cr.extract_visits({"graph": {"nodes": "not-a-dict"}}) is None


def test_extract_visits_none_when_nodes_empty():
    # An empty `nodes` dict is the signature of a DAG-less engine variant (sequential_react /
    # langgraph_react) that never populates execution.graph -- not a real zero-visit run.
    assert cr.extract_visits({"graph": _graph({})}) is None


def test_extract_visits_none_when_a_visit_node_lacks_url_key():
    nodes = {
        "a": _visit_node("a", "https://ex.com/x"),
        "b": _visit_node("b", None, has_url_key=False),
    }
    assert cr.extract_visits({"graph": _graph(nodes)}) is None


def test_build_record_distinct_urls_none_propagates_and_is_not_zero(tmp_path):
    d = _cell("999", graph=_graph({}), visits=3)
    rec = cr.build_record(d, "f.json", "run", tests_dir=tmp_path)
    assert rec["distinct_urls"] is None
    assert rec["duplicate_factor"] is None


# ---------------------------------------------------------------------------
# duplicate-factor arithmetic
# ---------------------------------------------------------------------------

def test_duplicate_factor_arithmetic(tmp_path):
    nodes = {
        "a": _visit_node("a", "https://ex.com/x"),
        "b": _visit_node("b", "https://ex.com/x"),
        "c": _visit_node("c", "https://ex.com/y"),
    }
    d = _cell("1", graph=_graph(nodes), visits=6)
    rec = cr.build_record(d, "f.json", "run", tests_dir=tmp_path)
    assert rec["distinct_urls"] == 2
    assert rec["duplicate_factor"] == 3.0  # 6 reported visits / 2 distinct urls


def test_duplicate_factor_none_when_distinct_urls_zero_would_divide_by_zero(tmp_path):
    # distinct_urls == 0 can't actually happen given the "empty nodes -> None" rule above, but
    # the arithmetic itself must still guard division by zero defensively.
    d = _cell("1", graph=_graph({"a": _non_visit_node("a")}), visits=5)
    rec = cr.build_record(d, "f.json", "run", tests_dir=tmp_path)
    assert rec["distinct_urls"] == 0
    assert rec["duplicate_factor"] is None


# ---------------------------------------------------------------------------
# infra_failed quarantine
# ---------------------------------------------------------------------------

def test_summarize_drops_infra_failed_cells(tmp_path):
    nodes = {"a": _visit_node("a", "https://ex.com/x")}
    recs = [
        cr.build_record(_cell("1", graph=_graph(nodes), visits=1, score=0.9), "f1", "run",
                        tests_dir=tmp_path),
        cr.build_record(_cell("2", graph=_graph(nodes), visits=1, score=0.1, infra_failed=True),
                        "f2", "run", tests_dir=tmp_path),
    ]
    summ = cr.summarize(recs)
    assert summ["n_cells"] == 1
    assert summ["n_infra_dropped"] == 1
    assert summ["score"]["mean"] == 0.9


# ---------------------------------------------------------------------------
# --exact anchoring (reused from compare_arms._result_files_for_id)
# ---------------------------------------------------------------------------

def test_exact_prevents_shorter_arm_matching_longer_one(tmp_path):
    _write(tmp_path, "bfx_r1_q7_good_adaptive", 1, "100", score=0.8)
    _write(tmp_path, "bfx_r1_q7_good_adaptive_breadth", 1, "100", score=0.2)

    loose, _ = cr.load_arm_records("bfx_r1_q7_good_adaptive", str(tmp_path))
    assert len(loose) == 2  # loose glob over-matches

    exact, _ = cr.load_arm_records("bfx_r1_q7_good_adaptive_rep1", str(tmp_path), exact=True)
    assert len(exact) == 1
    assert exact[0]["score"] == 0.8


def test_load_arm_records_skips_summary_and_reports_unreadable(tmp_path):
    _write(tmp_path, "runA", 1, "100", score=0.8)
    with open(tmp_path / "runA_summary.json", "w") as fh:
        json.dump({"bogus": True}, fh)
    with open(tmp_path / "runA_rep2_100_model_engine_cfgabc_r1.json", "w") as fh:
        fh.write("{not valid json")
    recs, unreadable = cr.load_arm_records("runA", str(tmp_path))
    assert len(recs) == 1
    assert len(unreadable) == 1


# ---------------------------------------------------------------------------
# A/B pairing by (task, rep)
# ---------------------------------------------------------------------------

def test_paired_metric_deltas_pairs_by_task_and_rep(tmp_path):
    nodes_a = _graph({"a": _visit_node("a", "https://ex.com/x"),
                       "b": _visit_node("b", "https://ex.com/y")})
    nodes_b = _graph({"a": _visit_node("a", "https://ex.com/x")})
    rec_a = cr.build_record(_cell("100", graph=nodes_a, visits=4, score=0.9), "fa",
                            "runA_rep1", tests_dir=tmp_path)
    rec_b = cr.build_record(_cell("100", graph=nodes_b, visits=4, score=0.5), "fb",
                            "runB_rep1", tests_dir=tmp_path)
    idx_a, idx_b = cr.index_records([rec_a]), cr.index_records([rec_b])
    deltas, used = cr.paired_metric_deltas(idx_a, idx_b, "distinct_urls")
    assert deltas == [1]  # 2 distinct - 1 distinct
    assert used == [("100", rec_a["rep"])]
    assert rec_a["rep"] == rec_b["rep"] == (1, None)


def test_paired_metric_deltas_excludes_infra_failed_and_unpaired_keys(tmp_path):
    nodes = _graph({"a": _visit_node("a", "https://ex.com/x")})
    rec_a1 = cr.build_record(_cell("100", graph=nodes, visits=1, score=0.9), "fa", "runA_rep1",
                             tests_dir=tmp_path)
    rec_a2 = cr.build_record(_cell("200", graph=nodes, visits=1, score=0.9), "fa2", "runA_rep1",
                             tests_dir=tmp_path)  # only-in-A key
    rec_b1 = cr.build_record(_cell("100", graph=nodes, visits=1, score=0.1, infra_failed=True),
                             "fb", "runB_rep1", tests_dir=tmp_path)
    idx_a, idx_b = cr.index_records([rec_a1, rec_a2]), cr.index_records([rec_b1])
    deltas, used = cr.paired_metric_deltas(idx_a, idx_b, "score")
    assert deltas == []  # the only shared key is infra_failed on B's side
    assert used == []


# ---------------------------------------------------------------------------
# n_items derivation
# ---------------------------------------------------------------------------

_TEST_SOURCE = '''
ENTITIES = ["a", "b", "c"]


def validate_coverage(result, observability):
    n = len(ENTITIES)
    return {"check": "coverage", "passed": True, "score": 1.0, "reason": f"{n}/{n} gathered"}
'''


def test_derive_n_items_from_test_source_ast(tmp_path):
    (tmp_path / "test_042_widget_thing.py").write_text(_TEST_SOURCE)
    n_items, source = cr.derive_n_items_from_test_source("042", tests_dir=tmp_path)
    assert n_items == 3
    assert source == "ast:ENTITIES"


def test_derive_n_items_falls_back_to_reason_string(tmp_path):
    grep_validations = [{"check": "coverage", "reason": "2/7 (x, y) gathered"}]
    n_items, source = cr.derive_n_items("999", grep_validations, tests_dir=tmp_path)
    assert n_items == 7
    assert source == "reason"


def test_derive_n_items_none_when_undeliverable(tmp_path):
    n_items, source = cr.derive_n_items("999", [], tests_dir=tmp_path)
    assert n_items is None
    assert source is None


# ---------------------------------------------------------------------------
# task_coverage_score / coverage_ratio_unreliable extraction
# ---------------------------------------------------------------------------

def test_build_record_carries_coverage_ratio_and_task_coverage_score(tmp_path):
    d = _cell("1", graph=_graph({"a": _visit_node("a", "https://ex.com/x")}), visits=1,
             coverage_ratio=1.0,
             grep_validations=[{"check": "coverage", "score": 0.4, "reason": "2/5 gathered"}])
    rec = cr.build_record(d, "f.json", "run", tests_dir=tmp_path)
    assert rec["coverage_ratio_unreliable"] == 1.0
    assert rec["task_coverage_score"] == 0.4
    assert rec["n_items"] == 5
    assert rec["deficit"] == 4  # 5 required - 1 distinct


# ---------------------------------------------------------------------------
# resolve_task_ids / apply_filters
# ---------------------------------------------------------------------------

def test_resolve_task_ids_alias_and_literal_list():
    assert "134" in cr.resolve_task_ids("core24")
    assert cr.resolve_task_ids("100, 200") == {"100", "200"}
    assert cr.resolve_task_ids(None) is None


def test_resolve_task_ids_raises_on_unparseable_spec():
    import pytest
    with pytest.raises(ValueError):
        cr.resolve_task_ids("   ")


def test_apply_filters_variant_and_task_ids(tmp_path):
    nodes = _graph({"a": _visit_node("a", "https://ex.com/x")})
    r1 = cr.build_record(_cell("100", graph=nodes, visits=1, variant="graph"), "f1", "run",
                         tests_dir=tmp_path)
    r2 = cr.build_record(_cell("200", graph=nodes, visits=1, variant="sequential_react"), "f2",
                         "run", tests_dir=tmp_path)
    out = cr.apply_filters([r1, r2], variant="graph")
    assert out == [r1]
    out = cr.apply_filters([r1, r2], task_ids={"200"})
    assert out == [r2]
