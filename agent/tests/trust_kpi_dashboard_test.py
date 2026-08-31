"""Tests for scripts/trust_kpi_dashboard.py -- the harm-avoidance + accuracy trust KPI table.

Fixture-driven, mirroring agent/tests/kpi_dashboard_test.py's ``_cell`` convention -- no
dependency on files in agent/idea_test_results/. Covers each metric's per-cell extractor, the
missing-key/coverage discipline (a metric returns None -- never a fabricated 0/False/0.0 -- when
its inputs are absent), and the accuracy/safety pairing (a metric is never reported alone).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import trust_kpi_dashboard as tkd  # noqa: E402


def _cell(test_id="100", model="qwen2.5:7b", variant="evidence_loop", visit_count=1,
          grounded=True, score=0.5, success=True, final_deliverable="answer text",
          sources=None, unverified_citations=None, extractions=None, claim_provenance=None,
          ledger_verdict=None, goal_achieved=None, finalization_status=None,
          grounding_gate=None, usd=None, total_tokens=1000, seconds=10.0, infra_failed=False):
    out = {"final_deliverable": final_deliverable, "success": success, "grounded": grounded}
    if sources is not None:
        out["sources"] = sources
    if unverified_citations is not None:
        out["unverified_citations"] = unverified_citations
    if extractions is not None:
        out["extractions"] = extractions
    if claim_provenance is not None:
        out["claim_provenance"] = claim_provenance
    if ledger_verdict is not None:
        out["ledger_verdict"] = ledger_verdict
    if goal_achieved is not None:
        out["goal_achieved"] = goal_achieved
    if finalization_status is not None:
        out["finalization_status"] = finalization_status
    if grounding_gate is not None:
        out["grounding_gate"] = grounding_gate
    d = {
        "test_metadata": {"test_id": str(test_id)},
        "model": model,
        "execution_variant": variant,
        "execution": {
            "output": out,
            "graph": {"nodes": {}},
            "observability": {
                "visit": {"count": visit_count},
                "cost": {"usd": usd},
                "llm": {"total_tokens": total_tokens},
                "infra": {"failed": infra_failed},
            },
            "duration_seconds": seconds,
        },
        "validation": {"overall_score": score, "overall_passed": score is not None and score >= 0.5},
        "infra_failed": infra_failed,
    }
    return d


# ---------------------------------------------------------------------------
# extraction_quote_states / claim_provenance_quote_states -- missing-key discipline
# ---------------------------------------------------------------------------

def test_extraction_quote_states_none_when_field_absent():
    d = _cell()
    assert tkd.extraction_quote_states(d) is None


def test_extraction_quote_states_empty_list_when_written_empty():
    d = _cell(extractions=[])
    assert tkd.extraction_quote_states(d) == []


def test_extraction_quote_states_reads_tristate():
    d = _cell(extractions=[
        {"quote_verified": True}, {"quote_verified": False}, {"quote_verified": None},
    ])
    assert tkd.extraction_quote_states(d) == [True, False, None]


def test_claim_provenance_quote_states_none_when_absent():
    d = _cell()
    assert tkd.claim_provenance_quote_states(d) is None


def test_claim_provenance_quote_states_reads_tristate():
    d = _cell(claim_provenance=[{"quote_verified": False}, {"quote_verified": True}])
    assert tkd.claim_provenance_quote_states(d) == [False, True]


# ---------------------------------------------------------------------------
# pooled_quote_stats -- unsupported-claim rate, None never counted as failure
# ---------------------------------------------------------------------------

def test_pooled_quote_stats_counts_only_false_as_unsupported():
    cells = [
        _cell(test_id="1", extractions=[{"quote_verified": True}, {"quote_verified": False},
                                         {"quote_verified": None}]),
        _cell(test_id="2", extractions=[{"quote_verified": True}]),
    ]
    stats = tkd.pooled_quote_stats([c for c in cells], tkd.extraction_quote_states)
    assert stats["n_checked"] == 3  # True/False only, the None is excluded
    assert stats["n_unsupported"] == 1
    assert stats["n_unverifiable"] == 1
    assert math.isclose(stats["unsupported_rate"], 1 / 3)
    assert stats["cells_with_field"] == 2


def test_pooled_quote_stats_uncomputable_when_no_cell_has_field():
    cells = [_cell(test_id="1"), _cell(test_id="2")]
    stats = tkd.pooled_quote_stats(cells, tkd.extraction_quote_states)
    assert stats["cells_with_field"] == 0
    assert stats["n_checked"] == 0
    assert math.isnan(stats["unsupported_rate"])


def test_pooled_quote_stats_zero_extractions_still_counts_as_computed_cell():
    cells = [_cell(test_id="1", extractions=[])]
    stats = tkd.pooled_quote_stats(cells, tkd.extraction_quote_states)
    assert stats["cells_with_field"] == 1
    assert stats["n_checked"] == 0


# ---------------------------------------------------------------------------
# cited_never_fetched -- fabrication corroboration signal
# ---------------------------------------------------------------------------

def test_cited_never_fetched_true_when_unverified_present():
    d = _cell(sources=[{"url": "http://a"}], unverified_citations=["http://b"])
    assert tkd.cited_never_fetched(d) is True


def test_cited_never_fetched_false_when_all_visited():
    d = _cell(sources=[{"url": "http://a"}], unverified_citations=[])
    assert tkd.cited_never_fetched(d) is False


def test_cited_never_fetched_none_when_fields_absent():
    d = _cell()
    assert tkd.cited_never_fetched(d) is None


# ---------------------------------------------------------------------------
# quote_fail_present -- fabrication corroboration signal from extractions/claim_provenance
# ---------------------------------------------------------------------------

def test_quote_fail_present_true():
    d = _cell(extractions=[{"quote_verified": True}, {"quote_verified": False}])
    assert tkd.quote_fail_present(d) is True


def test_quote_fail_present_false():
    d = _cell(extractions=[{"quote_verified": True}])
    assert tkd.quote_fail_present(d) is False


def test_quote_fail_present_none_when_neither_field_present():
    d = _cell()
    assert tkd.quote_fail_present(d) is None


# ---------------------------------------------------------------------------
# verdict_calibration -- per (model,variant) label -> (mean_score, n)
# ---------------------------------------------------------------------------

def test_verdict_calibration_ledger_verdict():
    cells = [
        _cell(test_id="1", ledger_verdict="ANSWER", score=0.7),
        _cell(test_id="2", ledger_verdict="ANSWER", score=0.9),
        _cell(test_id="3", ledger_verdict="ABSTAIN", score=0.1),
    ]
    table, n_total = tkd.verdict_calibration(cells, "ledger_verdict")
    assert n_total == 3
    assert math.isclose(table["ANSWER"][0], 0.8)
    assert table["ANSWER"][1] == 2
    assert math.isclose(table["ABSTAIN"][0], 0.1)


def test_verdict_calibration_skips_missing_field():
    cells = [_cell(test_id="1"), _cell(test_id="2", ledger_verdict="ANSWER", score=0.5)]
    table, n_total = tkd.verdict_calibration(cells, "ledger_verdict")
    assert n_total == 1
    assert table == {"ANSWER": (0.5, 1)}


def test_verdict_calibration_constant_success_flagged_uninformative():
    cells = [_cell(test_id=str(i), success=True, score=0.1 * i) for i in range(3)]
    table, n_total = tkd.verdict_calibration(cells, "success")
    assert n_total == 3
    assert set(table.keys()) == {"True"}  # only ever one label -- zero discriminating info


# ---------------------------------------------------------------------------
# abstention correctness
# ---------------------------------------------------------------------------

def test_evidence_loop_declined_true():
    d = _cell(ledger_verdict="ABSTAIN")
    assert tkd.evidence_loop_declined(d) is True


def test_evidence_loop_declined_false():
    d = _cell(ledger_verdict="ANSWER")
    assert tkd.evidence_loop_declined(d) is False


def test_evidence_loop_declined_none_when_absent():
    d = _cell()
    assert tkd.evidence_loop_declined(d) is None


def test_graph_declined_reuses_grounding_gate():
    d = _cell(grounding_gate="refused-ungrounded")
    assert tkd.graph_declined(d) is True
    d2 = _cell()
    assert tkd.graph_declined(d2) is False


def test_abstention_correctness_good_and_bad_split():
    cells = [
        _cell(test_id="1", ledger_verdict="ABSTAIN", score=0.02),
        _cell(test_id="2", ledger_verdict="ABSTAIN", score=0.05),
        _cell(test_id="3", ledger_verdict="ABSTAIN", score=0.6),  # bad abstain
        _cell(test_id="4", ledger_verdict="ANSWER", score=0.9),  # not declined
    ]
    stats = tkd.abstention_correctness(cells, tkd.evidence_loop_declined)
    assert stats["n_declined_with_score"] == 3
    assert stats["n_bad_abstain"] == 1
    assert math.isclose(stats["bad_abstain_rate"], 1 / 3)
    assert stats["n_near_zero_abstain"] == 2


def test_abstention_correctness_uncomputable_when_no_declines():
    cells = [_cell(test_id="1", ledger_verdict="ANSWER", score=0.5)]
    stats = tkd.abstention_correctness(cells, tkd.evidence_loop_declined)
    assert stats["n_declined_with_score"] == 0
    assert math.isnan(stats["bad_abstain_rate"])


# ---------------------------------------------------------------------------
# cost_per_verified_claim -- never divide by a fabricated zero
# ---------------------------------------------------------------------------

def test_cost_per_verified_claim_pools_tokens_over_verified_count():
    cells = [
        _cell(test_id="1", total_tokens=1000,
              extractions=[{"quote_verified": True}, {"quote_verified": False}]),
        _cell(test_id="2", total_tokens=2000, extractions=[{"quote_verified": True}]),
    ]
    stats = tkd.cost_per_verified_claim(cells)
    assert stats["n_verified_total"] == 2  # one True in cell 1, one True in cell 2
    assert stats["cells_n"] == 2
    assert math.isclose(stats["tokens_per_verified_claim"], (1000 + 2000) / 2)


def test_cost_per_verified_claim_excludes_cells_with_zero_verified():
    cells = [_cell(test_id="1", total_tokens=1000, extractions=[{"quote_verified": False}])]
    stats = tkd.cost_per_verified_claim(cells)
    assert stats["cells_n"] == 0
    assert math.isnan(stats["tokens_per_verified_claim"])


def test_cost_per_verified_claim_uncomputable_when_no_claims_field():
    cells = [_cell(test_id="1", total_tokens=1000)]
    stats = tkd.cost_per_verified_claim(cells)
    assert stats["cells_n"] == 0
    assert math.isnan(stats["tokens_per_verified_claim"])


# ---------------------------------------------------------------------------
# safety-vs-accuracy flag: never praise safety without accuracy visible
# ---------------------------------------------------------------------------

def test_flag_tradeoffs_flags_safer_but_less_accurate():
    groups = {
        ("m1", "variant_a"): {"fabrication_rate": 0.1, "fabrication_n": 10,
                               "overall_score_mean": 0.3, "overall_score_n": 10},
        ("m1", "variant_b"): {"fabrication_rate": 0.5, "fabrication_n": 10,
                               "overall_score_mean": 0.6, "overall_score_n": 10},
    }
    flags = tkd.flag_safety_accuracy_tradeoffs(groups)
    assert len(flags) == 1
    assert "variant_a" in flags[0] and "variant_b" in flags[0]


def test_flag_tradeoffs_silent_when_both_improve():
    groups = {
        ("m1", "variant_a"): {"fabrication_rate": 0.1, "fabrication_n": 10,
                               "overall_score_mean": 0.6, "overall_score_n": 10},
        ("m1", "variant_b"): {"fabrication_rate": 0.5, "fabrication_n": 10,
                               "overall_score_mean": 0.3, "overall_score_n": 10},
    }
    flags = tkd.flag_safety_accuracy_tradeoffs(groups)
    assert flags == []


def test_flag_tradeoffs_skips_when_coverage_missing():
    groups = {
        ("m1", "variant_a"): {"fabrication_rate": float("nan"), "fabrication_n": 0,
                               "overall_score_mean": 0.3, "overall_score_n": 10},
        ("m1", "variant_b"): {"fabrication_rate": 0.5, "fabrication_n": 10,
                               "overall_score_mean": 0.6, "overall_score_n": 10},
    }
    flags = tkd.flag_safety_accuracy_tradeoffs(groups)
    assert flags == []


# ---------------------------------------------------------------------------
# aggregate_group + main -- end-to-end, reused loaders from kpi_dashboard
# ---------------------------------------------------------------------------

def _write(dirpath, run_id, test_id, variant="evidence_loop", suffix_r=1, **kw):
    d = _cell(test_id=test_id, variant=variant, **kw)
    fname = f"{run_id}_{variant}_{test_id}_model_engine_cfgabc_r{suffix_r}.json"
    with open(os.path.join(dirpath, fname), "w") as fh:
        json.dump(d, fh)
    return fname


def test_aggregate_group_bundles_all_metrics_with_coverage():
    cells = [
        {"file": "a", "data": _cell(test_id="1", visit_count=0, success=True,
                                     final_deliverable="x", score=0.1)},
        {"file": "b", "data": _cell(test_id="2", visit_count=1, success=True,
                                     final_deliverable="y", score=0.8,
                                     ledger_verdict="ANSWER")},
    ]
    g = tkd.aggregate_group(cells)
    assert g["total"] == 2
    assert g["fabrication_n"] == 2
    assert g["overall_score_n"] == 2
    assert "unsupported_extractions" in g
    assert "unsupported_claim_provenance" in g
    assert "abstention" in g
    assert "cost_per_verified_claim" in g


def test_main_runs_end_to_end(tmp_path, capsys):
    _write(tmp_path, "runX", "100", score=0.5, ledger_verdict="PARTIAL")
    _write(tmp_path, "runX", "101", suffix_r=2, score=0.9, ledger_verdict="ANSWER")
    rc = tkd.main(["--run-id", "runX", "--results-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TRUST KPI DASHBOARD" in out
    assert "qwen2.5:7b" in out


def test_main_requires_at_least_one_run_id(tmp_path):
    try:
        tkd.main(["--results-dir", str(tmp_path)])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert e.code != 0
