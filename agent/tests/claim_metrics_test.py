"""Unit tests for scripts/claim_metrics.py.

Pipeline-edge precision/recall (retrieved -> extracted -> verified -> stated) follows
RAGChecker's decomposition (SS3.2/SS3.3, see the module docstring), computed from the
mechanically-verified fields ``execution_evidence_loop.py`` already produces -- never a fresh
LLM judge call. Also covers the fabricated-arithmetic rate built on Lane A's evidence graph.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import claim_metrics  # noqa: E402


def _page(page_id, url, text):
    return {"page_id": page_id, "url": url, "content_hash": "h", "chars": len(text),
           "stored_chars": len(text), "truncated": False, "text": text}


def _extraction(entity, field, value, *, page_id="p1", value_verified=True,
               quote_verified=False):
    return {"entity": entity, "field": field, "value": value, "verdict": "SUPPORTED",
           "source_url": "https://example.com", "quote": f'"{value}"',
           "quote_verified": quote_verified, "page_id": page_id, "quote_start": -1,
           "quote_end": -1, "quote_fail_reason": None, "unit": "", "value_verified": value_verified,
           "value_unit_bearing": False, "value_shape": "number", "value_fail_reason": None}


# ---------------------------------------------------------------------------------------------
# pipeline_edges: single-cell computation
# ---------------------------------------------------------------------------------------------

def test_returns_none_for_output_without_extractions_field():
    assert claim_metrics.pipeline_edges({"final_deliverable": "x"}) is None


def test_full_pipeline_all_stages_pass():
    output = {
        "final_deliverable": "Denali is 20310 feet tall.",
        "pages": [_page("p1", "https://example.com/denali", "Denali stands at 20310 feet.")],
        "extractions": [_extraction("Denali", "elevation_ft", "20310")],
    }
    edges = claim_metrics.pipeline_edges(output)
    assert edges["retrieved"] == 1
    assert edges["extracted"] == 1
    assert edges["verified"] == 1
    assert edges["stated"] == 1
    assert edges["orphaned_extractions"] == 0
    assert edges["retrieved_to_extracted_recall"] == 1.0
    assert edges["retrieved_to_extracted_precision"] == 1.0
    assert edges["extracted_to_verified_rate"] == 1.0
    assert edges["verified_to_stated_recall"] == 1.0


def test_verified_but_not_stated_when_final_text_omits_the_value():
    output = {
        "final_deliverable": "The mountain is quite tall.",  # value never actually stated
        "pages": [_page("p1", "https://example.com/denali", "Denali stands at 20310 feet.")],
        "extractions": [_extraction("Denali", "elevation_ft", "20310")],
    }
    edges = claim_metrics.pipeline_edges(output)
    assert edges["verified"] == 1
    assert edges["stated"] == 0
    assert edges["verified_to_stated_recall"] == 0.0


def test_extracted_but_not_verified_when_value_verified_false():
    output = {
        "final_deliverable": "The value is 999.",
        "pages": [_page("p1", "https://example.com/x", "unrelated content")],
        "extractions": [_extraction("X", "field", "999", value_verified=False)],
    }
    edges = claim_metrics.pipeline_edges(output)
    assert edges["extracted"] == 1
    assert edges["verified"] == 0
    assert edges["extracted_to_verified_rate"] == 0.0


def test_orphaned_extraction_citing_a_page_never_retrieved_is_flagged():
    output = {
        "final_deliverable": "x",
        "pages": [_page("p1", "https://example.com/a", "text a")],
        "extractions": [_extraction("X", "f", "1234", page_id="p_nonexistent")],
    }
    edges = claim_metrics.pipeline_edges(output)
    assert edges["orphaned_extractions"] == 1
    assert edges["retrieved_to_extracted_precision"] == 0.0


def test_extractions_with_blank_value_are_not_counted_as_extracted():
    output = {
        "final_deliverable": "x",
        "pages": [_page("p1", "https://example.com/a", "text a")],
        "extractions": [_extraction("X", "f", "")],
    }
    edges = claim_metrics.pipeline_edges(output)
    assert edges["extracted"] == 0


def test_stated_claim_precision_penalizes_fabricated_numbers_in_final_text():
    # Final text asserts TWO numbers; only one is backed by a verified extraction.
    output = {
        "final_deliverable": "The elevation is 20310 feet and 500000 people visit yearly.",
        "pages": [_page("p1", "https://example.com/denali", "Denali stands at 20310 feet.")],
        "extractions": [_extraction("Denali", "elevation_ft", "20310")],
    }
    edges = claim_metrics.pipeline_edges(output)
    assert edges["stated_claim_count"] == 2
    assert edges["stated_claim_precision"] == 0.5


# ---------------------------------------------------------------------------------------------
# aggregate_pipeline_edges
# ---------------------------------------------------------------------------------------------

def test_aggregate_skips_cells_without_extraction_records():
    cells = [
        {"output": {"final_deliverable": "x"}},  # no extractions field -> excluded
        {"output": {
            "final_deliverable": "20310",
            "pages": [_page("p1", "https://example.com/a", "value is 20310")],
            "extractions": [_extraction("A", "f", "20310")],
        }},
    ]
    agg = claim_metrics.aggregate_pipeline_edges(cells)
    assert agg["n"] == 1
    assert agg["total_extracted"] == 1


def test_aggregate_empty_input_reports_zero_n():
    assert claim_metrics.aggregate_pipeline_edges([]) == {"n": 0}


def test_aggregate_is_micro_averaged_not_macro():
    # Cell 1: 1/1 verified. Cell 2: 0/3 verified. Micro rate = 1/4, not the macro mean of 1.0
    # and 0.0 (0.5).
    cells = [
        {"output": {"final_deliverable": "x",
                   "pages": [_page("p1", "https://example.com/a", "5")],
                   "extractions": [_extraction("A", "f", "5")]}},
        {"output": {"final_deliverable": "x",
                   "pages": [_page("p1", "https://example.com/b", "nothing relevant")],
                   "extractions": [_extraction("B", "f", "1", value_verified=False),
                                  _extraction("B", "g", "2", value_verified=False),
                                  _extraction("B", "h", "3", value_verified=False)]}},
    ]
    agg = claim_metrics.aggregate_pipeline_edges(cells)
    assert agg["total_extracted"] == 4
    assert agg["total_verified"] == 1


# ---------------------------------------------------------------------------------------------
# derivation_fabrication_rate
# ---------------------------------------------------------------------------------------------

def test_fabrication_rate_none_without_evidence_graph():
    assert claim_metrics.derivation_fabrication_rate({"final_deliverable": "x"}) is None


def test_fabrication_rate_none_when_graph_has_no_derived_nodes():
    output = {"evidence_graph": {"nodes": [{"kind": "source", "derivation_valid": None}]}}
    assert claim_metrics.derivation_fabrication_rate(output) is None


def test_fabrication_rate_computed_from_derived_node_validity():
    output = {"evidence_graph": {"nodes": [
        {"kind": "source", "derivation_valid": None},
        {"kind": "derived", "derivation_valid": True},
        {"kind": "derived", "derivation_valid": False},
        {"kind": "derived", "derivation_valid": False},
    ]}}
    assert claim_metrics.derivation_fabrication_rate(output) == 2 / 3


def test_fabrication_rate_treats_unassessed_as_not_fabricated_but_not_valid_either():
    # derivation_valid=None (never checked) is neither counted as fabricated nor as valid.
    output = {"evidence_graph": {"nodes": [
        {"kind": "derived", "derivation_valid": None},
        {"kind": "derived", "derivation_valid": True},
    ]}}
    assert claim_metrics.derivation_fabrication_rate(output) == 0.0


# ---------------------------------------------------------------------------------------------
# load_cells integration
# ---------------------------------------------------------------------------------------------

def _write_cell(path, *, variant, extractions=None, pages=None):
    payload = {
        "test_metadata": {"test_id": "999"}, "model": "test-model",
        "execution_variant": variant,
        "execution": {"output": {"final_deliverable": "x", "extractions": extractions or [],
                                 "pages": pages or []}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_cells_scopes_by_run_id_and_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(claim_metrics.bench_common, "results_dir", lambda: tmp_path)
    _write_cell(tmp_path / "myrun_999_m_evidence_loop_cfgabc_r1.json", variant="evidence_loop")
    _write_cell(tmp_path / "myrun_999_m_graph_cfgabc_r1.json", variant="graph")
    cells = claim_metrics.load_cells(run_ids=["myrun"], variants=["evidence_loop"])
    assert len(cells) == 1
    assert cells[0]["variant"] == "evidence_loop"
