"""Tests for scripts/sample_unverified_quotes.py (Phase 0d of the ledger/DAG replan).

Everything runs against a synthetic results dir built in ``tmp_path`` (2 models x 2 variants),
never against ``agent/idea_test_results/`` -- the corpus-dependent-test trap is a known failure
mode in this repo (see feedback_verify_layers_on_a_real_cell), so these must pass in a bare
worktree with no stored cells at all.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "sample_unverified_quotes.py"
_spec = importlib.util.spec_from_file_location("sample_unverified_quotes", _SCRIPT_PATH)
suq = importlib.util.module_from_spec(_spec)
sys.modules["sample_unverified_quotes"] = suq
_spec.loader.exec_module(suq)


# ==============================================================================================
# Synthetic corpus
# ==============================================================================================

_PAGE_TEXT = (
    "A" * 500
    + " The reinforced concrete chimney is about 40 m taller than the Inco Superstack. "
    + "B" * 500
)

# (model, variant) -> how many unverified extraction records exist in that stratum.
_STRATUM_SIZES = {
    ("qwen2.5:7b", "evidence_loop"): 40,
    ("qwen2.5:7b", "sequential_react"): 30,
    ("llama3.2:3b", "evidence_loop"): 20,
    ("llama3.2:3b", "sequential_react"): 10,
}


def _cell(test_id, model, variant, n_unverified, n_verified=1):
    extractions = []
    for i in range(n_unverified):
        extractions.append({
            "entity": f"E{i}", "field": "height", "value": f"{100 + i} metres",
            "verdict": "SUPPORTED", "source_url": "https://en.wikipedia.org/wiki/X",
            "quote": "reinforced concrete chimney is about 40 m taller",
            "quote_verified": False, "page_id": "p1", "quote_start": -1, "quote_end": -1,
            "quote_fail_reason": "absent", "unit": "metres",
        })
    for i in range(n_verified):
        extractions.append({
            "entity": f"V{i}", "field": "height", "value": "381 metres",
            "verdict": "SUPPORTED", "source_url": "https://en.wikipedia.org/wiki/Y",
            "quote": "AAA", "quote_verified": True, "page_id": "p1",
            "quote_start": 0, "quote_end": 3, "quote_fail_reason": None, "unit": "metres",
        })
    return {
        "model": model, "execution_variant": variant,
        "test_metadata": {"test_id": test_id},
        "execution": {"output": {
            "extractions": extractions,
            "pages": [{"page_id": "p1", "url": "https://en.wikipedia.org/wiki/X",
                       "content_hash": "h1", "chars": len(_PAGE_TEXT), "text": _PAGE_TEXT}],
        }},
    }


@pytest.fixture()
def corpus(tmp_path):
    """A results dir with one cell per stratum plus decoy summary/report files."""
    rd = tmp_path / "results"
    rd.mkdir()
    for idx, ((model, variant), size) in enumerate(sorted(_STRATUM_SIZES.items())):
        name = f"synth01_2{10 + idx:02d}_{model}_{variant}_cfgaaa_r1.json"
        (rd / name).write_text(json.dumps(_cell(f"2{10 + idx:02d}", model, variant, size)))
    # Decoys that the loader must skip: both match the loose `{run_id}_*_*.json` glob.
    (rd / "synth01_run_summary.json").write_text(json.dumps({"cells": ["junk"]}))
    (rd / "synth01_210_qwen2.5:7b_report_v3.json").write_text(json.dumps({"final_output": "x"}))
    return rd


# ==============================================================================================
# Loading
# ==============================================================================================

def test_summary_and_report_files_are_skipped(corpus):
    paths = suq.discover_cell_files(["synth01"], str(corpus))
    names = sorted(p.split("/")[-1] for p in paths)
    assert len(names) == 4, names
    assert not any("_summary" in n or "_report_" in n for n in names)


def test_candidate_rows_only_include_unverified_quotes(corpus):
    rows = suq.collect_candidates(["synth01"], str(corpus))
    assert len(rows) == sum(_STRATUM_SIZES.values())
    assert {r["model"] for r in rows} == {"qwen2.5:7b", "llama3.2:3b"}
    assert {r["variant"] for r in rows} == {"evidence_loop", "sequential_react"}
    assert all(r["entity"].startswith("E") for r in rows)
    assert all(r["quote_fail_reason"] == "absent" for r in rows)
    assert all(r["test_id"] and r["cell_file"] for r in rows)


def test_unreadable_file_does_not_abort_collection(corpus):
    (corpus / "synth01_299_bad_evidence_loop_cfgz_r1.json").write_text("{not json")
    rows = suq.collect_candidates(["synth01"], str(corpus))
    assert len(rows) == sum(_STRATUM_SIZES.values())


# ==============================================================================================
# Stratification
# ==============================================================================================

def test_stratification_is_proportional(corpus):
    rows = suq.collect_candidates(["synth01"], str(corpus))
    picked = suq.stratified_sample(rows, n=20, seed=7)
    assert len(picked) == 20
    counts = suq.stratum_counts(picked)
    # 40/30/20/10 out of 100 -> 8/6/4/2 at n=20.
    assert counts[("qwen2.5:7b", "evidence_loop")] == 8
    assert counts[("qwen2.5:7b", "sequential_react")] == 6
    assert counts[("llama3.2:3b", "evidence_loop")] == 4
    assert counts[("llama3.2:3b", "sequential_react")] == 2


def test_sample_larger_than_population_returns_everything(corpus):
    rows = suq.collect_candidates(["synth01"], str(corpus))
    picked = suq.stratified_sample(rows, n=500, seed=7)
    assert len(picked) == len(rows)


def test_remainder_is_redistributed_when_a_stratum_is_exhausted():
    rows = ([{"model": "m1", "variant": "v", "i": i} for i in range(100)]
            + [{"model": "m2", "variant": "v", "i": i} for i in range(2)])
    picked = suq.stratified_sample(rows, n=50, seed=1)
    assert len(picked) == 50
    counts = suq.stratum_counts(picked)
    assert counts[("m2", "v")] <= 2
    assert counts[("m1", "v")] + counts[("m2", "v")] == 50


def test_seed_is_deterministic_and_seeds_differ(corpus):
    rows = suq.collect_candidates(["synth01"], str(corpus))
    a = suq.stratified_sample(rows, n=20, seed=7)
    b = suq.stratified_sample(rows, n=20, seed=7)
    c = suq.stratified_sample(rows, n=20, seed=8)
    key = lambda rs: [(r["cell_file"], r["entity"]) for r in rs]
    assert key(a) == key(b)
    assert key(a) != key(c)


# ==============================================================================================
# Window extraction
# ==============================================================================================

def test_window_centres_on_an_exact_quote_match():
    window, method = suq.locate_window(_PAGE_TEXT, "reinforced concrete chimney", radius=50)
    assert method == "exact"
    assert "reinforced concrete chimney" in window
    assert len(window) < 200
    assert not window.startswith("A" * 100)


def test_window_tolerates_whitespace_and_case_drift():
    window, method = suq.locate_window(_PAGE_TEXT, "REINFORCED   concrete\n chimney", radius=40)
    assert method in {"normalized", "fuzzy"}
    assert "concrete chimney" in window


def test_window_falls_back_to_page_head_when_quote_is_absent():
    window, method = suq.locate_window(_PAGE_TEXT, "zzz nothing like this at all zzz", radius=300)
    assert method == "page_head"
    assert window == _PAGE_TEXT[:600]


def test_window_reports_missing_page():
    window, method = suq.locate_window("", "anything")
    assert method == "no_page"
    assert window == ""


# ==============================================================================================
# CSV emission
# ==============================================================================================

def test_csv_has_the_audit_columns_and_empty_human_fields(corpus, tmp_path):
    rows = suq.collect_candidates(["synth01"], str(corpus))
    picked = suq.stratified_sample(rows, n=8, seed=3)
    out = tmp_path / "sample.csv"
    suq.write_csv(picked, str(out))
    with open(out, newline="", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 8
    assert list(got[0]) == list(suq.CSV_COLUMNS)
    for r in got:
        assert r["label"] == "" and r["supporting_span"] == "" and r["notes"] == ""
        assert "concrete chimney" in r["page_window"]
        assert r["window_method"] == "exact"


def test_main_end_to_end(corpus, tmp_path, capsys):
    out = tmp_path / "e2e.csv"
    rc = suq.main(["--prefixes", "synth01", "--n", "12", "--seed", "5",
                   "--out", str(out), "--results-dir", str(corpus)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "candidate rows" in text
    with open(out, newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 12
