"""Unit tests for scripts/build_corpus.py -- harvesting a frozen search corpus from run results.

The first corpus should cost nothing: hundreds of stored cells already carry the pages the agent
visited and the URLs it cited. Harvesting those turns existing spend into a reusable, deterministic
evidence universe, so a live recording pass is a top-up rather than a prerequisite.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_corpus  # noqa: E402


def _cell(tmp_path, name, pages=None, extractions=None):
    """Write a per-cell result JSON in the real nested shape the runner emits."""
    payload = {"execution": {"output": {
        "pages": pages or [],
        "extractions": extractions or [],
    }}}
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


PAGE = {"page_id": "p1", "url": "https://example.org/eiffel",
        "text": "The Eiffel Tower is 330 m tall.", "content_hash": "abc", "truncated": False}


def test_harvest_reads_stored_pages_from_a_result_cell(tmp_path):
    _cell(tmp_path, "run_122_model_variant_r1.json", pages=[PAGE])
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert len(docs) == 1
    assert docs[0]["url"] == "https://example.org/eiffel"
    assert "Eiffel Tower" in docs[0]["text"]


def test_harvest_counts_only_canonical_result_files(tmp_path):
    """A naive glob over the results dir is inflated by summary and trace files.

    ``*_summary.json`` reflects only the last cell of a multi-invocation run and ``*.jsonl`` are
    traces; counting either once produced a throughput figure 2.1x too high.
    """
    _cell(tmp_path, "run_122_model_variant_r1.json", pages=[PAGE])
    _cell(tmp_path, "run_summary.json", pages=[dict(PAGE, url="https://example.org/summary")])
    (tmp_path / "run_122_model_variant_r1.jsonl").write_text("{}\n", encoding="utf-8")
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert [doc["url"] for doc in docs] == ["https://example.org/eiffel"]


def test_harvest_deduplicates_the_same_url_across_cells(tmp_path):
    """The same page appears in every cell that visited it; the corpus should hold it once."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[PAGE])
    _cell(tmp_path, "b_122_m_v_r1.json", pages=[PAGE])
    assert len(build_corpus.harvest_documents(str(tmp_path))) == 1


def test_harvest_prefers_the_longest_text_for_a_repeated_url(tmp_path):
    """Cells truncate pages differently; the corpus should keep the most complete copy."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[dict(PAGE, text="short")])
    _cell(tmp_path, "b_122_m_v_r1.json", pages=[dict(PAGE, text="a considerably longer body")])
    docs = build_corpus.harvest_documents(str(tmp_path))
    assert docs[0]["text"] == "a considerably longer body"


def test_harvest_skips_pages_without_a_url(tmp_path):
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[dict(PAGE, url="")])
    assert build_corpus.harvest_documents(str(tmp_path)) == []


def test_harvest_survives_a_corrupt_result_file(tmp_path):
    """One unreadable cell must not abort an unattended corpus build."""
    _cell(tmp_path, "a_122_m_v_r1.json", pages=[PAGE])
    (tmp_path / "b_122_m_v_r1.json").write_text("{not json", encoding="utf-8")
    assert len(build_corpus.harvest_documents(str(tmp_path))) == 1


def test_write_corpus_emits_documents_jsonl_readable_by_the_backend(tmp_path):
    """The builder's output must load straight into the connector without a translation step."""
    from agent.app.connector_search_corpus import load_documents

    out = tmp_path / "corpus"
    build_corpus.write_corpus(str(out), [
        {"url": "https://example.org/a", "title": "A", "description": "d", "text": "body"}])
    loaded = load_documents(str(out))
    assert len(loaded) == 1
    assert loaded[0].url == "https://example.org/a"
