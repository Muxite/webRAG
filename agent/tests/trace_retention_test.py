"""Traces were deleted on every success, and their filenames could not tell cells apart.

Two defects that had to be fixed together:

* **Unconditional delete.** All six execution variants ended with a bare
  ``if trace_path.exists(): trace_path.unlink()`` on the success path, with no flag guarding
  any of them. A trace survived only when the run *crashed* before reaching the delete. The
  four-way baseline therefore had 0 of 96 cells with a recoverable trace, and the forensics
  pass had to reconstruct mechanism from result-graph topology alone.
* **Colliding filenames.** No trace path carried the repeat index or config fingerprint that
  the result JSON's name carries, while ``TraceRecorder`` opens in APPEND mode. Retaining
  traces without fixing the name would have silently interleaved concurrent cells into one
  corrupt file -- strictly worse than deleting them, because the corruption is invisible.

Retention is opt-in via ``IDEA_TEST_KEEP_TRACES`` so the default run keeps its current disk
footprint.

No network: paths and env are constructed directly.
"""
from __future__ import annotations

import pytest

from agent.app.trace_recorder import build_trace_path, traces_retained


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_KEEP_TRACES", raising=False)


def test_traces_are_deleted_by_default():
    assert traces_retained() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_retention_is_opt_in(monkeypatch, value):
    monkeypatch.setenv("IDEA_TEST_KEEP_TRACES", value)
    assert traces_retained() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "   "])
def test_falsey_values_do_not_retain(monkeypatch, value):
    monkeypatch.setenv("IDEA_TEST_KEEP_TRACES", value)
    assert traces_retained() is False


def test_cell_tag_disambiguates_repeats(tmp_path):
    """The collision that made retention unsafe: same cell, different repeat."""
    first = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "_cfgabc_r1")
    second = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "_cfgabc_r2")
    assert first != second


def test_cell_tag_disambiguates_config_fingerprints(tmp_path):
    covon = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "_cfgaaa_r1")
    covoff = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "_cfgbbb_r1")
    assert covon != covoff


def test_variant_and_model_still_disambiguate(tmp_path):
    graph = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "_r1")
    langgraph = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "langgraph_react", "_r1")
    other_model = build_trace_path(tmp_path, "run1", "152", "llama3.2:3b", "graph", "_r1")
    assert len({graph, langgraph, other_model}) == 3


def test_model_ids_with_separators_stay_one_path_component(tmp_path):
    """An OpenRouter id embeds a ``/`` -- unsanitized it silently creates a subdirectory."""
    path = build_trace_path(tmp_path, "run1", "152", "openai/gpt-5-mini", "graph", "_r1")
    assert path.parent == tmp_path
    assert "/" not in path.name


def test_extension_and_directory_are_stable(tmp_path):
    path = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "_r1")
    assert path.suffix == ".jsonl"
    assert path.parent == tmp_path


def test_an_empty_cell_tag_is_accepted(tmp_path):
    """Call sites that genuinely have no repeat context must still get a usable path."""
    path = build_trace_path(tmp_path, "run1", "152", "qwen2.5:7b", "graph", "")
    assert path.suffix == ".jsonl"
    assert "152" in path.name


def test_the_name_lines_up_with_the_result_json_naming(tmp_path):
    """Trace and result must be mappable to each other by name, not by guesswork."""
    tag = "_t2_cfgef66f4d7_r3"
    path = build_trace_path(tmp_path, "dagbase", "152", "qwen2.5:7b", "graph", tag)
    assert path.name == "dagbase_152_qwen2.5:7b_graph_t2_cfgef66f4d7_r3.jsonl"
