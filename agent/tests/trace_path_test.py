"""
Bug C regression tests: raw benchmark traces were silently lost for every run whose
model id contains a slash (e.g. OpenRouter ids like ``openai/gpt-5-mini``).

Root cause: ``trace_path = results_dir / f"..._{model_name}_....jsonl"`` interpolated
the RAW model id straight into the filename. ``TraceRecorder.__init__`` then does
``self.path.parent.mkdir(parents=True, exist_ok=True)`` before ``open(path, "a")`` —
for a model id with a ``/`` this silently creates an unintended subdirectory (e.g.
``agent/idea_test_results/<run>_openai/``) and writes the trace *inside* it instead
of raising. No exception is swallowed anywhere; the mkdir+open sequence just succeeds
against a path nobody intended. Combined with every execution module unconditionally
deleting the trace file on a successful run (``trace_path.unlink()``, by design, once
the rolled-up ``observability.timings`` histogram has captured what's needed), the
leftover artifact for a slash-containing model id is an orphaned, permanently empty
directory rather than nothing at all -- and on any run that raises before that unlink
step, the trace still exists but is unexpectedly nested one level deeper than callers
expect.

The fix (``agent.app.trace_recorder.sanitize_path_component``) collapses path
separators and whitespace into a single ``-`` so the composed trace filename is
always one path component, for every execution variant (react/graph, sequential
react, langgraph react, naive-discretion, compiled). ``:`` is left untouched by
default -- it is not a path separator on Linux/macOS, and preserving it keeps the
trace filename's model-id spelling identical to the result-JSON filename's
(``idea_test_runner.py``'s ``safe_model = normalized.replace("/", "-")``), instead
of the two conventions disagreeing on one cell (e.g. trace
``openai_gpt-5-mini`` vs result ``openai-gpt-5-mini`` pre-fix).
"""
import os
from pathlib import Path

import pytest

from agent.app.trace_recorder import TraceRecorder, sanitize_path_component


class TestSanitizePathComponent:
    def test_slash_separated_openrouter_id_has_no_path_separator(self):
        result = sanitize_path_component("openai/gpt-5-mini")
        assert "/" not in result
        assert "\\" not in result

    def test_slash_separated_id_is_still_identifying(self):
        result = sanitize_path_component("openai/gpt-5-mini")
        # Readable and unambiguous -- not a hash -- and both halves of the original
        # id are still present. "-" (not "_") to match the result-JSON filename
        # convention, where "_" is the field delimiter.
        assert result == "openai-gpt-5-mini"

    def test_multiple_slashes_all_replaced(self):
        result = sanitize_path_component("vendor/family/model-name")
        assert "/" not in result
        assert result == "vendor-family-model-name"

    def test_ordinary_model_id_without_slash_is_unchanged(self):
        # No regression for plain OpenRouter-style ids with no filesystem-hostile chars.
        assert sanitize_path_component("gpt-5-mini") == "gpt-5-mini"
        assert sanitize_path_component("claude-opus-5") == "claude-opus-5"

    def test_ollama_colon_tag_is_preserved_by_default(self):
        # qwen2.5:7b -- ":" is NOT a path separator on Linux (only "/" is), so it is
        # left untouched by default. This keeps the trace filename's model-id
        # spelling identical to the result-JSON filename's (idea_test_runner.py's
        # safe_model = normalized.replace("/", "-") also leaves ":" alone).
        result = sanitize_path_component("qwen2.5:7b")
        assert result == "qwen2.5:7b"

    def test_ollama_colon_tag_can_be_stricter_sanitized_on_request(self):
        # preserve_colon=False restores the original, stricter behavior for a
        # caller that genuinely needs portability (e.g. Windows/NTFS).
        result = sanitize_path_component("qwen2.5:7b", preserve_colon=False)
        assert ":" not in result
        assert result == "qwen2.5-7b"

    def test_ollama_colon_tag_without_other_hostile_chars_stable_shape(self):
        # No accidental over-sanitization: dots and dashes, which are filesystem-safe
        # and appear throughout real model ids, must survive untouched.
        result = sanitize_path_component("llama3.2:3b-instruct")
        assert result == "llama3.2:3b-instruct"

    def test_backslash_and_spaces_are_sanitized(self):
        result = sanitize_path_component("weird vendor\\model name")
        assert "\\" not in result
        assert " " not in result

    def test_empty_string_falls_back_to_placeholder(self):
        assert sanitize_path_component("") == "unknown"

    def test_leading_trailing_hostile_chars_are_stripped_not_left_as_underscores(self):
        assert sanitize_path_component("/openai/gpt-5-mini/") == "openai-gpt-5-mini"

    def test_custom_replacement_char(self):
        assert sanitize_path_component("openai/gpt-5-mini", replacement="_") == "openai_gpt-5-mini"


class TestTraceRecorderEndToEnd:
    """Prove a slash-containing model id produces a single readable file, not a
    directory that swallows the trace."""

    def test_sanitized_slash_model_id_writes_one_file_not_a_directory(self, tmp_path):
        results_dir = tmp_path / "idea_test_results"
        results_dir.mkdir()
        model_name = "openai/gpt-5-mini"

        # This mirrors the fixed composition in execution.py / execution_sequential.py /
        # execution_langgraph.py / execution_naive_discretion.py / execution_compiled.py.
        trace_path = results_dir / f"run1_031_{sanitize_path_component(model_name)}_graph.jsonl"

        tracer = TraceRecorder(trace_path)
        tracer.record("event_a", {"k": "v"})
        tracer.record("event_b", {"k": "v2"})
        tracer.close()

        assert trace_path.is_file()
        assert not trace_path.is_dir()
        # No nested subdirectory was created under results_dir -- everything the raw
        # model id would have implied lives in one flat filename.
        assert list(results_dir.iterdir()) == [trace_path]

        lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_unsanitized_slash_model_id_reproduces_the_original_bug(self, tmp_path):
        """Documents the bug this fix closes: composing the path with the RAW model id
        (no sanitization) creates a subdirectory and nests the trace file inside it,
        instead of the single flat file callers expect."""
        results_dir = tmp_path / "idea_test_results"
        results_dir.mkdir()
        model_name = "openai/gpt-5-mini"

        trace_path = results_dir / f"run1_031_{model_name}_graph.jsonl"  # unsanitized, pre-fix shape

        tracer = TraceRecorder(trace_path)
        tracer.record("event_a", {})
        tracer.close()

        # The "openai" segment became a real directory, not part of a filename.
        openai_dir = results_dir / "run1_031_openai"
        assert openai_dir.is_dir()
        assert (openai_dir / "gpt-5-mini_graph.jsonl").is_file()

    def test_colon_model_id_writes_one_file_not_a_directory(self, tmp_path):
        results_dir = tmp_path / "idea_test_results"
        results_dir.mkdir()
        model_name = "qwen2.5:7b"

        trace_path = results_dir / f"run1_014_{sanitize_path_component(model_name)}_sequential_react.jsonl"
        tracer = TraceRecorder(trace_path)
        tracer.record("event", {})
        tracer.close()

        assert trace_path.is_file()
        assert list(results_dir.iterdir()) == [trace_path]
