"""Regression test: idea_test_runner.py's result-JSON filename composition
(``safe_model = sanitize_model_component(normalized)``) must produce a
byte-identical string to the old, hand-rolled ``normalized.replace("/", "-")``
for every real model id shape the suite runs against.

Context: a separate investigation established that the trace-file sanitizer
(``sanitize_path_component``, ``_``-based, colon-collapsing) must NOT be applied
to result filenames -- 8 analysis scripts hardcode colon-bearing model tokens in
globs/regexes, ~1737 existing colon-bearing artifacts + 1171 provider-hyphen ones
would stop matching, and ``_`` is the field delimiter in the result filename
template. The only approved change here is routing the *existing* "/" -> "-"
convention through the shared helper (``sanitize_model_component``) instead of
three independent ``.replace("/", "-")`` copies -- never changing its output.
"""
from agent.app.trace_recorder import sanitize_model_component

REPRESENTATIVE_MODEL_IDS = [
    "openai/gpt-5-mini",
    "qwen2.5:7b",
    "google/gemini-3.1-pro-preview",
    "gpt-5-mini",
    "anthropic/claude-opus-5",
    "llama3.2:3b-instruct",
    "vendor/family/model-name",
]


def _old_safe_model(normalized: str) -> str:
    """The exact expression idea_test_runner.py used before this change."""
    return normalized.replace("/", "-")


class TestResultFilenameByteIdentical:
    def test_sanitize_model_component_matches_old_replace_for_every_representative_id(self):
        for model_id in REPRESENTATIVE_MODEL_IDS:
            assert sanitize_model_component(model_id) == _old_safe_model(model_id), (
                f"result-filename convention drifted for {model_id!r}: "
                f"{sanitize_model_component(model_id)!r} != {_old_safe_model(model_id)!r}"
            )

    def test_full_result_filename_template_unchanged(self):
        # Mirrors idea_test_runner.py's
        # f"{run_id}_{test_id}_{safe_model}_{execution_variant}{tier_tag}{cfg_tag}_r{repeat_index}.json"
        run_id, test_id, variant, repeat_index = "20260824_120000", "031", "graph", 1
        for model_id in REPRESENTATIVE_MODEL_IDS:
            old = f"{run_id}_{test_id}_{_old_safe_model(model_id)}_{variant}_r{repeat_index}.json"
            new = f"{run_id}_{test_id}_{sanitize_model_component(model_id)}_{variant}_r{repeat_index}.json"
            assert old == new
