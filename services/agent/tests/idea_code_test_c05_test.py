"""
Adversarial offline checks for codebench task c05 (word-frequency-report) — no Docker,
no LLM. Mirrors idea_code_test_c01_test.py's structure: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids
reference real tests, the compiled plan is well-formed) BEFORE anything ever reaches a
live sandbox, and exercise materialize_task.py end-to-end against this task. c05 is
hidden, so get_sandbox_fixture() is empty — the canonical test content lives only in
get_grading_payload()["tests"].

Because c05's assertions compare directly against dict literals (not simple string
literals), "does the embedded test file's ground truth match an independent
reimplementation" is checked by actually EXECUTING the embedded test file's test
functions against a reference `word_frequency` (imported as a fake `word_freq` module)
rather than regex-scraping literals — every `assert` in the canonical test file must
pass for real when run against the independently-written reference implementation.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

from agent.app.idea_code_tests import test_c05_word_freq as c05


def _independent_word_frequency(text: str) -> dict:
    """Reimplemented independently of task's own prose spec (manual char-by-char
    scanning with str.isalnum() gated to ASCII, not regex), to catch a hand-typed error
    in the task module's embedded expected values."""
    freq: dict = {}
    current = []
    for ch in text:
        if ch.isalnum() and ord(ch) < 128:
            current.append(ch.lower())
        else:
            if current:
                w = "".join(current)
                freq[w] = freq.get(w, 0) + 1
                current = []
    if current:
        w = "".join(current)
        freq[w] = freq.get(w, 0) + 1
    return freq


def test_ground_truth_values_are_internally_correct():
    cases = [
        (
            "The quick brown fox jumps over the lazy dog. The dog barks!",
            {
                "the": 3, "quick": 1, "brown": 1, "fox": 1, "jumps": 1,
                "over": 1, "lazy": 1, "dog": 2, "barks": 1,
            },
        ),
        (
            "Don't stop believing, don't you dare stop.",
            {"don": 2, "t": 2, "stop": 2, "believing": 1, "you": 1, "dare": 1},
        ),
        ("Cat cat CAT cat, dog DOG.", {"cat": 4, "dog": 2}),
        ("One 1 two 2 one 1!", {"one": 2, "1": 2, "two": 1, "2": 1}),
        ("", {}),
    ]
    for text, expected in cases:
        assert _independent_word_frequency(text) == expected, text


def test_embedded_test_file_asserts_match_ground_truth():
    content = c05.get_grading_payload()["tests"][c05.CANONICAL_TEST_PATH]

    fake_module = types.ModuleType("word_freq")
    fake_module.word_frequency = _independent_word_frequency
    original = sys.modules.get("word_freq")
    sys.modules["word_freq"] = fake_module
    try:
        namespace: dict = {}
        exec(compile(content, "<embedded c05 test file>", "exec"), namespace)
        test_functions = [
            v for k, v in namespace.items() if k.startswith("test_") and callable(v)
        ]
        assert test_functions, "expected at least one test_ function in the embedded file"
        for fn in test_functions:
            fn()  # raises AssertionError if an embedded expected value is wrong
    finally:
        if original is not None:
            sys.modules["word_freq"] = original
        else:
            sys.modules.pop("word_freq", None)


def test_spec_worked_example_matches_ground_truth():
    # get_task_statement() gives the agent a worked example inline since this task is
    # hidden — that example must itself be correct, or the agent is misled.
    assert _independent_word_frequency("Cat cat, CAT!") == {"cat": 3}
    assert '{"cat": 3}' in c05.get_task_statement()


def test_keystone_ids_reference_real_test_functions():
    content = c05.get_grading_payload()["tests"][c05.CANONICAL_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c05.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c05.CANONICAL_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_empty_case():
    # An empty (or word-free) text trivially returns {} under any reasonable
    # implementation, so it shouldn't gate the score.
    assert f"{c05.CANONICAL_TEST_PATH}::test_empty_text_returns_empty_dict" not in c05.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c05.get_visibility() == "hidden"


def test_sandbox_fixture_is_empty_for_hidden_task():
    assert c05.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c05.get_grading_payload()
    assert payload["tests"][c05.CANONICAL_TEST_PATH] == c05._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "word_freq", "functions": ["word_frequency"]}
    assert payload["keystone_test_ids"] == c05.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c05.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "word_freq.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["word_freq.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c05", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c05"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c05.get_task_statement()
    # hidden task: no test file (or any other starter file) leaked into public/repo
    assert list((public / "repo").rglob("*")) == []
    assert json.loads((public / "plan.json").read_text()) == c05.get_compiled_plan()

    assert (private / c05.CANONICAL_TEST_PATH).read_text() == c05.get_grading_payload()["tests"][c05.CANONICAL_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert manifest["test_file_globs"] == [c05.CANONICAL_TEST_PATH]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c05.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
