"""
Adversarial offline checks for codebench task c32 (email-fsm-validator) — no Docker, no LLM.
Mirrors idea_code_test_c03_test.py's structure exactly (c32 is hidden, same shape: a single
bool-returning validator function). The reimplementation below is a REGEX-based validator
(deliberately the opposite implementation style the task itself asks the agent to avoid),
structured completely differently from any state-machine walk, to catch a mistake in the
task module's embedded expected values via a genuinely independent method.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c32_email_fsm_validator as c32

_LOCAL_RE = re.compile(r"^[A-Za-z0-9._%+-]+$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9-]+$")
_LETTERS_RE = re.compile(r"^[A-Za-z]+$")


def _independent_is_valid_email(s: str) -> bool:
    """Deliberately regex-based (the style the task asks the agent NOT to use), to give a
    genuinely independent second opinion on every literal expected value below."""
    if s.count("@") != 1:
        return False
    local, domain = s.split("@")
    if not local or not _LOCAL_RE.match(local):
        return False
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    if not domain:
        return False
    labels = domain.split(".")
    if len(labels) < 2:
        return False
    for label in labels:
        if not label or not _LABEL_RE.match(label):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
    tld = labels[-1]
    if len(tld) < 2 or not _LETTERS_RE.match(tld):
        return False
    return True


_CASES = [
    ("a@b.co", True),
    ("john.doe@example.com", True),
    ("user_name+tag@sub-domain.example.org", True),
    ("a.b.c@example.co", True),
    ("x99@ab12.io", True),
    ("", False),
    ("noatsign.example.com", False),
    ("two@at@signs.com", False),
    ("@example.com", False),
    ("user@", False),
    (".user@example.com", False),
    ("user.@example.com", False),
    ("us..er@example.com", False),
    ("user@.example.com", False),
    ("user@example.com.", False),
    ("user@example..com", False),
    ("user@-example.com", False),
    ("user@example-.com", False),
    ("user@examplecom", False),
    ("user@example.c", False),
    ("user@example.c0m", False),
    ("user@example.co-m", False),
    ("us er@example.com", False),
    ("user@exa mple.com", False),
    ("user#name@example.com", False),
]


def test_ground_truth_values_are_internally_correct():
    for s, expected in _CASES:
        assert _independent_is_valid_email(s) is expected, s


def test_embedded_test_file_asserts_match_ground_truth():
    content = c32.get_grading_payload()["tests"][c32.CANONICAL_TEST_PATH]
    pairs = re.findall(r'is_valid_email\("((?:[^"\\]|\\.)*)"\) is (True|False)', content)
    assert len(pairs) == len(_CASES), (len(pairs), len(_CASES))
    for s, expected_str in pairs:
        expected = expected_str == "True"
        assert _independent_is_valid_email(s) is expected, (s, expected)
    # And every case in this file's own table is actually present in the embedded fixture.
    embedded_strings = {s for s, _ in pairs}
    for s, _ in _CASES:
        assert s in embedded_strings, s


def test_spec_worked_examples_match_ground_truth():
    statement = c32.get_task_statement()
    assert _independent_is_valid_email("a@b.co") is True
    assert '`is_valid_email("a@b.co")` -> `True`' in statement
    assert _independent_is_valid_email("user@example..com") is False
    assert '`is_valid_email("user@example..com")` -> \n`False`' in statement or (
        'is_valid_email("user@example..com")' in statement and "False" in statement
    )
    assert _independent_is_valid_email("user@example.c0m") is False
    assert 'is_valid_email("user@example.c0m")' in statement


def test_each_keystone_targets_a_distinct_rule_and_is_false():
    # Every keystone case must independently verify as False (they're all violation probes,
    # not valid-address examples) -- a keystone that's secretly True would be a authoring bug.
    content = c32.get_grading_payload()["tests"][c32.CANONICAL_TEST_PATH]
    false_only_keystones = {
        "test_local_starts_with_dot", "test_local_ends_with_dot",
        "test_local_has_consecutive_dots", "test_domain_leading_dot",
        "test_domain_trailing_dot", "test_domain_consecutive_dots",
        "test_domain_label_starts_with_hyphen", "test_domain_label_ends_with_hyphen",
        "test_domain_has_no_dot", "test_tld_too_short", "test_tld_contains_digit",
        "test_tld_contains_hyphen",
    }
    for node_id in c32.KEYSTONE_TEST_IDS:
        _, _, func = node_id.partition("::")
        if func in false_only_keystones:
            pattern = re.compile(rf"def {func}\(\):\s*\n\s*assert is_valid_email\(\"([^\"]*)\"\) is (True|False)")
            m = pattern.search(content)
            assert m, func
            assert m.group(2) == "False", func


def test_keystone_ids_reference_real_test_functions():
    content = c32.get_grading_payload()["tests"][c32.CANONICAL_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c32.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c32.CANONICAL_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_less_discriminating_cases():
    non_keystone = [
        "test_minimal_valid_address",
        "test_typical_valid_address",
        "test_empty_string",
        "test_missing_at_sign",
        "test_two_at_signs",
        "test_empty_local_part",
        "test_empty_domain",
        "test_space_in_local_part",
        "test_space_in_domain",
        "test_disallowed_local_character",
    ]
    for name in non_keystone:
        assert f"{c32.CANONICAL_TEST_PATH}::{name}" not in c32.KEYSTONE_TEST_IDS, name
    content = c32.get_grading_payload()["tests"][c32.CANONICAL_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 25
    assert len(c32.KEYSTONE_TEST_IDS) == 15
    assert len(non_keystone) + len(c32.KEYSTONE_TEST_IDS) == len(defined)


def test_visibility_is_hidden():
    assert c32.get_visibility() == "hidden"


def test_sandbox_fixture_is_empty_for_hidden_task():
    assert c32.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c32.get_grading_payload()
    assert payload["tests"][c32.CANONICAL_TEST_PATH] == c32._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "email_validator", "functions": ["is_valid_email"]}
    assert payload["keystone_test_ids"] == c32.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c32.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "email_validator.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["email_validator.py"]}
    json.dumps(plan)


def test_compiled_plan_does_not_leak_canonical_test_content():
    plan_text = json.dumps(c32.get_compiled_plan())
    assert "def test_" not in plan_text


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c32", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c32"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c32.get_task_statement()
    assert list((public / "repo").rglob("*")) == []
    assert json.loads((public / "plan.json").read_text()) == c32.get_compiled_plan()

    assert (private / c32.CANONICAL_TEST_PATH).read_text() == c32.get_grading_payload()["tests"][c32.CANONICAL_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert manifest["test_file_globs"] == [c32.CANONICAL_TEST_PATH]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c32.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
