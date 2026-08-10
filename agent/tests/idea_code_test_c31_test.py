"""
Adversarial offline checks for codebench task c31 (expr-tokenizer-negative-number-
disambiguation) — no Docker, no LLM. Mirrors idea_code_test_c01_test.py's structure: prove
the task module's own claims are internally consistent (ground truth is actually correct,
keystone ids reference real tests, the compiled plan is well-formed) BEFORE anything ever
reaches a live sandbox, plus an end-to-end exercise of materialize_task.py.

The reimplementation below is structured differently from a typical single-pass scanner: it
first splits the input into "raw lexeme spans" with a simple char-class walk that treats '-'
as always-standalone, THEN does a second pass merging a standalone MINUS into the following
NUMBER span when the disambiguation rule's conditions hold. This two-pass structure is
deliberately NOT how the task module's own docstring frames the one-pass algorithm (see its
get_compiled_plan() leaf instruction), specifically to catch a case where the docstring's
prose and the embedded test file's literal expectations disagree with each other.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c31_expr_tokenizer as c31

_OPERAND_EXPECTED_AFTER = {None, "PLUS", "MINUS", "STAR", "SLASH", "LPAREN"}
_SINGLE_CHAR = {"+": "PLUS", "*": "STAR", "/": "SLASH", "(": "LPAREN", ")": "RPAREN"}


def _independent_tokenize(s: str) -> list:
    """Two-pass reimplementation, independent of the one-pass scanner the task's own prose
    describes: pass 1 walks character classes treating every '-' as a standalone MINUS
    lexeme; pass 2 walks the resulting list and merges a MINUS into an immediately-following
    NUMBER lexeme (checking the ORIGINAL source text for absence of whitespace between them,
    not just adjacency in the token list) whenever the preceding lexeme (before the MINUS)
    was operand-expecting."""
    raw = []  # list of (type, text, start_index_in_s)
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "-":
            raw.append(("MINUS", "-", i))
            i += 1
            continue
        if ch in _SINGLE_CHAR:
            raw.append((_SINGLE_CHAR[ch], ch, i))
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            raw.append(("NUMBER", s[i:j], i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            raw.append(("IDENT", s[i:j], i))
            i = j
            continue
        raise ValueError(f"unexpected character {ch!r} at position {i}")

    merged = []
    k = 0
    while k < len(raw):
        ttype, text, start = raw[k]
        if (
            ttype == "MINUS"
            and k + 1 < len(raw)
            and raw[k + 1][0] == "NUMBER"
            and raw[k + 1][2] == start + 1  # no whitespace between '-' and the digits
            and (merged[-1][0] if merged else None) in _OPERAND_EXPECTED_AFTER
        ):
            _, num_text, _ = raw[k + 1]
            merged.append(("NUMBER", "-" + num_text))
            k += 2
        else:
            merged.append((ttype, text))
            k += 1
    return merged


_CASES = [
    ("", []),
    ("42", [("NUMBER", "42")]),
    ("foo", [("IDENT", "foo")]),
    ("3 + 4", [("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4")]),
    ("3-5", [("NUMBER", "3"), ("MINUS", "-"), ("NUMBER", "5")]),
    ("-5", [("NUMBER", "-5")]),
    ("3 - -5", [("NUMBER", "3"), ("MINUS", "-"), ("NUMBER", "-5")]),
    ("(-5 + x) * -2", [
        ("LPAREN", "("), ("NUMBER", "-5"), ("PLUS", "+"), ("IDENT", "x"),
        ("RPAREN", ")"), ("STAR", "*"), ("NUMBER", "-2"),
    ]),
    ("-x", [("MINUS", "-"), ("IDENT", "x")]),
    ("x - y", [("IDENT", "x"), ("MINUS", "-"), ("IDENT", "y")]),
    ("x -5", [("IDENT", "x"), ("MINUS", "-"), ("NUMBER", "5")]),
    ("2y", [("NUMBER", "2"), ("IDENT", "y")]),
    ("  3   +   4  ", [("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4")]),
    ("((1))", [
        ("LPAREN", "("), ("LPAREN", "("), ("NUMBER", "1"), ("RPAREN", ")"), ("RPAREN", ")"),
    ]),
    ("1 + 2 - 3 * 4 / 5", [
        ("NUMBER", "1"), ("PLUS", "+"), ("NUMBER", "2"), ("MINUS", "-"),
        ("NUMBER", "3"), ("STAR", "*"), ("NUMBER", "4"), ("SLASH", "/"), ("NUMBER", "5"),
    ]),
    ("x1 + 2y", [
        ("IDENT", "x1"), ("PLUS", "+"), ("NUMBER", "2"), ("IDENT", "y"),
    ]),
]


def test_ground_truth_values_are_internally_correct():
    for s, expected in _CASES:
        assert _independent_tokenize(s) == expected, s


def test_unrecognized_character_raises():
    import pytest
    for bad in ("3 @ 4", "3 & 4", "#", "3$4"):
        with pytest.raises(ValueError):
            _independent_tokenize(bad)


def test_embedded_test_file_literals_match_ground_truth():
    content = c31.get_sandbox_fixture()[c31.VISIBLE_TEST_PATH]
    # Every literal tokenize("...") == [...] assertion in the embedded file must match the
    # independent reimplementation. Parse conservatively: find each `tokenize("...")` call
    # followed by `== [` up to the matching closing bracket on a balanced-bracket basis.
    calls = list(re.finditer(r'tokenize\((".*?"|\'.*?\')\)\s*==\s*', content))
    assert len(calls) >= 15, f"expected at least 15 literal tokenize() assertions, found {len(calls)}"
    for m in calls:
        literal_src = m.group(1)
        s = eval(literal_src)  # noqa: S307 - trusted fixture content, simple string literal
        # Extract the following bracketed expression by bracket-balance counting.
        start = m.end()
        assert content[start] == "["
        depth = 0
        end = start
        for idx in range(start, len(content)):
            if content[idx] == "[":
                depth += 1
            elif content[idx] == "]":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        expr_src = content[start:end]
        expected = eval(expr_src)  # noqa: S307 - trusted fixture content, literal tuple/list
        assert _independent_tokenize(s) == [tuple(t) for t in expected], (s, expected)


def test_keystone_ids_reference_real_test_functions():
    content = c31.get_sandbox_fixture()[c31.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c31.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c31.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_and_bonus_cases():
    non_keystone = [
        "test_empty_string",
        "test_single_number",
        "test_single_identifier",
        "test_simple_addition",
        "test_identifier_minus_identifier",
        "test_whitespace_is_skipped",
        "test_nested_parens",
        "test_identifier_with_trailing_digit_and_number_with_trailing_letter",
        "test_unrecognized_character_raises",
    ]
    for name in non_keystone:
        assert f"{c31.VISIBLE_TEST_PATH}::{name}" not in c31.KEYSTONE_TEST_IDS, name
    content = c31.get_sandbox_fixture()[c31.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 17
    assert len(c31.KEYSTONE_TEST_IDS) == 8


def test_visibility_is_visible():
    assert c31.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c31.get_grading_payload()
    assert payload["tests"][c31.VISIBLE_TEST_PATH] == c31.get_sandbox_fixture()[c31.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "expr_tokenizer", "functions": ["tokenize"]}
    assert payload["keystone_test_ids"] == c31.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c31.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "expr_tokenizer.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["expr_tokenizer.py"]}
    json.dumps(plan)


def test_compiled_plan_does_not_leak_canonical_test_content():
    plan_text = json.dumps(c31.get_compiled_plan())
    assert "import pytest" not in plan_text
    assert "def test_" not in plan_text


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c31", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c31"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c31.get_task_statement()
    assert (public / "repo" / c31.VISIBLE_TEST_PATH).read_text() == c31.get_sandbox_fixture()[c31.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c31.get_compiled_plan()

    assert (private / c31.VISIBLE_TEST_PATH).read_text() == c31.get_grading_payload()["tests"][c31.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c31.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c31.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
