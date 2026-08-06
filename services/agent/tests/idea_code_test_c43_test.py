"""
Adversarial offline checks for codebench task c43 (tokenizer-parser-evaluator-chain) — no
Docker, no LLM.

Mirrors idea_code_test_c06_test.py / idea_code_test_c42_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference real
tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox. Second
THREE-leaf codebench task, so this file checks the leaf_a/leaf_b/leaf_c dependency wiring the
same way: exactly three leaves in a strict chain, leaf_c depending on leaf_b only (NOT leaf_a
directly -- the evaluator never calls tokenize()), and each dependent leaf's instruction
referencing its immediate upstream leaf via a ``{leaf_x}`` placeholder.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c43_tokenizer_parser_evaluator_chain as c43


# --- independent reimplementation (deliberately NOT importing the task module's own logic) -----

def _tokenize(expr: str):
    tokens = []
    i, n = 0, len(expr)
    single = {"+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH", "(": "LPAREN", ")": "RPAREN"}
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            tokens.append(("NUMBER", expr[i:j]))
            i = j
            continue
        if ch in single:
            tokens.append((single[ch], ch))
            i += 1
            continue
        raise ValueError(f"bad char {ch!r}")
    tokens.append(("EOF", ""))
    return tokens


class _P:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, t):
        tok = self.advance()
        if tok[0] != t:
            raise ValueError(f"expected {t}, got {tok[0]}")
        return tok

    def expr(self):
        node = self.term()
        while self.peek()[0] in ("PLUS", "MINUS"):
            op_tok = self.advance()
            op = "+" if op_tok[0] == "PLUS" else "-"
            node = ("binop", op, node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek()[0] in ("STAR", "SLASH"):
            op_tok = self.advance()
            op = "*" if op_tok[0] == "STAR" else "/"
            node = ("binop", op, node, self.factor())
        return node

    def factor(self):
        tok = self.peek()
        if tok[0] == "NUMBER":
            self.advance()
            return ("num", int(tok[1]))
        if tok[0] == "LPAREN":
            self.advance()
            node = self.expr()
            self.expect("RPAREN")
            return node
        raise ValueError(f"unexpected {tok[0]}")


def _parse(expr: str):
    p = _P(_tokenize(expr))
    node = p.expr()
    p.expect("EOF")
    return node


def _evaluate(expr: str):
    def walk(node):
        if node[0] == "num":
            return node[1]
        _, op, left, right = node
        lv, rv = walk(left), walk(right)
        return {"+": lv + rv, "-": lv - rv, "*": lv * rv, "/": lv / rv}[op]
    return walk(_parse(expr))


def test_ground_truth_tokenizer_is_internally_correct():
    assert _tokenize("3 + 4 * 2") == [
        ("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4"), ("STAR", "*"), ("NUMBER", "2"),
        ("EOF", ""),
    ]
    assert _tokenize("(3+4)*2") == [
        ("LPAREN", "("), ("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4"), ("RPAREN", ")"),
        ("STAR", "*"), ("NUMBER", "2"), ("EOF", ""),
    ]
    assert _tokenize("") == [("EOF", "")]
    import pytest
    with pytest.raises(ValueError):
        _tokenize("3 + @")


def test_ground_truth_parser_encodes_correct_precedence():
    assert _parse("3 + 4 * 2") == ("binop", "+", ("num", 3), ("binop", "*", ("num", 4), ("num", 2)))
    assert _parse("(3 + 4) * 2") == ("binop", "*", ("binop", "+", ("num", 3), ("num", 4)), ("num", 2))
    import pytest
    with pytest.raises(ValueError):
        _parse("(3 + 4")
    with pytest.raises(ValueError):
        _parse("3 +")


def test_ground_truth_evaluator_computes_correct_results():
    assert _evaluate("3 + 4 * 2") == 11
    assert _evaluate("(3 + 4) * 2") == 14
    assert _evaluate("10 / 4") == 2.5
    assert _evaluate("2 * (3 + (4 - 1))") == 12
    assert _evaluate("100 - 2 * 3 - 4") == 90
    import pytest
    with pytest.raises(ZeroDivisionError):
        _evaluate("1 / 0")
    with pytest.raises(ValueError):
        _evaluate("3 + @")


def test_embedded_test_file_asserts_match_ground_truth():
    content = c43.get_grading_payload()["tests"][c43._TEST_FILE_PATH]
    assert 'evaluate("3 + 4 * 2") == 11' in content
    assert 'evaluate("10 / 4") == 2.5' in content
    assert 'evaluate("100 - 2 * 3 - 4") == 90' in content
    assert ('("binop", "+", ("num", 3), ("binop", "*", ("num", 4), ("num", 2)))') in content


def test_keystone_ids_reference_real_test_functions():
    content = c43.get_grading_payload()["tests"][c43._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c43.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c43._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_basic_single_path_checks():
    for name in ("test_tokenize_basic_expression", "test_tokenize_rejects_unknown_character",
                 "test_parser_rejects_unbalanced_parens", "test_evaluator_division_by_zero_raises"):
        assert f"{c43._TEST_FILE_PATH}::{name}" not in c43.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c43.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c43.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c43.get_grading_payload()
    assert payload["tests"] == {c43._TEST_FILE_PATH: c43._TEST_FILE_CONTENT}
    assert payload["keystone_test_ids"] == c43.KEYSTONE_TEST_IDS
    modules = payload["entrypoint"]["modules"]
    assert {m["module"] for m in modules} == {"tokenizer", "parser", "evaluator"}
    assert next(m for m in modules if m["module"] == "tokenizer")["functions"] == ["tokenize"]
    assert next(m for m in modules if m["module"] == "parser")["functions"] == ["parse"]
    assert next(m for m in modules if m["module"] == "evaluator")["functions"] == ["evaluate"]


def test_compiled_plan_has_exactly_three_leaves_in_a_strict_chain():
    plan = c43.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["leaf_a", "leaf_b", "leaf_c"]
    leaf_a, leaf_b, leaf_c = plan["leaves"]
    assert leaf_a["depends_on"] == []
    assert leaf_b["depends_on"] == ["leaf_a"]
    # leaf_c depends on leaf_b ONLY -- the evaluator never calls tokenize() itself.
    assert leaf_c["depends_on"] == ["leaf_b"]


def test_leaf_b_instruction_references_leaf_a_via_placeholder_and_restates_the_api():
    plan = c43.get_compiled_plan()
    leaf_b = plan["leaves"][1]
    assert "{leaf_a}" in leaf_b["instruction"]
    for name in ("NUMBER", "PLUS", "LPAREN", "EOF", "tokenize"):
        assert name in leaf_b["instruction"], name
    assert "parser.py" in leaf_b["instruction"]


def test_leaf_c_instruction_references_leaf_b_via_placeholder_and_restates_the_api():
    plan = c43.get_compiled_plan()
    leaf_c = plan["leaves"][2]
    assert "{leaf_b}" in leaf_c["instruction"]
    assert "{leaf_a}" not in leaf_c["instruction"]
    assert "binop" in leaf_c["instruction"] and "num" in leaf_c["instruction"]
    assert "evaluator.py" in leaf_c["instruction"]


def test_leaf_a_instruction_names_the_exact_token_shape_downstream_leaves_will_rely_on():
    plan = c43.get_compiled_plan()
    leaf_a = plan["leaves"][0]
    for name in ("NUMBER", "PLUS", "MINUS", "STAR", "SLASH", "LPAREN", "RPAREN", "EOF"):
        assert name in leaf_a["instruction"], name
    assert "tokenizer.py" in leaf_a["instruction"]


def test_no_leaf_instruction_leaks_the_private_test_fixture_values():
    plan = c43.get_compiled_plan()
    all_instructions = " ".join(leaf["instruction"] for leaf in plan["leaves"])
    for leaked in ("100 - 2 * 3 - 4", "== 90", "== 11", "== 2.5"):
        assert leaked not in all_instructions, leaked


def test_compiled_plan_structure():
    plan = c43.get_compiled_plan()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {
        "op": "submit_files", "files": ["tokenizer.py", "parser.py", "evaluator.py"],
    }
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c43", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c43"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c43.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c43.get_compiled_plan()

    assert (private / c43._TEST_FILE_PATH).read_text() == c43._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c43._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c43.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
