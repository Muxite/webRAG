"""
Adversarial offline checks for codebench task c10 (markdown-table-formatter) — no Docker, no
LLM.

Mirrors the spirit of execution_compiled_*_validators_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
Also exercises codebench/materialize_task.py end-to-end against this task. c10 is
hidden, so (unlike c01/c09) the canonical test content lives ONLY in get_grading_payload(),
never in get_sandbox_fixture().
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c10_md_table as c10


def _independent_format_table(headers, rows):
    """Reimplemented with str.format width specifiers instead of str.ljust, and a
    transposed-column loop, to reduce the odds of a copy-correlated bug against the task's
    own reference implementation."""
    widths = []
    for col in range(len(headers)):
        col_cells = [headers[col]] + [row[col] for row in rows]
        widths.append(max(len(c) for c in col_cells))

    def render(cells):
        parts = ["{:<{w}}".format(c, w=w) for c, w in zip(cells, widths)]
        return "|" + "|".join(f" {p} " for p in parts) + "|"

    # separator is literal "---" per column, not width-padded — rendered directly, never via
    # render(), which assumes real content cells.
    separator = "|" + "|".join(" --- " for _ in widths) + "|"
    lines = [render(headers), separator]
    for row in rows:
        lines.append(render(row))
    return "\n".join(lines)


_WORKED = (["Name", "Age"], [["Alice", "30"], ["Bob", "5"]])
_UNEVEN = (
    ["Item", "Qty", "Unit Price"],
    [["Widget", "12", "3.50"], ["Extra Long Item Name", "1", "100"]],
)
_SINGLE_COL = (["Only"], [["x"], ["yy"]])

_EXPECTED_WORKED = "| Name  | Age |\n| --- | --- |\n| Alice | 30  |\n| Bob   | 5   |"
_EXPECTED_UNEVEN = (
    "| Item                 | Qty | Unit Price |\n"
    "| --- | --- | --- |\n"
    "| Widget               | 12  | 3.50       |\n"
    "| Extra Long Item Name | 1   | 100        |"
)
_EXPECTED_SINGLE_COL = "| Only |\n| --- |\n| x    |\n| yy   |"


def test_ground_truth_values_are_internally_correct():
    assert _independent_format_table(*_WORKED) == _EXPECTED_WORKED
    assert _independent_format_table(*_UNEVEN) == _EXPECTED_UNEVEN
    assert _independent_format_table(*_SINGLE_COL) == _EXPECTED_SINGLE_COL
    for out in (_EXPECTED_WORKED, _EXPECTED_UNEVEN, _EXPECTED_SINGLE_COL):
        for line in out.split("\n"):
            assert line == line.rstrip(), f"embedded expected value has trailing ws: {line!r}"


def test_task_statement_worked_example_matches_ground_truth():
    # The spec embeds the worked example between BEGIN/END markers character-for-character —
    # extract it and confirm it's byte-identical to the independently-verified value.
    statement = c10.get_task_statement()
    m = re.search(r"-----BEGIN-----\n(.*?)\n-----END-----", statement, re.DOTALL)
    assert m, "expected a BEGIN/END-delimited worked example in the task statement"
    assert m.group(1) == _EXPECTED_WORKED


def test_embedded_test_file_asserts_match_ground_truth():
    content = c10.get_grading_payload()["tests"][c10.VISIBLE_TEST_PATH]
    namespace = {"format_table": _independent_format_table}
    code = content.replace("from md_table import format_table\n", "")
    exec(compile(code, "<c10 embedded test>", "exec"), namespace)
    test_fns = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    assert len(test_fns) == 4, "expected all 4 embedded test_ functions to be present"
    for fn in test_fns:
        fn()


def test_keystone_ids_reference_real_test_functions():
    content = c10.get_grading_payload()["tests"][c10.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c10.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c10.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_gates_every_case():
    # Unlike c01/c09, this task IS the formatting exactness — no case is degenerate enough to
    # exclude, so all 4 embedded tests should be keystone.
    content = c10.get_grading_payload()["tests"][c10.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(c10.KEYSTONE_TEST_IDS) == len(defined)


def test_visibility_is_hidden():
    assert c10.get_visibility() == "hidden"


def test_sandbox_fixture_has_no_test_file_for_hidden_task():
    assert c10.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c10.get_grading_payload()
    assert payload["entrypoint"] == {"module": "md_table", "functions": ["format_table"]}
    assert payload["keystone_test_ids"] == c10.KEYSTONE_TEST_IDS
    assert c10.VISIBLE_TEST_PATH in payload["tests"]


def test_compiled_plan_structure():
    plan = c10.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "md_table.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["md_table.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c10", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c10"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c10.get_task_statement()
    # hidden task: no starter test file exposed in the public repo tree
    assert not (public / "repo" / c10.VISIBLE_TEST_PATH).exists()
    assert json.loads((public / "plan.json").read_text()) == c10.get_compiled_plan()

    assert (private / c10.VISIBLE_TEST_PATH).read_text() == c10.get_grading_payload()["tests"][c10.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c10.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c10.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
