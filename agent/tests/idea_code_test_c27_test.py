"""
Adversarial offline checks for codebench task c27 (ini-config-round-trip) — no Docker, no
LLM. Mirrors idea_code_test_c04_test.py's exec-based structure: the embedded canonical test
file's assertions compare structurally-parsed dicts, not simple string literals, so ground
truth is verified by actually EXECUTING every test_* function in the embedded file against
an INDEPENDENTLY-WRITTEN reference implementation (a different parsing strategy — regex
line-matching + explicit current-section state, rather than the task module's own
find()-based separator search) rather than regex-scraping literals out of the source text.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

from agent.app.idea_code_tests import test_c27_ini_config_round_trip as c27

_SECTION_RE = re.compile(r"^\[(.*)\]$")


def _independent_parse_ini(text: str) -> dict:
    """Independently-written (state-machine style, regex section match, str.partition for
    the separator search) rather than the task module's find()-based approach."""
    result: dict = {}
    order: list = []
    current = ""

    def touch(name):
        if name not in result:
            result[name] = {}
            order.append(name)

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in ("#", ";"):
            continue
        m = _SECTION_RE.match(stripped)
        if m:
            current = m.group(1).strip()
            touch(current)
            continue
        # whichever of '=' / ':' appears first
        eq_idx = stripped.find("=")
        colon_idx = stripped.find(":")
        candidates = [x for x in (eq_idx, colon_idx) if x != -1]
        if not candidates:
            continue
        sep_idx = min(candidates)
        key = stripped[:sep_idx].strip()
        value = stripped[sep_idx + 1:].strip()
        touch(current)
        result[current][key] = value

    if "" in result and not result[""]:
        del result[""]
    return result


def _independent_write_ini(config: dict) -> str:
    """Independently-written serializer, same required format but built via a fresh
    list-of-strings accumulator rather than the task module's block-join approach."""
    pieces: list = []
    if config.get("", None):
        block_lines = [f"{k} = {v}" for k, v in config[""].items()]
        pieces.append("\n".join(block_lines) + "\n")
    for section, kv in config.items():
        if section == "":
            continue
        block_lines = [f"[{section}]"]
        for k, v in kv.items():
            block_lines.append(f"{k} = {v}")
        pieces.append("\n".join(block_lines) + "\n")
    return "\n".join(pieces)


def _run_embedded_tests_against(parse_fn, write_fn):
    content = c27.get_sandbox_fixture()[c27.VISIBLE_TEST_PATH]
    fake_module = types.ModuleType("ini_mod")
    fake_module.parse_ini = parse_fn
    fake_module.write_ini = write_fn
    original = sys.modules.get("ini_mod")
    sys.modules["ini_mod"] = fake_module
    try:
        namespace: dict = {}
        exec(compile(content, "<embedded c27 test file>", "exec"), namespace)
        test_functions = [
            v for k, v in namespace.items() if k.startswith("test_") and callable(v)
        ]
        assert test_functions, "expected at least one test_ function in the embedded file"
        for fn in test_functions:
            fn()  # raises AssertionError if an embedded expected value is wrong
    finally:
        if original is not None:
            sys.modules["ini_mod"] = original
        else:
            sys.modules.pop("ini_mod", None)


def test_ground_truth_values_are_internally_correct():
    assert _independent_parse_ini("[a]\nx = 1\nx = 2\n") == {"a": {"x": "2"}}
    text2 = (
        "; leading comment\n\n[core]\n# comment inside section\n\nname = widget\n\n"
        "; trailing comment\n"
    )
    assert _independent_parse_ini(text2) == {"core": {"name": "widget"}}
    text3 = "debug = true\nverbose = false\n\n[core]\nname = widget\n"
    assert _independent_parse_ini(text3) == {
        "": {"debug": "true", "verbose": "false"},
        "core": {"name": "widget"},
    }
    text4 = "[core]\nname: widget\ncount = 5\n"
    assert _independent_parse_ini(text4) == {"core": {"name": "widget", "count": "5"}}


def test_ground_truth_write_ini_exact_format():
    cfg = {
        "": {"debug": "true"},
        "server": {"host": "localhost", "port": "8080"},
        "empty_section": {},
    }
    expected = "debug = true\n\n[server]\nhost = localhost\nport = 8080\n\n[empty_section]\n"
    assert _independent_write_ini(cfg) == expected


def test_ground_truth_round_trip():
    cfg = {"": {"a": "1"}, "sec1": {"x": "10", "y": "20"}, "sec2": {"z": "hello world"}}
    assert _independent_parse_ini(_independent_write_ini(cfg)) == cfg


def test_embedded_test_file_asserts_match_ground_truth():
    _run_embedded_tests_against(_independent_parse_ini, _independent_write_ini)


def test_keystone_ids_reference_real_test_functions():
    content = c27.get_sandbox_fixture()[c27.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c27.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c27.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_shallow_checks():
    # A trivial single-section parse and the alternate ':' separator are real behavior but
    # don't distinguish a correct implementation from a shallow one the way duplicate-key
    # policy, comment handling, the headerless top-level section, exact write_ini
    # formatting, and the round trip do — bonus credit only, not gating.
    assert f"{c27.VISIBLE_TEST_PATH}::test_basic_single_section" not in c27.KEYSTONE_TEST_IDS
    assert f"{c27.VISIBLE_TEST_PATH}::test_colon_separator_supported" not in c27.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c27.get_visibility() == "visible"


def test_sandbox_fixture_and_grading_tests_share_relpath():
    # Security constraint: get_sandbox_fixture() and get_grading_payload()["tests"] must
    # share relpaths for any file present in both.
    fixture = c27.get_sandbox_fixture()
    tests = c27.get_grading_payload()["tests"]
    assert set(fixture.keys()) == set(tests.keys())
    for path in fixture:
        assert fixture[path] == tests[path]


def test_no_canonical_values_leak_into_compiled_plan():
    plan = c27.get_compiled_plan()
    plan_text = json.dumps(plan)
    # None of the canonical INI text's specific literal values (from the embedded test
    # file) should be quoted verbatim inside the plan's leaf instructions — the plan
    # describes the FORMAT rules, not the test's own example data.
    for leaked in ("localhost", "8080", "empty_section", "hello world"):
        assert leaked not in plan_text, f"leaked canonical value into plan.json: {leaked!r}"


def test_grading_payload_shape():
    payload = c27.get_grading_payload()
    assert payload["tests"][c27.VISIBLE_TEST_PATH] == c27.get_sandbox_fixture()[c27.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {
        "module": "ini_mod",
        "functions": ["parse_ini", "write_ini"],
    }
    assert payload["keystone_test_ids"] == c27.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c27.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "ini_mod.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["ini_mod.py"]}
    json.dumps(plan)  # must be JSON-serializable as-is


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c27", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c27"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c27.get_task_statement()
    assert (public / "repo" / c27.VISIBLE_TEST_PATH).read_text() == c27.get_sandbox_fixture()[c27.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c27.get_compiled_plan()

    assert (private / c27.VISIBLE_TEST_PATH).read_text() == c27.get_grading_payload()["tests"][c27.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c27.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c27.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
