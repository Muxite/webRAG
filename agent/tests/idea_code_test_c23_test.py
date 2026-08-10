"""
Adversarial offline checks for codebench task c23 (request-query-header-parser) — no Docker,
no LLM.

Mirrors idea_code_test_c01_test.py's pattern (visible task, literal input/output pairs) but with
a SECOND, differently-coded re-derivation of ground truth: c23's own reference solution is
expected to compose urllib.parse.unquote/unquote_plus; this validator instead hand-rolls its own
percent-decoder from scratch (no urllib at all) as an independent cross-check that the embedded
test file's literal expected values are actually correct, not just self-consistent with one
particular (possibly also-wrong) implementation approach.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c23_request_query_header_parser as c23


def _independent_percent_decode(s: str, plus_is_space: bool) -> str:
    """Hand-rolled %XX decoder, deliberately NOT using urllib.parse at all -- an independent
    cross-check of whatever urllib-based approach c23's own reference solution takes."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "%" and i + 2 < n and all(c in "0123456789abcdefABCDEF" for c in s[i + 1:i + 3]):
            out.append(chr(int(s[i + 1:i + 3], 16)))
            i += 3
        elif ch == "+" and plus_is_space:
            out.append(" ")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _independent_parse_request(method, raw_path, header_lines):
    method_u = method.upper()
    if "?" in raw_path:
        path_part, query_part = raw_path.split("?", 1)
    else:
        path_part, query_part = raw_path, ""
    path = _independent_percent_decode(path_part, plus_is_space=False)

    query = {}
    if query_part:
        for pair in query_part.split("&"):
            if not pair:
                continue
            if "=" in pair:
                k, v = pair.split("=", 1)
            else:
                k, v = pair, ""
            k = _independent_percent_decode(k, plus_is_space=True)
            v = _independent_percent_decode(v, plus_is_space=True)
            query.setdefault(k, []).append(v)

    headers = {}
    for line in header_lines:
        if ":" not in line:
            raise ValueError("malformed header line")
        name, value = line.split(":", 1)
        name = name.strip().lower()
        value = value.strip()
        headers[name] = headers[name] + ", " + value if name in headers else value

    return {"method": method_u, "path": path, "query": query, "headers": headers}


def test_ground_truth_no_query_string():
    assert _independent_parse_request("get", "/hello", []) == {
        "method": "GET", "path": "/hello", "query": {}, "headers": {},
    }


def test_ground_truth_repeated_query_key_collects_list():
    result = _independent_parse_request(
        "GET", "/search?q=hello+world&tag=a&tag=b&page=2", []
    )
    assert result["path"] == "/search"
    assert result["query"] == {"q": ["hello world"], "tag": ["a", "b"], "page": ["2"]}


def test_ground_truth_path_percent_and_plus_handling():
    a = _independent_parse_request("GET", "/files/my%20doc.txt?name=foo+bar", [])
    assert a["path"] == "/files/my doc.txt"
    assert a["query"] == {"name": ["foo bar"]}

    b = _independent_parse_request("GET", "/note+book?title=foo", [])
    assert b["path"] == "/note+book"
    assert b["query"] == {"title": ["foo"]}


def test_ground_truth_bare_query_key():
    result = _independent_parse_request("GET", "/toggle?flag&on=1", [])
    assert result["query"] == {"flag": [""], "on": ["1"]}


def test_ground_truth_headers_case_insensitive_and_duplicate_joined():
    result = _independent_parse_request(
        "GET", "/x",
        ["Content-Type: application/json", "Accept: text/html", "Accept: application/xml",
         "X-Request-Id:   abc123  "],
    )
    assert result["headers"] == {
        "content-type": "application/json",
        "accept": "text/html, application/xml",
        "x-request-id": "abc123",
    }


def test_ground_truth_malformed_header_raises():
    with pytest.raises(ValueError):
        _independent_parse_request("GET", "/x", ["BadHeaderNoColon"])


def test_ground_truth_trailing_question_mark():
    result = _independent_parse_request("GET", "/plain?", [])
    assert result["path"] == "/plain"
    assert result["query"] == {}


def test_embedded_test_file_asserts_match_the_independent_reimplementation():
    content = c23.get_sandbox_fixture()[c23.VISIBLE_TEST_PATH]
    assert '"/search?q=hello+world&tag=a&tag=b&page=2"' in content
    assert '{"q": ["hello world"], "tag": ["a", "b"], "page": ["2"]}' in content
    assert '"/files/my%20doc.txt?name=foo+bar"' in content
    assert 'result_a["path"] == "/files/my doc.txt"' in content
    assert '"/note+book?title=foo"' in content
    assert 'result_b["path"] == "/note+book"' in content
    assert '{"flag": [""], "on": ["1"]}' in content
    assert '"text/html, application/xml"' in content
    assert "pytest.raises(ValueError)" in content
    assert '"BadHeaderNoColon"' in content


def test_keystone_ids_reference_real_test_functions():
    content = c23.get_sandbox_fixture()[c23.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 9
    for node_id in c23.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c23.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_non_discriminating_cases():
    non_keystone = [
        "test_no_query_string_returns_empty_dict",
        "test_bare_query_key_without_equals",
        "test_method_is_uppercased",
        "test_trailing_question_mark_empty_query",
    ]
    for name in non_keystone:
        assert f"{c23.VISIBLE_TEST_PATH}::{name}" not in c23.KEYSTONE_TEST_IDS
    assert len(c23.KEYSTONE_TEST_IDS) == 5


def test_visibility_is_visible():
    assert c23.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c23.get_grading_payload()
    assert payload["tests"][c23.VISIBLE_TEST_PATH] == c23.get_sandbox_fixture()[c23.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "request_parse", "functions": ["parse_request"]}
    assert payload["keystone_test_ids"] == c23.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c23.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "request_parse.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["request_parse.py"]}
    json.dumps(plan)  # must be plain JSON-safe data


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c23", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c23"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c23.get_task_statement()
    assert (public / "repo" / c23.VISIBLE_TEST_PATH).read_text() == c23.get_sandbox_fixture()[c23.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c23.get_compiled_plan()

    assert (private / c23.VISIBLE_TEST_PATH).read_text() == c23.get_grading_payload()["tests"][c23.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c23.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c23.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
