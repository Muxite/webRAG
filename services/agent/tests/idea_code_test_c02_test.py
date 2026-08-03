"""
Adversarial offline checks for codebench task c02 (rle-codec-round-trip) — no Docker, no
LLM. Mirrors idea_code_test_c01_test.py's structure: prove the task module's own claims
are internally consistent (ground truth is actually correct, keystone ids reference real
tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox, and
exercise materialize_task.py end-to-end against this task.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c02_rle_codec as c02


def _independent_encode(s: str) -> str:
    """Reimplemented independently of task's own prose spec (itertools.groupby-based),
    to catch a hand-arithmetic error in the task module's embedded expected values."""
    out = []
    for ch, grp in itertools.groupby(s):
        out.append(f"{sum(1 for _ in grp)}{ch}")
    return "".join(out)


def _independent_decode(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        j = i
        while j < n and s[j].isdigit():
            j += 1
        count = int(s[i:j])
        ch = s[j]
        out.append(ch * count)
        i = j + 1
    return "".join(out)


def test_ground_truth_values_are_internally_correct():
    encode_cases = [
        ("aaabbc", "3a2b1c"),
        ("c", "1c"),
        ("aaaa", "4a"),
        ("xyz", "1x1y1z"),
    ]
    for s, expected in encode_cases:
        assert _independent_encode(s) == expected, s

    decode_cases = [
        ("3a2b1c", "aaabbc"),
        ("1c", "c"),
        ("1a1b1c1a1b1c", "abcabc"),
    ]
    for s, expected in decode_cases:
        assert _independent_decode(s) == expected, s

    round_trip_cases = ["", "a", "aaaa", "xyz", "aabbaabbcc"]
    for s in round_trip_cases:
        assert _independent_decode(_independent_encode(s)) == s, s


def test_embedded_test_file_asserts_match_ground_truth():
    content = c02.get_sandbox_fixture()[c02.VISIBLE_TEST_PATH]

    encode_pairs = re.findall(r'encode\("([^"]*)"\) == "([^"]*)"', content)
    assert encode_pairs, "expected at least one literal encode() assertion"
    for s, expected in encode_pairs:
        assert _independent_encode(s) == expected, (s, expected)

    decode_pairs = re.findall(r'decode\("([^"]*)"\) == "([^"]*)"', content)
    assert decode_pairs, "expected at least one literal decode() assertion"
    for s, expected in decode_pairs:
        assert _independent_decode(s) == expected, (s, expected)

    round_trip_matches = re.findall(r'decode\(encode\("([^"]*)"\)\) == "([^"]*)"', content)
    assert round_trip_matches, "expected at least one round-trip assertion"
    for s, expected in round_trip_matches:
        assert s == expected, "round trip literal must assert decode(encode(s)) == s"
        assert _independent_decode(_independent_encode(s)) == s, s


def test_keystone_ids_reference_real_test_functions():
    content = c02.get_sandbox_fixture()[c02.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c02.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c02.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_round_trip():
    # decode(encode("")) is degenerate: both functions are no-ops on the empty string
    # regardless of implementation correctness, so it shouldn't gate the score.
    assert f"{c02.VISIBLE_TEST_PATH}::test_round_trip_empty" not in c02.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c02.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c02.get_grading_payload()
    assert payload["tests"][c02.VISIBLE_TEST_PATH] == c02.get_sandbox_fixture()[c02.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "rle_codec", "functions": ["encode", "decode"]}
    assert payload["keystone_test_ids"] == c02.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c02.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "rle_codec.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["rle_codec.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c02", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c02"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c02.get_task_statement()
    assert (public / "repo" / c02.VISIBLE_TEST_PATH).read_text() == c02.get_sandbox_fixture()[c02.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c02.get_compiled_plan()

    assert (private / c02.VISIBLE_TEST_PATH).read_text() == c02.get_grading_payload()["tests"][c02.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c02.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c02.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
