"""
Adversarial offline checks for codebench task c13 (roman-numeral-codec) — no Docker, no LLM.

Mirrors idea_code_test_c01_test.py exactly: prove the task module's own claims are internally
consistent (ground truth is actually correct, keystone ids reference real tests, the compiled
plan is well-formed) BEFORE anything ever reaches a live sandbox, plus an end-to-end exercise
of materialize_task.py against this task.

The reimplementation below is a per-digit lookup-table method — structurally different from a
greedy-subtraction walk — used purely to independently re-derive the literal Roman-numeral
strings embedded in the task module's docstring and test fixture, to catch a hand-transcription
error (subtractive notation, e.g. CD vs CCCC, is exactly the kind of thing easy to get subtly
wrong by hand). During authoring this was additionally cross-checked against a second,
greedy-subtraction implementation across the ENTIRE valid range 1..3999 with zero mismatches.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c13_roman_numeral_codec as c13

_THOUSANDS = ["", "M", "MM", "MMM"]
_HUNDREDS = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
_TENS = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
_ONES = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]

_SYM_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _independent_to_roman(n: int) -> str:
    """Reimplemented independently of the task's own prose spec, via a per-digit lookup
    table (not the greedy-subtraction style an implementation might use), to catch a
    hand-transcription error in the task module's embedded expected values."""
    if not (1 <= n <= 3999):
        raise ValueError(f"n must be in range 1..3999, got {n}")
    th, rem = divmod(n, 1000)
    h, rem = divmod(rem, 100)
    t, o = divmod(rem, 10)
    return _THOUSANDS[th] + _HUNDREDS[h] + _TENS[t] + _ONES[o]


def _independent_from_roman(s: str) -> int:
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _SYM_VALS[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def test_ground_truth_values_are_internally_correct():
    for n, expected in [
        (1, "I"), (4, "IV"), (9, "IX"), (40, "XL"), (90, "XC"),
        (400, "CD"), (900, "CM"), (3999, "MMMCMXCIX"),
    ]:
        assert _independent_to_roman(n) == expected, n
    assert _independent_from_roman("MMMDCCCLXXXVIII") == 3888
    for n in (44, 1994, 2023):
        assert _independent_from_roman(_independent_to_roman(n)) == n


def test_full_range_round_trips_cleanly():
    # Belt-and-suspenders: every value in the documented valid range round-trips through
    # the independent reimplementation, not just the handful embedded in the test fixture.
    for n in range(1, 4000):
        assert _independent_from_roman(_independent_to_roman(n)) == n


def test_embedded_encode_assertions_match_ground_truth():
    content = c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    pairs = re.findall(r'to_roman\((\d+)\) == "([A-Z]+)"', content)
    assert pairs, "expected at least one literal to_roman assertion in the embedded test file"
    for n_str, roman in pairs:
        assert _independent_to_roman(int(n_str)) == roman, (n_str, roman)


def test_embedded_decode_assertion_matches_ground_truth():
    content = c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    pairs = re.findall(r'from_roman\("([A-Z]+)"\) == (\d+)', content)
    assert pairs, "expected at least one literal from_roman assertion in the embedded test file"
    for roman, n_str in pairs:
        assert _independent_from_roman(roman) == int(n_str), (roman, n_str)


def test_embedded_round_trip_assertions_are_self_consistent():
    content = c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    ns = re.findall(r"from_roman\(to_roman\((\d+)\)\) == (\d+)", content)
    assert ns, "expected at least one round-trip assertion in the embedded test file"
    for n_str, expected_str in ns:
        n, expected = int(n_str), int(expected_str)
        assert n == expected
        assert _independent_from_roman(_independent_to_roman(n)) == expected


def test_value_error_contract_present():
    content = c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    assert content.count("pytest.raises(ValueError)") == 3
    assert "to_roman(0)" in content
    assert "to_roman(4000)" in content
    assert "to_roman(-5)" in content


def test_keystone_ids_reference_real_test_functions():
    content = c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c13.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c13.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_and_contract_cases():
    # n=1 is degenerate (no subtractive notation involved) and the three ValueError tests
    # exercise input validation, not codec logic — none should gate the score.
    assert f"{c13.VISIBLE_TEST_PATH}::test_to_roman_one" not in c13.KEYSTONE_TEST_IDS
    assert f"{c13.VISIBLE_TEST_PATH}::test_to_roman_zero_raises" not in c13.KEYSTONE_TEST_IDS
    assert f"{c13.VISIBLE_TEST_PATH}::test_to_roman_over_max_raises" not in c13.KEYSTONE_TEST_IDS
    assert f"{c13.VISIBLE_TEST_PATH}::test_to_roman_negative_raises" not in c13.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c13.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c13.get_grading_payload()
    assert payload["tests"][c13.VISIBLE_TEST_PATH] == c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {
        "module": "roman_codec",
        "functions": ["to_roman", "from_roman"],
    }
    assert payload["keystone_test_ids"] == c13.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c13.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "roman_codec.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["roman_codec.py"]}
    # Must be JSON-serializable as-is: materialize_task.py writes it verbatim to
    # public/plan.json with no transformation, and the agent image never imports this
    # module at all (see agents/badmodel/Dockerfile's SECURITY comment) — a plan that
    # isn't plain JSON-safe data would silently break at materialize time.
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c13", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c13"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c13.get_task_statement()
    assert (public / "repo" / c13.VISIBLE_TEST_PATH).read_text() == c13.get_sandbox_fixture()[c13.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c13.get_compiled_plan()

    assert (private / c13.VISIBLE_TEST_PATH).read_text() == c13.get_grading_payload()["tests"][c13.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c13.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c13.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
