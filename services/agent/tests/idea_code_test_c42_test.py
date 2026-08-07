"""
Adversarial offline checks for codebench task c42 (layered-field-record-batch-validation) — no
Docker, no LLM.

Mirrors idea_code_test_c06_test.py: prove the task module's own claims are internally consistent
(ground truth is actually correct, keystone ids reference real tests, the compiled plan is
well-formed) BEFORE anything ever reaches a live sandbox. c42 is the first THREE-leaf codebench
task, so this file additionally checks the leaf_a/leaf_b/leaf_c dependency wiring: exactly three
leaves in a strict chain, leaf_c depending on leaf_b (NOT leaf_a directly — matching the real
import graph), and each dependent leaf's instruction referencing its immediate upstream leaf via
a ``{leaf_x}`` placeholder (the mechanism ``execution_compiled_code.py``'s ``substitute_deps``/
``_dep_text`` fill in at runtime).

Second generation: the first version of this task was live-calibrated via aider (qwen2.5:14b) and
scored a perfect 1.0/1.0 -- aider gets the ENTIRE task statement in one message (see
``codebench/agents/aider/run_task.sh``; it never walks the leaf-by-leaf plan at all), so the
per-leaf ``{leaf_x}`` degradation this task's docstring describes only ever bites the badmodel
harness. The composition-level canonical test was hardened with a genuine cross-layer bug magnet
(NaN/bool handling in ``validate_in_range`` that only gets caught at the final batch-level
assertion), so this file adds a live mutation test (``test_nan_blind_range_check_is_caught``)
proving a plausible near-miss implementation is actually rejected by the canonical suite, not just
asserting the literal expected values look right.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c42_layered_field_record_batch_validation as c42


# --- independent reimplementation (deliberately NOT importing the task module's own strings) ---

def _validate_non_empty_string(value):
    if not isinstance(value, str):
        return "must be a string"
    if not value.strip():
        return "must not be empty"
    return None


def _validate_positive_int(value):
    if not isinstance(value, int) or isinstance(value, bool):
        return "must be an integer"
    if value <= 0:
        return "must be positive"
    return None


def _validate_in_range(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be a number"
    # NaN check written with the classic self-inequality trick (value != value is True only for
    # NaN) -- deliberately a DIFFERENT technique than field_validators.py's own math.isnan() /
    # "not (lo <= value <= hi)" approach, to keep this reimplementation genuinely independent.
    if value != value:
        return f"must be between {lo} and {hi}"
    if value < lo or value > hi:
        return f"must be between {lo} and {hi}"
    return None


def _validate_record(record):
    if not isinstance(record, dict):
        return ["record must be a dict"]
    errors = []
    for field in ("name", "age", "score"):
        if field not in record:
            errors.append(f"missing field: {field}")
    if "name" in record:
        err = _validate_non_empty_string(record["name"])
        if err:
            errors.append(f"name: {err}")
    if "age" in record:
        err = _validate_positive_int(record["age"])
        if err:
            errors.append(f"age: {err}")
    if "score" in record:
        err = _validate_in_range(record["score"], 0, 100)
        if err:
            errors.append(f"score: {err}")
    return errors


def _validate_batch(records):
    errors_by_index = {}
    for i, rec in enumerate(records):
        errs = _validate_record(rec)
        if errs:
            errors_by_index[i] = errs
    return {
        "total": len(records),
        "valid": len(records) - len(errors_by_index),
        "invalid": len(errors_by_index),
        "errors": errors_by_index,
    }


def test_ground_truth_field_validators_are_internally_correct():
    assert _validate_non_empty_string("   ") == "must not be empty"
    assert _validate_non_empty_string(5) == "must be a string"
    assert _validate_non_empty_string("Alice") is None
    assert _validate_positive_int(0) == "must be positive"
    assert _validate_positive_int(-1) == "must be positive"
    assert _validate_positive_int(True) == "must be an integer"
    assert _validate_positive_int(3) is None
    assert _validate_in_range(-1, 0, 100) == "must be between 0 and 100"
    assert _validate_in_range(101, 0, 100) == "must be between 0 and 100"
    assert _validate_in_range("x", 0, 100) == "must be a number"
    assert _validate_in_range(50, 0, 100) is None
    assert _validate_in_range(True, 0, 100) == "must be a number"


def test_ground_truth_positive_int_rejects_float_even_when_whole_and_positive():
    assert _validate_positive_int(25.0) == "must be an integer"
    assert _validate_positive_int(-2.5) == "must be an integer"
    assert isinstance(25.0, (int, float)) and not isinstance(25.0, int)  # confirms the trap


def test_ground_truth_nan_range_check_is_a_real_python_gotcha():
    # Prove the underlying claim independently, with math.isnan (yet another independent
    # technique) rather than reusing this file's own value != value reimplementation.
    nan = float("nan")
    assert math.isnan(nan)
    assert not (nan < 0 or nan > 100)   # the naive/buggy check: both comparisons are False
    assert not (0 <= nan <= 100)        # the correct check: this IS True (nan is out of range)
    assert _validate_in_range(nan, 0, 100) == "must be between 0 and 100"


def test_ground_truth_record_validator_composes_without_short_circuit():
    assert _validate_record({"name": "Eve", "age": 1, "score": 0}) == []
    assert _validate_record({}) == [
        "missing field: name", "missing field: age", "missing field: score",
    ]
    assert _validate_record({"name": "", "age": -1, "score": 200}) == [
        "name: must not be empty", "age: must be positive", "score: must be between 0 and 100",
    ]
    assert _validate_record("not a dict") == ["record must be a dict"]


def test_ground_truth_record_validator_propagates_nan_and_bool_score():
    assert _validate_record({"name": "X", "age": 1, "score": float("nan")}) == [
        "score: must be between 0 and 100",
    ]
    assert _validate_record({"name": "X", "age": 1, "score": True}) == [
        "score: must be a number",
    ]


def test_ground_truth_record_validator_propagates_float_age():
    assert _validate_record({"name": "X", "age": 25.0, "score": 50}) == [
        "age: must be an integer",
    ]


def test_ground_truth_batch_validator_mixed_report():
    records = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "", "age": 25, "score": 90},
        {"name": "Bob", "age": -5, "score": 50},
        {"name": "Carol", "age": 40, "score": 150},
        {"name": "Dan", "age": 22},
        {"name": "Eve", "age": 18, "score": float("nan")},
        {"name": "Frank", "age": 50, "score": True},
    ]
    assert _validate_batch(records) == {
        "total": 7,
        "valid": 1,
        "invalid": 6,
        "errors": {
            1: ["name: must not be empty"],
            2: ["age: must be positive"],
            3: ["score: must be between 0 and 100"],
            4: ["missing field: score"],
            5: ["score: must be between 0 and 100"],
            6: ["score: must be a number"],
        },
    }
    assert _validate_batch([]) == {"total": 0, "valid": 0, "invalid": 0, "errors": {}}


def test_embedded_test_file_asserts_match_ground_truth():
    content = c42.get_grading_payload()["tests"][c42._TEST_FILE_PATH]
    # Spot-check the same literal values against the independently-written reimplementation.
    assert 'validate_in_range(-1, 0, 100) == "must be between 0 and 100"' in content
    assert 'validate_record({"name": "", "age": -1, "score": 200})' in content
    assert '"missing field: name", "missing field: age", "missing field: score"' in content
    assert '"total": 7,' in content and '"invalid": 6,' in content
    # the hardened NaN/bool-in-range edge cases, at both the unit and composition levels
    assert 'validate_in_range(float("nan"), 0, 100) == "must be between 0 and 100"' in content
    assert 'validate_in_range(True, 0, 100) == "must be a number"' in content
    assert '"score": float("nan")' in content
    assert '"score": True' in content
    # the 2026-08-07 float-age hardening, at both the unit and composition levels
    assert 'validate_positive_int(25.0) == "must be an integer"' in content
    assert '"age": 25.0, "score": 50' in content
    assert '"age: must be an integer"' in content


def test_keystone_ids_reference_real_test_functions():
    content = c42.get_grading_payload()["tests"][c42._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c42.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c42._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_single_function_unit_checks():
    # bare field-validator unit tests don't exercise cross-module composition.
    for name in ("test_validate_non_empty_string_rejects_blank_and_wrong_type",
                 "test_validate_positive_int_rejects_non_positive_and_bool",
                 "test_validate_positive_int_rejects_float_even_when_whole_and_positive",
                 "test_validate_in_range_rejects_out_of_bounds_and_wrong_type",
                 "test_validate_in_range_rejects_nan",
                 "test_record_validator_reports_nan_and_bool_score",
                 "test_batch_validator_empty_list"):
        assert f"{c42._TEST_FILE_PATH}::{name}" not in c42.KEYSTONE_TEST_IDS


def test_keystone_includes_the_2026_08_07_float_age_composition_case():
    assert (
        f"{c42._TEST_FILE_PATH}::"
        "test_record_validator_rejects_float_age_despite_looking_numerically_valid"
        in c42.KEYSTONE_TEST_IDS
    )


def test_visibility_is_hidden():
    assert c42.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c42.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c42.get_grading_payload()
    assert payload["tests"] == {c42._TEST_FILE_PATH: c42._TEST_FILE_CONTENT}
    assert payload["keystone_test_ids"] == c42.KEYSTONE_TEST_IDS
    modules = payload["entrypoint"]["modules"]
    assert {m["module"] for m in modules} == {
        "field_validators", "record_validator", "batch_validator",
    }
    field_mod = next(m for m in modules if m["module"] == "field_validators")
    record_mod = next(m for m in modules if m["module"] == "record_validator")
    batch_mod = next(m for m in modules if m["module"] == "batch_validator")
    assert set(field_mod["functions"]) == {
        "validate_non_empty_string", "validate_positive_int", "validate_in_range",
    }
    assert record_mod["functions"] == ["validate_record"]
    assert batch_mod["functions"] == ["validate_batch"]


def test_compiled_plan_has_exactly_three_leaves_in_a_strict_chain():
    plan = c42.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["leaf_a", "leaf_b", "leaf_c"]
    leaf_a, leaf_b, leaf_c = plan["leaves"]
    assert leaf_a["depends_on"] == []
    assert leaf_b["depends_on"] == ["leaf_a"]
    # leaf_c depends on leaf_b ONLY -- it never touches field_validators directly, matching the
    # real import graph (batch_validator imports record_validator, not field_validators).
    assert leaf_c["depends_on"] == ["leaf_b"]


def test_leaf_b_instruction_references_leaf_a_via_placeholder_and_restates_the_api():
    plan = c42.get_compiled_plan()
    leaf_b = plan["leaves"][1]
    assert "{leaf_a}" in leaf_b["instruction"], (
        "leaf_b must template leaf_a's actual outcome via {leaf_a} -- substitute_deps() only "
        "replaces placeholders that literally match a dependency id"
    )
    for name in ("validate_non_empty_string", "validate_positive_int", "validate_in_range"):
        assert name in leaf_b["instruction"], name
    assert "record_validator.py" in leaf_b["instruction"]


def test_leaf_c_instruction_references_leaf_b_via_placeholder_and_restates_the_api():
    plan = c42.get_compiled_plan()
    leaf_c = plan["leaves"][2]
    assert "{leaf_b}" in leaf_c["instruction"], (
        "leaf_c must template leaf_b's actual outcome via {leaf_b}"
    )
    # And must NOT template a leaf it doesn't depend on.
    assert "{leaf_a}" not in leaf_c["instruction"]
    assert "validate_record" in leaf_c["instruction"]
    assert "batch_validator.py" in leaf_c["instruction"]


def test_leaf_a_instruction_names_the_exact_api_downstream_leaves_will_rely_on():
    plan = c42.get_compiled_plan()
    leaf_a = plan["leaves"][0]
    for name in ("validate_non_empty_string", "validate_positive_int", "validate_in_range"):
        assert name in leaf_a["instruction"], name
    assert "field_validators.py" in leaf_a["instruction"]


def test_no_leaf_instruction_leaks_the_private_test_fixture_values():
    # The specific batch scenario embedded in the private test (Alice/Bob/.../Frank, the exact
    # 7-record mix, the resulting index->errors mapping) must never appear verbatim in any
    # PUBLIC leaf instruction -- only the general rule (including the NaN/bool edge-case rule)
    # may appear there.
    plan = c42.get_compiled_plan()
    all_instructions = " ".join(leaf["instruction"] for leaf in plan["leaves"])
    for leaked in ("Alice", "Carol", "Dan", "Frank", '"total": 5', '"total": 7', '"invalid": 6'):
        assert leaked not in all_instructions, leaked


def test_task_statement_does_not_leak_the_private_test_fixture_values():
    # get_task_statement() becomes aider's ENTIRE prompt.md in one shot (see
    # codebench/agents/aider/run_task.sh) -- it must be held to the exact same leak bar as the
    # leaf instructions, not just described-in-general-terms.
    statement = c42.get_task_statement()
    for leaked in (
        "Alice", "Bob", "Carol", "Dan", "Eve", "Frank",
        '"total": 5', '"total": 7', '"invalid": 6', '"invalid": 4',
    ):
        assert leaked not in statement, leaked


def test_compiled_plan_structure():
    plan = c42.get_compiled_plan()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {
        "op": "submit_files",
        "files": ["field_validators.py", "record_validator.py", "batch_validator.py"],
    }
    # Must be JSON-serializable as-is -- see c01's identical check for why.
    json.dumps(plan)


# --- mutation test: a plausible near-miss must actually be caught -----------------------------

_NEAR_MISS_FIELD_VALIDATORS = '''\
def validate_non_empty_string(value):
    if not isinstance(value, str):
        return "must be a string"
    if not value.strip():
        return "must not be empty"
    return None


def validate_positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int):
        return "must be an integer"
    if value <= 0:
        return "must be positive"
    return None


def validate_in_range(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be a number"
    # the classic NaN-blind bounds check: every comparison against NaN is False in Python, so
    # this silently treats NaN as if it were in range.
    if value < lo or value > hi:
        return f"must be between {lo} and {hi}"
    return None
'''

_CORRECT_RECORD_VALIDATOR = '''\
from field_validators import validate_non_empty_string, validate_positive_int, validate_in_range


def validate_record(record):
    if not isinstance(record, dict):
        return ["record must be a dict"]
    errors = []
    for field in ("name", "age", "score"):
        if field not in record:
            errors.append(f"missing field: {field}")
    if "name" in record:
        err = validate_non_empty_string(record["name"])
        if err:
            errors.append(f"name: {err}")
    if "age" in record:
        err = validate_positive_int(record["age"])
        if err:
            errors.append(f"age: {err}")
    if "score" in record:
        err = validate_in_range(record["score"], 0, 100)
        if err:
            errors.append(f"score: {err}")
    return errors
'''

_CORRECT_BATCH_VALIDATOR = '''\
from record_validator import validate_record


def validate_batch(records):
    errors_by_index = {}
    for i, rec in enumerate(records):
        errs = validate_record(rec)
        if errs:
            errors_by_index[i] = errs
    return {
        "total": len(records),
        "valid": len(records) - len(errors_by_index),
        "invalid": len(errors_by_index),
        "errors": errors_by_index,
    }
'''


def test_nan_blind_range_check_is_caught(tmp_path):
    # A plausible near-miss: field_validators.py gets everything right except the NaN edge case
    # (a genuine, well-known Python gotcha, not an obscure trick). record_validator.py and
    # batch_validator.py are otherwise fully correct. Prove the canonical suite actually rejects
    # this -- and specifically fails the hardened composition-level checks, not just a stray
    # unrelated assertion -- rather than trusting that the new test text "looks like" it should
    # catch it.
    (tmp_path / "field_validators.py").write_text(_NEAR_MISS_FIELD_VALIDATORS)
    (tmp_path / "record_validator.py").write_text(_CORRECT_RECORD_VALIDATOR)
    (tmp_path / "batch_validator.py").write_text(_CORRECT_BATCH_VALIDATOR)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_batch_validator.py").write_text(c42._TEST_FILE_CONTENT)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_batch_validator.py"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    for expected_failure in (
        "test_validate_in_range_rejects_nan",
        "test_record_validator_reports_nan_and_bool_score",
        "test_batch_validator_mixed_report",
    ):
        assert expected_failure in result.stdout, result.stdout
    # And a correct implementation of the very same near-miss's siblings must still pass
    # everything else -- confirming the failures are specific to the NaN edge case, not a
    # collateral break caused by the mutation (17 tests total, exactly 3 fail).
    assert "14 passed" in result.stdout, result.stdout


# 2026-08-07 near-miss: field_validators.py gets NaN handling and bool-exclusion right in BOTH
# validate_positive_int and validate_in_range, but validate_positive_int's type check has been
# loosened from `isinstance(value, int)` to `isinstance(value, (int, float))` -- the exact
# numeric-type-check shape validate_in_range legitimately needs, reused by habit in the wrong
# place -- so it silently accepts floats as valid ages.
_FLOAT_AGE_NEAR_MISS_FIELD_VALIDATORS = '''\
def validate_non_empty_string(value):
    if not isinstance(value, str):
        return "must be a string"
    if not value.strip():
        return "must not be empty"
    return None


def validate_positive_int(value):
    # BUG: loosened to accept floats too, the same type-check shape validate_in_range needs.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be an integer"
    if value <= 0:
        return "must be positive"
    return None


def validate_in_range(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "must be a number"
    if not (lo <= value <= hi):
        return f"must be between {lo} and {hi}"
    return None
'''


def test_float_age_type_laxness_is_caught(tmp_path):
    # A plausible near-miss: field_validators.py gets NaN and bool-exclusion right everywhere,
    # but validate_positive_int's type check has been loosened to accept floats too. Prove the
    # canonical suite actually rejects this, and specifically via the hardened float-age cases,
    # not a stray unrelated assertion.
    (tmp_path / "field_validators.py").write_text(_FLOAT_AGE_NEAR_MISS_FIELD_VALIDATORS)
    (tmp_path / "record_validator.py").write_text(_CORRECT_RECORD_VALIDATOR)
    (tmp_path / "batch_validator.py").write_text(_CORRECT_BATCH_VALIDATOR)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_batch_validator.py").write_text(c42._TEST_FILE_CONTENT)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_batch_validator.py"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    for expected_failure in (
        "test_validate_positive_int_rejects_float_even_when_whole_and_positive",
        "test_record_validator_rejects_float_age_despite_looking_numerically_valid",
    ):
        assert expected_failure in result.stdout, result.stdout
    # Every OTHER case, including every NaN/bool case, is unaffected by this specific mutation
    # (17 tests total, exactly 2 fail).
    assert "15 passed" in result.stdout, result.stdout


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c42", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c42"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c42.get_task_statement()
    # Hidden task: no starter files under public/repo/ at all.
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c42.get_compiled_plan()

    assert (private / c42._TEST_FILE_PATH).read_text() == c42._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c42._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c42.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
