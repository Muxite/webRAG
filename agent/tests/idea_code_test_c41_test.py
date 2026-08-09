"""
Adversarial offline checks for codebench task c41 (bugfix-phone-dedupe) — no Docker, no LLM.

Second full replacement (see the task module's own docstring for why the previous design --
generic case/whitespace-insensitive string dedup -- was fully and correctly solved by BOTH real
round-3 submissions, which were algorithmically identical and simply correct). Same debug-task
validation shape as idea_code_test_c39_test.py / idea_code_test_c40_test.py:
  (1) the buggy starter genuinely fails Group A (order-preservation cases) when actually run,
  (2) the "obvious" partial fix (order-preserving walk, keyed either by plain string equality OR
      by the OLD task's fully-correct `.strip().lower()` formula) makes every Group A AND bonus
      case (visible AND hidden) agree with ground truth -- looks complete from the visible
      suite's point of view -- but still disagrees with ground truth on every Group B (hidden,
      digit-normalization-dependent) case, because neither formula does any digit-aware
      normalization at all,
  (3) three independently plausible near-miss mutants (missing country-code rule; over-
      generalized country-code rule; blank/no-digit entries dropped instead of deduped) each
      fail a narrow, predicted subset of Group B and nothing else,
  (4) the ACTUAL round-3 submitted source from both the badmodel and Aider harnesses (reproduced
      verbatim, adapted only to this task's function/module name) fails 7/9 Group B cases when
      run against the real canonical suite via pytest,
  (5) the fully correct fix agrees with ground truth on the ENTIRE canonical suite, and
  (6) the hidden cases are genuinely absent from the sandbox fixture while still being a strict
      superset built on top of the visible content at the identical relpath.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c41_bugfix_phone_dedupe as c41


# --------------------------------------------------------------------------------------------
# Ground truth, derived TWO independent ways (neither one is the task module's own source).
# --------------------------------------------------------------------------------------------
def _core_via_regex(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if len(digits) == 11 and digits[0] == "1":
        return digits[1:]
    return digits


def _core_via_char_filter(s: str) -> str:
    """Completely different code path: no regex engine at all, a manual character filter and
    manual length/prefix check."""
    digits = "".join(ch for ch in s if ch in "0123456789")
    if len(digits) == 11 and digits.startswith("1"):
        return digits[1:]
    return digits


def _independent_dedupe(numbers: list[str], core_fn) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for n in numbers:
        key = core_fn(n)
        if key not in seen:
            seen.add(key)
            result.append(n)
    return result


def _exec_dedupe(source: str):
    import importlib.util

    spec = importlib.util.spec_from_loader("c41_phone_dedupe_under_test", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)  # noqa: S102 — trusted in-repo fixture
    return module.dedupe_phone_numbers


def _extract_cases(content: str) -> list[tuple[list[str], list[str]]]:
    """Parse with ast (not regex) so both call shapes — an inline list literal, or a local
    `numbers = [...]` referenced by name — resolve correctly."""
    tree = ast.parse(content)
    cases: list[tuple[list[str], list[str]]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        local_vars: dict[str, object] = {}
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                local_vars[stmt.targets[0].id] = ast.literal_eval(stmt.value)
            elif isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare):
                call = stmt.test.left
                if (
                    isinstance(call, ast.Call)
                    and getattr(call.func, "id", None) == "dedupe_phone_numbers"
                    and len(call.args) == 1
                    and len(stmt.test.comparators) == 1
                ):
                    arg = call.args[0]
                    numbers = local_vars[arg.id] if isinstance(arg, ast.Name) else ast.literal_eval(arg)
                    expected = ast.literal_eval(stmt.test.comparators[0])
                    cases.append((numbers, expected))
    return cases


_GROUP_A = {
    "basic_repeats": (["415-555-0111", "212-555-0122", "415-555-0111"],),
    "no_dup_unsorted": (["999-555-0133", "212-555-0122", "415-555-0111"],),
    "order_reversed": (["444-555-0144", "333-555-0155", "222-555-0166", "444-555-0144", "333-555-0155"],),
}
_BONUS_VISIBLE = {
    "all_same": (["555-555-0177"] * 4,),
    "empty": ([],),
    "single": (["555-555-0188"],),
    "clustered_end": (["111-555-0199", "222-555-0200", "333-555-0211", "111-555-0199", "222-555-0200"],),
    "interleaved": (
        ["111-555-0222", "222-555-0233", "111-555-0222", "333-555-0244",
         "222-555-0233", "111-555-0222", "444-555-0255"],
    ),
}
_GROUP_B = {
    "format_variants": (["(415) 555-0100", "415-555-0100", "415.555.0100"],),
    "country_code_variant": (["+1 415-555-0100", "415-555-0100"],),
    "country_code_no_punct": (["4155550100", "14155550100"],),
    "eleven_not_starting_with_one": (["21415550100", "1415550100"],),
    "blank_and_nondigit_deduped_not_dropped": (["", "415-555-0100", "n/a", "415.555.0100"],),
    "multiple_nondigit_variants": (["n/a", "unknown", "555-1234"],),
}
_BONUS_HIDDEN = {
    "spaces_variant": (["415 555 0100", "4155550100"],),
    "country_code_no_punct_reversed": (["14155550100", "415-555-0100"],),
    "ten_digit_starting_with_one_untouched": (["1234567890", "234567890"],),
}
# Group B minus "eleven_not_starting_with_one": that one case is a NEGATIVE/anti-over-merge
# check (correct answer keeps two numbers DISTINCT) whose expected output happens to coincide
# with plain string equality and with case/whitespace-only normalization -- it exists
# specifically to catch a mutant that over-generalizes the country-code rule (see
# test_mutant_always_drop_first_of_eleven_fails_exactly_the_predicted_case), not to catch an
# implementation that never normalizes at all. Every case in this subset, by contrast, requires
# genuine digit-based MERGING that neither plain nor case/whitespace equality would ever produce.
_GROUP_B_REQUIRES_MERGE = {
    name: value for name, value in _GROUP_B.items() if name != "eleven_not_starting_with_one"
}


def _check_all(fn, table, core_fn=_core_via_regex):
    for name, (numbers,) in table.items():
        expected = _independent_dedupe(numbers, core_fn)
        actual = fn(list(numbers))
        yield name, expected, actual


def test_the_two_independent_ground_truth_implementations_agree():
    for table in (_GROUP_A, _GROUP_B, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, (numbers,) in table.items():
            regex_result = _independent_dedupe(numbers, _core_via_regex)
            filter_result = _independent_dedupe(numbers, _core_via_char_filter)
            assert regex_result == filter_result, name


def test_embedded_visible_test_file_asserts_match_ground_truth():
    cases = _extract_cases(c41._TEST_FILE_CONTENT)
    assert len(cases) == 8
    for numbers, expected in cases:
        assert _independent_dedupe(numbers, _core_via_regex) == expected


def test_embedded_hidden_addendum_asserts_match_ground_truth():
    cases = _extract_cases(c41._HIDDEN_TEST_ADDENDUM)
    assert len(cases) == 9
    for numbers, expected in cases:
        assert _independent_dedupe(numbers, _core_via_regex) == expected


def test_group_b_requires_merge_cases_each_genuinely_require_digit_normalization():
    """Every case in _GROUP_B_REQUIRES_MERGE must have a different (shorter) answer under
    digit-core comparison than under plain OR case/whitespace-insensitive string comparison --
    else it wouldn't discriminate the normalization rule at all."""
    for name, (numbers,) in _GROUP_B_REQUIRES_MERGE.items():
        digit_expected = _independent_dedupe(numbers, _core_via_regex)
        plain_len = len(dict.fromkeys(numbers))
        casefold_len = len(dict.fromkeys(n.strip().lower() for n in numbers))
        assert len(digit_expected) < plain_len or len(digit_expected) < casefold_len, (
            f"{name}: {numbers!r} does not discriminate digit-normalization from plain/casefold "
            f"equality (digit={digit_expected!r})"
        )


def test_eleven_not_starting_with_one_is_a_deliberate_anti_over_merge_case():
    """Unlike the rest of Group B, this case's correct answer (stay distinct) coincides with
    plain AND case/whitespace-insensitive equality -- it exists to catch a DIFFERENT mutant (one
    that over-generalizes the country-code rule), not a missing-normalization one."""
    numbers, = _GROUP_B["eleven_not_starting_with_one"]
    digit_expected = _independent_dedupe(numbers, _core_via_regex)
    assert digit_expected == numbers  # nothing merged
    assert len(dict.fromkeys(numbers)) == len(numbers)  # plain equality already agrees


def test_canonical_test_file_is_visible_plus_hidden_with_no_extra_cases():
    canonical_cases = _extract_cases(c41._CANONICAL_TEST_FILE_CONTENT)
    visible_cases = _extract_cases(c41._TEST_FILE_CONTENT)
    hidden_cases = _extract_cases(c41._HIDDEN_TEST_ADDENDUM)
    assert len(canonical_cases) == 17
    assert canonical_cases == visible_cases + hidden_cases


def test_starter_module_actually_buggy_on_group_a_when_run():
    buggy = _exec_dedupe(c41._PHONE_DEDUPE_PY_CONTENT)
    for name, expected, actual in _check_all(buggy, _GROUP_A):
        assert actual != expected, f"{name} was expected to be BROKEN by the order bug"


def test_starter_module_actually_buggy_on_group_b_when_run():
    buggy = _exec_dedupe(c41._PHONE_DEDUPE_PY_CONTENT)
    for name, expected, actual in _check_all(buggy, _GROUP_B):
        assert actual != expected, f"{name} was expected to be BROKEN on the unmodified starter"


def test_starter_module_passes_bonus_cases_when_run():
    buggy = _exec_dedupe(c41._PHONE_DEDUPE_PY_CONTENT)
    for name, expected, actual in _check_all(buggy, _BONUS_VISIBLE):
        assert actual == expected, name


# --------------------------------------------------------------------------------------------
# The "obvious" partial fix: order-preserving walk, plain string equality -- the natural first
# response to "it doesn't preserve order," and completely blind to normalization of any kind.
# --------------------------------------------------------------------------------------------
_PLAIN_EQUALITY_FIX_SOURCE = (
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = set()\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        if n not in seen:\n"
    "            seen.add(n)\n"
    "            result.append(n)\n"
    "    return result\n"
)

# The formula that was FULLY CORRECT for this task's previous (now-retired) design -- order-
# preserving, case/whitespace-insensitive -- but has nothing to do with phone-number digit
# normalization.
_OLD_TASK_FORMULA_SOURCE = (
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = set()\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        key = n.strip().lower()\n"
    "        if key not in seen:\n"
    "            seen.add(key)\n"
    "            result.append(n)\n"
    "    return result\n"
)

_FULL_FIX_SOURCE = (
    "import re\n\n"
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = {}\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        digits = re.sub(r'\\D', '', n)\n"
    "        if len(digits) == 11 and digits[0] == '1':\n"
    "            digits = digits[1:]\n"
    "        if digits not in seen:\n"
    "            seen[digits] = n\n"
    "            result.append(n)\n"
    "    return result\n"
)


def test_plain_equality_fix_looks_complete_from_the_visible_suite_alone():
    fixed = _exec_dedupe(_PLAIN_EQUALITY_FIX_SOURCE)
    for table in (_GROUP_A, _BONUS_VISIBLE):
        for name, expected, actual in _check_all(fixed, table):
            assert actual == expected, name


def test_plain_equality_fix_fails_every_group_b_requires_merge_case():
    fixed = _exec_dedupe(_PLAIN_EQUALITY_FIX_SOURCE)
    for name, expected, actual in _check_all(fixed, _GROUP_B_REQUIRES_MERGE):
        assert actual != expected, f"{name} should still be WRONG under the plain-equality fix"
    # the anti-over-merge case coincidentally still passes under plain equality (see
    # test_eleven_not_starting_with_one_is_a_deliberate_anti_over_merge_case)
    numbers, expected = _GROUP_B["eleven_not_starting_with_one"][0], _independent_dedupe(
        _GROUP_B["eleven_not_starting_with_one"][0], _core_via_regex
    )
    assert fixed(list(numbers)) == expected


def test_old_task_formula_also_looks_complete_from_the_visible_suite_alone():
    """The exact formula that was the FULLY CORRECT answer for this task's previous design is
    STILL wrong here -- confirms the replacement genuinely changed what 'correct' requires, not
    just the surface-level test data."""
    fixed = _exec_dedupe(_OLD_TASK_FORMULA_SOURCE)
    for table in (_GROUP_A, _BONUS_VISIBLE):
        for name, expected, actual in _check_all(fixed, table):
            assert actual == expected, name


def test_old_task_formula_fails_every_group_b_requires_merge_case():
    fixed = _exec_dedupe(_OLD_TASK_FORMULA_SOURCE)
    for name, expected, actual in _check_all(fixed, _GROUP_B_REQUIRES_MERGE):
        assert actual != expected, f"{name} should still be WRONG under the old task's formula"


def test_full_fix_matches_ground_truth_on_every_case():
    fixed = _exec_dedupe(_FULL_FIX_SOURCE)
    for table in (_GROUP_A, _GROUP_B, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, expected, actual in _check_all(fixed, table):
            assert actual == expected, name


# --------------------------------------------------------------------------------------------
# Three independently plausible near-miss mutants, each with a narrow, predicted failure set.
# --------------------------------------------------------------------------------------------
_MUTANT_NO_COUNTRY_CODE_SOURCE = (
    "import re\n\n"
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = set()\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        digits = re.sub(r'\\D', '', n)\n"
    "        if digits not in seen:\n"
    "            seen.add(digits)\n"
    "            result.append(n)\n"
    "    return result\n"
)

_MUTANT_ALWAYS_DROP_FIRST_OF_ELEVEN_SOURCE = (
    "import re\n\n"
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = set()\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        digits = re.sub(r'\\D', '', n)\n"
    "        if len(digits) == 11:\n"
    "            digits = digits[1:]\n"
    "        if digits not in seen:\n"
    "            seen.add(digits)\n"
    "            result.append(n)\n"
    "    return result\n"
)

_MUTANT_SKIP_BLANK_SOURCE = (
    "import re\n\n"
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = set()\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        digits = re.sub(r'\\D', '', n)\n"
    "        if not digits:\n"
    "            continue\n"
    "        if len(digits) == 11 and digits[0] == '1':\n"
    "            digits = digits[1:]\n"
    "        if digits not in seen:\n"
    "            seen.add(digits)\n"
    "            result.append(n)\n"
    "    return result\n"
)


def test_mutant_no_country_code_fails_exactly_the_predicted_cases():
    fixed = _exec_dedupe(_MUTANT_NO_COUNTRY_CODE_SOURCE)
    failing = {
        name for name, expected, actual in _check_all(fixed, _GROUP_B) if actual != expected
    }
    assert failing == {"country_code_variant", "country_code_no_punct"}, failing


def test_mutant_always_drop_first_of_eleven_fails_exactly_the_predicted_case():
    fixed = _exec_dedupe(_MUTANT_ALWAYS_DROP_FIRST_OF_ELEVEN_SOURCE)
    failing = {
        name for name, expected, actual in _check_all(fixed, _GROUP_B) if actual != expected
    }
    assert failing == {"eleven_not_starting_with_one"}, failing


def test_mutant_skip_blank_fails_exactly_the_predicted_cases():
    fixed = _exec_dedupe(_MUTANT_SKIP_BLANK_SOURCE)
    failing = {
        name for name, expected, actual in _check_all(fixed, _GROUP_B) if actual != expected
    }
    assert failing == {
        "blank_and_nondigit_deduped_not_dropped", "multiple_nondigit_variants",
    }, failing


# --------------------------------------------------------------------------------------------
# The REAL round-3 submitted source from both harnesses, reproduced verbatim from
# badmodel-lab/codebench/results/runs/coordinator_batch4/c41__{aider,badmodel}__qwen2.5_14b/
# submission/dedupe_list.py (renamed function/import only) -- the strongest possible evidence
# that this replacement discriminates the actual failure mode observed live, not a hypothetical.
# --------------------------------------------------------------------------------------------
_REAL_AIDER_SUBMISSION_SOURCE = (
    "from collections import OrderedDict\n\n"
    "def dedupe_phone_numbers(numbers):\n"
    "    seen = OrderedDict()\n"
    "    for n in numbers:\n"
    "        normalized_item = n.strip().lower()\n"
    "        if normalized_item not in seen:\n"
    "            seen[normalized_item] = n\n"
    "    return list(seen.values())\n"
)

_REAL_BADMODEL_SUBMISSION_SOURCE = (
    "def dedupe_phone_numbers(numbers):\n"
    "    seen_items = set()\n"
    "    output = []\n"
    "    for n in numbers:\n"
    "        clean_item = n.lower().strip()\n"
    "        if clean_item not in seen_items:\n"
    "            output.append(n)\n"
    "            seen_items.add(clean_item)\n"
    "    return output\n"
)


def test_real_aider_round3_submission_fails_seven_of_nine_hidden_cases():
    """The strongest possible evidence: the ACTUAL round-3 submitted source (reproduced
    verbatim) fails 7 of the 9 hidden cases combined across Group B and the bonus-hidden set --
    every case that requires genuine digit-based normalization, and none of the two cases whose
    correct answer happens to coincide with plain/casefold string distinctness."""
    fixed = _exec_dedupe(_REAL_AIDER_SUBMISSION_SOURCE)
    failing_group_b = {
        name for name, expected, actual in _check_all(fixed, _GROUP_B) if actual != expected
    }
    failing_bonus = {
        name for name, expected, actual in _check_all(fixed, _BONUS_HIDDEN) if actual != expected
    }
    assert failing_group_b == set(_GROUP_B_REQUIRES_MERGE)  # every merge-requiring case fails
    assert "eleven_not_starting_with_one" not in failing_group_b  # coincidentally passes
    assert failing_bonus == {"spaces_variant", "country_code_no_punct_reversed"}
    assert "ten_digit_starting_with_one_untouched" not in failing_bonus  # coincidentally passes
    assert len(failing_group_b) + len(failing_bonus) == 7


def test_real_badmodel_round3_submission_fails_the_same_way_as_aider():
    fixed_aider = _exec_dedupe(_REAL_AIDER_SUBMISSION_SOURCE)
    fixed_badmodel = _exec_dedupe(_REAL_BADMODEL_SUBMISSION_SOURCE)
    for table in (_GROUP_A, _GROUP_B, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, (numbers,) in table.items():
            assert fixed_aider(list(numbers)) == fixed_badmodel(list(numbers)), name


def test_keystone_ids_reference_real_test_functions_in_the_canonical_grading_file():
    canonical_content = c41.get_grading_payload()["tests"][c41.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", canonical_content, re.MULTILINE))
    assert len(defined) == 17
    for node_id in c41.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c41.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the canonical file"
    assert set(c41.KEYSTONE_TEST_IDS) == {
        f"{c41.VISIBLE_TEST_PATH}::test_basic_repeats_preserve_first_occurrence_order",
        f"{c41.VISIBLE_TEST_PATH}::test_no_duplicates_but_unsorted_input_stays_unsorted",
        f"{c41.VISIBLE_TEST_PATH}::test_order_reversed_relative_to_lexicographic_sort",
        f"{c41.VISIBLE_TEST_PATH}::test_formatting_variants_with_different_punctuation_are_duplicates_keep_first",
        f"{c41.VISIBLE_TEST_PATH}::test_country_code_variant_is_duplicate_of_bare_ten_digit",
        f"{c41.VISIBLE_TEST_PATH}::test_country_code_variant_with_no_punctuation_still_recognized",
        f"{c41.VISIBLE_TEST_PATH}::test_eleven_digit_number_not_starting_with_one_stays_distinct",
        f"{c41.VISIBLE_TEST_PATH}::test_blank_and_non_digit_entries_are_deduped_not_dropped",
        f"{c41.VISIBLE_TEST_PATH}::test_multiple_non_digit_variants_keep_first_verbatim",
    }


def test_category_is_hard():
    assert c41.get_test_metadata()["category"] == "hard"


def test_visibility_is_visible():
    assert c41.get_visibility() == "visible"


def test_sandbox_fixture_excludes_hidden_tests():
    fixture = c41.get_sandbox_fixture()
    assert fixture[c41.STARTER_MODULE_PATH] == c41._PHONE_DEDUPE_PY_CONTENT
    assert fixture[c41.VISIBLE_TEST_PATH] == c41._TEST_FILE_CONTENT
    assert "sorted(set(numbers))" in fixture[c41.STARTER_MODULE_PATH]
    hidden_names = set(re.findall(r"^def (test_\w+)\(", c41._HIDDEN_TEST_ADDENDUM, re.MULTILINE))
    assert len(hidden_names) == 9
    for name in hidden_names:
        assert name not in fixture[c41.VISIBLE_TEST_PATH]
    for leaked in ("(415) 555-0100", "+1 415-555-0100", "21415550100", "n/a"):
        assert leaked not in fixture[c41.VISIBLE_TEST_PATH]


def test_grading_payload_is_a_genuine_superset_of_the_visible_fixture():
    payload = c41.get_grading_payload()
    canonical = payload["tests"][c41.VISIBLE_TEST_PATH]
    visible = c41.get_sandbox_fixture()[c41.VISIBLE_TEST_PATH]
    assert canonical != visible
    assert canonical.startswith(visible)
    assert canonical == c41._CANONICAL_TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "phone_dedupe", "functions": ["dedupe_phone_numbers"]}
    assert payload["keystone_test_ids"] == c41.KEYSTONE_TEST_IDS


def test_compiled_plan_structure_and_no_leaked_values():
    plan = c41.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["fix_bug"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "phone_dedupe.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["phone_dedupe.py"]}

    instruction = leaf["instruction"]
    for numbers, expected in _extract_cases(c41._CANONICAL_TEST_FILE_CONTENT):
        if len(numbers) >= 2:
            assert str(numbers) not in instruction
            assert str(expected) not in instruction
    # never the literal mechanism/fix
    for leaked in (
        "re.sub", "\\d", "strip()", "lower()", "casefold", "digits[1:]", "digits[0]",
        "11 and", "len(digits)",
    ):
        assert leaked.lower() not in instruction.lower(), f"plan leaks the mechanism via {leaked!r}"
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c41", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c41"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c41.get_task_statement()
    assert (public / "repo" / c41.STARTER_MODULE_PATH).read_text() == c41._PHONE_DEDUPE_PY_CONTENT
    assert (public / "repo" / c41.VISIBLE_TEST_PATH).read_text() == c41._TEST_FILE_CONTENT
    assert json.loads((public / "plan.json").read_text()) == c41.get_compiled_plan()

    assert (private / c41.VISIBLE_TEST_PATH).read_text() == c41.get_grading_payload()["tests"][c41.VISIBLE_TEST_PATH]
    assert (private / c41.VISIBLE_TEST_PATH).read_text() == c41._CANONICAL_TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c41.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c41.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()


# --------------------------------------------------------------------------------------------
# Live pytest run of the ACTUAL round-3 submitted source against the REAL embedded canonical
# test file (not the reimplementations above) -- proves the actual shipped fixture, not just
# this validator's model of it, catches the real observed failure live.
# --------------------------------------------------------------------------------------------
def _run_pytest_against_impl(tmp_path, impl_source: str):
    (tmp_path / "phone_dedupe.py").write_text(impl_source)
    test_content = c41.get_grading_payload()["tests"][c41.VISIBLE_TEST_PATH]
    (tmp_path / "test_phone_dedupe.py").write_text(test_content)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", str(tmp_path / "test_phone_dedupe.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    return result.stdout + result.stderr


def test_reference_full_fix_passes_every_canonical_test_live(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _FULL_FIX_SOURCE)
    assert "17 passed" in output, output


def test_real_aider_submission_fails_live_against_real_canonical_suite(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _REAL_AIDER_SUBMISSION_SOURCE)
    assert "passed" in output
    assert re.search(r"(\d+) failed", output)
    failed_count = int(re.search(r"(\d+) failed", output).group(1))
    assert failed_count == 7, output
