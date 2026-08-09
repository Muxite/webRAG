"""
codebench task c18 — soft/visible, behavior-preserving refactor.

Unlike c01-c13 (hard tasks with a single deterministic correct output), this is a SOFT
task: there is no one "correct" refactor, so grading leans on two different signals —
(1) a lightweight regression suite that pins down the STARTER module's exact behavior
(the one hard constraint: don't change what calc() returns) and (2) get_judge_rubric(),
an LLM judge scoring the refactor's clarity/structure, which is the real signal here.

The starter module (``messy.py``) is deliberately ugly-but-correct: deeply nested
if/elif, a repeated magic-number special-case pattern (0/1 edge branches that duplicate
what the general-case arithmetic already computes), unclear variable names (x1, tmp2),
and one genuinely unreachable branch (the final ``else`` under the sub op's
>/</== chain — for any two real numbers exactly one of those three always holds, so
that branch can never execute; kept as an explicit "remove this dead code" target).

Every value embedded in ``_TEST_FILE_CONTENT`` below was produced by actually IMPORTING
and RUNNING this exact messy.py (not hand-computed) — see
idea_code_test_c18_test.py::test_regression_values_match_running_the_real_starter_module,
which re-derives them the same way at test time.
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_messy.py"
STARTER_MODULE_PATH = "messy.py"

_MESSY_PY_CONTENT = '''\
def calc(a, b, op):
    x1 = a
    tmp2 = b
    if op == "add":
        if tmp2 == 0:
            result = x1 + 0
        else:
            result = x1 + tmp2
    elif op == "sub":
        if x1 > tmp2:
            result = x1 - tmp2
        elif x1 < tmp2:
            result = x1 - tmp2
        elif x1 == tmp2:
            result = 0
        else:
            result = -999
    elif op == "mul":
        if tmp2 == 1:
            result = x1 * 1
        else:
            result = x1 * tmp2
    elif op == "div":
        if tmp2 == 0:
            result = None
        elif tmp2 == 1:
            result = x1 / 1
        else:
            result = x1 / tmp2
    else:
        result = None
    return result
'''

_TEST_FILE_CONTENT = '''\
from messy import calc


def test_add_basic():
    assert calc(2, 3, "add") == 5


def test_add_with_zero():
    assert calc(5, 0, "add") == 5


def test_sub_positive():
    assert calc(10, 4, "sub") == 6


def test_sub_negative():
    assert calc(4, 10, "sub") == -6


def test_sub_equal():
    assert calc(7, 7, "sub") == 0


def test_mul_basic():
    assert calc(3, 4, "mul") == 12


def test_mul_by_one():
    assert calc(3, 1, "mul") == 3


def test_div_basic():
    assert calc(10, 2, "div") == 5.0


def test_div_by_zero():
    assert calc(10, 0, "div") is None


def test_div_by_one():
    assert calc(10, 1, "div") == 10.0


def test_unknown_op():
    assert calc(3, 5, "unknown") is None
'''

# All eleven cases gate the score: this task's whole point is "behavior must not change,"
# so partial credit on the regression suite has no soft-task analogue the way a degenerate
# case would on a hard task — every assertion here is load-bearing for that one hard
# constraint, and the OPEN-ended part (actual refactor quality) is judged separately via
# get_judge_rubric(), not through pytest at all.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_add_basic",
    f"{VISIBLE_TEST_PATH}::test_add_with_zero",
    f"{VISIBLE_TEST_PATH}::test_sub_positive",
    f"{VISIBLE_TEST_PATH}::test_sub_negative",
    f"{VISIBLE_TEST_PATH}::test_sub_equal",
    f"{VISIBLE_TEST_PATH}::test_mul_basic",
    f"{VISIBLE_TEST_PATH}::test_mul_by_one",
    f"{VISIBLE_TEST_PATH}::test_div_basic",
    f"{VISIBLE_TEST_PATH}::test_div_by_zero",
    f"{VISIBLE_TEST_PATH}::test_div_by_one",
    f"{VISIBLE_TEST_PATH}::test_unknown_op",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c18",
        "title": "refactor-messy-module",
        "category": "soft",
    }


def get_task_statement() -> str:
    return (
        "Refactor `messy.py` for clarity — the existing behavior must NOT change.\n\n"
        "`messy.py` already works correctly (it computes the right answers), but the "
        "code itself is bad: deeply nested if/elif chains, unclear variable names "
        "(`x1`, `tmp2`), a repeated magic-number special-casing pattern that duplicates "
        "logic the general case already handles correctly, and at least one dead/"
        "unreachable branch.\n\n"
        "Improve it: extract the logic into smaller, well-named functions; rename "
        "variables to something descriptive; remove the redundant magic-number special "
        "cases; remove the unreachable branch. Do NOT change what `calc(a, b, op)` "
        "returns for any input — a visible regression test file is already present at "
        "tests/test_messy.py that pins down the CURRENT behavior exactly, and it must "
        "keep passing, unchanged, after your refactor.\n\n"
        "Run it (run_pytest) after every change and keep revising until every test in "
        "it still passes, then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts: the messy module
    itself (the agent edits this in place — same relpath, same import name `messy`, so
    the regression suite's `from messy import calc` keeps working) plus the visible
    regression test file. Grading still re-injects a pristine canonical copy of the test
    file from private/tests/ regardless (see materialize_task.py / run_grade.sh) — the
    agent's own copy of tests/test_messy.py gets manifest-dropped before grading, same
    as c01's convention."""
    return {
        STARTER_MODULE_PATH: _MESSY_PY_CONTENT,
        VISIBLE_TEST_PATH: _TEST_FILE_CONTENT,
    }


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "messy", "functions": ["calc"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_judge_rubric() -> dict:
    return {
        "criteria": [
            "calc's logic is broken into smaller, well-named helper functions rather "
            "than left as one long nested if/elif chain",
            "variable names are descriptive — no x1/tmp2-style placeholders remain",
            "the repeated magic-number special-casing (the 0/1 edge branches that "
            "duplicate what the general-case arithmetic already computes correctly) is "
            "removed, not just relabeled",
            "the unreachable branch (the final else under the sub op's >/</== chain) is "
            "removed",
            "no new magic numbers, dead code, or needless complexity were introduced by "
            "the refactor itself",
        ],
        "notes": (
            "Behavior preservation is already fully covered by the visible regression "
            "suite (tests/test_messy.py) and does not need re-judging here — score "
            "purely on the clarity and structure of the refactor."
        ),
    }


def get_compiled_plan() -> dict:
    """Single leaf: the refactor is one cohesive unit of work on one small file, so it
    doesn't decompose into independent pieces the way a multi-file task would."""
    return {
        "leaves": [
            {
                "id": "refactor",
                "instruction": (
                    "Refactor messy.py for clarity: extract calc's logic into smaller, "
                    "well-named helper functions; replace unclear variable names (x1, "
                    "tmp2, etc.) with descriptive ones; remove the repeated "
                    "magic-number special-casing (the 0/1 edge branches that just "
                    "duplicate what the general-case arithmetic already computes); "
                    "remove the unreachable dead branch. Do NOT change calc's "
                    "observable behavior for any input. The regression suite at "
                    "tests/test_messy.py pins down the CURRENT behavior exactly — use "
                    "read_file to review messy.py, patch_file/write_file to refactor "
                    "it, then run_pytest on tests/test_messy.py after every change and "
                    "keep revising until it fully passes, then finish."
                ),
                "expect": (
                    "messy.py refactored into smaller/well-named pieces with no dead "
                    "code or magic-number special-casing; tests/test_messy.py still "
                    "fully passes"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm messy.py exists and report the pytest pass/fail summary against "
            "tests/test_messy.py."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["messy.py"]},
    }
