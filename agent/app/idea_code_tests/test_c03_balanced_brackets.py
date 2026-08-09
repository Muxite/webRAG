"""
codebench task c03 — hard/hidden, string validation with a stack.

Ground truth for ``is_balanced`` (True iff every (), [], {} in the string is validly
matched and nested; all other characters are ignored/pass through) was computed by two
independently-written reference implementations (a groupby-free explicit stack with a
dict lookup on the closer, vs. an index-based loop mapping openers to their expected
closer) that were run and cross-checked to agree exactly before any value below was
written down — see idea_code_test_c03_test.py for the reimplementation that re-verifies
this at test time. Do not hand-edit these literals without re-deriving.

    is_balanced("")                    -> True   (degenerate: nothing to unbalance)
    is_balanced("()[]{}")              -> True   (all matched, flat)
    is_balanced("())")                 -> False  (unmatched close)
    is_balanced("((")                  -> False  (unmatched open)
    is_balanced("([)]")                -> False  (wrong nesting order)
    is_balanced("a(b[c]d)e")           -> True   (brackets mixed with other text)
    is_balanced("[{(})]")              -> False  (brackets mixed with other text, wrong order)
    is_balanced("(((((((())))))))")    -> True   (deeply nested valid)
"""
from __future__ import annotations

CANONICAL_TEST_PATH = "tests/test_brackets.py"

_TEST_FILE_CONTENT = '''\
from brackets import is_balanced


def test_empty_string():
    assert is_balanced("") is True


def test_all_matched():
    assert is_balanced("()[]{}") is True


def test_unmatched_close():
    assert is_balanced("())") is False


def test_unmatched_open():
    assert is_balanced("((") is False


def test_wrong_nesting_order():
    assert is_balanced("([)]") is False


def test_mixed_with_text_valid():
    assert is_balanced("a(b[c]d)e") is True


def test_mixed_with_text_invalid():
    assert is_balanced("[{(})]") is False


def test_deeply_nested_valid():
    assert is_balanced("(((((((())))))))") is True
'''

# The unmatched-close/open, wrong-nesting-order, and mixed-with-text cases in both
# directions are what distinguish "implemented a real matching stack" from "counted
# brackets" or "only handles one bracket type"; the empty-string case is degenerate
# (trivially True regardless of implementation) and is bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{CANONICAL_TEST_PATH}::test_all_matched",
    f"{CANONICAL_TEST_PATH}::test_unmatched_close",
    f"{CANONICAL_TEST_PATH}::test_unmatched_open",
    f"{CANONICAL_TEST_PATH}::test_wrong_nesting_order",
    f"{CANONICAL_TEST_PATH}::test_mixed_with_text_valid",
    f"{CANONICAL_TEST_PATH}::test_mixed_with_text_invalid",
    f"{CANONICAL_TEST_PATH}::test_deeply_nested_valid",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c03",
        "title": "balanced-brackets-validator",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `brackets.py` that defines a function "
        "`is_balanced(s: str) -> bool`.\n\n"
        "`is_balanced(s)` must return True if and only if every bracket in `s` is "
        "validly matched and correctly nested, where the three bracket pairs are "
        "`()`, `[]`, and `{}`. Every character in `s` that is NOT one of these six "
        "bracket characters is an ordinary, non-bracket character: it is completely "
        "ignored by the matching logic — it neither opens nor closes anything, it is "
        "simply skipped over as if it weren't there, and its presence never affects "
        "the result.\n\n"
        "A string is balanced when: (1) every closing bracket has a corresponding "
        "previously-unmatched opening bracket of the SAME type immediately available "
        "to match it (most-recently-opened-first, i.e. proper nesting), and (2) no "
        "opening bracket is left unmatched at the end.\n\n"
        "Worked examples:\n"
        "  - `is_balanced(\"a(b)c\")` -> `True` (the letters are ignored; `(` and `)` "
        "match).\n"
        "  - `is_balanced(\"([)]\")` -> `False` (the `(` and `[` open in that order, "
        "but then `)` tries to close before the more-recently-opened `[` has been "
        "closed — wrong nesting order, even though each bracket type appears an equal "
        "number of times).\n"
        "  - `is_balanced(\"{[()()]}\")` -> `True` (properly nested: two `()` pairs "
        "inside one `[]` pair, all inside one `{}` pair).\n\n"
        "No test file is visible for this task. Write brackets.py implementing "
        "is_balanced exactly as specified above, then use run_pytest to check your own "
        "reasoning about correctness before finishing (there is no pre-existing test "
        "file at tests/ — the grader supplies its own tests after you finish)."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {CANONICAL_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "brackets", "functions": ["is_balanced"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write brackets.py implementing is_balanced(s: str) -> bool: return "
                    "True iff every (), [], {} in s is validly matched and correctly "
                    "nested (most-recently-opened bracket must close first); every "
                    "non-bracket character is ignored and never affects the result. Use "
                    "a stack: push on an opener, on a closer pop and check it matches "
                    "the expected type (fail if the stack is empty or the type doesn't "
                    "match); the string is balanced iff the stack is empty at the end. "
                    "Use write_file to create it. There is no pre-existing test file — "
                    "write a few of your own sanity-check calls (e.g. via run_pytest "
                    "against a small scratch test you author, or by reasoning through "
                    "the worked examples in the task statement) to validate your logic, "
                    "then finish."
                ),
                "expect": "brackets.py written implementing is_balanced per the spec",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm brackets.py exists and defines is_balanced.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["brackets.py"]},
    }
