"""
codebench task c15 — soft/hidden, open-ended build (arithmetic expression evaluator).

Unlike the hard tasks (c01/c02/c13), there is no single "correct" implementation here — the
spec deliberately leaves the parsing strategy up to the agent. `get_grading_payload()["tests"]`
is therefore a small SANITY suite only (imports cleanly, a couple of basic arithmetic cases,
confirms *some* exception is raised on bad input) rather than an attempt to pin exact behavior.
Task-specific judging beyond that lives in `get_judge_rubric()`, consumed by the (separately
built) rubric-judge module — NOT parsed here.

Visibility is "hidden": `get_sandbox_fixture()` is empty, so the agent sees only the task
statement and must build calculator.py entirely from the prose spec (no starter test file is
placed in the sandbox for it to iterate against — self-checking is done with its own throwaway
checks, per the compiled plan's leaf instruction below).
"""
from __future__ import annotations

SMOKE_TEST_PATH = "tests/test_calculator_smoke.py"

_TEST_FILE_CONTENT = '''\
import pytest
from calculator import evaluate


def test_basic_addition():
    assert evaluate("2 + 2") == 4


def test_parens_and_precedence():
    assert evaluate("2 * (3 + 4)") == 14


def test_division_by_zero_raises_something():
    # Soft task: we don't pin the exact exception type, only that evaluate() surfaces
    # *some* exception for a division-by-zero rather than crashing uninformatively
    # (e.g. hanging, returning a silently wrong value, or killing the process).
    with pytest.raises(Exception):
        evaluate("2 / 0")
'''

# All three cases are basic sanity/smoke checks of the spec's explicitly-named requirements
# (arithmetic + precedence/parens + division-by-zero handling) — none is a corner case, so all
# three gate the score for this soft task.
KEYSTONE_TEST_IDS = [
    f"{SMOKE_TEST_PATH}::test_basic_addition",
    f"{SMOKE_TEST_PATH}::test_parens_and_precedence",
    f"{SMOKE_TEST_PATH}::test_division_by_zero_raises_something",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c15",
        "title": "build-a-calculator-cli",
        "category": "soft",
    }


def get_task_statement() -> str:
    return (
        "Build a Python module `calculator.py` that defines a function "
        "`evaluate(expression: str) -> float`.\n\n"
        "`evaluate` should parse and evaluate a basic arithmetic expression string. At "
        "minimum it must support the operators `+`, `-`, `*`, `/`, and parentheses for "
        "grouping, with standard operator precedence (multiplication and division bind "
        "tighter than addition and subtraction) and support for nested parentheses.\n\n"
        "Do NOT use Python's built-in `eval()` (or `exec()`) to implement this — write a "
        "real small parser/evaluator instead (for example: tokenize the string, then "
        "evaluate with a recursive-descent or shunting-yard approach). The exact internal "
        "design is up to you; this is intentionally open-ended.\n\n"
        "On invalid input (a malformed expression, unbalanced parentheses, unrecognized "
        "characters) or on division by zero, `evaluate` must raise a sensible exception "
        "rather than crashing uninformatively, hanging, or silently returning a wrong "
        "value. Any exception type is acceptable as long as it's a real, catchable "
        "exception.\n\n"
        "There is no starter file or test file provided — build calculator.py entirely "
        "from this description. Before finishing, sanity-check your own work (e.g. with "
        "run_python, or by writing a couple of your own quick checks and running them "
        "with run_pytest) against a few basic cases: simple addition, an expression with "
        "parentheses and precedence, and a division-by-zero case."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files (no test file, no partial implementation) — the agent
    builds calculator.py from the task statement alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {SMOKE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "calculator", "functions": ["evaluate"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_judge_rubric() -> dict:
    return {
        "criteria": [
            "Supports all four operators (+, -, *, /) with standard operator precedence "
            "(multiplication/division before addition/subtraction).",
            "Correctly handles parentheses, including nested parentheses.",
            "Does not implement evaluation by delegating to Python's eval()/exec() — a "
            "real parser/evaluator was written.",
            "Raises a clear, catchable exception (not a hang, a silent wrong answer, or "
            "an uninformative crash) on malformed input and on division by zero.",
            "Exposes evaluate(expression: str) -> float from calculator.py exactly as "
            "specified, so it's importable and callable without guessing.",
        ],
        "notes": (
            "This is an open-ended build task — there is no single correct implementation, "
            "so judge whether a reasonable calculator was actually built and self-tested, "
            "not whether it matches any particular parsing strategy. The spec explicitly "
            "calls out division-by-zero and malformed-input handling, so weight missing or "
            "crash-prone handling of those meaningfully; an uncaught low-level exception "
            "(e.g. a bare ZeroDivisionError propagating up unguarded) still counts as "
            "'raises a sensible exception' since it is catchable and informative."
        ),
    }


def get_compiled_plan() -> dict:
    """Single leaf: this is a small, open-ended build with no independent sub-parts to
    decompose across. Since the task is hidden (no canonical test in the sandbox), the leaf
    instruction has the model self-check its own work rather than iterate against a fixture."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Build calculator.py implementing evaluate(expression: str) -> float: "
                    "support +, -, *, /, and parentheses (with correct precedence and "
                    "nesting), without using eval()/exec(). Raise a sensible exception on "
                    "invalid input or division by zero. There is no starter test file — "
                    "use write_file to create calculator.py from the task description. "
                    "Then self-check your own work: use run_python (or write a few quick "
                    "checks of your own and run them with run_pytest) to confirm a simple "
                    "addition, an expression with parentheses and precedence, and a "
                    "division-by-zero case all behave as described. If something's wrong, "
                    "use read_file/patch_file to fix calculator.py and re-check. Once "
                    "you're satisfied it behaves correctly, finish."
                ),
                "expect": (
                    "calculator.py written; agent's own self-check runs confirm basic "
                    "arithmetic, precedence/parens, and division-by-zero handling"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm calculator.py exists and report the agent's own self-check results."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["calculator.py"]},
    }
