"""
codebench task c01 — hard/visible, sequence computation.

First task in the coding-agent benchmark (codebench/), proving out the
task-module contract materialize_task.py consumes (see its module docstring). Deliberately
the simplest possible hard task: one pure function, no I/O, no multi-file structure — the
point of c01 is to validate the harness plumbing end-to-end, not to be discriminating.

Ground truth for ``nth_even_fib_sum`` (0-indexed, F0=0, F1=1, sum of EVEN-valued terms among
F0..Fn inclusive) verified by hand here — do not change these without re-deriving:
    F: 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610
    n=0 -> 0            (F0=0)
    n=3 -> 2            (0+2)
    n=6 -> 10           (0+2+8)
    n=9 -> 44           (0+2+8+34)
    n=12 -> 188         (0+2+8+34+144)
    n=15 -> 798         (0+2+8+34+144+610)
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_fib_mod.py"

_TEST_FILE_CONTENT = '''\
import pytest
from fib_mod import nth_even_fib_sum


def test_n0():
    assert nth_even_fib_sum(0) == 0


def test_n3():
    assert nth_even_fib_sum(3) == 2


def test_n6():
    assert nth_even_fib_sum(6) == 10


def test_n9():
    assert nth_even_fib_sum(9) == 44


def test_n15():
    assert nth_even_fib_sum(15) == 798


def test_negative_raises():
    with pytest.raises(ValueError):
        nth_even_fib_sum(-1)
'''

# The four cases that actually distinguish "computes the right sum" from "got lucky on a
# trivial n" gate the score; n0 (degenerate) and the negative-input contract are bonus
# credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_n3",
    f"{VISIBLE_TEST_PATH}::test_n6",
    f"{VISIBLE_TEST_PATH}::test_n9",
    f"{VISIBLE_TEST_PATH}::test_n15",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c01",
        "title": "nth-even-fib-sum",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `fib_mod.py` that defines a function "
        "`nth_even_fib_sum(n: int) -> int`.\n\n"
        "The Fibonacci sequence is 0-indexed with F(0) = 0 and F(1) = 1 "
        "(so it begins 0, 1, 1, 2, 3, 5, 8, 13, ...).\n\n"
        "`nth_even_fib_sum(n)` must return the sum of every EVEN-valued term among "
        "F(0) through F(n) inclusive.\n\n"
        "If `n` is negative, raise `ValueError`.\n\n"
        "A visible test file is already present at tests/test_fib_mod.py — run it "
        "(run_pytest) and keep revising fib_mod.py until every test in it passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts. For a visible
    task this includes the real test file (grading still re-injects a pristine canonical
    copy from private/tests/ regardless — see materialize_task.py / run_grade.sh)."""
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "fib_mod", "functions": ["nth_even_fib_sum"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Hand-authored offline plan (mirrors idea_tests/'s get_compiled_plan() convention) —
    a single leaf is enough for a one-function task; multi-leaf plans are for tasks that
    genuinely decompose into independent pieces."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write fib_mod.py implementing nth_even_fib_sum(n): the sum of "
                    "even-valued Fibonacci terms F(0)..F(n) inclusive (0-indexed, F0=0, "
                    "F1=1); raise ValueError for negative n. Use write_file to create it. "
                    "Then use run_pytest on tests/test_fib_mod.py. If any test fails, use "
                    "read_file/patch_file to fix fib_mod.py and run_pytest again. Once "
                    "every test passes, finish."
                ),
                "expect": "fib_mod.py written; tests/test_fib_mod.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm fib_mod.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["fib_mod.py"]},
    }
