"""
codebench task c11 — hard/visible, nth-prime lookup.

Ground truth for ``nth_prime`` verified with an independent throwaway reference script using
TWO separately-coded algorithms (naive trial division AND a sieve of Eratosthenes up to a
fixed bound) and confirming they agree, before being embedded here — not hand-counted. Do not
change these expected values without re-running that cross-check.
    n=1  -> 2
    n=2  -> 3
    n=6  -> 13
    n=25 -> 97   (cross-checked: sieve_primes_up_to(600)[24] == 97)
    n<=0 -> ValueError
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_prime_sieve.py"

_TEST_FILE_CONTENT = '''\
import pytest
from prime_sieve import nth_prime


def test_n1():
    assert nth_prime(1) == 2


def test_n2():
    assert nth_prime(2) == 3


def test_n6():
    assert nth_prime(6) == 13


def test_n25():
    assert nth_prime(25) == 97


def test_zero_raises():
    with pytest.raises(ValueError):
        nth_prime(0)


def test_negative_raises():
    with pytest.raises(ValueError):
        nth_prime(-3)
'''

# The three cases that distinguish "actually finds the n-th prime" from "hardcoded the first
# couple of primes" gate the score; the ValueError contract cases test input validation, not
# the prime-finding logic, so they're bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_n2",
    f"{VISIBLE_TEST_PATH}::test_n6",
    f"{VISIBLE_TEST_PATH}::test_n25",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c11",
        "title": "nth-prime-sieve",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `prime_sieve.py` that defines a function "
        "`nth_prime(n: int) -> int` returning the n-th prime number, 1-indexed "
        "(`nth_prime(1) == 2`, `nth_prime(2) == 3`, `nth_prime(6) == 13`).\n\n"
        "It must handle at least up to `n=100` efficiently enough to finish well within a "
        "test timeout. A naive O(n^2)-ish trial-division primality check (try dividing by "
        "every integer up to sqrt(candidate)) is completely fine for this range — you do "
        "not need a true sieve of Eratosthenes or any other optimization.\n\n"
        "If `n` is zero or negative, raise `ValueError`.\n\n"
        "A visible test file is already present at tests/test_prime_sieve.py — run it "
        "(run_pytest) and keep revising prime_sieve.py until every test in it passes, then "
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
        "entrypoint": {"module": "prime_sieve", "functions": ["nth_prime"]},
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
                    "Write prime_sieve.py implementing nth_prime(n): the n-th prime number "
                    "(1-indexed, nth_prime(1)==2). A naive trial-division primality check "
                    "(divide candidates by every integer up to sqrt(candidate)) is fine for "
                    "n up to 100. Raise ValueError for n <= 0. Use write_file to create it. "
                    "Then use run_pytest on tests/test_prime_sieve.py. If any test fails, "
                    "use read_file/patch_file to fix prime_sieve.py and run_pytest again. "
                    "Once every test passes, finish."
                ),
                "expect": "prime_sieve.py written; tests/test_prime_sieve.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm prime_sieve.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["prime_sieve.py"]},
    }
