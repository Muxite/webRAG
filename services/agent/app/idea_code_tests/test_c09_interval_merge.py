"""
codebench task c09 — hard/visible, interval merging.

Ground truth for ``merge_intervals`` verified with an independent throwaway reference script
(sort-then-sweep, run against every case below) before being embedded here — not hand-computed.
Do not change these expected values without re-running that verification.

Merge rule (explicit, since touching endpoints is the one place this spec could be ambiguous):
two intervals [a, b] and [c, d] with a <= c overlap-or-touch, and therefore MUST merge, whenever
``c <= b`` — i.e. [1, 2] and [2, 3] DO merge into [1, 3] because they share the endpoint 2.
    no_overlap:  [[1,2],[4,5],[7,8]]        -> [[1,2],[4,5],[7,8]]   (already sorted, unchanged)
    full_chain:  [[1,4],[2,5],[3,6]]        -> [[1,6]]               (all three chain-merge)
    touching:    [[1,2],[2,3]]              -> [[1,3]]               (c<=b rule: 2<=2 merges)
    unsorted:    [[5,6],[1,3],[8,10]]       -> [[1,3],[5,6],[8,10]]  (sorted by start, no merges)
    single:      [[1,1]]                    -> [[1,1]]
    empty:       []                         -> []
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_interval_merge.py"

_TEST_FILE_CONTENT = '''\
from interval_merge import merge_intervals


def test_no_overlap():
    assert merge_intervals([[1, 2], [4, 5], [7, 8]]) == [[1, 2], [4, 5], [7, 8]]


def test_full_chain_merge():
    assert merge_intervals([[1, 4], [2, 5], [3, 6]]) == [[1, 6]]


def test_touching_endpoints_merge():
    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]


def test_unsorted_input_gets_sorted():
    assert merge_intervals([[5, 6], [1, 3], [8, 10]]) == [[1, 3], [5, 6], [8, 10]]


def test_single_interval():
    assert merge_intervals([[1, 1]]) == [[1, 1]]


def test_empty_list():
    assert merge_intervals([]) == []
'''

# The four cases that actually distinguish "implements the sort+merge sweep" from "got lucky on
# a trivial input" gate the score; single-interval and empty-list are degenerate (any correct
# passthrough handles them trivially) and are bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_no_overlap",
    f"{VISIBLE_TEST_PATH}::test_full_chain_merge",
    f"{VISIBLE_TEST_PATH}::test_touching_endpoints_merge",
    f"{VISIBLE_TEST_PATH}::test_unsorted_input_gets_sorted",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c09",
        "title": "interval-merge",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `interval_merge.py` that defines a function "
        "`merge_intervals(intervals: list[list[int]]) -> list[list[int]]`.\n\n"
        "Each interval is `[start, end]` with `start <= end`, inclusive on both ends. "
        "`merge_intervals` must merge all overlapping intervals and return the merged "
        "list sorted by start.\n\n"
        "Two intervals `[a, b]` and `[c, d]` (with `a <= c`) overlap-or-touch, and "
        "therefore MUST be merged into one, whenever `c <= b`. This means touching "
        "endpoints DO count as overlapping: `[1, 2]` and `[2, 3]` merge into `[1, 3]` "
        "because they share the boundary point 2. Only strictly separated intervals "
        "(`c > b`) stay distinct.\n\n"
        "The input list may be given in any order and may contain zero, one, or many "
        "intervals; the output must always be sorted by start.\n\n"
        "A visible test file is already present at tests/test_interval_merge.py — run it "
        "(run_pytest) and keep revising interval_merge.py until every test in it passes, "
        "then finish."
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
        "entrypoint": {"module": "interval_merge", "functions": ["merge_intervals"]},
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
                    "Write interval_merge.py implementing merge_intervals(intervals): sort "
                    "intervals by start, then sweep left to right merging any interval whose "
                    "start is <= the running merged interval's end (touching endpoints DO "
                    "merge, e.g. [1,2] and [2,3] -> [1,3]); return the merged list sorted by "
                    "start. Use write_file to create it. Then use run_pytest on "
                    "tests/test_interval_merge.py. If any test fails, use read_file/patch_file "
                    "to fix interval_merge.py and run_pytest again. Once every test passes, "
                    "finish."
                ),
                "expect": "interval_merge.py written; tests/test_interval_merge.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm interval_merge.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["interval_merge.py"]},
    }
