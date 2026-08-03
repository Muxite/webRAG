"""
codebench task c12 — hard/hidden, flat dict key-diff tool.

Ground truth for ``diff_keys`` verified with an independent throwaway reference script
before being embedded here — not hand-derived. Do not change these expected values without
re-running that script.
    old={"a":1,"b":2,"c":3}, new={"a":1,"b":20,"d":4}
        -> {"added": ["d"], "removed": ["c"], "changed": ["b"]}   (worked example; "a" is
           equal in both so it is omitted entirely)
    pure addition:  old={"a":1}, new={"a":1,"b":2}
        -> {"added": ["b"], "removed": [], "changed": []}
    pure removal:   old={"a":1,"b":2}, new={"a":1}
        -> {"added": [], "removed": ["b"], "changed": []}
    pure change:    old={"a":1}, new={"a":2}
        -> {"added": [], "removed": [], "changed": ["a"]}
    no difference:  old=new={"a":1,"b":2}
        -> {"added": [], "removed": [], "changed": []}
    list value diff: old={"a":[1,2]}, new={"a":[1,3]} -> changed=["a"]
    int/float equal: old={"a":1}, new={"a":1.0} -> all empty (Python's 1 == 1.0, so `!=`
        treats them as equal and the key is omitted from every list)
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_json_diff.py"

_TEST_FILE_CONTENT = '''\
from json_diff import diff_keys


def test_worked_example_mix():
    old = {"a": 1, "b": 2, "c": 3}
    new = {"a": 1, "b": 20, "d": 4}
    assert diff_keys(old, new) == {"added": ["d"], "removed": ["c"], "changed": ["b"]}


def test_pure_addition():
    assert diff_keys({"a": 1}, {"a": 1, "b": 2}) == {
        "added": ["b"], "removed": [], "changed": [],
    }


def test_pure_removal():
    assert diff_keys({"a": 1, "b": 2}, {"a": 1}) == {
        "added": [], "removed": ["b"], "changed": [],
    }


def test_pure_change():
    assert diff_keys({"a": 1}, {"a": 2}) == {
        "added": [], "removed": [], "changed": ["a"],
    }


def test_no_difference():
    same = {"a": 1, "b": 2}
    assert diff_keys(same, dict(same)) == {"added": [], "removed": [], "changed": []}


def test_list_value_change_detected():
    assert diff_keys({"a": [1, 2]}, {"a": [1, 3]}) == {
        "added": [], "removed": [], "changed": ["a"],
    }


def test_int_and_float_equal_value_omitted():
    assert diff_keys({"a": 1}, {"a": 1.0}) == {"added": [], "removed": [], "changed": []}
'''

# Everything except the int/float edge case gates the score: added/removed/changed detection
# and the "equal values are omitted, not listed as changed" contract are the actual task.
# The int-vs-float case pins a Python-semantics edge case (1 == 1.0) that's genuinely easy to
# get "wrong" in a defensible way (e.g. type(old[k]) != type(new[k])), so it's bonus credit,
# not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_worked_example_mix",
    f"{VISIBLE_TEST_PATH}::test_pure_addition",
    f"{VISIBLE_TEST_PATH}::test_pure_removal",
    f"{VISIBLE_TEST_PATH}::test_pure_change",
    f"{VISIBLE_TEST_PATH}::test_no_difference",
    f"{VISIBLE_TEST_PATH}::test_list_value_change_detected",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c12",
        "title": "json-schema-diff-tool",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `json_diff.py` that defines a function "
        "`diff_keys(old: dict, new: dict) -> dict`.\n\n"
        "Both `old` and `new` are FLAT dicts — single level, no nested dicts or lists of "
        "dicts to recurse into, though values themselves may be any JSON-like type "
        "(str, int, float, bool, None, list). Compare them key by key and return a dict "
        "with exactly three keys:\n"
        "- `\"added\"`: a sorted list of keys present in `new` but not in `old`.\n"
        "- `\"removed\"`: a sorted list of keys present in `old` but not in `new`.\n"
        "- `\"changed\"`: a sorted list of keys present in BOTH `old` and `new` whose "
        "values differ, i.e. `old[k] != new[k]`.\n\n"
        "A key whose value is equal in both dicts must NOT appear in any of the three "
        "lists — it is simply omitted entirely.\n\n"
        "Worked example: for `old = {\"a\": 1, \"b\": 2, \"c\": 3}` and "
        "`new = {\"a\": 1, \"b\": 20, \"d\": 4}`, the function must return EXACTLY "
        "`{\"added\": [\"d\"], \"removed\": [\"c\"], \"changed\": [\"b\"]}` — key `\"a\"` "
        "is equal in both (1 == 1) so it does not appear anywhere in the result.\n\n"
        "No test file is visible for this task — write json_diff.py from this spec, then "
        "use run_pytest on any test file you create yourself to sanity-check your own "
        "understanding before finishing (the canonical grading tests are hidden and will "
        "be run separately)."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter test file is exposed to the agent. It works purely from
    get_task_statement()'s spec text; canonical tests only ever live in the grading
    payload / private/tests/ (see materialize_task.py's module docstring)."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "json_diff", "functions": ["diff_keys"]},
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
                    "Write json_diff.py implementing diff_keys(old, new) for FLAT dicts: "
                    "added = sorted keys in new not in old; removed = sorted keys in old not "
                    "in new; changed = sorted keys present in both where old[k] != new[k]; "
                    "keys with equal values are omitted from all three lists. Return "
                    "{'added': [...], 'removed': [...], 'changed': [...]}. Match the worked "
                    "example in the task statement exactly. Use write_file to create it. "
                    "Write your own quick test file and use run_pytest to sanity-check the "
                    "worked example before finishing — the real grading tests are hidden."
                ),
                "expect": "json_diff.py written implementing diff_keys per the exact spec",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm json_diff.py exists and defines diff_keys.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["json_diff.py"]},
    }
