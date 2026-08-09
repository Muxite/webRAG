"""
codebench task c07 — hard/visible, "generate N files matching a template mold" without real
file I/O — the function returns the dict-of-dicts directly, so grading never touches a
filesystem, keeping this task's grading as simple as c01's.

Ground truth for ``generate_configs`` verified independently with a throwaway reference script
(NOT hand-computed) before being embedded here — see the offline test file's own
reimplementation for the second, independent check. Verified behavior:
    generate_configs({"region": "us-east-1", "replicas": 3}, ["web-01", "web-02", "worker-01"])
    -> {
         "web-01":    {"region": "us-east-1", "replicas": 3, "instance_name": "web-01"},
         "web-02":    {"region": "us-east-1", "replicas": 3, "instance_name": "web-02"},
         "worker-01": {"region": "us-east-1", "replicas": 3, "instance_name": "worker-01"},
       }
    generate_configs(base, []) -> {}
    base itself is never mutated by the call.
    Two returned per-name dicts are independent objects: mutating one's keys leaves the other
    (and base) untouched — a shared-reference implementation (e.g. building one dict and reusing
    it, or doing `cfg = base; cfg["instance_name"] = name`) is exactly the bug this catches.
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_fleet_gen.py"

_TEST_FILE_CONTENT = '''\
from fleet_gen import generate_configs

BASE = {"region": "us-east-1", "replicas": 3}
NAMES = ["web-01", "web-02", "worker-01"]


def test_returns_dict_keyed_by_names():
    result = generate_configs(BASE, NAMES)
    assert set(result.keys()) == set(NAMES)


def test_each_config_has_base_keys_plus_instance_name():
    result = generate_configs(BASE, NAMES)
    for name in NAMES:
        cfg = result[name]
        assert cfg["region"] == "us-east-1"
        assert cfg["replicas"] == 3
        assert cfg["instance_name"] == name


def test_configs_are_independent_copies_not_shared_references():
    result = generate_configs(BASE, NAMES)
    result["web-01"]["region"] = "eu-west-1"
    assert result["web-02"]["region"] == "us-east-1"
    assert result["worker-01"]["region"] == "us-east-1"


def test_base_dict_is_not_mutated():
    base_copy = dict(BASE)
    generate_configs(BASE, NAMES)
    assert BASE == base_copy


def test_empty_names_returns_empty_dict():
    assert generate_configs(BASE, []) == {}
'''

# The shared-reference bug is the whole point of this task, so it (plus the two basic shape
# checks it depends on making sense of) gates the score. Not-mutating-base and the empty-list
# edge case are real contract requirements but bonus credit only, same convention as c01.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_returns_dict_keyed_by_names",
    f"{VISIBLE_TEST_PATH}::test_each_config_has_base_keys_plus_instance_name",
    f"{VISIBLE_TEST_PATH}::test_configs_are_independent_copies_not_shared_references",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c07",
        "title": "config-yaml-fleet-gen",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `fleet_gen.py` that defines a function "
        "`generate_configs(base: dict, names: list[str]) -> dict`.\n\n"
        "This models generating N per-instance config files from one template, without doing "
        "any real file I/O: `generate_configs` returns a dict-of-dicts instead of writing "
        "files.\n\n"
        "For each name in `names`, the returned dict must have an entry `{name: config}` where "
        "`config` is a COPY of `base` with one extra key added: `instance_name` set to that "
        "name. So the return value has exactly `len(names)` entries, one per name in `names`.\n\n"
        "Requirements:\n"
        "- Every returned per-name config must contain all of `base`'s original keys/values, "
        "PLUS `instance_name`.\n"
        "- Each per-name config must be an independent copy — mutating one entry's dict must "
        "NOT affect any other entry's dict, and must not affect `base` itself.\n"
        "- `base` itself must never be mutated by the call.\n"
        "- If `names` is empty, return an empty dict.\n\n"
        "A visible test file is already present at tests/test_fleet_gen.py — run it (run_pytest) "
        "and keep revising fleet_gen.py until every test in it passes, then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "fleet_gen", "functions": ["generate_configs"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one pure function, no I/O, no multi-file structure — same rationale as c01."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write fleet_gen.py implementing generate_configs(base: dict, "
                    "names: list[str]) -> dict: for each name in names, the result maps name to "
                    "a COPY of base with an added 'instance_name' key set to that name. Each "
                    "returned per-name dict must be an independent object (mutating one must not "
                    "affect another or affect base); base itself must never be mutated; an empty "
                    "names list returns an empty dict. Use write_file to create it. Then use "
                    "run_pytest on tests/test_fleet_gen.py. If any test fails, use "
                    "read_file/patch_file to fix fleet_gen.py and run_pytest again. Once every "
                    "test passes, finish."
                ),
                "expect": "fleet_gen.py written; tests/test_fleet_gen.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm fleet_gen.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["fleet_gen.py"]},
    }
