"""
codebench task c08 — hard/hidden, precise eviction-order semantics.

Ground truth for ``LRUCache`` verified independently with a throwaway ``collections.OrderedDict``
reference implementation (NOT hand-computed) before being embedded here — see the offline test
file's own reimplementation for the second, independent check. Verified traces:
    capacity=1: put(1,'a'); put(2,'b') -> get(1) == -1, get(2) == 'b'
    capacity=2: put(1,'a'); put(2,'b'); get(1) -> 'a' (touches 1, so 2 becomes LRU);
                put(3,'c') evicts 2 (NOT 1) -> get(2) == -1, get(1) == 'a', get(3) == 'c'
                (this is the case that would fail if only put() updated recency)
    capacity=2: put(1,'a'); put(2,'b'); put(1,'z') (overwrite) -> get(1) == 'z', get(2) == 'b'
                (overwrite updates the value and must NOT evict anything)
    capacity=2: put(1,'a'); put(2,'b'); put(1,'z') (overwrite also touches 1's recency);
                put(3,'c') evicts 2 (NOT 1) -> get(2) == -1, get(1) == 'z', get(3) == 'c'
    get() on a key that was never put -> -1
    Worked example (also given to the agent verbatim in the task statement, since this is a
    hidden task with no visible test to anchor against):
        cache = LRUCache(2)
        cache.put(1, 'a'); cache.put(2, 'b')
        cache.get(1)      -> 'a'   (touches 1; 2 is now the least-recently-used entry)
        cache.put(3, 'c')  -> evicts key 2 (capacity exceeded, 2 is LRU)
        cache.get(2)      -> -1    (evicted)
        cache.get(3)      -> 'c'
        cache.get(1)      -> 'a'
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_lru_cache.py"

_TEST_FILE_CONTENT = '''\
from lru_cache import LRUCache


def test_capacity_one_eviction():
    cache = LRUCache(1)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == -1
    assert cache.get(2) == "b"


def test_get_counts_as_recency_use():
    # This would fail if only put() updated recency: get(1) below must make key 2 the LRU
    # entry, so the next put() evicts 2, not 1.
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    cache.put(3, "c")
    assert cache.get(2) == -1
    assert cache.get(1) == "a"
    assert cache.get(3) == "c"


def test_overwrite_existing_key_does_not_evict():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(1, "z")
    assert cache.get(1) == "z"
    assert cache.get(2) == "b"


def test_overwrite_counts_as_recency_use():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(1, "z")   # overwrite touches key 1's recency -> key 2 becomes the LRU entry
    cache.put(3, "c")    # capacity exceeded -> evicts 2, not 1
    assert cache.get(2) == -1
    assert cache.get(1) == "z"
    assert cache.get(3) == "c"


def test_missing_key_returns_negative_one():
    cache = LRUCache(2)
    assert cache.get(999) == -1


def test_worked_example_from_spec():
    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(1) == "a"
    cache.put(3, "c")
    assert cache.get(2) == -1
    assert cache.get(3) == "c"
    assert cache.get(1) == "a"
'''

# The worked example is handed to the agent verbatim in the task statement (necessary anchor
# for a hidden task with no visible test), so a model that merely echoes it back isn't
# demonstrating real understanding — excluded from the gate, same convention c01 uses for its
# degenerate n=0 case. The four behaviors that actually pin down LRU semantics precisely gate
# the score.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_capacity_one_eviction",
    f"{_TEST_FILE_PATH}::test_get_counts_as_recency_use",
    f"{_TEST_FILE_PATH}::test_overwrite_existing_key_does_not_evict",
    f"{_TEST_FILE_PATH}::test_overwrite_counts_as_recency_use",
    f"{_TEST_FILE_PATH}::test_missing_key_returns_negative_one",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c08",
        "title": "lru-cache-impl",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `lru_cache.py` that defines a class `LRUCache` with:\n"
        "- `__init__(self, capacity: int)` — creates a cache holding at most `capacity` "
        "key-value pairs.\n"
        "- `get(self, key)` — return the value for `key` if present, else return `-1`.\n"
        "- `put(self, key, value)` — insert or update `key`'s value. If inserting a NEW key "
        "would push the cache over `capacity`, first evict the LEAST RECENTLY USED entry.\n\n"
        "Recency rule: BOTH `get` and `put` count as \"using\" a key — each call to either "
        "method makes that key the most-recently-used one, moving whatever was previously the "
        "least-recently-used entry one step closer to eviction. Overwriting an existing key's "
        "value via `put` updates the value and counts as using it, but never evicts anything by "
        "itself (the number of distinct keys hasn't grown).\n\n"
        "Worked example:\n"
        "```\n"
        "cache = LRUCache(2)\n"
        "cache.put(1, 'a')\n"
        "cache.put(2, 'b')\n"
        "cache.get(1)       # -> 'a'   (touches 1; key 2 is now the least-recently-used entry)\n"
        "cache.put(3, 'c')  # capacity exceeded -> evicts key 2 (the LRU entry)\n"
        "cache.get(2)       # -> -1    (evicted)\n"
        "cache.get(3)       # -> 'c'\n"
        "cache.get(1)       # -> 'a'\n"
        "```\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "implementation yourself — reproduce the worked example above and confirm you get "
        "exactly those results, and try a capacity-1 cache and an overwrite of an existing key — "
        "before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "lru_cache", "classes": ["LRUCache"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one class, no I/O, no multi-file structure — same rationale as c01."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write lru_cache.py implementing class LRUCache with __init__(self, "
                    "capacity: int), get(self, key) (return value or -1 if missing), and "
                    "put(self, key, value) (insert/update; if a NEW key would exceed capacity, "
                    "evict the least-recently-used entry first). BOTH get and put count as "
                    "'using' a key for recency purposes - an implementation that only reorders "
                    "on put is WRONG. Overwriting an existing key's value via put must update "
                    "the value, count as a use, and never evict anything by itself. Worked "
                    "example to match exactly: capacity=2; put(1,'a'); put(2,'b'); get(1) -> "
                    "'a'; put(3,'c') evicts key 2 (not 1, because get(1) touched it more "
                    "recently); then get(2) -> -1, get(3) -> 'c', get(1) -> 'a'. Use write_file "
                    "to create lru_cache.py, then use run_python to reproduce that worked "
                    "example plus a capacity-1 case and an overwrite case, and confirm the "
                    "results match. Fix issues with patch_file/run_python until confident, then "
                    "finish."
                ),
                "expect": "lru_cache.py written, defining LRUCache with correct get/put "
                          "eviction-order semantics, sanity-checked with run_python against the "
                          "worked example",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm lru_cache.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["lru_cache.py"]},
    }
