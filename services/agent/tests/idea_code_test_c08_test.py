"""
Adversarial offline checks for codebench task c08 (lru-cache-impl) — no Docker, no LLM.

Mirrors idea_code_test_c01_test.py's structure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

from agent.app.idea_code_tests import test_c08_lru_cache_impl as c08


class _IndependentLRUCache:
    """Reimplemented independently of the task module's own prose spec (via
    collections.OrderedDict rather than any hand-rolled linked-list/dict pairing), to catch a
    mistake in the embedded canonical test file's own expectations."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: "OrderedDict[object, object]" = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return -1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)


def test_ground_truth_capacity_one_eviction():
    c = _IndependentLRUCache(1)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == -1
    assert c.get(2) == "b"


def test_ground_truth_get_counts_as_recency():
    c = _IndependentLRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == "a"
    c.put(3, "c")
    assert c.get(2) == -1
    assert c.get(1) == "a"
    assert c.get(3) == "c"


def test_ground_truth_overwrite_does_not_evict():
    c = _IndependentLRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    c.put(1, "z")
    assert c.get(1) == "z"
    assert c.get(2) == "b"


def test_ground_truth_overwrite_counts_as_recency():
    c = _IndependentLRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    c.put(1, "z")
    c.put(3, "c")
    assert c.get(2) == -1
    assert c.get(1) == "z"
    assert c.get(3) == "c"


def test_ground_truth_missing_key_returns_negative_one():
    c = _IndependentLRUCache(2)
    assert c.get(999) == -1


def test_ground_truth_matches_worked_example_in_task_statement():
    statement = c08.get_task_statement()
    assert "cache.get(2)       # -> -1" in statement
    c = _IndependentLRUCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == "a"
    c.put(3, "c")
    assert c.get(2) == -1
    assert c.get(3) == "c"
    assert c.get(1) == "a"


def test_get_only_recency_bug_is_actually_caught():
    """Prove test_get_counts_as_recency_use is discriminating: a plausible buggy implementation
    that only reorders on put() (ignoring get()) must produce the OPPOSITE eviction decision."""

    class PutOnlyRecencyCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self._data = OrderedDict()

        def get(self, key):
            return self._data.get(key, -1)  # BUG: never moves key to the end

        def put(self, key, value):
            self._data[key] = value
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)

    c = PutOnlyRecencyCache(2)
    c.put(1, "a")
    c.put(2, "b")
    assert c.get(1) == "a"   # ignored for recency by the buggy impl
    c.put(3, "c")             # buggy impl evicts key 1 (oldest by put-order), not key 2
    assert c.get(1) == -1     # WRONG per spec (correct impl keeps 1, evicts 2)
    assert c.get(2) == "b"    # WRONG per spec


def test_embedded_test_file_asserts_match_ground_truth():
    content = c08.get_grading_payload()["tests"][c08._TEST_FILE_PATH]
    pairs = re.findall(r'cache\.get\((\d+)\) == ("[a-z]"|-1)', content)
    assert pairs, "expected at least one literal get() assertion in the embedded test file"


def test_keystone_ids_reference_real_test_functions():
    content = c08.get_grading_payload()["tests"][c08._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c08.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c08._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_spoiled_worked_example():
    # The worked example is handed to the agent verbatim in the prompt, so replaying it back
    # isn't a discriminating signal -- excluded from the gate.
    assert f"{c08._TEST_FILE_PATH}::test_worked_example_from_spec" not in c08.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c08.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c08.get_sandbox_fixture() == {}


def test_task_statement_contains_a_worked_example_as_an_anchor():
    statement = c08.get_task_statement()
    assert "cache = LRUCache(2)" in statement
    assert "cache.put(1, 'a')" in statement
    assert "cache.get(1)" in statement


def test_grading_payload_shape():
    payload = c08.get_grading_payload()
    assert payload["tests"] == {c08._TEST_FILE_PATH: c08._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {"module": "lru_cache", "classes": ["LRUCache"]}
    assert payload["keystone_test_ids"] == c08.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c08.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "lru_cache.py" in leaf["instruction"]
    assert "LRUCache" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["lru_cache.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c08", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c08"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c08.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c08.get_compiled_plan()

    assert (private / c08._TEST_FILE_PATH).read_text() == c08._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c08._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c08.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
