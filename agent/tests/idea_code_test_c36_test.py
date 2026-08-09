"""
Adversarial offline checks for codebench task c36 (trie-prefix-tree) — no Docker, no LLM.

Mirrors idea_code_test_c08_test.py's structure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c36_trie_prefix_tree as c36


class _IndependentTrie:
    """Reimplemented independently of the task module's own node/children-tree prose spec, via
    a flat Python ``set`` of complete words plus a linear prefix scan, rather than any nested
    node structure — to catch a mistake in the embedded canonical test file's own expectations
    without sharing a bug-class with a real trie implementation."""

    def __init__(self):
        self._words: set[str] = set()

    def insert(self, word: str) -> None:
        self._words.add(word)

    def search(self, word: str) -> bool:
        return word in self._words

    def starts_with(self, prefix: str) -> bool:
        return any(w.startswith(prefix) for w in self._words)

    def delete(self, word: str) -> bool:
        if word not in self._words:
            return False
        self._words.discard(word)
        return True


def _build_car_family():
    t = _IndependentTrie()
    for w in ["cat", "car", "card", "care", "dog"]:
        t.insert(w)
    return t


def test_ground_truth_basic_insert_and_search():
    t = _build_car_family()
    assert t.search("cat") is True
    assert t.search("ca") is False


def test_ground_truth_starts_with():
    t = _build_car_family()
    assert t.starts_with("ca") is True
    assert t.starts_with("do") is True
    assert t.starts_with("z") is False


def test_ground_truth_delete_leaf_word_does_not_affect_siblings():
    t = _build_car_family()
    assert t.delete("car") is True
    assert t.search("car") is False
    assert t.starts_with("car") is True
    assert t.search("card") is True
    assert t.search("care") is True


def test_ground_truth_delete_word_that_is_prefix_of_another():
    t = _IndependentTrie()
    t.insert("a")
    t.insert("ab")
    assert t.delete("a") is True
    assert t.search("a") is False
    assert t.search("ab") is True
    assert t.starts_with("a") is True


def test_ground_truth_delete_trims_dead_branch():
    t = _IndependentTrie()
    t.insert("xyz")
    assert t.delete("xyz") is True
    assert t.search("xyz") is False
    assert t.starts_with("xy") is False
    assert t.starts_with("x") is False


def test_ground_truth_delete_missing_word_is_a_noop():
    t = _IndependentTrie()
    t.insert("cat")
    t.insert("dog")
    assert t.delete("nonexistent") is False
    assert t.search("cat") is True
    assert t.search("dog") is True


def test_ground_truth_full_cascading_delete_sequence():
    t = _build_car_family()
    assert t.delete("car") is True
    assert t.delete("card") is True
    assert t.search("card") is False
    assert t.starts_with("card") is False
    assert t.starts_with("car") is True
    assert t.delete("care") is True
    assert t.starts_with("car") is False
    assert t.delete("car") is False
    assert t.search("dog") is True


def test_ground_truth_delete_then_reinsert():
    t = _IndependentTrie()
    t.insert("hi")
    assert t.delete("hi") is True
    assert t.search("hi") is False
    t.insert("hi")
    assert t.search("hi") is True


def test_set_backed_delete_bug_is_actually_caught():
    """Prove the delete-prefix-safety keystone is discriminating: a plausible buggy
    implementation that deletes every word sharing a prefix (e.g. wipes the whole node
    subtree instead of just clearing the is-word flag) produces the OPPOSITE result."""

    class WipeSubtreeBugTrie:
        """Simulates a common trie delete bug: treats delete(word) as delete-everything-under-
        this-branch instead of delete-just-this-word, via a naive set-based stand-in."""

        def __init__(self):
            self._words: set[str] = set()

        def insert(self, word):
            self._words.add(word)

        def search(self, word):
            return word in self._words

        def starts_with(self, prefix):
            return any(w.startswith(prefix) for w in self._words)

        def delete(self, word):
            if word not in self._words:
                return False
            # BUG: removes every word that starts with `word`, not just `word` itself.
            self._words = {w for w in self._words if not w.startswith(word)}
            return True

    t = WipeSubtreeBugTrie()
    for w in ["car", "card", "care"]:
        t.insert(w)
    t.delete("car")
    assert t.search("card") is False  # WRONG per spec (correct impl keeps "card")
    assert t.search("care") is False  # WRONG per spec (correct impl keeps "care")


def test_embedded_test_file_covers_both_directions_of_the_prefix_relationship():
    content = c36.get_grading_payload()["tests"][c36._TEST_FILE_PATH]
    assert '"a"' in content and '"ab"' in content
    assert "test_delete_word_that_is_a_prefix_of_another_word_keeps_the_longer_word" in content
    assert "test_delete_leaf_word_does_not_affect_sibling_words_sharing_prefix" in content


def test_keystone_ids_reference_real_test_functions():
    content = c36.get_grading_payload()["tests"][c36._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c36.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c36._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_basic_and_bonus_cases():
    non_keystone = {
        "test_basic_insert_and_search",
        "test_starts_with_true_and_false",
        "test_delete_missing_word_returns_false_and_changes_nothing",
        "test_delete_then_reinsert_works",
    }
    for name in non_keystone:
        assert f"{c36._TEST_FILE_PATH}::{name}" not in c36.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c36.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c36.get_sandbox_fixture() == {}


def test_task_statement_contains_a_worked_example_as_an_anchor():
    statement = c36.get_task_statement()
    assert "t = Trie()" in statement
    assert 't.delete("car")' in statement
    assert 't.starts_with("car")' in statement


def test_grading_payload_shape():
    payload = c36.get_grading_payload()
    assert payload["tests"] == {c36._TEST_FILE_PATH: c36._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {"module": "trie", "classes": ["Trie"]}
    assert payload["keystone_test_ids"] == c36.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c36.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "trie.py" in leaf["instruction"]
    assert "Trie" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["trie.py"]}
    json.dumps(plan)


def test_compiled_plan_does_not_leak_hidden_keystone_values():
    """The compiled plan is mounted read-only into the agent's sandbox (public/plan.json), so
    it must not state any literal keystone-test assertion that isn't already public via the
    task statement's own worked example. In particular the dead-branch-trimming keystone
    ("card"/"xyz" cases) and the full-cascade sequence's literal end states must not appear
    verbatim in the plan — only the same "car"/"ab" worked example already in prompt.md."""
    plan_text = json.dumps(c36.get_compiled_plan())
    assert "xyz" not in plan_text
    assert "starts_with('card')" not in plan_text and 'starts_with("card")' not in plan_text


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c36", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c36"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c36.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c36.get_compiled_plan()

    assert (private / c36._TEST_FILE_PATH).read_text() == c36._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c36._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c36.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
