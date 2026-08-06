"""
codebench task c36 — hard/hidden, data structure implementation (prefix tree).

Data-structure-implementations category, beyond c08's LRU-cache coverage (see that task's
docstring for the category's calibration conventions this one follows).

Ground truth for ``Trie`` verified independently with a throwaway nested-dict-node reference
script (NOT hand-computed) before being embedded here — see the offline test file's own
reimplementation (a Python ``set``-backed oracle, structurally unrelated to any node/children
tree) for the second, independent check. Verified traces:
    insert("cat"), insert("car"), insert("card"), insert("care"), insert("dog"):
        search("cat") -> True         search("ca") -> False (never inserted as a full word)
        starts_with("ca") -> True     starts_with("do") -> True   starts_with("z") -> False
        delete("car") -> True         search("car") -> False (word itself is gone)
        starts_with("car") -> True    (KEY CASE: "card"/"care" still extend this prefix, so the
                                        branch must survive even though "car" itself was deleted)
        search("card") -> True        search("care") -> True     (both untouched by the delete)
        delete("card") -> True        search("card") -> False    starts_with("card") -> False
        starts_with("car") -> True    (still — "care" alone keeps this prefix alive)
        delete("care") -> True        starts_with("car") -> False (now nothing extends "car")
        delete("car") again -> False  (already gone)              search("dog") -> True
        delete("nonexistent") -> False
    Prefix-of-another-word case (the OTHER direction of the same trickiness):
        insert("a"), insert("ab"); delete("a") -> True
        search("a") -> False   search("ab") -> True (must survive deleting its own prefix word!)
        starts_with("a") -> True
    Dead-branch trimming case (delete must not just flip a flag and leave orphaned nodes):
        insert("xyz"); delete("xyz") -> True
        search("xyz") -> False   starts_with("xy") -> False   starts_with("x") -> False
    Delete-then-reinsert case:
        insert("hi"); delete("hi"); insert("hi"); search("hi") -> True
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_trie.py"

_TEST_FILE_CONTENT = '''\
from trie import Trie


def _build_car_family():
    t = Trie()
    for w in ["cat", "car", "card", "care", "dog"]:
        t.insert(w)
    return t


def test_basic_insert_and_search():
    t = _build_car_family()
    assert t.search("cat") is True
    assert t.search("ca") is False


def test_starts_with_true_and_false():
    t = _build_car_family()
    assert t.starts_with("ca") is True
    assert t.starts_with("do") is True
    assert t.starts_with("z") is False


def test_delete_leaf_word_does_not_affect_sibling_words_sharing_prefix():
    t = _build_car_family()
    assert t.delete("car") is True
    assert t.search("car") is False
    assert t.starts_with("car") is True
    assert t.search("card") is True
    assert t.search("care") is True


def test_delete_word_that_is_a_prefix_of_another_word_keeps_the_longer_word():
    t = Trie()
    t.insert("a")
    t.insert("ab")
    assert t.delete("a") is True
    assert t.search("a") is False
    assert t.search("ab") is True
    assert t.starts_with("a") is True


def test_delete_trims_dead_branch_from_prefix_search():
    t = Trie()
    t.insert("xyz")
    assert t.delete("xyz") is True
    assert t.search("xyz") is False
    assert t.starts_with("xy") is False
    assert t.starts_with("x") is False


def test_delete_missing_word_returns_false_and_changes_nothing():
    t = Trie()
    t.insert("cat")
    t.insert("dog")
    assert t.delete("nonexistent") is False
    assert t.search("cat") is True
    assert t.search("dog") is True


def test_full_cascading_delete_sequence():
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


def test_delete_then_reinsert_works():
    t = Trie()
    t.insert("hi")
    assert t.delete("hi") is True
    assert t.search("hi") is False
    t.insert("hi")
    assert t.search("hi") is True
'''

# The two contract-violation shapes a naive delete() gets wrong gate the score, plus the
# integration sequence that chains both of them together: (1) deleting a word must not
# collapse a shared branch other words still need, (2) deleting a word must not leave dead
# nodes behind that make starts_with() lie. Basic insert/search, the missing-word no-op, and
# delete-then-reinsert are supporting/bonus credit — a broken delete() can still pass those.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_delete_leaf_word_does_not_affect_sibling_words_sharing_prefix",
    f"{_TEST_FILE_PATH}::test_delete_word_that_is_a_prefix_of_another_word_keeps_the_longer_word",
    f"{_TEST_FILE_PATH}::test_delete_trims_dead_branch_from_prefix_search",
    f"{_TEST_FILE_PATH}::test_full_cascading_delete_sequence",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c36",
        "title": "trie-prefix-tree",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `trie.py` that defines a class `Trie` implementing a prefix "
        "tree with:\n"
        "- `Trie()` — construct an empty trie.\n"
        "- `insert(self, word: str) -> None` — add `word` to the trie.\n"
        "- `search(self, word: str) -> bool` — return True iff `word` was previously inserted "
        "as a COMPLETE word (not merely a prefix of some longer inserted word).\n"
        "- `starts_with(self, prefix: str) -> bool` — return True iff at least one inserted "
        "word begins with `prefix` (an inserted word that equals `prefix` exactly also "
        "counts).\n"
        "- `delete(self, word: str) -> bool` — remove `word` if it was previously inserted as "
        "a complete word, and return True. If `word` was never inserted as a complete word, "
        "do nothing and return False.\n\n"
        "`delete` is the trickiest part, and must satisfy BOTH of these rules:\n"
        "1. Deleting a word must NEVER affect any other word that shares structure with it, "
        "in either direction — deleting a word that is itself a prefix of a longer inserted "
        "word (e.g. deleting \"a\" when \"ab\" is also present) must leave the longer word "
        "fully searchable, AND deleting a word that shares an earlier branch with other, "
        "unrelated words (e.g. deleting \"car\" when \"card\" and \"care\" are also present) "
        "must leave both of those fully searchable, with `starts_with(\"car\")` still True "
        "because \"card\"/\"care\" extend it.\n"
        "2. Deleting a word must actually clean up after itself: once you delete the ONLY "
        "word that ever used a given branch of the tree, `starts_with` on that now-dead "
        "prefix must return False, not True — do not just flip an internal flag on the "
        "deleted word's own node and leave orphaned nodes behind.\n\n"
        "Worked example:\n"
        "```\n"
        "t = Trie()\n"
        "for w in [\"cat\", \"car\", \"card\", \"care\", \"dog\"]:\n"
        "    t.insert(w)\n"
        "t.search(\"car\")        # -> True\n"
        "t.delete(\"car\")        # -> True (removes the word \"car\")\n"
        "t.search(\"car\")        # -> False (the word itself is gone)\n"
        "t.starts_with(\"car\")   # -> True (\"card\" and \"care\" still extend this prefix)\n"
        "t.search(\"card\")       # -> True (completely untouched)\n"
        "t.delete(\"card\"); t.delete(\"care\")\n"
        "t.starts_with(\"car\")   # -> False now (nothing left extends \"car\")\n"
        "t.search(\"dog\")        # -> True (never touched by any of this)\n"
        "```\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "implementation yourself — reproduce the worked example above, and also try "
        "inserting both \"a\" and \"ab\" then deleting \"a\" (confirm \"ab\" is still findable "
        "via both `search` and `starts_with`) — before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "trie", "classes": ["Trie"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one class, no I/O, no multi-file structure — same rationale as c08."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write trie.py implementing class Trie with __init__(self) (empty "
                    "trie), insert(self, word: str) -> None, search(self, word: str) -> "
                    "bool (True iff word was inserted as a COMPLETE word, not just a "
                    "prefix), starts_with(self, prefix: str) -> bool (True iff any "
                    "inserted word begins with prefix), and delete(self, word: str) -> "
                    "bool (removes word if it was a complete word and returns True, else "
                    "returns False and changes nothing). delete() is the hard part and "
                    "must get BOTH of these right: (1) deleting a word must never affect "
                    "any other word that shares structure with it in either direction — "
                    "deleting a word that is a prefix of a longer inserted word (e.g. "
                    "delete 'a' when 'ab' is also present) must leave the longer word "
                    "fully searchable; deleting a word that shares an earlier branch with "
                    "unrelated words (e.g. delete 'car' when 'card'/'care' are also "
                    "present) must leave those fully searchable and starts_with('car') "
                    "must stay True because they extend it; (2) after deleting the ONLY "
                    "word that used a given branch, starts_with on that now-dead prefix "
                    "must become False — actually remove the orphaned nodes, don't just "
                    "flip a flag. Use write_file to create trie.py, then use run_python to "
                    "reproduce this worked example: insert cat/car/card/care/dog; delete "
                    "'car' -> search('car') is False but starts_with('car') is True and "
                    "search('card')/search('care') are still True; then also insert 'a' "
                    "and 'ab' into a fresh Trie, delete 'a', and confirm search('ab') is "
                    "still True. Fix issues with patch_file/run_python until confident, "
                    "then finish."
                ),
                "expect": "trie.py written, defining Trie with insert/search/starts_with/"
                          "delete, delete's prefix-safety and dead-branch-trimming both "
                          "sanity-checked with run_python",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm trie.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["trie.py"]},
    }
