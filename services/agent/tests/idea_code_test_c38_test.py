"""
Adversarial offline checks for codebench task c38 (union-find-disjoint-set) — no Docker, no LLM.

Mirrors idea_code_test_c08_test.py's structure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c38_union_find_disjoint_set as c38


class _IndependentDisjointSet:
    """Reimplemented independently of the task module's own iterative-list-array prose spec,
    via RECURSIVE find and a plain dict instead of a list, to catch a mistake in the embedded
    canonical test file's own expectations without sharing an implementation bug-class with a
    typical list-array union-find."""

    def __init__(self, n: int):
        self._parent = {i: i for i in range(n)}
        self._rank = {i: 0 for i in range(n)}

    def find(self, x: int) -> int:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # recursive path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self._rank[rx] < self._rank[ry]:
            self._parent[rx] = ry
        elif self._rank[rx] > self._rank[ry]:
            self._parent[ry] = rx
        else:
            self._parent[ry] = rx
            self._rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def parent_of(self, x: int) -> int:
        return self._parent[x]

    def rank_of(self, x: int) -> int:
        return self._rank[x]


def _build_eight_element_forest(cls=_IndependentDisjointSet):
    ds = cls(8)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds.union(a, b)
    return ds


def test_ground_truth_initial_state_all_singletons():
    ds = _IndependentDisjointSet(5)
    for i in range(5):
        assert ds.find(i) == i
    assert ds.connected(0, 1) is False


def test_ground_truth_union_reports_true_and_merges():
    ds = _IndependentDisjointSet(3)
    assert ds.union(0, 1) is True
    assert ds.connected(0, 1) is True


def test_ground_truth_union_already_connected_is_noop():
    ds = _IndependentDisjointSet(3)
    ds.union(0, 1)
    assert ds.union(0, 1) is False
    assert ds.connected(0, 1) is True


def test_ground_truth_tie_break_convention():
    ds = _IndependentDisjointSet(2)
    assert ds.union(0, 1) is True
    assert ds.parent_of(1) == 0
    assert ds.parent_of(0) == 0
    assert ds.rank_of(0) == 1


def test_ground_truth_set_membership_after_unions():
    ds = _build_eight_element_forest()
    assert ds.connected(0, 3) is True
    assert ds.connected(1, 2) is True
    assert ds.connected(4, 7) is True
    assert ds.connected(3, 7) is True
    assert ds.connected(1, 5) is True


def test_ground_truth_raw_parent_pointers_before_any_find_are_multilevel():
    """This is the load-bearing fact the path-compression keystone depends on: our recursive/
    dict-based reference (structurally unrelated to the task's iterative/list reference)
    independently reproduces the SAME multi-level chain 7 -> 6 -> 4 -> 0 before find() is ever
    called on 7, confirming this isn't an artifact of one particular find() implementation."""
    ds = _build_eight_element_forest()
    assert ds.parent_of(7) == 6
    assert ds.parent_of(6) == 4
    assert ds.parent_of(4) == 0
    assert ds.parent_of(3) == 2


def test_ground_truth_find_flattens_the_path():
    ds = _build_eight_element_forest()
    root = ds.find(7)
    assert root == 0
    assert ds.parent_of(7) == 0
    assert ds.parent_of(6) == 0
    assert ds.parent_of(3) == 2  # untouched sibling branch


def test_ground_truth_second_find_flattens_its_own_untouched_chain():
    ds = _build_eight_element_forest()
    ds.find(7)
    assert ds.parent_of(3) == 2
    root = ds.find(3)
    assert root == 0
    assert ds.parent_of(3) == 0
    assert ds.parent_of(2) == 0


def test_ground_truth_union_on_merged_roots_stays_false():
    ds = _build_eight_element_forest()
    ds.find(7)
    ds.find(3)
    assert ds.union(3, 7) is False
    assert ds.connected(3, 7) is True


def test_ground_truth_union_by_rank_resolves_roots_before_comparing_ranks():
    ds = _IndependentDisjointSet(9)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds.union(a, b)
    ds.find(7)
    ds.find(3)
    assert ds.rank_of(0) == 3
    assert ds.rank_of(3) == 0  # 3's own stored rank is stale -- it hasn't been a root
    assert ds.union(3, 8) is True
    assert ds.rank_of(0) == 3  # true root ranks (3 vs 0) were not tied -- no increment
    assert ds.parent_of(8) == 0
    assert ds.parent_of(3) == 0


def test_a_raw_argument_rank_bug_is_actually_caught():
    """Prove the resolved-root-rank keystone is discriminating: a plausible buggy union()
    that resolves rx/ry correctly for ATTACHMENT but compares the raw stored rank of the
    original x/y arguments (not rx/ry) for the tie-break decision diverges from the correct
    answer exactly at this scenario, even though it passes every other case in this suite
    (every other union() call here happens to be invoked with x/y that are already roots)."""

    class BuggyRawArgRank:
        def __init__(self, n):
            self._parent = list(range(n))
            self._rank = [0] * n

        def find(self, x):
            root = x
            while self._parent[root] != root:
                root = self._parent[root]
            while self._parent[x] != root:
                self._parent[x], x = root, self._parent[x]
            return root

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            # BUG: compares the rank of the raw x/y arguments, not the resolved roots rx/ry.
            if self._rank[x] < self._rank[y]:
                self._parent[rx] = ry
            elif self._rank[x] > self._rank[y]:
                self._parent[ry] = rx
            else:
                self._parent[ry] = rx
                self._rank[rx] += 1
            return True

        def connected(self, x, y):
            return self.find(x) == self.find(y)

        def parent_of(self, x):
            return self._parent[x]

        def rank_of(self, x):
            return self._rank[x]

    ds = BuggyRawArgRank(9)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds.union(a, b)
    ds.find(7)
    ds.find(3)
    ds.union(3, 8)
    assert ds.rank_of(0) != 3  # WRONG per spec (correct impl leaves rank_of(0) at 3)


class _BuggyYOnlyRaw:
    """A half-fix: resolves the FIRST argument (x) to its root before comparing ranks, but
    forgets to resolve the SECOND (y), comparing against y's raw stored rank instead. Attachment
    itself is still always between the correctly-resolved roots -- only the COMPARISON is wrong."""

    def __init__(self, n):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x):
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # BUG: rx is correctly resolved, but the SECOND operand still uses raw self._rank[y].
        if self._rank[rx] < self._rank[y]:
            self._parent[rx] = ry
        elif self._rank[rx] > self._rank[y]:
            self._parent[ry] = rx
        else:
            self._parent[ry] = rx
            self._rank[rx] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def parent_of(self, x):
        return self._parent[x]

    def rank_of(self, x):
        return self._rank[x]


def test_y_only_raw_bug_survives_the_2026_08_06_case_but_is_caught_by_the_2026_08_07_case():
    """Prove the 2026-08-07 hardening actually closes a real gap left by the 2026-08-06 case
    alone: a union() that resolves its FIRST argument but forgets its SECOND is INDISTINGUISHABLE
    from a fully correct implementation on the 2026-08-06 scenario (because that scenario's
    second argument, a fresh singleton, is already its own root -- raw and resolved rank are the
    same number either way), but diverges sharply on the 2026-08-07 scenario (whose second
    argument, node 6, is genuinely not a root)."""
    ds_old_scenario = _BuggyYOnlyRaw(9)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds_old_scenario.union(a, b)
    ds_old_scenario.find(7)
    ds_old_scenario.find(3)
    ds_old_scenario.union(3, 8)
    assert ds_old_scenario.rank_of(0) == 3  # matches CORRECT behavior -- this bug is invisible here

    ds_new_scenario = _BuggyYOnlyRaw(10)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds_new_scenario.union(a, b)
    ds_new_scenario.find(7)
    ds_new_scenario.find(3)
    ds_new_scenario.union(8, 9)
    ds_new_scenario.union(8, 6)
    assert ds_new_scenario.parent_of(0) == 8   # WRONG per spec -- correct impl leaves 0 as root
    assert ds_new_scenario.rank_of(8) == 2     # WRONG per spec -- correct impl leaves rank_of(8) at 1


def test_no_path_compression_bug_is_actually_caught():
    """Prove the path-compression keystone is discriminating: a plausible buggy
    implementation that finds the root correctly but never rewrites parent pointers along the
    way leaves the raw chain exactly as deep as before -- the opposite of what's required."""

    class NoCompressionDisjointSet:
        def __init__(self, n):
            self._parent = list(range(n))
            self._rank = [0] * n

        def find(self, x):
            while self._parent[x] != x:  # BUG: never repoints anything, just walks
                x = self._parent[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self._rank[rx] < self._rank[ry]:
                self._parent[rx] = ry
            elif self._rank[rx] > self._rank[ry]:
                self._parent[ry] = rx
            else:
                self._parent[ry] = rx
                self._rank[rx] += 1
            return True

        def connected(self, x, y):
            return self.find(x) == self.find(y)

        def parent_of(self, x):
            return self._parent[x]

        def rank_of(self, x):
            return self._rank[x]

    ds = _build_eight_element_forest(cls=NoCompressionDisjointSet)
    ds.find(7)
    assert ds.parent_of(7) == 6  # WRONG per spec (correct impl compresses this to 0)


def test_embedded_test_file_asserts_match_ground_truth_reference():
    content = c38.get_grading_payload()["tests"][c38._TEST_FILE_PATH]
    for literal in [
        "assert ds.parent_of(7) == 6",
        "assert ds.parent_of(6) == 4",
        "assert ds.parent_of(4) == 0",
        "assert ds.parent_of(3) == 2",
    ]:
        assert literal in content


def test_embedded_test_file_contains_the_hardened_resolved_root_rank_case():
    content = c38.get_grading_payload()["tests"][c38._TEST_FILE_PATH]
    for literal in [
        "DisjointSet(9)",
        "assert ds.union(3, 8) is True",
        "assert ds.rank_of(0) == 3",
        "test_union_by_rank_resolves_arguments_to_their_current_roots_before_comparing_ranks",
    ]:
        assert literal in content


def test_embedded_test_file_contains_the_2026_08_07_second_argument_case():
    content = c38.get_grading_payload()["tests"][c38._TEST_FILE_PATH]
    for literal in [
        "DisjointSet(10)",
        "ds.union(8, 9)",
        "assert ds.union(8, 6) is True",
        "assert ds.parent_of(8) == 0",
        "assert ds.parent_of(0) == 0",
        "test_union_by_rank_also_resolves_the_second_argument_not_just_the_first",
    ]:
        assert literal in content


def test_keystone_ids_reference_real_test_functions():
    content = c38.get_grading_payload()["tests"][c38._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c38.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c38._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_basic_and_bonus_cases():
    non_keystone = {
        "test_initial_state_all_singletons",
        "test_union_merges_two_singletons_and_reports_true",
        "test_path_compression_is_transitive_on_the_next_find_too",
        "test_union_returns_false_and_structure_stable_when_called_on_already_merged_roots",
    }
    for name in non_keystone:
        assert f"{c38._TEST_FILE_PATH}::{name}" not in c38.KEYSTONE_TEST_IDS


def test_keystone_includes_the_hardened_2026_08_06_case():
    # The resolved-root-rank case is the one added after live calibration showed a strong
    # coding agent acing the previous suite, which never forced union() to resolve a non-root
    # argument to its current root before comparing ranks -- it must gate the score.
    assert (
        f"{c38._TEST_FILE_PATH}::"
        "test_union_by_rank_resolves_arguments_to_their_current_roots_before_comparing_ranks"
        in c38.KEYSTONE_TEST_IDS
    )


def test_keystone_includes_the_hardened_2026_08_07_case():
    # The second-argument-resolution case is the one added after round-2 live calibration
    # showed a strong coding agent still acing the suite, because the 2026-08-06 case's second
    # argument happened to already be a root -- it must gate the score too.
    assert (
        f"{c38._TEST_FILE_PATH}::"
        "test_union_by_rank_also_resolves_the_second_argument_not_just_the_first"
        in c38.KEYSTONE_TEST_IDS
    )


def test_visibility_is_hidden():
    assert c38.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c38.get_sandbox_fixture() == {}


def test_task_statement_contains_a_worked_example_as_an_anchor():
    statement = c38.get_task_statement()
    assert "ds = DisjointSet(4)" in statement
    assert "ds.parent_of(3)" in statement
    assert "ds.find(3)" in statement


def test_grading_payload_shape():
    payload = c38.get_grading_payload()
    assert payload["tests"] == {c38._TEST_FILE_PATH: c38._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {"module": "union_find", "classes": ["DisjointSet"]}
    assert payload["keystone_test_ids"] == c38.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c38.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "union_find.py" in leaf["instruction"]
    assert "DisjointSet" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["union_find.py"]}
    json.dumps(plan)


def test_compiled_plan_does_not_leak_hidden_keystone_values():
    """The compiled plan is mounted read-only into the agent's sandbox (public/plan.json). It
    may restate the small 4-element worked example already public in prompt.md, but must not
    state the hidden 8-element keystone's literal parent-pointer values (6, 4-as-parent-of-6,
    etc.), nor the hardened 9-element/rank_of(0)==3 resolved-root-rank case's literal values,
    that only appear in the private canonical test file."""
    plan_text = json.dumps(c38.get_compiled_plan())
    assert "parent_of(7)" not in plan_text
    assert "parent_of(6)" not in plan_text
    assert "DisjointSet(9)" not in plan_text
    assert "union(3, 8)" not in plan_text
    assert "rank_of(0) == 3" not in plan_text
    assert "DisjointSet(10)" not in plan_text
    assert "union(8, 6)" not in plan_text
    assert "rank_of(8)" not in plan_text


def test_task_statement_does_not_leak_hardened_keystone_values():
    statement = c38.get_task_statement()
    assert "DisjointSet(9)" not in statement
    assert "union(3, 8)" not in statement
    assert "rank_of(0) == 3" not in statement
    assert "DisjointSet(10)" not in statement
    assert "union(8, 6)" not in statement
    assert "rank_of(8)" not in statement


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c38", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c38"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c38.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c38.get_compiled_plan()

    assert (private / c38._TEST_FILE_PATH).read_text() == c38._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c38._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c38.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
