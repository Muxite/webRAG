"""
Adversarial offline checks for codebench task c37 (lane-dispatcher) -- no Docker, no LLM.

Second full replacement (see the task module's own docstring for why the LFU-with-insertion-
order-tiebreak design, itself already a full replacement of an earlier binary-heap design, was
STILL too tractable for a Python dict-and-min()-reaching-for implementer, even against a genuine
live-calibration submission). Same validation shape as idea_code_test_c22_test.py /
idea_code_test_c37_test.py's first version: independently re-derive ground truth with a SECOND,
differently-coded oracle (rotates a slice of the lane-order list on every dispatch call, rather
than the task module's own reference design of modulo index arithmetic against a lane-identity
pointer), run it against the scenarios embedded in the canonical test file, then separately
confirm those scenarios are what's actually embedded there. Also proves -- by actually
constructing and running two independently plausible near-miss mutants (one directly informed by
Aider's own real c37 round-3 submission's general implementation SHAPE -- a stale numeric list
index invalidated by promote()'s reorder -- and one from a natural misreading of the word
"promote") against the REAL canonical test file via pytest, not by asserting from authorial
confidence, exactly which tests each one fails.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c37_lane_dispatcher as c37


# ---------------------------------------------------------------------------
# Independent re-derivation: a from-scratch oracle (rotates a SLICE of the order list every
# dispatch call, tracks "due lane" by identity) plus two independently-coded mutants.
# ---------------------------------------------------------------------------


class _SliceRotationOracle:
    def __init__(self, lanes):
        self.order = list(lanes)
        self.queues = {lane: [] for lane in lanes}
        self.next_lane = self.order[0] if self.order else None

    def enqueue(self, lane, item):
        if lane not in self.queues:
            raise KeyError(lane)
        self.queues[lane].append(item)

    def dispatch(self):
        if not self.order:
            raise IndexError("no lanes")
        start = self.order.index(self.next_lane)
        rotated = self.order[start:] + self.order[:start]
        for lane in rotated:
            if self.queues[lane]:
                item = self.queues[lane].pop(0)
                served_idx = self.order.index(lane)
                self.next_lane = self.order[(served_idx + 1) % len(self.order)]
                return item
        raise IndexError("all lanes empty")

    def promote(self, lane):
        if lane not in self.queues:
            raise KeyError(lane)
        self.order.remove(lane)
        self.order.insert(0, lane)

    def pending_count(self, lane):
        if lane not in self.queues:
            raise KeyError(lane)
        return len(self.queues[lane])


class _StaleIndexMutant:
    """Plausible near-miss directly informed by Aider's real c37 round-3 submission SHAPE
    (a per-key scalar bookkeeping value that looks like it tracks order but doesn't survive a
    structural reorder): tracks "due lane" as a raw numeric list index, which promote()
    silently invalidates."""

    def __init__(self, lanes):
        self.lanes = list(lanes)
        self.queues = {lane: [] for lane in lanes}
        self.cursor = 0

    def enqueue(self, lane, item):
        if lane not in self.queues:
            raise KeyError(lane)
        self.queues[lane].append(item)

    def dispatch(self):
        n = len(self.lanes)
        if n == 0:
            raise IndexError("no lanes")
        for offset in range(n):
            idx = (self.cursor + offset) % n
            lane = self.lanes[idx]
            if self.queues[lane]:
                item = self.queues[lane].pop(0)
                self.cursor = (idx + 1) % n
                return item
        raise IndexError("all lanes empty")

    def promote(self, lane):
        if lane not in self.queues:
            raise KeyError(lane)
        self.lanes.remove(lane)
        self.lanes.insert(0, lane)

    def pending_count(self, lane):
        if lane not in self.queues:
            raise KeyError(lane)
        return len(self.queues[lane])


class _PromoteCutsLineMutant:
    """Plausible near-miss: reads "promote" as "prioritize this lane starting right now,"
    setting the due lane to the promoted lane immediately instead of merely reordering future
    scans."""

    def __init__(self, lanes):
        self._order = list(lanes)
        self._queues = {lane: [] for lane in lanes}
        self._next_lane = self._order[0] if self._order else None

    def enqueue(self, lane, item):
        if lane not in self._queues:
            raise KeyError(lane)
        self._queues[lane].append(item)

    def dispatch(self):
        n = len(self._order)
        if n == 0:
            raise IndexError("no lanes")
        start_idx = self._order.index(self._next_lane)
        for offset in range(n):
            idx = (start_idx + offset) % n
            lane = self._order[idx]
            if self._queues[lane]:
                item = self._queues[lane].pop(0)
                next_idx = (idx + 1) % n
                self._next_lane = self._order[next_idx]
                return item
        raise IndexError("all lanes empty")

    def promote(self, lane):
        if lane not in self._queues:
            raise KeyError(lane)
        self._order.remove(lane)
        self._order.insert(0, lane)
        self._next_lane = lane  # BUG: cuts in line immediately

    def pending_count(self, lane):
        if lane not in self._queues:
            raise KeyError(lane)
        return len(self._queues[lane])


def test_ground_truth_worked_scenario_matches_oracle():
    oracle = _SliceRotationOracle(["a", "b", "c"])
    oracle.enqueue("a", 1)
    oracle.enqueue("a", 2)
    oracle.enqueue("b", 10)
    oracle.enqueue("c", 100)
    assert oracle.dispatch() == 1
    oracle.promote("c")
    assert oracle.dispatch() == 10
    assert oracle.dispatch() == 100
    assert oracle.dispatch() == 2
    with pytest.raises(IndexError):
        oracle.dispatch()


def test_stale_index_mutant_diverges_on_the_worked_scenario():
    mutant = _StaleIndexMutant(["a", "b", "c"])
    mutant.enqueue("a", 1)
    mutant.enqueue("a", 2)
    mutant.enqueue("b", 10)
    mutant.enqueue("c", 100)
    assert mutant.dispatch() == 1
    mutant.promote("c")
    # BUG: serves "a"'s leftover item instead of "b", since the numeric cursor now (post-reorder)
    # points at "a"'s new position rather than "b"
    assert mutant.dispatch() == 2
    with pytest.raises(AssertionError):
        assert mutant.dispatch() == 100  # further diverges from the correct sequence


def test_promote_cuts_line_mutant_diverges_immediately_on_any_non_due_promote():
    mutant = _PromoteCutsLineMutant(["a", "b", "c"])
    mutant.enqueue("a", "a1")
    mutant.enqueue("b", "b1")
    mutant.enqueue("c", "c1")
    assert mutant.dispatch() == "a1"  # "b" now due, per spec
    mutant.promote("c")  # BUG: cuts in line
    assert mutant.dispatch() == "c1"  # wrong: should have been "b1"


def _fuzz(cls_under_test, seed, steps=300, lanes=("a", "b", "c", "d")):
    rng = random.Random(seed)
    a = cls_under_test(list(lanes))
    b = _SliceRotationOracle(list(lanes))
    for i in range(steps):
        op = rng.choice(["enqueue", "dispatch", "promote", "pending_count"])
        if op == "enqueue":
            lane = rng.choice(lanes)
            a.enqueue(lane, i)
            b.enqueue(lane, i)
        elif op == "dispatch":
            a_exc = b_exc = None
            a_val = b_val = None
            try:
                a_val = a.dispatch()
            except IndexError:
                a_exc = "IndexError"
            try:
                b_val = b.dispatch()
            except IndexError:
                b_exc = "IndexError"
            if a_exc != b_exc or a_val != b_val:
                return False, i
        elif op == "promote":
            lane = rng.choice(lanes)
            a.promote(lane)
            b.promote(lane)
        else:
            lane = rng.choice(lanes)
            if a.pending_count(lane) != b.pending_count(lane):
                return False, i
    return True, steps


def test_reference_shaped_implementation_matches_oracle_across_fixed_seeds():
    """The task module's own worked example (see get_task_statement) is structurally identical
    to a lane-identity-tracking implementation; confirm that SHAPE agrees with the independent
    oracle over many fixed seeds (never asserted from confidence alone)."""

    class _LaneIdentityRef:
        def __init__(self, lanes):
            self._order = list(lanes)
            self._queues = {lane: [] for lane in lanes}
            self._next_lane = self._order[0] if self._order else None

        def enqueue(self, lane, item):
            if lane not in self._queues:
                raise KeyError(lane)
            self._queues[lane].append(item)

        def dispatch(self):
            n = len(self._order)
            if n == 0:
                raise IndexError("no lanes")
            start_idx = self._order.index(self._next_lane)
            for offset in range(n):
                idx = (start_idx + offset) % n
                lane = self._order[idx]
                if self._queues[lane]:
                    item = self._queues[lane].pop(0)
                    self._next_lane = self._order[(idx + 1) % n]
                    return item
            raise IndexError("all lanes empty")

        def promote(self, lane):
            if lane not in self._queues:
                raise KeyError(lane)
            self._order.remove(lane)
            self._order.insert(0, lane)

        def pending_count(self, lane):
            if lane not in self._queues:
                raise KeyError(lane)
            return len(self._queues[lane])

    for seed in range(30):
        ok, step = _fuzz(_LaneIdentityRef, seed)
        assert ok, f"seed={seed} diverged at step {step}"


def test_stale_index_mutant_diverges_on_every_fuzz_seed():
    diverged = 0
    for seed in range(30):
        ok, _ = _fuzz(_StaleIndexMutant, seed)
        if not ok:
            diverged += 1
    assert diverged == 30, f"only {diverged}/30 seeds caught the stale-index mutant"


def test_promote_cuts_line_mutant_diverges_on_every_fuzz_seed():
    diverged = 0
    for seed in range(30):
        ok, _ = _fuzz(_PromoteCutsLineMutant, seed)
        if not ok:
            diverged += 1
    assert diverged == 30, f"only {diverged}/30 seeds caught the promote-cuts-line mutant"


def test_keystone_ids_reference_real_test_functions():
    content = c37.get_grading_payload()["tests"][c37._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 14
    for node_id in c37.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c37._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_basic_and_bonus_cases():
    non_keystone = {
        "test_basic_fifo_order_within_a_single_lane",
        "test_dispatch_raises_index_error_when_every_lane_is_empty",
        "test_unknown_lane_name_raises_key_error_everywhere",
        "test_plain_round_robin_with_no_promote_calls",
        "test_empty_lane_is_skipped_over_in_the_scan",
        "test_pending_count_tracks_state_through_enqueue_and_dispatch",
        "test_promoting_a_lane_already_at_the_front_is_a_harmless_no_op",
        "test_dispatch_after_all_lanes_drain_then_refill_still_works",
        "test_zero_configured_lanes_dispatch_and_enqueue_both_error",
    }
    for name in non_keystone:
        assert f"{c37._TEST_FILE_PATH}::{name}" not in c37.KEYSTONE_TEST_IDS
    assert set(c37.KEYSTONE_TEST_IDS) == {
        f"{c37._TEST_FILE_PATH}::test_promote_does_not_let_a_lane_cut_in_line_ahead_of_the_lane_already_due",
        f"{c37._TEST_FILE_PATH}::test_promote_reorders_future_scans_without_disturbing_the_currently_due_lane",
        f"{c37._TEST_FILE_PATH}::test_promote_with_a_pending_item_still_sitting_in_the_reordered_lane",
        f"{c37._TEST_FILE_PATH}::test_promoting_the_currently_due_lane_itself_changes_nothing_observable",
        f"{c37._TEST_FILE_PATH}::test_long_randomized_sequence_matches_an_independent_oracle",
    }


def test_visibility_is_hidden():
    assert c37.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c37.get_sandbox_fixture() == {}


def test_task_statement_worked_example_distinct_from_hidden_keystone_literals():
    statement = c37.get_task_statement()
    assert "north" in statement
    assert "south" in statement
    assert "east" in statement
    # the statement's worked example uses its own item labels ("n1"/"s1"/"e1") and never the
    # hidden canonical test's own scenario item labels/values -- keeps the public worked
    # example and the private discriminating literals textually distinct. The public example
    # is also deliberately SIMPLER than the keystone "stale index" scenario (single item per
    # lane, no leftover item sitting in a reordered lane) -- it demonstrates the core rule
    # (promote reorders future scans without disturbing who's already due) without handing the
    # harder leftover-item interaction the keystone test actually gates on.
    for leaked in ('"a1"', '"b1"', '"c1"', '"d1"', "d.dispatch()             # -> 1", "-> 10", "-> 100"):
        assert leaked not in statement, leaked


def test_grading_payload_shape():
    payload = c37.get_grading_payload()
    assert payload["tests"] == {c37._TEST_FILE_PATH: c37._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {"module": "lane_dispatcher", "classes": ["LaneDispatcher"]}
    assert payload["keystone_test_ids"] == c37.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c37.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "lane_dispatcher.py" in leaf["instruction"]
    assert "LaneDispatcher" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["lane_dispatcher.py"]}
    json.dumps(plan)


def test_compiled_plan_does_not_leak_hidden_keystone_literals():
    plan_text = json.dumps(c37.get_compiled_plan())
    for leaked in ('"a"', '"b"', '"c"', '"d"', "20260807", "north", "south", "east"):
        assert leaked not in plan_text, leaked
    # never the literal mechanism/fix (store the due lane by identity, not by index)
    for leaked in ("by identity", "lane identity", ".index(", "store the due lane"):
        assert leaked.lower() not in plan_text.lower(), f"plan leaks the mechanism via {leaked!r}"


# ---------------------------------------------------------------------------
# Live pytest runs against the REAL embedded canonical test file (not the reimplementations
# above) -- proves the actual shipped fixture, not just this validator's model of it, has the
# claimed discriminating power.
# ---------------------------------------------------------------------------

_REFERENCE_SOURCE = '''\
class LaneDispatcher:
    def __init__(self, lanes):
        if len(set(lanes)) != len(lanes):
            raise ValueError("lane names must be unique")
        self._order = list(lanes)
        self._queues = {lane: [] for lane in lanes}
        self._next_lane = self._order[0] if self._order else None

    def enqueue(self, lane, item):
        if lane not in self._queues:
            raise KeyError(lane)
        self._queues[lane].append(item)

    def dispatch(self):
        n = len(self._order)
        if n == 0:
            raise IndexError("no lanes configured")
        start_idx = self._order.index(self._next_lane)
        for offset in range(n):
            idx = (start_idx + offset) % n
            lane = self._order[idx]
            if self._queues[lane]:
                item = self._queues[lane].pop(0)
                next_idx = (idx + 1) % n
                self._next_lane = self._order[next_idx]
                return item
        raise IndexError("all lanes empty")

    def promote(self, lane):
        if lane not in self._queues:
            raise KeyError(lane)
        self._order.remove(lane)
        self._order.insert(0, lane)

    def pending_count(self, lane):
        if lane not in self._queues:
            raise KeyError(lane)
        return len(self._queues[lane])
'''

_STALE_INDEX_MUTANT_SOURCE = '''\
class LaneDispatcher:
    def __init__(self, lanes):
        self.lanes = list(lanes)
        self.queues = {lane: [] for lane in lanes}
        self.cursor = 0

    def enqueue(self, lane, item):
        if lane not in self.queues:
            raise KeyError(lane)
        self.queues[lane].append(item)

    def dispatch(self):
        n = len(self.lanes)
        if n == 0:
            raise IndexError("no lanes")
        for offset in range(n):
            idx = (self.cursor + offset) % n
            lane = self.lanes[idx]
            if self.queues[lane]:
                item = self.queues[lane].pop(0)
                self.cursor = (idx + 1) % n
                return item
        raise IndexError("all lanes empty")

    def promote(self, lane):
        if lane not in self.queues:
            raise KeyError(lane)
        self.lanes.remove(lane)
        self.lanes.insert(0, lane)

    def pending_count(self, lane):
        if lane not in self.queues:
            raise KeyError(lane)
        return len(self.queues[lane])
'''

_PROMOTE_CUTS_LINE_MUTANT_SOURCE = '''\
class LaneDispatcher:
    def __init__(self, lanes):
        self._order = list(lanes)
        self._queues = {lane: [] for lane in lanes}
        self._next_lane = self._order[0] if self._order else None

    def enqueue(self, lane, item):
        if lane not in self._queues:
            raise KeyError(lane)
        self._queues[lane].append(item)

    def dispatch(self):
        n = len(self._order)
        if n == 0:
            raise IndexError("no lanes")
        start_idx = self._order.index(self._next_lane)
        for offset in range(n):
            idx = (start_idx + offset) % n
            lane = self._order[idx]
            if self._queues[lane]:
                item = self._queues[lane].pop(0)
                next_idx = (idx + 1) % n
                self._next_lane = self._order[next_idx]
                return item
        raise IndexError("all lanes empty")

    def promote(self, lane):
        if lane not in self._queues:
            raise KeyError(lane)
        self._order.remove(lane)
        self._order.insert(0, lane)
        self._next_lane = lane

    def pending_count(self, lane):
        if lane not in self._queues:
            raise KeyError(lane)
        return len(self._queues[lane])
'''


def _run_pytest_against_impl(tmp_path, impl_source: str):
    (tmp_path / "lane_dispatcher.py").write_text(impl_source)
    test_content = c37.get_grading_payload()["tests"][c37._TEST_FILE_PATH]
    (tmp_path / "test_lane_dispatcher.py").write_text(test_content)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", str(tmp_path / "test_lane_dispatcher.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    return result.stdout + result.stderr


def _failed_test_names(pytest_output: str):
    return set(re.findall(r"FAILED test_lane_dispatcher\.py::(test_\w+)", pytest_output))


def test_reference_implementation_passes_every_canonical_test(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _REFERENCE_SOURCE)
    assert "14 passed" in output, output


def test_stale_index_mutant_fails_the_expected_tests_against_real_fixture(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _STALE_INDEX_MUTANT_SOURCE)
    failed = _failed_test_names(output)
    assert failed == {
        "test_promote_with_a_pending_item_still_sitting_in_the_reordered_lane",
        "test_promoting_the_currently_due_lane_itself_changes_nothing_observable",
        "test_long_randomized_sequence_matches_an_independent_oracle",
    }, output


def test_promote_cuts_line_mutant_fails_the_expected_tests_against_real_fixture(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _PROMOTE_CUTS_LINE_MUTANT_SOURCE)
    failed = _failed_test_names(output)
    assert failed == {
        "test_promote_does_not_let_a_lane_cut_in_line_ahead_of_the_lane_already_due",
        "test_promote_reorders_future_scans_without_disturbing_the_currently_due_lane",
        "test_promote_with_a_pending_item_still_sitting_in_the_reordered_lane",
        "test_long_randomized_sequence_matches_an_independent_oracle",
    }, output


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c37", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c37"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c37.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c37.get_compiled_plan()

    assert (private / c37._TEST_FILE_PATH).read_text() == c37._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c37._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c37.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
