"""
Adversarial offline checks for codebench task c25 (bounded-blocking-queue) -- no Docker
(besides the materialize_task subprocess check), no LLM.

Mirrors idea_code_test_c24_test.py's spirit: re-run the actual grading scenarios against a
hand-written correct reference and hand-written broken references, confirming empirically
that the canonical test file's discriminating power is real, not assumed. See this task
module's own docstring for why c25 leans on plain thread-join-with-timeout (deterministic)
rather than a probabilistic race window (c24's approach) for its primary signal: a
naive check-then-mutate race on a plain list essentially never manifests reliably under
CPython's GIL scheduling in practice (confirmed empirically while authoring c24), so this
task instead targets two 100%-reproducible bugs: "bounded" in name only (put() never
blocks), and holding the lock while waiting (an immediate, total deadlock).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from agent.app.idea_code_tests import test_c25_bounded_blocking_queue as c25


class _CorrectQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self._items = []
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def put(self, item):
        with self._not_full:
            while len(self._items) >= self.capacity:
                self._not_full.wait()
            self._items.append(item)
            self._not_empty.notify()

    def get(self):
        with self._not_empty:
            while len(self._items) == 0:
                self._not_empty.wait()
            item = self._items.pop(0)
            self._not_full.notify()
            return item

    def qsize(self):
        with self._lock:
            return len(self._items)


class _NoCapacityQueue:
    """'Bounded' in name only -- put() never actually blocks."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._items = []
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            self._items.append(item)

    def get(self):
        while True:
            with self._lock:
                if self._items:
                    return self._items.pop(0)
            time.sleep(0)

    def qsize(self):
        with self._lock:
            return len(self._items)


class _LockHeldWhileWaitingQueue:
    """Holds the lock WHILE waiting instead of releasing it -- deadlocks immediately."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._items = []
        self._lock = threading.Lock()

    def put(self, item):
        with self._lock:
            while len(self._items) >= self.capacity:
                time.sleep(0.01)
            self._items.append(item)

    def get(self):
        with self._lock:
            while len(self._items) == 0:
                time.sleep(0.01)
            return self._items.pop(0)

    def qsize(self):
        with self._lock:
            return len(self._items)


def _join_all_with_deadline(threads, deadline_s, poll=0.02):
    end = time.time() + deadline_s
    while time.time() < end:
        if all(not t.is_alive() for t in threads):
            return True
        time.sleep(poll)
    return all(not t.is_alive() for t in threads)


def _blocks_when_full(cls, capacity=2, deadline=2.0):
    q = cls(capacity)
    q.put("a")
    q.put("b")
    done = threading.Event()

    def putter():
        q.put("c")
        done.set()

    t = threading.Thread(target=putter, daemon=True)
    t.start()
    if done.wait(timeout=0.3):
        return False
    result = {}

    def getter():
        result["item"] = q.get()

    g = threading.Thread(target=getter, daemon=True)
    g.start()
    if not _join_all_with_deadline([g], deadline):
        return False
    if not _join_all_with_deadline([t], deadline):
        return False
    return True


def _stress_no_loss(cls, capacity=5, n_producers=6, items_per_producer=60, n_consumers=6,
                     deadline=3.0):
    q = cls(capacity)
    total_items = n_producers * items_per_producer
    sentinel = ("__STOP__",)
    lock = threading.Lock()
    consumed = []

    def producer(pid):
        for s in range(items_per_producer):
            q.put((pid, s))

    def consumer():
        while True:
            item = q.get()
            if item == sentinel:
                return
            with lock:
                consumed.append(item)

    def sentinel_pusher():
        for _ in range(n_consumers):
            q.put(sentinel)

    prod = [threading.Thread(target=producer, args=(p,), daemon=True) for p in range(n_producers)]
    cons = [threading.Thread(target=consumer, daemon=True) for _ in range(n_consumers)]
    for t in cons:
        t.start()
    for t in prod:
        t.start()
    prod_ok = _join_all_with_deadline(prod, deadline)
    sp = threading.Thread(target=sentinel_pusher, daemon=True)
    sp.start()
    sp_ok = _join_all_with_deadline([sp], deadline)
    cons_ok = _join_all_with_deadline(cons, deadline)

    expected = {(p, s) for p in range(n_producers) for s in range(items_per_producer)}
    consumed_set = set(consumed)
    ok = (prod_ok and sp_ok and cons_ok and len(consumed) == total_items == len(consumed_set)
          and consumed_set == expected)
    return ok


def test_correct_reference_passes_blocking_and_stress_scenarios():
    for _ in range(3):
        assert _blocks_when_full(_CorrectQueue)
    for _ in range(3):
        assert _stress_no_loss(_CorrectQueue)


def test_no_capacity_reference_reliably_fails_blocking_check():
    failures = sum(1 for _ in range(5) if not _blocks_when_full(_NoCapacityQueue))
    assert failures == 5, (
        "calibration invariant violated: a queue that never enforces capacity on put() "
        "passed the blocking check at least once -- the check is not reliable"
    )


def test_lock_held_while_waiting_reference_reliably_deadlocks():
    failures = sum(1 for _ in range(3) if not _blocks_when_full(_LockHeldWhileWaitingQueue))
    assert failures == 3, (
        "calibration invariant violated: a queue that holds its lock while waiting did not "
        "deadlock every trial -- the deadlock-detection check is not reliable"
    )
    assert not _stress_no_loss(_LockHeldWhileWaitingQueue, deadline=2.0)


def test_embedded_test_file_uses_bounded_deadline_joins_not_bare_join():
    content = c25.get_grading_payload()["tests"][c25._TEST_FILE_PATH]
    assert "_join_all_with_deadline" in content
    # A bare `.join()` with no timeout anywhere in the canonical file would let a genuinely
    # deadlocked submission hang the whole grading run instead of failing fast.
    assert re.search(r"\.join\(\)", content) is None


def test_embedded_test_file_never_names_a_correct_implementation():
    content = c25.get_grading_payload()["tests"][c25._TEST_FILE_PATH]
    assert "class BoundedQueue" not in content
    assert "def put(self" not in content
    assert "def get(self" not in content
    assert "threading.Condition" not in content


def test_keystone_ids_reference_real_test_functions():
    content = c25.get_grading_payload()["tests"][c25._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c25.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c25._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_non_contending_sanity_check():
    assert f"{c25._TEST_FILE_PATH}::test_basic_single_threaded_fifo_order" not in c25.KEYSTONE_TEST_IDS


def test_keystone_is_exactly_the_three_discriminating_checks():
    assert set(c25.KEYSTONE_TEST_IDS) == {
        f"{c25._TEST_FILE_PATH}::test_put_blocks_when_full",
        f"{c25._TEST_FILE_PATH}::test_get_blocks_when_empty",
        f"{c25._TEST_FILE_PATH}::test_stress_no_lost_or_duplicated_items_under_heavy_load",
    }


def test_visibility_is_hidden():
    assert c25.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c25.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c25.get_grading_payload()
    assert payload["tests"] == {c25._TEST_FILE_PATH: c25._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "bounded_queue",
        "class": "BoundedQueue",
        "methods": ["put", "get", "qsize"],
    }
    assert payload["keystone_test_ids"] == c25.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c25.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "bounded_queue.py" in leaf["instruction"]
    assert "Condition" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["bounded_queue.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_answer_code():
    # Naming the required class/methods in prose is expected (see c24's identical check for
    # why) -- what must never appear is literal working implementation source.
    instruction = c25.get_compiled_plan()["leaves"][0]["instruction"]
    assert "def put(self" not in instruction
    assert "def get(self" not in instruction
    assert "self._items = []" not in instruction
    assert "self._not_full" not in instruction


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c25", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c25"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c25.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c25.get_compiled_plan()

    assert (private / c25._TEST_FILE_PATH).read_text() == c25._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c25._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c25.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
