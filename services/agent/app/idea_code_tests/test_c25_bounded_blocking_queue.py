"""
codebench task c25 — hard/hidden, concurrency (bounded blocking producer-consumer queue).

Second concurrency-shaped codebench task (see test_c24_call_coalescer.py for the
first). Where c24 is about single-flight call coalescing, c25 is about a bounded buffer with
genuine BLOCKING semantics: put() must wait while full, get() must wait while empty, and
neither producer nor consumer may ever observe a lost or duplicated item.

Authoring finding (see this task's own offline validator for the full calibration evidence,
mirroring c24's): a queue built on a plain Python list has essentially the SAME "GIL rarely
interrupts a straight-line non-yielding statement sequence" property discovered while
authoring c24, which means a naive check-then-mutate race on `len(self._items)` almost never
actually corrupts data in practice on either python3.12 (this host) or python:3.10-slim (the
actual codebench sandbox image) -- so this task does NOT lean on that kind of subtle,
probabilistic race for its primary signal. Instead its two most reliable, 100%-reproducible
failure modes (confirmed live, see the offline test) are both DETERMINISTIC and caught via
plain thread-join-with-timeout, no scheduling luck required at all:
  1. "bounded" in name only: a `put()` that never actually blocks once the queue is full
     (e.g. an unbounded list with no capacity check at all) -- caught by
     test_put_blocks_when_full firing the moment the blocking assertion is checked.
  2. Holding the queue's lock WHILE waiting for space/items (a very plausible mistake for
     someone who hasn't internalized why threading.Condition exists, e.g. writing
     `with self._lock: while full: time.sleep(x)`) -- this deadlocks the ENTIRE queue
     immediately (the thread that should free/fill a slot can never acquire the same lock),
     caught by every one of this task's tests via their bounded join-timeout.
A correct Condition-variable-based implementation (release the lock while waiting, re-check
the predicate after waking) passes every test here in a fraction of a second; both broken
references above were confirmed live to fail 100% of trials (see offline validator).
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_bounded_queue.py"

_TEST_FILE_CONTENT = '''\
import threading
import time

import pytest
from bounded_queue import BoundedQueue


def _join_all_with_deadline(threads, deadline_s, poll=0.02):
    """Wait for every thread in `threads` to finish, but NEVER longer than deadline_s total
    (regardless of how many threads there are) -- a queue that deadlocks must make this
    test fail fast, not hang for minutes."""
    end = time.time() + deadline_s
    while time.time() < end:
        if all(not t.is_alive() for t in threads):
            return True
        time.sleep(poll)
    return all(not t.is_alive() for t in threads)


def test_basic_single_threaded_fifo_order():
    q = BoundedQueue(capacity=3)
    q.put("a")
    q.put("b")
    assert q.qsize() == 2
    assert q.get() == "a"
    assert q.get() == "b"
    assert q.qsize() == 0


def test_put_blocks_when_full():
    q = BoundedQueue(capacity=2)
    q.put("a")
    q.put("b")
    assert q.qsize() == 2

    done = threading.Event()

    def putter():
        q.put("c")
        done.set()

    t = threading.Thread(target=putter, daemon=True)
    t.start()
    # A correctly-bounded queue must NOT have accepted "c" yet -- there is no room.
    assert not done.wait(timeout=0.3), "put() did not block on a full queue"
    assert q.qsize() == 2

    # Free a slot from a SEPARATE thread with its own bounded timeout, so a queue that
    # deadlocks (e.g. holds its lock while "waiting") fails this test fast instead of
    # hanging the whole suite.
    result = {}

    def getter():
        result["item"] = q.get()

    g = threading.Thread(target=getter, daemon=True)
    g.start()
    assert _join_all_with_deadline([g], 3.0), "get() deadlocked on a non-empty queue"
    assert result["item"] == "a"

    assert _join_all_with_deadline([t], 3.0), "put() never unblocked after space was freed"
    assert done.is_set()
    assert q.qsize() == 2


def test_get_blocks_when_empty():
    q = BoundedQueue(capacity=2)
    result = {}
    done = threading.Event()

    def getter():
        result["item"] = q.get()
        done.set()

    g = threading.Thread(target=getter, daemon=True)
    g.start()
    assert not done.wait(timeout=0.3), "get() did not block on an empty queue"

    p = threading.Thread(target=lambda: q.put("x"), daemon=True)
    p.start()
    assert _join_all_with_deadline([p], 3.0), "put() deadlocked on a non-full queue"
    assert _join_all_with_deadline([g], 3.0), "get() never unblocked after an item was put"
    assert result["item"] == "x"


def _run_stress_scenario(n_producers=6, items_per_producer=60, n_consumers=6, capacity=5,
                          deadline=3.0):
    q = BoundedQueue(capacity)
    total_items = n_producers * items_per_producer
    sentinel = ("__STOP__",)
    lock = threading.Lock()
    consumed = []

    def producer(pid):
        for seq in range(items_per_producer):
            q.put((pid, seq))

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

    prod_threads = [threading.Thread(target=producer, args=(p,), daemon=True)
                     for p in range(n_producers)]
    cons_threads = [threading.Thread(target=consumer, daemon=True) for _ in range(n_consumers)]

    for t in cons_threads:
        t.start()
    for t in prod_threads:
        t.start()

    producers_ok = _join_all_with_deadline(prod_threads, deadline)

    sp = threading.Thread(target=sentinel_pusher, daemon=True)
    sp.start()
    sentinels_ok = _join_all_with_deadline([sp], deadline)

    consumers_ok = _join_all_with_deadline(cons_threads, deadline)

    expected = {(p, s) for p in range(n_producers) for s in range(items_per_producer)}
    consumed_set = set(consumed)
    return {
        "producers_ok": producers_ok,
        "sentinels_ok": sentinels_ok,
        "consumers_ok": consumers_ok,
        "total_items": total_items,
        "n_consumed": len(consumed),
        "n_consumed_unique": len(consumed_set),
        "lost": len(expected - consumed_set),
    }


def test_stress_no_lost_or_duplicated_items_under_heavy_load():
    r = _run_stress_scenario()
    assert r["producers_ok"], "a producer thread never finished (deadlock?)"
    assert r["sentinels_ok"], "could not drain the queue with sentinels (deadlock?)"
    assert r["consumers_ok"], "a consumer thread never finished (deadlock?)"
    assert r["lost"] == 0, f"{r['lost']} item(s) were never delivered to any consumer"
    assert r["n_consumed"] == r["total_items"], (
        f"expected exactly {r['total_items']} items consumed, got {r['n_consumed']} "
        "(items were lost or duplicated)"
    )
    assert r["n_consumed_unique"] == r["n_consumed"], "some item was delivered more than once"
'''

# The blocking-discipline tests and the heavy-load stress test are the core of this task --
# gate the score. The basic single-threaded FIFO sanity check is supporting/bonus credit
# only: it never actually contends, so a naively-locked (or even unlocked) implementation
# could pass it by accident.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_put_blocks_when_full",
    f"{_TEST_FILE_PATH}::test_get_blocks_when_empty",
    f"{_TEST_FILE_PATH}::test_stress_no_lost_or_duplicated_items_under_heavy_load",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c25",
        "title": "bounded-blocking-queue",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `bounded_queue.py` that defines a class `BoundedQueue` "
        "implementing a fixed-capacity, thread-safe, BLOCKING FIFO queue for producer/"
        "consumer use.\n\n"
        "Implement exactly this API:\n"
        "- `BoundedQueue(capacity: int)` — construct an empty queue that will never hold "
        "more than `capacity` items at once.\n"
        "- `put(self, item) -> None` — add `item` to the back of the queue. If the queue "
        "is already at capacity, this call must BLOCK (wait) until another thread removes "
        "an item and makes room — it must NOT raise, drop the item, or silently exceed "
        "capacity.\n"
        "- `get(self) -> object` — remove and return the item at the front of the queue "
        "(FIFO order — the item that has been waiting longest comes out first). If the "
        "queue is empty, this call must BLOCK (wait) until another thread puts an item — "
        "it must NOT raise or return a placeholder value.\n"
        "- `qsize(self) -> int` — return the current number of items in the queue "
        "(a thread-safe snapshot; it's fine if it's stale by the time the caller reads "
        "it).\n\n"
        "This must be genuinely correct under real concurrent load: many threads may call "
        "`put`/`get` on the SAME `BoundedQueue` instance at the same time. No item may "
        "ever be lost, silently duplicated, or corrupted, no matter how many producer and "
        "consumer threads are running concurrently or how the OS happens to schedule them. "
        "A naive 'check the size, then mutate' approach without a lock spanning both steps "
        "is not safe. Blocking must be implemented properly (e.g. with "
        "`threading.Condition`) — a thread that is 'waiting' must not be holding a lock "
        "that would prevent the very thread it's waiting on from making progress (that "
        "would deadlock the whole queue).\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "queue yourself before finishing: construct a small BoundedQueue, put a few items "
        "and get them back out in FIFO order; from a single thread, confirm calling get() "
        "on an empty queue or put() on a full queue would block (you can test this with a "
        "background thread and a short timeout — if a call returns instantly when it "
        "should have blocked, or never returns at all once space/an item becomes "
        "available, something is wrong). Fix issues with patch_file/run_python until "
        "you're confident, then finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "bounded_queue",
            "class": "BoundedQueue",
            "methods": ["put", "get", "qsize"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write bounded_queue.py defining a class BoundedQueue implementing a "
                    "fixed-capacity, thread-safe, BLOCKING FIFO queue. Required API: "
                    "`BoundedQueue(capacity)` (empty queue, never holds more than "
                    "`capacity` items); `put(self, item)` (adds to the back; if already at "
                    "capacity, BLOCKS/waits until a slot frees up -- must not raise, drop "
                    "the item, or exceed capacity); `get(self)` (removes and returns the "
                    "front item, FIFO order; if empty, BLOCKS/waits until an item is "
                    "available -- must not raise or return a placeholder); `qsize(self)` "
                    "(current item count, thread-safe snapshot). Implement blocking "
                    "correctly with threading.Condition (or an equivalent primitive): a "
                    "waiting thread must release its lock WHILE waiting, not hold it -- "
                    "holding a lock across a wait loop deadlocks the whole queue, since the "
                    "thread that needs to make room/add an item can never acquire that same "
                    "lock. Under real concurrent load from many producer AND consumer "
                    "threads at once, no item may ever be lost, duplicated, or corrupted, "
                    "regardless of scheduling. Use write_file to create bounded_queue.py. "
                    "Since there is no visible test file, use run_python to sanity-check: "
                    "basic FIFO put/get order from a single thread; then verify blocking "
                    "behavior using a couple of background threads with short timeouts -- "
                    "put()/get() should NOT return instantly when they should be blocked, "
                    "and SHOULD return promptly once the blocking condition is resolved by "
                    "another thread. Fix issues with patch_file/run_python until confident, "
                    "then finish."
                ),
                "expect": "bounded_queue.py written, defining BoundedQueue with "
                          "put/get/qsize, correctly blocking via a Condition-style "
                          "primitive, sanity-checked with run_python",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm bounded_queue.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["bounded_queue.py"]},
    }
