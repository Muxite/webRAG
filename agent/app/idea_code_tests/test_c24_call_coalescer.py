"""
codebench task c24 — hard/visible, concurrency (single-flight call coalescing).

REPLACEMENT NOTE (2026-08-06/07): this task used to be a thread-safe shared counter
(`SafeCounter.increment`/`decrement`, one shared lock held across each compound
read-modify-write). Across TWO separate live-calibration rounds -- including a V2 hardening
pass that stripped every mechanism word from the prompt and added a second, distinctly-shaped
compound operation specifically to defeat a naive "one lock" fix -- qwen2.5:14b scored 1.0/1.0
on BOTH the badmodel and Aider codebench harnesses both times. Inspecting the actual submitted
solution confirmed it was not a fluke or a leak: a fully correct, textbook `SafeCounter` with
one shared `Lock` held across the whole compound operation for both methods. Thread-safe
counters (and bank-account-style increment/decrement with a shared lock) are among the most
canonical, over-represented concurrency exercises in existence -- there is no "obvious but
wrong" intermediate step for a competent model to fall into; it just goes straight to correct,
regardless of how the prompt is worded. This is a full replacement, same id/category
conventions ("hard", concurrency), a genuinely different mechanism and a genuinely different
trap.

The new task: implement request/call COALESCING for a keyed set of expensive calls -- given a
`key` and a callable `fn` with no arguments, at most one call to `fn()` may be in flight AT
A TIME for that `key`; any other caller that shows up while a call for that same `key` is
already running must NOT invoke `fn()` again -- it must instead wait for the ALREADY-RUNNING
call to finish and receive that exact same outcome (return value, or the same exception,
re-raised). Once that call finishes, the key becomes available again: the very next caller for
that key (as long as nothing else is concurrently in flight for it right this moment) starts a
brand new call to `fn()`, from scratch -- this is a coalescer for concurrent DUPLICATE work,
never a permanent cache.

Why this shape resists memorized pattern-matching where thread-safe-counter did not: it is a
real, documented production pattern (request coalescing / duplicate-suppression, used to
protect a slow origin/backend from a "thundering herd" of simultaneous identical requests), but
it is NOT a canonical from-scratch toy coding exercise the way a shared counter or a
producer/consumer queue is -- in practice almost everyone either reaches for a library or never
implements the leader/follower result-sharing logic by hand. Crucially, the training-data
reflex this task's own author-brief calls for ("share a mutable resource across threads -> put
a lock around it") is NECESSARY here (you do need to protect the bookkeeping dict) but is NOT
SUFFICIENT -- a fully plausible, structurally-sound-looking implementation can apply exactly
that reflex (give each key its own lock, have every caller for that key acquire the lock and
then call `fn()` itself once inside it) and end up with something that is perfectly race-free
and crash-free, yet is completely wrong for the stated contract: it serializes duplicate work
instead of sharing a single result, so `fn()` still gets called once PER CALLER instead of once
per truly-concurrent batch. This is the exact kind of "gets the structure right, silently wrong
on the interaction the spec actually cares about" trap the c22 replacement (conditional-ETag
resource store) demonstrated works, ported to the concurrency category. A second, independent
near-miss -- correctly sharing the in-flight result among concurrent callers, but never
clearing the bookkeeping entry once the call finishes -- turns the coalescer into a permanent,
un-invalidating cache instead of a coalescer for truly-simultaneous callers only.

Ground truth is a set of observable invariants (never a numeric sequence to hand-derive),
verified three ways, exactly as the task-authoring brief's calibration bar requires: (a) a
hand-written reference implementation (Event-based leader/follower signaling, entry cleared on
every completion path including exceptions) passes every canonical test; (b) a "protect
duplicate work with a per-key lock, but every caller still invokes fn() itself" mutant --
the single most plausible near-miss for a model reflexively applying the
counter-style-lock-around-a-critical-section pattern -- was actually run live against the
canonical 11-test suite (not just reasoned about) and FAILS 8 of 11 tests deterministically,
including every coalescing-count and shared-result assertion (not via a rare race window: with
N concurrent callers it calls fn() exactly N times, every time, on every run, since that is
what the mutant's code structurally always does -- there is no timing sensitivity to tune
here, unlike the old counter task's race-window flakiness); only the three purely-sequential,
no-real-contention sanity checks survive, since this mutant is genuinely, broadly wrong for the
stated contract, not merely wrong at one narrow edge; (c) a second, more surgical mutant that
coalesces and shares results correctly but never clears the bookkeeping entry after completion
was likewise run live and passes all 6 keystone tests except exactly ONE -- it deterministically
fails "the key must be available again afterward" (and that test's two purely-sequential bonus
corollaries), always returning the first batch's stale result forever, while every genuinely
concurrent coalescing assertion still passes. This second mutant is the narrow, single-keystone
near-miss; the first is a broad, nearly-total failure -- both were measured directly rather than
assumed, and both figures are reproduced in
agent/tests/idea_code_test_c24_test.py. This replacement has not yet been through a
live-calibration round against
qwen2.5:14b (offline mutation testing above is the authoring-time evidence; live calibration
against both the badmodel and aider harnesses is the coordinating session's next step, per its
own instructions, not run from here).
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_call_coalescer.py"

_TEST_FILE_CONTENT = '''\
import threading
import time

import pytest
from call_coalescer import CallCoalescer


N_THREADS = 25
LEADER_SLEEP_S = 0.05


class _CountingFn:
    """A slow, thread-safe call counter used as the `fn` argument below. Each call sleeps
    briefly (simulating real work -- e.g. an origin fetch) before returning a fresh,
    monotonically increasing "result-N" string -- so two DISTINCT calls to this object always
    return two DIFFERENT strings. That is what lets these tests detect whether the submission
    is really coalescing concurrent callers into a single underlying call, or secretly
    invoking fn() once per caller (or forever replaying one stale result)."""

    def __init__(self, sleep_s=LEADER_SLEEP_S):
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self.call_count = 0

    def __call__(self):
        with self._lock:
            self.call_count += 1
            my_number = self.call_count
        time.sleep(self._sleep_s)
        return f"result-{my_number}"


class _RaisingFn:
    def __init__(self, sleep_s=LEADER_SLEEP_S):
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self.call_count = 0

    def __call__(self):
        with self._lock:
            self.call_count += 1
        time.sleep(self._sleep_s)
        raise ValueError("boom")


def _run_concurrent_batch(coalescer, key, fn, n_threads=N_THREADS):
    barrier = threading.Barrier(n_threads)
    results = [None] * n_threads
    errors = [None] * n_threads

    def worker(idx):
        barrier.wait()
        try:
            results[idx] = coalescer.run(key, fn)
        except Exception as e:  # noqa: BLE001 - captured for the test to inspect
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    for t in threads:
        assert not t.is_alive(), "a worker thread never finished -- possible deadlock"
    return results, errors


def test_basic_single_call_returns_value_and_invokes_fn_once():
    coalescer = CallCoalescer()
    fn = _CountingFn()
    assert coalescer.run("k", fn) == "result-1"
    assert fn.call_count == 1


def test_basic_sequential_calls_reinvoke_fn_every_time():
    coalescer = CallCoalescer()
    fn = _CountingFn()
    first = coalescer.run("k", fn)
    second = coalescer.run("k", fn)
    third = coalescer.run("k", fn)
    assert [first, second, third] == ["result-1", "result-2", "result-3"]
    assert fn.call_count == 3


def test_two_different_keys_do_not_share_state():
    coalescer = CallCoalescer()
    fn_a, fn_b = _CountingFn(), _CountingFn()
    assert coalescer.run("a", fn_a) == "result-1"
    assert coalescer.run("b", fn_b) == "result-1"
    assert fn_a.call_count == 1
    assert fn_b.call_count == 1


def test_concurrent_batch_shares_a_single_fn_invocation_trial_1():
    coalescer = CallCoalescer()
    fn = _CountingFn()
    _results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert errors == [None] * N_THREADS
    assert fn.call_count == 1


def test_concurrent_batch_shares_a_single_fn_invocation_trial_2():
    coalescer = CallCoalescer()
    fn = _CountingFn()
    _results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert errors == [None] * N_THREADS
    assert fn.call_count == 1


def test_concurrent_batch_all_waiters_receive_the_identical_leader_result():
    coalescer = CallCoalescer()
    fn = _CountingFn()
    results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert errors == [None] * N_THREADS
    assert results == ["result-1"] * N_THREADS


def test_key_is_cleared_after_a_concurrent_batch_for_a_fresh_call():
    coalescer = CallCoalescer()
    fn = _CountingFn()
    batch_results, batch_errors = _run_concurrent_batch(coalescer, "k", fn)
    assert batch_errors == [None] * N_THREADS
    assert fn.call_count == 1
    # A call AFTER the whole batch has joined -- nothing is concurrently in flight for "k"
    # any more -- must invoke fn() fresh, not replay the batch's cached result forever.
    followup = coalescer.run("k", fn)
    assert fn.call_count == 2
    assert followup == "result-2"
    assert followup not in batch_results


def test_concurrent_exception_propagates_to_every_waiter_and_leader_runs_once():
    coalescer = CallCoalescer()
    fn = _RaisingFn()
    results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert results == [None] * N_THREADS
    assert fn.call_count == 1
    for err in errors:
        assert isinstance(err, ValueError)
        assert str(err) == "boom"


def test_key_is_cleared_after_an_exception_for_a_fresh_successful_call():
    coalescer = CallCoalescer()
    raising_fn = _RaisingFn()
    _run_concurrent_batch(coalescer, "k", raising_fn)
    assert raising_fn.call_count == 1
    ok_fn = _CountingFn()
    assert coalescer.run("k", ok_fn) == "result-1"
    assert ok_fn.call_count == 1


def test_two_keys_coalesce_independently_under_concurrent_mixed_load():
    coalescer = CallCoalescer()
    fn_a, fn_b = _CountingFn(), _CountingFn()
    half = N_THREADS // 2
    barrier = threading.Barrier(half * 2)
    results = [None] * (half * 2)

    def worker(idx, key, fn):
        barrier.wait()
        results[idx] = coalescer.run(key, fn)

    threads = []
    for i in range(half):
        threads.append(threading.Thread(target=worker, args=(i, "a", fn_a)))
    for i in range(half):
        threads.append(threading.Thread(target=worker, args=(half + i, "b", fn_b)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    for t in threads:
        assert not t.is_alive(), "a worker thread never finished -- possible deadlock"

    assert fn_a.call_count == 1
    assert fn_b.call_count == 1
    assert results[:half] == ["result-1"] * half
    assert results[half:] == ["result-1"] * half


def test_repeated_cycles_of_concurrent_batches_stay_correct():
    coalescer = CallCoalescer()
    for _cycle in range(3):
        fn = _CountingFn()
        results, errors = _run_concurrent_batch(coalescer, "k", fn, n_threads=10)
        assert errors == [None] * 10
        assert fn.call_count == 1
        assert results == ["result-1"] * 10
'''

# The concurrent-coalescing invariants are the entire point of this task; they gate the score:
# sharing exactly one fn() call across a truly-concurrent batch (two repeated trials, since
# thread scheduling means any one run is worth confirming twice), every waiter receiving the
# identical shared result (catches the per-caller-invokes-fn-itself near-miss), the key
# becoming available again for a fresh call once the batch has fully joined (catches the
# never-clears-the-entry near-miss), concurrent exception propagation to every waiter with the
# leader itself only running once, and two keys coalescing independently under mixed
# concurrent load. The basic single-threaded cases, the sequential-reinvoke sanity check, the
# clears-after-exception corollary, and the repeated-cycles reuse check are supporting/bonus
# credit: real behavior, but none of them requires genuine concurrent coalescing to pass by
# accident the way the keystone set does -- see the module docstring's mutant analysis.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_concurrent_batch_shares_a_single_fn_invocation_trial_1",
    f"{VISIBLE_TEST_PATH}::test_concurrent_batch_shares_a_single_fn_invocation_trial_2",
    f"{VISIBLE_TEST_PATH}::test_concurrent_batch_all_waiters_receive_the_identical_leader_result",
    f"{VISIBLE_TEST_PATH}::test_key_is_cleared_after_a_concurrent_batch_for_a_fresh_call",
    f"{VISIBLE_TEST_PATH}::test_concurrent_exception_propagates_to_every_waiter_and_leader_runs_once",
    f"{VISIBLE_TEST_PATH}::test_two_keys_coalesce_independently_under_concurrent_mixed_load",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c24",
        "title": "call-coalescer",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `call_coalescer.py` that defines a class `CallCoalescer` for "
        "COALESCING duplicate concurrent work -- when many threads all want the result of the "
        "same expensive operation at (nearly) the same time, only ONE of them should actually "
        "do the work; the rest should share that single outcome instead of each redoing it "
        "themselves.\n\n"
        "Implement:\n"
        "- `CallCoalescer()` — construct an empty coalescer.\n"
        "- `run(self, key, fn)` — `fn` is a callable that takes no arguments. The required "
        "behavior:\n"
        "  - At any given moment, at most ONE call to `fn()` may be in progress for a given "
        "`key`. If a call for `key` is already in progress (started by some other thread and "
        "not yet finished) when `run` is called again with the same `key`, the new caller must "
        "NOT invoke `fn()` a second time -- instead it must wait for the ALREADY-IN-PROGRESS "
        "call to finish, and then return exactly what that call returned (or, if that call "
        "raised an exception, raise an exception of the same type with the same message, "
        "rather than returning anything).\n"
        "  - This applies transitively to any number of concurrent callers: if 50 threads all "
        "call `run(\"k\", fn)` at close to the same moment while nothing for `\"k\"` is "
        "currently in progress, `fn()` must be invoked exactly ONCE, and all 50 threads must "
        "receive that one call's result (or its exception).\n"
        "  - Once the in-progress call for `key` finishes (successfully or with an exception), "
        "`key` becomes available again: the very next call to `run(key, ...)` — as long as "
        "nothing else happens to be concurrently in progress for `key` at that exact moment — "
        "must invoke `fn()` fresh, from scratch. `CallCoalescer` coalesces truly-simultaneous "
        "duplicate work; it must NEVER become a permanent cache that keeps replaying one old "
        "result forever after the call that produced it has already finished.\n"
        "  - Two different `key` values must be completely independent of each other: an "
        "in-progress call for one key must never block, delay, or share a result with a call "
        "for a different key.\n\n"
        "A visible test file already exists at tests/test_call_coalescer.py (run_pytest). It "
        "launches many threads at once against a slow `fn` (via a shared `threading.Barrier`, "
        "so they all arrive at `run()` at essentially the same moment) and checks: how many "
        "times the underlying `fn` was actually invoked, whether every concurrent caller got "
        "back the identical result, and whether a call made strictly AFTER a whole batch has "
        "finished gets a fresh new result rather than a stale cached one. It also checks that "
        "an exception raised by `fn()` reaches every concurrent caller, and that two different "
        "keys under concurrent load never interfere with each other. Correctly protecting your "
        "own internal bookkeeping from concurrent access is necessary here, but on its own it "
        "is NOT sufficient -- serializing duplicate callers so they each safely take a turn "
        "calling `fn()` themselves is a different behavior than making them share ONE call's "
        "result, and the tests are specifically built to tell those two apart. Keep revising "
        "call_coalescer.py until every test in tests/test_call_coalescer.py passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "call_coalescer",
            "class": "CallCoalescer",
            "methods": ["run"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write call_coalescer.py defining a class CallCoalescer with run(self, "
                    "key, fn) -> Any, where fn is a zero-argument callable, that COALESCES "
                    "duplicate concurrent work keyed by `key`. Required behavior: at any "
                    "moment at most one call to fn() may be in progress for a given key; any "
                    "other thread calling run() with the same key while that call is still in "
                    "progress must NOT invoke fn() itself -- it must wait for the in-progress "
                    "call to finish and then receive that exact same outcome (its return "
                    "value, or the same exception re-raised) instead of computing its own. "
                    "This must work for any number of simultaneous callers sharing one key: "
                    "if N threads all call run(key, fn) at close to the same moment while "
                    "nothing is in progress for key yet, fn() must be invoked exactly once "
                    "and all N callers receive that one outcome. Once the in-progress call for "
                    "a key finishes (success or exception), the key must become available "
                    "again -- the next call to run(key, ...), once nothing is concurrently in "
                    "progress for it, must invoke fn() fresh rather than replaying an old "
                    "result forever; this is a coalescer for truly-simultaneous duplicate "
                    "work, not a permanent cache. Different keys must be fully independent "
                    "under concurrent access. Protecting your own bookkeeping from concurrent "
                    "access safely is necessary but not sufficient on its own -- merely taking "
                    "turns (each caller safely but separately invoking fn() itself, one after "
                    "another) is a different, unwanted behavior from making concurrent callers "
                    "share a single call's outcome; make sure your implementation genuinely "
                    "does the latter, not the former. Use write_file to create "
                    "call_coalescer.py, then use run_pytest on tests/test_call_coalescer.py "
                    "-- it stress-tests your implementation with many threads launched at "
                    "once via a barrier, checking the underlying call count, whether every "
                    "concurrent caller received the identical shared result, whether a call "
                    "made after a batch has fully finished gets a fresh result, concurrent "
                    "exception propagation, and independence between different keys. If any "
                    "test fails, use read_file/patch_file to fix call_coalescer.py and "
                    "run_pytest again. Once every test passes, finish."
                ),
                "expect": "call_coalescer.py written; tests/test_call_coalescer.py fully "
                          "passes, including the concurrent single-invocation coalescing, "
                          "shared-result, key-clears-after-completion, and concurrent "
                          "exception-propagation tests",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm call_coalescer.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["call_coalescer.py"]},
    }
