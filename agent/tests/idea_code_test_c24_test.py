"""
Adversarial offline checks for codebench task c24 (call-coalescer) -- no Docker (besides the
materialize_task subprocess check), no LLM.

Mirrors idea_code_test_c22_test.py's structure for this task family: independently re-derive
ground truth (a SEPARATE, differently-coded reference implementation and TWO separately-coded
plausible near-miss mutants, never importing the task module's own embedded test file's
helper classes), then separately confirm via string search that the scenarios these mutants
probe are what's actually embedded in the canonical test file, then prove -- by actually
running pytest against each variant, not by asserting from authorial confidence -- that the
canonical suite's keystone set discriminates a "structurally looks reasonable" implementation
from a genuinely correct one.

c24 is the second CONCURRENCY-shaped codebench task (after the counter/lock task this module
replaces). "Ground truth" here is not a numeric sequence to hand-derive -- it is the
discriminating power of the embedded stress tests themselves, exactly as for the task this one
replaces. Unlike that task's race-window-timing-sensitive mutants (which needed repeated
trials and generous margins to reliably manifest, and were the subject of extensive tuning
documented in that task's own history), the two mutants here fail (or pass) DETERMINISTICALLY
-- their wrongness (or correctness) follows directly from what their code structurally always
does on every run, not from winning or losing a race window. This was a deliberate design
choice for this replacement (see the task module's own docstring) specifically to avoid the
flakiness class that made the counter task's calibration so fragile.
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

from agent.app.idea_code_tests import test_c24_call_coalescer as c24


# ---------------------------------------------------------------------------
# Independent re-derivation: a reference implementation and two plausible near-miss mutants,
# each written from scratch here (not imported from the task module, which ships no
# implementation of its own at all -- c24 is a "write it from spec" visible task).
# ---------------------------------------------------------------------------

N_THREADS = 25
LEADER_SLEEP_S = 0.05


class _CountingFn:
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
        except Exception as e:  # noqa: BLE001
            errors[idx] = e

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    for t in threads:
        assert not t.is_alive(), "a worker thread never finished -- possible deadlock"
    return results, errors


class _ReferenceCoalescer:
    """Independently written correct reference: Event-based leader/follower signaling,
    bookkeeping entry cleared on every completion path (success or exception)."""

    class _Call:
        def __init__(self):
            self.event = threading.Event()
            self.value = None
            self.error = None

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = {}

    def run(self, key, fn):
        with self._lock:
            call = self._in_flight.get(key)
            if call is not None:
                is_leader = False
            else:
                call = self._Call()
                self._in_flight[key] = call
                is_leader = True
        if not is_leader:
            call.event.wait()
            if call.error is not None:
                raise call.error
            return call.value
        try:
            value = fn()
        except Exception as exc:
            call.error = exc
            with self._lock:
                if self._in_flight.get(key) is call:
                    del self._in_flight[key]
            call.event.set()
            raise
        else:
            call.value = value
            with self._lock:
                if self._in_flight.get(key) is call:
                    del self._in_flight[key]
            call.event.set()
            return value


class _PerKeyLockSerializeMutant:
    """The single most plausible near-miss: applies the "protect a shared resource with a
    lock" reflex, but each caller still invokes fn() itself once it gets the lock -- correctly
    race-free, but serializes duplicate work instead of sharing one result."""

    def __init__(self):
        self._meta_lock = threading.Lock()
        self._locks = {}

    def _get_lock(self, key):
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def run(self, key, fn):
        lock = self._get_lock(key)
        with lock:
            return fn()


class _NeverClearsMutant:
    """Correctly coalesces AND shares the in-flight result, but never removes the bookkeeping
    entry once the call finishes -- becomes a permanent memoizing cache instead of a
    coalescer for truly-simultaneous callers only."""

    class _Call:
        def __init__(self):
            self.event = threading.Event()
            self.value = None
            self.error = None

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = {}

    def run(self, key, fn):
        with self._lock:
            call = self._in_flight.get(key)
            if call is not None:
                is_leader = False
            else:
                call = self._Call()
                self._in_flight[key] = call
                is_leader = True
        if not is_leader:
            call.event.wait()
            if call.error is not None:
                raise call.error
            return call.value
        try:
            value = fn()
        except Exception as exc:
            call.error = exc
            call.event.set()  # BUG: never removes self._in_flight[key]
            raise
        else:
            call.value = value
            call.event.set()  # BUG: never removes self._in_flight[key]
            return value


def test_ground_truth_reference_coalesces_a_concurrent_batch_to_one_call():
    coalescer = _ReferenceCoalescer()
    fn = _CountingFn()
    results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert errors == [None] * N_THREADS
    assert fn.call_count == 1
    assert results == ["result-1"] * N_THREADS


def test_ground_truth_reference_reinvokes_fn_after_batch_completes():
    coalescer = _ReferenceCoalescer()
    fn = _CountingFn()
    _run_concurrent_batch(coalescer, "k", fn)
    assert fn.call_count == 1
    followup = coalescer.run("k", fn)
    assert fn.call_count == 2
    assert followup == "result-2"


def test_ground_truth_reference_propagates_exceptions_to_every_waiter():
    coalescer = _ReferenceCoalescer()
    fn = _RaisingFn()
    results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert results == [None] * N_THREADS
    assert fn.call_count == 1
    for err in errors:
        assert isinstance(err, ValueError)
        assert str(err) == "boom"


def test_per_key_lock_serialize_mutant_fails_deterministically_not_by_race_luck():
    """The plausible near-miss must fail the coalescing-count assertion on every independent
    run, since its wrongness follows directly from its structure (every caller calls fn()
    itself), not from winning a timing race."""
    for _trial in range(3):
        coalescer = _PerKeyLockSerializeMutant()
        fn = _CountingFn()
        _results, errors = _run_concurrent_batch(coalescer, "k", fn)
        assert errors == [None] * N_THREADS
        assert fn.call_count == N_THREADS, (
            "calibration invariant violated: the per-key-lock-serialize mutant coalesced "
            "calls it should not have been able to -- it always invokes fn() once per caller "
            "by construction"
        )


def test_never_clears_mutant_coalesces_correctly_but_never_frees_the_key():
    coalescer = _NeverClearsMutant()
    fn = _CountingFn()
    results, errors = _run_concurrent_batch(coalescer, "k", fn)
    assert errors == [None] * N_THREADS
    assert fn.call_count == 1
    assert results == ["result-1"] * N_THREADS
    # the defining bug: a call strictly AFTER the batch has finished still replays the stale
    # cached result instead of invoking fn() again
    followup = coalescer.run("k", fn)
    assert fn.call_count == 1, (
        "calibration invariant violated: the never-clears mutant unexpectedly invoked fn() "
        "again after the batch completed -- it should be permanently stuck on the cached "
        "result for this test to be a valid demonstration of the bug"
    )
    assert followup == "result-1"


# ---------------------------------------------------------------------------
# Task-authoring hygiene: canonical suite content, keystone shape, plan/payload shape, no
# leaked mechanism words, live pytest runs of the reference and both mutants against the
# REAL embedded canonical test file.
# ---------------------------------------------------------------------------


def test_embedded_test_file_defines_the_three_scenario_helpers():
    content = c24.get_grading_payload()["tests"][c24.VISIBLE_TEST_PATH]
    assert "_CountingFn" in content
    assert "_RaisingFn" in content
    assert "threading.Barrier" in content
    assert "N_THREADS = 25" in content


def test_embedded_test_file_never_names_a_correct_implementation():
    # The fixture's own _CountingFn/_RaisingFn helpers legitimately use threading.Lock() to
    # protect THEIR OWN call-count bookkeeping -- that is test-harness plumbing, not a leak of
    # the CallCoalescer solution itself, so it is not checked against here. What must never
    # appear is the actual CallCoalescer class/run() implementation.
    content = c24.get_grading_payload()["tests"][c24.VISIBLE_TEST_PATH]
    assert "class CallCoalescer" not in content
    assert "def run(self" not in content
    assert "_in_flight" not in content
    assert "is_leader" not in content


def test_task_statement_and_plan_do_not_leak_synchronization_mechanism():
    statement = c24.get_task_statement()
    plan = c24.get_compiled_plan()
    instruction = plan["leaves"][0]["instruction"]
    forbidden = ["threading.lock", "threading.event", "threading.condition", "mutex",
                 "semaphore", "gil"]
    for text, label in [(statement, "task statement"), (instruction, "plan instruction")]:
        lowered = text.lower()
        for word in forbidden:
            assert word not in lowered, (
                f"{label} leaks synchronization mechanism {word!r} -- describe the required "
                "BEHAVIOR/INVARIANTS only, never how to implement it"
            )


def test_keystone_ids_reference_real_test_functions():
    content = c24.get_grading_payload()["tests"][c24.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(defined) == 11
    for node_id in c24.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c24.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_non_discriminating_sequential_sanity_checks():
    non_keystone = [
        "test_basic_single_call_returns_value_and_invokes_fn_once",
        "test_basic_sequential_calls_reinvoke_fn_every_time",
        "test_two_different_keys_do_not_share_state",
        "test_key_is_cleared_after_an_exception_for_a_fresh_successful_call",
        "test_repeated_cycles_of_concurrent_batches_stay_correct",
    ]
    for name in non_keystone:
        assert f"{c24.VISIBLE_TEST_PATH}::{name}" not in c24.KEYSTONE_TEST_IDS, name
    assert set(c24.KEYSTONE_TEST_IDS) == {
        f"{c24.VISIBLE_TEST_PATH}::test_concurrent_batch_shares_a_single_fn_invocation_trial_1",
        f"{c24.VISIBLE_TEST_PATH}::test_concurrent_batch_shares_a_single_fn_invocation_trial_2",
        f"{c24.VISIBLE_TEST_PATH}::test_concurrent_batch_all_waiters_receive_the_identical_leader_result",
        f"{c24.VISIBLE_TEST_PATH}::test_key_is_cleared_after_a_concurrent_batch_for_a_fresh_call",
        f"{c24.VISIBLE_TEST_PATH}::test_concurrent_exception_propagates_to_every_waiter_and_leader_runs_once",
        f"{c24.VISIBLE_TEST_PATH}::test_two_keys_coalesce_independently_under_concurrent_mixed_load",
    }


def test_visibility_is_visible():
    assert c24.get_visibility() == "visible"


def test_sandbox_fixture_matches_grading_tests():
    assert c24.get_sandbox_fixture() == {c24.VISIBLE_TEST_PATH: c24._TEST_FILE_CONTENT}


def test_grading_payload_shape():
    payload = c24.get_grading_payload()
    assert payload["tests"][c24.VISIBLE_TEST_PATH] == c24.get_sandbox_fixture()[c24.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {
        "module": "call_coalescer",
        "class": "CallCoalescer",
        "methods": ["run"],
    }
    assert payload["keystone_test_ids"] == c24.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c24.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "call_coalescer.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["call_coalescer.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_answer_code():
    plan = c24.get_compiled_plan()
    instruction = plan["leaves"][0]["instruction"]
    assert "self._lock =" not in instruction
    assert "self._in_flight" not in instruction
    assert "threading.Event()" not in instruction
    assert "threading.Lock()" not in instruction


def _run_pytest_against_impl(tmp_path, impl_source: str):
    (tmp_path / "call_coalescer.py").write_text(impl_source)
    test_content = c24.get_grading_payload()["tests"][c24.VISIBLE_TEST_PATH]
    (tmp_path / "test_call_coalescer.py").write_text(test_content)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", str(tmp_path / "test_call_coalescer.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    return result.stdout + result.stderr


def _failed_test_names(pytest_output: str):
    return set(re.findall(r"FAILED test_call_coalescer\.py::(test_\w+)", pytest_output))


_REFERENCE_IMPL_SOURCE = '''\
import threading


class CallCoalescer:
    class _Call:
        def __init__(self):
            self.event = threading.Event()
            self.value = None
            self.error = None

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = {}

    def run(self, key, fn):
        with self._lock:
            call = self._in_flight.get(key)
            if call is not None:
                is_leader = False
            else:
                call = self._Call()
                self._in_flight[key] = call
                is_leader = True

        if not is_leader:
            call.event.wait()
            if call.error is not None:
                raise call.error
            return call.value

        try:
            value = fn()
        except Exception as exc:
            call.error = exc
            with self._lock:
                if self._in_flight.get(key) is call:
                    del self._in_flight[key]
            call.event.set()
            raise
        else:
            call.value = value
            with self._lock:
                if self._in_flight.get(key) is call:
                    del self._in_flight[key]
            call.event.set()
            return value
'''

_PER_KEY_LOCK_SERIALIZE_MUTANT_SOURCE = '''\
import threading


class CallCoalescer:
    def __init__(self):
        self._meta_lock = threading.Lock()
        self._locks = {}

    def _get_lock(self, key):
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def run(self, key, fn):
        lock = self._get_lock(key)
        with lock:
            return fn()
'''

_NEVER_CLEARS_MUTANT_SOURCE = '''\
import threading


class CallCoalescer:
    class _Call:
        def __init__(self):
            self.event = threading.Event()
            self.value = None
            self.error = None

    def __init__(self):
        self._lock = threading.Lock()
        self._in_flight = {}

    def run(self, key, fn):
        with self._lock:
            call = self._in_flight.get(key)
            if call is not None:
                is_leader = False
            else:
                call = self._Call()
                self._in_flight[key] = call
                is_leader = True
        if not is_leader:
            call.event.wait()
            if call.error is not None:
                raise call.error
            return call.value
        try:
            value = fn()
        except Exception as exc:
            call.error = exc
            call.event.set()
            raise
        else:
            call.value = value
            call.event.set()
            return value
'''


def test_reference_implementation_passes_every_canonical_test(tmp_path):
    output = _run_pytest_against_impl(tmp_path, _REFERENCE_IMPL_SOURCE)
    assert "11 passed" in output, output


def test_per_key_lock_serialize_mutant_fails_broadly_against_the_real_canonical_file(tmp_path):
    """Run against the ACTUAL embedded canonical test file (not the validator's own
    reimplementation above) -- confirms this mutant fails every keystone plus the
    keystone-adjacent bonus cases that also depend on real coalescing, and nothing else."""
    output = _run_pytest_against_impl(tmp_path, _PER_KEY_LOCK_SERIALIZE_MUTANT_SOURCE)
    failed = _failed_test_names(output)
    keystone_names = {node_id.split("::")[1] for node_id in c24.KEYSTONE_TEST_IDS}
    assert keystone_names.issubset(failed), (keystone_names - failed, output)
    assert failed == {
        "test_concurrent_batch_shares_a_single_fn_invocation_trial_1",
        "test_concurrent_batch_shares_a_single_fn_invocation_trial_2",
        "test_concurrent_batch_all_waiters_receive_the_identical_leader_result",
        "test_key_is_cleared_after_a_concurrent_batch_for_a_fresh_call",
        "test_concurrent_exception_propagates_to_every_waiter_and_leader_runs_once",
        "test_two_keys_coalesce_independently_under_concurrent_mixed_load",
        "test_key_is_cleared_after_an_exception_for_a_fresh_successful_call",
        "test_repeated_cycles_of_concurrent_batches_stay_correct",
    }, output


def test_never_clears_mutant_fails_exactly_one_targeted_keystone(tmp_path):
    """The surgical near-miss must fail EXACTLY the one keystone built to catch it (plus that
    keystone's two purely-sequential bonus corollaries), and pass every other keystone --
    proving this is a genuine, narrow, structurally-interacting trap rather than a broad
    correctness gap."""
    output = _run_pytest_against_impl(tmp_path, _NEVER_CLEARS_MUTANT_SOURCE)
    failed = _failed_test_names(output)
    assert failed == {
        "test_basic_sequential_calls_reinvoke_fn_every_time",
        "test_key_is_cleared_after_a_concurrent_batch_for_a_fresh_call",
        "test_key_is_cleared_after_an_exception_for_a_fresh_successful_call",
        "test_repeated_cycles_of_concurrent_batches_stay_correct",
    }, output
    keystones_failed = failed & {node_id.split("::")[1] for node_id in c24.KEYSTONE_TEST_IDS}
    assert keystones_failed == {"test_key_is_cleared_after_a_concurrent_batch_for_a_fresh_call"}


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c24", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c24"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c24.get_task_statement()
    assert (public / "repo" / c24.VISIBLE_TEST_PATH).read_text() == c24.get_sandbox_fixture()[c24.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c24.get_compiled_plan()

    assert (private / c24.VISIBLE_TEST_PATH).read_text() == c24.get_grading_payload()["tests"][c24.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c24.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c24.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
