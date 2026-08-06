"""
codebench task c26 — hard/hidden, concurrency (deadlock avoidance via lock-ordering).

Third concurrency-shaped codebench task (see test_c24_thread_safe_counter.py and
test_c25_bounded_blocking_queue.py for the first two). Where c24/c25 are about NOT losing
data under contention, c26 is about a different failure mode entirely: two locks acquired
in an inconsistent order across threads, causing a genuine, permanent, circular-wait
deadlock (the textbook "bank transfer between two accounts" puzzle) -- an easy way for a
model to write code that "looks right" and even passes light manual testing (a single
transfer works fine) but locks up completely the instant two transfers run concurrently in
opposite directions between the same two accounts.

Unlike c24/c25's data-corruption races, a REAL circular-wait deadlock in CPython does not
need any artificial widening: `Lock.acquire()` when the lock is already held is a genuine
blocking call that releases the GIL, so once thread A holds lock 1 and blocks waiting for
lock 2 while thread B holds lock 2 and blocks waiting for lock 1, that condition is
PERMANENT (barring external intervention) -- there is no scheduling luck involved once the
circular hold-and-wait actually forms, only in HOW LIKELY it is to form at all within a
bounded number of trials/threads. This task's offline validator confirms live (both on this
host's python3.12 and the actual codebench sandbox's python:3.10-slim image) that a naive
`with src.lock: with dst.lock: ...` implementation (locks acquired in caller-argument order,
not a consistent global order) deadlocks in 100% of 6 independent trials when ~24 threads
fire concurrent transfers in both directions across a small pool of accounts, while a
correct implementation (locks acquired in a consistent order derived from account identity,
e.g. sorted by account id, regardless of which account is `src` vs `dst`) never deadlocks
across the same trials and completes in milliseconds.

The grading invariant deliberately does NOT pin down HOW the agent avoids the deadlock (any
consistent global lock-ordering scheme works, and so would coarser-grained locking) -- it
only checks the OBSERVABLE, scheduling-independent contract: every thread completes within a
generous deadline (no deadlock), total money across all accounts is conserved (no lost or
duplicated balance from a racy read-modify-write), and no account balance ever goes
negative.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_bank_transfer.py"

_TEST_FILE_CONTENT = '''\
import random
import threading
import time

import pytest
from bank import Account, transfer


def _join_all_with_deadline(threads, deadline_s, poll=0.02):
    """Wait for every thread to finish, but never longer than deadline_s total -- a
    deadlocked transfer() must make this test fail fast, not hang the whole suite."""
    end = time.time() + deadline_s
    while time.time() < end:
        if all(not t.is_alive() for t in threads):
            return True
        time.sleep(poll)
    return all(not t.is_alive() for t in threads)


def test_basic_single_threaded_transfer():
    a = Account("a", 100)
    b = Account("b", 50)
    transfer(a, b, 30)
    assert a.balance == 70
    assert b.balance == 80


def test_insufficient_funds_raises_and_leaves_balances_unchanged():
    a = Account("a", 10)
    b = Account("b", 5)
    with pytest.raises(ValueError):
        transfer(a, b, 999)
    assert a.balance == 10
    assert b.balance == 5


def _run_bidirectional_stress(n_accounts=4, n_threads=24, iters_per_thread=30,
                               start_balance=10_000, deadline=4.0):
    accounts = [Account(f"acct{i}", start_balance) for i in range(n_accounts)]
    total_before = sum(acc.balance for acc in accounts)

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(iters_per_thread):
            i, j = rng.sample(range(n_accounts), 2)
            amount = rng.randint(1, 5)
            try:
                transfer(accounts[i], accounts[j], amount)
            except ValueError:
                pass  # insufficient funds is an expected, non-corrupting outcome

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for t in threads:
        t.start()
    finished = _join_all_with_deadline(threads, deadline)

    total_after = sum(acc.balance for acc in accounts)
    any_negative = any(acc.balance < 0 for acc in accounts)
    return finished, total_before, total_after, any_negative


def test_no_deadlock_under_concurrent_bidirectional_transfers_trial_1():
    finished, _, _, _ = _run_bidirectional_stress()
    assert finished, "not every transfer thread finished in time -- likely deadlock"


def test_no_deadlock_under_concurrent_bidirectional_transfers_trial_2():
    finished, _, _, _ = _run_bidirectional_stress()
    assert finished, "not every transfer thread finished in time -- likely deadlock"


def test_total_balance_conserved_under_concurrent_load():
    finished, total_before, total_after, _ = _run_bidirectional_stress()
    assert finished, "not every transfer thread finished in time -- likely deadlock"
    assert total_after == total_before, (
        f"total balance changed from {total_before} to {total_after} -- "
        "a concurrent transfer lost or duplicated money"
    )


def test_no_account_ever_goes_negative_under_concurrent_load():
    finished, _, _, any_negative = _run_bidirectional_stress()
    assert finished, "not every transfer thread finished in time -- likely deadlock"
    assert not any_negative, "some account balance went negative under concurrent load"
'''

# The two deadlock-freedom trials plus the conservation and no-negative-balance checks (each
# run against the SAME kind of concurrent bidirectional load) are the core of this task --
# gate the score. The basic single-threaded transfer and the insufficient-funds contract are
# supporting/bonus credit only: neither involves any real concurrency, so they don't
# distinguish a correct lock-ordering scheme from a naive, deadlock-prone one.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_no_deadlock_under_concurrent_bidirectional_transfers_trial_1",
    f"{_TEST_FILE_PATH}::test_no_deadlock_under_concurrent_bidirectional_transfers_trial_2",
    f"{_TEST_FILE_PATH}::test_total_balance_conserved_under_concurrent_load",
    f"{_TEST_FILE_PATH}::test_no_account_ever_goes_negative_under_concurrent_load",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c26",
        "title": "deadlock-free-bank-transfer",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `bank.py` implementing a small in-memory bank that supports "
        "concurrent transfers between accounts WITHOUT ever deadlocking.\n\n"
        "Implement:\n"
        "- `Account(acct_id: str, balance: int)` — a class representing one account. Each "
        "instance must have (at least) a `.acct_id` attribute (the string passed in) and a "
        "`.balance` attribute (an int, starting at the given `balance`) that reflects the "
        "account's current balance. Multiple threads may read and modify the same "
        "`Account`'s balance concurrently (via `transfer`, below) -- access to it must be "
        "properly synchronized.\n"
        "- `transfer(src: Account, dst: Account, amount: int) -> None` — atomically moves "
        "`amount` from `src`'s balance to `dst`'s balance. If `src.balance < amount`, raise "
        "`ValueError` and leave BOTH accounts' balances unchanged.\n\n"
        "The critical requirement: `transfer` will be called concurrently, from many "
        "threads, on the SAME small pool of accounts, in BOTH directions (e.g. one thread "
        "calling `transfer(A, B, ...)` while another simultaneously calls "
        "`transfer(B, A, ...)`). Each transfer needs to hold both accounts' locks at once "
        "while it moves the money (to prevent a torn read-modify-write). If you always "
        "acquire `src`'s lock first and then `dst`'s lock (i.e. in the order the caller "
        "happened to pass the arguments), two concurrent transfers going in opposite "
        "directions between the same two accounts WILL deadlock: each thread ends up "
        "holding one account's lock while waiting forever for the other. To avoid this, "
        "acquire the two accounts' locks in a CONSISTENT order that does not depend on "
        "which one is `src` and which is `dst` — for example, order them by `acct_id` — so "
        "every thread, regardless of transfer direction, always tries to lock the same "
        "account first.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check "
        "yourself before finishing: a single transfer moves the right amount between two "
        "accounts; transferring more than the available balance raises ValueError and "
        "changes nothing; and — most importantly — spin up a handful of threads doing "
        "transfers back and forth between the same two or three accounts concurrently and "
        "confirm with `thread.join(timeout=...)` that every thread actually finishes (does "
        "not hang) within a few seconds. Fix issues with patch_file/run_python until "
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
            "module": "bank",
            "class": "Account",
            "functions": ["transfer"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write bank.py implementing a small in-memory bank supporting "
                    "concurrent transfers WITHOUT deadlocking. Define `Account(acct_id, "
                    "balance)` with `.acct_id` and `.balance` attributes, and each account "
                    "needs its own synchronization primitive (e.g. threading.Lock) since "
                    "its balance will be read/modified from multiple threads concurrently. "
                    "Define `transfer(src, dst, amount)`: atomically move `amount` from "
                    "src's balance to dst's balance; raise ValueError (leaving both "
                    "balances unchanged) if src.balance < amount. The critical part: "
                    "transfer() will be called concurrently from many threads, in BOTH "
                    "directions, on the same small set of accounts (e.g. thread 1 calls "
                    "transfer(A, B, ...) while thread 2 simultaneously calls "
                    "transfer(B, A, ...)). Each call needs both accounts' locks held at "
                    "once. If you acquire them in caller-argument order (src's lock, then "
                    "dst's lock), opposite-direction concurrent transfers on the same "
                    "account pair WILL deadlock (classic circular wait: each thread holds "
                    "one lock and blocks forever waiting for the other). Avoid this by "
                    "acquiring the two accounts' locks in a CONSISTENT global order that "
                    "does NOT depend on which is src/dst -- e.g. sort the two accounts by "
                    "acct_id and always lock the lower one first, regardless of transfer "
                    "direction. Use write_file to create bank.py. Since there is no "
                    "visible test file, use run_python to sanity-check: a single transfer "
                    "works and updates both balances correctly; insufficient funds raises "
                    "ValueError without changing anything; and spin up several threads "
                    "doing transfers back and forth between the same accounts "
                    "concurrently, joining each with a timeout, to confirm none of them "
                    "hang. Fix issues with patch_file/run_python until confident, then "
                    "finish."
                ),
                "expect": "bank.py written, defining Account and transfer(src, dst, amount) "
                          "with consistent cross-account lock ordering, sanity-checked with "
                          "run_python including a concurrent no-hang check",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm bank.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["bank.py"]},
    }
