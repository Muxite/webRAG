"""
Adversarial offline checks for codebench task c26 (deadlock-free-bank-transfer) -- no Docker
(besides the materialize_task subprocess check), no LLM.

Mirrors idea_code_test_c24_test.py / idea_code_test_c25_test.py: re-run the actual grading
scenario against a hand-written correct (consistent lock-ordering) reference and a
hand-written broken (caller-argument-order locking) reference, confirming empirically that a
REAL circular-wait deadlock forms reliably for the broken reference and never for the
correct one, and that the conservation-of-total-balance / no-negative-balance invariants are
independently sound checks (not just deadlock detection in disguise).
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from agent.app.idea_code_tests import test_c26_deadlock_free_bank_transfer as c26


class _Account:
    def __init__(self, acct_id, balance):
        self.acct_id = acct_id
        self.balance = balance
        self.lock = threading.Lock()


def _transfer_correct(src, dst, amount):
    first, second = (src, dst) if src.acct_id < dst.acct_id else (dst, src)
    with first.lock:
        with second.lock:
            if src.balance < amount:
                raise ValueError("insufficient funds")
            src.balance -= amount
            dst.balance += amount


def _transfer_naive(src, dst, amount):
    """Locks in caller-argument order, not a consistent global order -- classic circular
    wait when two threads transfer in opposite directions between the same two accounts."""
    with src.lock:
        time.sleep(0.001)
        with dst.lock:
            if src.balance < amount:
                raise ValueError("insufficient funds")
            src.balance -= amount
            dst.balance += amount


def _join_all_with_deadline(threads, deadline_s, poll=0.02):
    end = time.time() + deadline_s
    while time.time() < end:
        if all(not t.is_alive() for t in threads):
            return True
        time.sleep(poll)
    return all(not t.is_alive() for t in threads)


def _run_scenario(transfer_fn, n_accounts=4, n_threads=24, iters_per_thread=30,
                   start_balance=10_000, deadline=4.0):
    accounts = [_Account(f"acct{i}", start_balance) for i in range(n_accounts)]
    total_before = sum(a.balance for a in accounts)

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(iters_per_thread):
            i, j = rng.sample(range(n_accounts), 2)
            amt = rng.randint(1, 5)
            try:
                transfer_fn(accounts[i], accounts[j], amt)
            except ValueError:
                pass

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_threads)]
    for t in threads:
        t.start()
    finished = _join_all_with_deadline(threads, deadline)
    total_after = sum(a.balance for a in accounts)
    any_negative = any(a.balance < 0 for a in accounts)
    return finished, total_before, total_after, any_negative


def test_correct_reference_never_deadlocks_and_conserves_balance():
    for _ in range(3):
        finished, before, after, any_negative = _run_scenario(_transfer_correct)
        assert finished, "correct reference deadlocked -- calibration invariant violated"
        assert before == after
        assert not any_negative


def test_naive_argument_order_reference_reliably_deadlocks():
    failures = 0
    for _ in range(4):
        finished, _, _, _ = _run_scenario(_transfer_naive)
        if not finished:
            failures += 1
    assert failures == 4, (
        "calibration invariant violated: a caller-argument-order lock acquisition scheme "
        "did not deadlock every trial -- the deadlock scenario is not reliable enough to "
        "gate a score on"
    )


def test_conservation_check_would_catch_a_lost_update_bug_independent_of_deadlock():
    # A transfer function that races on the balance update itself (no locking at all, but
    # also no deadlock since there's nothing to hold) must fail the conservation check even
    # though every thread finishes promptly -- proves the invariant isn't just a disguised
    # deadlock check. NOTE: the sleep must sit BETWEEN the read and the write of both
    # balances, not before the read -- a sleep before the read only delays an otherwise
    # atomic `-=`/`+=` statement pair (LOAD_ATTR/BINARY_OP/STORE_ATTR never gets interrupted
    # mid-statement in practice, the same finding this whole codebench concurrency trio's
    # docstrings describe for c24's bare counter), so it doesn't open a real race window.
    # Also: with only 2 accounts total, "total balance" is trivially conserved no matter how
    # badly two transfers between that SAME pair interleave (each transfer's own math is a
    # closed system over exactly those two accounts) -- corruption is only externally
    # OBSERVABLE via the sum once a third account can absorb a lost cross-account update,
    # which is exactly why the real task's scenario uses n_accounts=4, not 2.
    def racy_transfer(src, dst, amount):
        if src.balance < amount:
            raise ValueError("insufficient funds")
        src_bal, dst_bal = src.balance, dst.balance
        time.sleep(0.001)
        src.balance = src_bal - amount
        dst.balance = dst_bal + amount

    saw_corruption = False
    for _ in range(5):
        finished, before, after, _ = _run_scenario(racy_transfer, deadline=3.0)
        assert finished, "unlocked transfer should never deadlock (nothing to hold)"
        if before != after:
            saw_corruption = True
            break
    assert saw_corruption, (
        "calibration invariant violated: an entirely unlocked transfer() never corrupted "
        "the total balance across 5 trials -- the conservation check may not be exercising "
        "real contention"
    )


def test_embedded_test_file_uses_bounded_deadline_joins_not_bare_join():
    content = c26.get_grading_payload()["tests"][c26._TEST_FILE_PATH]
    assert "_join_all_with_deadline" in content
    assert re.search(r"\.join\(\)", content) is None


def test_embedded_test_file_never_names_a_correct_implementation():
    content = c26.get_grading_payload()["tests"][c26._TEST_FILE_PATH]
    assert "def transfer(src" not in content
    assert "acct_id < dst.acct_id" not in content
    assert "sorted(" not in content


def test_keystone_ids_reference_real_test_functions():
    content = c26.get_grading_payload()["tests"][c26._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c26.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c26._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_non_contending_sanity_checks():
    assert f"{c26._TEST_FILE_PATH}::test_basic_single_threaded_transfer" not in c26.KEYSTONE_TEST_IDS
    assert (f"{c26._TEST_FILE_PATH}::test_insufficient_funds_raises_and_leaves_balances_unchanged"
            not in c26.KEYSTONE_TEST_IDS)


def test_keystone_is_exactly_the_four_concurrency_checks():
    assert set(c26.KEYSTONE_TEST_IDS) == {
        f"{c26._TEST_FILE_PATH}::test_no_deadlock_under_concurrent_bidirectional_transfers_trial_1",
        f"{c26._TEST_FILE_PATH}::test_no_deadlock_under_concurrent_bidirectional_transfers_trial_2",
        f"{c26._TEST_FILE_PATH}::test_total_balance_conserved_under_concurrent_load",
        f"{c26._TEST_FILE_PATH}::test_no_account_ever_goes_negative_under_concurrent_load",
    }


def test_visibility_is_hidden():
    assert c26.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c26.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c26.get_grading_payload()
    assert payload["tests"] == {c26._TEST_FILE_PATH: c26._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "bank",
        "class": "Account",
        "functions": ["transfer"],
    }
    assert payload["keystone_test_ids"] == c26.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c26.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "bank.py" in leaf["instruction"]
    assert "acct_id" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["bank.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_answer_code():
    instruction = c26.get_compiled_plan()["leaves"][0]["instruction"]
    assert "def transfer(src" not in instruction
    assert "sorted(" not in instruction


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c26", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c26"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c26.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c26.get_compiled_plan()

    assert (private / c26._TEST_FILE_PATH).read_text() == c26._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c26._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c26.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
