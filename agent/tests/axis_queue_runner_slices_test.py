"""Tests for scripts/axis_queue_runner.py's --slices feature: round-robin task dealing,
concurrent per-slice execution, and the print-only dry-run path. No subprocess is ever
actually spawned for a real benchmark here -- subprocess.Popen/subprocess.run are patched.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import axis_queue_runner as qr  # noqa: E402


# --------------------------------------------------------------------- deal_round_robin -----
def test_deal_round_robin_disjoint_balanced_covers_every_task():
    tasks = [f"{n:03d}" for n in range(122, 146)]  # 24 tasks, matches core24
    slices = qr.deal_round_robin(tasks, 3)
    assert len(slices) == 3
    covered = sorted(t for s in slices for t in s)
    assert covered == sorted(tasks)
    # disjoint
    seen = set()
    for s in slices:
        assert not (seen & set(s))
        seen.update(s)
    # balanced: 24 / 3 == 8 exactly
    assert [len(s) for s in slices] == [8, 8, 8]


def test_deal_round_robin_interleaves_not_contiguous():
    tasks = ["122", "123", "124", "125", "126", "127"]
    slices = qr.deal_round_robin(tasks, 2)
    # round-robin: evens to slice 0, odds to slice 1 -- not a contiguous split
    assert slices[0] == ["122", "124", "126"]
    assert slices[1] == ["123", "125", "127"]


def test_deal_round_robin_rejects_n_below_one():
    with pytest.raises(ValueError):
        qr.deal_round_robin(["122"], 0)


# -------------------------------------------------------------------- resolve_entry_tasks ---
def test_resolve_entry_tasks_from_task_set():
    entry = {"run_id": "r", "task_set": "smoke8"}
    assert qr.resolve_entry_tasks(entry) == qr.TASK_SETS["smoke8"]


def test_resolve_entry_tasks_defaults_to_core24():
    entry = {"run_id": "r"}
    assert qr.resolve_entry_tasks(entry) == qr.TASK_SETS["core24"]


def test_resolve_entry_tasks_from_explicit_tasks_string():
    entry = {"run_id": "r", "tasks": "158,159,160"}
    assert qr.resolve_entry_tasks(entry) == ["158", "159", "160"]


def test_resolve_entry_tasks_explicit_tasks_overrides_task_set():
    entry = {"run_id": "r", "task_set": "smoke8", "tasks": "305"}
    assert qr.resolve_entry_tasks(entry) == ["305"]


def test_resolve_entry_tasks_unknown_task_set_raises():
    entry = {"run_id": "r", "task_set": "no_such_set"}
    with pytest.raises(KeyError):
        qr.resolve_entry_tasks(entry)


# -------------------------------------------------------------------------- expand_entry ----
def test_expand_entry_n1_returns_same_object_unchanged():
    entry = {"run_id": "dagv3p0_mech", "task_set": "smoke8", "jobs": 1}
    out = qr.expand_entry(entry, 1)
    assert out == [entry]
    assert out[0] is entry  # identity: no copy, no run_id suffix, no tasks rewrite


def test_expand_entry_default_slices_absent_means_one():
    # main()'s default is 1; expand_entry(entry, 1) must be the whole story.
    entry = {"run_id": "r", "task_set": "core24"}
    out = qr.expand_entry(entry, 1)
    assert "_s0" not in out[0]["run_id"]
    assert out[0]["run_id"] == "r"


def test_expand_entry_suffixes_run_id_uniquely():
    entry = {"run_id": "dagv3p0_mech", "task_set": "smoke8", "jobs": 1}
    out = qr.expand_entry(entry, 4)
    run_ids = [e["run_id"] for e in out]
    assert len(run_ids) == len(set(run_ids))
    assert all(rid.startswith("dagv3p0_mech_s") for rid in run_ids)


def test_expand_entry_slices_task_set_into_tasks():
    entry = {"run_id": "r", "task_set": "smoke8", "jobs": 1}
    out = qr.expand_entry(entry, 2)
    covered = []
    for e in out:
        assert "task_set" not in e
        assert e.get("tasks")
        covered += e["tasks"].split(",")
    assert sorted(covered) == sorted(qr.TASK_SETS["smoke8"])


def test_expand_entry_slices_explicit_tasks():
    entry = {"run_id": "r", "tasks": "158,159,160,302,303,304,305", "jobs": 1}
    out = qr.expand_entry(entry, 3)
    covered = []
    for e in out:
        covered += e["tasks"].split(",")
    assert sorted(covered) == sorted(["158", "159", "160", "302", "303", "304", "305"])


def test_expand_entry_preserves_other_keys():
    entry = {"run_id": "r", "task_set": "smoke8", "jobs": 1, "axis": "phase0_local",
              "arms": "good_adaptive"}
    out = qr.expand_entry(entry, 2)
    for e in out:
        assert e["axis"] == "phase0_local"
        assert e["arms"] == "good_adaptive"
        assert e["jobs"] == 1


def test_expand_entry_more_slices_than_tasks_degrades_without_empty_commands():
    entry = {"run_id": "r", "tasks": "158,159", "jobs": 1}
    out = qr.expand_entry(entry, 5)
    assert len(out) == 2  # only 2 non-empty slices, not 5
    for e in out:
        assert e["tasks"]  # never empty
        cmd = qr.build_cell_command(e)
        assert "--tasks" in cmd
        assert cmd[cmd.index("--tasks") + 1]  # non-empty argument


def test_expand_entry_rejects_n_below_one():
    with pytest.raises(ValueError):
        qr.expand_entry({"run_id": "r", "task_set": "smoke8"}, 0)


# ------------------------------------------------------------------------ process_entry -----
class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


def test_process_entry_n1_uses_subprocess_run_unchanged(monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(("run", cmd, cwd))
        return _FakeCompleted(0)

    def fake_popen(*a, **k):
        raise AssertionError("Popen must not be used when slices==1")

    monkeypatch.setattr(qr.subprocess, "run", fake_run)
    monkeypatch.setattr(qr.subprocess, "Popen", fake_popen)

    entry = {"run_id": "r", "task_set": "smoke8", "jobs": 1}
    logs = []
    results = qr.process_entry(0, 1, entry, 1, logs.append)

    assert len(calls) == 1
    assert results == [(entry, 0, results[0][2])]
    assert not any("slice" in line.lower() for line in logs)


class _FakePopen:
    """Records concurrency: every instance is created before any is waited on, proving the
    caller launches all slices before blocking on the first."""
    active = 0
    max_active = 0

    def __init__(self, cmd, cwd=None, returncode=0):
        self.cmd = cmd
        self.cwd = cwd
        self._returncode = returncode
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)

    def wait(self):
        type(self).active -= 1
        return self._returncode


def test_process_entry_slices_launch_concurrently(monkeypatch):
    _FakePopen.active = 0
    _FakePopen.max_active = 0

    def fake_popen(cmd, cwd=None):
        return _FakePopen(cmd, cwd=cwd, returncode=0)

    def fake_run(*a, **k):
        raise AssertionError("subprocess.run must not be used when slicing")

    monkeypatch.setattr(qr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(qr.subprocess, "run", fake_run)

    entry = {"run_id": "r", "task_set": "smoke8", "jobs": 1}
    logs = []
    results = qr.process_entry(0, 1, entry, 3, logs.append)

    assert len(results) == 3
    assert all(rc == 0 for _, rc, _ in results)
    # all 3 slice subprocesses were live at once before any was waited on
    assert _FakePopen.max_active == 3


def test_process_entry_per_slice_failure_isolated_and_surfaced(monkeypatch):
    returncodes = {"r_s0": 0, "r_s1": 7, "r_s2": 0}

    def fake_popen(cmd, cwd=None):
        run_id = cmd[cmd.index("--run-id") + 1]
        return _FakePopen(cmd, cwd=cwd, returncode=returncodes[run_id])

    monkeypatch.setattr(qr.subprocess, "Popen", fake_popen)

    entry = {"run_id": "r", "task_set": "smoke8", "jobs": 1}
    logs = []
    results = qr.process_entry(0, 1, entry, 3, logs.append)

    rc_by_run_id = {e["run_id"]: rc for e, rc, _ in results}
    assert rc_by_run_id == returncodes
    assert any("FAILED rc=7" in line and "r_s1" in line for line in logs)
    # the other two slices' success is still reported, not swallowed by the failure
    assert sum(1 for line in logs if line.startswith(f"[1/1] OK")) == 2


def test_process_entry_launch_failure_isolated(monkeypatch):
    def fake_popen(cmd, cwd=None):
        run_id = cmd[cmd.index("--run-id") + 1]
        if run_id.endswith("_s1"):
            raise OSError("no such interpreter")
        return _FakePopen(cmd, cwd=cwd, returncode=0)

    monkeypatch.setattr(qr.subprocess, "Popen", fake_popen)

    entry = {"run_id": "r", "task_set": "smoke8", "jobs": 1}
    logs = []
    results = qr.process_entry(0, 1, entry, 3, logs.append)

    assert len(results) == 3
    rc_by_run_id = {e["run_id"]: rc for e, rc, _ in results}
    assert rc_by_run_id["r_s1"] == -1
    assert rc_by_run_id["r_s0"] == 0
    assert rc_by_run_id["r_s2"] == 0
    assert any("FAILED-TO-LAUNCH" in line and "r_s1" in line for line in logs)
