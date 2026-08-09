"""Offline unit tests for the code-domain sandbox connector (app/connector_sandbox.py) — free.

The load-bearing test here is the CONFINEMENT guard: ``confine()`` is the only thing standing
between a model-authored relpath and the rest of the filesystem for our file abstractions, so it
gets adversarial coverage (``..`` traversal, absolute paths, symlink escape, and the same escapes
attempted through the public actions). The rest pins the action contract the leaf loop relies on:
every action returns a dict and never raises, the write budgets are enforced, and the subprocess
actions run inside the workdir with truncated output and a real timeout kill.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.app.connector_sandbox import (
    DIR_ENTRY_CAP,
    OUTPUT_CHAR_CAP,
    SandboxConnector,
    _truncate,
)
from agent.app.idea_policies.config import SandboxActionConfig

#: A lone surrogate, constructed the way one actually reaches this connector: a model emits it,
#: ``json.loads`` accepts it happily, and every UTF-8 encoder downstream refuses it.
LONE_SURROGATE = json.loads('{"content": "\\ud800"}')["content"]


def _sandbox(tmp_path: Path, **limit_overrides) -> SandboxConnector:
    root = tmp_path / "work"
    limits = SandboxActionConfig(workdir_root=str(root), **limit_overrides)
    return SandboxConnector(root, limits=limits)


# --- confinement (security-critical) -----------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "../escape.txt",
    "../../escape.txt",
    "sub/../../escape.txt",
    "/etc/passwd",
    "/tmp/escape.txt",
    "..",
])
def test_confine_refuses_escapes(tmp_path, bad):
    sb = _sandbox(tmp_path)
    assert sb.confine(bad) is None


@pytest.mark.parametrize("good", ["", ".", "a.py", "pkg/mod.py", "./pkg/../a.py"])
def test_confine_allows_paths_inside_the_workdir(tmp_path, good):
    sb = _sandbox(tmp_path)
    resolved = sb.confine(good)
    assert resolved is not None
    base = sb.workdir.resolve()
    assert resolved == base or base in resolved.parents


def test_confine_refuses_symlink_escape(tmp_path):
    sb = _sandbox(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("canonical test content")
    (sb.workdir / "link").symlink_to(outside)
    assert sb.confine("link/secret.txt") is None
    result = asyncio.run(sb.read_file("link/secret.txt"))
    assert result["ok"] is False and "escapes the workdir" in result["error"]


def test_write_file_escape_is_denied_and_writes_nothing(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.write_file("../pwned.txt", "x"))
    assert result["ok"] is False and "escapes the workdir" in result["error"]
    assert not (tmp_path / "pwned.txt").exists()


def test_run_pytest_escape_is_denied(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_pytest("../../tests"))
    assert result["ok"] is False and "escapes the workdir" in result["error"]


def test_run_python_file_escape_is_denied(tmp_path):
    sb = _sandbox(tmp_path)
    (tmp_path / "evil.py").write_text("print('nope')")
    result = asyncio.run(sb.run_python("../evil.py"))
    assert result["ok"] is False and "escapes the workdir" in result["error"]


# --- file actions ------------------------------------------------------------------------------
def test_write_then_read_round_trip(tmp_path):
    sb = _sandbox(tmp_path)
    written = asyncio.run(sb.write_file("pkg/mod.py", "value = 42\n"))
    assert written["ok"] is True and written["path"] == "pkg/mod.py"
    assert (sb.workdir / "pkg" / "mod.py").read_text() == "value = 42\n"
    read = asyncio.run(sb.read_file("pkg/mod.py"))
    assert read["ok"] is True and read["output"] == "value = 42\n"


def test_read_missing_file_is_a_result_not_an_exception(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.read_file("nope.py"))
    assert result["ok"] is False and "no such file" in result["error"]


def test_list_dir_lists_root_and_marks_directories(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("a.py", "a"))
    asyncio.run(sb.write_file("pkg/b.py", "b"))
    (sb.workdir / "__pycache__").mkdir()
    result = asyncio.run(sb.list_dir(""))
    assert result["ok"] is True
    assert result["entries"] == ["a.py", "pkg/"]  # __pycache__ is noise, never listed


def test_patch_file_replaces_first_occurrence_only(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("m.py", "x = 1\ny = 1\n"))
    result = asyncio.run(sb.patch_file("m.py", "= 1", "= 2"))
    assert result["ok"] is True and result["occurrences"] == 2
    assert (sb.workdir / "m.py").read_text() == "x = 2\ny = 1\n"


def test_patch_file_miss_is_actionable(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("m.py", "x = 1\n"))
    result = asyncio.run(sb.patch_file("m.py", "nowhere", "here"))
    assert result["ok"] is False and "not found" in result["error"]
    assert (sb.workdir / "m.py").read_text() == "x = 1\n"


# --- malformed model-authored input is a result, never an exception -----------------------------
def test_write_file_with_unencodable_content_fails_cleanly(tmp_path):
    """A lone surrogate used to raise ``UnicodeEncodeError`` straight out of the connector.

    ``UnicodeEncodeError`` is a ``UnicodeError``/``ValueError``, so it slipped past the
    ``except OSError`` here; on the engine's non-parallel path that took down the WHOLE run rather
    than the one leaf that authored the bad content.
    """
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.write_file("bad.txt", LONE_SURROGATE))
    assert result["ok"] is False
    assert "cannot be encoded" in result["error"]
    assert not (sb.workdir / "bad.txt").exists(), "refuse, do not silently rewrite the content"


def test_patch_file_with_unencodable_replacement_fails_cleanly(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("m.py", "x = 1\n"))
    result = asyncio.run(sb.patch_file("m.py", "x = 1", LONE_SURROGATE))
    assert result["ok"] is False and "cannot be encoded" in result["error"]
    assert (sb.workdir / "m.py").read_text() == "x = 1\n", "the file must be left untouched"


@pytest.mark.parametrize("bad, expected", [
    (LONE_SURROGATE, "cannot be encoded"),
    ("\x00", "null byte"),
])
def test_run_python_with_malformed_inline_code_fails_cleanly(tmp_path, bad, expected):
    """The SAME bug class one layer down, in the shared ``_run`` subprocess launcher.

    ``create_subprocess_exec`` also only had ``except OSError`` around it, and every read-only
    action goes through it too — so a surrogate (``UnicodeEncodeError``) or an embedded NUL (a
    bare ``ValueError("embedded null byte")``) in model-authored argv crashed the run.
    """
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_python(f"print('x'){bad}"))
    assert result["ok"] is False
    assert "could not start" in result["error"] and expected in result["error"]


@pytest.mark.parametrize("bad, expected", [
    (LONE_SURROGATE, "cannot be encoded"),
    ("\x00", "null byte"),
])
def test_readonly_command_with_a_malformed_pattern_fails_cleanly(tmp_path, bad, expected):
    """Reachable from the NATIVE engine today: ``count_lines``/``find_files`` pass a
    model-authored pattern straight into argv."""
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("notes.txt", "alpha\n"))
    result = asyncio.run(sb.run_readonly("count_lines", ["grep", "-c", "--", f"a{bad}", "notes.txt"]))
    assert result["ok"] is False
    assert "could not start" in result["error"] and expected in result["error"]


def test_a_real_valueerror_is_not_swallowed_as_a_model_mistake(tmp_path, monkeypatch):
    """Widening the catch must not turn a genuine bug into "your argument was malformed"."""
    sb = _sandbox(tmp_path)

    async def boom(*args, **kwargs):
        raise ValueError("a real bug in our own code")

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    with pytest.raises(ValueError, match="a real bug"):
        asyncio.run(sb.run_python("print(1)"))


# --- caps --------------------------------------------------------------------------------------
def test_read_file_refuses_an_oversize_file_before_reading_it(tmp_path):
    """The pre-read guard ``write_file``/``patch_file`` always had: a file can enter the workdir
    without passing the write budget (starter fixture, a ``run_python`` body's own ``open()``), so
    the read side cannot assume everything on disk is under the limit."""
    sb = _sandbox(tmp_path, max_file_bytes=32)
    (sb.workdir / "big.txt").write_text("y" * 500, encoding="utf-8")
    result = asyncio.run(sb.read_file("big.txt"))
    assert result["ok"] is False
    assert "over the 32-byte limit" in result["error"] and result["bytes"] == 500
    assert "output" not in result, "an oversize file must not be materialized into the result"
    # ...and a file under the limit still reads normally
    asyncio.run(sb.write_file("small.txt", "ok\n"))
    assert asyncio.run(sb.read_file("small.txt"))["output"] == "ok\n"


def test_list_dir_caps_its_entries_and_says_how_many_it_dropped(tmp_path):
    sb = _sandbox(tmp_path)
    for i in range(DIR_ENTRY_CAP + 7):
        (sb.workdir / f"f{i:04d}.txt").write_text("x", encoding="utf-8")
    result = asyncio.run(sb.list_dir(""))
    assert result["ok"] is True
    assert len(result["entries"]) == DIR_ENTRY_CAP
    assert result["total_entries"] == DIR_ENTRY_CAP + 7 and result["truncated"] is True
    assert result["output"].endswith("...and 7 more"), "a partial listing must say it is partial"


def test_list_dir_under_the_cap_is_not_marked_truncated(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("a.py", "a"))
    result = asyncio.run(sb.list_dir(""))
    assert result["truncated"] is False and result["total_entries"] == 1
    assert "more" not in result["output"]


def test_max_file_bytes_rejects_an_oversize_write(tmp_path):
    sb = _sandbox(tmp_path, max_file_bytes=10)
    result = asyncio.run(sb.write_file("big.py", "x" * 11))
    assert result["ok"] is False and "over the 10-byte limit" in result["error"]
    assert not (sb.workdir / "big.py").exists()


def test_max_files_per_leaf_caps_distinct_files_but_allows_rewrites(tmp_path):
    sb = _sandbox(tmp_path, max_files_per_leaf=2)
    assert asyncio.run(sb.write_file("a.py", "1"))["ok"] is True
    assert asyncio.run(sb.write_file("b.py", "2"))["ok"] is True
    # rewriting an already-counted file is always allowed (the write->test->fix loop needs it)
    assert asyncio.run(sb.write_file("a.py", "1b"))["ok"] is True
    blocked = asyncio.run(sb.write_file("c.py", "3"))
    assert blocked["ok"] is False and "budget exhausted" in blocked["error"]
    # a new leaf gets a fresh budget
    sb.reset_leaf_budget()
    assert asyncio.run(sb.write_file("c.py", "3"))["ok"] is True


def test_truncate_keeps_head_and_tail(tmp_path):
    text = "H" * 100 + "M" * 5000 + "T" * 100
    out = _truncate(text, 300)
    assert out.startswith("H") and out.endswith("T") and "chars omitted" in out
    assert len(out) < len(text)


# --- subprocess actions ------------------------------------------------------------------------
def test_run_python_inline_code_runs_in_the_workdir(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_python("import os; print(os.getcwd())"))
    assert result["ok"] is True and result["exit_code"] == 0
    assert str(sb.workdir.resolve()) in result["stdout"]


def test_run_python_file_path_runs_the_file(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("go.py", "print('from-file')\n"))
    result = asyncio.run(sb.run_python("go.py"))
    assert result["ok"] is True and "from-file" in result["stdout"]


def test_run_python_failure_reports_exit_code_and_stderr(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_python("raise SystemExit(3)"))
    assert result["ok"] is False and result["exit_code"] == 3


def test_run_python_timeout_is_a_result_not_a_hang(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_python("import time; time.sleep(30)", timeout_s=1))
    assert result["ok"] is False and result["timed_out"] is True
    assert "timed out" in result["error"]


#: A ``run_python`` body that daemonizes the classic way — double fork, let the intermediate parent
#: exit, drop the inherited stdio — and then hangs so the action times out. Killing only the DIRECT
#: child leaves that heartbeat-writing grandchild running long past the declared timeout, untracked
#: by the connector; it stays in the child's process group, so a process-group kill reaches it.
#: The daemon self-limits to ~20s so a regression cannot leave a spinner behind.
_DAEMONIZING_BODY = """
import os, time

pid = os.fork()
if pid == 0:
    if os.fork() > 0:
        os._exit(0)                       # intermediate parent exits -> the grandchild is orphaned
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)              # release the inherited pipes, like a real daemon
    for _ in range(400):
        with open("heartbeat", "w") as fh:
            fh.write(str(time.time()))
        time.sleep(0.05)
    os._exit(0)
os.waitpid(pid, 0)
time.sleep(60)                            # the direct child hangs, forcing the timeout
"""


def test_timeout_kills_the_whole_process_group_not_just_the_direct_child(tmp_path):
    """A timed-out action must not leave a detached descendant running.

    The declared per-action timeout is a real bound, so a body that daemonizes itself cannot buy
    unbounded runtime by outliving the one process we kill.
    """
    sb = _sandbox(tmp_path)
    started = time.monotonic()
    result = asyncio.run(sb.run_python(_DAEMONIZING_BODY, timeout_s=1))
    elapsed = time.monotonic() - started
    assert result["ok"] is False and result["timed_out"] is True
    assert elapsed < 8, "the timed-out action held the caller for the daemon's whole lifetime"

    heartbeat = sb.workdir / "heartbeat"
    assert heartbeat.exists(), "the detached child never ran — this test would prove nothing"
    time.sleep(0.4)                       # it beats every 50ms; if it is alive, mtime moves
    first = heartbeat.stat().st_mtime_ns
    time.sleep(0.6)
    assert heartbeat.stat().st_mtime_ns == first, "the orphaned daemon survived the timeout kill"


def test_run_python_output_is_truncated(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_python("print('x' * 50000)", timeout_s=30))
    assert len(result["stdout"]) <= OUTPUT_CHAR_CAP + 64


def test_run_pytest_reports_pass_and_fail(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("test_ok.py", "def test_ok():\n    assert 1 == 1\n"))
    passed = asyncio.run(sb.run_pytest("test_ok.py", timeout_s=120))
    assert passed["ok"] is True and passed["exit_code"] == 0

    asyncio.run(sb.write_file("test_bad.py", "def test_bad():\n    assert 1 == 2\n"))
    failed = asyncio.run(sb.run_pytest("test_bad.py", timeout_s=120))
    assert failed["ok"] is False and failed["exit_code"] != 0
    assert "test_bad" in failed["output"]


def test_run_pytest_missing_target_is_reported(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_pytest("tests/test_nope.py"))
    assert result["ok"] is False and "no such test path" in result["error"]


def test_subprocess_env_disables_bytecode_and_puts_workdir_first(tmp_path):
    sb = _sandbox(tmp_path)
    env = sb._subprocess_env()
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPATH"].split(":")[0] == str(sb.workdir.resolve())


# --- web action --------------------------------------------------------------------------------
def test_search_web_delegates_to_the_injected_search_connector(tmp_path):
    sb = _sandbox(tmp_path)
    sb.connector_search = type("S", (), {})()
    sb.connector_search.query_search = AsyncMock(return_value=[
        {"title": "Docs", "url": "https://example.test/x", "description": "how to"},
    ])
    result = asyncio.run(sb.search_web("python itertools"))
    assert result["ok"] is True
    assert "https://example.test/x" in result["output"]
    sb.connector_search.query_search.assert_awaited_once()


def test_search_web_without_a_connector_is_a_clean_miss(tmp_path):
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.search_web("anything"))
    assert result["ok"] is False and "not available" in result["error"]


def test_search_web_absorbs_a_backend_failure(tmp_path):
    sb = _sandbox(tmp_path)
    sb.connector_search = type("S", (), {})()
    sb.connector_search.query_search = AsyncMock(side_effect=RuntimeError("searxng down"))
    result = asyncio.run(sb.search_web("q"))
    assert result["ok"] is False and "searxng down" in result["error"]


# --- introspection helpers used by the submit composer ------------------------------------------
def test_exists_and_file_size_are_confined(tmp_path):
    sb = _sandbox(tmp_path)
    asyncio.run(sb.write_file("out.py", "abc"))
    assert sb.exists("out.py") is True
    assert sb.file_size("out.py") == 3
    (tmp_path / "outside.py").write_text("abc")
    assert sb.exists("../outside.py") is False
    assert sb.file_size("../outside.py") is None


def test_sandbox_uses_the_running_interpreter(tmp_path):
    """Subprocess actions must run the SAME interpreter the harness runs under (no PATH lookup)."""
    sb = _sandbox(tmp_path)
    result = asyncio.run(sb.run_python("import sys; print(sys.executable)"))
    assert sys.executable in result["stdout"]
