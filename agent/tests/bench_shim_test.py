"""Regression tests for the ./bench repo-root shell shim (invoked via subprocess -- it's a
POSIX sh script, not Python)."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"


def _run(*args):
    return subprocess.run([str(BENCH), *args], cwd=str(REPO),
                           capture_output=True, text=True, timeout=30)


def test_unknown_script_name_errors_sensibly():
    result = _run("totally_bogus_script_name")
    assert result.returncode != 0
    assert "no such script" in result.stderr


def test_no_args_lists_available_scripts():
    result = _run()
    assert result.returncode == 0
    assert "bench_status" in result.stdout


def test_list_flag_lists_available_scripts():
    result = _run("--list")
    assert result.returncode == 0
    assert "bench_status" in result.stdout


def test_accepts_script_name_with_py_suffix():
    result = _run("bench_status.py", "--limit", "0")
    assert result.returncode == 0


def test_accepts_script_name_without_py_suffix():
    result = _run("bench_status", "--limit", "0")
    assert result.returncode == 0


def test_resolves_repo_root_from_script_location_not_cwd(tmp_path):
    # Invoked with a different cwd -- must still find scripts/ next to the shim itself.
    result = subprocess.run([str(BENCH), "bench_status", "--limit", "0"],
                             cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
