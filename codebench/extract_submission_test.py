"""Tests for extract_submission.py — plain pytest / sys.path-injection convention,
matching badmodel-lab/analyze_calibration_test.py for other standalone scripts that live
outside the services.* package tree.

The symlink-rejection tests are the load-bearing ones: an adversarial review found this
script would dereference and copy a symlink's TARGET content into the submission tree
(shutil.copy2's default follow_symlinks=True), letting a single `os.symlink(...)` action
inside the agent's sandbox exfiltrate any file readable by whoever runs the harness, since
/work is a real host bind mount. Verified live against the actual script before/after the
fix; these pin that fix in place.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_submission as es  # noqa: E402


def test_kept_files_are_copied_verbatim(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fib_mod.py").write_text("def f(): return 1\n")

    out = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"test_file_globs": []}))

    kept = es_main(raw, manifest, out)
    assert kept == {"fib_mod.py"}
    assert (out / "fib_mod.py").read_text() == "def f(): return 1\n"


def test_manifest_matched_files_are_dropped(tmp_path):
    raw = tmp_path / "raw"
    (raw / "tests").mkdir(parents=True)
    (raw / "tests" / "test_fib_mod.py").write_text("tampered content")
    (raw / "fib_mod.py").write_text("real submission")

    out = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"test_file_globs": ["tests/test_fib_mod.py"]}))

    kept = es_main(raw, manifest, out)
    assert kept == {"fib_mod.py"}


def test_symlinked_file_is_rejected_not_dereferenced(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET HOST CONTENT")
    (raw / "leaked.txt").symlink_to(secret)
    (raw / "normal.py").write_text("fine")

    out = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"test_file_globs": []}))

    kept = es_main(raw, manifest, out)
    assert kept == {"normal.py"}
    assert not (out / "leaked.txt").exists()


def test_symlinked_directory_is_rejected_not_walked_into(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("host content")
    (raw / "linked_dir").symlink_to(outside)
    (raw / "normal.py").write_text("fine")

    out = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"test_file_globs": []}))

    kept = es_main(raw, manifest, out)
    assert kept == {"normal.py"}
    assert not (out / "linked_dir").exists()
    assert not any(out.rglob("secret.txt"))


def test_is_dropped_matches_fnmatch_glob():
    assert es.is_dropped("tests/test_fib_mod.py", ["tests/test_fib_mod.py"])
    assert not es.is_dropped("tests/Test_fib_mod.py", ["tests/test_fib_mod.py"])  # case-sensitive
    assert not es.is_dropped("fib_mod.py", ["tests/test_fib_mod.py"])


def test_load_manifest_missing_file_is_empty(tmp_path):
    assert es.load_manifest(tmp_path / "nope.json") == []


def es_main(raw_dir: Path, manifest: Path, out: Path) -> set[str]:
    """Run the real script as a subprocess (matches how run_matrix.sh invokes it) and
    return the set of relative paths that ended up in --out."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "extract_submission.py"),
         "--raw-dir", str(raw_dir), "--manifest", str(manifest), "--out", str(out)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
