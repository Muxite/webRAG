"""
Adversarial offline checks for codebench task c28 (brec-binary-record-format) — no Docker,
no LLM. Mirrors idea_code_test_c04_test.py's exec-based structure: every test_* function in
the embedded canonical test file is run for real against an INDEPENDENTLY-WRITTEN reference
implementation (built directly from the byte layout, not by importing the task module's own
private helper), to catch a hand-arithmetic error in the embedded worked-example bytes or
the checksum literal.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import types
from pathlib import Path

from agent.app.idea_code_tests import test_c28_brec_binary_format as c28

_MAGIC = b"BREC"


def _independent_build_records(strings: list) -> bytes:
    out = bytearray()
    out += _MAGIC
    out.append(1)
    out += struct.pack(">H", len(strings))
    for s in strings:
        raw = s.encode("utf-8")
        out += struct.pack(">H", len(raw))
        out += raw
    checksum = sum(out) % (2 ** 32)
    out += struct.pack(">I", checksum)
    return bytes(out)


def _independent_parse_records(data: bytes) -> list:
    if len(data) < 7:
        raise ValueError("too short")
    if data[:4] != _MAGIC:
        raise ValueError("bad magic")
    if data[4] != 1:
        raise ValueError("bad version")
    (count,) = struct.unpack(">H", data[5:7])
    pos = 7
    out = []
    for _ in range(count):
        if pos + 2 > len(data):
            raise ValueError("truncated length prefix")
        (flen,) = struct.unpack(">H", data[pos:pos + 2])
        pos += 2
        if pos + flen > len(data):
            raise ValueError("truncated field")
        chunk = data[pos:pos + flen]
        pos += flen
        try:
            out.append(chunk.decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError("invalid utf-8")
    remaining = len(data) - pos
    if remaining != 4:
        raise ValueError("checksum field is not exactly 4 bytes")
    expected = sum(data[:pos]) % (2 ** 32)
    (actual,) = struct.unpack(">I", data[pos:pos + 4])
    if expected != actual:
        raise ValueError("checksum mismatch")
    return out


def test_ground_truth_round_trip_cases():
    assert _independent_parse_records(_independent_build_records(["hello", "world"])) == [
        "hello", "world",
    ]
    assert _independent_parse_records(_independent_build_records([])) == []
    strings = ["héllo", "日本語", ""]
    assert _independent_parse_records(_independent_build_records(strings)) == strings


def test_ground_truth_worked_example_byte_layout():
    data = _independent_build_records(["ab"])
    expected = (
        b"BREC" + b"\x01" + b"\x00\x01" + b"\x00\x02" + b"ab" + b"\x00\x00\x01\xe3"
    )
    assert data == expected, data.hex()
    # cross-check the checksum arithmetic explicitly: sum of the 11 preceding bytes
    preceding = b"BREC" + b"\x01" + b"\x00\x01" + b"\x00\x02" + b"ab"
    assert len(preceding) == 11
    assert sum(preceding) % (2 ** 32) == 483 == 0x000001E3


def test_ground_truth_malformed_inputs_rejected():
    good = _independent_build_records(["hello", "world"])

    bad_magic = bytearray(good)
    bad_magic[0:4] = b"XXXX"
    try:
        _independent_parse_records(bytes(bad_magic))
        raise AssertionError("bad magic should have raised ValueError")
    except ValueError:
        pass

    bad_version = bytearray(good)
    bad_version[4] = 2
    try:
        _independent_parse_records(bytes(bad_version))
        raise AssertionError("bad version should have raised ValueError")
    except ValueError:
        pass

    truncated = good[:-2]
    try:
        _independent_parse_records(truncated)
        raise AssertionError("truncated data should have raised ValueError")
    except ValueError:
        pass

    corrupted_trailer = bytearray(good)
    corrupted_trailer[-1] ^= 0xFF
    try:
        _independent_parse_records(bytes(corrupted_trailer))
        raise AssertionError("corrupted checksum trailer should have raised ValueError")
    except ValueError:
        pass


def test_embedded_test_file_asserts_match_ground_truth():
    content = c28.get_grading_payload()["tests"][c28._TEST_FILE_PATH]
    fake_module = types.ModuleType("brec_mod")
    fake_module.build_records = _independent_build_records
    fake_module.parse_records = _independent_parse_records
    original = sys.modules.get("brec_mod")
    sys.modules["brec_mod"] = fake_module
    try:
        namespace: dict = {}
        exec(compile(content, "<embedded c28 test file>", "exec"), namespace)
        test_functions = [
            v for k, v in namespace.items() if k.startswith("test_") and callable(v)
        ]
        assert test_functions, "expected at least one test_ function in the embedded file"
        for fn in test_functions:
            fn()  # raises if an embedded expected value / pytest.raises() is wrong
    finally:
        if original is not None:
            sys.modules["brec_mod"] = original
        else:
            sys.modules.pop("brec_mod", None)


def test_keystone_ids_reference_real_test_functions():
    content = c28.get_grading_payload()["tests"][c28._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c28.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c28._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_worked_example_echo_and_duplicate_variant():
    # test_output_byte_layout_matches_spec directly replays the worked example given
    # verbatim in the prompt (echo risk, same convention as c08's excluded worked example),
    # and test_rejects_corrupted_payload_breaks_checksum is a deeper duplicate of the same
    # checksum-mismatch behavior test_rejects_corrupted_checksum_trailer already covers.
    assert (
        f"{c28._TEST_FILE_PATH}::test_output_byte_layout_matches_spec"
        not in c28.KEYSTONE_TEST_IDS
    )
    assert (
        f"{c28._TEST_FILE_PATH}::test_rejects_corrupted_payload_breaks_checksum"
        not in c28.KEYSTONE_TEST_IDS
    )


def test_visibility_is_hidden():
    assert c28.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c28.get_sandbox_fixture() == {}


def test_task_statement_contains_worked_example_as_an_anchor():
    statement = c28.get_task_statement()
    assert "build_records([\"ab\"])" in statement
    assert "483" in statement
    assert "BREC" in statement


def test_no_canonical_test_file_content_leaks_into_task_statement():
    # Security constraint: the prompt gives the byte-layout SPEC and one worked example,
    # but must not leak the canonical test file's own hidden assertions (e.g. the literal
    # unicode strings used in the round-trip test, or the specific corruption offsets).
    statement = c28.get_task_statement()
    for leaked in ("héllo", "日本語", "hello", "world"):
        assert leaked not in statement, f"leaked canonical test content: {leaked!r}"


def test_no_canonical_values_leak_into_compiled_plan():
    plan = c28.get_compiled_plan()
    plan_text = json.dumps(plan)
    for leaked in ("hello", "world", "héllo", "日本語"):
        assert leaked not in plan_text, f"leaked canonical value into plan.json: {leaked!r}"


def test_grading_payload_shape():
    payload = c28.get_grading_payload()
    assert payload["tests"] == {c28._TEST_FILE_PATH: c28._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "brec_mod",
        "functions": ["build_records", "parse_records"],
    }
    assert payload["keystone_test_ids"] == c28.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c28.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "brec_mod.py" in leaf["instruction"]
    assert "checksum" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["brec_mod.py"]}
    json.dumps(plan)  # must be JSON-serializable as-is


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c28", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c28"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c28.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c28.get_compiled_plan()

    assert (private / c28._TEST_FILE_PATH).read_text() == c28._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c28._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c28.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
