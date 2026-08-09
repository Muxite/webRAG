"""
codebench task c28 — hard/hidden, fixed-header binary record format (magic bytes,
length-prefixed fields, checksum validation, malformed-input rejection).

File-format coverage gap this fills: c04 (CSV), c07 (fleet-config gen), c10 (markdown
table), c12 (JSON diff), c27 (INI) are all TEXT formats. This is the suite's first BINARY
format — a genuinely different skill (struct packing/unpacking, big-endian integers,
byte-offset bookkeeping, decode-error handling) from anything else in idea_code_tests/.

## Exact byte layout (BREC v1) — restated verbatim in the task statement below, so this
## docstring and the agent-visible prompt cannot silently drift apart:

    offset 0..3   (4 bytes)  magic       ASCII b"BREC"
    offset 4      (1 byte)   version     unsigned 8-bit int, must be exactly 1
    offset 5..6   (2 bytes)  rec_count   unsigned 16-bit big-endian int
    then, repeated rec_count times, back to back:
        (2 bytes)  field_len   unsigned 16-bit big-endian int
        (field_len bytes)  field_data   raw UTF-8 encoded bytes of that record's string
    then, immediately after the last record (or right after rec_count if rec_count == 0):
        (4 bytes)  checksum    unsigned 32-bit big-endian int =
                                (sum of every byte from offset 0 up to, but NOT
                                 including, this checksum field) mod (2**32)
    No bytes may follow the checksum field.

Ground truth for ``build_records``/``parse_records`` (and the worked-example byte string
below) verified by actually RUNNING a reference implementation built on Python's own
``struct`` module (NOT hand-computed) — see
``/tmp/claude-1000/-home-muk-projects-webRAG/cebc60e0-6c4f-4b3e-9be6-882dd9f08d84/scratchpad/ref/brec_mod.py``
for the throwaway script this was derived from, and idea_code_test_c28_test.py for the
independent second reimplementation that re-verifies every literal below at test time.
Verified behavior:
    build_records(["hello", "world"])
        -> b'BREC\\x01\\x00\\x02\\x00\\x05hello\\x00\\x05world\\x00\\x00\\x05e'
    build_records([]) -> b'BREC\\x01\\x00\\x00\\x00\\x00\\x01\\x1d'          (header + checksum only)
    build_records(["ab"])
        -> b'BREC\\x01\\x00\\x01\\x00\\x02ab\\x00\\x00\\x01\\xe3'            (worked example below)
    parse_records(build_records(strings)) == strings for any list of strings, including
        unicode multi-byte UTF-8 strings and empty-string fields (field_len == 0).
    Malformed input -> ValueError: bad magic, bad version, truncated data (cut off mid
        field or mid checksum), corrupted checksum trailer, and a payload byte changed
        after the checksum was computed (checksum no longer matches) all raise ValueError.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_brec_mod.py"

_TEST_FILE_CONTENT = '''\
import pytest
from brec_mod import parse_records, build_records


def test_round_trip_basic():
    data = build_records(["hello", "world"])
    assert parse_records(data) == ["hello", "world"]


def test_round_trip_empty_list():
    data = build_records([])
    assert parse_records(data) == []


def test_round_trip_unicode_and_empty_string_field():
    strings = ["h\\u00e9llo", "\\u65e5\\u672c\\u8a9e", ""]
    data = build_records(strings)
    assert parse_records(data) == strings


def test_output_byte_layout_matches_spec():
    data = build_records(["ab"])
    assert data == (
        b"BREC" + b"\\x01" + b"\\x00\\x01" + b"\\x00\\x02" + b"ab" + b"\\x00\\x00\\x01\\xe3"
    )


def test_rejects_bad_magic():
    data = bytearray(build_records(["hello", "world"]))
    data[0:4] = b"XXXX"
    with pytest.raises(ValueError):
        parse_records(bytes(data))


def test_rejects_bad_version():
    data = bytearray(build_records(["hello", "world"]))
    data[4] = 2
    with pytest.raises(ValueError):
        parse_records(bytes(data))


def test_rejects_corrupted_checksum_trailer():
    data = bytearray(build_records(["hello", "world"]))
    data[-1] ^= 0xFF
    with pytest.raises(ValueError):
        parse_records(bytes(data))


def test_rejects_corrupted_payload_breaks_checksum():
    data = bytearray(build_records(["hello", "world"]))
    assert chr(data[9]) == "h"
    data[9] = ord("H")
    with pytest.raises(ValueError):
        parse_records(bytes(data))


def test_rejects_truncated_data():
    data = build_records(["hello", "world"])
    with pytest.raises(ValueError):
        parse_records(data[:-2])
'''

# The round-trip cases (basic, empty list, unicode/empty-field) and every malformed-input
# rejection (bad magic, bad version, truncated) are the core spec-following behaviors and
# gate the score. The exact-byte-layout check reuses a worked example given verbatim in
# the task statement (a model could echo it without truly implementing the general spec,
# same convention c08 uses for its worked example), and the payload-corruption variant is
# a deeper duplicate of the same checksum-mismatch behavior the trailer-corruption case
# already covers — both bonus credit only.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_round_trip_basic",
    f"{_TEST_FILE_PATH}::test_round_trip_empty_list",
    f"{_TEST_FILE_PATH}::test_round_trip_unicode_and_empty_string_field",
    f"{_TEST_FILE_PATH}::test_rejects_bad_magic",
    f"{_TEST_FILE_PATH}::test_rejects_bad_version",
    f"{_TEST_FILE_PATH}::test_rejects_corrupted_checksum_trailer",
    f"{_TEST_FILE_PATH}::test_rejects_truncated_data",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c28",
        "title": "brec-binary-record-format",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `brec_mod.py` that defines two functions, "
        "`build_records(strings: list[str]) -> bytes` and "
        "`parse_records(data: bytes) -> list[str]`, implementing a small fixed-layout "
        "binary format called BREC (v1). The exact byte layout is specified below — "
        "follow it precisely, byte for byte.\n\n"
        "## BREC v1 byte layout\n\n"
        "```\n"
        "offset 0..3   (4 bytes)  magic       ASCII bytes b\"BREC\"\n"
        "offset 4      (1 byte)   version     unsigned 8-bit int, must be exactly 1\n"
        "offset 5..6   (2 bytes)  rec_count   unsigned 16-bit BIG-ENDIAN int\n"
        "then, repeated rec_count times, back to back:\n"
        "    (2 bytes)  field_len   unsigned 16-bit BIG-ENDIAN int\n"
        "    (field_len bytes)  field_data   that record's string, UTF-8 encoded\n"
        "then, immediately after the last record (or right after rec_count if it is 0):\n"
        "    (4 bytes)  checksum    unsigned 32-bit BIG-ENDIAN int, equal to\n"
        "                           (sum of every byte from offset 0 up to, but NOT\n"
        "                            including, this checksum field) mod (2**32)\n"
        "No bytes may follow the checksum field.\n"
        "```\n\n"
        "## build_records(strings) -> bytes\n\n"
        "Encode `strings` (a list of Python str, possibly containing multi-byte UTF-8 "
        "characters, possibly including empty strings) into a `bytes` object matching the "
        "layout above exactly: magic, version=1, rec_count = len(strings), then each "
        "string's UTF-8-encoded bytes preceded by its 2-byte big-endian length, then the "
        "4-byte big-endian checksum computed over everything written so far.\n\n"
        "Worked example — `build_records([\"ab\"])` must return exactly this 15-byte "
        "value (shown as hex-escaped bytes):\n"
        "```\n"
        "b'BREC' + b'\\\\x01' + b'\\\\x00\\\\x01' + b'\\\\x00\\\\x02' + b'ab' + b'\\\\x00\\\\x00\\\\x01\\\\xe3'\n"
        "```\n"
        "Breaking that down: `BREC` (magic) / `\\\\x01` (version 1) / `\\\\x00\\\\x01` "
        "(rec_count=1) / `\\\\x00\\\\x02` (field_len=2, since \"ab\" is 2 UTF-8 bytes) / "
        "`ab` (the field data) / `\\\\x00\\\\x00\\\\x01\\\\xe3` (checksum = 483, i.e. the "
        "sum of every one of the 11 preceding bytes' unsigned values, mod 2**32).\n\n"
        "## parse_records(data) -> list[str]\n\n"
        "Parse `data` back into the list of strings (the inverse of build_records) by "
        "reading the layout above. Raise `ValueError` (not any other exception type — "
        "catch and convert if needed) whenever `data` does not represent a valid BREC v1 "
        "record stream, including but not limited to: `data` too short to even contain "
        "the fixed header; the magic bytes are not exactly `b\"BREC\"`; the version byte "
        "is not exactly 1; a field's declared length would read past the end of `data` "
        "(before the checksum); there are not EXACTLY 4 bytes left for the checksum after "
        "the last record (too few, or extra trailing bytes); a field's bytes are not "
        "valid UTF-8; or the trailing 4-byte checksum does not match the checksum "
        "recomputed from the bytes that precede it.\n\n"
        "`parse_records(build_records(strings)) == strings` must hold for any list of "
        "strings, including an empty list and strings containing multi-byte UTF-8 "
        "characters and empty strings.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check both "
        "functions yourself before finishing: round-trip a couple of string lists "
        "(including an empty list and a string with non-ASCII characters), reproduce the "
        "worked example above and confirm the exact bytes match, and confirm that feeding "
        "parse_records some deliberately corrupted bytes (e.g. flipped magic, flipped "
        "version, truncated data, or a flipped byte in an otherwise-valid blob) raises "
        "ValueError in every case."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the byte-layout
    spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "brec_mod",
            "functions": ["build_records", "parse_records"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: two small, tightly-coupled functions in one file, no external I/O —
    same single-leaf rationale as c01/c04/c07/c27."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write brec_mod.py implementing build_records(strings: list[str]) -> "
                    "bytes and parse_records(data: bytes) -> list[str] for the BREC v1 "
                    "binary format. Exact layout: 4 bytes magic b'BREC', 1 byte version "
                    "(must be 1), 2 bytes rec_count (big-endian uint16), then per record "
                    "back to back: 2 bytes field_len (big-endian uint16) + field_len bytes "
                    "of that string's UTF-8 encoding, then finally 4 bytes checksum "
                    "(big-endian uint32) = sum of every byte from offset 0 up to but not "
                    "including the checksum field, mod 2**32. No bytes may follow the "
                    "checksum. build_records encodes this exactly (use Python's struct "
                    "module, '>H' for uint16 and '>I' for uint32, big-endian). "
                    "parse_records is the inverse and must raise ValueError (not any other "
                    "exception) for: too-short data, wrong magic, wrong version, a field "
                    "length that reads past the end of data, anything other than exactly "
                    "4 bytes remaining for the checksum after the last record, invalid "
                    "UTF-8 in a field, or a checksum that doesn't match what's recomputed "
                    "from the preceding bytes. parse_records(build_records(x)) == x must "
                    "hold for any list of strings including [] and unicode/empty-string "
                    "entries. Use write_file to create brec_mod.py, then use run_python to "
                    "round-trip a few string lists, verify build_records(['ab']) produces "
                    "exactly b'BREC' + b'\\x01' + b'\\x00\\x01' + b'\\x00\\x02' + b'ab' + "
                    "b'\\x00\\x00\\x01\\xe3', and confirm several deliberately-corrupted "
                    "byte strings (bad magic, bad version, truncated, flipped byte) all "
                    "raise ValueError from parse_records. Fix issues with patch_file/"
                    "run_python until confident, then finish."
                ),
                "expect": "brec_mod.py written, defining build_records/parse_records "
                          "matching the BREC v1 byte layout, sanity-checked with run_python",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm brec_mod.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["brec_mod.py"]},
    }
