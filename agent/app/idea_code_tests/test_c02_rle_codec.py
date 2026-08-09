"""
codebench task c02 — hard/visible, string encode/decode round-trip.

Ground truth for ``encode``/``decode`` (run-length encoding: consecutive repeated
characters become ``<count><char>``; a single non-repeated char still gets a count of 1)
was computed by two independently-written reference implementations (itertools.groupby
vs. manual char-scanning) that were run and cross-checked to agree exactly before any
value below was written down — see idea_code_test_c02_test.py for the reimplementation
that re-verifies this at test time. Do not hand-edit these literals without re-deriving.

    encode("aaabbc")       -> "3a2b1c"
    encode("c")            -> "1c"
    encode("aaaa")         -> "4a"
    encode("xyz")          -> "1x1y1z"   (no repeats: every run has count 1)
    decode("3a2b1c")       -> "aaabbc"
    decode("1c")           -> "c"
    decode("1a1b1c1a1b1c") -> "abcabc"
    round trips (decode(encode(s)) == s): "", "a", "aaaa", "xyz", "aabbaabbcc"
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_rle_codec.py"

_TEST_FILE_CONTENT = '''\
from rle_codec import encode, decode


def test_encode_repeated_runs():
    assert encode("aaabbc") == "3a2b1c"


def test_encode_single_char():
    assert encode("c") == "1c"


def test_encode_all_same_char():
    assert encode("aaaa") == "4a"


def test_encode_no_repeats():
    assert encode("xyz") == "1x1y1z"


def test_decode_repeated_runs():
    assert decode("3a2b1c") == "aaabbc"


def test_decode_single_char():
    assert decode("1c") == "c"


def test_decode_no_repeats():
    assert decode("1a1b1c1a1b1c") == "abcabc"


def test_round_trip_empty():
    assert decode(encode("")) == ""


def test_round_trip_single_char():
    assert decode(encode("a")) == "a"


def test_round_trip_all_same_char():
    assert decode(encode("aaaa")) == "aaaa"


def test_round_trip_no_repeats():
    assert decode(encode("xyz")) == "xyz"


def test_round_trip_mixed():
    assert decode(encode("aabbaabbcc")) == "aabbaabbcc"
'''

# The core encode/decode behaviors and the round trips on non-trivial inputs are what
# actually distinguish "implemented RLE correctly" from "got lucky on one shape"; the
# empty-string round trip is a degenerate edge case (both encode and decode are no-ops
# on "") and is bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_encode_repeated_runs",
    f"{VISIBLE_TEST_PATH}::test_encode_single_char",
    f"{VISIBLE_TEST_PATH}::test_encode_all_same_char",
    f"{VISIBLE_TEST_PATH}::test_encode_no_repeats",
    f"{VISIBLE_TEST_PATH}::test_decode_repeated_runs",
    f"{VISIBLE_TEST_PATH}::test_decode_single_char",
    f"{VISIBLE_TEST_PATH}::test_decode_no_repeats",
    f"{VISIBLE_TEST_PATH}::test_round_trip_single_char",
    f"{VISIBLE_TEST_PATH}::test_round_trip_all_same_char",
    f"{VISIBLE_TEST_PATH}::test_round_trip_no_repeats",
    f"{VISIBLE_TEST_PATH}::test_round_trip_mixed",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c02",
        "title": "rle-codec-round-trip",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `rle_codec.py` that defines two functions: "
        "`encode(s: str) -> str` and `decode(s: str) -> str`, implementing run-length "
        "encoding.\n\n"
        "`encode(s)` must replace every maximal run of consecutive identical characters "
        "with `<count><char>`, where `<count>` is the run length written in decimal with "
        "no leading zeros. A single non-repeated character still gets a count of 1 "
        "(e.g. a run of length 1 is written as `1x`, not just `x`).\n\n"
        "For example: `encode(\"aaabbc\")` must return `\"3a2b1c\"`, and `encode(\"c\")` "
        "must return `\"1c\"`.\n\n"
        "`decode(s)` must be the exact inverse of `encode`: given a string of "
        "`<count><char>` pairs concatenated together, it reconstructs the original "
        "string by repeating each char count times, in order. For all strings s made up "
        "only of the characters this codec is used on, `decode(encode(s)) == s` must "
        "hold.\n\n"
        "The empty string is a valid input to both functions: `encode(\"\")` returns "
        "`\"\"` and `decode(\"\")` returns `\"\"`.\n\n"
        "A visible test file is already present at tests/test_rle_codec.py — run it "
        "(run_pytest) and keep revising rle_codec.py until every test in it passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "rle_codec", "functions": ["encode", "decode"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write rle_codec.py implementing encode(s: str) -> str and "
                    "decode(s: str) -> str for run-length encoding: encode() maps each "
                    "maximal run of identical characters to '<count><char>' (a single "
                    "char still gets count 1, e.g. 'c' -> '1c'); decode() is the exact "
                    "inverse of encode(). Both must handle the empty string. Use "
                    "write_file to create it. Then use run_pytest on "
                    "tests/test_rle_codec.py. If any test fails, use "
                    "read_file/patch_file to fix rle_codec.py and run_pytest again. Once "
                    "every test passes, finish."
                ),
                "expect": "rle_codec.py written; tests/test_rle_codec.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm rle_codec.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["rle_codec.py"]},
    }
