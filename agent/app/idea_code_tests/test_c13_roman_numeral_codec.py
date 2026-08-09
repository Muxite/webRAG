"""
codebench task c13 — hard/visible, format codec (Roman numerals).

Ground truth for ``to_roman``/``from_roman`` verified with TWO structurally independent
reference implementations (greedy-subtraction table walk, and a per-digit lookup-table
method), cross-checked against each other across the ENTIRE valid range 1..3999 with zero
mismatches before any value below was embedded — do not change these without re-verifying
the same way (subtractive notation, e.g. CD vs CCCC, is exactly the kind of thing that's
easy to get subtly wrong by hand):
    to_roman(1)    -> "I"
    to_roman(4)    -> "IV"
    to_roman(9)    -> "IX"
    to_roman(40)   -> "XL"
    to_roman(90)   -> "XC"
    to_roman(400)  -> "CD"
    to_roman(900)  -> "CM"
    to_roman(3999) -> "MMMCMXCIX"
    from_roman("MMMDCCCLXXXVIII") -> 3888
    round-trips: from_roman(to_roman(n)) == n for n in {44, 1994, 2023}
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_roman_codec.py"

_TEST_FILE_CONTENT = '''\
import pytest
from roman_codec import to_roman, from_roman


def test_to_roman_one():
    assert to_roman(1) == "I"


def test_to_roman_iv():
    assert to_roman(4) == "IV"


def test_to_roman_ix():
    assert to_roman(9) == "IX"


def test_to_roman_xl():
    assert to_roman(40) == "XL"


def test_to_roman_xc():
    assert to_roman(90) == "XC"


def test_to_roman_cd():
    assert to_roman(400) == "CD"


def test_to_roman_cm():
    assert to_roman(900) == "CM"


def test_to_roman_max():
    assert to_roman(3999) == "MMMCMXCIX"


def test_from_roman_direct():
    assert from_roman("MMMDCCCLXXXVIII") == 3888


def test_round_trip_44():
    assert from_roman(to_roman(44)) == 44


def test_round_trip_1994():
    assert from_roman(to_roman(1994)) == 1994


def test_round_trip_2023():
    assert from_roman(to_roman(2023)) == 2023


def test_to_roman_zero_raises():
    with pytest.raises(ValueError):
        to_roman(0)


def test_to_roman_over_max_raises():
    with pytest.raises(ValueError):
        to_roman(4000)


def test_to_roman_negative_raises():
    with pytest.raises(ValueError):
        to_roman(-5)
'''

# The subtractive-notation cases, the max-value case, and the round-trip/direct-decode cases
# are what actually distinguish "implemented the encoding correctly" from "got lucky on a
# simple additive case"; test_to_roman_one (degenerate, no subtraction involved) and the
# three ValueError boundary contracts (input validation, not codec logic) are bonus credit
# only, not keystone — mirrors c01's convention exactly.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_to_roman_iv",
    f"{VISIBLE_TEST_PATH}::test_to_roman_ix",
    f"{VISIBLE_TEST_PATH}::test_to_roman_xl",
    f"{VISIBLE_TEST_PATH}::test_to_roman_xc",
    f"{VISIBLE_TEST_PATH}::test_to_roman_cd",
    f"{VISIBLE_TEST_PATH}::test_to_roman_cm",
    f"{VISIBLE_TEST_PATH}::test_to_roman_max",
    f"{VISIBLE_TEST_PATH}::test_from_roman_direct",
    f"{VISIBLE_TEST_PATH}::test_round_trip_44",
    f"{VISIBLE_TEST_PATH}::test_round_trip_1994",
    f"{VISIBLE_TEST_PATH}::test_round_trip_2023",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c13",
        "title": "roman-numeral-codec",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `roman_codec.py` that defines two functions:\n"
        "`to_roman(n: int) -> str` and `from_roman(s: str) -> int`.\n\n"
        "Both must implement standard Roman numeral encoding/decoding, including "
        "subtractive notation: 4 is `IV` (not `IIII`), 9 is `IX`, 40 is `XL`, 90 is `XC`, "
        "400 is `CD` (not `CCCC`), and 900 is `CM`. The seven symbols and their values are "
        "I=1, V=5, X=10, L=50, C=100, D=500, M=1000.\n\n"
        "`to_roman` is only defined for `1 <= n <= 3999`. If `n` is outside that range "
        "(including 0, negative numbers, or anything >= 4000), `to_roman` must raise "
        "`ValueError`.\n\n"
        "`from_roman` takes a valid Roman numeral string (uppercase, using the subtractive "
        "forms above) and returns the integer it represents. You do not need to validate "
        "malformed input to `from_roman` — only well-formed Roman numerals in the range "
        "1..3999 will be passed to it.\n\n"
        "A visible test file is already present at tests/test_roman_codec.py — run it "
        "(run_pytest) and keep revising roman_codec.py until every test in it passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts. For a visible
    task this includes the real test file (grading still re-injects a pristine canonical
    copy from private/tests/ regardless — see materialize_task.py / run_grade.sh)."""
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "roman_codec", "functions": ["to_roman", "from_roman"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Hand-authored offline plan (mirrors idea_tests/'s get_compiled_plan() convention) —
    a single leaf is enough for a two-function codec task; multi-leaf plans are for tasks
    that genuinely decompose into independent pieces."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write roman_codec.py implementing to_roman(n: int) -> str and "
                    "from_roman(s: str) -> int using standard Roman numeral rules with "
                    "subtractive notation (4=IV, 9=IX, 40=XL, 90=XC, 400=CD, 900=CM). "
                    "to_roman must raise ValueError for n outside 1..3999. Use write_file "
                    "to create it. Then use run_pytest on tests/test_roman_codec.py. If "
                    "any test fails, use read_file/patch_file to fix roman_codec.py and "
                    "run_pytest again. Once every test passes, finish."
                ),
                "expect": "roman_codec.py written; tests/test_roman_codec.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm roman_codec.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["roman_codec.py"]},
    }
