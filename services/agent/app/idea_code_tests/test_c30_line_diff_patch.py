"""
codebench task c30 — hard/hidden, string/text algorithms: line-based diff + patch round trip.

New category coverage beyond the existing suite's RLE (c02) / balanced-brackets (c03) /
Roman-numeral (c13) string tasks — this one is about EDIT SCRIPTS between two sequences of
text lines, not encoding a single string.

Ground truth for ``diff_lines``/``apply_patch`` verified with an independently-written DP
longest-common-subsequence (LCS) reference implementation (a bottom-up table, walked forward
producing ops, cross-checked against Python's own ``difflib.SequenceMatcher`` match lengths
for every case below with zero mismatches) BEFORE any value here was written down — see
idea_code_test_c30_test.py for the reimplementation that re-derives this independently again.
Do not hand-edit these literals without re-deriving.

Op format: ``diff_lines(old, new)`` returns a list of ``[tag, line]`` pairs, tag in
{"equal", "delete", "insert"}. Applying them left to right against an old-index pointer i
(starting at 0): "equal" requires old[i] == line, copies it to the output, advances i;
"delete" requires old[i] == line, advances i without copying; "insert" copies line to the
output without advancing i. A valid diff consumes old exactly (i must equal len(old) at the
end) and its ops reconstruct new exactly.

Some cases below admit only ONE minimal edit script (identical sequences, pure append, pure
prepend, insert-into-empty, delete-all) — those are pinned as an EXACT expected op list.
Others admit multiple equally-minimal edit scripts (when the longest common subsequence
itself is not unique, e.g. duplicate lines, or when a delete/insert pair at the same position
could be emitted in either order) — those are graded structurally instead, via an
``is_valid_diff(old, new, ops)`` checker (mirrors c06's ``is_valid_topo_order`` pattern) plus
an ``op_counts(ops)`` tuple checked against the independently-computed LCS length, so a
"delete everything, reinsert everything" non-diff still fails even though it round-trips.

    diff_lines([], [])                                       -> []
    diff_lines(["a","b","c"], ["a","b","c"])                  -> all-equal, no edits
    diff_lines(["a","b","c"], ["a","b","c","d"])              -> equal,equal,equal,insert(d)
    diff_lines(["b","c","d"], ["a","b","c","d"])              -> insert(a),equal,equal,equal
    diff_lines([], ["a","b","c"])                             -> insert(a),insert(b),insert(c)
    diff_lines(["a","b","c"], [])                             -> delete(a),delete(b),delete(c)
    diff_lines(["a","b","c","d","e"], ["a","x","c","d","f"])  -> counts (equal=3, delete=2, insert=2)
    diff_lines(["a","b"], ["x","y"])                          -> counts (equal=0, delete=2, insert=2)
    diff_lines(["a","a","b"], ["a","b","a"])                  -> counts (equal=2, delete=1, insert=1)
        (LCS of ["a","a","b"]/["a","b","a"] has length 2 -- either "ab" or "aa" -- ambiguous
        which one, so only the counts are pinned, not the exact op list)

apply_patch contract errors (ops that do not exactly consume old raise ValueError):
    apply_patch(["a","b"], [["equal","a"]])                    -> ValueError (old not fully consumed)
    apply_patch(["a"], [["equal","a"],["equal","b"]])          -> ValueError (ran out of old)
"""
from __future__ import annotations

CANONICAL_TEST_PATH = "tests/test_line_diff.py"

_TEST_FILE_CONTENT = '''\
import pytest
from line_diff import diff_lines, apply_patch


def is_valid_diff(old: list, new: list, ops: list) -> bool:
    """True iff `ops` is a well-formed edit script from `old` to `new`: every 'equal'/'delete'
    op must reference the next unconsumed line of `old` (in order, matching by value), every
    'equal'/'insert' op contributes its line to the output in order, `old` is consumed exactly
    once end to end (no skipping, no reordering, no reuse), and the reconstructed output
    equals `new` exactly. Used below for cases where more than one minimal edit script exists,
    so a single hardcoded op list can't be the expected value."""
    out = []
    i = 0
    for op in ops:
        tag, line = op[0], op[1]
        if tag == "equal":
            if i >= len(old) or old[i] != line:
                return False
            out.append(line)
            i += 1
        elif tag == "delete":
            if i >= len(old) or old[i] != line:
                return False
            i += 1
        elif tag == "insert":
            out.append(line)
        else:
            return False
    if i != len(old):
        return False
    return out == new


def op_counts(ops: list) -> tuple:
    equal = sum(1 for op in ops if op[0] == "equal")
    delete = sum(1 for op in ops if op[0] == "delete")
    insert = sum(1 for op in ops if op[0] == "insert")
    return equal, delete, insert


def test_round_trip_empty_to_empty():
    assert apply_patch([], diff_lines([], [])) == []


def test_identical_sequences_produce_no_edits():
    old = ["a", "b", "c"]
    ops = diff_lines(old, old)
    assert ops == [["equal", "a"], ["equal", "b"], ["equal", "c"]]


def test_pure_append_at_end():
    old, new = ["a", "b", "c"], ["a", "b", "c", "d"]
    ops = diff_lines(old, new)
    assert ops == [["equal", "a"], ["equal", "b"], ["equal", "c"], ["insert", "d"]]
    assert apply_patch(old, ops) == new


def test_pure_prepend_at_start():
    old, new = ["b", "c", "d"], ["a", "b", "c", "d"]
    ops = diff_lines(old, new)
    assert ops == [["insert", "a"], ["equal", "b"], ["equal", "c"], ["equal", "d"]]
    assert apply_patch(old, ops) == new


def test_insert_into_empty_old():
    ops = diff_lines([], ["a", "b", "c"])
    assert ops == [["insert", "a"], ["insert", "b"], ["insert", "c"]]


def test_delete_all_from_old():
    ops = diff_lines(["a", "b", "c"], [])
    assert ops == [["delete", "a"], ["delete", "b"], ["delete", "c"]]


def test_middle_change_finds_real_common_subsequence():
    old, new = ["a", "b", "c", "d", "e"], ["a", "x", "c", "d", "f"]
    ops = diff_lines(old, new)
    assert op_counts(ops) == (3, 2, 2)
    assert apply_patch(old, ops) == new
    assert is_valid_diff(old, new, ops)


def test_complete_replacement_no_common_lines():
    old, new = ["a", "b"], ["x", "y"]
    ops = diff_lines(old, new)
    assert op_counts(ops) == (0, 2, 2)
    assert apply_patch(old, ops) == new


def test_duplicate_lines_ambiguous_lcs_still_round_trips():
    old, new = ["a", "a", "b"], ["a", "b", "a"]
    ops = diff_lines(old, new)
    assert op_counts(ops) == (2, 1, 1)
    assert is_valid_diff(old, new, ops)


def test_apply_patch_direct_with_hand_built_ops():
    old = ["a", "b", "c"]
    ops = [["equal", "a"], ["delete", "b"], ["insert", "x"], ["equal", "c"]]
    assert apply_patch(old, ops) == ["a", "x", "c"]


def test_apply_patch_rejects_incomplete_patch():
    with pytest.raises(ValueError):
        apply_patch(["a", "b"], [["equal", "a"]])


def test_apply_patch_rejects_overrunning_patch():
    with pytest.raises(ValueError):
        apply_patch(["a"], [["equal", "a"], ["equal", "b"]])


def test_round_trip_on_larger_multiline_pair():
    old = ["line1", "line2", "line3", "line4", "line5", "line6"]
    new = ["line1", "lineX", "line3", "lineY", "line5", "line6", "line7"]
    ops = diff_lines(old, new)
    assert apply_patch(old, ops) == new
    assert is_valid_diff(old, new, ops)


def test_round_trip_on_reordered_lines():
    old, new = ["alpha", "beta", "gamma"], ["gamma", "beta", "alpha"]
    ops = diff_lines(old, new)
    assert apply_patch(old, ops) == new
    assert is_valid_diff(old, new, ops)
'''

# Cases with exactly one minimal edit script (identical / pure append / pure prepend), the two
# genuine-LCS cases that distinguish "found the real longest common subsequence" from "deleted
# everything and reinserted it" (middle-change, complete-replacement), the tie-broken duplicate-
# lines case, apply_patch tested directly (independent of diff_lines), the incomplete-patch
# ValueError contract, and the larger multi-line round trip are what actually gate the score.
# The two empty/insert-only/delete-only degenerate cases, the second (overrunning) ValueError
# variant, and the reordered-lines round trip are bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{CANONICAL_TEST_PATH}::test_identical_sequences_produce_no_edits",
    f"{CANONICAL_TEST_PATH}::test_pure_append_at_end",
    f"{CANONICAL_TEST_PATH}::test_pure_prepend_at_start",
    f"{CANONICAL_TEST_PATH}::test_middle_change_finds_real_common_subsequence",
    f"{CANONICAL_TEST_PATH}::test_complete_replacement_no_common_lines",
    f"{CANONICAL_TEST_PATH}::test_duplicate_lines_ambiguous_lcs_still_round_trips",
    f"{CANONICAL_TEST_PATH}::test_apply_patch_direct_with_hand_built_ops",
    f"{CANONICAL_TEST_PATH}::test_apply_patch_rejects_incomplete_patch",
    f"{CANONICAL_TEST_PATH}::test_round_trip_on_larger_multiline_pair",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c30",
        "title": "line-diff-patch-round-trip",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `line_diff.py` that defines two functions:\n"
        "`diff_lines(old: list, new: list) -> list` and "
        "`apply_patch(old: list, ops: list) -> list`.\n\n"
        "Both `old` and `new` are lists of strings (think: the lines of two versions of a "
        "text file). `diff_lines` computes an edit script that transforms `old` into `new`; "
        "`apply_patch` applies that edit script back to `old` to reconstruct `new`.\n\n"
        "## Op format\n\n"
        "An edit script is a list of `[tag, line]` pairs, where `tag` is one of the strings "
        "`\"equal\"`, `\"delete\"`, or `\"insert\"`, and `line` is one of the actual string "
        "lines involved.\n\n"
        "`apply_patch(old, ops)` must process `ops` in order while walking an index `i` "
        "into `old`, starting at `i = 0`:\n"
        "- `[\"equal\", line]` — `old[i]` must equal `line`; append it to the output, then "
        "advance `i` by 1.\n"
        "- `[\"delete\", line]` — `old[i]` must equal `line`; advance `i` by 1 WITHOUT "
        "appending anything to the output.\n"
        "- `[\"insert\", line]` — append `line` to the output; do NOT advance `i` (this line "
        "isn't coming from `old` at all).\n\n"
        "After processing every op, `i` must equal `len(old)` exactly — every line of `old` "
        "was accounted for by exactly one `equal` or `delete` op, in order, none skipped and "
        "none reused. If at any point `old[i]` would be out of range (there's no line left to "
        "match), OR after processing all ops `i` is still less than `len(old)` (some lines of "
        "`old` were never consumed), `apply_patch` must raise `ValueError`. When "
        "`apply_patch` succeeds, the output list it returns is exactly what `diff_lines` was "
        "asked to transform `old` into.\n\n"
        "## What `diff_lines` must compute\n\n"
        "`diff_lines(old, new)` must return an edit script (in the format above) such that "
        "`apply_patch(old, diff_lines(old, new)) == new`. That alone is not the whole "
        "requirement, though — a `diff_lines` that always emits `len(old)` deletes followed "
        "by `len(new)` inserts would technically satisfy that round trip, but it is not a "
        "real diff. Instead, `diff_lines` must be based on the LONGEST COMMON SUBSEQUENCE "
        "(LCS) of `old` and `new`: a common subsequence is a sequence of lines that appears "
        "in both `old` and `new`, in the same relative order (not necessarily contiguous in "
        "either), and the LCS is the longest such sequence. The number of `\"equal\"` ops in "
        "your returned edit script must equal the length of the LCS of `old` and `new` — "
        "every line that is genuinely part of some longest common subsequence should end up "
        "matched by an `\"equal\"` op rather than a `delete`+`insert` pair. Lines from `old` "
        "that are NOT part of the LCS you chose become `\"delete\"` ops (in `old`'s order); "
        "lines from `new` that are NOT part of the LCS you chose become `\"insert\"` ops (in "
        "`new`'s order); the `equal`/`delete`/`insert` ops must be interleaved so that "
        "`apply_patch` reconstructs `new` exactly.\n\n"
        "When `old` and `new` share no common lines, `diff_lines` must return zero `equal` "
        "ops. When `old == new`, it must return all `equal` ops and zero `delete`/`insert` "
        "ops. When the longest common subsequence is not unique (e.g. duplicate lines create "
        "more than one longest common subsequence of the same length), any one of the "
        "maximal-length choices is acceptable — the exact op list isn't graded in that "
        "case, only that the round trip holds and the `equal` count matches the true LCS "
        "length.\n\n"
        "Worked examples:\n"
        "  - `diff_lines([\"a\",\"b\",\"c\"], [\"a\",\"b\",\"c\",\"d\"])` -> "
        "`[[\"equal\",\"a\"],[\"equal\",\"b\"],[\"equal\",\"c\"],[\"insert\",\"d\"]]` (pure "
        "append: the whole common prefix stays `equal`, then one `insert`).\n"
        "  - `diff_lines([\"b\",\"c\",\"d\"], [\"a\",\"b\",\"c\",\"d\"])` -> "
        "`[[\"insert\",\"a\"],[\"equal\",\"b\"],[\"equal\",\"c\"],[\"equal\",\"d\"]]` (pure "
        "prepend).\n"
        "  - `diff_lines([\"a\",\"b\"], [\"x\",\"y\"])` -> two `delete`s and two `insert`s, "
        "zero `equal` ops (nothing in common).\n\n"
        "There is no visible test file for this task. Implement a standard dynamic-"
        "programming LCS table (or an equivalent approach) to find the LCS length and an "
        "actual longest common subsequence between `old` and `new`, then derive the edit "
        "script from it. Use run_python to sanity-check both functions yourself — build a "
        "handful of `old`/`new` pairs (including at least one where they share no lines, one "
        "where they're identical, and one with a real middle change), confirm "
        "`apply_patch(old, diff_lines(old, new)) == new` for each, and confirm the `equal` "
        "op count matches the common-line count you'd expect by inspection — before "
        "finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {CANONICAL_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "line_diff", "functions": ["diff_lines", "apply_patch"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write line_diff.py implementing two functions.\n\n"
                    "diff_lines(old: list, new: list) -> list: return an edit script — a "
                    "list of [tag, line] pairs, tag in 'equal'/'delete'/'insert' — based on "
                    "the LONGEST COMMON SUBSEQUENCE (LCS) of old and new (build a standard "
                    "DP LCS table, e.g. dp[i][j] = length of LCS of old[i:] and new[j:], "
                    "then walk it to emit ops: where old[i]==new[j] emit an 'equal' op and "
                    "advance both; otherwise follow whichever of dp[i+1][j]/dp[i][j+1] is "
                    "larger, emitting 'delete' (advance old only) or 'insert' (advance new "
                    "only) respectively; when both old and new are exhausted stop, then flush "
                    "any remaining old lines as deletes and any remaining new lines as "
                    "inserts). The number of 'equal' ops in the result MUST equal the true "
                    "LCS length of old and new — do not just delete everything and reinsert "
                    "it, that has zero 'equal' ops and will be graded as wrong even though it "
                    "happens to round-trip.\n\n"
                    "apply_patch(old: list, ops: list) -> list: walk an index i into old "
                    "starting at 0. For ['equal', line]: old[i] must equal line, append it to "
                    "the output, i += 1. For ['delete', line]: old[i] must equal line, i += 1 "
                    "(nothing appended). For ['insert', line]: append line to the output, i "
                    "unchanged. If old[i] would be out of range when needed, OR i != len(old) "
                    "after all ops are processed, raise ValueError (the patch doesn't apply "
                    "cleanly). Otherwise return the output list.\n\n"
                    "Use write_file to create line_diff.py. There is no pre-existing test "
                    "file — use run_python to sanity-check: build several old/new pairs of "
                    "your own (identical lists, completely disjoint lists, a list with a "
                    "real change in the middle, an empty list on one side), confirm "
                    "apply_patch(old, diff_lines(old, new)) == new for each, and for the "
                    "middle-change case count by hand how many lines are common between old "
                    "and new (in order) and confirm the number of 'equal' ops in your output "
                    "matches that hand count exactly -- do not just delete everything and "
                    "reinsert it, which round-trips correctly but has zero 'equal' ops and "
                    "will be graded as wrong. Also confirm apply_patch raises ValueError when "
                    "given an incomplete op list that doesn't fully consume old. Fix issues "
                    "with patch_file/run_python until confident, then finish."
                ),
                "expect": "line_diff.py written, implementing an LCS-based diff_lines and a "
                          "strict apply_patch with a ValueError contract on malformed patches",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm line_diff.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["line_diff.py"]},
    }
