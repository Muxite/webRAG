"""
codebench task c40 — hard/visible, DEBUG an existing broken module (date-range overlap).

REVISION NOTE (2026-08-06): the original version of this task (one `b_start < a_end` vs
`b_start <= a_end` boundary flip, with the visible test file textually identical to the
canonical grading test file, AND — the real smoking gun — a compiled-plan leaf instruction
that spelled out the exact fix formula in prose) scored 1.0/1.0 for qwen2.5:14b on BOTH the
badmodel and aider codebench harnesses in live calibration. This revision does two things:
(1) the compiled plan below no longer states the formula or names which comparison is wrong
— see get_compiled_plan()'s docstring; (2) the function grows one more piece of documented
contract — an optional `touching_counts` flag — that the shipped starter silently ignores
(a realistic "added the parameter to the signature, forgot to actually wire it into the
logic" bug), and that flag is exercised ONLY by canonical test cases that are graded but
never shown to the agent (get_grading_payload()["tests"] is now a strict superset of
get_sandbox_fixture(), at the same relpath — the agent's own copy of the test file is still
manifest-dropped before grading, so this cannot be defeated by editing the visible copy).

``range_overlap.py`` implements the standard "do two inclusive integer ranges overlap" check
(ranges represent e.g. day-number spans). The full, documented contract is:
``ranges_overlap(a_start, a_end, b_start, b_end, touching_counts=True)`` — when
``touching_counts`` is True (the default, and the only value the VISIBLE tests exercise),
ranges that merely touch at a shared boundary day count as overlapping; when it is False,
they must share more than a single boundary point.

The shipped starter has TWO problems, both real, neither one alone sufficient:
  (1) the touching_counts=True formula uses `b_start < a_end` where it must be `<=` — a
      too-strict boundary comparison, the "obvious" bug that the visible failing tests point
      straight at.
  (2) `touching_counts` is accepted as a parameter but never read anywhere in the function
      body — the function behaves identically regardless of what is passed for it.

Fixing ONLY (1) (the natural first move when chasing the visible red tests) makes every
VISIBLE test pass, because none of them pass touching_counts at all — but the function still
silently ignores the flag, so it is wrong for any caller that actually relies on
touching_counts=False, which only the hidden canonical tests exercise. Genuinely reading and
implementing the documented contract (not just pattern-matching the shown assertions) is
required to pass the hidden set.

Ground truth for ``ranges_overlap`` verified two independent ways in
idea_code_test_c40_test.py: a boolean-negation formulation (``not (a_end < b_start or
b_end < a_start)`` for the touching_counts=True case, and literal calendar-day-set
intersection — no algebra at all — for the touching_counts=False case) AND a max/min-overlap-
window formulation (``max(a_start, b_start) < min(a_end, b_end)`` style, generalized to both
flag values) — both must agree with every literal value embedded here. That file also actually
EXECUTES the shipped buggy starter, the "obvious boundary-only" partial fix, and the fully
correct joint fix, and proves each one's exact pass/fail signature against both the visible
and hidden case sets.

REVISION NOTE (2026-08-07): round-2 live calibration showed Aider still acing this task (1.0)
even with the touching_counts-wiring trap above in place, and its own actual submitted fix for
the touching_counts=False branch — ``a_start <= b_end and b_start <= a_end and a_start !=
b_end and b_start != a_end`` — is REVEALING: it correctly wires the flag in and correctly
handles every boundary-touch case in this suite, but it is still WRONG in general. That formula
(and, it turns out, this task's own previously-"fully correct" reference fix,
``a_start < b_end and b_start < a_end``) both silently mishandle a degenerate case: when ONE of
the two ranges is itself a SINGLE DAY (`a_start == a_end`) sitting entirely inside the other,
away from either of its boundaries (e.g. `[5, 5]` inside `[1, 10]`), the two ranges can share AT
MOST that one day — by definition, NOT "more than a single day" — yet both of those formulas
return True (verified by brute-force day-set enumeration in idea_code_test_c40_test.py: 0
mismatches against the correct window formula across 400,000 random cases, EXCEPT this exact
degenerate-single-day-range shape, which is where both of those plausible-looking two-term
formulas diverge from ground truth). The genuinely correct touching_counts=False condition
needs the OVERLAP WINDOW itself to have positive width (``max(a_start, b_start) <
min(a_end, b_end)``), not merely "the two ranges don't touch at a_start/b_end or b_start/a_end"
— those are only the same thing when NEITHER range is degenerate, an assumption both of the
above two-term formulas silently make without checking. A NEW hidden-only case exercises this;
it discriminates Aider's own actual submitted fix specifically, not a hypothetical mutant.
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_range_overlap.py"
STARTER_MODULE_PATH = "range_overlap.py"

_RANGE_OVERLAP_PY_CONTENT = '''\
def ranges_overlap(a_start, a_end, b_start, b_end, touching_counts=True):
    """Two inclusive integer ranges [a_start, a_end] and [b_start, b_end].

    Returns True if they overlap, False if they are strictly disjoint. Assumes
    a_start <= a_end and b_start <= b_end.

    If touching_counts is True (the default), ranges that merely touch at a shared boundary
    point (e.g. [1, 5] and [5, 10], which share day 5) count as overlapping. If False, the
    ranges must share more than a single boundary point to count as overlapping.
    """
    return a_start <= b_end and b_start < a_end
'''

# Shown to the agent in the sandbox (get_sandbox_fixture) AND graded (it is a strict subset of
# the canonical content below, at the SAME relpath — see get_grading_payload). Every call here
# omits touching_counts (uses the default).
_TEST_FILE_CONTENT = '''\
from range_overlap import ranges_overlap


def test_clearly_overlapping():
    assert ranges_overlap(1, 10, 5, 8) is True


def test_clearly_disjoint():
    assert ranges_overlap(1, 5, 10, 15) is False


def test_b_starts_exactly_where_a_ends():
    assert ranges_overlap(1, 5, 5, 10) is True


def test_a_starts_exactly_where_b_ends():
    assert ranges_overlap(5, 10, 1, 5) is True


def test_touching_boundary_scaled_up():
    assert ranges_overlap(10, 20, 20, 30) is True


def test_single_day_ranges_same_day():
    assert ranges_overlap(5, 5, 5, 5) is True


def test_identical_ranges():
    assert ranges_overlap(3, 7, 3, 7) is True


def test_a_contains_b():
    assert ranges_overlap(1, 10, 4, 6) is True


def test_b_contains_a():
    assert ranges_overlap(4, 6, 1, 10) is True


def test_adjacent_with_gap_not_overlapping():
    assert ranges_overlap(1, 5, 6, 10) is False
'''

# NEVER shown to the agent (absent from get_sandbox_fixture) — graded only, appended to the
# visible content above to form the canonical private test file. These are the only cases in
# the whole suite that pass touching_counts explicitly, so they are the only cases that can
# catch the "flag accepted but never wired in" bug.
_HIDDEN_TEST_ADDENDUM = '''\
def test_touching_boundary_not_counted_when_flag_false():
    assert ranges_overlap(1, 5, 5, 10, touching_counts=False) is False


def test_touching_boundary_scaled_up_not_counted_when_flag_false():
    assert ranges_overlap(10, 20, 20, 30, touching_counts=False) is False


def test_single_day_ranges_not_counted_when_flag_false():
    assert ranges_overlap(5, 5, 5, 5, touching_counts=False) is False


def test_real_overlap_still_counts_when_flag_false():
    assert ranges_overlap(1, 10, 5, 8, touching_counts=False) is True


def test_explicit_flag_true_matches_default_behavior():
    assert ranges_overlap(1, 10, 5, 8, touching_counts=True) is True


def test_real_gap_still_not_overlapping_when_flag_false():
    assert ranges_overlap(1, 5, 6, 10, touching_counts=False) is False


def test_single_day_range_fully_inside_another_not_counted_when_flag_false():
    # a is a single-day range (a_start == a_end) sitting entirely inside b, away from EITHER of
    # b's own boundaries. The two ranges can share at most that one day -- by definition not
    # "more than a single day" -- even though b visually extends well past both sides of a. A
    # two-term fix that only checks "does a_start equal b_end, or b_start equal a_end" (the
    # natural-looking way to rule out boundary-touch cases) misses this: neither of those two
    # specific equalities holds here, so it wrongly concludes there's a real overlap.
    assert ranges_overlap(5, 5, 1, 10, touching_counts=False) is False


def test_single_day_range_fully_inside_another_reversed_args_not_counted_when_flag_false():
    assert ranges_overlap(1, 10, 5, 5, touching_counts=False) is False
'''

_CANONICAL_TEST_FILE_CONTENT = _TEST_FILE_CONTENT + "\n\n" + _HIDDEN_TEST_ADDENDUM

# Group A: broken on the unmodified starter by bug (1) alone (the boundary comparison) — the
# "obvious" cases the visible red tests point straight at. Group B: hidden-only, and the ONLY
# cases in the suite that can distinguish "fixed the boundary AND wired in touching_counts"
# from "fixed the boundary but still ignores touching_counts" (a fix that does only the
# former passes every visible test and still fails every one of these) — this now also
# includes the 2026-08-07 degenerate-single-day-range case, which additionally discriminates a
# fix that DOES correctly wire in touching_counts but implements the False branch with a
# plausible two-term "boundary equality" formula instead of a genuine positive-width-overlap
# check (Aider's own actual submitted fix for this task has exactly this shape). Both groups
# gate the score. The remaining cases (plain overlap/disjoint, the boundary touching the other
# direction, containment, identical ranges, a real gap — visible and hidden) already pass
# under a boundary-only fix and stay correct under the full fix too, so they are bonus credit
# only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_b_starts_exactly_where_a_ends",
    f"{VISIBLE_TEST_PATH}::test_touching_boundary_scaled_up",
    f"{VISIBLE_TEST_PATH}::test_single_day_ranges_same_day",
    f"{VISIBLE_TEST_PATH}::test_touching_boundary_not_counted_when_flag_false",
    f"{VISIBLE_TEST_PATH}::test_touching_boundary_scaled_up_not_counted_when_flag_false",
    f"{VISIBLE_TEST_PATH}::test_single_day_ranges_not_counted_when_flag_false",
    f"{VISIBLE_TEST_PATH}::test_single_day_range_fully_inside_another_not_counted_when_flag_false",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c40",
        "title": "bugfix-date-range-overlap",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "The file `range_overlap.py` already contains an implementation of "
        "`ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int, "
        "touching_counts: bool = True) -> bool` — two inclusive integer ranges (think "
        "day-number spans) `[a_start, a_end]` and `[b_start, b_end]`.\n\n"
        "When `touching_counts` is True (the default), two ranges overlap-or-touch, and the "
        "function must return True, whenever neither range is strictly to one side of the "
        "other — this INCLUDES the case where they merely touch at a shared boundary day "
        "(e.g. `[1, 5]` and `[5, 10]` overlap, because they share day 5). Only ranges with a "
        "real gap between them (one strictly ends before the other strictly begins) should "
        "return False in that mode.\n\n"
        "When `touching_counts` is False, merely touching at a single shared boundary day is "
        "NOT enough — the two ranges must share more than one day (a real, non-degenerate "
        "overlap) for the function to return True. Be careful with a DEGENERATE range that is "
        "itself only a single day (`a_start == a_end`, or `b_start == b_end`): it can never "
        "share more than one day with anything, no matter where it sits relative to the other "
        "range's boundaries, so `touching_counts=False` must return False for it every time — "
        "double-check this against a case where the single-day range sits well INSIDE the "
        "other range (not merely touching one of its endpoints), since a formula that only "
        "checks for a shared BOUNDARY POINT (rather than actually reasoning about how many "
        "days are shared) can get this specific shape wrong even though it looks correct on "
        "every boundary-touching case.\n\n"
        "There is a bug in the existing implementation: a visible test file at "
        "tests/test_range_overlap.py currently FAILS on several cases. Find the bug and fix "
        "it — do not rewrite the function from scratch. Note that the visible test file only "
        "exercises the default `touching_counts=True` behavior; make sure your fix actually "
        "implements the FULL documented contract above (both flag values), not just "
        "whatever makes the shown tests pass, since the grading criteria cover the full "
        "contract.\n\n"
        "Run the tests (run_pytest) and keep revising range_overlap.py until every test in "
        "tests/test_range_overlap.py passes, then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts: the buggy module
    itself (the agent edits this in place — same relpath, same import name `range_overlap`)
    plus the VISIBLE SUBSET of the test file. Grading re-injects the larger canonical copy of
    the test file from private/tests/ regardless (see materialize_task.py / run_grade.sh) —
    the agent's own copy of tests/test_range_overlap.py gets manifest-dropped before grading,
    same as c01/c18/c39's convention — and that canonical copy has MORE tests than this one,
    at the identical relpath (see get_grading_payload)."""
    return {
        STARTER_MODULE_PATH: _RANGE_OVERLAP_PY_CONTENT,
        VISIBLE_TEST_PATH: _TEST_FILE_CONTENT,
    }


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _CANONICAL_TEST_FILE_CONTENT},
        "entrypoint": {"module": "range_overlap", "functions": ["ranges_overlap"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: locating and fixing the bug(s) in one small file is one cohesive unit of
    work, same shape as c18/c39's single-leaf plans. Unlike this task's original version, the
    instruction below deliberately does NOT state the overlap formula or say which comparison
    (or which parameter) is broken — it restates the CONTRACT (already public in the task
    prompt) and a debugging methodology only, so a cheap executor still has to read the code,
    run the tests, and reason about the fix rather than transcribe an answer out of the plan."""
    return {
        "leaves": [
            {
                "id": "fix_bug",
                "instruction": (
                    "range_overlap.py implements ranges_overlap(a_start, a_end, b_start, "
                    "b_end, touching_counts=True) for two inclusive integer ranges, but has "
                    "at least one bug. Use read_file to review its current logic against "
                    "its own docstring/contract (pay attention to EVERY documented "
                    "parameter, not just the ones the visible tests happen to call with "
                    "non-default values), then run_pytest on tests/test_range_overlap.py to "
                    "see what's failing. Form a hypothesis about the root cause, make a "
                    "targeted change with patch_file, then run_pytest again to confirm. "
                    "Getting every visible test to pass is necessary but may not be "
                    "sufficient — double check with run_python that your fix actually "
                    "matches the full documented behavior for cases the visible test file "
                    "doesn't happen to cover, not just the specific cases you were shown "
                    "failing. In particular, don't just pattern-match a two-term inequality "
                    "formula from the boundary-touching cases you can see -- explicitly try a "
                    "case with touching_counts=False where ONE range is a single day sitting "
                    "well inside the other one (not touching either of its endpoints), and "
                    "reason from the actual definition (how many days do they really share?) "
                    "rather than assuming your formula generalizes. Keep iterating until "
                    "you're confident the fix is complete, then finish."
                ),
                "expect": "range_overlap.py fixed; tests/test_range_overlap.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm range_overlap.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["range_overlap.py"]},
    }
