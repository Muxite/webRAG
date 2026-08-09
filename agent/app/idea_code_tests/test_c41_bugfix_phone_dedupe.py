"""
codebench task c41 — hard/visible, DEBUG an existing broken module (dedupe phone numbers,
preserve order, with a bespoke multi-step normalization rule).

SECOND REPLACEMENT NOTE (2026-08-07): the previous version of this task (`dedupe_list.py`,
generic case/whitespace-insensitive string dedup) went through two hardening rounds and was
STILL fully aced (1.0/1.0) by qwen2.5:14b on BOTH the badmodel and Aider codebench harnesses in
round 3. Inspecting the two real round-3 submissions showed why: they are algorithmically
IDENTICAL in shape and genuinely, fully correct —
```python
# Aider's submission
from collections import OrderedDict
def dedupe_preserve_order(items):
    seen = OrderedDict()
    for item in items:
        normalized_item = item.strip().lower()
        if normalized_item not in seen:
            seen[normalized_item] = item
    return list(seen.values())

# badmodel's submission
def dedupe_preserve_order(items):
    seen_items = set()
    output = []
    for item in items:
        clean_item = item.lower().strip()
        if clean_item not in seen_items:
            output.append(item)
            seen_items.add(clean_item)
    return output
```
Both correctly implement "walk in order, key a seen-table by `.strip().lower()`, keep first
occurrence's original string" — including blank/whitespace-only items, which neither one special-
cases or drops. `.strip().lower()`-keyed order-preserving dedup is an extremely common, nearly
one-line Python idiom; once a model has correctly identified "this needs case/whitespace-
insensitive normalization," there is essentially only one natural way to write it, and that one
natural way is simply correct. No amount of additional test cases WITHIN that exact normalization
rule (more case variants, more whitespace variants, more blank-item variants) can catch an
implementation that is not almost-right but flat-out right. This is a full domain replacement
(same precedent as c37/c39's full replacements after repeated hardening rounds against the same
underlying shape): the normalization rule itself is now bespoke and multi-step, deliberately
chosen so that ".strip().lower()" — or any comparison based on the STRING's characters directly —
has nothing to do with the actual equivalence relation, so the "one natural correct one-liner"
escape hatch that defeated the previous design does not exist here.

The new task: `phone_dedupe.py` implements `dedupe_phone_numbers(numbers: list[str]) -> list[str]`
— dedupe a list of phone-number strings, preserving first-occurrence order, where two entries are
the SAME phone number if their NORMALIZED DIGIT CORES match:
  1. Strip every character that is not a digit 0-9 (drop spaces, dashes, dots, parentheses, plus
     signs, letters, everything).
  2. If the remaining digit string is EXACTLY 11 digits long AND its first digit is specifically
     `'1'` (a US country code), drop that leading `1` — the remaining 10 digits are the core.
  3. Otherwise (any other length, or an 11-digit string that does NOT start with `1`), the digit
     string from step 1 is the core, UNCHANGED.
The value kept in the output is always the exact original string (formatting untouched) from
wherever its core first appeared.

The shipped starter has the same "obvious," visible-test-breaking bug as the previous design
(`sorted(set(numbers))` — exact-string dedup, order thrown away, no normalization at all,
directly visible from the first `run_pytest`). Fixing ONLY the ordering — an order-preserving
walk keyed by exact string equality, or even one keyed by `.strip().lower()` (the natural
first move once you notice "the bug is about order," and the SAME formula that was fully
correct for the previous version of this task) — makes every VISIBLE test pass (all visible
duplicates are byte-for-byte identical) while getting every single hidden case wrong, because
case/whitespace normalization has nothing to do with recognizing that "(415) 555-0100" and
"415-555-0100" and "+1 415-555-0100" are the same phone number.

Ground truth for `dedupe_phone_numbers` verified two independent ways in
idea_code_test_c41_test.py: a `re.sub(r"\\D", "", s)` digit-extraction formulation and a
completely separate `"".join(ch for ch in s if ch.isdigit())` formulation (no regex engine
involved at all) — both must agree with every literal value embedded here. That file also
actually EXECUTES the shipped buggy starter, the "obvious order-preserving-but-unnormalized"
fix, THREE independently plausible near-miss mutants (forgets the country-code rule entirely;
over-generalizes it to unconditionally drop the first digit of any 11-digit string regardless of
whether it's a `1`; and drops blank/no-digit entries instead of deduping them), and the fully
correct fix — proving each one's exact pass/fail signature against the real canonical suite via
pytest. It also directly re-runs the ACTUAL round-3 submitted source code from both the badmodel
and Aider harnesses (reproduced verbatim from
badmodel-lab/codebench/results/runs/coordinator_batch4/c41__{aider,badmodel}__qwen2.5_14b/
submission/dedupe_list.py, adapted only to this task's new function/module name) against the new
canonical suite: both fail 7 of the 9 hidden cases, since neither implements ANY digit-based
normalization at all.

KNOWN LIMITATION (2026-08-07): live round-4 calibration on this phone-dedupe redesign showed
badmodel 0.5882 (genuinely hard for it) but Aider/qwen2.5:14b 1.0/1.0 — it read the multi-step
normalization rule from the task statement and transcribed it into fully correct code on the
first attempt (verified: its submission is a direct, correct implementation of the 3-step rule
above, not a lucky near-miss). This is the third codebench task id (after c22's two replacements
and c37's two replacements) where Aider paired with qwen2.5:14b reliably converts an unambiguous,
precisely-specified rule into correct code regardless of how unusual the domain is chosen to be —
by this point a strong, general finding about this harness/model combination's ceiling on
single-function, fully-specified debugging tasks, not a fixable flaw in any one task's design.
Per the project's own calibration discipline (a task must be reworked/rejected, not shipped
indefinitely, if it keeps failing to discriminate — see CODEBENCH_HANDOFF.md), this task is being
ACCEPTED AS-IS after 4 live-calibration rounds (2 designs x 2 rounds each) rather than getting a
3rd full replacement: it still has genuine, real discriminating power against badmodel (0.59, a
meaningfully hard result, not a giveaway), and per-harness ceilings differing is itself useful
signal about relative agent-harness capability, not solely a task defect. A future session could
still choose a 3rd replacement domain if a stronger idea presents itself.
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_phone_dedupe.py"
STARTER_MODULE_PATH = "phone_dedupe.py"

_PHONE_DEDUPE_PY_CONTENT = '''\
def dedupe_phone_numbers(numbers):
    """Return a new list of phone-number strings with duplicates removed, keeping only each
    number's FIRST occurrence, in the same relative order it first appeared in ``numbers``.

    Two entries represent the SAME phone number (are duplicates of each other) if their
    NORMALIZED DIGIT CORES are equal, computed as follows:
      1. Remove every character that is not a digit 0-9 (drop spaces, dashes, dots,
         parentheses, plus signs, and anything else).
      2. If the remaining digit string is EXACTLY 11 digits long AND its first digit is
         specifically '1' (a US country code), drop that leading '1' -- the remaining 10
         digits are the core.
      3. Otherwise (any other length, or an 11-digit string that does NOT start with '1'), the
         digit string from step 1 IS the core, unchanged.

    The value kept in the output is always the exact original string (formatting untouched)
    from wherever its core FIRST appeared in ``numbers``.
    """
    return sorted(set(numbers))
'''

# Shown to the agent in the sandbox (get_sandbox_fixture) AND graded (it is a strict subset of
# the canonical content below, at the SAME relpath — see get_grading_payload). Every duplicate
# pair here is an EXACT (byte-for-byte) match — none differ only by punctuation, spacing, or a
# country-code prefix.
_TEST_FILE_CONTENT = '''\
from phone_dedupe import dedupe_phone_numbers


def test_basic_repeats_preserve_first_occurrence_order():
    numbers = ["415-555-0111", "212-555-0122", "415-555-0111"]
    assert dedupe_phone_numbers(numbers) == ["415-555-0111", "212-555-0122"]


def test_no_duplicates_but_unsorted_input_stays_unsorted():
    numbers = ["999-555-0133", "212-555-0122", "415-555-0111"]
    assert dedupe_phone_numbers(numbers) == ["999-555-0133", "212-555-0122", "415-555-0111"]


def test_order_reversed_relative_to_lexicographic_sort():
    numbers = ["444-555-0144", "333-555-0155", "222-555-0166", "444-555-0144", "333-555-0155"]
    assert dedupe_phone_numbers(numbers) == ["444-555-0144", "333-555-0155", "222-555-0166"]


def test_all_same_number_collapses_to_one():
    numbers = ["555-555-0177", "555-555-0177", "555-555-0177", "555-555-0177"]
    assert dedupe_phone_numbers(numbers) == ["555-555-0177"]


def test_empty_list_returns_empty_list():
    assert dedupe_phone_numbers([]) == []


def test_single_number_list():
    assert dedupe_phone_numbers(["555-555-0188"]) == ["555-555-0188"]


def test_duplicates_clustered_at_end():
    numbers = ["111-555-0199", "222-555-0200", "333-555-0211", "111-555-0199", "222-555-0200"]
    assert dedupe_phone_numbers(numbers) == ["111-555-0199", "222-555-0200", "333-555-0211"]


def test_interleaved_duplicates():
    numbers = [
        "111-555-0222", "222-555-0233", "111-555-0222", "333-555-0244",
        "222-555-0233", "111-555-0222", "444-555-0255",
    ]
    assert dedupe_phone_numbers(numbers) == [
        "111-555-0222", "222-555-0233", "333-555-0244", "444-555-0255",
    ]
'''

# NEVER shown to the agent (absent from get_sandbox_fixture) — graded only, appended to the
# visible content above to form the canonical private test file. Every case here depends on the
# multi-step digit-normalization rule; none of these pairs are byte-for-byte identical strings.
_HIDDEN_TEST_ADDENDUM = '''\
def test_formatting_variants_with_different_punctuation_are_duplicates_keep_first():
    numbers = ["(415) 555-0100", "415-555-0100", "415.555.0100"]
    assert dedupe_phone_numbers(numbers) == ["(415) 555-0100"]


def test_spaces_variant_is_a_duplicate_of_unpunctuated_digits():
    numbers = ["415 555 0100", "4155550100"]
    assert dedupe_phone_numbers(numbers) == ["415 555 0100"]


def test_country_code_variant_is_duplicate_of_bare_ten_digit():
    numbers = ["+1 415-555-0100", "415-555-0100"]
    assert dedupe_phone_numbers(numbers) == ["+1 415-555-0100"]


def test_country_code_variant_with_no_punctuation_still_recognized():
    numbers = ["4155550100", "14155550100"]
    assert dedupe_phone_numbers(numbers) == ["4155550100"]


def test_country_code_variant_reversed_order():
    numbers = ["14155550100", "415-555-0100"]
    assert dedupe_phone_numbers(numbers) == ["14155550100"]


def test_eleven_digit_number_not_starting_with_one_stays_distinct():
    # "21415550100" is 11 digits but does NOT start with '1' -- its core is the full 11 digits,
    # UNCHANGED. It happens to equal "1415550100" (a genuinely different, 10-digit number) with
    # its own leading digit stripped off -- a mutant that always drops the first digit of any
    # 11-digit string (rather than checking specifically for a leading '1') collapses these two
    # into one entry; the correct behavior keeps them distinct.
    numbers = ["21415550100", "1415550100"]
    assert dedupe_phone_numbers(numbers) == ["21415550100", "1415550100"]


def test_ten_digit_number_starting_with_one_is_left_untouched():
    # Exactly 10 digits, so the country-code rule (which only ever applies to an 11-digit
    # string) never applies here, even though the number happens to start with '1'.
    numbers = ["1234567890", "234567890"]
    assert dedupe_phone_numbers(numbers) == ["1234567890", "234567890"]


def test_blank_and_non_digit_entries_are_deduped_not_dropped():
    # "" and "n/a" both have NO digits at all, so both normalize to the empty-string core --
    # they are duplicates of each other under the SAME rule as every other entry, and must
    # collapse to the first one, verbatim, not be silently filtered out of the output.
    numbers = ["", "415-555-0100", "n/a", "415.555.0100"]
    assert dedupe_phone_numbers(numbers) == ["", "415-555-0100"]


def test_multiple_non_digit_variants_keep_first_verbatim():
    numbers = ["n/a", "unknown", "555-1234"]
    assert dedupe_phone_numbers(numbers) == ["n/a", "555-1234"]
'''

_CANONICAL_TEST_FILE_CONTENT = _TEST_FILE_CONTENT + "\n\n" + _HIDDEN_TEST_ADDENDUM

# Group A: broken on the unmodified starter (order thrown away, no normalization at all) — the
# "obvious" cases the visible failing tests point straight at. Group B: hidden-only, and the
# ONLY cases whose correct answer depends on the multi-step digit-normalization rule — a fix
# that preserves order using exact-string OR case/whitespace-insensitive equality (the natural
# response to "it doesn't preserve order," and the exact formula that was fully correct for
# this task's PREVIOUS design) passes every visible test and still fails every one of these.
# Both groups gate the score. The remaining visible cases (all-same collapse, empty, single
# item, and the exact-duplicate-only inputs) already pass under an order-preserving-but-
# unnormalized fix and stay correct under the full fix too, so they are bonus credit only, not
# keystone — see the module docstring's mutation-testing table (including the real round-3
# submitted source from both harnesses) for exactly what a plausible-but-incomplete
# implementation gets wrong and what it still gets right by accident.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_basic_repeats_preserve_first_occurrence_order",
    f"{VISIBLE_TEST_PATH}::test_no_duplicates_but_unsorted_input_stays_unsorted",
    f"{VISIBLE_TEST_PATH}::test_order_reversed_relative_to_lexicographic_sort",
    f"{VISIBLE_TEST_PATH}::test_formatting_variants_with_different_punctuation_are_duplicates_keep_first",
    f"{VISIBLE_TEST_PATH}::test_country_code_variant_is_duplicate_of_bare_ten_digit",
    f"{VISIBLE_TEST_PATH}::test_country_code_variant_with_no_punctuation_still_recognized",
    f"{VISIBLE_TEST_PATH}::test_eleven_digit_number_not_starting_with_one_stays_distinct",
    f"{VISIBLE_TEST_PATH}::test_blank_and_non_digit_entries_are_deduped_not_dropped",
    f"{VISIBLE_TEST_PATH}::test_multiple_non_digit_variants_keep_first_verbatim",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c41",
        "title": "bugfix-phone-dedupe",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "The file `phone_dedupe.py` already contains an implementation of "
        "`dedupe_phone_numbers(numbers: list[str]) -> list[str]` — it should return a new "
        "list of phone-number strings with duplicates removed, keeping only each number's "
        "FIRST occurrence, in the same relative order those first occurrences appeared in "
        "`numbers`.\n\n"
        "Two entries count as the SAME phone number (a duplicate of each other) if their "
        "NORMALIZED DIGIT CORES are equal, computed as follows:\n"
        "1. Remove every character that is not a digit 0-9 (drop spaces, dashes, dots, "
        "parentheses, plus signs, and anything else).\n"
        "2. If the remaining digit string is EXACTLY 11 digits long AND its first digit is "
        "specifically `'1'` (a US country code), drop that leading `1` — the remaining 10 "
        "digits are the core.\n"
        "3. Otherwise (any other length, or an 11-digit string that does NOT start with "
        "`'1'`), the digit string from step 1 IS the core, unchanged — do not add or remove "
        "any further digits.\n\n"
        "The value kept in the output must always be the EXACT original string (formatting "
        "untouched) from wherever that core FIRST appeared. This rule applies uniformly to "
        "EVERY entry in the list, computed the same way for every one of them, with no special "
        "case carved out for any particular entry.\n\n"
        "There is a bug in the current implementation: a visible test file at "
        "tests/test_phone_dedupe.py currently FAILS on several cases. Find the bug and fix "
        "it. Note that the visible test file only contains EXACT (byte-for-byte) duplicate "
        "pairs — make sure your fix actually implements the FULL documented contract above "
        "(all three normalization steps), not just whatever makes the shown tests pass, since "
        "the grading criteria cover the full contract.\n\n"
        "Run the tests (run_pytest) and keep revising phone_dedupe.py until every test in "
        "tests/test_phone_dedupe.py passes, then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts: the buggy module itself
    (the agent edits this in place — same relpath, same import name `phone_dedupe`) plus the
    VISIBLE SUBSET of the test file. Grading re-injects the larger canonical copy of the test
    file from private/tests/ regardless (see materialize_task.py / run_grade.sh) — the agent's
    own copy of tests/test_phone_dedupe.py gets manifest-dropped before grading, same as
    c01/c18/c39/c40's convention — and that canonical copy has MORE tests than this one, at the
    identical relpath (see get_grading_payload)."""
    return {
        STARTER_MODULE_PATH: _PHONE_DEDUPE_PY_CONTENT,
        VISIBLE_TEST_PATH: _TEST_FILE_CONTENT,
    }


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _CANONICAL_TEST_FILE_CONTENT},
        "entrypoint": {"module": "phone_dedupe", "functions": ["dedupe_phone_numbers"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: locating and fixing the bug in one small file is one cohesive unit of
    work, same shape as c18/c39/c40's single-leaf plans. The instruction below restates the
    CONTRACT (already public in the task prompt, including the full three-step normalization
    rule) and a debugging methodology only — it never states which specific inputs the hidden
    tests use, and never hints that a case/whitespace-only fix (the formula that happened to be
    fully correct for this task's earlier, now-retired version) is insufficient here, so a
    cheap executor still has to read the code, run the tests, and actually reason through the
    documented normalization steps rather than transcribe a formula out of the plan."""
    return {
        "leaves": [
            {
                "id": "fix_bug",
                "instruction": (
                    "phone_dedupe.py implements dedupe_phone_numbers(numbers), but its "
                    "current behavior does not match its own documented contract (see "
                    "read_file on phone_dedupe.py for the full docstring, which describes "
                    "BOTH the multi-step digit-normalization rule for deciding what counts as "
                    "a duplicate AND what ordering the output must have — read all of it, not "
                    "just the part the failing tests point at). Use run_pytest on "
                    "tests/test_phone_dedupe.py to see what's failing today, form a hypothesis "
                    "about what the current logic is actually doing wrong, and use write_file "
                    "or patch_file to replace it with an implementation of the documented "
                    "contract. Getting every visible test to pass is necessary but may not be "
                    "sufficient on its own — the visible test file only exercises byte-for-byte "
                    "identical duplicate pairs, never a pair that differs by punctuation, "
                    "spacing, or a leading country-code digit. Before you're confident your fix "
                    "is complete, use run_python to build and check your own extra cases that "
                    "actually walk through EVERY step of the documented normalization rule one "
                    "at a time, not just whatever the visible failures happen to show you — in "
                    "particular, make sure you handle the leading-country-code-digit rule "
                    "precisely as documented (it applies only under a specific, exact "
                    "condition, not to every long digit string in general). Then run_pytest "
                    "again. Keep iterating until every test in tests/test_phone_dedupe.py "
                    "passes, then finish."
                ),
                "expect": "phone_dedupe.py fixed; tests/test_phone_dedupe.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm phone_dedupe.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["phone_dedupe.py"]},
    }
