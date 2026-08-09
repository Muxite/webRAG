"""
codebench task c39 — hard/visible, DEBUG an existing broken module (a tiny template-string
renderer with two independent, unrelated seeded bugs).

REPLACEMENT NOTE (2026-08-06/07): this task used to be a broken binary search (a classic
`lo < hi` vs `lo <= hi` loop-guard off-by-one). It was already hardened once -- a second,
independent, latent `hi = mid` vs `hi = mid - 1` bug was seeded specifically so that the
"obvious" one-line fix (correcting only the loop guard) would make every originally-failing
VISIBLE test pass while causing a previously-passing test to hang (infinite loop) -- yet
qwen2.5:14b STILL scored 1.0/1.0 on BOTH the badmodel and Aider codebench harnesses in a second
live-calibration round. Inspecting the actual submission confirmed a fully correct, textbook
binary search (`while lo <= hi`, `hi = mid - 1`). Binary search is quite possibly the single
most memorized algorithm in existence; a competent model does not need to trace the GIVEN
buggy code's actual logic at all -- it recognizes "this is trying to be binary search," already
knows byte-for-byte what correct binary search looks like, and reproduces that from memory,
bypassing the debugging exercise entirely regardless of how the bug (or the second, masked
bug) is constructed. This is a full replacement, same id/category conventions ("hard",
debugging an existing broken module, visible test file + hidden held-back superset), applied
to a bespoke, made-up little format with no single canonical "this is what correct code looks
like" burned into training data the way binary search has.

The new task: `find_index.py` is replaced by `template_render.py`, implementing
`render(template: str, values: dict) -> str` -- a small template-string interpolator for a
made-up mini-language (`{name}` / `{name:default}` placeholders, `{{` / `}}` doubled-brace
escaping for literal braces). This is not a well-known library or a textbook algorithm; there
is no single memorized "correct" implementation of THIS SPECIFIC bespoke contract for a model
to fall back on, so actually fixing it requires tracing the GIVEN code's specific logic against
the stated contract, not recognizing a famous shape.

Two independent, seeded bugs, chosen so that neither can mask or "accidentally fix" the other,
and so that NEITHER bug can ever cause the loop to fail to terminate (every branch of the
scanning loop unconditionally advances its index by at least one character per iteration --
this task is deliberately engineered to have zero hang risk, unlike its predecessor, which
needed a live-measured wall-clock-bounded mutation-testing harness to safely PROVE an infinite
loop existed without ever letting a test process actually block):
  (1) when extracting the text between a placeholder's `{` and its matching `}`, the starter
      slices `template[i:j]` where it should slice `template[i+1:j]` -- it forgets to skip the
      opening brace itself, so every extracted placeholder name is corrupted with a leading
      `{` character it should never have. This breaks EVERY test that exercises a real
      `{name}` placeholder (wrong output, or the wrong KeyError argument) and is directly,
      immediately visible from the very first `run_pytest` -- the "obvious" bug.
  (2) when consuming a doubled-open-brace escape `{{`, the starter advances its scan index by
      only 1 character instead of 2 -- it only consumes ONE of the two `{` characters, leaving
      the second one to be misinterpreted as the start of a brand new (bogus) placeholder on
      the very next loop iteration. This bug lives entirely inside the escaping code path,
      which NONE of the visible tests exercise at all (every visible test uses only plain
      `{name}` / `{name:default}` placeholders, never `{{` / `}}`) -- so it is completely
      latent with respect to the visible suite, regardless of whether bug (1) has been fixed.

Because the two bugs are structurally independent (they live in different, non-overlapping
branches of the scanning loop and never share mutable state), fixing bug (1) alone makes EVERY
visible test pass cleanly -- a fix that looks completely done from the point of view of anyone
who only runs the visible suite -- while every hidden test exercising `{{`/`}}` escaping still
fails, because bug (2) was never touched. This was verified by actually EXECUTING all three
variants (unmodified starter, bug-(1)-only "obvious" fix, both bugs fixed) against the full
13-case canonical suite: the starter fails 10/13 (only the escaping-unrelated, non-placeholder
cases pass by luck), the bug-(1)-only fix passes all 9 VISIBLE cases but fails all 4 HIDDEN
(escaping) cases, and the fully correct fix passes all 13. See
agent/tests/idea_code_test_c39_test.py for the same verification re-derived
independently (never importing this module's own buggy source as ground truth) plus the
keystone/plan/materialize_task shape checks. This replacement has not yet been through a
live-calibration round against qwen2.5:14b (offline mutation testing above is the
authoring-time evidence; live calibration against both the badmodel and aider harnesses is the
coordinating session's next step, per its own instructions, not run from here).
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_template_render.py"
STARTER_MODULE_PATH = "template_render.py"

_TEMPLATE_RENDER_PY_CONTENT = '''\
def render(template, values):
    """Render a tiny template mini-language.

    Placeholders look like ``{name}`` or ``{name:default}`` and are replaced with
    ``str(values[name])`` (or the literal default text if ``name`` is not in ``values`` --
    raising ``KeyError(name)`` if there is also no default). A doubled brace ``{{`` or ``}}``
    renders as a single literal ``{`` or ``}`` and never opens or closes a placeholder. An
    unmatched single ``{`` or ``}`` raises ``ValueError``.
    """
    out = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            if template[i:i + 2] == "{{":
                out.append("{")
                i += 1
                continue
            j = template.find("}", i)
            if j == -1:
                raise ValueError(f"unmatched '{{' at position {i}")
            inner = template[i:j]
            if ":" in inner:
                name, default = inner.split(":", 1)
            else:
                name, default = inner, None
            if name in values:
                out.append(str(values[name]))
            elif default is not None:
                out.append(default)
            else:
                raise KeyError(name)
            i = j + 1
        elif ch == "}":
            if template[i:i + 2] == "}}":
                out.append("}")
                i += 2
            else:
                raise ValueError(f"unmatched '}}' at position {i}")
        else:
            out.append(ch)
            i += 1
    return "".join(out)
'''

# Shown to the agent in the sandbox (get_sandbox_fixture) AND graded (it is a strict subset of
# the canonical content below, at the SAME relpath — see get_grading_payload). Deliberately
# never exercises the `{{` / `}}` escaping rule at all — see module docstring.
_TEST_FILE_CONTENT = '''\
import pytest
from template_render import render


def test_plain_text_with_no_placeholders_is_unchanged():
    assert render("hello world", {}) == "hello world"


def test_single_placeholder_is_substituted():
    assert render("hi {name}!", {"name": "Ann"}) == "hi Ann!"


def test_multiple_placeholders_are_all_substituted():
    assert render("{a}-{b}-{c}", {"a": "1", "b": "2", "c": "3"}) == "1-2-3"


def test_non_string_value_is_stringified():
    assert render("count={n}", {"n": 5}) == "count=5"


def test_default_is_used_when_name_is_missing():
    assert render("hi {name:stranger}!", {}) == "hi stranger!"


def test_default_is_ignored_when_name_is_present():
    assert render("hi {name:stranger}!", {"name": "Bo"}) == "hi Bo!"


def test_missing_name_without_default_raises_key_error():
    with pytest.raises(KeyError) as exc_info:
        render("{missing}", {})
    assert exc_info.value.args[0] == "missing"


def test_placeholder_at_start_and_end_of_template():
    assert render("{a}middle{b}", {"a": "X", "b": "Y"}) == "XmiddleY"


def test_unmatched_open_brace_raises_value_error():
    with pytest.raises(ValueError):
        render("bad {open", {})
'''

# NEVER shown to the agent (absent from get_sandbox_fixture) — graded only, appended to the
# visible content above to form the canonical private test file. Every one of these exercises
# the `{{` / `}}` escaping code path that bug (2) lives in and that the visible suite never
# touches at all.
_HIDDEN_TEST_ADDENDUM = '''\
def test_escaped_braces_alone_render_as_literal_braces():
    assert render("{{not a placeholder}}", {}) == "{not a placeholder}"


def test_escaped_braces_combined_with_a_real_placeholder():
    assert render("{{literal}} then {name}", {"name": "Q"}) == "{literal} then Q"


def test_consecutive_escaped_pairs_render_correctly():
    assert render("{{{{}}}}", {}) == "{{}}"


def test_unmatched_close_brace_after_valid_escapes_raises_value_error():
    with pytest.raises(ValueError):
        render("{{ok}}extra}", {})
'''

_CANONICAL_TEST_FILE_CONTENT = _TEST_FILE_CONTENT + "\n\n" + _HIDDEN_TEST_ADDENDUM

# Group A: broken on the unmodified starter by bug (1) alone (VISIBLE, "obvious" from the very
# first run_pytest). Group B: bug (1) fixed in isolation ("the obvious fix") makes every one of
# these pass, since none of them touch escaping -- but bug (2) is a completely SEPARATE, latent
# defect that only the HIDDEN escaping cases can ever catch; both groups gate the score. The
# remaining visible cases (plain text, default-used-when-missing, unmatched-open-brace) already
# pass on the unmodified starter AND stay correct under either partial or full fixes, so they
# are bonus credit only, not keystone — see the module docstring's live-executed verification
# table.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_single_placeholder_is_substituted",
    f"{VISIBLE_TEST_PATH}::test_multiple_placeholders_are_all_substituted",
    f"{VISIBLE_TEST_PATH}::test_non_string_value_is_stringified",
    f"{VISIBLE_TEST_PATH}::test_default_is_ignored_when_name_is_present",
    f"{VISIBLE_TEST_PATH}::test_missing_name_without_default_raises_key_error",
    f"{VISIBLE_TEST_PATH}::test_placeholder_at_start_and_end_of_template",
    f"{VISIBLE_TEST_PATH}::test_escaped_braces_alone_render_as_literal_braces",
    f"{VISIBLE_TEST_PATH}::test_escaped_braces_combined_with_a_real_placeholder",
    f"{VISIBLE_TEST_PATH}::test_consecutive_escaped_pairs_render_correctly",
    f"{VISIBLE_TEST_PATH}::test_unmatched_close_brace_after_valid_escapes_raises_value_error",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c39",
        "title": "bugfix-template-renderer",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "The file `template_render.py` already contains an implementation of "
        "`render(template: str, values: dict) -> str` — a tiny template-string interpolator. "
        "Its full contract:\n\n"
        "- Scan `template` left to right, character by character.\n"
        "- The two-character sequence `{{` renders as a single literal `{` and does not open "
        "a placeholder.\n"
        "- The two-character sequence `}}` renders as a single literal `}` and does not close "
        "a placeholder (nothing is open at that point for it to close).\n"
        "- A single `{` (not immediately followed by another `{`) opens a PLACEHOLDER that "
        "runs up to the next `}` character. Inside a placeholder the text is either just "
        "`NAME`, or `NAME:DEFAULT` (a name, a single colon, then default text — placeholders "
        "and default text never themselves contain `{` or `}`).\n"
        "  - If `NAME` is a key in `values`: the placeholder is replaced by "
        "`str(values[NAME])` (any default text present is ignored).\n"
        "  - If `NAME` is not a key in `values` and a default was given (a colon was "
        "present): the placeholder is replaced by the default text, exactly as written.\n"
        "  - If `NAME` is not a key in `values` and no default was given: raise "
        "`KeyError(NAME)`.\n"
        "- If a `{` is opened but no matching `}` is ever found before the template ends, "
        "raise `ValueError`.\n"
        "- A `}` that is not part of a doubled `}}` and is not already accounted for by "
        "closing an open placeholder above is itself an error: raise `ValueError`.\n"
        "- Every other character is copied through to the output unchanged.\n\n"
        "The implementation is buggy: a visible test file at tests/test_template_render.py "
        "currently FAILS on several cases. Find the bug and fix it — do not rewrite the "
        "function from scratch.\n\n"
        "Important: the visible test file only exercises PART of the full contract above — in "
        "particular, none of its cases involve the doubled-brace `{{` / `}}` escaping rule. "
        "Passing every visible test is necessary but is not, on its own, good evidence that "
        "your fix is complete. Re-read the full contract above, and use run_python to build "
        "and check your own extra cases exercising the parts the visible tests don't — "
        "especially `{{` and `}}` — before you're confident the fix is right. More than one "
        "thing can be wrong with a small piece of code like this at the same time; fixing "
        "whichever one your first test run happens to point at does not guarantee the other "
        "isn't still there, waiting in a part of the contract nothing has exercised yet.\n\n"
        "Run the tests (run_pytest) and keep revising template_render.py until every test in "
        "tests/test_template_render.py passes cleanly, then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts: the buggy module itself
    (the agent edits this in place — same relpath, same import name `template_render`) plus
    the VISIBLE SUBSET of the test file. Grading re-injects the larger canonical copy of the
    test file from private/tests/ regardless (see materialize_task.py / run_grade.sh) — the
    agent's own copy of tests/test_template_render.py gets manifest-dropped before grading,
    same as c01/c18/c39's original convention — and that canonical copy has MORE tests than
    this one, at the identical relpath (see get_grading_payload)."""
    return {
        STARTER_MODULE_PATH: _TEMPLATE_RENDER_PY_CONTENT,
        VISIBLE_TEST_PATH: _TEST_FILE_CONTENT,
    }


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _CANONICAL_TEST_FILE_CONTENT},
        "entrypoint": {"module": "template_render", "functions": ["render"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: locating and fixing the bug(s) in one small file is one cohesive unit of
    work, same shape as c18's single-leaf refactor plan and the original c39's. The
    instruction intentionally describes CONTRACT and METHODOLOGY only (what the function must
    do, how to investigate, how to recognize a fix that only covers what's currently visible
    vs. one that covers the whole stated contract) — never the mechanism of either seeded bug,
    and never a literal input/output pair, so a cheap executor cannot just copy an answer out
    of the plan instead of deriving it from the sandbox."""
    return {
        "leaves": [
            {
                "id": "fix_bug",
                "instruction": (
                    "template_render.py implements a small template-string interpolator "
                    "(placeholders like {name} or {name:default}, plus {{ / }} doubled-brace "
                    "escaping for literal braces), but has at least one bug — quite possibly "
                    "more than one, and they need not be related to each other. Use read_file "
                    "to review template_render.py's current logic, then run_pytest on "
                    "tests/test_template_render.py to see what's "
                    "failing. Form a hypothesis about the root cause, make a minimal targeted "
                    "change with patch_file, then run_pytest again on the WHOLE file (not "
                    "just the case you targeted) to check whether that actually fixed things. "
                    "Critically: passing every test currently in tests/test_template_render.py "
                    "only proves the parts of the contract THAT FILE happens to exercise are "
                    "correct — it says nothing about parts of the documented contract (in "
                    "particular, the doubled-brace {{ / }} escaping rule) that file might "
                    "never touch at all. Before concluding you're done, use run_python to "
                    "build and check a few of your own extra cases that exercise parts of the "
                    "full contract the visible test file doesn't, especially escaping — a fix "
                    "that only chases the failures the visible suite happens to show you is "
                    "not the same thing as a fix that covers the whole stated contract. Keep "
                    "iterating until the whole visible test file passes cleanly AND your own "
                    "extra spot checks of the untested parts of the contract look right, then "
                    "finish."
                ),
                "expect": "template_render.py fixed; tests/test_template_render.py fully "
                          "passes, and the fix covers the full documented contract "
                          "(placeholders, defaults, missing-key errors, AND doubled-brace "
                          "escaping), not only whatever the visible test file happens to show",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm template_render.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["template_render.py"]},
    }
