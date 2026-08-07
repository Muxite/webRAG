"""
codebench task c29 — hard/visible, `{{placeholder}}` mail-merge templating format
(substitution, whitespace-tolerant keys, backslash-escaped literal braces, strict vs.
non-strict missing-key policy, unterminated-placeholder rejection, and an
escape-adjacent-to-real-placeholder interaction that a naive two-pass
unescape-then-substitute implementation gets wrong).

File-format coverage gap this fills: c04 (CSV), c07 (fleet-config gen), c10 (markdown
table), c12 (JSON diff), c27 (INI), c28 (BREC binary) are all either tabular/structured
data formats or a binary record format. None of them is a text SUBSTITUTION/templating
mini-language with its own escaping rules — a different parsing skill (single left-to-right
scan with lookahead/escape handling) from any of those.

Ground truth for ``render_template`` verified by actually RUNNING a reference
implementation (NOT hand-computed) — see
``/tmp/claude-1000/-home-muk-projects-webRAG/cebc60e0-6c4f-4b3e-9be6-882dd9f08d84/scratchpad/ref/mailmerge.py``
for the throwaway script this was derived from, and idea_code_test_c29_test.py for the
independent second reimplementation that re-verifies every literal below at test time.
Verified behavior:
    render_template("Hello, {{name}}!", {"name": "Alice"}) -> "Hello, Alice!"
    render_template("Hi {{ name }}", {"name": "Bob"}) -> "Hi Bob"              (inner ws stripped)
    render_template("{{a}}-{{b}}-{{a}}", {"a": "x", "b": "y"}) -> "x-y-x"
    render_template("Count: {{n}}", {"n": 42}) -> "Count: 42"                  (str()'d)
    render_template(r"Use \\{\\{ and \\}\\} for literals", {}) -> "Use {{ and }} for literals"
    render_template("plain text, no braces here", {}) -> "plain text, no braces here"
    render_template("{{missing}}", {}) -> raises KeyError               (strict=True default)
    render_template("{{missing}}", {}, strict=False) -> "{{missing}}"   (left verbatim)
    render_template("{{oops", {"oops": "x"}) -> raises ValueError       (unterminated)
    render_template(r"\\{{not_a_var}}{{real}}", {"real": "R"}) -> "{{not_a_var}}R"
        (subtle: a backslash-escape (rule 1) consumes exactly ONE following brace
        character, not a "{{" pair. So the escape at the start only eats the first
        "{"; the leftover second "{" can't pair with the "n" right after it to open a
        placeholder (rule 2 needs the CURRENT char and the NEXT char to both be "{"),
        so "not_a_var" and the "}}" after it fall through to rule 3 and are emitted as
        plain literal text -- "not_a_var" is never looked up as a key at all, so
        strict=True must NOT raise for it even though it's absent from context. Only
        the trailing "{{real}}" is a genuine placeholder. A two-pass implementation
        that unescapes braces first and THEN regex-substitutes "{{...}}" placeholders
        (a common, idiomatic way to write this) gets this wrong: unescaping first
        turns the input into "{{not_a_var}}{{real}}", which then looks like two
        genuine placeholders, incorrectly raising KeyError('not_a_var').)
    render_template("Dear {{missing}}, thanks", {}, strict=False) ->
        "Dear {{missing}}, thanks"   (unchanged; the non-strict verbatim-fallback span
        must be sliced relative to where the "{{" itself starts, not relative to 0 --
        this is trivially easy to get right when the placeholder happens to be the
        very first thing in the template, and easy to get subtly wrong (e.g. an
        off-by-the-placeholder's-own-start-index slice) once there is leading text
        before it, which then corrupts/duplicates the trailing text after it too.
        LIVE-CALIBRATION NOTE: this is not a hypothetical -- an actual submitted
        solution for this task got every original canonical test right (including the
        non-strict-missing-key one, whose only case had the placeholder flush at
        index 0) while shipping exactly this off-by-index slicing bug, which only
        this later, differently-positioned case exposes.)
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_mailmerge.py"

_TEST_FILE_CONTENT = r'''
import pytest
from mailmerge import render_template


def test_basic_substitution():
    assert render_template("Hello, {{name}}!", {"name": "Alice"}) == "Hello, Alice!"


def test_whitespace_inside_braces_stripped():
    assert render_template("Hi {{ name }}", {"name": "Bob"}) == "Hi Bob"


def test_multiple_placeholders_repeated_key():
    assert render_template("{{a}}-{{b}}-{{a}}", {"a": "x", "b": "y"}) == "x-y-x"


def test_non_string_value_stringified():
    assert render_template("Count: {{n}}", {"n": 42}) == "Count: 42"


def test_escaped_braces_literal():
    template = r"Use \{\{ and \}\} for literals"
    assert render_template(template, {}) == "Use {{ and }} for literals"


def test_no_placeholders_passthrough():
    text = "plain text, no braces here"
    assert render_template(text, {}) == text


def test_missing_key_strict_raises_keyerror():
    with pytest.raises(KeyError):
        render_template("{{missing}}", {})


def test_missing_key_non_strict_leaves_placeholder():
    assert render_template("{{missing}}", {}, strict=False) == "{{missing}}"


def test_unterminated_placeholder_raises_value_error():
    with pytest.raises(ValueError):
        render_template("{{oops", {"oops": "x"})


def test_escaped_brace_immediately_before_real_placeholder():
    # A backslash-escape (rule 1) consumes exactly ONE following brace character, not a
    # "{{" pair. So in r"\{{not_a_var}}{{real}}" the escape only eats the leading "\{";
    # the leftover "{" right after it can't pair with "n" to open a placeholder, so
    # "not_a_var" (and the "}}" after it) are plain literal text, never looked up as a
    # key -- strict mode must NOT raise for it. Only the trailing "{{real}}" is a real
    # placeholder.
    template = r"\{{not_a_var}}{{real}}"
    assert render_template(template, {"real": "R"}) == "{{not_a_var}}R"


def test_non_strict_missing_placeholder_not_at_template_start():
    # The non-strict verbatim-fallback span must be sliced relative to the
    # placeholder's OWN start position, not relative to index 0 -- easy to get right
    # by coincidence when the placeholder happens to be the first thing in the
    # template (as in test_missing_key_non_strict_leaves_placeholder above) and easy
    # to get subtly wrong once there's leading text before it, which then also
    # corrupts the text that follows the placeholder.
    template = "Dear {{missing}}, thanks"
    assert render_template(template, {}, strict=False) == template
'''

# Basic substitution, escaped-brace literals, both missing-key policies, unterminated-
# placeholder rejection, the escape-adjacent-to-real-placeholder interaction, and the
# non-strict fallback at a non-zero offset are what actually distinguish a real,
# correctly-indexed single-pass priority-ordered scanning implementation from a
# naive/subtly-off one (which typically gets escaping, the strict/non-strict branch, the
# ordering between escaping and placeholder-detection, or the fallback slice's offset
# wrong first) — those seven gate the score. The escape-adjacent case catches the common
# (and otherwise entirely plausible) "unescape braces in one pass, then regex-substitute
# placeholders in a second pass" implementation strategy, which passes every other test
# here but silently mishandles a backslash sitting right next to a real placeholder. The
# non-zero-offset case catches an off-by-the-placeholder's-own-index slicing mistake in
# the non-strict fallback branch that is invisible when the ONLY non-strict test case
# has its placeholder flush at index 0 (a real submitted solution for this exact task
# reproduced precisely this bug during live calibration and passed every other test
# here). Whitespace-trimming, repeated-key reuse, non-string stringification, and plain
# passthrough are real but shallower checks — bonus credit only, same convention as
# c04's/c27's bonus tests.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_basic_substitution",
    f"{VISIBLE_TEST_PATH}::test_escaped_braces_literal",
    f"{VISIBLE_TEST_PATH}::test_missing_key_strict_raises_keyerror",
    f"{VISIBLE_TEST_PATH}::test_missing_key_non_strict_leaves_placeholder",
    f"{VISIBLE_TEST_PATH}::test_unterminated_placeholder_raises_value_error",
    f"{VISIBLE_TEST_PATH}::test_escaped_brace_immediately_before_real_placeholder",
    f"{VISIBLE_TEST_PATH}::test_non_strict_missing_placeholder_not_at_template_start",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c29",
        "title": "mailmerge-template-substitution",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `mailmerge.py` that defines a function "
        "`render_template(template: str, context: dict, strict: bool = True) -> str`, a "
        "small mail-merge / templating engine using `{{placeholder}}` substitution.\n\n"
        "Scan `template` left to right, ONE character at a time, building the output as "
        "you go, using these rules IN THIS PRIORITY ORDER at each position:\n\n"
        "1. If the current character is a backslash `\\` and the NEXT character is `{` or "
        "`}`: emit that `{` or `}` literally (as plain output text) and advance past BOTH "
        "characters (the backslash itself is consumed, never emitted).\n"
        "2. Otherwise, if the current character and the next character are both `{` "
        "(i.e. you see the two-character sequence `{{`, and rule 1 did not already "
        "consume it): this begins a placeholder. Search forward from just after this `{{` "
        "for the next occurrence of `}}` anywhere later in the template.\n"
        "   - If no `}}` is found anywhere later in the template, raise `ValueError` "
        "(an unterminated placeholder is malformed input).\n"
        "   - Otherwise, the text strictly between the `{{` and that `}}` is the "
        "placeholder's raw key text. Strip leading/trailing whitespace from it to get "
        "`key`.\n"
        "     - If `key` is a key of `context`: emit `str(context[key])` and advance past "
        "the whole `{{...}}` span (to just after the `}}`).\n"
        "     - If `key` is NOT a key of `context`:\n"
        "       - if `strict` is `True` (the default): raise `KeyError(key)`.\n"
        "       - if `strict` is `False`: emit the ORIGINAL text of the whole `{{...}}` "
        "span exactly as it appeared in the template (including its own `{{`/`}}` "
        "delimiters and any inner whitespace, UN-stripped), then advance past it.\n"
        "3. Otherwise: emit the current character exactly as-is and advance one "
        "character. (This covers a lone `{`, a lone `}`, and a backslash that is not "
        "immediately followed by `{` or `}` — none of those are special on their own.)\n\n"
        "Notes: `key` lookup is an exact string match after stripping (context keys are "
        "not stripped/normalized on the other side, so define your context dicts with "
        "already-clean key names, matching the examples below). Values from `context` are "
        "converted with `str(...)` before being inserted, so non-string values (e.g. "
        "ints) work.\n\n"
        "Examples:\n"
        '- `render_template("Hello, {{name}}!", {"name": "Alice"})` -> `"Hello, Alice!"`\n'
        '- `render_template("Hi {{ name }}", {"name": "Bob"})` -> `"Hi Bob"`\n'
        '- `render_template(r"Use \\{\\{ and \\}\\} for literals", {})` -> '
        '`"Use {{ and }} for literals"`\n'
        '- `render_template("{{missing}}", {})` -> raises `KeyError`\n'
        '- `render_template("{{missing}}", {}, strict=False)` -> `"{{missing}}"`\n'
        '- `render_template("{{oops", {"oops": "x"})` -> raises `ValueError`\n\n'
        "A visible test file is already present at tests/test_mailmerge.py — run it "
        "(run_pytest) and keep revising mailmerge.py until every test in it passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "mailmerge", "functions": ["render_template"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one function (with a default keyword arg), no I/O — same single-leaf
    rationale as c01/c04/c07/c27."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write mailmerge.py implementing render_template(template: str, "
                    "context: dict, strict: bool = True) -> str. Scan template left to "
                    "right: a backslash immediately followed by '{' or '}' emits that "
                    "brace literally and consumes both characters (backslash dropped). "
                    "Otherwise '{{' starts a placeholder: find the next '}}' later in the "
                    "template (if none exists, raise ValueError - unterminated); the text "
                    "strictly between them, whitespace-stripped, is the key. If the key is "
                    "in context, emit str(context[key]) and skip past the whole "
                    "'{{...}}' span. If the key is missing: when strict is True (default), "
                    "raise KeyError(key); when strict is False, emit the original "
                    "'{{...}}' text verbatim (un-stripped) instead. Any other character "
                    "(including a lone '{' or '}') is emitted as-is. Use write_file to "
                    "create mailmerge.py. Then use run_pytest on tests/test_mailmerge.py. "
                    "If any test fails, use read_file/patch_file to fix mailmerge.py and "
                    "run_pytest again. Once every test passes, finish."
                ),
                "expect": "mailmerge.py written; tests/test_mailmerge.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm mailmerge.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["mailmerge.py"]},
    }
