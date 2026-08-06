"""
codebench task c31 — hard/visible, string/text algorithms: expression-language tokenizer.

New category coverage beyond the existing suite's RLE (c02) / balanced-brackets (c03) /
Roman-numeral (c13) string tasks, and distinct from c30 (line diff) — this one is a
character-stream SCANNER with a context-sensitive disambiguation rule (unary-minus /
negative-number literal vs. binary subtraction), the classic tokenizer edge case named in
this task's design brief.

Ground truth for ``tokenize`` verified with an independently-written reference scanner
(same algorithm shape necessarily, since the disambiguation rule IS the spec — but every
token-by-token result below was produced by running that reference script, not hand-derived,
and cross-checked against a second manual token-by-token trace for every case) before being
embedded here — see idea_code_test_c31_test.py for the reimplementation that re-verifies this
independently. Do not hand-edit these literals without re-running that script.

The disambiguation rule: a `-` character is the START of a NEGATIVE NUMBER token (merged with
the digits immediately following it, no whitespace allowed in between) if and only if (a) it
is immediately followed by a digit, AND (b) the token immediately preceding it (if any) is
one of PLUS/MINUS/STAR/SLASH/LPAREN, or there IS no preceding token (start of input) — i.e.
a `-` is only ever a negative-number sign in a position where an operand (not an operator) is
expected. In every other case, `-` is its own MINUS token.

    ""                    -> []
    "42"                  -> [(NUMBER,"42")]
    "foo"                 -> [(IDENT,"foo")]
    "3 + 4"               -> [(NUMBER,"3"),(PLUS,"+"),(NUMBER,"4")]
    "3-5"                 -> [(NUMBER,"3"),(MINUS,"-"),(NUMBER,"5")]        (subtraction, no space)
    "-5"                  -> [(NUMBER,"-5")]                                (negative literal, start of input)
    "3 - -5"              -> [(NUMBER,"3"),(MINUS,"-"),(NUMBER,"-5")]       (subtraction then negative literal)
    "(-5 + x) * -2"       -> [(LPAREN,"("),(NUMBER,"-5"),(PLUS,"+"),(IDENT,"x"),(RPAREN,")"),(STAR,"*"),(NUMBER,"-2")]
    "-x"                  -> [(MINUS,"-"),(IDENT,"x")]                      (not followed by a digit -> operator)
    "x - y"               -> [(IDENT,"x"),(MINUS,"-"),(IDENT,"y")]
    "x -5"                -> [(IDENT,"x"),(MINUS,"-"),(NUMBER,"5")]         (after IDENT -> always operator position)
    "2y"                  -> [(NUMBER,"2"),(IDENT,"y")]                     (digits don't extend into letters)
    "  3   +   4  "       -> [(NUMBER,"3"),(PLUS,"+"),(NUMBER,"4")]         (whitespace skipped)
    "((1))"               -> [(LPAREN,"("),(LPAREN,"("),(NUMBER,"1"),(RPAREN,")"),(RPAREN,")")]
    "1 + 2 - 3 * 4 / 5"   -> all four operators, MINUS still an operator (follows NUMBER)
    "x1 + 2y"             -> [(IDENT,"x1"),(PLUS,"+"),(NUMBER,"2"),(IDENT,"y")]
    "3 @ 4"               -> raises ValueError (unrecognized character '@')
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_expr_tokenizer.py"

_TEST_FILE_CONTENT = '''\
import pytest
from expr_tokenizer import tokenize


def test_empty_string():
    assert tokenize("") == []


def test_single_number():
    assert tokenize("42") == [("NUMBER", "42")]


def test_single_identifier():
    assert tokenize("foo") == [("IDENT", "foo")]


def test_simple_addition():
    assert tokenize("3 + 4") == [("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4")]


def test_subtraction_no_spaces():
    assert tokenize("3-5") == [("NUMBER", "3"), ("MINUS", "-"), ("NUMBER", "5")]


def test_leading_negative_number():
    assert tokenize("-5") == [("NUMBER", "-5")]


def test_subtraction_then_negative_number():
    assert tokenize("3 - -5") == [
        ("NUMBER", "3"), ("MINUS", "-"), ("NUMBER", "-5"),
    ]


def test_compound_expression_with_negatives_and_parens():
    assert tokenize("(-5 + x) * -2") == [
        ("LPAREN", "("), ("NUMBER", "-5"), ("PLUS", "+"), ("IDENT", "x"),
        ("RPAREN", ")"), ("STAR", "*"), ("NUMBER", "-2"),
    ]


def test_unary_minus_before_identifier_is_an_operator():
    assert tokenize("-x") == [("MINUS", "-"), ("IDENT", "x")]


def test_identifier_minus_identifier():
    assert tokenize("x - y") == [("IDENT", "x"), ("MINUS", "-"), ("IDENT", "y")]


def test_identifier_then_negative_looking_number_is_subtraction():
    assert tokenize("x -5") == [("IDENT", "x"), ("MINUS", "-"), ("NUMBER", "5")]


def test_number_immediately_followed_by_letter_splits():
    assert tokenize("2y") == [("NUMBER", "2"), ("IDENT", "y")]


def test_whitespace_is_skipped():
    assert tokenize("  3   +   4  ") == [("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4")]


def test_nested_parens():
    assert tokenize("((1))") == [
        ("LPAREN", "("), ("LPAREN", "("), ("NUMBER", "1"),
        ("RPAREN", ")"), ("RPAREN", ")"),
    ]


def test_all_four_operators_in_one_expression():
    assert tokenize("1 + 2 - 3 * 4 / 5") == [
        ("NUMBER", "1"), ("PLUS", "+"), ("NUMBER", "2"), ("MINUS", "-"),
        ("NUMBER", "3"), ("STAR", "*"), ("NUMBER", "4"), ("SLASH", "/"), ("NUMBER", "5"),
    ]


def test_identifier_with_trailing_digit_and_number_with_trailing_letter():
    assert tokenize("x1 + 2y") == [
        ("IDENT", "x1"), ("PLUS", "+"), ("NUMBER", "2"), ("IDENT", "y"),
    ]


def test_unrecognized_character_raises():
    with pytest.raises(ValueError):
        tokenize("3 @ 4")
'''

# The negative-number-vs-subtraction disambiguation cases (no-space subtraction, leading
# negative, subtraction-then-negative, the compound expression combining several of these,
# unary-minus-before-identifier, identifier-then-negative-looking-number) are what actually
# distinguish "implemented the context-sensitive rule correctly" from "always treats '-' as
# one or the other"; the number/letter boundary and all-four-operators cases are also
# discriminating (a naive scanner might over- or under-consume). The degenerate single-token
# cases, whitespace handling, nested parens, and the unrecognized-character contract are
# bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_subtraction_no_spaces",
    f"{VISIBLE_TEST_PATH}::test_leading_negative_number",
    f"{VISIBLE_TEST_PATH}::test_subtraction_then_negative_number",
    f"{VISIBLE_TEST_PATH}::test_compound_expression_with_negatives_and_parens",
    f"{VISIBLE_TEST_PATH}::test_unary_minus_before_identifier_is_an_operator",
    f"{VISIBLE_TEST_PATH}::test_identifier_then_negative_looking_number_is_subtraction",
    f"{VISIBLE_TEST_PATH}::test_number_immediately_followed_by_letter_splits",
    f"{VISIBLE_TEST_PATH}::test_all_four_operators_in_one_expression",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c31",
        "title": "expr-tokenizer-negative-number-disambiguation",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `expr_tokenizer.py` that defines a function "
        "`tokenize(s: str) -> list`.\n\n"
        "`tokenize` scans a small arithmetic-expression source string into a list of "
        "`(TYPE, TEXT)` tuples, where `TYPE` is one of: `NUMBER`, `IDENT`, `PLUS`, `MINUS`, "
        "`STAR`, `SLASH`, `LPAREN`, `RPAREN`. `TEXT` is always the exact substring of `s` "
        "that token was scanned from.\n\n"
        "## Basic token shapes\n\n"
        "- Whitespace (spaces) between tokens is skipped entirely — it never produces a "
        "token.\n"
        "- A `NUMBER` is one or more consecutive decimal digits (`0`-`9`). Digits never "
        "extend into letters — `\"2y\"` scans as a `NUMBER` `\"2\"` immediately followed by "
        "an `IDENT` `\"y\"`, not one combined token.\n"
        "- An `IDENT` is one letter or underscore, followed by zero or more letters, digits, "
        "or underscores (so `\"x1\"` is a single `IDENT` token, not `IDENT` `\"x\"` followed "
        "by `NUMBER` `\"1\"`).\n"
        "- `+`, `*`, `/`, `(`, `)` are always their own single-character token (`PLUS`, "
        "`STAR`, `SLASH`, `LPAREN`, `RPAREN` respectively) — no ambiguity involved.\n"
        "- Any character that isn't whitespace, a digit, a letter/underscore, `+`, `-`, `*`, "
        "`/`, `(`, or `)` is invalid: raise `ValueError`.\n\n"
        "## The `-` disambiguation rule\n\n"
        "`-` is special: depending on context, it is EITHER the start of a NEGATIVE NUMBER "
        "literal (merged together with the digits immediately following it into a single "
        "`NUMBER` token whose text includes the leading `-`, e.g. `\"-5\"`) OR its own "
        "standalone `MINUS` operator token. The rule, applied at the moment a `-` character "
        "is encountered:\n\n"
        "1. Look at the TYPE of the most recently emitted token so far (or \"nothing yet\" "
        "if this `-` is the very first character of the (non-whitespace) input).\n"
        "2. If that previous token is `PLUS`, `MINUS`, `STAR`, `SLASH`, or `LPAREN`, or "
        "there is no previous token at all — i.e. the position is one where an OPERAND "
        "(a value), not an operator, is expected next — AND the `-` is immediately followed "
        "(no whitespace in between) by at least one digit, then merge the `-` together with "
        "that run of digits into a single `NUMBER` token.\n"
        "3. Otherwise (the previous token is `NUMBER`, `IDENT`, or `RPAREN` — an operator is "
        "expected next, not an operand — OR the `-` is not immediately followed by a digit), "
        "emit `-` as its own `MINUS` token.\n\n"
        "Concretely, this means:\n"
        "  - `\"3-5\"` and `\"3 - 5\"` both tokenize identically as subtraction: "
        "`[(\"NUMBER\",\"3\"), (\"MINUS\",\"-\"), (\"NUMBER\",\"5\")]` — the previous token "
        "before `-` is `NUMBER`, so it's always an operator regardless of spacing or what "
        "follows it.\n"
        "  - `\"-5\"` (nothing before it) tokenizes as a single negative-number token: "
        "`[(\"NUMBER\",\"-5\")]`.\n"
        "  - `\"3 - -5\"` tokenizes as `[(\"NUMBER\",\"3\"), (\"MINUS\",\"-\"), "
        "(\"NUMBER\",\"-5\")]` — the FIRST `-` follows a `NUMBER` so it's an operator; the "
        "SECOND `-` follows that `MINUS` operator (an operand-expected position) and is "
        "immediately followed by a digit, so it merges into `\"-5\"`.\n"
        "  - `\"-x\"` tokenizes as `[(\"MINUS\",\"-\"), (\"IDENT\",\"x\")]` — even though "
        "the position expects an operand, `-` is followed by a letter, not a digit, so it "
        "does NOT merge; it's emitted as its own `MINUS` token.\n"
        "  - `\"x -5\"` tokenizes as `[(\"IDENT\",\"x\"), (\"MINUS\",\"-\"), "
        "(\"NUMBER\",\"5\")]` — the previous token is `IDENT`, an operator-expected "
        "position, so `-` is a `MINUS` regardless of what follows it; the remaining `\"5\"` "
        "is then its own ordinary (non-negative) `NUMBER`.\n\n"
        "A visible test file is already present at tests/test_expr_tokenizer.py — run it "
        "(run_pytest) and keep revising expr_tokenizer.py until every test in it passes, "
        "then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "expr_tokenizer", "functions": ["tokenize"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write expr_tokenizer.py implementing tokenize(s: str) -> list, "
                    "returning a list of (TYPE, TEXT) tuples: TYPE in NUMBER/IDENT/PLUS/"
                    "MINUS/STAR/SLASH/LPAREN/RPAREN. Whitespace is skipped between tokens. "
                    "NUMBER = one or more digits (never extends into letters). IDENT = a "
                    "letter/underscore followed by letters/digits/underscores. "
                    "+ * / ( ) are always their own single-char operator token. Any other "
                    "character raises ValueError.\n\n"
                    "The tricky part is '-': scan char by char, tracking the TYPE of the "
                    "most recently emitted token (or None if nothing emitted yet, i.e. start "
                    "of input, ignoring whitespace). When you hit '-': if the previous "
                    "token's type is one of PLUS/MINUS/STAR/SLASH/LPAREN, or there is no "
                    "previous token, THEN check whether the very next character (no "
                    "whitespace skipped) is a digit -- if so, consume '-' plus that whole "
                    "run of digits as one NUMBER token whose text is e.g. '-5'. In every "
                    "other case (previous token is NUMBER/IDENT/RPAREN, or '-' isn't "
                    "immediately followed by a digit), emit '-' alone as a MINUS token. "
                    "This means after a NUMBER, IDENT, or RPAREN, '-' is ALWAYS a MINUS "
                    "operator regardless of what follows it (e.g. 'x -5' is IDENT, MINUS, "
                    "NUMBER('5') -- subtraction, not IDENT followed by a negative number).\n\n"
                    "Use write_file to create it. Then use run_pytest on "
                    "tests/test_expr_tokenizer.py. If any test fails, use "
                    "read_file/patch_file to fix expr_tokenizer.py and run_pytest again -- "
                    "pay special attention to the negative-number-vs-subtraction test cases, "
                    "they are the crux of this task. Once every test passes, finish."
                ),
                "expect": "expr_tokenizer.py written; tests/test_expr_tokenizer.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm expr_tokenizer.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["expr_tokenizer.py"]},
    }
