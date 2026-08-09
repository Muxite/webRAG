"""
codebench task c43 — hard/hidden, compiler-lite chain, THREE-leaf dependency chain.

Third multi-leaf task (after c06's two-leaf precedent and c42's three-leaf validation chain).
This one is a classic front-half-of-a-compiler pipeline over a tiny arithmetic-expression
language (non-negative integers, `+ - * /`, parentheses, standard precedence):
  * leaf_a builds ``tokenizer.py`` — ``tokenize(expr) -> list[tuple[str, str]]``, a lexer that
    turns a string into ``(TYPE, text)`` tuples, always ending with an ``("EOF", "")`` sentinel.
    Independently useful/testable on its own.
  * leaf_b (``depends_on: ["leaf_a"]``) builds ``parser.py``'s ``parse(expr) -> tuple``, which
    IMPORTS ``tokenizer.tokenize`` and runs a standard recursive-descent parse
    (``expr := term (('+'|'-') term)*``, ``term := factor (('*'|'/') factor)*``,
    ``factor := NUMBER | '(' expr ')'``) into a nested-tuple AST (``("num", n)`` /
    ``("binop", op, left, right)``) — this is where operator precedence actually gets encoded
    structurally, not just token order.
  * leaf_c (``depends_on: ["leaf_b"]``) builds ``evaluator.py``'s ``evaluate(expr) -> int|float``,
    which IMPORTS ``parser.parse`` and recursively walks the AST leaf_b produced to compute a
    number. leaf_c does NOT depend on leaf_a directly — it never calls ``tokenize`` itself,
    exactly mirroring the real dependency graph of a tokenizer/parser/evaluator pipeline (the
    evaluator only ever talks to the parser).

See ``agent/app/testing/execution_compiled_code.py``'s ``_dep_text``/
``UPSTREAM_INCOMPLETE_MARKER`` for exactly what lands in a ``{leaf_x}`` placeholder at runtime.
Because that text is not fully reliable even in the happy path, both dependent leaves'
instructions independently RESTATE the exact token/AST shape they rely on, exactly like c06 and
c42 do, rather than trusting the upstream leaf's self-report alone.

Ground truth below was derived by actually RUNNING a reference tokenizer/parser/evaluator (not
worked out by hand) — see the offline validator test file's own independently-written
reimplementation for the second, separate check. Key verified values:
    tokenize("3 + 4 * 2")
        -> [("NUMBER","3"), ("PLUS","+"), ("NUMBER","4"), ("STAR","*"), ("NUMBER","2"), ("EOF","")]
    tokenize("(3+4)*2")
        -> [("LPAREN","("), ("NUMBER","3"), ("PLUS","+"), ("NUMBER","4"), ("RPAREN",")"),
            ("STAR","*"), ("NUMBER","2"), ("EOF","")]
    tokenize("3 + @") -> raises ValueError (unknown character)
    parse("3 + 4 * 2") -> ("binop","+",("num",3),("binop","*",("num",4),("num",2)))
        (multiplication binds tighter -- proves real precedence, not left-to-right token grouping)
    parse("(3 + 4) * 2") -> ("binop","*",("binop","+",("num",3),("num",4)),("num",2))
    parse("(3 + 4") -> raises ValueError (unbalanced parens)
    parse("3 +") -> raises ValueError (missing right operand)
    evaluate("3 + 4 * 2") == 11
    evaluate("(3 + 4) * 2") == 14
    evaluate("10 / 4") == 2.5           (true division, float result)
    evaluate("2 * (3 + (4 - 1))") == 12
    evaluate("100 - 2 * 3 - 4") == 90   (left-associative +/- at equal precedence)
    evaluate("1 / 0") -> raises ZeroDivisionError
    evaluate("3 + @") -> raises ValueError (propagated from the tokenizer, through the parser)
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_evaluator.py"

_TEST_FILE_CONTENT = '''\
import pytest
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate


# --- leaf_a: tokenizer.py -------------------------------------------------------------------

def test_tokenize_basic_expression():
    assert tokenize("3 + 4 * 2") == [
        ("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4"), ("STAR", "*"), ("NUMBER", "2"),
        ("EOF", ""),
    ]


def test_tokenize_handles_parentheses_and_ignores_whitespace():
    assert tokenize("(3+4)*2") == [
        ("LPAREN", "("), ("NUMBER", "3"), ("PLUS", "+"), ("NUMBER", "4"), ("RPAREN", ")"),
        ("STAR", "*"), ("NUMBER", "2"), ("EOF", ""),
    ]


def test_tokenize_rejects_unknown_character():
    with pytest.raises(ValueError):
        tokenize("3 + @")


def test_tokenize_empty_expression_is_just_eof():
    assert tokenize("") == [("EOF", "")]
    assert tokenize("   ") == [("EOF", "")]


# --- leaf_b: parser.py -----------------------------------------------------------------------

def test_parser_builds_correct_precedence_ast():
    # multiplication must bind tighter than addition -- this is what proves the parser encodes
    # real operator precedence, not just left-to-right token order.
    assert parse("3 + 4 * 2") == ("binop", "+", ("num", 3), ("binop", "*", ("num", 4), ("num", 2)))


def test_parser_handles_nested_parentheses():
    assert parse("(3 + 4) * 2") == ("binop", "*", ("binop", "+", ("num", 3), ("num", 4)), ("num", 2))
    assert parse("2 * (3 + (4 - 1))") == (
        "binop", "*", ("num", 2),
        ("binop", "+", ("num", 3), ("binop", "-", ("num", 4), ("num", 1))),
    )


def test_parser_rejects_unbalanced_parens():
    with pytest.raises(ValueError):
        parse("(3 + 4")


def test_parser_rejects_missing_operand():
    with pytest.raises(ValueError):
        parse("3 +")


# --- leaf_c: evaluator.py --------------------------------------------------------------------

def test_evaluator_simple_arithmetic():
    assert evaluate("3 + 4 * 2") == 11
    assert evaluate("(3 + 4) * 2") == 14


def test_evaluator_respects_precedence_and_associativity():
    assert evaluate("2 * (3 + (4 - 1))") == 12
    assert evaluate("100 - 2 * 3 - 4") == 90


def test_evaluator_division_produces_float():
    assert evaluate("10 / 4") == 2.5


def test_evaluator_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        evaluate("1 / 0")


def test_evaluator_propagates_syntax_errors():
    with pytest.raises(ValueError):
        evaluate("3 + @")
'''

# Basic single-token/single-error-path checks are supporting/bonus credit -- a broken parser or
# evaluator could still pass them by accident (e.g. hardcoding). The four that gate the score
# exercise the parts a shortcut implementation would plausibly get wrong: real operator
# PRECEDENCE structurally encoded in the AST (not just left-to-right grouping), nested
# parentheses, end-to-end precedence+associativity through the full pipeline, and a genuine
# float (true-division) result.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_parser_builds_correct_precedence_ast",
    f"{_TEST_FILE_PATH}::test_parser_handles_nested_parentheses",
    f"{_TEST_FILE_PATH}::test_evaluator_respects_precedence_and_associativity",
    f"{_TEST_FILE_PATH}::test_evaluator_division_produces_float",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c43",
        "title": "tokenizer-parser-evaluator-chain",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You are building a tiny arithmetic-expression interpreter, split across three files: "
        "a tokenizer, a parser that builds an AST from the tokens, and an evaluator that walks "
        "the AST. The language: non-negative integers, the binary operators `+ - * /`, and "
        "parentheses, with standard precedence (`*`/`/` bind tighter than `+`/`-`) and "
        "left-to-right associativity within the same precedence level.\n\n"
        "## Part 1 — tokenizer.py\n\n"
        "Write `tokenizer.py` defining `tokenize(expr: str) -> list[tuple[str, str]]`. Skip "
        "whitespace. A maximal run of ASCII digits becomes `(\"NUMBER\", \"<digits>\")`. Each "
        "of `+ - * / ( )` becomes one of `(\"PLUS\", \"+\")`, `(\"MINUS\", \"-\")`, "
        "`(\"STAR\", \"*\")`, `(\"SLASH\", \"/\")`, `(\"LPAREN\", \"(\")`, `(\"RPAREN\", "
        "\")\")`. After every real token (even for an empty or whitespace-only `expr`), append "
        "a final `(\"EOF\", \"\")` sentinel tuple. Any other character must raise `ValueError`.\n\n"
        "## Part 2 — parser.py\n\n"
        "Write `parser.py` that does `from tokenizer import tokenize` and defines `parse(expr: "
        "str) -> tuple`. Tokenize `expr` first, then run a standard recursive-descent parse:\n"
        "`expr := term (('+' | '-') term)*`\n"
        "`term := factor (('*' | '/') factor)*`\n"
        "`factor := NUMBER | '(' expr ')'`\n"
        "Build a nested-tuple AST: a number becomes `(\"num\", <int value>)`; a binary "
        "operation becomes `(\"binop\", <op string, one of \"+\"/\"-\"/\"*\"/\"/\">, <left "
        "node>, <right node>)`, built left-associatively for a chain at the same precedence "
        "level (so `a - b - c` parses as `(a - b) - c`, not `a - (b - c)`). After parsing the "
        "whole expression, the next token must be `EOF` — anything else (unbalanced "
        "parentheses, a missing operand, trailing garbage) must raise `ValueError`.\n\n"
        "## Part 3 — evaluator.py\n\n"
        "Write `evaluator.py` that does `from parser import parse` and defines `evaluate(expr: "
        "str) -> int | float`. Parse `expr`, then recursively walk the AST: a `\"num\"` node "
        "evaluates to its stored value; a `\"binop\"` node evaluates both children and applies "
        "the operator (`/` is ordinary Python true division, so it can produce a float, and "
        "dividing by zero should raise `ZeroDivisionError` — just letting Python's own `/` "
        "operator do that is fine, don't catch it). A syntax error from the tokenizer/parser "
        "should propagate up as `ValueError` (do not catch and swallow it).\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check all three "
        "modules yourself — tokenize a couple of expressions and print the token list; parse an "
        "expression combining `+`, `*` and parentheses and print the AST to confirm precedence "
        "is structurally correct (multiplication nested inside, not siblings in a flat list); "
        "evaluate a handful of expressions including one dividing to a non-integer result and "
        "one dividing by zero — and confirm every output matches this spec before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "modules": [
                {"module": "tokenizer", "functions": ["tokenize"]},
                {"module": "parser", "functions": ["parse"]},
                {"module": "evaluator", "functions": ["evaluate"]},
            ]
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Three leaves, a strict linear chain (leaf_a -> leaf_b -> leaf_c) matching the real
    dependency graph of a tokenizer/parser/evaluator pipeline: leaf_b imports leaf_a's
    ``tokenize``, leaf_c imports leaf_b's ``parse`` (NOT leaf_a's ``tokenize`` directly — the
    evaluator only ever talks to the parser, exactly like a real interpreter front end). Each
    dependent leaf both templates its upstream leaf's actual outcome via ``{leaf_x}`` and
    independently restates the token/AST shape it depends on, since a weak model's own summary
    of what it built is not fully reliable even when it finished honestly — same pattern as c06
    and c42.
    """
    return {
        "leaves": [
            {
                "id": "leaf_a",
                "instruction": (
                    "Write tokenizer.py, a lexer for a tiny arithmetic-expression language, "
                    "for later use by a parser. Define tokenize(expr: str) -> list[tuple[str, "
                    "str]]. Skip whitespace. A maximal run of ASCII digits becomes (\"NUMBER\", "
                    "\"<digits>\"). Each of + - * / ( ) becomes one of (\"PLUS\", \"+\"), "
                    "(\"MINUS\", \"-\"), (\"STAR\", \"*\"), (\"SLASH\", \"/\"), (\"LPAREN\", "
                    "\"(\"), (\"RPAREN\", \")\"). After every real token (even for an empty or "
                    "whitespace-only expr), append a final (\"EOF\", \"\") sentinel tuple as "
                    "the last element. Any other character must raise ValueError. This module "
                    "must be genuinely correct and usable on its own — a later step imports it "
                    "with `from tokenizer import tokenize` and relies on the exact tuple shapes "
                    "above, so the type strings and structure must match EXACTLY. Use "
                    "write_file to create it, then use run_python to tokenize a couple of "
                    "expressions (including one with parentheses and one with an invalid "
                    "character) and print the results to confirm they match this spec. Then "
                    "finish, and in your summary explicitly describe the token tuple shape."
                ),
                "expect": "tokenizer.py written, defining tokenize(expr) -> list of (TYPE, "
                          "text) tuples ending in an EOF sentinel, sanity-checked with "
                          "run_python",
                "depends_on": [],
            },
            {
                "id": "leaf_b",
                "instruction": (
                    "Leaf A already wrote tokenizer.py, a lexer for a tiny arithmetic-"
                    "expression language. Leaf A reported: {leaf_a}\n\n"
                    "Regardless of exactly what that summary said, tokenizer.py is expected to "
                    "define tokenize(expr: str) -> list[tuple[str, str]]: a maximal run of "
                    "digits becomes (\"NUMBER\", \"<digits>\"); each of + - * / ( ) becomes one "
                    "of (\"PLUS\",\"+\")/(\"MINUS\",\"-\")/(\"STAR\",\"*\")/(\"SLASH\",\"/\")/"
                    "(\"LPAREN\",\"(\")/(\"RPAREN\",\")\"); whitespace is skipped; the list "
                    "always ends with (\"EOF\", \"\"); an unrecognized character raises "
                    "ValueError. If tokenizer.py does not actually expose that API, use "
                    "read_file to inspect it first and adapt your usage accordingly — do not "
                    "guess blindly.\n\n"
                    "Now write parser.py that does `from tokenizer import tokenize` and "
                    "implements parse(expr: str) -> tuple. Tokenize expr first, then run a "
                    "standard recursive-descent parse:\n"
                    "expr := term (('+' | '-') term)*\n"
                    "term := factor (('*' | '/') factor)*\n"
                    "factor := NUMBER | '(' expr ')'\n"
                    "Build a nested-tuple AST: a number becomes (\"num\", <int value>); a "
                    "binary operation becomes (\"binop\", <op string, one of "
                    "\"+\"/\"-\"/\"*\"/\"/\">, <left node>, <right node>), built "
                    "left-associatively for a chain at the same precedence level (so `a - b - "
                    "c` parses as `(a - b) - c`, not `a - (b - c)`). After parsing the whole "
                    "expression, the next token must be EOF — anything else (unbalanced "
                    "parentheses, a missing operand, trailing garbage) must raise ValueError. "
                    "Note that multiplication/division must end up NESTED INSIDE an "
                    "addition/subtraction node in the tree when they appear together (e.g. "
                    "`3 + 4 * 2` must NOT parse as a flat 3-operand structure) — that nesting "
                    "is what encodes correct operator precedence.\n\n"
                    "Use write_file to create parser.py, then use run_python to parse a couple "
                    "of expressions (including one mixing + and *, one with nested parentheses, "
                    "and one malformed one like unbalanced parens) and print the resulting AST "
                    "or the raised error to confirm they match this spec. Fix issues with "
                    "patch_file/run_python until you're confident, then finish."
                ),
                "expect": "parser.py written, importing tokenizer.tokenize and implementing "
                          "parse(expr) -> nested-tuple AST via recursive descent with correct "
                          "precedence",
                "depends_on": ["leaf_a"],
            },
            {
                "id": "leaf_c",
                "instruction": (
                    "Leaf B already wrote parser.py, importing tokenizer and implementing "
                    "parse(expr) -> tuple: a number is (\"num\", <int>), a binary op is "
                    "(\"binop\", <op str>, <left>, <right>) with op one of \"+\"/\"-\"/\"*\"/"
                    "\"/\", nested to reflect correct precedence, raising ValueError on "
                    "malformed input. Leaf B reported: {leaf_b}\n\n"
                    "Regardless of exactly what that summary said, if parser.py does not "
                    "actually expose parse(expr) -> tuple with that AST shape, use read_file to "
                    "inspect it first and adapt your usage accordingly — do not guess "
                    "blindly.\n\n"
                    "Now write evaluator.py that does `from parser import parse` and "
                    "implements evaluate(expr: str) -> int | float. Parse expr, then "
                    "recursively walk the AST: a (\"num\", value) node evaluates to value; a "
                    "(\"binop\", op, left, right) node evaluates both children and applies the "
                    "operator (op is one of \"+\"/\"-\"/\"*\"/\"/\" — \"/\" is ordinary Python "
                    "true division, so it can legitimately produce a float, and dividing by "
                    "zero should raise ZeroDivisionError; just letting Python's own `/` "
                    "operator do that is fine, do not catch it). A ValueError raised by "
                    "tokenizing/parsing should propagate up out of evaluate() unmodified (do "
                    "not catch and swallow it).\n\n"
                    "Use write_file to create evaluator.py, then use run_python to evaluate a "
                    "handful of expressions: a couple mixing + and * with parentheses, one "
                    "dividing to a non-integer float result, one dividing by zero (confirm "
                    "ZeroDivisionError is raised), and one with invalid syntax (confirm "
                    "ValueError propagates). Fix issues with patch_file/run_python until "
                    "confident, then finish."
                ),
                "expect": "evaluator.py written, importing parser.parse and implementing "
                          "evaluate(expr) -> int|float by recursively walking the AST",
                "depends_on": ["leaf_b"],
            },
        ],
        "aggregation": "Confirm tokenizer.py, parser.py, and evaluator.py all exist and report "
                        "the sanity-check results from each leaf.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files",
                         "files": ["tokenizer.py", "parser.py", "evaluator.py"]},
    }
