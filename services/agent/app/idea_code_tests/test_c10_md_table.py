"""
codebench task c10 — hard/hidden, exact-format markdown table generation.

Ground truth for ``format_table`` verified with an independent throwaway reference script
(build widths, ljust each cell, wrap/join with " | ") before being embedded here — not
hand-typed. Do not change these expected strings without re-running that script; a single
stray space silently breaks byte-for-byte grading on a formatting-exactness task.

Algorithm the reference implements (also spelled out in the task statement below):
  1. For each column, width = max(len(cell)) across the header AND every data row in that
     column.
  2. Header/data cell rendering: " " + content.ljust(width) + " " (single space pad each
     side, content left-justified to the column's width), cells joined with "|", row wrapped
     with a leading and trailing "|".
  3. Separator row: literal "---" per column (NOT padded to width), same " | "-join/wrap.
  4. No line has trailing whitespace before its newline (or before EOF on the last line).

Worked example (headers=["Name","Age"], rows=[["Alice","30"],["Bob","5"]]):
    | Name  | Age |
    | --- | --- |
    | Alice | 30  |
    | Bob   | 5   |
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_md_table.py"

_TEST_FILE_CONTENT = '''\
from md_table import format_table


def test_worked_example_exact():
    out = format_table(["Name", "Age"], [["Alice", "30"], ["Bob", "5"]])
    assert out == '| Name  | Age |\\n| --- | --- |\\n| Alice | 30  |\\n| Bob   | 5   |'


def test_uneven_column_widths():
    out = format_table(
        ["Item", "Qty", "Unit Price"],
        [["Widget", "12", "3.50"], ["Extra Long Item Name", "1", "100"]],
    )
    assert out == (
        '| Item                 | Qty | Unit Price |\\n'
        '| --- | --- | --- |\\n'
        '| Widget               | 12  | 3.50       |\\n'
        '| Extra Long Item Name | 1   | 100        |'
    )


def test_single_column():
    out = format_table(["Only"], [["x"], ["yy"]])
    assert out == '| Only |\\n| --- |\\n| x    |\\n| yy   |'


def test_no_trailing_whitespace_on_any_line():
    out = format_table(
        ["Item", "Qty", "Unit Price"],
        [["Widget", "12", "3.50"], ["Extra Long Item Name", "1", "100"]],
    )
    for line in out.split("\\n"):
        assert line == line.rstrip(), f"trailing whitespace in line: {line!r}"
'''

# This task IS the formatting exactness — every case gates the score. There is no
# "degenerate" case to exclude the way c01/c09 exclude trivial n=0/empty-list checks: even
# the single-column case pins real behavior (does the separator/pad logic generalize past
# two columns).
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_worked_example_exact",
    f"{VISIBLE_TEST_PATH}::test_uneven_column_widths",
    f"{VISIBLE_TEST_PATH}::test_single_column",
    f"{VISIBLE_TEST_PATH}::test_no_trailing_whitespace_on_any_line",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c10",
        "title": "markdown-table-formatter",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `md_table.py` that defines a function "
        "`format_table(headers: list[str], rows: list[list[str]]) -> str`.\n\n"
        "It must produce a GitHub-flavored markdown table as a single string (rows joined "
        "by `\\n`, no leading or trailing newline): the header row, then a separator row, "
        "then one row per entry in `rows`.\n\n"
        "Exact formatting rules:\n"
        "1. For each column, compute `width` = the length of the longest cell in that "
        "column, counting the header cell too.\n"
        "2. Render every header/data cell as a single space, then the cell's text "
        "left-justified (padded with spaces on the right) to `width`, then a single "
        "space. Join a row's rendered cells with `|`, and add a leading `|` and a "
        "trailing `|` to the whole row.\n"
        "3. The separator row uses the literal text `---` for every column (NOT padded "
        "to the column width) — rendered and joined the same way as rule 2 (single space "
        "each side, `|`-joined, wrapped in leading/trailing `|`).\n"
        "4. No line may have trailing whitespace immediately before its `\\n` (or before "
        "the end of the string, on the last line).\n\n"
        "Worked example — for `headers=[\"Name\", \"Age\"]`, "
        "`rows=[[\"Alice\", \"30\"], [\"Bob\", \"5\"]]`, the function must return EXACTLY "
        "this string (shown here between markers so whitespace is unambiguous; do not "
        "include the marker lines themselves):\n"
        "-----BEGIN-----\n"
        "| Name  | Age |\n"
        "| --- | --- |\n"
        "| Alice | 30  |\n"
        "| Bob   | 5   |\n"
        "-----END-----\n\n"
        "No test file is visible for this task — write md_table.py from this spec, then "
        "use run_pytest on any test file you create yourself to sanity-check your own "
        "understanding before finishing (the canonical grading tests are hidden and will "
        "be run separately)."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter test file is exposed to the agent. It works purely from
    get_task_statement()'s spec text; canonical tests only ever live in the grading
    payload / private/tests/ (see materialize_task.py's module docstring)."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "md_table", "functions": ["format_table"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Hand-authored offline plan (mirrors idea_tests/'s get_compiled_plan() convention) —
    a single leaf is enough for a one-function task; multi-leaf plans are for tasks that
    genuinely decompose into independent pieces."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write md_table.py implementing format_table(headers, rows): compute "
                    "per-column width = max(len) across header and every row's cell in that "
                    "column; render each cell as ' ' + content.ljust(width) + ' ', join a "
                    "row's cells with '|', wrap with leading/trailing '|'; the separator row "
                    "uses literal '---' per column (not padded to width) rendered the same "
                    "way; join all rows with '\\n', no leading/trailing newline, no trailing "
                    "whitespace on any line. Match the worked example in the task statement "
                    "character-for-character. Use write_file to create it. Write your own "
                    "quick test file and use run_pytest to sanity-check the worked example "
                    "before finishing — the real grading tests are hidden."
                ),
                "expect": "md_table.py written implementing format_table per the exact spec",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm md_table.py exists and defines format_table.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["md_table.py"]},
    }
