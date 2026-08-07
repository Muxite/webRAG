"""
codebench task c44 — hard/hidden, small data-processing pipeline, THREE-leaf dependency chain.

Fourth multi-leaf task (after c06's two-leaf precedent, c42's validation chain, and c43's
compiler-lite chain). This one is a classic small ETL pipeline over comma-separated
"date,category,amount" sales-log lines:
  * leaf_a builds ``schema.py`` — ``parse_row(fields: list[str]) -> dict``, which validates and
    converts ONE already-split row of three raw strings into a typed record, raising
    ``ValueError`` on any schema violation. Independently useful/testable on its own.
  * leaf_b (``depends_on: ["leaf_a"]``) builds ``loader.py``'s
    ``load_records(text) -> tuple[list[dict], list[str]]``, which IMPORTS ``schema.parse_row``
    and applies it line-by-line to a whole text blob, skipping blank lines and collecting a
    ``"line N: <reason>"`` entry for every line that fails to parse rather than aborting the
    whole load — this is where per-row validation becomes resilient batch loading.
  * leaf_c (``depends_on: ["leaf_b"]``) builds ``report.py``'s ``build_report(text) -> dict``,
    which IMPORTS ``loader.load_records`` and folds the parsed records into a summary (total
    amount, a per-category breakdown, record/error counts). leaf_c does NOT depend on leaf_a
    directly — it never calls ``parse_row`` itself, exactly mirroring the real dependency graph
    of a schema/loader/reporter pipeline (the reporter only ever talks to the loader).

See ``services/agent/app/testing/execution_compiled_code.py``'s ``_dep_text``/
``UPSTREAM_INCOMPLETE_MARKER`` for exactly what lands in a ``{leaf_x}`` placeholder at runtime.
Because that text is not fully reliable even in the happy path, both dependent leaves'
instructions independently RESTATE the exact contract of the module they depend on, exactly like
c06/c42/c43 do, rather than trusting the upstream leaf's self-report alone.

Deliberately loose on the exact WORDING of ``schema.py``'s ``ValueError`` messages (only the
TYPE is asserted at that layer) — pinning exact prose there would over-constrain leaf_a for no
real signal. What IS pinned exactly is the loader's ``"line N: ..."`` PREFIX format (a real
structural contract downstream code depends on) and every numeric aggregation in the report.

Ground truth below was derived by actually RUNNING a reference schema/loader/report chain (not
worked out by hand) — see the offline validator test file's own independently-written
reimplementation for the second, separate check. Key verified values for the loader (a 7-line
blob with one blank line and two schema-invalid lines):
    text = "2026-01-01,food,12.50\\n2026-01-02,transport,7.25\\nbadline\\n"
           "2026-01-03,food,3.50\\n2026-01-04,transport,-1\\n\\n2026-01-05,books,20"
    load_records(text) -> (
        [{"date":"2026-01-01","category":"food","amount":12.5},
         {"date":"2026-01-02","category":"transport","amount":7.25},
         {"date":"2026-01-03","category":"food","amount":3.5},
         {"date":"2026-01-05","category":"books","amount":20.0}],
        [<a "line 3: ..." error>, <a "line 5: ..." error>],   # exactly 2, correctly numbered
    )

Hardening note (this is the second generation of this task — the first version was live-verified
to score a perfect 1.0/1.0 via aider, so the report-level canonical check below was sharpened with
a larger, floating-point-precision-sensitive dataset): the report test now uses a 10-line blob
that adds three more rows on top of the loader scenario above, with amounts of "1.001", "2.025",
and "0.075" — deliberately chosen so that "sum every record's raw amount, then round the sum ONCE"
(the spec's actual contract) and the very plausible-looking alternative "round each amount to 2
decimal places as it's parsed/added, then sum the already-rounded pieces" produce DIFFERENT final
totals, purely because 2-3-decimal-digit amounts don't split into cents evenly. Getting this right
requires two things working together across the chain: schema.py (leaf_a) must NOT round the
amount when parsing (return the raw ``float()`` result), and report.py (leaf_c) must sum the raw
per-record amounts and round only the final sums — never round an intermediate/per-record value.
Either module rounding early, even though each individually still "looks correct" in isolation,
silently shifts the final report by a cent.

CAUTION for anyone touching these numbers again: floating-point summation in Python is NOT
associative, and CPython 3.12+'s builtin ``sum()`` uses a compensated (Neumaier) summation
algorithm for floats that is measurably more accurate than a naive ``total += x`` loop — so a
badly-chosen dataset can make the "correct" (sum-raw-then-round-once) total_amount itself depend
on *which* correct-in-spirit summation strategy an implementation happens to use, which would be
genuinely unfair, not hard. The values below were verified to agree across ``sum()``, a manual
accumulator loop (both forward and reverse iteration order), and ``math.fsum()`` (true
arbitrary-precision-equivalent summation) — see the offline validator's own stability check. Any
future edit to these numbers MUST re-verify that agreement, not just re-run one summation style.
Key verified values for the (larger) report scenario:
    report_text = "2026-01-01,food,12.50\\n2026-01-02,transport,7.25\\nbadline\\n"
                  "2026-01-03,food,3.50\\n2026-01-04,transport,-1\\n\\n2026-01-05,books,20\\n"
                  "2026-01-06,food,1.001\\n2026-01-07,food,2.025\\n2026-01-08,transport,0.075"
    build_report(report_text) -> {"total_amount": 46.35,
                                   "by_category": {"books": 20.0, "food": 19.03,
                                                    "transport": 7.33},
                                   "record_count": 7, "error_count": 2}
        (category keys come out alphabetical: books, food, transport; a round-per-record-then-sum
        implementation instead produces total_amount 46.34, food 19.02, and transport 7.32 — all
        three wrong, by exactly one cent each)
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_report.py"

_TEST_FILE_CONTENT = '''\
import pytest
from schema import parse_row
from loader import load_records
from report import build_report


# --- leaf_a: schema.py -----------------------------------------------------------------------

def test_schema_parses_valid_row():
    assert parse_row(["2026-01-01", "food", "12.50"]) == {
        "date": "2026-01-01", "category": "food", "amount": 12.5,
    }


def test_schema_rejects_wrong_field_count():
    with pytest.raises(ValueError):
        parse_row(["2026-01-01", "food"])


def test_schema_rejects_empty_date_or_category():
    with pytest.raises(ValueError):
        parse_row(["", "food", "5"])
    with pytest.raises(ValueError):
        parse_row(["2026-01-01", "  ", "5"])


def test_schema_rejects_unparsable_amount():
    with pytest.raises(ValueError):
        parse_row(["2026-01-01", "food", "abc"])


def test_schema_rejects_negative_amount():
    with pytest.raises(ValueError):
        parse_row(["2026-01-01", "food", "-5"])


# --- leaf_b: loader.py ------------------------------------------------------------------------

def test_loader_parses_valid_lines_and_skips_blanks():
    text = "2026-01-01,food,12.50\\n\\n2026-01-02,transport,7.25"
    records, errors = load_records(text)
    assert records == [
        {"date": "2026-01-01", "category": "food", "amount": 12.5},
        {"date": "2026-01-02", "category": "transport", "amount": 7.25},
    ]
    assert errors == []


def test_loader_collects_errors_with_correct_line_numbers():
    text = (
        "2026-01-01,food,12.50\\n"
        "2026-01-02,transport,7.25\\n"
        "badline\\n"
        "2026-01-03,food,3.50\\n"
        "2026-01-04,transport,-1\\n"
        "\\n"
        "2026-01-05,books,20"
    )
    records, errors = load_records(text)
    assert len(records) == 4
    assert len(errors) == 2
    assert errors[0].startswith("line 3:")
    assert errors[1].startswith("line 5:")


# --- leaf_c: report.py ------------------------------------------------------------------------

# Three more rows than the loader scenario above -- amounts chosen so that summing the RAW
# per-record amounts and rounding once (the spec's contract) diverges from rounding each amount
# as it is parsed/added and then summing the rounded pieces (a very plausible-looking but wrong
# alternative), purely because of binary floating-point representation.
_REPORT_TEXT = (
    "2026-01-01,food,12.50\\n"
    "2026-01-02,transport,7.25\\n"
    "badline\\n"
    "2026-01-03,food,3.50\\n"
    "2026-01-04,transport,-1\\n"
    "\\n"
    "2026-01-05,books,20\\n"
    "2026-01-06,food,1.001\\n"
    "2026-01-07,food,2.025\\n"
    "2026-01-08,transport,0.075"
)


def test_report_rounds_the_final_sum_once_not_per_record():
    # Isolates the precision trap from the rest of the pipeline: 7.25 + 0.075, rounded once at
    # the end, is 7.33 -- but round(7.25, 2) + round(0.075, 2) is 7.32, a plausible-looking wrong
    # answer that only shows up with amounts carrying more than 2 decimal digits.
    text = "2026-01-01,transport,7.25\\n2026-01-02,transport,0.075"
    report = build_report(text)
    assert report == {
        "total_amount": 7.33,
        "by_category": {"transport": 7.33},
        "record_count": 2,
        "error_count": 0,
    }


def test_report_aggregates_by_category():
    report = build_report(_REPORT_TEXT)
    assert report == {
        "total_amount": 46.35,
        "by_category": {"books": 20.0, "food": 19.03, "transport": 7.33},
        "record_count": 7,
        "error_count": 2,
    }


def test_report_orders_categories_alphabetically():
    report = build_report(_REPORT_TEXT)
    assert list(report["by_category"].keys()) == ["books", "food", "transport"]
'''

# The five schema-level tests only exercise ONE row at a time, and the isolated rounding-order
# check just above is a supporting/bonus diagnostic -- a stub loader/reporter could still fail
# every one of the four below even while all of those pass. The four that gate the score exercise
# the actual COMPOSITION at each layer: the loader correctly skipping blanks while still parsing
# real rows, the loader's line-number bookkeeping surviving a mix of blank and failing lines (an
# off-by-one here is a classic bug), and the reporter's full aggregation + alphabetical ordering,
# which only passes if leaf_c correctly folds leaf_b's parsed records AND sums their raw (never
# early-rounded) amounts, rounding only the final per-category/total sums -- a plausible-looking
# "round as you go" implementation still produces a total/by_category dict that looks completely
# reasonable, just off by a cent, so this can't be caught by eyeballing the output.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_loader_parses_valid_lines_and_skips_blanks",
    f"{_TEST_FILE_PATH}::test_loader_collects_errors_with_correct_line_numbers",
    f"{_TEST_FILE_PATH}::test_report_aggregates_by_category",
    f"{_TEST_FILE_PATH}::test_report_orders_categories_alphabetically",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c44",
        "title": "sales-pipeline-schema-loader-report",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You are building a small sales-log ETL pipeline, split across three files: a "
        "per-row schema validator, a loader that applies it across a whole text blob, and a "
        "reporter that aggregates the result. Each row represents one sale as three "
        "comma-separated fields: date, category, amount.\n\n"
        "## Part 1 — schema.py\n\n"
        "Write `schema.py` defining `parse_row(fields: list[str]) -> dict`. `fields` is a list "
        "(or tuple) of exactly 3 raw strings, in order: date, category, amount. Strip leading/"
        "trailing whitespace from each of the three before using it. Raise `ValueError` if: "
        "`fields` does not have exactly 3 elements; the stripped date is empty; the stripped "
        "category is empty; the stripped amount string does not parse as a float (use Python's "
        "`float()`, don't hand-roll number parsing); or the parsed amount is negative. On "
        "success, return `{\"date\": <stripped date>, \"category\": <stripped category>, "
        "\"amount\": <amount as float>}` — store `amount` EXACTLY as `float()` returns it, with "
        "no rounding of any kind. Rounding, if any, happens later, in report.py, not here — "
        "resist the temptation to \"clean up\" a money value at parse time, since rounding at "
        "the wrong stage of the pipeline can silently shift a downstream sum by a cent.\n\n"
        "## Part 2 — loader.py\n\n"
        "Write `loader.py` that does `from schema import parse_row` and defines "
        "`load_records(text: str) -> tuple[list[dict], list[str]]`. Split `text` into lines, "
        "numbering them starting at 1 for the first line — count EVERY line of the input "
        "(including ones you end up skipping) so line numbers always match the original text. "
        "For each line: if it is empty or all-whitespace, skip it (produces nothing). "
        "Otherwise split it on commas into fields (plain `str.split(\",\")`, no CSV quoting "
        "needed) and call `parse_row(fields)`. If it succeeds, append the returned record to "
        "the records list. If it raises `ValueError`, do NOT let the whole load fail — catch "
        "it, append the string `f\"line {lineno}: {exc}\"` (where `exc` is the caught "
        "exception, so `str(exc)` is its message) to the errors list, and continue to the next "
        "line. Return `(records, errors)` once every line has been processed.\n\n"
        "## Part 3 — report.py\n\n"
        "Write `report.py` that does `from loader import load_records` and defines "
        "`build_report(text: str) -> dict`. Call `load_records(text)` to get `(records, "
        "errors)`, then return a dict with EXACTLY these four keys: `\"total_amount\"` (float, "
        "the SUM of every record's raw `\"amount\"`, rounded to 2 decimal places ONCE, after "
        "summing), `\"by_category\"` (dict mapping each distinct category present in `records` "
        "to the SUM of that category's raw amounts, each rounded to 2 decimal places ONCE, "
        "after summing, with keys in ALPHABETICAL ascending order), `\"record_count\"` (int, "
        "`len(records)`), and `\"error_count\"` (int, `len(errors)`).\n\n"
        "Precision matters here: sum the raw, un-rounded amounts first, and call `round(..., 2)` "
        "exactly once per resulting total (once for `total_amount`, once per category). Do NOT "
        "round each record's amount before adding it to a running total, and do NOT round an "
        "intermediate running sum on every addition — either of those produces a dict that still "
        "looks entirely plausible but is wrong by a cent on some inputs, purely because of how "
        "Python's binary floating-point representation interacts with `round()`: summing raw "
        "values and rounding once is NOT always the same as rounding pieces along the way and "
        "combining the rounded pieces, once amounts have more than 2 decimal digits. Don't "
        "assume you know which of your own test amounts happen to trigger this — verify your "
        "actual `report.py` output with run_python rather than trusting mental arithmetic.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check all three "
        "modules yourself — call `parse_row` on a couple of valid and invalid field lists; call "
        "`load_records` on a small multi-line blob with at least one valid line, one blank "
        "line, and one line that fails schema validation, and confirm the line numbering in "
        "`errors` is correct; call `build_report` on a blob spanning at least two categories, "
        "including at least two same-category rows whose amounts have 3+ decimal digits, and "
        "confirm the four keys, category grouping, rounding (computed by actually running the "
        "code, not by hand), alphabetical key order, and counts all match this spec — before "
        "finishing."
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
                {"module": "schema", "functions": ["parse_row"]},
                {"module": "loader", "functions": ["load_records"]},
                {"module": "report", "functions": ["build_report"]},
            ]
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Three leaves, a strict linear chain (leaf_a -> leaf_b -> leaf_c) matching the real
    dependency graph of a schema/loader/reporter pipeline: leaf_b imports leaf_a's
    ``parse_row``, leaf_c imports leaf_b's ``load_records`` (NOT leaf_a's ``parse_row``
    directly — the reporter only ever talks to the loader, exactly like a real ETL pipeline).
    Each dependent leaf both templates its upstream leaf's actual outcome via ``{leaf_x}`` and
    independently restates the contract it depends on, since a weak model's own summary of what
    it built is not fully reliable even when it finished honestly — same pattern as c06/c42/c43.
    """
    return {
        "leaves": [
            {
                "id": "leaf_a",
                "instruction": (
                    "Write schema.py, a per-row validator for a small sales-log ETL pipeline, "
                    "for later use by a loader. Define parse_row(fields: list[str]) -> dict. "
                    "fields is a list (or tuple) of exactly 3 raw strings, in order: date, "
                    "category, amount. Strip leading/trailing whitespace from each of the "
                    "three before using it. Raise ValueError if: fields does not have exactly "
                    "3 elements; the stripped date is empty; the stripped category is empty; "
                    "the stripped amount string does not parse as a float (use Python's "
                    "float(), don't hand-roll number parsing); or the parsed amount is "
                    "negative. On success, return {\"date\": <stripped date>, \"category\": "
                    "<stripped category>, \"amount\": <amount as float>} -- store amount "
                    "EXACTLY as float() returns it, with no rounding of any kind. A later step "
                    "sums many of these amounts and rounds the sums, not the individual "
                    "records, so rounding here (even to \"clean up\" a money value) would "
                    "silently corrupt that later arithmetic. This module must be genuinely "
                    "correct and usable on its own — a later step imports it with `from schema "
                    "import parse_row` and relies on this exact behavior, so the function name, "
                    "argument shape, and success/failure conditions must match EXACTLY (the "
                    "exact wording of your ValueError messages does not matter, only that "
                    "ValueError is raised in each of those cases). Use write_file to create it, "
                    "then use run_python to call parse_row on a couple of valid field lists "
                    "(including at least one with an amount that has 3+ decimal digits, e.g. "
                    "\"9.999\", and confirm the returned amount is NOT rounded) and a couple of "
                    "invalid ones (wrong count, empty category, unparsable amount, negative "
                    "amount), "
                    "and print the results/errors to confirm they match this spec. Then finish, "
                    "and in your summary explicitly describe the returned dict's three keys."
                ),
                "expect": "schema.py written, defining parse_row(fields) -> {date, category, "
                          "amount} with the described validation and NO rounding of amount, "
                          "sanity-checked with run_python",
                "depends_on": [],
            },
            {
                "id": "leaf_b",
                "instruction": (
                    "Leaf A already wrote schema.py, a per-row validator for a sales-log "
                    "pipeline. Leaf A reported: {leaf_a}\n\n"
                    "Regardless of exactly what that summary said, schema.py is expected to "
                    "define parse_row(fields: list[str]) -> dict: given a list of exactly 3 "
                    "raw strings [date, category, amount], it strips each, and returns "
                    "{\"date\": ..., \"category\": ..., \"amount\": <float>} where amount is "
                    "the raw, un-rounded result of float(); it raises ValueError if there "
                    "aren't exactly 3 fields, if the stripped date or category is empty, if "
                    "the amount string doesn't parse as a float, or if the parsed amount is "
                    "negative. If schema.py does not actually expose that API, use read_file to "
                    "inspect it first and adapt your usage accordingly — do not guess blindly.\n\n"
                    "Now write loader.py that does `from schema import parse_row` and "
                    "implements load_records(text: str) -> tuple[list[dict], list[str]]. Split "
                    "text into lines, numbering them starting at 1 for the first line — count "
                    "EVERY line of the input (including ones you end up skipping) so line "
                    "numbers always match the original text. For each line: if it is empty or "
                    "all-whitespace, skip it (produces nothing — no record, no error). "
                    "Otherwise split it on commas into fields (plain str.split(\",\"), no CSV "
                    "quoting needed) and call parse_row(fields). If it succeeds, append the "
                    "returned record dict to the records list. If it raises ValueError, do NOT "
                    "let the whole load fail — catch it, append the string f\"line {lineno}: "
                    "{exc}\" (where exc is the caught exception, so str(exc) is its message) "
                    "to the errors list, and continue on to the next line. Return (records, "
                    "errors) once every line has been processed.\n\n"
                    "Use write_file to create loader.py, then use run_python to load a small "
                    "multi-line text blob containing at least one fully valid line, one blank "
                    "line, and one line that fails schema validation, and print the (records, "
                    "errors) result to confirm: valid lines produce records in the right "
                    "order, the blank line produces nothing, and the failing line produces an "
                    "errors entry prefixed with its correct 1-based line number (count the "
                    "blank line too when computing that number). Fix issues with patch_file/"
                    "run_python until you're confident, then finish."
                ),
                "expect": "loader.py written, importing schema.parse_row and implementing "
                          "load_records(text) -> (records, errors) that skips blank lines and "
                          "collects per-line errors with correct 1-based line numbers",
                "depends_on": ["leaf_a"],
            },
            {
                "id": "leaf_c",
                "instruction": (
                    "Leaf B already wrote loader.py, importing schema and implementing "
                    "load_records(text) -> (records, errors): records is a list of "
                    "{\"date\",\"category\",\"amount\"} dicts for every successfully-parsed "
                    "non-blank line, errors is a list of \"line N: <reason>\" strings (N is "
                    "1-based) for every line that failed to parse (blank lines produce "
                    "neither). Leaf B reported: {leaf_b}\n\n"
                    "Regardless of exactly what that summary said, if loader.py does not "
                    "actually expose load_records(text) -> (records, errors) with that "
                    "behavior, use read_file to inspect it first and adapt your usage "
                    "accordingly — do not guess blindly.\n\n"
                    "Now write report.py that does `from loader import load_records` and "
                    "implements build_report(text: str) -> dict. Call load_records(text) to "
                    "get (records, errors), then return a dict with EXACTLY these four keys: "
                    "\"total_amount\" (float, the SUM of every record's raw \"amount\", rounded "
                    "to 2 decimal places ONCE, after summing), \"by_category\" (dict mapping "
                    "each distinct category present in records to the SUM of that category's "
                    "raw amounts, each rounded to 2 decimal places ONCE, after summing, with "
                    "keys in ALPHABETICAL ascending order — build it by inserting the "
                    "categories in sorted order, since a plain Python dict preserves insertion "
                    "order), \"record_count\" (int, len(records)), and \"error_count\" (int, "
                    "len(errors)).\n\n"
                    "Precision matters here: sum the raw, un-rounded amounts first, and call "
                    "round(..., 2) exactly once per resulting total (once for total_amount, "
                    "once per category). Do NOT round each record's amount before adding it to "
                    "a running total, and do NOT round a running sum on every addition -- either "
                    "of those produces a dict that still looks entirely plausible but is off by "
                    "a cent on inputs with more than 2 decimal digits, purely because of how "
                    "Python's binary floating-point representation interacts with round(): "
                    "summing raw values and rounding once is NOT always the same as rounding "
                    "pieces along the way and combining the rounded pieces. Don't assume you "
                    "know which of your own test amounts happen to trigger this -- verify your "
                    "actual output with run_python rather than trusting mental arithmetic.\n\n"
                    "Use write_file to create report.py, then use run_python to build a report "
                    "from a small multi-line text blob spanning at least two different "
                    "categories, including one invalid line and at least two same-category rows "
                    "whose amounts have 3+ decimal digits, and print the result to confirm the "
                    "four keys, the category grouping, the rounding (computed by actually "
                    "running the code), the alphabetical key order, and the counts all match "
                    "this spec. Fix issues with patch_file/run_python until confident, then "
                    "finish."
                ),
                "expect": "report.py written, importing loader.load_records and implementing "
                          "build_report(text) -> {total_amount, by_category, record_count, "
                          "error_count} exactly as specified, rounding sums once from raw "
                          "amounts (never rounding per-record or per-addition)",
                "depends_on": ["leaf_b"],
            },
        ],
        "aggregation": "Confirm schema.py, loader.py, and report.py all exist and report the "
                        "sanity-check results from each leaf.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files",
                         "files": ["schema.py", "loader.py", "report.py"]},
    }
