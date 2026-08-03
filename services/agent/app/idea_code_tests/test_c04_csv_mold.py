"""
codebench task c04 — hard/visible, structured text transformation.

Ground truth for ``csv_to_json`` (parse CSV text with the first row as header, return
`json.dumps(records, sort_keys=False)` of an array of objects, one per data row, with
keys in header order) was computed by two independently-written reference
implementations (csv.DictReader vs. csv.reader + manual header-zip) built entirely on
Python's own `csv` and `json` modules, run and cross-checked (via json.loads structural
equality, since dict key order across the two zip strategies need not be byte-identical)
to agree exactly before any value below was written down — see idea_code_test_c04_test.py
for the reimplementation that re-verifies this at test time. Do not hand-edit these
literals without re-deriving.

    csv_to_json("name,age,city\\nAlice,30,NYC\\nBob,25,LA\\n")
        -> parses to [{"name": "Alice", "age": "30", "city": "NYC"},
                       {"name": "Bob", "age": "25", "city": "LA"}]

    csv_to_json(<CSV with a quoted field "Doe, Jane" and a quoted field containing an
                 escaped internal double-quote pair around the word hi>)
        -> parses to [{"name": "Doe, Jane", "quote": 'She said "hi"', "age": "41"},
                       {"name": "Bob", "quote": "plain", "age": "25"}]
        (see idea_code_test_c04_test.py / the test file's own literal for the exact
        CSV-quoted source text — omitted here to avoid an accidental triple-double-quote
        inside this triple-double-quoted docstring)

    csv_to_json("a,b\\n1,2\\n")
        -> parses to [{"a": "1", "b": "2"}]
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_csv_mold.py"

_TEST_FILE_CONTENT = '''\
import json

from csv_mold import csv_to_json


def test_basic_table():
    csv_text = "name,age,city\\nAlice,30,NYC\\nBob,25,LA\\n"
    expected = [
        {"name": "Alice", "age": "30", "city": "NYC"},
        {"name": "Bob", "age": "25", "city": "LA"},
    ]
    assert json.loads(csv_to_json(csv_text)) == expected


def test_quoted_fields_with_embedded_comma_and_quotes():
    csv_text = (
        'name,quote,age\\n'
        '"Doe, Jane","She said ""hi""",41\\n'
        'Bob,plain,25\\n'
    )
    expected = [
        {"name": "Doe, Jane", "quote": 'She said "hi"', "age": "41"},
        {"name": "Bob", "quote": "plain", "age": "25"},
    ]
    assert json.loads(csv_to_json(csv_text)) == expected


def test_single_data_row():
    csv_text = "a,b\\n1,2\\n"
    expected = [{"a": "1", "b": "2"}]
    assert json.loads(csv_to_json(csv_text)) == expected


def test_returns_valid_json_string():
    csv_text = "a,b\\n1,2\\n"
    result = csv_to_json(csv_text)
    assert isinstance(result, str)
    reparsed = json.loads(result)
    assert isinstance(reparsed, list)


def test_key_order_matches_header_order():
    csv_text = "z,a,m\\n1,2,3\\n"
    result = csv_to_json(csv_text)
    reparsed = json.loads(result)
    assert list(reparsed[0].keys()) == ["z", "a", "m"]
'''

# The basic-table, quoted-field, and key-order cases are what actually distinguish a
# real csv.DictReader-based parse from a naive split-on-comma hack; the
# returns-valid-json-string check is a bonus format sanity check only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_basic_table",
    f"{VISIBLE_TEST_PATH}::test_quoted_fields_with_embedded_comma_and_quotes",
    f"{VISIBLE_TEST_PATH}::test_single_data_row",
    f"{VISIBLE_TEST_PATH}::test_key_order_matches_header_order",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c04",
        "title": "csv-to-json-mold",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `csv_mold.py` that defines a function "
        "`csv_to_json(csv_text: str) -> str`.\n\n"
        "`csv_text` is a CSV-formatted string whose first row is a header row. "
        "`csv_to_json` must parse it and return a JSON string: an array of objects, one "
        "object per data row (i.e. every row after the header), where each object's "
        "keys are the header columns IN THE SAME ORDER as they appear in the header "
        "row, and each object's values are that row's cell values as strings.\n\n"
        "Produce the JSON string using `json.dumps(records, sort_keys=False)` "
        "semantics exactly — do not reorder keys alphabetically or otherwise; the key "
        "order in each output object must match the header's column order, not be "
        "sorted.\n\n"
        "Use Python's built-in `csv` module to do the actual CSV parsing (e.g. "
        "`csv.DictReader`) rather than hand-rolling a comma-splitter — this matters "
        "because real CSV fields can be quoted and can contain embedded commas or "
        "escaped quote characters, and `csv.DictReader` already handles that correctly.\n\n"
        "A visible test file is already present at tests/test_csv_mold.py — run it "
        "(run_pytest) and keep revising csv_mold.py until every test in it passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "csv_mold", "functions": ["csv_to_json"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write csv_mold.py implementing csv_to_json(csv_text: str) -> str: "
                    "parse csv_text with Python's csv module (e.g. csv.DictReader, "
                    "first row is the header), build a list of one dict per data row "
                    "with keys in header column order, and return "
                    "json.dumps(records, sort_keys=False). Do not hand-roll CSV parsing "
                    "— use the csv module so quoted fields with embedded commas/quotes "
                    "are handled correctly. Use write_file to create it. Then use "
                    "run_pytest on tests/test_csv_mold.py. If any test fails, use "
                    "read_file/patch_file to fix csv_mold.py and run_pytest again. Once "
                    "every test passes, finish."
                ),
                "expect": "csv_mold.py written; tests/test_csv_mold.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm csv_mold.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["csv_mold.py"]},
    }
