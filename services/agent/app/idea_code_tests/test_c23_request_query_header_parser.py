"""
codebench task c23 — hard/visible, HTTP-shaped API logic: request line + query + header parsing.

Third task in the API/HTTP-server-logic gap, and deliberately the most "parse untyped input into
a typed structure" shaped of the three (distinct from c21's dispatch logic and c22's stateful
throttle logic). No real network I/O: a "raw request" is just a (method, raw_path, header_lines)
tuple of plain strings passed straight to `parse_request`.

The two edge cases this task is built to catch are both about percent-encoding and `+`-as-space,
which behave DIFFERENTLY in the path vs the query string of a URL, and both about headers, which
are case-insensitive and may legally repeat:
  1. `%XX` percent-escapes decode the same way everywhere (e.g. `%20` -> a literal space,
     anywhere in the URL). But a literal `+` character means SPACE only inside the query string
     (a `application/x-www-form-urlencoded` convention) — inside the PATH, `+` is just a literal
     plus sign and must NOT be decoded to a space. A solution that decodes the whole raw string
     with one blanket "+-means-space" pass gets the path wrong.
  2. HTTP header names are case-insensitive (`Content-Type` and `content-type` name the same
     header) and the SAME header name may legally appear on multiple lines, in which case the
     values must be joined with `", "` in the order they appeared — not silently overwritten by
     whichever one comes last, which is what a naive `dict(...)`-style header parse does.

Query parameters always come back as a LIST of decoded values under each key (even when the key
appears exactly once), since a query string may repeat any key any number of times and a caller
needs one consistent shape regardless of how many times a given key showed up.

Ground truth for every embedded case was independently confirmed by hand-tracing the spec's own
stated algorithm character by character (partition-on-first-`?`, partition-on-first-`:`, explicit
`unquote` vs `unquote_plus` at each step) AND by running the exact canonical test file against a
deliberately naive implementation built out of `urllib.parse.urlsplit` + `parse_qs` +
`unquote_plus` applied uniformly (a plausible first instinct given Python's stdlib) plus a plain
`dict` for headers — it failed 4 of 9 tests, including three of the five keystone cases, which
confirms path-vs-query `+` handling and header case/duplicate handling are not redundant with the
easier no-query-string case.
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_request_parse.py"

_TEST_FILE_CONTENT = '''\
import pytest
from request_parse import parse_request


def test_no_query_string_returns_empty_dict():
    result = parse_request("get", "/hello", [])
    assert result == {"method": "GET", "path": "/hello", "query": {}, "headers": {}}


def test_repeated_query_key_collects_list():
    result = parse_request("GET", "/search?q=hello+world&tag=a&tag=b&page=2", [])
    assert result["path"] == "/search"
    assert result["query"] == {"q": ["hello world"], "tag": ["a", "b"], "page": ["2"]}


def test_path_percent_and_plus_handling():
    result_a = parse_request("GET", "/files/my%20doc.txt?name=foo+bar", [])
    assert result_a["path"] == "/files/my doc.txt"
    assert result_a["query"] == {"name": ["foo bar"]}

    result_b = parse_request("GET", "/note+book?title=foo", [])
    assert result_b["path"] == "/note+book"
    assert result_b["query"] == {"title": ["foo"]}


def test_bare_query_key_without_equals():
    result = parse_request("GET", "/toggle?flag&on=1", [])
    assert result["query"] == {"flag": [""], "on": ["1"]}


def test_headers_case_insensitive_lookup():
    result = parse_request("GET", "/x", ["Content-Type: application/json"])
    assert result["headers"] == {"content-type": "application/json"}


def test_duplicate_headers_joined_with_comma():
    result = parse_request(
        "GET",
        "/x",
        ["Accept: text/html", "Accept: application/xml", "X-Request-Id:   abc123  "],
    )
    assert result["headers"]["accept"] == "text/html, application/xml"
    assert result["headers"]["x-request-id"] == "abc123"


def test_malformed_header_line_raises_value_error():
    with pytest.raises(ValueError):
        parse_request("GET", "/x", ["BadHeaderNoColon"])


def test_method_is_uppercased():
    result = parse_request("post", "/x", [])
    assert result["method"] == "POST"


def test_trailing_question_mark_empty_query():
    result = parse_request("GET", "/plain?", [])
    assert result["path"] == "/plain"
    assert result["query"] == {}
'''

# The path-vs-query +/percent-decoding split and the header case-insensitivity/duplicate-joining
# rules are the sharp edges a naive "decode everything the same way, dict() the headers"
# implementation gets wrong (see module docstring's naive-mutant note) — that's
# test_repeated_query_key_collects_list, test_path_percent_and_plus_handling,
# test_headers_case_insensitive_lookup, and test_duplicate_headers_joined_with_comma. The
# malformed-header contract is also load-bearing (silently accepting garbage input is a real
# correctness bug, not a cosmetic one) so it gates too. The no-query-string base case, the bare
# key without '=', the method-case normalization, and the trailing-'?' edge case are supporting/
# bonus credit.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_repeated_query_key_collects_list",
    f"{VISIBLE_TEST_PATH}::test_path_percent_and_plus_handling",
    f"{VISIBLE_TEST_PATH}::test_headers_case_insensitive_lookup",
    f"{VISIBLE_TEST_PATH}::test_duplicate_headers_joined_with_comma",
    f"{VISIBLE_TEST_PATH}::test_malformed_header_line_raises_value_error",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c23",
        "title": "request-query-header-parser",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `request_parse.py` that defines a function "
        "`parse_request(method: str, raw_path: str, header_lines: list) -> dict`.\n\n"
        "This models the parsing step at the front of an HTTP API server, turning raw text "
        "into a typed structure — no real network I/O is involved; `header_lines` is just a "
        "plain list of strings, each one line of a raw HTTP header block (e.g. "
        "`\"Content-Type: application/json\"`).\n\n"
        "`parse_request` must return a dict with exactly these four keys:\n\n"
        "- `\"method\"`: `method` upper-cased (e.g. `\"get\"` -> `\"GET\"`).\n"
        "- `\"path\"`: the path portion of `raw_path` (everything before a literal `?`, or "
        "the whole string if there is no `?`), with any `%XX` percent-escapes decoded (e.g. "
        "`%20` -> a space). A literal `+` character in the path is NOT a space — leave it "
        "exactly as `+` (percent-encoding, `%2B`, would be how a literal `+` in a real path "
        "segment gets escaped in practice, but a bare `+` character itself must be left "
        "alone here).\n"
        "- `\"query\"`: a dict built from everything after the `?` (empty dict if there is "
        "no `?`, or nothing after it). Split on `&` into `key=value` pairs (a pair with no "
        "`=` is a bare key with an empty-string value). For BOTH the key and the value: "
        "decode `%XX` escapes, AND ALSO decode a literal `+` character to a space (this is "
        "the standard query-string convention — note this is the OPPOSITE of how `+` is "
        "handled in the path above). Every key in the returned dict maps to a LIST of its "
        "decoded values in the order they appeared — even if a key only appears once, its "
        "value is still a one-element list (a query string may legally repeat the same key "
        "more than once, e.g. `?tag=a&tag=b`, and callers need one consistent shape "
        "regardless of how many times a key showed up).\n"
        "- `\"headers\"`: a dict built from `header_lines`. Each line must contain a `:` — "
        "split it at the FIRST `:` into a name and a value, strip leading/trailing "
        "whitespace from the value, and lower-case the name (HTTP header names are "
        "case-insensitive, so `\"Content-Type\"` and `\"content-type\"` are the same header "
        "and the returned dict's keys must always be lower-case). If the SAME header name "
        "(after lower-casing) appears on more than one line, its values must be JOINED with "
        "`\", \"` in the order the lines appeared — not overwritten by the last one. If any "
        "line contains no `:` at all, raise `ValueError`.\n\n"
        "A visible test file is already present at tests/test_request_parse.py — run it "
        "(run_pytest) and keep revising request_parse.py until every test in it passes, "
        "then finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    """Starter files copied into /work before the agent's loop starts. For a visible task
    this includes the real test file (grading still re-injects a pristine canonical copy
    from private/tests/ regardless — see materialize_task.py / run_grade.sh)."""
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "request_parse", "functions": ["parse_request"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one pure parsing function, no independent sub-parts worth decomposing."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write request_parse.py defining parse_request(method, raw_path, "
                    "header_lines) -> dict with keys 'method' (upper-cased), 'path' "
                    "(everything before the first '?' in raw_path, or all of raw_path if "
                    "there's no '?', with %XX escapes decoded but a literal '+' left "
                    "UNCHANGED), 'query' (dict from everything after the first '?', split on "
                    "'&' into key=value pairs — a pair with no '=' means an empty-string "
                    "value — with BOTH key and value %XX-decoded AND '+' decoded to a space "
                    "(opposite of the path's '+' handling); every key maps to a LIST of its "
                    "decoded values in order of appearance, even for a key that only appears "
                    "once), and 'headers' (dict from header_lines: split each line at the "
                    "FIRST ':', lower-case the name, strip whitespace from the value; if the "
                    "same lower-cased name appears on more than one line, JOIN the values "
                    "with ', ' in order rather than overwriting; a line with no ':' at all "
                    "must raise ValueError). Python's urllib.parse module has useful pieces "
                    "(unquote for plain %XX decoding, unquote_plus for %XX-plus-space "
                    "decoding) but you must apply them to the RIGHT pieces — using the same "
                    "decoding rule everywhere will get the path/query '+' distinction wrong. "
                    "Use write_file to create request_parse.py. Then use run_pytest on "
                    "tests/test_request_parse.py. If any test fails, use read_file/"
                    "patch_file to fix request_parse.py and run_pytest again. Once every "
                    "test passes, finish."
                ),
                "expect": (
                    "request_parse.py written; tests/test_request_parse.py fully passes"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm request_parse.py exists and report the pytest pass/fail summary."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["request_parse.py"]},
    }
