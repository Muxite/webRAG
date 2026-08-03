"""
codebench task c14 — hard/hidden, fixed-format log parsing into a count mold.

Ground truth for ``aggregate_by_level`` verified with an independent reference script
(services/agent/app/idea_code_tests, exercised via badmodel-lab scratchpad during authoring —
see idea_code_test_c14_test.py's own reimplementation for the offline re-derivation) — do not
change these without re-running that check:
    mixed levels, varied counts               -> {"INFO": 3, "WARN": 1, "ERROR": 2}
    one level with zero occurrences (WARN)     -> {"INFO": 2, "WARN": 0, "ERROR": 1}
    a malformed line mixed in (silently skipped, not counted, no error) ->
        {"INFO": 1, "WARN": 1, "ERROR": 1}
    lowercase/mixed-case level prefixes do NOT count (exact-case only) ->
        {"INFO": 1, "WARN": 1, "ERROR": 0}

This is a HIDDEN task: no test file is exposed to the agent's sandbox (get_sandbox_fixture()
returns {}) — the canonical test below lives only in private/tests/ and is never mounted into
the agent's /work. The spec in get_task_statement() must therefore spell out every edge case
explicitly, since the agent cannot infer them from a visible test.
"""
from __future__ import annotations

TEST_PATH = "tests/test_log_agg.py"

_TEST_FILE_CONTENT = '''\
from log_agg import aggregate_by_level


def test_mixed_levels_varied_counts():
    lines = [
        "INFO: server started",
        "WARN: low disk space",
        "ERROR: connection refused",
        "INFO: request handled",
        "INFO: request handled again",
        "ERROR: timeout",
    ]
    assert aggregate_by_level(lines) == {"INFO": 3, "WARN": 1, "ERROR": 2}


def test_zero_occurrences_of_one_level_still_present_as_key():
    lines = [
        "INFO: booted",
        "INFO: ready",
        "ERROR: crash",
    ]
    assert aggregate_by_level(lines) == {"INFO": 2, "WARN": 0, "ERROR": 1}


def test_malformed_line_is_silently_skipped():
    lines = [
        "INFO: ok",
        "this line has no level prefix at all",
        "WARN: careful",
        "DEBUG: not a recognized level",
        "ERROR: bad",
        "INFO:missing the space after colon",
    ]
    assert aggregate_by_level(lines) == {"INFO": 1, "WARN": 1, "ERROR": 1}


def test_lowercase_level_prefix_does_not_count():
    lines = [
        "INFO: real one",
        "info: lowercase should not count",
        "Info: mixed case should not count",
        "WARN: real warn",
    ]
    assert aggregate_by_level(lines) == {"INFO": 1, "WARN": 1, "ERROR": 0}
'''

# All four cases are load-bearing for a hidden task: each one confirms a distinct part of the
# contract the agent cannot see (always-present keys, silent-skip on malformed input, and
# exact-case matching) rather than just "does the happy path work" — so, unlike c01/c13, none
# are demoted to bonus credit here.
KEYSTONE_TEST_IDS = [
    f"{TEST_PATH}::test_mixed_levels_varied_counts",
    f"{TEST_PATH}::test_zero_occurrences_of_one_level_still_present_as_key",
    f"{TEST_PATH}::test_malformed_line_is_silently_skipped",
    f"{TEST_PATH}::test_lowercase_level_prefix_does_not_count",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c14",
        "title": "log-line-aggregator",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `log_agg.py` that defines a function "
        "`aggregate_by_level(log_lines: list[str]) -> dict`.\n\n"
        "Each line in `log_lines` is expected to have the fixed format "
        "`\"<LEVEL>: <message>\"`, where `LEVEL` is exactly one of `INFO`, `WARN`, or "
        "`ERROR`, and the delimiter between the level and the message is exactly `\": \"` "
        "(a colon followed by a single space). For example, the line "
        "`\"ERROR: disk full\"` has level `ERROR` and message `disk full`.\n\n"
        "`aggregate_by_level` must return a dict with exactly three keys, `\"INFO\"`, "
        "`\"WARN\"`, and `\"ERROR\"`, mapping to the count of lines with that level. All "
        "three keys must always be present in the returned dict, even if a given level "
        "never appears in `log_lines` (its count is then 0).\n\n"
        "Matching is exact-case only: the level prefix must be uppercase exactly as shown "
        "above. A line like `\"info: booted\"` or `\"Info: booted\"` does NOT count as an "
        "INFO line, because the prefix does not match `\"INFO: \"` exactly.\n\n"
        "Any line that does not start with one of `\"INFO: \"`, `\"WARN: \"`, or "
        "`\"ERROR: \"` (including lines with an unrecognized level, missing the space "
        "after the colon, or no level prefix at all) must be silently skipped: it is not "
        "counted toward any level, and it must NOT raise an error.\n\n"
        "Worked example: given "
        "`[\"INFO: up\", \"WARN: mem high\", \"garbage line\", \"ERROR: down\", "
        "\"INFO: up again\"]`, the function must return "
        "`{\"INFO\": 2, \"WARN\": 1, \"ERROR\": 1}` — the `\"garbage line\"` entry is "
        "skipped entirely.\n\n"
        "There is no visible test file for this task. Implement log_agg.py directly from "
        "this specification, then use run_pytest to sanity-check your own understanding "
        "before finishing (write your own scratch test if you find that useful — it will "
        "not be graded)."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter test file is copied into /work. The agent implements
    log_agg.py purely from get_task_statement()'s spec; the canonical test lives only in
    private/tests/ (see get_grading_payload()) and grading re-runs it in a fresh container
    the agent never had access to."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "log_agg", "functions": ["aggregate_by_level"]},
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
                    "Write log_agg.py implementing aggregate_by_level(log_lines: "
                    "list[str]) -> dict. Each line has the format '<LEVEL>: <message>' "
                    "with LEVEL one of INFO, WARN, ERROR and the delimiter exactly ': ' "
                    "(colon-space). Return {'INFO': n, 'WARN': n, 'ERROR': n} with all "
                    "three keys always present (0 if a level never appears). Matching is "
                    "exact-case only ('info:' does not count). A line that doesn't start "
                    "with one of the three exact prefixes is silently skipped: not "
                    "counted, no error raised. Use write_file to create log_agg.py. There "
                    "is no visible test file — optionally use run_pytest against a scratch "
                    "test you write yourself to sanity-check, then finish."
                ),
                "expect": "log_agg.py written implementing the documented contract",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm log_agg.py exists and defines aggregate_by_level.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["log_agg.py"]},
    }
