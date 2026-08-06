"""
codebench task c27 — hard/visible, structured text file-format parsing (INI).

File-format coverage gap this fills: c04 (CSV), c07 (fleet-config generation, not really
YAML *parsing*), c10 (markdown table formatting), c12 (JSON schema diff) already exist —
none of them exercise a sectioned key-value config format with comments, duplicate-key
policy, and a real write-then-reparse round trip. This is the first.

Ground truth for ``parse_ini``/``write_ini`` verified by actually RUNNING a reference
implementation (not hand-computed) — see
``/tmp/claude-1000/-home-muk-projects-webRAG/cebc60e0-6c4f-4b3e-9be6-882dd9f08d84/scratchpad/ref/ini_mod.py``
for the throwaway script this was derived from, and idea_code_test_c27_test.py for the
independent second reimplementation that re-verifies every literal below at test time.
Verified behavior:
    parse_ini("[a]\\nx = 1\\nx = 2\\n") -> {"a": {"x": "2"}}                    (last dup wins)
    parse_ini("; c\\n\\n[core]\\n# c2\\n\\nname = widget\\n\\n; c3\\n")
        -> {"core": {"name": "widget"}}                                        (comments/blanks ignored)
    parse_ini("debug = true\\nverbose = false\\n\\n[core]\\nname = widget\\n")
        -> {"": {"debug": "true", "verbose": "false"}, "core": {"name": "widget"}}
    parse_ini("[core]\\nname: widget\\ncount = 5\\n")
        -> {"core": {"name": "widget", "count": "5"}}                          (':' also a separator)
    write_ini({"": {"debug": "true"}, "server": {"host": "localhost", "port": "8080"},
               "empty_section": {}})
        -> "debug = true\\n\\n[server]\\nhost = localhost\\nport = 8080\\n\\n[empty_section]\\n"
    parse_ini(write_ini(cfg)) == cfg for any dict-of-dicts cfg (round trip)
"""
from __future__ import annotations

VISIBLE_TEST_PATH = "tests/test_ini_mod.py"

_TEST_FILE_CONTENT = '''\
from ini_mod import parse_ini, write_ini


def test_basic_single_section():
    text = "[core]\\nname = widget\\n"
    assert parse_ini(text) == {"core": {"name": "widget"}}


def test_duplicate_key_last_value_wins():
    text = "[a]\\nx = 1\\nx = 2\\n"
    assert parse_ini(text) == {"a": {"x": "2"}}


def test_comments_and_blank_lines_ignored():
    text = (
        "; leading comment\\n"
        "\\n"
        "[core]\\n"
        "# comment inside section\\n"
        "\\n"
        "name = widget\\n"
        "\\n"
        "; trailing comment\\n"
    )
    assert parse_ini(text) == {"core": {"name": "widget"}}


def test_top_level_section_before_any_header():
    text = "debug = true\\nverbose = false\\n\\n[core]\\nname = widget\\n"
    assert parse_ini(text) == {
        "": {"debug": "true", "verbose": "false"},
        "core": {"name": "widget"},
    }


def test_colon_separator_supported():
    text = "[core]\\nname: widget\\ncount = 5\\n"
    assert parse_ini(text) == {"core": {"name": "widget", "count": "5"}}


def test_write_ini_exact_format():
    cfg = {
        "": {"debug": "true"},
        "server": {"host": "localhost", "port": "8080"},
        "empty_section": {},
    }
    expected = (
        "debug = true\\n"
        "\\n"
        "[server]\\n"
        "host = localhost\\n"
        "port = 8080\\n"
        "\\n"
        "[empty_section]\\n"
    )
    assert write_ini(cfg) == expected


def test_round_trip_parse_write_parse():
    cfg = {
        "": {"a": "1"},
        "sec1": {"x": "10", "y": "20"},
        "sec2": {"z": "hello world"},
    }
    assert parse_ini(write_ini(cfg)) == cfg
'''

# The duplicate-key policy, comment/blank-line stripping, the top-level (headerless)
# section, exact write_ini serialization, and the write-then-reparse round trip are what
# actually distinguish a real INI implementation from a naive line-splitter, so those five
# gate the score. A trivial single-section parse and the alternate ':' separator are real
# but shallower checks — bonus credit only, same convention as c04's format-only bonus test.
KEYSTONE_TEST_IDS = [
    f"{VISIBLE_TEST_PATH}::test_duplicate_key_last_value_wins",
    f"{VISIBLE_TEST_PATH}::test_comments_and_blank_lines_ignored",
    f"{VISIBLE_TEST_PATH}::test_top_level_section_before_any_header",
    f"{VISIBLE_TEST_PATH}::test_write_ini_exact_format",
    f"{VISIBLE_TEST_PATH}::test_round_trip_parse_write_parse",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c27",
        "title": "ini-config-round-trip",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `ini_mod.py` that defines two functions, "
        "`parse_ini(text: str) -> dict` and `write_ini(config: dict) -> str`, for a "
        "simplified INI config-file format.\n\n"
        "## Format\n\n"
        "- A section header is a line of the form `[section_name]` (after stripping "
        "leading/trailing whitespace from the line). Everything after it, up to the next "
        "section header (or end of text), belongs to that section.\n"
        "- Any key-value lines that appear BEFORE the first section header belong to a "
        "special top-level section, whose name is the empty string `\"\"`.\n"
        "- A key-value line has the form `key = value` OR `key: value`. To parse it, find "
        "whichever of `=` or `:` occurs FIRST in the line, split there, and strip "
        "whitespace from both the key (left of the separator) and the value (right of "
        "it). Do not strip or otherwise treat any `=`/`:` that occurs later in the value.\n"
        "- A line whose first non-whitespace character is `;` or `#` is a comment — ignore "
        "it entirely (it is not a key-value pair). Comments are only recognized as WHOLE "
        "lines; a `;` or `#` appearing after real content on a key-value line is just part "
        "of that line's value (no inline-comment stripping).\n"
        "- Blank lines (empty after stripping) are ignored.\n"
        "- Duplicate keys within the same section: the LAST occurrence in the text wins "
        "(its value overwrites any earlier value for that key in that section).\n"
        "- All values are plain strings — do not coerce types.\n\n"
        "## parse_ini(text) -> dict\n\n"
        "Return a dict mapping section name (str, `\"\"` for the top-level section) to an "
        "inner dict mapping key (str) to value (str), in the format described above. If "
        "the text has no content before the first section header, do NOT include a `\"\"` "
        "entry in the result at all (no empty top-level section).\n\n"
        "## write_ini(config) -> str\n\n"
        "`config` has the same shape as `parse_ini`'s return value: a dict mapping section "
        "name to an inner dict of key/value strings. Serialize it back to INI text, "
        "EXACTLY as follows:\n"
        "- If `config` has a `\"\"` key with a NON-EMPTY inner dict, write its entries "
        "FIRST, one per line as `key = value`, with NO `[...]` header line for it.\n"
        "- Then write every other section (in `config`'s own iteration order, skipping "
        "`\"\"`) as a `[section_name]` header line followed by its `key = value` lines (in "
        "that section's own iteration order) — even if that section's inner dict is empty "
        "(write just the header line, no key lines under it).\n"
        "- Separate each section's block (including the `\"\"` block, if present) from the "
        "next with exactly ONE blank line. There is no blank line before the very first "
        "block and no trailing blank line after the very last line of output.\n"
        "- If `config` has no `\"\"` key, or its inner dict is empty, omit that block "
        "entirely (no header, no stray blank line for it).\n\n"
        "`parse_ini(write_ini(config))` must equal `config` for any well-formed config "
        "dict of this shape (a genuine round trip) — build `write_ini` and `parse_ini` so "
        "that holds, not just so each one looks right in isolation.\n\n"
        "A visible test file is already present at tests/test_ini_mod.py — run it "
        "(run_pytest) and keep revising ini_mod.py until every test in it passes, then "
        "finish."
    )


def get_visibility() -> str:
    return "visible"


def get_sandbox_fixture() -> dict:
    return {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT}


def get_grading_payload() -> dict:
    return {
        "tests": {VISIBLE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "ini_mod", "functions": ["parse_ini", "write_ini"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: two small, tightly-coupled functions in one file, no I/O — same
    single-leaf rationale as c01/c04/c07."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write ini_mod.py implementing parse_ini(text: str) -> dict and "
                    "write_ini(config: dict) -> str for a simplified INI format. "
                    "parse_ini: section headers are `[name]` lines; key-value lines are "
                    "`key = value` or `key: value` (split on whichever of '=' or ':' "
                    "comes first in the line, strip both sides); lines before the first "
                    "header belong to a top-level '' section (omit '' from the result if "
                    "it would be empty); lines starting with ';' or '#' (after stripping) "
                    "and blank lines are ignored; within a section, a later duplicate key "
                    "overwrites an earlier one. write_ini: inverse serialization — the ''"
                    " section (if non-empty) first with no header, one 'key = value' line "
                    "per entry, then every other section as '[name]' plus its "
                    "'key = value' lines (even if empty, header-only), each block "
                    "separated from the next by exactly one blank line, no leading or "
                    "trailing blank line. parse_ini(write_ini(config)) must equal config "
                    "for any well-formed config. Use write_file to create ini_mod.py. "
                    "Then use run_pytest on tests/test_ini_mod.py. If any test fails, use "
                    "read_file/patch_file to fix ini_mod.py and run_pytest again. Once "
                    "every test passes, finish."
                ),
                "expect": "ini_mod.py written; tests/test_ini_mod.py fully passes",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm ini_mod.py exists and report the pytest pass/fail summary.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["ini_mod.py"]},
    }
