"""
codebench task c17 — soft/hidden, open-ended build (small-subset Markdown -> HTML renderer).

Same soft-task shape as c15/c16: no single correct implementation (exact whitespace/tag
choices are explicitly left open), so grading is a light, substring-based smoke suite plus a
`get_judge_rubric()` for task-specific judging. The spec deliberately scopes the "must support"
set to exactly three features — headings (h1-h3), bold, and paragraphs — so this is NOT a
CommonMark-compliance task.
"""
from __future__ import annotations

SMOKE_TEST_PATH = "tests/test_md_render_smoke.py"

_TEST_FILE_CONTENT = '''\
from md_render import render


def test_h1_heading():
    html = render("# Hello")
    assert "<h1>" in html
    assert "Hello" in html


def test_bold_text():
    html = render("**bold**")
    assert "bold" in html
    assert "<b>" in html or "<strong>" in html


def test_plain_paragraph():
    html = render("This is a plain paragraph.")
    assert "This is a plain paragraph." in html
    assert "<p>" in html
'''

# All three are direct smoke checks of the three explicitly-named MUST-support features
# (heading, bold, paragraph) — none is a corner case, so all three gate the score.
KEYSTONE_TEST_IDS = [
    f"{SMOKE_TEST_PATH}::test_h1_heading",
    f"{SMOKE_TEST_PATH}::test_bold_text",
    f"{SMOKE_TEST_PATH}::test_plain_paragraph",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c17",
        "title": "markdown-to-html-renderer",
        "category": "soft",
    }


def get_task_statement() -> str:
    return (
        "Build a Python module `md_render.py` that defines a function "
        "`render(markdown_text: str) -> str` converting a SMALL SUBSET of Markdown to "
        "HTML. This is explicitly NOT a full CommonMark-compliant renderer — support "
        "exactly these three features, and nothing more is required:\n\n"
        "1. Headings: a line starting with `#`, `##`, or `###` (one to three hashes) "
        "becomes an `<h1>`, `<h2>`, or `<h3>` respectively — e.g. `# Heading` -> "
        "`<h1>Heading</h1>`.\n"
        "2. Bold: `**text**` becomes bold-tagged text — e.g. `**bold**` -> "
        "`<b>bold</b>` (using `<strong>` instead of `<b>` is also fine, your choice).\n"
        "3. Plain paragraphs: any other line(s) of plain text get wrapped in "
        "`<p>...</p>`.\n\n"
        "Exact whitespace/formatting of the output HTML is not important — this is an "
        "open-ended build, not a byte-for-byte spec. Internal parsing approach (regex, "
        "line-by-line scanning, a small state machine, etc.) is entirely up to you.\n\n"
        "There is no starter file or test file provided — build md_render.py entirely "
        "from this description. Before finishing, self-check your own work (e.g. with "
        "run_python, or a couple of quick checks run with run_pytest) against a heading "
        "line, a line with bold text, and a plain paragraph."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files — the agent builds md_render.py from the task
    statement alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {SMOKE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "md_render", "functions": ["render"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_judge_rubric() -> dict:
    return {
        "criteria": [
            "Headings work for all three levels (#, ##, ###), producing <h1>, <h2>, "
            "<h3> respectively — not just a single hardcoded level.",
            "Bold text (**text**) is rendered with some bold-indicating tag (<b> or "
            "<strong> — either is acceptable).",
            "Plain paragraph text is wrapped in a <p>...</p> (or equivalent block-level "
            "wrapper) rather than emitted bare or silently dropped.",
            "Stays scoped to the stated 3-feature subset rather than crashing or "
            "misbehaving on other Markdown syntax (e.g. lists, links, code fences) that "
            "the spec explicitly does not require support for — passing those through "
            "unchanged or wrapping them in a paragraph is fine; erroring out is not.",
            "Output is well-formed enough to be usable as embedded HTML (tags that are "
            "opened are also closed).",
        ],
        "notes": (
            "This is an open-ended build task scoped to a small 3-feature Markdown "
            "subset, not full CommonMark compliance — do not penalize the absence of "
            "features (lists, links, images, code blocks, tables, etc.) the spec never "
            "asked for. Judge whether the three required features actually work, and "
            "whether unsupported input degrades gracefully rather than crashing."
        ),
    }


def get_compiled_plan() -> dict:
    """Single leaf: no independent sub-parts to decompose across for a module this small."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Build md_render.py implementing render(markdown_text: str) -> str: "
                    "convert a small Markdown subset to HTML — `#`/`##`/`###` headings "
                    "to <h1>/<h2>/<h3>, **bold** to a bold tag (<b> or <strong>), and "
                    "plain paragraphs wrapped in <p>...</p>. This is a small 3-feature "
                    "subset, not full CommonMark. There is no starter test file — use "
                    "write_file to create md_render.py from the task description. Then "
                    "self-check your own work: use run_python (or write a couple of "
                    "quick checks and run them with run_pytest) against a heading line, "
                    "a bold-text line, and a plain paragraph. If something's wrong, use "
                    "read_file/patch_file to fix md_render.py and re-check. Once you're "
                    "satisfied it behaves correctly, finish."
                ),
                "expect": (
                    "md_render.py written; agent's own self-check confirms headings, "
                    "bold, and plain paragraphs all render as described"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm md_render.py exists and report the agent's own self-check results."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["md_render.py"]},
    }
