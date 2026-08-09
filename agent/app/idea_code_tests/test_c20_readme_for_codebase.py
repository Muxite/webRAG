"""
codebench task c20 — soft/hidden, write documentation for a given mini-codebase.

Deliberately a DIFFERENT KIND of deliverable than every other task in this suite: the
output is natural-language documentation (README.md), not more code. There is no
function to call, no behavior to assert on, so this task pushes on the edge of what the
hard-task-shaped pytest grading pipeline can usefully check at all — see the comment on
KEYSTONE_TEST_IDS / get_grading_payload() below. The real signal for this task is
get_judge_rubric(), not the smoke test.

get_sandbox_fixture() hands the agent a tiny hand-written "given codebase" (a 2-function
utility module + a tiny class, ~25 lines combined) to document. These starter files are
intentionally simple and self-explanatory to a human reader, so a judge can straightforwardly
check whether a README actually describes what's really there.
"""
from __future__ import annotations

README_PATH = "README.md"
SMOKE_TEST_PATH = "tests/test_readme_exists.py"

_UTILS_PY_CONTENT = '''\
def slugify(text):
    """Convert text into a lowercase, hyphen-separated slug."""
    return "-".join(text.lower().split())


def clamp(value, low, high):
    """Clamp value to the inclusive [low, high] range."""
    return max(low, min(high, value))
'''

_STACK_PY_CONTENT = '''\
class Stack:
    """A simple last-in-first-out stack."""

    def __init__(self):
        self._items = []

    def push(self, item):
        """Push item onto the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raises IndexError if empty."""
        return self._items.pop()

    def is_empty(self):
        """Return True if the stack has no items."""
        return len(self._items) == 0
'''

# Not a real Python-behavior check at all — there is no function to call for a README
# task, so this exercises the boundary of what this benchmark's pytest-based grading
# pipeline can usefully verify. It confirms only that a README.md was actually produced
# and isn't a one-line stub; it says nothing about accuracy, completeness, or whether the
# usage example is real. That signal comes from get_judge_rubric() below, judged by an
# LLM reading the actual README content against the actual starter files.
_TEST_FILE_CONTENT = '''\
import os

def test_readme_exists_and_is_nontrivial():
    # Submission files land at /grade/ root alongside tests/ (see run_grade.sh: the
    # submission is staged into the grading container's /grade, and this canonical test
    # file is injected at /grade/tests/test_readme_exists.py) — so README.md is expected
    # one directory up from this test file, at the grade root, not nested under tests/.
    readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
    assert os.path.exists(readme_path), "README.md missing from submission"
    assert os.path.getsize(readme_path) > 100, (
        "README.md exists but is suspiciously short to be real documentation"
    )
'''

KEYSTONE_TEST_IDS = [
    f"{SMOKE_TEST_PATH}::test_readme_exists_and_is_nontrivial",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c20",
        "title": "write-a-readme-for-given-codebase",
        "category": "soft",
    }


def get_task_statement() -> str:
    return (
        "Write documentation for the given mini-codebase.\n\n"
        "This working directory already contains two small Python files: `utils.py` "
        "(two standalone utility functions) and `stack.py` (a small Stack class). Read "
        "both, then write `README.md` documenting:\n"
        "  - What this small codebase does, overall.\n"
        "  - Its public functions/classes: what each one is for and what "
        "parameters/behavior it has.\n"
        "  - At least one concrete usage example showing how to actually call/use "
        "them.\n\n"
        "There is no code to write or tests to pass here — the deliverable is "
        "README.md itself. Use read_file on utils.py and stack.py first so the README "
        "accurately reflects what the code really does, not a guess from the file "
        "names. Before finishing, read your own README.md back and double check it "
        "matches the real functions/classes rather than describing something you "
        "assumed or invented."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """The 'given mini-codebase' the agent must document — not a test file (there is
    none to show; this is a hidden task), just the source it has to read and describe."""
    return {
        "utils.py": _UTILS_PY_CONTENT,
        "stack.py": _STACK_PY_CONTENT,
    }


def get_grading_payload() -> dict:
    return {
        "tests": {SMOKE_TEST_PATH: _TEST_FILE_CONTENT},
        # No function/module to call here — the deliverable is a markdown file, not
        # importable code, so "entrypoint" doesn't fit the module/functions convention
        # every other task uses. This shape is purely informational (written verbatim
        # into meta.json; nothing downstream parses it structurally for this task).
        "entrypoint": {"deliverable": README_PATH},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_judge_rubric() -> dict:
    return {
        "criteria": [
            "README.md documents every public function/class actually present in "
            "utils.py and stack.py (slugify, clamp, Stack with push/pop/is_empty) — "
            "nothing invented, nothing real left out",
            "at least one concrete, plausible usage example is present (e.g. showing "
            "slugify/clamp called with sample arguments, or Stack pushed/popped)",
            "the description of each function/class matches what the code actually "
            "does (e.g. clamp's range behavior, Stack's LIFO order, pop()'s behavior "
            "on empty) rather than just restating the name",
            "reasonably organized and readable — not one undifferentiated paragraph, "
            "and not padded with irrelevant boilerplate",
        ],
        "notes": (
            "This task's pytest-based grading only checks that README.md exists and "
            "is non-trivially long (get_grading_payload()'s tests dict) — it cannot "
            "verify accuracy, completeness, or usefulness of the documentation. This "
            "rubric IS the real signal for this task; weigh it accordingly."
        ),
    }


def get_compiled_plan() -> dict:
    """Single leaf: reading two tiny files and writing one doc is one cohesive unit of
    work, not something that decomposes into independent pieces."""
    return {
        "leaves": [
            {
                "id": "document",
                "instruction": (
                    "Read utils.py and stack.py (already present in the working "
                    "directory) using read_file. Then write README.md (write_file) "
                    "documenting: what this small codebase does overall, each public "
                    "function/class it exposes (name, purpose, parameters/behavior), "
                    "and at least one concrete usage example showing how to call/use "
                    "them. Before finishing, read your own README.md back with "
                    "read_file and sanity-check it actually describes the real "
                    "functions/classes present in utils.py and stack.py, not invented "
                    "ones, then finish."
                ),
                "expect": (
                    "README.md written, accurately documenting utils.py's and "
                    "stack.py's real public functions/classes with a usage example"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm README.md exists and report a brief summary of its contents.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": [README_PATH]},
    }
