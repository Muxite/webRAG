"""
codebench task c16 — soft/hidden, open-ended build (in-memory todo-list API).

Same soft-task shape as c15: no single correct implementation, so grading is a light smoke
suite (import + exercise a basic flow) plus a `get_judge_rubric()` for task-specific judging.
The one wrinkle is that a truly open API shape (agent picks class vs. functions, names
whatever it likes) would leave nothing for an offline smoke test to safely import — so the
spec both invites the agent to design its own API AND gives a concrete fallback minimal
surface (module-level add/list_todos/mark_done/remove) to fall back on when it has no strong
preference of its own. The smoke test below targets that fallback surface directly.
"""
from __future__ import annotations

SMOKE_TEST_PATH = "tests/test_todo_api_smoke.py"

_TEST_FILE_CONTENT = '''\
from todo_api import add, list_todos, mark_done, remove


def _ids(todos):
    """Todo entries might be dicts, simple objects, or bare ids depending on how the
    agent modeled a "todo" internally — extract an id from any of those shapes."""
    ids = []
    for t in todos:
        if isinstance(t, dict) and "id" in t:
            ids.append(t["id"])
        elif hasattr(t, "id"):
            ids.append(t.id)
        else:
            ids.append(t)
    return ids


def test_add_returns_id_like_value():
    todo_id = add("buy milk")
    assert todo_id is not None


def test_list_reflects_addition():
    before = len(list_todos())
    todo_id = add("write report")
    after = list_todos()
    assert len(after) == before + 1
    assert todo_id in _ids(after)


def test_mark_done_does_not_crash():
    todo_id = add("water plants")
    mark_done(todo_id)  # should not raise


def test_remove_removes_it():
    todo_id = add("temporary task")
    remove(todo_id)
    assert todo_id not in _ids(list_todos())
'''

# All four are direct smoke checks of the four explicitly-required operations (add, list,
# mark done, remove) named in the spec — none is a corner case, so all four gate the score.
KEYSTONE_TEST_IDS = [
    f"{SMOKE_TEST_PATH}::test_add_returns_id_like_value",
    f"{SMOKE_TEST_PATH}::test_list_reflects_addition",
    f"{SMOKE_TEST_PATH}::test_mark_done_does_not_crash",
    f"{SMOKE_TEST_PATH}::test_remove_removes_it",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c16",
        "title": "build-a-todo-api",
        "category": "soft",
    }


def get_task_statement() -> str:
    return (
        "Build a Python module `todo_api.py` implementing an IN-MEMORY todo-list API "
        "(no real database, no file I/O, no network — state just lives in memory for the "
        "life of the process). It must support:\n\n"
        "- adding a todo (given some text, returns an id for the new todo)\n"
        "- listing all todos\n"
        "- marking a todo done, given its id\n"
        "- removing a todo, given its id\n\n"
        "Design and name the API yourself — a class with methods, or a set of "
        "module-level functions operating on module-level state, are both fine. Pick "
        "whichever shape you'd naturally reach for.\n\n"
        "If you have no strong preference, use exactly this minimal shape so it's easy to "
        "use unambiguously — module-level functions:\n\n"
        "    add(text) -> id          # create a todo, return its id\n"
        "    list_todos() -> list     # return all todos currently stored\n"
        "    mark_done(id)            # mark the todo with this id as done\n"
        "    remove(id)               # delete the todo with this id\n\n"
        "There is no starter file or test file provided — build todo_api.py entirely "
        "from this description. Before finishing, self-check your own work (e.g. with "
        "run_python, or a couple of quick checks run with run_pytest): add a todo, "
        "confirm it shows up when listing, mark it done without it crashing, then remove "
        "it and confirm it's gone from the list."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files — the agent builds todo_api.py from the task
    statement alone (which includes a concrete fallback API shape, see above)."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {SMOKE_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "todo_api",
            "functions": ["add", "list_todos", "mark_done", "remove"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_judge_rubric() -> dict:
    return {
        "criteria": [
            "Supports all four core operations: add a todo (returns an id), list all "
            "todos, mark a todo done by id, and remove a todo by id.",
            "State is genuinely in-memory only — no database, file, or network I/O, per "
            "the spec's explicit constraint.",
            "If the agent chose an API shape other than the suggested fallback "
            "(module-level add/list_todos/mark_done/remove), that shape is still "
            "coherent and discoverable (sensible naming and/or docstrings) rather than "
            "requiring the reader to guess.",
            "mark_done and remove behave sensibly when given an unknown or "
            "already-removed id — a clear error or a harmless no-op is fine; silent data "
            "corruption (e.g. mutating the wrong entry) is not.",
            "The list-all operation accurately reflects prior adds/removes/mark-done "
            "calls (no stale or duplicated entries).",
        ],
        "notes": (
            "This is an open-ended build task with a deliberately loose API contract — "
            "the smoke tests only check the concrete fallback surface named in the spec "
            "(module-level add/list_todos/mark_done/remove), since that's the one thing "
            "guaranteed importable. If the agent designed a different but reasonable API "
            "instead (e.g. a TodoList class) the smoke tests will fail to import and "
            "score 0 on the deterministic side — that's an accepted tradeoff of the open "
            "design choice offered in the spec, so the judge should weigh the rubric "
            "criteria above (does the chosen API actually do the job well) rather than "
            "penalizing the deviation itself."
        ),
    }


def get_compiled_plan() -> dict:
    """Single leaf: no independent sub-parts to decompose across for a module this small.
    The instruction repeats the fallback API names verbatim so a weak executor model has a
    concrete target if it doesn't confidently prefer its own design."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Build todo_api.py: an in-memory todo-list API (no file/db/network "
                    "I/O) supporting add-a-todo (returns an id), list-all-todos, "
                    "mark-a-todo-done-by-id, and remove-a-todo-by-id. Design the API "
                    "shape yourself, or — if unsure — use module-level functions "
                    "add(text) -> id, list_todos() -> list, mark_done(id), remove(id). "
                    "There is no starter test file — use write_file to create "
                    "todo_api.py from the task description. Then self-check your own "
                    "work: use run_python (or write a couple of quick checks and run "
                    "them with run_pytest) to add a todo, confirm it appears when "
                    "listing, mark it done without crashing, then remove it and confirm "
                    "it's gone. If something's wrong, use read_file/patch_file to fix "
                    "todo_api.py and re-check. Once you're satisfied it behaves "
                    "correctly, finish."
                ),
                "expect": (
                    "todo_api.py written; agent's own self-check confirms add/list/"
                    "mark_done/remove all behave sensibly"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm todo_api.py exists and report the agent's own self-check results."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["todo_api.py"]},
    }
