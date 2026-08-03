"""
codebench task c19 — soft/hidden, in-memory key-value store with REAL disk persistence.

Open-ended by design: any reasonable class-or-module-API shape that supports set/get plus
save-to-disk/load-from-disk satisfies the task statement. The task statement gives a
concrete fallback module-level API (same convention as the plan doc's c16 note — an
implementer without a strong API preference has something exact to target) so this
module's SMOKE tests (grading tests, hidden from the agent) have a fixed contract to
import against: ``kv_store.py`` exposing ``set_value(key, value)``,
``get_value(key) -> value or None``, ``save(path)``, ``load(path)``.

Unlike c16 (a plain in-memory store, no disk I/O), this task requires REAL file I/O:
save(path) must write to an actual file on disk and load(path) must read one back — the
smoke tests below prove this with a real round trip through a temp file, not just an
in-process value check.

Smoke tests only (soft task — no single correct implementation to pin down beyond this
minimal contract): does the store hold what you put in it, and does persistence actually
round-trip through disk. Real judging of API quality / error handling / persistence
robustness is get_judge_rubric()'s job, not these tests. Temp files use Python's
``tempfile`` module directly (mkstemp/mkdtemp) rather than pytest's ``tmp_path`` fixture,
since these test files are consumed as flat pytest-discoverable modules by the grading
container and this benchmark's other canonical test files (see c01/c03/c09/c13) never rely
on pytest fixtures either — keeping these self-contained avoids any hidden dependency on
how the grading harness configures pytest.
"""
from __future__ import annotations

CANONICAL_TEST_PATH = "tests/test_kv_store.py"

_TEST_FILE_CONTENT = '''\
import os
import tempfile

from kv_store import get_value, load, save, set_value


def test_set_then_get_returns_the_value():
    set_value("smoke_alpha", 1)
    assert get_value("smoke_alpha") == 1


def test_get_missing_key_returns_none():
    assert get_value("smoke_key_that_was_never_set") is None


def test_save_then_load_round_trips_through_a_real_file():
    set_value("smoke_beta", 42)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save(path)
        assert os.path.getsize(path) > 0, "save(path) did not write anything to disk"
        # Mutate the in-memory value so a passing load() below can only be explained by
        # it actually reading the value back off disk, not by the in-memory state
        # coincidentally already holding the right answer.
        set_value("smoke_beta", -1)
        load(path)
        assert get_value("smoke_beta") == 42
    finally:
        os.unlink(path)
'''

# Both the round-trip and the plain set/get case are the entire smoke contract for this
# task — there is no larger suite to draw a subset from, so every test here is keystone.
KEYSTONE_TEST_IDS = [
    f"{CANONICAL_TEST_PATH}::test_set_then_get_returns_the_value",
    f"{CANONICAL_TEST_PATH}::test_get_missing_key_returns_none",
    f"{CANONICAL_TEST_PATH}::test_save_then_load_round_trips_through_a_real_file",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c19",
        "title": "key-value-store-with-persistence",
        "category": "soft",
    }


def get_task_statement() -> str:
    return (
        "Build a key-value store with real disk persistence.\n\n"
        "Write `kv_store.py` implementing an in-memory key-value store that ALSO "
        "supports saving to, and loading from, a JSON file on disk — this means REAL "
        "file I/O (actually reading and writing a file at a given path), not just an "
        "in-memory simulation of persistence.\n\n"
        "You may design the API however you like (a class, a module singleton, "
        "whatever fits), but if you don't have a strong preference, implement these "
        "module-level functions so any caller has something concrete to target:\n"
        "  - `set_value(key, value)` — store `value` under `key`.\n"
        "  - `get_value(key)` — return the value stored under `key`, or `None` if the "
        "key was never set.\n"
        "  - `save(path)` — write the store's current contents to a real JSON file at "
        "`path`.\n"
        "  - `load(path)` — read a JSON file previously written by `save(path)` and "
        "restore the store's contents from it.\n\n"
        "A save-then-load round trip (through an actual file on disk) must restore "
        "exactly what was saved. There is no visible test file for this task — use "
        "run_python or run_pytest to write and run your own quick checks (set a value, "
        "get it back; save to a temp file, mutate the in-memory value, load the temp "
        "file back, confirm the original value returns) before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {CANONICAL_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "kv_store",
            "functions": ["set_value", "get_value", "save", "load"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_judge_rubric() -> dict:
    return {
        "criteria": [
            "save(path)/load(path) perform REAL file I/O against the given path (read "
            "the code, don't just trust that the smoke test happened to pass) — not an "
            "in-memory stub that ignores the path argument",
            "load(path) actually repopulates the store's contents from the file's data "
            "rather than being a no-op or only partially restoring state",
            "reasonable error handling: e.g. get_value on a missing key doesn't raise, "
            "and load() on a missing or malformed file fails predictably rather than "
            "silently corrupting existing in-memory state",
            "the on-disk format is a real, human-readable JSON file (matching the task "
            "statement), not a pickled or otherwise opaque blob",
            "the API is usable and consistent (if a class was chosen instead of the "
            "suggested module functions, it is still easy to use correctly)",
        ],
        "notes": (
            "The smoke tests only prove a single value round-trips through a single "
            "save/load cycle — they cannot catch a store that silently drops keys, "
            "mishandles types across the JSON boundary, or fakes persistence. Weigh "
            "the code itself, not just whether the smoke tests passed."
        ),
    }


def get_compiled_plan() -> dict:
    """Single leaf: one small module with one cohesive responsibility."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write kv_store.py: an in-memory key-value store that also "
                    "persists to/from a real JSON file on disk. If you don't have a "
                    "strong preference for your own API shape, implement module-level "
                    "functions set_value(key, value), get_value(key) -> value or None, "
                    "save(path) (writes the store to a real JSON file at path), and "
                    "load(path) (reads that file back and restores the store's "
                    "contents from it). Use write_file to create it. Then self-check "
                    "your own work with run_python or run_pytest: set a value and get "
                    "it back; save to a real temp file path, change the in-memory "
                    "value, load the temp file back, and confirm the ORIGINAL saved "
                    "value is what comes back (proving load actually reads from disk "
                    "rather than being a no-op). Once you're confident it works, "
                    "finish."
                ),
                "expect": (
                    "kv_store.py written; set/get works in-memory; save(path)/"
                    "load(path) round-trip a value through a real file on disk"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm kv_store.py exists and report the self-check results (in-memory "
            "set/get, and a real save/load round trip through disk)."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["kv_store.py"]},
    }
