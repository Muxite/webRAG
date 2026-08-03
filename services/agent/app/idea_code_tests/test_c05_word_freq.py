"""
codebench task c05 — hard/hidden, tokenization + counting.

Ground truth for ``word_frequency`` (lowercase every word; a "word" is a maximal run of
ASCII letters and digits, everything else — including apostrophes — is a separator and
is discarded) was computed by two independently-written reference implementations
(regex findall on `[A-Za-z0-9]+` vs. manual char-by-char scanning with `str.isalnum()`
gated to `ord(ch) < 128`) that were run and cross-checked to agree exactly before any
value below was written down — see idea_code_test_c05_test.py for the reimplementation
that re-verifies this at test time. Do not hand-edit these literals without re-deriving.

    word_frequency("The quick brown fox jumps over the lazy dog. The dog barks!")
        -> {"the": 3, "quick": 1, "brown": 1, "fox": 1, "jumps": 1, "over": 1,
            "lazy": 1, "dog": 2, "barks": 1}

    word_frequency("Don't stop believing, don't you dare stop.")
        -> {"don": 2, "t": 2, "stop": 2, "believing": 1, "you": 1, "dare": 1}
        (apostrophe is a separator: "Don't" tokenizes to "don", "t")

    word_frequency("Cat cat CAT cat, dog DOG.")
        -> {"cat": 4, "dog": 2}

    word_frequency("One 1 two 2 one 1!")
        -> {"one": 2, "1": 2, "two": 1, "2": 1}
        (digits count as word characters too, per spec)
"""
from __future__ import annotations

CANONICAL_TEST_PATH = "tests/test_word_freq.py"

_TEST_FILE_CONTENT = '''\
from word_freq import word_frequency


def test_repeated_words_and_punctuation():
    text = "The quick brown fox jumps over the lazy dog. The dog barks!"
    expected = {
        "the": 3,
        "quick": 1,
        "brown": 1,
        "fox": 1,
        "jumps": 1,
        "over": 1,
        "lazy": 1,
        "dog": 2,
        "barks": 1,
    }
    assert word_frequency(text) == expected


def test_apostrophe_contraction_splits_into_two_words():
    text = "Don't stop believing, don't you dare stop."
    expected = {
        "don": 2,
        "t": 2,
        "stop": 2,
        "believing": 1,
        "you": 1,
        "dare": 1,
    }
    assert word_frequency(text) == expected


def test_mixed_case_folds_to_lowercase():
    text = "Cat cat CAT cat, dog DOG."
    expected = {"cat": 4, "dog": 2}
    assert word_frequency(text) == expected


def test_digits_count_as_word_characters():
    text = "One 1 two 2 one 1!"
    expected = {"one": 2, "1": 2, "two": 1, "2": 1}
    assert word_frequency(text) == expected


def test_empty_text_returns_empty_dict():
    assert word_frequency("") == {}
'''

# The repeated-words, apostrophe-splitting, case-folding, and digit-handling cases each
# pin down a distinct piece of the tokenization spec that a naive str.split() or
# regex-only-letters implementation would get wrong; the empty-text case is degenerate
# (trivially {} regardless of implementation) and is bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{CANONICAL_TEST_PATH}::test_repeated_words_and_punctuation",
    f"{CANONICAL_TEST_PATH}::test_apostrophe_contraction_splits_into_two_words",
    f"{CANONICAL_TEST_PATH}::test_mixed_case_folds_to_lowercase",
    f"{CANONICAL_TEST_PATH}::test_digits_count_as_word_characters",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c05",
        "title": "word-frequency-report",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `word_freq.py` that defines a function "
        "`word_frequency(text: str) -> dict`.\n\n"
        "`word_frequency(text)` must return a dict mapping each distinct word found in "
        "`text` to how many times it occurs, with every word LOWERCASED before "
        "counting (so `\"Cat\"` and `\"cat\"` are the same word and contribute to the "
        "same count).\n\n"
        "Tokenization must work exactly as follows: a \"word\" is a maximal run of "
        "ASCII letters (a-z, A-Z) and digits (0-9). Every other character — spaces, "
        "punctuation, and in particular the apostrophe character `'` — is a separator "
        "and is simply discarded; it does NOT get treated as part of a word and it does "
        "NOT glue two words together. This means a contraction like `\"don't\"` is NOT "
        "one word: the apostrophe splits it into two separate words, `\"don\"` and "
        "`\"t\"`, each counted on their own.\n\n"
        "Worked example: `word_frequency(\"Cat cat, CAT!\")` must return "
        "`{\"cat\": 3}` — all three occurrences fold to the same lowercase word "
        "despite the differing capitalization and trailing punctuation.\n\n"
        "If `text` contains no words at all (including the empty string), return an "
        "empty dict `{}`.\n\n"
        "No test file is visible for this task. Write word_freq.py implementing "
        "word_frequency exactly as specified above, then use run_pytest to check your "
        "own reasoning about correctness before finishing (there is no pre-existing "
        "test file at tests/ — the grader supplies its own tests after you finish)."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {CANONICAL_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "word_freq", "functions": ["word_frequency"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write word_freq.py implementing word_frequency(text: str) -> dict: "
                    "tokenize text into maximal runs of ASCII letters and digits only "
                    "(use a regex like re.findall(r'[A-Za-z0-9]+', text) or equivalent "
                    "manual scanning), lowercase every token, then count occurrences "
                    "into a dict. Apostrophes and all other non-alphanumeric characters "
                    "are separators and must NOT be treated as part of a word — "
                    "\"don't\" must tokenize to \"don\" and \"t\" as two separate words. "
                    "Use write_file to create it. There is no pre-existing test file — "
                    "write a few of your own sanity-check calls (e.g. via run_pytest "
                    "against a small scratch test you author, or by reasoning through "
                    "the worked example in the task statement, especially the "
                    "apostrophe-splitting rule) to validate your logic, then finish."
                ),
                "expect": "word_freq.py written implementing word_frequency per the spec",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm word_freq.py exists and defines word_frequency.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["word_freq.py"]},
    }
