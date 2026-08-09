"""
codebench task c42 — hard/hidden, layered-validation-system, THREE-leaf dependency chain.

Second-generation multi-leaf task (c06 is the only precedent — two leaves). This one goes one
layer deeper: a strict, linear dependency chain of three modules, each genuinely importing and
composing the previous one, not three independently-bundled files:
  * leaf_a builds ``field_validators.py`` — three pure, single-field validator functions
    (``validate_non_empty_string``/``validate_positive_int``/``validate_in_range``), each
    returning ``None`` on success or a short error string otherwise. Independently useful.
  * leaf_b (``depends_on: ["leaf_a"]``) builds ``record_validator.py``'s
    ``validate_record(record) -> list[str]``, which IMPORTS all three field validators and
    composes them over a fixed record shape (name/age/score), collecting EVERY field's error
    rather than short-circuiting on the first one.
  * leaf_c (``depends_on: ["leaf_b"]``) builds ``batch_validator.py``'s
    ``validate_batch(records) -> dict``, which IMPORTS ``validate_record`` and folds it over a
    list of records into a summary report (total/valid/invalid/errors-by-index). leaf_c does NOT
    depend on leaf_a directly — exactly mirroring the real import graph (batch_validator only
    ever touches record_validator; it never imports field_validators itself).

See ``agent/app/testing/execution_compiled_code.py``'s ``_dep_text``/
``UPSTREAM_INCOMPLETE_MARKER`` for exactly what lands in a ``{leaf_x}`` placeholder at runtime
(the model's own ``finish(summary)`` text, plus a deterministic files-written note, or the
incomplete marker if the upstream leaf never finished). Because that text is not fully reliable
even in the happy path, both leaf_b's and leaf_c's instructions independently RESTATE the exact
API contract of the module they depend on, exactly like c06 does, rather than trusting the
upstream leaf's self-report alone.

Ground truth below was derived by actually RUNNING a reference implementation (not worked out by
hand) — see the offline validator test file's own independent reimplementation for the second,
separately-written check. Key verified values:
    validate_non_empty_string("  ") -> "must not be empty"
    validate_non_empty_string(5) -> "must be a string"
    validate_positive_int(True) -> "must be an integer"   (bool excluded even though bool < int)
    validate_positive_int(-1) -> "must be positive"
    validate_in_range(101, 0, 100) -> "must be between 0 and 100"
    validate_in_range(True, 0, 100) -> "must be a number"   (bool excluded here too)
    validate_in_range(float("nan"), 0, 100) -> "must be between 0 and 100"   (NOT None — see below)
    validate_record({}) -> ["missing field: name", "missing field: age", "missing field: score"]
    validate_record({"name": "", "age": -1, "score": 200})
        -> ["name: must not be empty", "age: must be positive", "score: must be between 0 and 100"]
        (all three errors reported — proves the composition does not short-circuit)
    validate_record("not a dict") -> ["record must be a dict"]
    validate_batch([]) -> {"total": 0, "valid": 0, "invalid": 0, "errors": {}}
    validate_batch(<7 mixed records: 1 valid, 1 empty-name, 1 non-positive-age,
                    1 out-of-range-score, 1 missing-score, 1 NaN-score, 1 bool-score>)
        -> {"total": 7, "valid": 1, "invalid": 6, "errors": {
                1: ["name: must not be empty"], 2: ["age: must be positive"],
                3: ["score: must be between 0 and 100"], 4: ["missing field: score"],
                5: ["score: must be between 0 and 100"], 6: ["score: must be a number"]}}

Hardening note (this is the second generation of this task — the first version was live-verified
to score a perfect 1.0/1.0 via aider, so the composition-level canonical checks below were
sharpened): a NaN value is technically a ``float`` (``isinstance(float("nan"), float)`` is
``True``), but it satisfies NEITHER bound of a range check, because every comparison against NaN
in Python — ``<``, ``>``, ``<=``, ``>=``, ``==`` — evaluates to ``False``. A bounds check written
as the intuitive ``if value < lo or value > hi: return error`` therefore silently treats NaN as
"in range" (neither branch fires), while the logically-equivalent-looking ``if not (lo <= value
<= hi): return error`` correctly rejects it. This is a genuine, well-known Python floating-point
gotcha, not an invented trick — and because ``validate_record``/``validate_batch`` do nothing NaN
-specific themselves (they just call ``validate_in_range`` and check its return value), a wrong
answer at the ``field_validators.py`` layer silently propagates all the way through record- and
batch-level composition, and is only ever CAUGHT by the private test at the final, batch-level
assertion — exactly the kind of cross-leaf interaction this task is meant to exercise. The
bool-excluded-from-range-check requirement was already stated in the spec from the first
generation but was previously untested at the composition level; it is now folded into the same
hardened batch scenario (a record can also fail because its score is a bool, which is a *number*
under the hood but must still be rejected).

Hardening note (third generation, 2026-08-07): round-2 live calibration showed badmodel
(qwen2.5:14b) now acing this task (1.0/1.0) after the NaN/bool hardening above, while Aider sat
at 0.8 — the opposite gap from every other task in this batch, so THIS task specifically needed
one more trap to close badmodel's remaining gap without further punishing Aider (whose own
round-2 gap is elsewhere). Live mutation-testing a battery of plausible near-misses against the
round-2 canonical suite (see idea_code_test_c42_test.py) found one that survives every existing
case unchanged: ``validate_positive_int`` is specified as accepting an ``int`` (excluding
``bool``) — but a plausible implementation that writes its bool-exclusion check correctly and
then, by habit (the exact same numeric-type-check shape ``validate_in_range`` legitimately
needs), loosens the TYPE check to ``isinstance(value, (int, float))`` instead of
``isinstance(value, int)`` alone, silently starts accepting FLOATS as valid ages — e.g.
``validate_positive_int(25.0)`` wrongly returns ``None`` instead of ``"must be an integer"``.
Nothing in the existing suite ever calls a field validator, record, or batch with a float `age`,
so this passes every unit-level AND composition-level check unchanged. Exactly like the NaN
case, this is a silent field-layer defect that propagates unchanged through record_validator and
batch_validator (neither layer does its own type-checking) and is only ever caught by an
assertion that actually supplies a float age at the record-composition level.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_batch_validator.py"

_TEST_FILE_CONTENT = '''\
from field_validators import validate_non_empty_string, validate_positive_int, validate_in_range
from record_validator import validate_record
from batch_validator import validate_batch


# --- leaf_a: field_validators.py -----------------------------------------------------------

def test_validate_non_empty_string_rejects_blank_and_wrong_type():
    assert validate_non_empty_string("   ") == "must not be empty"
    assert validate_non_empty_string(5) == "must be a string"


def test_validate_non_empty_string_accepts_good_value():
    assert validate_non_empty_string("Alice") is None


def test_validate_positive_int_rejects_non_positive_and_bool():
    assert validate_positive_int(0) == "must be positive"
    assert validate_positive_int(-1) == "must be positive"
    assert validate_positive_int(True) == "must be an integer"


def test_validate_positive_int_accepts_good_value():
    assert validate_positive_int(3) is None


def test_validate_positive_int_rejects_float_even_when_whole_and_positive():
    # 25.0 LOOKS like a valid positive integer value, but it is a float, not an int -- the spec
    # requires the type itself to be int, not merely "numerically a positive whole number".
    assert validate_positive_int(25.0) == "must be an integer"
    assert validate_positive_int(-2.5) == "must be an integer"


def test_validate_in_range_rejects_out_of_bounds_and_wrong_type():
    assert validate_in_range(-1, 0, 100) == "must be between 0 and 100"
    assert validate_in_range(101, 0, 100) == "must be between 0 and 100"
    assert validate_in_range("x", 0, 100) == "must be a number"
    assert validate_in_range(True, 0, 100) == "must be a number"


def test_validate_in_range_rejects_nan():
    # float("nan") passes isinstance(value, float), but every comparison against it is False in
    # Python -- a naive "value < lo or value > hi" check silently misses it.
    assert validate_in_range(float("nan"), 0, 100) == "must be between 0 and 100"


def test_validate_in_range_accepts_good_value():
    assert validate_in_range(50, 0, 100) is None


# --- leaf_b: record_validator.py -----------------------------------------------------------

def test_record_validator_valid_record():
    assert validate_record({"name": "Eve", "age": 1, "score": 0}) == []
    assert validate_record({"name": "Eve", "age": 1, "score": 100}) == []


def test_record_validator_reports_missing_fields():
    assert validate_record({}) == [
        "missing field: name", "missing field: age", "missing field: score",
    ]


def test_record_validator_collects_all_three_field_errors():
    # every field is wrong at once -- proves the composition does NOT short-circuit
    errors = validate_record({"name": "", "age": -1, "score": 200})
    assert errors == [
        "name: must not be empty",
        "age: must be positive",
        "score: must be between 0 and 100",
    ]


def test_record_validator_rejects_non_dict():
    assert validate_record("not a dict") == ["record must be a dict"]


def test_record_validator_reports_nan_and_bool_score():
    # score=NaN is technically a float but out of range; score=True is technically a number but
    # a bool -- both must be rejected, exercising validate_in_range's edge cases through the
    # record-level composition (not just directly).
    assert validate_record({"name": "X", "age": 1, "score": float("nan")}) == [
        "score: must be between 0 and 100",
    ]
    assert validate_record({"name": "X", "age": 1, "score": True}) == [
        "score: must be a number",
    ]


def test_record_validator_rejects_float_age_despite_looking_numerically_valid():
    # age=25.0 is numerically a positive whole number, but it is a float, not an int -- a
    # field_validators.py that accidentally loosens validate_positive_int's type check to
    # accept any (int, float) -- the same shape validate_in_range legitimately needs -- passes
    # this silently through as valid. Nothing else in this suite ever supplies a float age.
    assert validate_record({"name": "X", "age": 25.0, "score": 50}) == [
        "age: must be an integer",
    ]


# --- leaf_c: batch_validator.py ------------------------------------------------------------

def test_batch_validator_empty_list():
    assert validate_batch([]) == {"total": 0, "valid": 0, "invalid": 0, "errors": {}}


def test_batch_validator_all_valid():
    records = [{"name": "A", "age": 1, "score": 1}, {"name": "B", "age": 2, "score": 2}]
    assert validate_batch(records) == {"total": 2, "valid": 2, "invalid": 0, "errors": {}}


def test_batch_validator_mixed_report():
    records = [
        {"name": "Alice", "age": 30, "score": 85},
        {"name": "", "age": 25, "score": 90},
        {"name": "Bob", "age": -5, "score": 50},
        {"name": "Carol", "age": 40, "score": 150},
        {"name": "Dan", "age": 22},
        {"name": "Eve", "age": 18, "score": float("nan")},
        {"name": "Frank", "age": 50, "score": True},
    ]
    report = validate_batch(records)
    assert report == {
        "total": 7,
        "valid": 1,
        "invalid": 6,
        "errors": {
            1: ["name: must not be empty"],
            2: ["age: must be positive"],
            3: ["score: must be between 0 and 100"],
            4: ["missing field: score"],
            5: ["score: must be between 0 and 100"],
            6: ["score: must be a number"],
        },
    }
'''

# Individual field-validator sanity checks (including the NaN/bool-range and float-age unit
# checks), the record-level NaN/bool bonus check, and the fully-degenerate empty-batch case are
# all supporting/bonus credit: a broken record- or batch-level composition could still pass
# every one of them. The five that gate the score exercise the actual COMPOSITION at each layer:
# not short-circuiting on multiple simultaneous field errors, missing-field detection, the full
# mixed-batch report, and (2026-08-07) the float-age propagation case. That mixed-batch report is
# deliberately harder than a first glance suggests -- two of its seven records (NaN score, bool
# score) only get classified correctly end-to-end if validate_in_range's edge-case handling (see
# the NaN/bool-exclusion note above) is threaded correctly through record_validator's per-field
# composition and batch_validator's index/error folding. A validate_in_range that's subtly wrong
# on just NaN or just bool can still look "basically right" against ad hoc manual testing, yet
# this single assertion catches it. The float-age keystone is the SAME shape of trap applied to
# validate_positive_int instead: nothing else in this suite ever supplies a float age, so a
# field_validators.py whose validate_positive_int accidentally accepts floats (the same
# (int, float) type-check shape validate_in_range legitimately needs) passes every other test
# unchanged and is only caught here.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_record_validator_reports_missing_fields",
    f"{_TEST_FILE_PATH}::test_record_validator_collects_all_three_field_errors",
    f"{_TEST_FILE_PATH}::test_batch_validator_all_valid",
    f"{_TEST_FILE_PATH}::test_batch_validator_mixed_report",
    f"{_TEST_FILE_PATH}::test_record_validator_rejects_float_age_despite_looking_numerically_valid",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c42",
        "title": "layered-field-record-batch-validation",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You are building a small three-layer validation library, split across three files: "
        "field-level validators, a record validator that composes them, and a batch validator "
        "that composes the record validator into a summary report.\n\n"
        "## Part 1 — field_validators.py\n\n"
        "Write `field_validators.py` defining exactly three functions, each returning `None` "
        "when the value is valid or a short error-message string otherwise:\n"
        "- `validate_non_empty_string(value)` — `None` if `value` is a string that is not empty "
        "or all-whitespace; `\"must be a string\"` if `value` isn't a string at all; "
        "`\"must not be empty\"` if it is a string but empty/whitespace-only.\n"
        "- `validate_positive_int(value)` — `None` if `value` is an `int` (booleans do NOT "
        "count, even though `bool` is a subclass of `int` in Python) strictly greater than 0; "
        "`\"must be an integer\"` if it isn't an int (or is a bool); `\"must be positive\"` if "
        "it's an int `<= 0`.\n"
        "- `validate_in_range(value, lo, hi)` — `None` if `value` is an `int`/`float` (not "
        "bool) with `lo <= value <= hi`; `\"must be a number\"` if it isn't a number (or is a "
        "bool); otherwise exactly `f\"must be between {lo} and {hi}\"`. This must also handle "
        "`float('nan')` correctly: it passes `isinstance(value, float)`, but it does not "
        "actually satisfy `lo <= value <= hi` — treat it as out of range (the same "
        "`f\"must be between {lo} and {hi}\"` message), never `None`. Be careful here: in "
        "Python, every comparison against NaN (`<`, `>`, `<=`, `>=`, `==`) evaluates to "
        "`False`, so a bounds check written as `if value < lo or value > hi: ...` silently "
        "treats NaN as if it were in range — test this case explicitly with run_python rather "
        "than assuming your check handles it.\n\n"
        "## Part 2 — record_validator.py\n\n"
        "Write `record_validator.py` that does `from field_validators import "
        "validate_non_empty_string, validate_positive_int, validate_in_range` and defines "
        "`validate_record(record) -> list[str]`. A valid record is a dict with fields `\"name\"` "
        "(checked with `validate_non_empty_string`), `\"age\"` (checked with "
        "`validate_positive_int`), and `\"score\"` (checked with `validate_in_range(value, 0, "
        "100)`).\n\n"
        "If `record` is not a dict, return `[\"record must be a dict\"]` immediately. Otherwise, "
        "for each of `name`, `age`, `score` (in that order) missing from `record`, append "
        "`f\"missing field: {field}\"`. For each present field, run its validator; if it returns "
        "a non-`None` error, append `f\"{field}: {error}\"`. Return the accumulated list — do "
        "NOT stop at the first problem, a record can have more than one field wrong at once. "
        "An empty list means the record is fully valid.\n\n"
        "## Part 3 — batch_validator.py\n\n"
        "Write `batch_validator.py` that does `from record_validator import validate_record` "
        "and defines `validate_batch(records: list) -> dict`, returning EXACTLY these four "
        "keys: `\"total\"` (int, length of `records`), `\"valid\"` (int, how many records "
        "`validate_record` returned `[]` for), `\"invalid\"` (int, `total - valid`), and "
        "`\"errors\"` (dict mapping each INVALID record's 0-based index in `records` to the "
        "list `validate_record` returned for it — valid records are not keys at all).\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check all three "
        "modules yourself (call each field validator on a couple of valid/invalid values, "
        "including `float('nan')` and `True` passed to `validate_in_range`; call "
        "`validate_record` on a fully valid record, one missing every field, and one where "
        "every present field is wrong; call `validate_batch` on an empty list and a small mixed "
        "list) and confirm the outputs match this spec exactly before finishing."
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
                {"module": "field_validators",
                 "functions": ["validate_non_empty_string", "validate_positive_int",
                               "validate_in_range"]},
                {"module": "record_validator", "functions": ["validate_record"]},
                {"module": "batch_validator", "functions": ["validate_batch"]},
            ]
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Three leaves, a strict linear chain (leaf_a -> leaf_b -> leaf_c) with REAL structural
    dependencies matching the actual import graph: leaf_b imports leaf_a's module, leaf_c imports
    leaf_b's module (NOT leaf_a's — batch_validator never touches field_validators directly,
    exactly like a real layered codebase). Each dependent leaf both templates its upstream
    leaf's actual outcome via ``{leaf_x}`` and independently restates the API it depends on, since
    a weak model's own summary of what it built is not fully reliable even when it finished
    honestly — same pattern as c06.
    """
    return {
        "leaves": [
            {
                "id": "leaf_a",
                "instruction": (
                    "Write field_validators.py, a small module of pure field-level validator "
                    "functions, for later use by a record-level validator. Expose EXACTLY these "
                    "three functions, each returning None on success or a short error string "
                    "otherwise:\n"
                    "- validate_non_empty_string(value): None if value is a non-empty, "
                    "non-whitespace-only string; \"must be a string\" if value is not a str at "
                    "all; \"must not be empty\" if it is a str but empty/whitespace-only.\n"
                    "- validate_positive_int(value): None if value is an int (a bool does NOT "
                    "count, even though bool is a subclass of int in Python) strictly greater "
                    "than 0; \"must be an integer\" if it is not an int or is a bool; \"must be "
                    "positive\" if it is an int <= 0.\n"
                    "- validate_in_range(value, lo, hi): None if value is an int or float (not "
                    "bool) with lo <= value <= hi; \"must be a number\" if it is not a number or "
                    "is a bool; otherwise exactly f\"must be between {lo} and {hi}\". This must "
                    "also correctly handle float('nan'): it passes isinstance(value, float), but "
                    "it does not satisfy lo <= value <= hi -- treat it as out of range (the same "
                    "f\"must be between {lo} and {hi}\" message), never None. Be careful: every "
                    "comparison against NaN (<, >, <=, >=, ==) evaluates to False in Python, so a "
                    "check written as 'if value < lo or value > hi' silently treats NaN as if it "
                    "were in range -- test this explicitly with run_python rather than assuming.\n"
                    "This module must be genuinely correct and usable on its own — a later step "
                    "imports all three functions with `from field_validators import "
                    "validate_non_empty_string, validate_positive_int, validate_in_range` and "
                    "calls them exactly as described, so names and returned strings must match "
                    "EXACTLY. Use write_file to create it, then use run_python to call each "
                    "function on a couple of valid and invalid values (including float('nan') "
                    "and True passed to validate_in_range) and print the results to confirm they "
                    "match this spec. Then finish, and in your summary explicitly name the three "
                    "functions."
                ),
                "expect": "field_validators.py written, defining validate_non_empty_string/"
                          "validate_positive_int/validate_in_range exactly as specified "
                          "(including correct NaN and bool handling in validate_in_range), "
                          "sanity-checked with run_python",
                "depends_on": [],
            },
            {
                "id": "leaf_b",
                "instruction": (
                    "Leaf A already wrote field_validators.py, a small module of field-level "
                    "validator functions. Leaf A reported: {leaf_a}\n\n"
                    "Regardless of exactly what that summary said, field_validators.py is "
                    "expected to define three functions, each returning None when the value is "
                    "valid or a short error string otherwise: validate_non_empty_string(value) "
                    "(None if a non-empty/non-whitespace str, else \"must be a string\" or "
                    "\"must not be empty\"), validate_positive_int(value) (None if a non-bool "
                    "int > 0, else \"must be an integer\" or \"must be positive\"), and "
                    "validate_in_range(value, lo, hi) (None if a non-bool int/float within "
                    "[lo, hi], else \"must be a number\" or f\"must be between {lo} and "
                    "{hi}\" -- including correctly rejecting float('nan') as out of range, "
                    "since NaN passes the float type check but fails every bound comparison). "
                    "If field_validators.py does not actually expose that API, use read_file to "
                    "inspect it first and adapt your usage accordingly — do not guess blindly.\n\n"
                    "Now write record_validator.py that does `from field_validators import "
                    "validate_non_empty_string, validate_positive_int, validate_in_range` and "
                    "implements validate_record(record) -> list[str]. A valid record is a dict "
                    "with exactly three fields: \"name\" (checked with "
                    "validate_non_empty_string), \"age\" (checked with validate_positive_int), "
                    "and \"score\" (checked with validate_in_range(value, 0, 100)).\n\n"
                    "validate_record(record) must: (1) if record is not a dict, return the "
                    "single-element list [\"record must be a dict\"] immediately; (2) "
                    "otherwise, for each of \"name\", \"age\", \"score\" (in that order) MISSING "
                    "from record, append f\"missing field: {field}\" to the result list; (3) "
                    "for each of \"name\", \"age\", \"score\" that IS present, run the matching "
                    "field validator on its value, and if it returns a non-None error string, "
                    "append f\"{field}: {error}\" to the result list; (4) return the "
                    "accumulated list. Do NOT stop at the first error — check and report EVERY "
                    "field's problem, since a record can have more than one thing wrong with it "
                    "at once. An empty list means the record is fully valid.\n\n"
                    "Use write_file to create record_validator.py, then use run_python to call "
                    "validate_record on a fully valid record, a record missing all three "
                    "fields, a record where all three present fields are invalid, and a record "
                    "whose score is an edge-case numeric value (e.g. float('nan') or True -- "
                    "both must be treated as invalid, per validate_in_range's contract), and "
                    "print the results to confirm they match this spec exactly. Fix issues with "
                    "patch_file/run_python until you're confident, then finish."
                ),
                "expect": "record_validator.py written, importing field_validators and "
                          "implementing validate_record(record) -> list[str] that composes all "
                          "three field validators without short-circuiting",
                "depends_on": ["leaf_a"],
            },
            {
                "id": "leaf_c",
                "instruction": (
                    "Leaf B already wrote record_validator.py, importing field_validators and "
                    "implementing validate_record(record) -> list[str] (empty list = fully "
                    "valid; otherwise a list of \"field: message\" / \"missing field: field\" "
                    "strings, one per problem, never short-circuiting). Leaf B reported: "
                    "{leaf_b}\n\n"
                    "Regardless of exactly what that summary said, if record_validator.py does "
                    "not actually expose validate_record(record) -> list[str] with that "
                    "behavior, use read_file to inspect it first and adapt your usage "
                    "accordingly — do not guess blindly.\n\n"
                    "Now write batch_validator.py that does `from record_validator import "
                    "validate_record` and implements validate_batch(records: list) -> dict. "
                    "Given a list of records (each meant to be a dict, validated the same way "
                    "validate_record validates one), it must return a summary dict with EXACTLY "
                    "these four keys:\n"
                    "- \"total\": int, the number of records in the input list.\n"
                    "- \"valid\": int, how many records validate_record returned an empty list "
                    "for.\n"
                    "- \"invalid\": int, how many records validate_record returned a non-empty "
                    "list for (total - valid).\n"
                    "- \"errors\": dict mapping each INVALID record's integer index in the "
                    "input list (0-based) to the list of error strings validate_record returned "
                    "for it. Valid records do NOT appear as keys; the dict has exactly "
                    "`invalid` keys.\n\n"
                    "Use write_file to create batch_validator.py, then use run_python to call "
                    "validate_batch on an empty list, a list of all-valid records, and a mixed "
                    "list containing both valid and invalid records (including at least one "
                    "record broken in more than one field, at least one record missing a field "
                    "entirely, and at least one record whose score is an edge-case numeric value "
                    "like float('nan') or True), and print the results to confirm the \"total\"/"
                    "\"valid\"/\"invalid\"/\"errors\" shape matches this spec exactly for every "
                    "one of those records. Fix issues with patch_file/run_python until "
                    "confident, then finish."
                ),
                "expect": "batch_validator.py written, importing record_validator and "
                          "implementing validate_batch(records) -> {total, valid, invalid, "
                          "errors} exactly as specified",
                "depends_on": ["leaf_b"],
            },
        ],
        "aggregation": "Confirm field_validators.py, record_validator.py, and "
                        "batch_validator.py all exist and report the sanity-check results from "
                        "each leaf.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files",
                         "files": ["field_validators.py", "record_validator.py",
                                   "batch_validator.py"]},
    }
