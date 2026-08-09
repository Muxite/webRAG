"""
Offline unit tests for the numbered-list-marker fix to the count/band keystone gates — free.

BUG (fixed across test_070/072/073/078/082/083/087/089/090): several keystone validators do
``KEYSTONE_COUNT in _int_values(_primary_text(result))`` (or the band-membership variant
``any(n in KEYSTONE_BAND for n in _int_values(...))``), and ``_int_values`` extracts EVERY plain
integer anywhere in the answer text with zero context filtering. If the model's answer includes a
numbered list, the list's own enumeration markers ("1. Lake A ... 2. Lake B ... 3. ...") can
satisfy the membership check even when the model's actually-asserted count is something else
entirely.

FIX: ``_strip_list_markers`` drops a leading enumeration marker from the start of each line, but
ONLY when the text looks like an actual numbered list (>= 2 such markers present) — a single
leading digit-marker on an otherwise terse answer (e.g. ``"4."`` or ``"4. Lakes exceed the
threshold"``) is far more likely a genuine short answer than list enumeration and must NOT be
stripped. An earlier, unconditional version of this fix was found via adversarial review to break
exactly that terse-answer case; this conditional (>=2 markers) version was verified to fix the
original bug while preserving the terse-answer case.

This file tests:
  * ``_strip_list_markers`` directly, table-driven, including the terse-answer non-regression case;
  * the EXACT reproduced bug scenario against the REAL ``test_072``/``test_078`` keystone gates —
    a synthetic answer that asserts the wrong count in prose ("three") while incidentally numbering
    an unrelated 6-item list 1-6 (so the correct keystone integer 4 appears ONLY as a list marker);
  * a regression guard: a realistic ``_compose_count_threshold``-rendered answer (the real composer
    from ``agent.app.testing.execution_compiled``, fed the real leaves/composition from
    ``test_072.get_compiled_plan()``) still passes the keystone gate after the fix.
"""
import pytest

from agent.app.idea_tests import test_072_tier5_count_with_condition as t072
from agent.app.idea_tests import test_078_tier5_count_with_condition_b as t078
from agent.app.testing.execution_compiled import _compose_count_threshold


# ── _strip_list_markers: table-driven cases ─────────────────────────────────────────────────────

_SIX_LINE_LIST = (
    "1. Lake Alpha\n"
    "2. Lake Beta\n"
    "3. Lake Gamma\n"
    "4. Lake Delta\n"
    "5. Lake Epsilon\n"
    "6. Lake Zeta\n"
)

_SIX_LINE_LIST_STRIPPED = (
    "Lake Alpha\n"
    "Lake Beta\n"
    "Lake Gamma\n"
    "Lake Delta\n"
    "Lake Epsilon\n"
    "Lake Zeta\n"
)

STRIP_CASES = [
    # (label, input_text, expected_output)
    ("six_line_numbered_list_stripped", _SIX_LINE_LIST, _SIX_LINE_LIST_STRIPPED),
    ("bare_digit_dot_unchanged", "4.", "4."),
    ("terse_digit_dot_sentence_unchanged", "4. Lakes exceed the threshold", "4. Lakes exceed the threshold"),
    ("no_marker_present_unchanged", "4 of the 7 lakes exceed the threshold", "4 of the 7 lakes exceed the threshold"),
]


@pytest.mark.parametrize("label,text,expected", STRIP_CASES, ids=[c[0] for c in STRIP_CASES])
def test_strip_list_markers_table(label, text, expected):
    assert t072._strip_list_markers(text) == expected


def test_strip_list_markers_requires_at_least_two_markers():
    """A single leading marker must survive untouched -- only >=2 markers trigger stripping."""
    single = "4. Lakes exceed the threshold"
    assert len(t072._LIST_MARKER_RX.findall(single)) == 1
    assert t072._strip_list_markers(single) == single


def test_strip_list_markers_parenthesised_and_close_paren_forms():
    """'(3) ' and '2) ' marker forms are recognised too, once >=2 are present."""
    text = "(1) Lake A\n2) Lake B\n3. Lake C\n"
    stripped = t072._strip_list_markers(text)
    assert "Lake A" in stripped and "Lake B" in stripped and "Lake C" in stripped
    assert "(1)" not in stripped and "2)" not in stripped and "3." not in stripped


# ── reproduced bug scenario (test_072) ──────────────────────────────────────────────────────────

_OBS = {"visit": {"count": 7}}


def _unrelated_six_item_list_body(prose_count_word: str) -> str:
    """An answer that asserts the wrong count IN PROSE while incidentally numbering an unrelated
    six-item list 1-6 -- so the correct keystone integer (4) never appears anywhere in the text
    EXCEPT as that list's own enumeration marker on line 4."""
    return (
        f"Answer: {prose_count_word} lakes exceed the threshold.\n\n"
        "For reference, here are the seven lakes considered in this task:\n"
        "1. Lake Alpha\n"
        "2. Lake Beta\n"
        "3. Lake Gamma\n"
        "4. Lake Delta\n"
        "5. Lake Epsilon\n"
        "6. Lake Zeta\n"
    )


def test_072_reproduced_bug_scenario_now_correctly_fails():
    """EXACT reproduced bug: the model's real asserted count is THREE (spelled out in prose, not
    the correct answer 4), but it also prints an unrelated numbered list running 1-6, whose own
    marker '4.' would (pre-fix) satisfy 'KEYSTONE_COUNT in _int_values(...)' and wrongly pass.
    Post-fix, ``_strip_list_markers`` removes the list markers (>=2 present) before extraction, so
    no digit '4' survives anywhere in the text and the keystone correctly fails."""
    body = _unrelated_six_item_list_body("three")
    result = {"deliverables": [body], "output": {"final_deliverable": body}}

    # Demonstrate the bug would have fired on the RAW (unstripped) text: the enumeration markers
    # alone contain every digit 1-6, including the keystone value 4.
    assert t072.KEYSTONE_COUNT in t072._int_values(body)

    # The REAL keystone gate and validator, post-fix, must NOT be fooled by those markers.
    assert t072._keystone_ok(result, _OBS) is False
    v = t072.validate_keystone_count(result, _OBS)
    assert v["passed"] is False
    assert v["score"] == 0.0


def test_078_reproduced_bug_scenario_now_correctly_fails():
    """Same reproduced bug shape against test_078's (islands) keystone gate, for the same reason:
    a prose count of 'three' plus an unrelated numbered list 1-6 must not be credited as count=4."""
    body = _unrelated_six_item_list_body("three")
    result = {"deliverables": [body], "output": {"final_deliverable": body}}

    assert t078.KEYSTONE_COUNT in t078._int_values(body)  # raw text still "contains" 4 via markers

    assert t078._keystone_ok(result, _OBS) is False
    v = t078.validate_keystone_count(result, _OBS)
    assert v["passed"] is False
    assert v["score"] == 0.0


def test_072_reproduced_bug_scenario_correct_prose_count_still_passes():
    """Sanity counterpart: if the model's prose count genuinely IS four (not stated as a digit,
    just to keep this test independent of the numeral itself) alongside the same unrelated list,
    the keystone must still be gated on the actual digit '4' appearing outside the list -- i.e.
    this test only asserts the fix doesn't overcorrect into false negatives when '4' legitimately
    appears elsewhere in the text too."""
    body = (
        "Answer: 4 lakes exceed the threshold.\n\n"
        "For reference, here are the seven lakes considered in this task:\n"
        "1. Lake Alpha\n"
        "2. Lake Beta\n"
        "3. Lake Gamma\n"
        "4. Lake Delta\n"
        "5. Lake Epsilon\n"
        "6. Lake Zeta\n"
    )
    result = {"deliverables": [body], "output": {"final_deliverable": body}}
    assert t072._keystone_ok(result, _OBS) is True
    assert t072.validate_keystone_count(result, _OBS)["score"] == 1.0


# ── regression guard: a realistic composed answer still passes ─────────────────────────────────

def test_regression_guard_real_composed_answer_still_passes():
    """A realistic answer in the ACTUAL render shape produced by ``_compose_count_threshold``
    (the real composer, fed the real leaves/composition from test_072's own
    ``get_compiled_plan()``) must still pass the keystone gate after the list-marker fix -- the
    composer's per-item lines are 'Name: value=... (...)', never a bare leading digit marker, so
    stripping must be a no-op here."""
    plan = t072.get_compiled_plan()
    leaves = plan["leaves"]
    composition = plan["composition"]
    results = {
        "matano_depth": "590 m — source: https://en.wikipedia.org/wiki/Lake_Matano",
        "hornindalsvatnet_depth": "514 m — source: https://en.wikipedia.org/wiki/Hornindalsvatnet",
        "quesnel_depth": "511 m — source: https://en.wikipedia.org/wiki/Quesnel_Lake",
        "sarez_depth": "505 m — source: https://en.wikipedia.org/wiki/Sarez_Lake",
        "tinnsja_depth": "460 m — source: https://en.wikipedia.org/wiki/Tinnsj%C3%A5",
        "manapouri_depth": "444 m — source: https://en.wikipedia.org/wiki/Lake_Manapouri",
        "ohrid_depth": "288 m — source: https://en.wikipedia.org/wiki/Lake_Ohrid",
    }

    composed = _compose_count_threshold(leaves, results, composition)
    assert composed is not None, "composer should resolve cleanly for a fully-populated fact set"
    assert composed.startswith("4 of the 7 lake")  # sanity: composer really produced the count

    result = {"deliverables": [composed], "output": {"final_deliverable": composed}}
    obs = {"visit": {"count": 7}}

    assert t072._keystone_ok(result, obs) is True
    v = t072.validate_keystone_count(result, obs)
    assert v["passed"] is True
    assert v["score"] == 1.0
