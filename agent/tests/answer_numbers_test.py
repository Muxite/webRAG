"""Tests for `agent.app.answer_numbers`: pure extraction/trivia/appropriateness over an ANSWER
string, no I/O, no model, offline. Adversarial fixtures target the two real certified-but-wrong
cases named in the module docstring: a -30.55 km "height difference" (wrong operand order) and a
dimensionless ratio (2.66) given where a difference was asked."""
from __future__ import annotations

from agent.app.answer_numbers import (
    extract_answer_numbers,
    is_trivial_number,
    operation_appropriateness,
    strip_urls,
)


class TestStripUrls:
    def test_removes_a_url_substring_leaving_the_rest_of_the_line(self):
        assert strip_urls("38.7 metres, see https://example.com/1991/page") == "38.7 metres, see  "

    def test_no_url_is_a_no_op(self):
        assert strip_urls("38.7 metres") == "38.7 metres"

    def test_empty_and_none_are_safe(self):
        assert strip_urls("") == ""
        assert strip_urls(None) == ""


class TestExtractAnswerNumbers:
    def test_a_unit_bearing_number_keeps_its_unit(self):
        entries = extract_answer_numbers("The tower is 38.7 metres tall.")
        assert {"text": "38.7", "value": 38.7, "unit": "metres"} in entries

    def test_a_bare_number_has_no_unit(self):
        entries = extract_answer_numbers("It was built in 1889.")
        assert {"text": "1889", "value": 1889.0, "unit": ""} in entries

    def test_thousands_separators_are_read_as_one_token(self):
        entries = extract_answer_numbers("Population is 8,336,817.")
        assert {"text": "8,336,817", "value": 8336817.0, "unit": ""} in entries

    def test_a_negative_number_keeps_its_sign_and_unit(self):
        entries = extract_answer_numbers("The height difference is -30.55 km.")
        assert {"text": "-30.55", "value": -30.55, "unit": "km"} in entries

    def test_a_url_digit_run_does_not_masquerade_as_an_answer_number(self):
        entries = extract_answer_numbers(
            "The height is 38.7 metres. See https://example.com/1991/page for the source.")
        values = [e["value"] for e in entries]
        assert 38.7 in values
        assert 1991.0 not in values  # only appears inside the stripped URL path

    def test_a_url_on_the_same_line_as_the_answer_number_does_not_drop_the_number(self):
        """The line-dropping bug ledger_risk_coverage.strip_urls's docstring warns about: URL
        substrings are stripped, never whole lines, so a same-line "answer + citation" survives."""
        entries = extract_answer_numbers("38.7 metres https://example.com/cite")
        assert any(e["value"] == 38.7 for e in entries)

    def test_a_following_ordinary_word_is_never_captured_as_a_unit(self):
        """`extract_unit`'s split alone is whitelist-free and would accept "and" as a unit; this
        module must reject it via the quantity_index whitelist (the trap ledger_tools's own
        module docstring names: "a whitelist is REQUIRED rather than 'whatever word follows the
        number'")."""
        entries = extract_answer_numbers("The count was 2 and population was 8,336,817.")
        two = next(e for e in entries if e["value"] == 2.0)
        assert two["unit"] == ""

    def test_an_area_unit_with_a_digit_suffix_is_captured_whole(self):
        entries = extract_answer_numbers("Area is 8,372 km2.")
        assert any(e["value"] == 8372.0 and e["unit"] == "km2" for e in entries)

    def test_a_computed_answer_line_yields_all_three_numbers_in_order(self):
        entries = extract_answer_numbers("419.7 metres - 381 metres = 38.7 metres")
        assert [e["value"] for e in entries] == [419.7, 381.0, 38.7]
        assert all(e["unit"] == "metres" for e in entries)

    def test_empty_text_yields_no_numbers(self):
        assert extract_answer_numbers("") == []
        assert extract_answer_numbers(None) == []


class TestIsTrivialNumber:
    def test_a_bare_year_is_trivial(self):
        assert is_trivial_number({"value": 1991.0, "unit": ""}) is True

    def test_a_year_like_integer_with_a_unit_is_not_trivial(self):
        """1991 m is a real elevation, not a year -- unit presence overrides the year shape."""
        assert is_trivial_number({"value": 1991.0, "unit": "m"}) is False

    def test_a_small_bare_integer_is_trivial(self):
        assert is_trivial_number({"value": 50.0, "unit": ""}) is True

    def test_a_bare_integer_at_or_above_the_ceiling_is_not_trivial(self):
        assert is_trivial_number({"value": 100.0, "unit": ""}) is False
        assert is_trivial_number({"value": 500.0, "unit": ""}) is False

    def test_a_decimal_is_never_trivial_even_in_year_range(self):
        assert is_trivial_number({"value": 1991.5, "unit": ""}) is False

    def test_a_unit_bearing_number_is_never_trivial(self):
        assert is_trivial_number({"value": 38.7, "unit": "metres"}) is False

    def test_outside_the_year_window_a_large_bare_integer_is_not_trivial(self):
        assert is_trivial_number({"value": 2500.0, "unit": ""}) is False


class TestOperationAppropriateness:
    """Targets the two real certified-but-wrong cases: a -30.55 km "height difference" (wrong
    operand order) and a dimensionless ratio 2.66 given where a difference was asked."""

    def test_negative_difference_for_a_magnitude_question_is_sign_implausible(self):
        result = operation_appropriateness(
            "What is the height difference between Tower A and Tower B?",
            "difference", -30.55, "km")
        assert result["sign_plausible"] is False

    def test_positive_difference_for_a_magnitude_question_is_sign_plausible(self):
        result = operation_appropriateness(
            "What is the height difference between Tower A and Tower B?",
            "difference", 30.55, "km")
        assert result["sign_plausible"] is True

    def test_how_much_taller_phrasing_also_triggers_the_sign_check(self):
        result = operation_appropriateness(
            "How much taller is Tower A than Tower B?", "difference", -12.0, "m")
        assert result["sign_plausible"] is False

    def test_a_ratio_where_a_difference_was_asked_fails_shape_match(self):
        result = operation_appropriateness(
            "What is the height difference between Tower A and Tower B?",
            "quotient", 2.66, "")
        assert result["operation_shape_match"] is False

    def test_ratio_alias_also_fails_shape_match_on_a_difference_mandate(self):
        result = operation_appropriateness(
            "What is the gap between the two rivers' lengths?", "ratio", 2.66, "")
        assert result["operation_shape_match"] is False

    def test_a_difference_where_a_ratio_was_asked_fails_shape_match(self):
        result = operation_appropriateness(
            "How many times taller is Tower A than Tower B?", "difference", 5.0, "")
        assert result["operation_shape_match"] is False

    def test_matching_shape_reports_true(self):
        result = operation_appropriateness(
            "What is the height difference between Tower A and Tower B?",
            "difference", 30.55, "km")
        assert result["operation_shape_match"] is True

        result = operation_appropriateness(
            "How many times taller is Tower A than Tower B?", "quotient", 2.66, "")
        assert result["operation_shape_match"] is True

    def test_no_unambiguous_cue_reports_none_for_both(self):
        result = operation_appropriateness(
            "What is the height of Tower A?", "quotient", 2.66, "")
        assert result["sign_plausible"] is None
        assert result["operation_shape_match"] is None

    def test_a_non_numeric_value_reports_none_for_sign_plausible(self):
        result = operation_appropriateness(
            "What is the height difference?", "difference", "not-a-number", "")
        assert result["sign_plausible"] is None

    def test_an_unrecognized_operation_under_a_magnitude_cue_is_not_penalized(self):
        """An operation this module does not recognize as ratio-shaped is not FALSELY flagged --
        only a KNOWN mismatch (quotient/ratio under a magnitude cue) fires False."""
        result = operation_appropriateness(
            "What is the height difference?", "sum", 30.0, "")
        assert result["operation_shape_match"] is None
