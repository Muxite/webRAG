"""Tests for `agent.app.answer_numbers`: pure extraction/trivia/appropriateness over an ANSWER
string, no I/O, no model, offline. Adversarial fixtures target the two real certified-but-wrong
cases named in the module docstring: a -30.55 km "height difference" (wrong operand order) and a
dimensionless ratio (2.66) given where a difference was asked."""
from __future__ import annotations

from agent.app.answer_numbers import (
    extract_answer_numbers,
    is_trivial_number,
    mandate_demanded_operation,
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


class TestGluedPrefixGuard:
    """"GRES-2" is a NAME; extracting -2.0 from it poisoned the smoke cell's predicate
    (an unbackable non-trivial number in every deliverable that mentions the station)."""

    def test_hyphenated_name_suffix_is_not_a_negative_number(self):
        values = [n["value"] for n in extract_answer_numbers("GRES-2 Power Station is tall")]
        assert values == []

    def test_superscript_unit_digit_is_not_extracted(self):
        values = [n["value"] for n in extract_answer_numbers("area 8,372 km2 and population")]
        assert values == [8372.0]

    def test_real_negative_numbers_survive(self):
        values = [n["value"] for n in extract_answer_numbers("delta = -30.55 km exactly")]
        assert values == [-30.55]


class TestMandateDemandedOperation:
    """Fixture tests against the REAL `get_task_statement()` of the 12 tier5 shape-derive tasks
    (test_210 .. test_221) -- import-safe pure functions, no model/network involved. Each mandate
    is checked against its expected shape per the corpus's own operation vocabulary: 210/212/213
    are absolute-difference mandates, 211 is a sum, 214-217 are quotient/ratio mandates, and
    218-221 are argmax/comparison mandates over five entities -- none of the four should ever
    resolve to a two-operand operation even though their prose separately uses a quotient- or
    ratio-shaped word per entity ("per km^2", "aspect ratio", "per floor")."""

    def _mandate(self, module_name: str) -> str:
        import importlib

        module = importlib.import_module(f"agent.app.idea_tests.{module_name}")
        return module.get_task_statement()

    def test_210_chimney_height_difference_is_absolute_difference(self):
        result = mandate_demanded_operation(
            self._mandate("test_210_tier5_chimney_height_difference"))
        assert result["operation"] == "difference"
        assert result["absolute"] is True

    def test_211_lake_depth_sum_is_sum(self):
        result = mandate_demanded_operation(self._mandate("test_211_tier5_lake_depth_sum"))
        assert result["operation"] == "sum"
        assert result["absolute"] is False

    def test_212_tunnel_length_difference_is_absolute_difference(self):
        """Also the mandate that names the false-positive trap for a bare sum cue: it warns the
        reader NOT to use "the combined total of all access shafts" as the tunnel's length --
        a bare `\\bcombined\\b`/`\\btotal\\b` cue would have fired here alongside the difference
        cue and forced this mandate to "ambiguous" (`None`)."""
        result = mandate_demanded_operation(
            self._mandate("test_212_tier5_tunnel_length_difference"))
        assert result["operation"] == "difference"
        assert result["absolute"] is True

    def test_213_national_park_area_difference_is_absolute_difference(self):
        result = mandate_demanded_operation(
            self._mandate("test_213_tier5_national_park_area_difference"))
        assert result["operation"] == "difference"
        assert result["absolute"] is True

    def test_214_dam_capacity_ratio_is_quotient(self):
        result = mandate_demanded_operation(self._mandate("test_214_tier5_dam_capacity_ratio"))
        assert result["operation"] == "quotient"
        assert result["absolute"] is False

    def test_215_stadium_cost_per_seat_is_quotient(self):
        result = mandate_demanded_operation(self._mandate("test_215_tier5_stadium_cost_per_seat"))
        assert result["operation"] == "quotient"

    def test_216_rail_average_speed_is_quotient(self):
        result = mandate_demanded_operation(self._mandate("test_216_tier5_rail_average_speed"))
        assert result["operation"] == "quotient"

    def test_217_lake_area_ratio_is_quotient(self):
        result = mandate_demanded_operation(self._mandate("test_217_tier5_lake_area_ratio"))
        assert result["operation"] == "quotient"

    def test_218_river_length_density_argmax_is_none(self):
        result = mandate_demanded_operation(
            self._mandate("test_218_tier5_river_length_density_argmax"))
        assert result["operation"] is None
        assert result["reason"] == "argmax_phrasing"

    def test_219_waterfall_aspect_ratio_argmax_is_none(self):
        """The mandate literally contains "aspect ratio" and "computed ratio value" -- a bare
        `\\bratio\\b` cue fires, but the argmax phrasing ("determine which waterfall has the
        HIGHEST ... ratio") must unconditionally override it."""
        result = mandate_demanded_operation(
            self._mandate("test_219_tier5_waterfall_aspect_ratio_argmax"))
        assert result["operation"] is None
        assert result["reason"] == "argmax_phrasing"

    def test_220_bridge_span_fraction_argmax_is_none(self):
        result = mandate_demanded_operation(
            self._mandate("test_220_tier5_bridge_span_fraction_argmax"))
        assert result["operation"] is None
        assert result["reason"] == "argmax_phrasing"

    def test_221_skyscraper_floor_height_argmax_is_none(self):
        """The mandate contains "in metres per floor" -- a `per` quotient cue fires, but the
        argmax phrasing must still override it to `None`."""
        result = mandate_demanded_operation(
            self._mandate("test_221_tier5_skyscraper_floor_height_argmax"))
        assert result["operation"] is None
        assert result["reason"] == "argmax_phrasing"

    def test_empty_mandate_is_none(self):
        result = mandate_demanded_operation("")
        assert result["operation"] is None
        assert result["reason"] == "no_cue"

    def test_none_mandate_never_raises(self):
        result = mandate_demanded_operation(None)
        assert result["operation"] is None

    def test_two_families_at_once_is_ambiguous(self):
        result = mandate_demanded_operation(
            "Compute the absolute difference and the ratio of the two values.")
        assert result["operation"] is None
        assert result["reason"] == "ambiguous_cue"

    def test_a_ratio_cue_without_absolute_wording_reports_absolute_false(self):
        result = mandate_demanded_operation("What is the ratio of A to B?")
        assert result["operation"] == "quotient"
        assert result["absolute"] is False

    def test_a_bare_difference_without_absolute_wording_reports_absolute_false(self):
        result = mandate_demanded_operation("Compute the difference between A and B.")
        assert result["operation"] == "difference"
        assert result["absolute"] is False
