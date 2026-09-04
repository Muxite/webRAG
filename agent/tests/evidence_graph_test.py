"""Tests for the evidence derivation graph (W1: the SOURCE half; layer 4: recomputed arithmetic)."""

import pytest

import agent.app.testing.evidence_graph as eg

from agent.app.idea_policies.contract_satisfaction import StepContract
from agent.app.testing.evidence_graph import (
    KIND_DERIVED,
    KIND_SOURCE,
    VALUE_FAIL_ABSENT,
    VALUE_FAIL_AMBIGUOUS,
    VALUE_FAIL_EMPTY,
    VALUE_FAIL_JUNK,
    VALUE_FAIL_NO_PAGE,
    EvidenceGraph,
    _candidates,
    extract_unit,
    is_junk_value,
    is_unit_bearing,
    normalize_for_match,
    numeric_value,
    reverify_graph,
    source_node_id,
    value_shape,
    verify_value,
    verify_value_against_stored_page,
)

PAGE = "The Eiffel Tower is 1,991 metres from the Seine and 330 m tall."


def _graph() -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.add_page("p1", "https://example.org/eiffel", PAGE)
    return graph


class TestVerifyValue:
    def test_exact_substring_verifies_with_raw_offsets(self):
        match = verify_value(PAGE, "1,991 metres")
        assert match.verified is True
        assert PAGE[match.start:match.end] == "1,991 metres"

    def test_absent_value_is_false_and_absent(self):
        match = verify_value(PAGE, "1,992 metres")
        assert match.verified is False
        assert match.fail_reason == VALUE_FAIL_ABSENT

    def test_missing_page_is_unverifiable_never_false(self):
        match = verify_value("", "1,991 metres")
        assert match.verified is None
        assert match.fail_reason == VALUE_FAIL_NO_PAGE

    def test_empty_value_is_unverifiable_never_false(self):
        match = verify_value(PAGE, "   ")
        assert match.verified is None
        assert match.fail_reason == VALUE_FAIL_EMPTY

    @pytest.mark.parametrize("value", ["1991 metres", "1,991 Metres", "1991metres"])
    def test_light_normalization_cases(self, value):
        match = verify_value(PAGE, value)
        assert match.verified is True
        assert PAGE[match.start:match.end] == "1,991 metres"

    def test_unit_spacing_value_matches_spaced_page(self):
        match = verify_value("Height: 330m tall", "330 m")
        assert match.verified is True

    def test_nfkc_fullwidth_digits_match(self):
        match = verify_value("Population １２３", "123")
        assert match.verified is True

    def test_dash_unification(self):
        match = verify_value("open 1889–1890 season", "1889-1890")
        assert match.verified is True

    def test_digit_tokens_scattered_on_page_do_not_verify(self):
        # The REJECTED relaxation: "1991" and "metres" both appear, apart. Not a match.
        match = verify_value("built in 1991; height given in metres", "1991 metres")
        assert match.verified is False

    def test_paraphrase_never_verifies(self):
        match = verify_value(PAGE, "roughly two thousand metres")
        assert match.verified is False
        assert match.fail_reason == VALUE_FAIL_ABSENT


class TestJunkValues:
    @pytest.mark.parametrize("value", [
        "https://example.org/eiffel",
        "http://example.org/a?b=c",
        "['https://a.example', 'https://b.example']",
        "[]",
    ])
    def test_junk_shapes_are_rejected_unadjudicated(self, value):
        assert is_junk_value(value) is True
        match = verify_value(f"see {value} for details", value)
        assert match.verified is None
        assert match.fail_reason == VALUE_FAIL_JUNK

    @pytest.mark.parametrize("value", ["330 m", "Gustave Eiffel", "1,991 metres"])
    def test_real_values_are_not_junk(self, value):
        assert is_junk_value(value) is False


class TestNormalizeForMatch:
    def test_collapses_whitespace_and_case(self):
        assert normalize_for_match("  A\n B  ") == "a b"

    def test_strips_digit_group_separators(self):
        assert normalize_for_match("1,991") == "1991"


class TestSourceAdmission:
    def test_present_value_admits_source_node(self):
        graph = _graph()
        node = graph.add_source("p1", "1991 metres", quote="is 1,991 metres from")
        assert node is not None
        assert node.kind == KIND_SOURCE
        assert node.page_id == "p1"
        assert node.source_url == "https://example.org/eiffel"
        assert node.verified is True
        assert PAGE[node.start:node.end] == "1,991 metres"
        assert node.quote_verified is True
        assert graph.nodes()[0].id == node.id

    def test_absent_value_is_rejected_and_recorded(self):
        graph = _graph()
        assert graph.add_source("p1", "2,991 metres") is None
        assert graph.nodes() == []
        assert graph.rejections[0]["fail_reason"] == VALUE_FAIL_ABSENT
        assert graph.rejections[0]["verified"] is False

    def test_junk_value_is_rejected_even_when_present_on_page(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "see https://example.org/eiffel here")
        assert graph.add_source("p1", "https://example.org/eiffel") is None
        assert graph.rejections[0]["fail_reason"] == VALUE_FAIL_JUNK
        assert graph.rejections[0]["verified"] is None

    def test_unknown_page_is_unverifiable_never_failed(self):
        graph = _graph()
        assert graph.add_source("p9", "1991 metres") is None
        assert graph.rejections[0]["verified"] is None
        assert graph.rejections[0]["fail_reason"] == VALUE_FAIL_NO_PAGE

    def test_miss_on_truncated_page_is_unverifiable(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", PAGE, max_chars=10)
        assert graph.add_source("p1", "330 m") is None
        assert graph.rejections[0]["verified"] is None
        assert graph.rejections[0]["fail_reason"] == VALUE_FAIL_NO_PAGE

    def test_paraphrased_quote_does_not_block_a_located_value(self):
        graph = _graph()
        node = graph.add_source("p1", "330 m", quote="it stands about a thousand feet tall")
        assert node is not None
        assert node.verified is True
        assert node.quote_verified is False

    def test_page_identity_guard_rejects_the_wrong_page(self):
        graph = _graph()
        contract = StepContract(text="height of the Eiffel Tower",
                                subject_tokens=["brooklyn", "bridge"])
        assert graph.add_source("p1", "330 m", contract=contract) is None
        assert graph.rejections[0]["fail_reason"] == "page_identity"

    def test_page_identity_guard_admits_the_right_page(self):
        graph = _graph()
        contract = StepContract(text="height", subject_tokens=["eiffel"])
        assert graph.add_source("p1", "330 m", contract=contract) is not None


class TestContentAddressing:
    def test_same_page_offsets_and_value_dedupe_to_one_node(self):
        graph = _graph()
        first = graph.add_source("p1", "1991 metres")
        second = graph.add_source("p1", "1,991 metres", quote="1,991 metres")
        assert first.id == second.id == source_node_id("p1", first.start, first.end, first.value)
        assert len(graph.nodes()) == 1

    def test_different_values_on_one_page_are_different_nodes(self):
        graph = _graph()
        first = graph.add_source("p1", "1991 metres")
        second = graph.add_source("p1", "330 m")
        assert first.id != second.id
        assert len(graph.nodes()) == 2

    def test_dedup_keeps_the_first_node_and_its_quote(self):
        graph = _graph()
        first = graph.add_source("p1", "330 m", quote="330 m tall")
        again = graph.add_source("p1", "330 m", quote="a paraphrase nobody wrote")
        assert again is first
        assert graph.node(first.id).quote_verified is True


class TestDerivedSeam:
    def test_derived_node_links_all_of_its_inputs(self):
        graph = _graph()
        a = graph.add_source("p1", "1991 metres")
        b = graph.add_source("p1", "330 m")
        derived = graph.add_derived("2321", "sum", [a.id, b.id])
        assert derived.kind == KIND_DERIVED
        assert derived.input_ids == (a.id, b.id)
        assert set(graph.edges()) == {(a.id, derived.id, "sum"), (b.id, derived.id, "sum")}

    def test_derived_dedupes_on_operation_inputs_and_value(self):
        graph = _graph()
        a = graph.add_source("p1", "330 m")
        first = graph.add_derived("330", "to_number", [a.id])
        second = graph.add_derived("330", "to_number", [a.id])
        assert first is second
        assert len(graph.nodes()) == 2

    def test_sources_of_follows_every_parent_not_just_the_first(self):
        graph = _graph()
        a = graph.add_source("p1", "1991 metres")
        b = graph.add_source("p1", "330 m")
        mid = graph.add_derived("2321", "sum", [a.id, b.id])
        top = graph.add_derived("2321 total", "label", [mid.id])
        assert {node.id for node in graph.sources_of(top.id)} == {a.id, b.id}

    def test_derived_on_an_unknown_input_is_refused(self):
        graph = _graph()
        with pytest.raises(ValueError):
            graph.add_derived("x", "sum", ["nope"])

    def test_derived_needs_at_least_one_input(self):
        graph = _graph()
        with pytest.raises(ValueError):
            graph.add_derived("x", "sum", [])


class TestReverify:
    def test_reverify_graph_re_derives_verification_from_the_artifact_alone(self):
        graph = _graph()
        node = graph.add_source("p1", "1991 metres", quote="1,991 metres")
        artifact = graph.to_dict()
        report = reverify_graph(artifact)
        assert report["counts"]["verified"] == 1
        assert report["counts"]["failed"] == 0
        assert report["nodes"][0]["node_id"] == node.id
        assert report["nodes"][0]["verified"] is True

    def test_reverify_detects_a_node_whose_value_left_the_page(self):
        graph = _graph()
        graph.add_source("p1", "330 m")
        artifact = graph.to_dict()
        artifact["pages"][0]["text"] = "an entirely different page"
        report = reverify_graph(artifact)
        assert report["counts"]["failed"] == 1
        assert report["nodes"][0]["verified"] is False

    def test_reverify_never_calls_a_missing_page_a_failure(self):
        graph = _graph()
        graph.add_source("p1", "330 m")
        artifact = graph.to_dict()
        artifact["pages"] = []
        report = reverify_graph(artifact)
        assert report["counts"]["failed"] == 0
        assert report["counts"]["unchecked"] == 1
        assert report["nodes"][0]["fail_reason"] == VALUE_FAIL_NO_PAGE

    def test_reverify_reports_derived_nodes_through_their_sources(self):
        graph = _graph()
        a = graph.add_source("p1", "330 m")
        graph.add_derived("330", "to_number", [a.id])
        report = reverify_graph(graph.to_dict())
        assert report["counts"]["verified"] == 1
        assert report["counts"]["derived"] == 1

    def test_round_trip_restores_the_graph(self):
        graph = _graph()
        a = graph.add_source("p1", "330 m")
        graph.add_derived("330", "to_number", [a.id])
        restored = EvidenceGraph.from_dict(graph.to_dict())
        assert [n.id for n in restored.nodes()] == [n.id for n in graph.nodes()]
        assert restored.edges() == graph.edges()


class TestStoredPageHelper:
    def test_value_against_stored_page_uses_the_same_tri_state(self):
        graph = _graph()
        page = graph.page("p1")
        assert verify_value_against_stored_page(page, "330 m").verified is True
        assert verify_value_against_stored_page(page, "331 m").verified is False
        assert verify_value_against_stored_page(None, "330 m").fail_reason == VALUE_FAIL_NO_PAGE


class TestBoundaryIntegrity:
    """A numeric value must sit at non-alphanumeric boundaries — no matches inside a digit run."""

    @pytest.mark.parametrize("page,value", [
        ("11991", "1991"),
        ("19912003", "1991"),
        ("1950", "50"),
        ("1,300", "3"),
        ("built 41991 times", "1991"),
        ("330000 people", "330"),
    ])
    def test_numeric_value_inside_a_digit_run_does_not_verify(self, page, value):
        assert verify_value(page, value).verified is False

    @pytest.mark.parametrize("page", ["1991", "in 1991.", "(1991)", "1991, Paris", "x=1991"])
    def test_numeric_value_at_real_boundaries_still_verifies(self, page):
        assert verify_value(page, "1991").verified is True

    @pytest.mark.parametrize("page,value", [
        ("Molar heat capacity 26.74 J/(mol K)", "74"),
        ("Bulk modulus 41.2 GPa", "41"),
        ("Lattice constant c = 569.66 pm", "66"),
        ("a toll of 1,300 yen", "300"),
        ("version 4.11.2 released", "11"),
    ])
    def test_a_number_continuing_past_a_decimal_point_is_not_a_standalone_match(self, page, value):
        assert verify_value(page, value).verified is False

    @pytest.mark.parametrize("page,value", [
        ("built in 1991. Later it closed.", "1991"),
        ("Molar heat capacity 26.74 J", "26.74"),
        ("scores were 3, 4 and 5", "4"),
        ("a toll of 1,300 yen", "1,300"),
    ])
    def test_a_number_ending_at_punctuation_still_verifies(self, page, value):
        assert verify_value(page, value).verified is True

    def test_the_corpus_case_the_old_rule_admitted(self):
        # Every record the tightening drops from the real 405-record corpus is this one: an
        # atomic number "verified" against the tail of an isotope label and a decimal fraction.
        page = "isotopes 235 92U, and 239 92U). Molar heat capacity 28.91 J/(mol K)"
        assert verify_value(page, "39").verified is False

    def test_word_value_inside_a_longer_word_does_not_verify(self):
        assert verify_value("the metropolis grew", "metro").verified is False

    def test_boundary_check_scans_past_a_bad_first_occurrence(self):
        match = verify_value("11991 and then 1991 alone", "1991")
        assert match.verified is True
        assert match.start == 15


class TestDigitDigitSpaceFusion:
    """Deleting a digit-digit space fused infobox rows into one digit string. It must not."""

    @pytest.mark.parametrize("page,value", [
        ("1985 1991 2003", "19912003"),
        ("1985 1991 2003", "8519"),
        ("460 590", "460590"),
        ("12 345", "12345"),
    ])
    def test_adjacent_numbers_do_not_fuse(self, page, value):
        assert verify_value(page, value).verified is False

    def test_benign_newline_unit_case_still_verifies(self):
        page = "Height 285\nm (935\nft) above the river"
        match = verify_value(page, "285 m (935 ft)")
        assert match.verified is True
        assert page[match.start:match.end] == "285\nm (935\nft)"

    def test_digit_group_separator_still_collapses(self):
        assert verify_value("1,991 metres", "1991 metres").verified is True

    def test_normalize_keeps_the_space_between_two_numbers(self):
        assert normalize_for_match("460 590") == "460 590"

    def test_normalize_still_drops_the_space_before_a_unit(self):
        assert normalize_for_match("285 m") == "285m"


class TestSuperscriptFolding:
    """NFKC folds superscripts to digits. A superscript must not supply a digit for a match."""

    def test_superscript_does_not_supply_a_digit(self):
        assert verify_value("2", "²").verified is False
        assert verify_value("²", "2").verified is False

    def test_squared_unit_does_not_become_a_number(self):
        assert verify_value("area of 5 km²", "5 km2").verified is False

    def test_fullwidth_digits_still_fold(self):
        assert verify_value("Population １２３", "123").verified is True


class TestOccurrenceCount:
    def test_single_occurrence_is_counted_once(self):
        assert verify_value("built in 1991.", "1991").occurrences == 1

    def test_repeated_value_reports_every_boundary_valid_occurrence(self):
        match = verify_value("1991 census; revised 1991; cited 1991", "1991")
        assert match.verified is True
        assert match.occurrences == 3

    def test_occurrences_ignore_hits_inside_a_digit_run(self):
        match = verify_value("11991 and 1991", "1991")
        assert match.occurrences == 1

    def test_absent_value_reports_zero_occurrences(self):
        assert verify_value("nothing here", "1991").occurrences == 0


class TestUnitBearing:
    @pytest.mark.parametrize("value,expected", [
        ("330 m", True), ("330m", True), ("1,991 metres", True),
        ("330", False), ("1991", False), ("Gustave Eiffel", False), ("", False),
    ])
    def test_is_unit_bearing(self, value, expected):
        assert is_unit_bearing(value) is expected

    @pytest.mark.parametrize("value,expected", [
        ("1991", "bare_year"), ("330", "bare_number"), ("1,300.5", "bare_number"),
        ("330 m", "number_with_unit"), ("Gustave Eiffel", "text"), ("  ", "empty"),
    ])
    def test_value_shape(self, value, expected):
        assert value_shape(value) == expected

    def test_bare_value_verifies_and_reports_itself_as_bare(self):
        match = verify_value(PAGE, "330")
        assert match.verified is True
        assert match.unit_bearing is False

    def test_value_plus_unit_verifies_as_one_span(self):
        match = verify_value(PAGE, "330", unit="m")
        assert match.verified is True
        assert match.unit_bearing is True
        assert PAGE[match.start:match.end] == "330 m"

    def test_unit_bearing_value_needs_the_unit_adjacent(self):
        assert verify_value("330 people and a mast in m", "330", unit="m").unit_bearing is False

    def test_a_wrong_unit_falls_back_to_the_bare_value_never_silently_rejects(self):
        match = verify_value(PAGE, "330", unit="ft")
        assert match.verified is True
        assert match.unit_bearing is False


LABEL_PAGE = ("Cited 1991 in the references. " + "x" * 400 +
              " The bridge opened to traffic in 1991 after a long delay.")


class TestLabelProximity:
    def test_label_token_near_the_match_is_reported(self):
        match = verify_value(LABEL_PAGE, "1991", label="opened bridge")
        assert match.verified is True
        assert match.label_nearby is True
        assert LABEL_PAGE[match.start - 20:match.start].endswith("to traffic in ")

    def test_label_absent_from_every_window_is_reported_false(self):
        match = verify_value(LABEL_PAGE, "1991", label="elevation")
        assert match.verified is True
        assert match.label_nearby is False

    def test_no_label_leaves_the_signal_unchecked(self):
        assert verify_value(LABEL_PAGE, "1991").label_nearby is None

    def test_label_of_only_stopwords_leaves_the_signal_unchecked(self):
        assert verify_value(LABEL_PAGE, "1991", label="of the").label_nearby is None


class TestAmbiguityRefusal:
    def test_flag_off_admits_an_ambiguous_value_exactly_as_before(self):
        page = "1991 census; revised 1991"
        default = verify_value(page, "1991")
        explicit = verify_value(page, "1991", refuse_ambiguous=False)
        assert default == explicit
        assert default.verified is True
        assert default.occurrences == 2

    def test_flag_on_refuses_a_repeated_value_with_no_label_nearby(self):
        match = verify_value("1991 census; revised 1991", "1991", refuse_ambiguous=True)
        assert match.verified is None
        assert match.fail_reason == VALUE_FAIL_AMBIGUOUS
        assert match.occurrences == 2

    def test_flag_on_admits_a_repeated_value_the_label_disambiguates(self):
        match = verify_value(LABEL_PAGE, "1991", label="opened bridge", refuse_ambiguous=True)
        assert match.verified is True

    def test_flag_on_admits_a_unique_value(self):
        assert verify_value("built in 1991.", "1991", refuse_ambiguous=True).verified is True

    def test_refusal_is_unverifiable_never_a_failure(self):
        match = verify_value("1991 census; revised 1991", "1991", refuse_ambiguous=True)
        assert match.verified is not False


class TestSignalsOnAdmittedNodes:
    def test_node_carries_occurrence_and_shape_signals(self):
        graph = _graph()
        node = graph.add_source("p1", "330", unit="m", label="tall")
        assert node.occurrences == 1
        assert node.unit_bearing is True
        assert node.label_nearby is True
        assert node.as_dict()["unit_bearing"] is True

    def test_signals_survive_a_round_trip(self):
        graph = _graph()
        graph.add_source("p1", "330", unit="m", label="tall")
        restored = EvidenceGraph.from_dict(graph.to_dict())
        assert restored.nodes()[0].unit_bearing is True
        assert restored.nodes()[0].label_nearby is True

    def test_ambiguity_refusal_is_recorded_as_a_rejection(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "1991 census; revised 1991")
        assert graph.add_source("p1", "1991", refuse_ambiguous=True) is None
        assert graph.rejections[0]["fail_reason"] == VALUE_FAIL_AMBIGUOUS
        assert graph.rejections[0]["verified"] is None


class TestNumericValue:
    @pytest.mark.parametrize("value,unit,expected", [
        ("330", "", 330.0), ("1,991", "", 1991.0), ("330 m", "", 330.0),
        ("330 m", "m", 330.0), ("42.5", "", 42.5), ("330m", "", 330.0),
    ])
    def test_parses_the_numeric_part(self, value, unit, expected):
        assert numeric_value(value, unit) == expected

    @pytest.mark.parametrize("value", ["Gustave Eiffel", "", "  ", "roughly 330"])
    def test_non_numeric_value_returns_none(self, value):
        assert numeric_value(value) is None


class TestExtractUnit:
    @pytest.mark.parametrize("value,expected", [
        ("330 m", "m"), ("1,991 metres", "metres"), ("330m", "m"),
        ("330", ""), ("Gustave Eiffel", ""), ("", ""),
    ])
    def test_extract_unit(self, value, expected):
        assert extract_unit(value) == expected


class TestSourceUnitField:
    def test_explicit_unit_kwarg_is_stored_on_the_node(self):
        graph = _graph()
        node = graph.add_source("p1", "330", unit="m")
        assert node.unit == "m"

    def test_unit_is_inferred_from_a_unit_bearing_value_with_no_kwarg(self):
        graph = _graph()
        node = graph.add_source("p1", "1,991 metres")
        assert node.unit == "metres"

    def test_bare_value_with_no_unit_kwarg_has_no_unit(self):
        graph = _graph()
        node = graph.add_source("p1", "330")
        assert node.unit == ""


class TestValueShapeUnitField:
    """FIX 1: a bare number with an explicit ``unit`` field is unit-bearing coverage too —
    734 numeric extractions from the gpu0831 block showed 24.1% carrying the unit embedded in the
    value string but a further 38.0% carrying it ONLY in the separate ``unit`` field, so the old
    value-string-only shape undercounted coverage by more than half."""

    @pytest.mark.parametrize("value,unit,expected", [
        ("330", "m", "number_with_unit"),
        ("1991", "people", "number_with_unit"),
        ("330", "", "bare_number"),
        ("1991", "", "bare_year"),
        ("330 m", "ft", "number_with_unit"),
        ("Gustave Eiffel", "m", "text"),
        ("", "m", "empty"),
    ])
    def test_value_shape_with_explicit_unit_field(self, value, unit, expected):
        assert value_shape(value, unit) == expected


class TestValueShapeDates:
    """FIX 3: a date is not a unit-bearing number. ``value_shape`` used to classify
    ``"28 October 1981"`` as ``number_with_unit`` (with ``"October 1981"`` read as its "unit"),
    which would corrupt any future arithmetic/compare op over dates and pollutes the unit-coverage
    metric."""

    @pytest.mark.parametrize("value", [
        "28 October 1981", "3 January 2005", "1 December 1999", "1981-10-28", "2005-01-03",
    ])
    def test_dates_get_their_own_shape(self, value):
        assert value_shape(value) == "date"

    def test_a_date_is_never_number_with_unit_even_with_a_unit_field(self):
        assert value_shape("28 October 1981", "October 1981") == "date"
        assert value_shape("28 October 1981", "1981-10-28") == "date"

    def test_a_date_is_not_unit_bearing(self):
        assert is_unit_bearing("28 October 1981") is False


class TestDualUnitParentheticalNormalization:
    """FIX 2: a Wikipedia infobox lists both unit systems in one string
    (``"m (423 ft)"``), so comparing the raw unit strings raises a FALSE mismatch and refuses a
    valid derivation. Normalizing to the leading token before comparison fixes the false refusal
    without weakening the genuine 'm' vs 'ft' mismatch check."""

    @pytest.mark.parametrize("unit_a,unit_b", [
        ("m (423 ft)", "m"),
        ("m (ft)", "m"),
        ("metres (2,051 ft)", "metres"),
        ("m (1,940 ft)", "m"),
    ])
    def test_real_corpus_dual_unit_strings_do_not_falsely_mismatch(self, unit_a, unit_b):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "The first is 100 tall. The second is 50 tall.")
        a = graph.add_source("p1", "100", unit=unit_a)
        b = graph.add_source("p1", "50", unit=unit_b)
        node = graph.add_arith("sum", [a.id, b.id])
        assert node.derivation_valid is True
        assert "mismatched" not in (node.derivation_detail or "")

    def test_both_sides_carrying_the_same_dual_unit_string_still_agree(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "The first is 100 tall. The second is 50 tall.")
        a = graph.add_source("p1", "100", unit="m (423 ft)")
        b = graph.add_source("p1", "50", unit="m (423 ft)")
        node = graph.add_arith("sum", [a.id, b.id])
        assert node.derivation_valid is True

    def test_genuinely_different_units_still_mismatch(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "590 m and 1,940 ft")
        metres = graph.add_source("p1", "590 m")
        feet = graph.add_source("p1", "1,940 ft")
        with pytest.raises(ValueError):
            graph.add_arith("sum", [metres.id, feet.id])

    def test_a_dual_unit_string_still_mismatches_a_genuinely_different_leading_unit(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "The first is 100 tall. The second is 50 tall.")
        a = graph.add_source("p1", "100", unit="m (423 ft)")
        b = graph.add_source("p1", "50", unit="ft")
        with pytest.raises(ValueError):
            graph.add_arith("sum", [a.id, b.id])


class TestLayer1RegressionCorruptThousandsSplit:
    """FIX 4: a real caught corruption. A page reading "59,000" was mis-extracted as
    value='59' unit='000' (the thousands separator misread as a split point). The hardened
    verifier correctly refuses this — pinned here so a future normalization change cannot start
    admitting it."""

    PAGE_59000 = "The population grew to 59,000 residents by the next census."

    def test_thousands_separator_split_value_and_unit_does_not_verify(self):
        match = verify_value(self.PAGE_59000, "59", unit="000")
        assert match.verified is False
        assert match.fail_reason == VALUE_FAIL_ABSENT

    def test_add_source_rejects_the_split_value_with_an_empty_quote(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/pop", self.PAGE_59000)
        node = graph.add_source("p1", "59", quote="", unit="000")
        assert node is None
        assert graph.rejections[-1]["verified"] is False


def _arith_graph():
    graph = EvidenceGraph()
    graph.add_page("p1", "https://example.org/stats",
                   "Team A scored 400 goals. Team B scored 424 goals. "
                   "Player X made 42 goals in 50 appearances. "
                   "Tower A is 590 m tall. Tower B is 1,940 ft tall.")
    return graph


class TestAddArith:
    def test_sum_is_recomputed_from_inputs(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id])
        assert node.kind == KIND_DERIVED
        assert node.value == "824"
        assert node.derivation_valid is True

    def test_difference_is_recomputed(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "424 goals")
        b = graph.add_source("p1", "400 goals")
        node = graph.add_arith("difference", [a.id, b.id])
        assert node.value == "24"
        assert node.derivation_valid is True

    def test_product_is_recomputed(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "42 goals")
        node = graph.add_arith("product", [a.id, b.id])
        assert node.value == "16800"

    def test_quotient_is_recomputed_with_a_composed_rate_unit(self):
        graph = _arith_graph()
        goals = graph.add_source("p1", "42 goals")
        apps = graph.add_source("p1", "50 appearances")
        node = graph.add_arith("quotient", [goals.id, apps.id])
        assert node.value == "0.84"
        assert node.unit == "goals/appearances"

    def test_ratio_is_an_alias_for_the_same_division(self):
        graph = _arith_graph()
        goals = graph.add_source("p1", "42 goals")
        apps = graph.add_source("p1", "50 appearances")
        node = graph.add_arith("ratio", [goals.id, apps.id])
        assert node.value == "0.84"

    def test_quotient_by_zero_raises(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        zero_page = EvidenceGraph()
        zero_page.add_page("p1", "https://example.org/x", "Score: 0 goals total")
        z = zero_page.add_source("p1", "0 goals")
        with pytest.raises(ValueError):
            graph.add_arith("quotient", [a.id, z.id])

    def test_the_motivating_hallucination_is_caught_by_recomputation(self):
        # Reference-solver motivating bug: the model stated "1594" against a true sum of 824.
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id], proposed_value="1594")
        assert node.value == "824"
        assert node.derivation_valid is False
        assert "1594" in node.derivation_detail
        assert "824" in node.derivation_detail

    def test_a_proposed_value_within_tolerance_is_not_a_disagreement(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id], proposed_value="824.0000001")
        assert node.derivation_valid is True
        assert node.derivation_detail == ""

    def test_mismatched_units_fail_loudly_and_create_no_node(self):
        graph = _arith_graph()
        metres = graph.add_source("p1", "590 m")
        feet = graph.add_source("p1", "1,940 ft")
        before = len(graph.nodes())
        with pytest.raises(ValueError):
            graph.add_arith("sum", [metres.id, feet.id])
        assert len(graph.nodes()) == before

    def test_matching_units_are_carried_onto_the_derived_node(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id])
        assert node.unit == "goals"

    def test_unknown_input_id_raises(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(ValueError):
            graph.add_arith("sum", [a.id, "nope"])

    def test_non_numeric_input_raises(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "Gustave Eiffel built it in 1889.")
        name = graph.add_source("p1", "Gustave Eiffel")
        year = graph.add_source("p1", "1889")
        with pytest.raises(ValueError):
            graph.add_arith("sum", [name.id, year.id])

    def test_difference_needs_exactly_two_inputs(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(ValueError):
            graph.add_arith("difference", [a.id])


class TestAddCount:
    def test_count_is_the_number_of_distinct_inputs(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_count([a.id, b.id])
        assert node.kind == KIND_DERIVED
        assert node.value == "2"
        assert node.derivation_valid is True

    def test_count_filters_duplicate_input_ids(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        node = graph.add_count([a.id, a.id, a.id])
        assert node.value == "1"

    def test_count_of_unknown_input_raises(self):
        graph = _arith_graph()
        with pytest.raises(ValueError):
            graph.add_count(["nope"])


class TestAddExtremum:
    def test_max_picks_the_largest_input_value_verbatim(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_extremum([a.id, b.id], "max")
        assert node.value == "424 goals"
        assert node.derivation_valid is True

    def test_min_picks_the_smallest_input_value(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_extremum([a.id, b.id], "min")
        assert node.value == "400 goals"

    def test_winner_is_one_of_the_inputs(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_extremum([a.id, b.id], "max")
        assert node.value in (a.value, b.value)

    def test_mismatched_units_fail_loudly(self):
        graph = _arith_graph()
        metres = graph.add_source("p1", "590 m")
        feet = graph.add_source("p1", "1,940 ft")
        with pytest.raises(ValueError):
            graph.add_extremum([metres.id, feet.id], "max")

    def test_unknown_mode_raises(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(ValueError):
            graph.add_extremum([a.id], "middle")


class TestAddCompare:
    def test_gt_is_recomputed(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "424 goals")
        b = graph.add_source("p1", "400 goals")
        node = graph.add_compare(a.id, b.id, "gt")
        assert node.value == "true"
        assert node.derivation_valid is True

    def test_lt_is_recomputed(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_compare(a.id, b.id, "lt")
        assert node.value == "true"

    def test_eq_uses_the_same_tolerance_as_arith(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        node = graph.add_compare(a.id, a.id, "eq")
        assert node.value == "true"

    def test_both_sides_must_be_graph_nodes(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(ValueError):
            graph.add_compare(a.id, "nope", "gt")

    def test_mismatched_units_fail_loudly(self):
        graph = _arith_graph()
        metres = graph.add_source("p1", "590 m")
        feet = graph.add_source("p1", "1,940 ft")
        with pytest.raises(ValueError):
            graph.add_compare(metres.id, feet.id, "gt")

    def test_unknown_mode_raises(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(ValueError):
            graph.add_compare(a.id, a.id, "spicier")


class TestDerivationValidity:
    def test_empty_graph_reports_none(self):
        graph = _arith_graph()
        assert graph.derivation_validity() is None

    def test_all_valid_derivations_report_1(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        graph.add_arith("sum", [a.id, b.id])
        assert graph.derivation_validity() == 1.0

    def test_a_disagreement_lowers_the_ratio(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        graph.add_arith("sum", [a.id, b.id], proposed_value="1594")
        assert graph.derivation_validity() == 0.0

    def test_a_derived_node_with_no_recorded_validity_does_not_pass(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        graph.add_derived("400", "to_number", [a.id])
        assert graph.derivation_validity() == 0.0

    def test_invalidity_propagates_to_a_downstream_derivation(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        bad = graph.add_arith("sum", [a.id, b.id], proposed_value="1594")
        c = graph.add_source("p1", "42 goals")
        downstream = graph.add_arith("sum", [bad.id, c.id])
        assert downstream.derivation_valid is False
        assert "invalid" in downstream.derivation_detail


class TestDerivedNodeSerialization:
    def test_unit_and_validity_survive_a_round_trip(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        graph.add_arith("sum", [a.id, b.id])
        restored = EvidenceGraph.from_dict(graph.to_dict())
        derived = [n for n in restored.nodes() if n.kind == KIND_DERIVED][0]
        assert derived.unit == "goals"
        assert derived.derivation_valid is True


class TestTypedDerivationRefusals:
    """Every ``add_*`` refusal carries a machine-readable CODE, not just prose.

    The consumer is the evidence loop's ``derive`` action, which has to turn a refusal into an
    observation the model can act on ("your two operands are in different units") rather than a
    dead step. Matching on the message text would couple that mapping to wording; these are typed
    at the raise site instead. Every one still subclasses ``ValueError``, so the pre-existing
    ``pytest.raises(ValueError)`` assertions above keep their meaning.
    """

    def test_every_derivation_error_is_still_a_value_error(self):
        for cls in (eg.UnitMismatch, eg.MissingOperand, eg.NonNumeric, eg.DivisionByZero,
                    eg.UnknownOperation, eg.WrongArity):
            assert issubclass(cls, eg.DerivationError)
            assert issubclass(cls, ValueError)

    def test_mismatched_units_raise_unit_mismatch(self):
        graph = _arith_graph()
        metres = graph.add_source("p1", "590 m")
        feet = graph.add_source("p1", "1,940 ft")
        with pytest.raises(eg.UnitMismatch) as excinfo:
            graph.add_arith("difference", [metres.id, feet.id])
        assert excinfo.value.code == "UNIT_MISMATCH"

    def test_unknown_input_id_raises_missing_operand(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(eg.MissingOperand) as excinfo:
            graph.add_arith("sum", [a.id, "no-such-node"])
        assert excinfo.value.code == "MISSING_OPERAND"

    def test_non_numeric_operand_raises_non_numeric(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/x", "Gustave Eiffel built it in 1889.")
        name = graph.add_source("p1", "Gustave Eiffel")
        year = graph.add_source("p1", "1889")
        with pytest.raises(eg.NonNumeric) as excinfo:
            graph.add_arith("sum", [name.id, year.id])
        assert excinfo.value.code == "NON_NUMERIC"

    def test_division_by_zero_raises_division_by_zero(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        graph.add_page("p2", "https://example.org/zero", "Score: 0 goals total")
        zero = graph.add_source("p2", "0 goals")
        with pytest.raises(eg.DivisionByZero) as excinfo:
            graph.add_arith("quotient", [a.id, zero.id])
        assert excinfo.value.code == "DIVISION_BY_ZERO"

    def test_unknown_operation_raises_unknown_operation(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(eg.UnknownOperation) as excinfo:
            graph.add_arith("exponentiate", [a.id])
        assert excinfo.value.code == "UNKNOWN_OPERATION"

    def test_binary_op_with_wrong_arity_raises_wrong_arity(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        c = graph.add_source("p1", "42 goals")
        with pytest.raises(eg.WrongArity) as excinfo:
            graph.add_arith("difference", [a.id, b.id, c.id])
        assert excinfo.value.code == "WRONG_ARITY"

    def test_arith_with_no_inputs_raises_wrong_arity(self):
        graph = _arith_graph()
        with pytest.raises(eg.WrongArity) as excinfo:
            graph.add_arith("sum", [])
        assert excinfo.value.code == "WRONG_ARITY"

    def test_extremum_mode_and_arity_are_typed(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        with pytest.raises(eg.UnknownOperation):
            graph.add_extremum([a.id], "median")
        with pytest.raises(eg.WrongArity):
            graph.add_extremum([], "max")

    def test_extremum_unit_mismatch_is_typed(self):
        graph = _arith_graph()
        metres = graph.add_source("p1", "590 m")
        feet = graph.add_source("p1", "1,940 ft")
        with pytest.raises(eg.UnitMismatch):
            graph.add_extremum([metres.id, feet.id], "max")

    def test_compare_mode_and_operands_are_typed(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        with pytest.raises(eg.UnknownOperation):
            graph.add_compare(a.id, b.id, "approx")
        with pytest.raises(eg.MissingOperand):
            graph.add_compare(a.id, "no-such-node", "gt")

    def test_count_on_an_unknown_input_is_missing_operand(self):
        graph = _arith_graph()
        with pytest.raises(eg.MissingOperand):
            graph.add_count(["no-such-node"])

    def test_add_derived_unknown_input_is_missing_operand(self):
        graph = _graph()
        with pytest.raises(eg.MissingOperand):
            graph.add_derived("x", "sum", ["nope"])


class TestDeclaredUnitRespelling:
    """A value whose unit is ABBREVIATED differently from the page must still verify.

    Found on a real cell (task 130, qwen2.5:7b, evidence_loop): the USGS page says
    ``"20,310 feet"``; the extractor reported ``value="20,310 ft"`` with ``unit="feet"`` — the
    right number, the unit correctly declared in its own field, and merely a different spelling
    in the value string. ``_candidates`` only ever tried ``"20,310 ft"``, so the value was
    reported ABSENT and admitted no node. Six of six extractions failed that way and the graph
    admitted nothing at all, which would leave the whole derivation layer inert in production
    while every offline test stayed green.

    The fix stays inside the module's no-fuzzy-matching rule: the extra spelling is built from
    the value's own numeric half plus the unit the MODEL ITSELF declared, and it is still matched
    as one exact, boundary-checked span. No conversion, no synonym table, no token-overlap.
    """

    PAGE = ("The official height for Denali has been measured at 20,310 feet, just 10 feet "
            "less than the previous elevation of 20,320 feet.")

    def test_an_abbreviated_unit_verifies_against_the_pages_spelling(self):
        match = verify_value(self.PAGE, "20,310 ft", unit="feet")
        assert match.verified is True
        assert self.PAGE[match.start:match.end] == "20,310 feet"
        assert match.unit_bearing is True, "the re-spelled match is still one unit-bearing span"

    def test_the_pages_own_spelling_still_verifies_unchanged(self):
        assert verify_value(self.PAGE, "20,310 feet", unit="feet").verified is True

    def test_a_wrong_number_is_still_refused_however_the_unit_is_spelled(self):
        assert verify_value(self.PAGE, "20,315 ft", unit="feet").verified is False

    def test_a_genuinely_absent_unit_family_is_still_refused(self):
        # The page is imperial only; a metric figure must not be manufactured from it.
        assert verify_value(self.PAGE, "6,194 m", unit="metres").verified is False

    def test_a_declared_unit_that_matches_the_value_adds_no_new_spelling(self):
        assert _candidates("20,310 feet", "feet") == [("20,310 feet", True)]

    def test_the_respelling_is_offered_after_the_models_own_spelling(self):
        cands = _candidates("20,310 ft", "feet")
        assert cands[0] == ("20,310 ft", True), "the model's literal spelling is still tried first"
        assert ("20,310 feet", True) in cands

    def test_a_bare_value_with_a_declared_unit_is_unchanged(self):
        assert _candidates("20,310", "feet")[0] == ("20,310 feet", True)

    def test_the_admitted_node_records_the_unit_it_was_given(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/denali", self.PAGE)
        node = graph.add_source("p1", "20,310 ft", unit="feet")
        assert node is not None
        assert graph.rejections == []


class TestDerivationRefusalsAreRecorded:
    """A REFUSED derivation must survive into the artifact, like a refused SOURCE already does.

    Found by auditing the `ledgernum22` campaign: tasks 222-224 exist to test that incompatible
    units are REFUSED rather than converted, and their cells carried zero DERIVED nodes -- which
    is consistent both with "correctly refused" and with "never attempted". Nothing distinguished
    them, because a refused derivation left no trace anywhere: the graph records `rejections` for
    SOURCE admission, the loop's scratchpad is not persisted in the result JSON, and a refusal by
    definition creates no node. The unit-mismatch refusal endpoint was therefore unmeasurable.
    """

    def test_a_recorded_refusal_carries_its_code_operation_and_operands(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "590 m")
        b = graph.add_source("p1", "1,940 ft")
        try:
            graph.add_arith("difference", [a.id, b.id])
        except eg.DerivationError as exc:
            graph.record_refusal("difference", [a.id, b.id], exc)
        assert len(graph.derivation_refusals) == 1
        row = graph.derivation_refusals[0]
        assert row["code"] == "UNIT_MISMATCH"
        assert row["operation"] == "difference"
        assert row["input_ids"] == [a.id, b.id]
        assert "mismatched units" in row["message"]

    def test_a_refusal_creates_no_node(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "590 m")
        b = graph.add_source("p1", "1,940 ft")
        before = len(graph.nodes())
        try:
            graph.add_arith("difference", [a.id, b.id])
        except eg.DerivationError as exc:
            graph.record_refusal("difference", [a.id, b.id], exc)
        assert len(graph.nodes()) == before

    def test_refusals_survive_the_artifact_roundtrip(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "590 m")
        b = graph.add_source("p1", "1,940 ft")
        try:
            graph.add_arith("sum", [a.id, b.id])
        except eg.DerivationError as exc:
            graph.record_refusal("sum", [a.id, b.id], exc)
        restored = EvidenceGraph.from_dict(graph.to_dict())
        assert [r["code"] for r in restored.derivation_refusals] == ["UNIT_MISMATCH"]

    def test_the_artifact_exposes_refusals_as_their_own_key(self):
        graph = _arith_graph()
        assert graph.to_dict()["derivation_refusals"] == []

    def test_refusal_counts_are_reported_by_code(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "590 m")
        b = graph.add_source("p1", "1,940 ft")
        for op in ("sum", "difference"):
            try:
                graph.add_arith(op, [a.id, b.id])
            except eg.DerivationError as exc:
                graph.record_refusal(op, [a.id, b.id], exc)
        try:
            graph.add_arith("integrate", [a.id])
        except eg.DerivationError as exc:
            graph.record_refusal("integrate", [a.id], exc)
        assert graph.refusal_counts() == {"UNIT_MISMATCH": 2, "UNKNOWN_OPERATION": 1}

    def test_a_graph_with_no_refusals_counts_nothing(self):
        assert _arith_graph().refusal_counts() == {}


class TestParseQuantityConservation:
    """`parse_quantity` must account for EVERY character or refuse.

    The defect this replaces produced confidently wrong numbers marked valid:
    ``sum("121 crore", "162753003")`` recomputed to 162,753,124 with ``derivation_valid: True``
    against a true 1,372,753,003 -- wrong by 8.4x -- because scale words were treated as opaque
    unit suffixes and the magnitude was silently discarded. A parser that cannot lose silently
    makes that class of bug impossible rather than patching the one instance.
    """

    def test_a_plain_number_parses_with_no_dimensions(self):
        q = eg.parse_quantity("162,753,003")
        assert q.ok and q.magnitude == 162753003.0
        assert (q.currency, q.unit, q.scale_name, q.residue) == ("", "", "", "")

    def test_a_unit_bearing_value_keeps_its_unit(self):
        q = eg.parse_quantity("590 m")
        assert q.ok and q.magnitude == 590.0 and q.unit == "m"

    def test_unparsed_residue_refuses_rather_than_guessing(self):
        q = eg.parse_quantity("1,645 ft or 501 m (from Wikipedia)")
        assert not q.ok and q.residue

    def test_prose_is_not_a_quantity(self):
        assert not eg.parse_quantity("Directly counting all stars is not feasible").ok
        assert not eg.parse_quantity("Gustave Eiffel").ok


class TestParseQuantityRanges:
    """A range is not a quantity. Today `numeric_value('1 trillion to 2.6 trillion')` silently
    returns 1.0 -- it takes the lower bound. Ranges must be detected BEFORE number extraction,
    because that is exactly how the lower bound leaks through."""

    @pytest.mark.parametrize("text", [
        "1 trillion to 2.6 trillion", "100–400 billion", "11 to 18%", "11–18%",
        "2.3 – 4.0", "~5.6", "100 or 400", "±3", "between 5 and 9",
    ])
    def test_ranges_and_approximations_refuse(self, text):
        q = eg.parse_quantity(text)
        assert not q.ok, f"{text!r} must not parse to a single magnitude"

    def test_a_range_never_silently_becomes_its_lower_bound(self):
        assert eg.parse_quantity("100–400 billion").magnitude in (None,)
        assert eg.numeric_value("1 trillion to 2.6 trillion") is None


class TestParseQuantityScaleWords:
    """Scale words join the MAGNITUDE, never the unit."""

    @pytest.mark.parametrize("text,expected", [
        ("2.5 thousand", 2_500.0), ("237.8 million", 237_800_000.0),
        ("100 billion", 100_000_000_000.0), ("1 trillion", 1_000_000_000_000.0),
        ("5 lakh", 500_000.0), ("121 crore", 1_210_000_000.0), ("116.5 crore", 1_165_000_000.0),
    ])
    def test_full_scale_words_multiply_the_magnitude(self, text, expected):
        q = eg.parse_quantity(text)
        assert q.ok and q.magnitude == expected
        assert q.unit == "", "a scale word is magnitude, not a unit"

    def test_the_scale_name_is_kept_for_provenance(self):
        assert eg.parse_quantity("121 crore").scale_name == "crore"

    def test_values_at_different_scales_become_comparable(self):
        """Today this raises a SPURIOUS unit mismatch because 'million' and 'billion' look like
        different units. They are the same dimension at different magnitudes."""
        assert eg.parse_quantity("2 billion").magnitude > eg.parse_quantity("500 million").magnitude

    def test_a_scale_word_with_a_real_unit_keeps_both(self):
        q = eg.parse_quantity("1.5 million tonnes")
        assert q.ok and q.magnitude == 1_500_000.0 and q.unit == "tonnes"


class TestParseQuantityBareLetterTrap:
    """THE dangerous rule. 245 stored values end in a bare letter and are overwhelmingly METRES.

    A naive scale parser reading `m` as "million" would turn `1,158 m` into 1.158 billion and
    corrupt every height and elevation task in the suite. Single-letter abbreviations scale ONLY
    when a currency prefix was consumed.
    """

    @pytest.mark.parametrize("text,magnitude,unit", [
        ("1,158 m", 1158.0, "m"), ("1,410 m", 1410.0, "m"), ("100 m", 100.0, "m"),
        ("590 m", 590.0, "m"), ("2 k", 2.0, "k"), ("5 bn", 5.0, "bn"),
    ])
    def test_a_bare_letter_after_a_plain_number_is_a_unit_not_a_scale(self, text, magnitude, unit):
        q = eg.parse_quantity(text)
        assert q.ok and q.magnitude == magnitude, f"{text!r} must stay {magnitude}"
        assert q.unit == unit and q.scale_name == ""

    @pytest.mark.parametrize("text,expected", [
        ("$5m", 5_000_000.0), ("$1.2bn", 1_200_000_000.0), ("£250k", 250_000.0),
    ])
    def test_a_bare_letter_behind_a_currency_IS_a_scale(self, text, expected):
        q = eg.parse_quantity(text)
        assert q.ok and q.magnitude == expected and q.currency

    def test_the_metres_corpus_is_unharmed(self):
        """Regression guard over the real shape that dominates the stored corpus."""
        for n in ("1,280", "1,298", "1,470", "1,594", "1,642", "1,675.15", "1,777"):
            q = eg.parse_quantity(f"{n} m")
            assert q.ok and q.magnitude == float(n.replace(",", "")) and q.unit == "m"


class TestParseQuantityCurrency:
    """Currency is a DIMENSION, not decoration -- so GBP vs USD is a real mismatch."""

    @pytest.mark.parametrize("text,magnitude,currency", [
        ("$162,753,003", 162753003.0, "USD"), ("£7,481,396", 7481396.0, "GBP"),
        ("€533 million", 533_000_000.0, "EUR"), ("Rs 116.5 crore", 1_165_000_000.0, "INR"),
        ("INR 121 crore", 1_210_000_000.0, "INR"), ("¥500", 500.0, "JPY"),
    ])
    def test_a_currency_prefixed_value_parses(self, text, magnitude, currency):
        q = eg.parse_quantity(text)
        assert q.ok and q.magnitude == magnitude and q.currency == currency

    def test_currency_is_not_conflated_with_the_unit(self):
        q = eg.parse_quantity("£2,395,000")
        assert q.currency == "GBP" and q.unit == ""

    def test_two_currencies_are_different_dimensions(self):
        assert eg.parse_quantity("$100").currency != eg.parse_quantity("£100").currency


class TestParseQuantityRestatement:
    """`"1,308 m (4,291 ft)"` -- 359 stored occurrences, the largest recoverable set. The module
    already knows this shape: `_UNIT_PARENTHETICAL` strips it before comparing units."""

    def test_a_dual_unit_parenthetical_keeps_the_primary_quantity(self):
        q = eg.parse_quantity("1,308 m (4,291 ft)")
        assert q.ok and q.magnitude == 1308.0 and q.unit == "m"

    def test_the_restatement_is_recorded_not_discarded(self):
        q = eg.parse_quantity("1,308 m (4,291 ft)")
        assert q.restatement is not None
        assert q.restatement.magnitude == 4291.0 and q.restatement.unit == "ft"

    def test_a_prose_parenthetical_is_residue_and_refuses(self):
        q = eg.parse_quantity("1,645 ft (from a comparison table on another page)")
        assert not q.ok and q.residue

    def test_a_spelled_out_restatement_still_parses(self):
        q = eg.parse_quantity("1,642 m (5,387 feet)")
        assert q.ok and q.magnitude == 1642.0


class TestParseQuantityWiring:
    """The parse feeds ARITHMETIC only. Span location keeps today's semantics exactly."""

    def test_numeric_value_delegates_to_the_parser(self):
        assert eg.numeric_value("121 crore") == 1_210_000_000.0
        assert eg.numeric_value("Rs 116.5 crore") == 1_165_000_000.0
        assert eg.numeric_value("1,158 m") == 1158.0

    def test_matching_semantics_are_untouched(self):
        """`verify_value`/`is_unit_bearing`/`value_shape` must not shift: they locate spans."""
        assert eg.is_unit_bearing("121 crore") is True
        assert eg.value_shape("1,158 m") == "number_with_unit"
        page = "The tower is 121 crore rupees and 1,158 m tall."
        assert eg.verify_value(page, "1,158 m").verified is True
        assert eg.verify_value(page, "121 crore").verified is True

    def test_the_motivating_bug_is_fixed(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "Chhaava 121 crore. Minecraft 162753003 dollars.")
        a = graph.add_source("p1", "121 crore")
        b = graph.add_source("p1", "162753003")
        assert a is not None and b is not None
        node = graph.add_arith("sum", [a.id, b.id])
        assert float(node.value) == 121 * 10**7 + 162753003

    def test_mixed_scales_now_compare_correctly_instead_of_refusing(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "Alpha 500 million. Beta 2 billion.")
        a = graph.add_source("p1", "500 million")
        b = graph.add_source("p1", "2 billion")
        node = graph.add_extremum([a.id, b.id], "max")
        assert "2 billion" in node.value

    def test_currency_mismatch_is_a_unit_mismatch_not_a_parse_failure(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "Wimbledon £2,395,000 and US Open $90,535,500.")
        a = graph.add_source("p1", "£2,395,000")
        b = graph.add_source("p1", "$90,535,500")
        assert a is not None and b is not None
        with pytest.raises(eg.UnitMismatch):
            graph.add_arith("sum", [a.id, b.id])

    def test_same_currency_still_combines(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "First £2,395,000 then £7,481,396.")
        a = graph.add_source("p1", "£2,395,000")
        b = graph.add_source("p1", "£7,481,396")
        node = graph.add_arith("sum", [a.id, b.id])
        assert float(node.value) == 2395000 + 7481396


class TestQuotientDimensionalAnnotation:
    """W1: quotient/ratio stays exempt from `UnitMismatch` (a rate is the point), but is no
    longer silent about its own dimensions -- `derivation_detail` carries a machine-readable
    `unit_note` and, for a same-dimension ratio, the composed unit is corrected to dimensionless.
    None of this ever touches `derivation_valid`: annotation, not refusal."""

    def test_same_dimension_ratio_is_flagged_dimensionless(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "Plant A generates 500 MW. Plant B generates 250 MW.")
        a = graph.add_source("p1", "500 MW")
        b = graph.add_source("p1", "250 MW")
        node = graph.add_arith("ratio", [a.id, b.id])
        assert node.unit == ""
        assert "unit_note=same_dimension_ratio" in node.derivation_detail
        assert "MW/MW" in node.derivation_detail
        assert node.derivation_valid is True

    def test_cross_dimension_quotient_is_flagged_not_refused(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "Cost was £500. Length was 10 m.")
        cost = graph.add_source("p1", "£500")
        length = graph.add_source("p1", "10 m")
        node = graph.add_arith("quotient", [cost.id, length.id])
        assert node.value == "50"
        assert "unit_note=cross_dimension_quotient" in node.derivation_detail
        assert node.derivation_valid is True

    def test_missing_unit_on_either_side_is_flagged_unassessed(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "A is 500. B is 20 m.")
        a = graph.add_source("p1", "500")
        b = graph.add_source("p1", "20 m")
        node = graph.add_arith("quotient", [a.id, b.id])
        assert "unit_note=unassessed_units" in node.derivation_detail
        assert node.derivation_valid is True

    def test_both_sides_missing_units_is_also_unassessed(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "A is 500. B is 20.")
        a = graph.add_source("p1", "500")
        b = graph.add_source("p1", "20")
        node = graph.add_arith("quotient", [a.id, b.id])
        assert "unit_note=unassessed_units" in node.derivation_detail

    def test_a_genuine_disagreement_still_composes_with_the_unit_note(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x", "Cost was £500. Length was 10 m.")
        cost = graph.add_source("p1", "£500")
        length = graph.add_source("p1", "10 m")
        node = graph.add_arith("quotient", [cost.id, length.id], proposed_value="99")
        assert node.derivation_valid is False
        assert "disagrees" in node.derivation_detail
        assert "unit_note=cross_dimension_quotient" in node.derivation_detail


class TestScaleWordNeverSurvivesAsUnit:
    """The live bug: a model derived `381 - 25` where `25` came from `25 million`, and the pair
    passed as unit-consistent because the scale word was split off and dropped before the value
    ever reached `parse_quantity`. `extract_unit` is the split point (used directly by
    `LedgerToolkit._locate`), so a scale word must never come back as its unit."""

    def test_a_bare_scale_word_is_not_a_unit(self):
        assert extract_unit("25 million") == ""

    def test_a_scale_word_followed_by_a_real_unit_keeps_only_the_unit(self):
        assert extract_unit("1.5 million tonnes") == "tonnes"

    def test_a_real_unit_is_unaffected(self):
        assert extract_unit("330 m") == "m"
        assert extract_unit("1,991 metres") == "metres"

    def test_25_million_parses_to_its_full_magnitude(self):
        assert numeric_value("25 million") == 25_000_000.0

    def test_381_million_minus_25_million_is_computed_at_full_scale(self):
        graph = EvidenceGraph()
        graph.add_page("p1", "https://e.org/x",
                       "Alpha revenue was 381 million dollars. Beta revenue was 25 million dollars.")
        a = graph.add_source("p1", "381 million")
        b = graph.add_source("p1", "25 million")
        node = graph.add_arith("difference", [a.id, b.id])
        assert node.value == "356000000"

    def test_ledger_toolkit_locate_keeps_the_scale_word_with_the_number(self):
        """Regression for the exact live shape: `LedgerToolkit.derive` splitting a literal operand
        on `extract_unit` used to mint a SOURCE node holding only `"25"`, discarding the x10^6."""
        from agent.app.ledger_tools import LedgerToolkit

        toolkit = LedgerToolkit()
        toolkit.register_page(
            "https://e.org/x",
            "Alpha revenue was 25 million dollars. Beta revenue was 381 million dollars.")
        observation = toolkit.derive("difference", ["381 million", "25 million"])
        assert "356000000" in observation
        node = toolkit._graph.nodes()[-1]
        assert node.value == "356000000"
        assert node.derivation_valid is True


class TestOperandSupported:
    """A DERIVED node's `operand_supported` is a separate axis from `derivation_valid`: it asks
    whether the operands are still grounded in located page text, not whether the arithmetic
    checked out."""

    def test_a_freshly_minted_derivation_is_supported(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id])
        assert node.operand_supported is True
        assert graph.operand_support_rate() == 1.0

    def test_a_graph_with_no_derived_nodes_reports_none(self):
        graph = _arith_graph()
        assert graph.operand_support_rate() is None

    def test_as_dict_round_trips_operand_supported(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id])
        restored = eg.EvidenceNode.from_dict(node.as_dict())
        assert restored.operand_supported is True

    def test_missing_field_deserializes_gracefully(self):
        """An artifact minted before this field existed has no `operand_supported` key at all."""
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        node = graph.add_arith("sum", [a.id, b.id])
        data = node.as_dict()
        del data["operand_supported"]
        restored = eg.EvidenceNode.from_dict(data)
        assert restored.operand_supported is None

    def test_reverify_graph_recomputes_operand_supported_on_a_legacy_artifact(self):
        """A stored artifact from before this field existed carries no `operand_supported` on its
        DERIVED node, and reverification must fill in the real answer rather than leave it None."""
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        graph.add_arith("sum", [a.id, b.id])
        artifact = graph.to_dict()
        for node_data in artifact["nodes"]:
            node_data.pop("operand_supported", None)

        result = reverify_graph(artifact)
        derived_rows = [row for row in result["nodes"] if row["kind"] == KIND_DERIVED]
        assert len(derived_rows) == 1
        assert derived_rows[0]["operand_supported"] is True
        assert result["operand_support_rate"] == 1.0

    def test_reverify_graph_reports_false_when_a_source_no_longer_verifies(self):
        graph = _arith_graph()
        a = graph.add_source("p1", "400 goals")
        b = graph.add_source("p1", "424 goals")
        graph.add_arith("sum", [a.id, b.id])
        artifact = graph.to_dict()
        artifact["pages"][0]["text"] = "This page no longer mentions either figure."
        artifact["pages"][0]["truncated"] = False

        result = reverify_graph(artifact)
        derived_rows = [row for row in result["nodes"] if row["kind"] == KIND_DERIVED]
        assert derived_rows[0]["operand_supported"] is False
        assert result["operand_support_rate"] == 0.0
