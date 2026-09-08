"""Tests for :mod:`agent.app.quantity_index` — the host-neutral quantity extraction index.

Pinned cases straight from the brief: the real infobox slice ``Max.\\ndepth\\n1,642\\nm
(5,387\\nft)``, a date row that must yield nothing, an ordinal that must yield nothing, an inline
prose quantity, id stability, and ``render_index`` respecting ``max_chars``.
"""
import pytest

from agent.app.quantity_index import QuantityRef, build_index, lookup, render_index


class TestInfobox:
    def test_max_depth_slice_yields_label_value_unit(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        entry = entries[0]
        assert entry.label == "Max. depth"
        assert entry.value == "1,642"
        assert entry.unit == "m"
        assert entry.source == "infobox"

    def test_max_depth_parenthetical_secondary_unit_also_indexed(self):
        # Both values are literally on the page -- capture, not conversion (see module
        # docstring). The primary "m" entry stays first; the parenthetical "ft" restatement
        # is now ALSO indexed as its own entry, not silently discarded.
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        assert len(entries) == 2
        assert entries[1].label == "Max. depth"
        assert entries[1].value == "5,387"
        assert entries[1].unit == "ft"
        assert entries[1].source == "infobox"
        assert text[entries[1].start:entries[1].end] == "5,387"

    def test_max_depth_offsets_point_at_raw_value_span(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        entry = entries[0]
        assert text[entry.start:entry.end] == "1,642"

    def test_date_row_yields_nothing(self):
        text = "August\n1943"
        assert build_index(text) == []

    def test_ordinal_yields_nothing(self):
        text = "1st"
        assert build_index(text) == []

    def test_ordinal_embedded_in_infobox_shape_yields_nothing(self):
        text = "Rank\n1st\nCity\nMotihari"
        assert build_index(text) == []

    def test_single_line_label(self):
        text = "Average depth\n744.4\nm (2,442\nft)"
        entries = build_index(text)
        assert entries[0].label == "Average depth"
        assert entries[0].value == "744.4"
        assert entries[0].unit == "m"
        # Parenthetical restatement is captured too (see TestParentheticalCapture below).
        assert len(entries) == 2

    def test_bare_count_row_with_no_unit_yields_nothing(self):
        # "Ward(s)\n46" — an infobox row that is genuinely unit-less; the index must not
        # invent a unit out of the next unrelated label line.
        text = "Ward(s)\n46\nEstablished\n1866\nNamed after\n'Champak'"
        assert build_index(text) == []

    def test_value_and_unit_sharing_one_line(self):
        text = "Installed\ncapacity\n13,860 MW\nAnnual\ngeneration\n55.2\nTWh\n(2015)"
        entries = build_index(text)
        first = entries[0]
        assert first.value == "13,860"
        assert first.unit == "MW"
        assert first.source == "infobox"

    def test_two_infobox_rows_both_recovered(self):
        text = "Max.\nlength\n28\nkm (17\nmi)\nMax.\nwidth\n8\nkm (5.0\nmi)"
        entries = build_index(text)
        primary = [e for e in entries if e.source == "infobox" and e.unit == "km"]
        assert [e.value for e in primary] == ["28", "8"]
        assert [e.label for e in primary] == ["Max. length", "Max. width"]
        # Both parenthetical restatements are captured too.
        secondary = [e for e in entries if e.unit == "mi"]
        assert [e.value for e in secondary] == ["17", "5.0"]
        assert [e.label for e in secondary] == ["Max. length", "Max. width"]


class TestEmbeddedExtraNumberRow:
    """Item 2: an embedded second number on the unit line must not sink the whole row."""

    def test_humber_bridge_shape_recovers_all_three_values(self):
        text = "Total\nlength\n7,280\nft; 1.38\nmi (2,220\nm)"
        entries = build_index(text)
        values_units = {(e.value, e.unit) for e in entries}
        assert ("7,280", "ft") in values_units
        assert ("1.38", "mi") in values_units
        assert ("2,220", "m") in values_units

    def test_primary_value_offset_still_correct(self):
        text = "Total\nlength\n7,280\nft; 1.38\nmi (2,220\nm)"
        entries = build_index(text)
        primary = next(e for e in entries if e.value == "7,280")
        assert text[primary.start:primary.end] == "7,280"


class TestTrailingAnnexRow:
    """Item 3: a trailing non-quantity annex degrades to the longest leading quantity."""

    def test_floor_count_with_maintenance_annex_yields_leading_number(self):
        text = "Floor count\n154 + 9 maintenance"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].value == "154"
        assert entries[0].label == "Floor count"


class TestCompoundDuration:
    """Item 4: an "N hours M minutes" phrase folds into one decimal-hours quantity."""

    def test_two_hours_twenty_one_minutes_becomes_decimal_hours(self):
        text = "The train took 2 hours 21 minutes for the journey."
        entries = build_index(text)
        combined = [e for e in entries if e.unit == "h"]
        assert len(combined) == 1
        assert combined[0].value == "2.35"


class TestConservativeLabeledCounts:
    """Item 5: bare unit-less counts are indexed only under a whitelisted label."""

    def test_stadium_capacity_bare_count_is_indexed(self):
        text = "Capacity\n67,215"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].value == "67,215"
        assert entries[0].unit == "count"
        assert entries[0].label == "Capacity"

    def test_floor_count_bare_count_is_indexed(self):
        text = "Floor count\n154"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].value == "154"
        assert entries[0].unit == "count"

    def test_unwhitelisted_label_bare_count_still_yields_nothing(self):
        # Regression guard for the wildcard-suppression concern: an arbitrary label must NOT
        # newly qualify for bare-count capture.
        text = "Ward(s)\n46\nEstablished\n1866\nNamed after\n'Champak'"
        assert build_index(text) == []


class TestProse:
    def test_inline_prose_quantity_found_with_correct_offsets(self):
        text = "The tower reaches 419.7 metres above the ground."
        entries = build_index(text)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.value == "419.7"
        assert entry.unit == "metres"
        assert entry.source == "prose"
        assert text[entry.start:entry.end] == "419.7"

    def test_month_name_after_number_is_not_a_unit(self):
        text = "He was born on 25 June 1903 in Motihari."
        entries = build_index(text)
        assert entries == []

    def test_bare_number_with_no_trailing_word_yields_nothing(self):
        text = "There were 46 wards in total, roughly."
        # "wards" is not in the whitelist -> no quantity.
        assert build_index(text) == []

    def test_unit_at_end_of_sentence_strips_trailing_period(self):
        text = "The current height is approximately ~345 meters. It was reduced."
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].value == "345"
        assert entries[0].unit == "meters"


class TestDedup:
    def test_infobox_preferred_over_duplicate_prose_mention(self):
        text = (
            "Max.\ndepth\n1,642\nm (5,387\nft)\n"
            "The lake's maximum depth is 1,642 m, a striking figure."
        )
        entries = build_index(text)
        sources = [e.source for e in entries if e.value == "1,642" and e.unit == "m"]
        assert sources == ["infobox"]


class TestDeterminism:
    def test_build_index_is_deterministic(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)\nMax.\nwidth\n8\nkm (5.0\nmi)"
        first = build_index(text)
        second = build_index(text)
        assert first == second

    def test_ids_stable_across_calls(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)\nMax.\nwidth\n8\nkm (5.0\nmi)"
        entries_a = build_index(text)
        entries_b = build_index(text)
        rendered_a = render_index(entries_a)
        rendered_b = render_index(entries_b)
        assert rendered_a == rendered_b
        assert rendered_a.startswith("q1: Max. depth = 1,642 m")


class TestCap:
    def test_build_index_respects_limit(self):
        rows = "\n".join(f"Row{i}\n{i}\nm" for i in range(1, 100))
        entries = build_index(rows, limit=5)
        assert len(entries) == 5


class TestAbsence:
    def test_empty_text_returns_empty_list(self):
        assert build_index("") == []

    def test_text_with_no_quantities_returns_empty_list(self):
        assert build_index("Just some prose with no numbers at all.") == []

    def test_render_index_of_empty_list_is_empty_string(self):
        assert render_index([]) == ""


class TestRender:
    def test_render_format(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        rendered = render_index(entries)
        assert rendered == "q1: Max. depth = 1,642 m\nq2: Max. depth = 5,387 ft"

    def test_render_respects_max_chars(self):
        rows = "\n".join(f"Row{i}\n{i}\nm" for i in range(1, 50))
        entries = build_index(rows, limit=40)
        rendered = render_index(entries, max_chars=40)
        assert len(rendered) <= 40

    def test_render_no_unit_omits_trailing_space(self):
        entry = QuantityRef(label="", value="42", unit="", start=0, end=2, source="prose")
        assert render_index([entry]) == "q1: 42"


class TestLookup:
    def test_lookup_plain(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)\nMax.\nwidth\n8\nkm (5.0\nmi)"
        entries = build_index(text)
        assert lookup(entries, "q2") is entries[1]

    def test_lookup_tolerant_of_case_and_dot(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)\nMax.\nwidth\n8\nkm (5.0\nmi)"
        entries = build_index(text)
        assert lookup(entries, "Q2") is entries[1]
        assert lookup(entries, "q2.") is entries[1]

    def test_lookup_out_of_range_is_none(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        assert lookup(entries, "q9") is None

    def test_lookup_garbage_is_none(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        assert lookup(entries, "not a ref") is None

    def test_lookup_empty_entries_is_none(self):
        assert lookup([], "q1") is None


class TestNoUnitConversion:
    def test_dual_unit_restatement_reports_only_primary_unit_as_written(self):
        # Non-goal: never convert ft to m or vice versa. The restatement is now ALSO indexed
        # (see TestInfobox.test_max_depth_parenthetical_secondary_unit_also_indexed) as its own
        # entry in ITS OWN written unit -- but the primary entry's unit is never rewritten to
        # (or blended with) the restatement's, and no entry ever holds a computed conversion.
        text = "Max.\ndepth\n1,642\nm (5,387\nft)"
        entries = build_index(text)
        assert entries[0].unit == "m"
        assert "ft" not in entries[0].unit


class TestSuperscriptUnits:
    """km<sup>2</sup> flattens to ``km\\n2``: the unit window must keep growing past the lone
    superscript digit instead of returning at the shorter ``km`` parse."""

    @staticmethod
    def _river_page(name, length, basin):
        # The exact fixture shape from host_derive_test.py.
        return (f"{name}\n{name} is a major river.\nLength\n{length}\nkm\n"
                f"Basin size\n{basin}\nkm\n2\n")

    def test_basin_size_indexes_as_km2_and_length_as_km(self):
        from agent.app.testing.evidence_graph import canonical_unit
        text = self._river_page("Nile", "6,650", "3,400,000")
        entries = build_index(text)
        # The label of the first row also absorbs the preceding "... river." sentence via the
        # existing period-continuation rule; only the label's tail is pinned here.
        by_label = {e.label.split()[-1]: e for e in entries if e.source == "infobox"}
        by_label["Basin size"] = by_label.pop("size")
        assert canonical_unit(by_label["Length"].unit) == "km"
        assert by_label["Length"].value == "6,650"
        assert canonical_unit(by_label["Basin size"].unit) == "km2"
        assert by_label["Basin size"].value == "3,400,000"
        assert text[by_label["Basin size"].start:by_label["Basin size"].end] == "3,400,000"

    def test_following_label_is_never_swallowed_as_a_unit(self):
        text = "Length\n4,909\nkm\nBasin size\n795,000\nkm\n2\n"
        entries = build_index(text)
        length = next(e for e in entries if e.label == "Length")
        assert length.unit == "km"

    @pytest.mark.parametrize("unit_line,digit,expected", [
        ("m", "2", "m2"), ("mi", "2", "mi2"), ("km", "3", "km3"), ("km", "²", "km2"),
    ])
    def test_other_superscript_units(self, unit_line, digit, expected):
        from agent.app.testing.evidence_graph import canonical_unit
        text = f"Area\n1,234\n{unit_line}\n{digit}\nElevation\n12\nm\n"
        entries = build_index(text)
        area = next(e for e in entries if e.label == "Area")
        assert canonical_unit(area.unit) == expected
        elevation = next(e for e in entries if e.label == "Elevation")
        assert elevation.unit == "m"

    def test_superscript_digit_followed_by_punctuation(self):
        text = "Basin size\n795,000\nkm\n2\n)\nLength\n5\nkm\n"
        entries = build_index(text)
        basin = next(e for e in entries if e.label == "Basin size")
        assert basin.unit == "km2"

    def test_real_infobox_slice_with_dual_unit_restatement(self):
        text = "Basin size\n795,000\nkm\n2\n(307,000\nmi\n2\n)\nLength\n5\nkm\n"
        entries = build_index(text)
        basin = next(e for e in entries if e.label == "Basin size")
        assert basin.value == "795,000"
        assert basin.unit == "km2"


class TestCurrency:
    """Currency values must reach ``parse_quantity`` through every digit-anchored gate."""

    def test_infobox_euro_prefix_with_scale(self):
        text = "Construction cost\n€533 million\nOpened\n2009\n"
        entries = build_index(text)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.label == "Construction cost"
        assert entry.currency == "EUR"
        assert entry.scale == "million"
        assert text[entry.start:entry.end] == entry.value
        # The value+unit recombination consumers parse must keep currency AND scale.
        from agent.app.testing.evidence_graph import parse_quantity
        parsed = parse_quantity(f"{entry.value} {entry.unit}")
        assert parsed.ok and parsed.currency == "EUR" and parsed.magnitude == 533e6

    def test_infobox_dollar_prefix_with_scale(self):
        text = "Cost\n$1.2 billion\n"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].currency == "USD"
        assert entries[0].scale == "billion"
        assert entries[0].label == "Cost"

    def test_infobox_prefix_split_across_lines(self):
        text = "Cost\n€533\nmillion\nOpened\n2009\n"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].currency == "EUR"
        assert entries[0].scale == "million"
        assert entries[0].value == "€533"

    def test_infobox_suffix_symbol(self):
        text = "Ticket price\n100 €\n"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].currency == "EUR"
        assert entries[0].value == "100"
        from agent.app.testing.evidence_graph import numeric_value
        assert numeric_value(f"{entries[0].value} {entries[0].unit}") == 100

    def test_infobox_suffix_code_with_scale(self):
        text = "Cost\n533 million EUR\n"
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].currency == "EUR"
        assert entries[0].scale == "million"
        from agent.app.testing.evidence_graph import numeric_value
        assert numeric_value(f"{entries[0].value} {entries[0].unit}") == 533e6

    def test_prose_euro_prefix_with_scale(self):
        text = "The stadium cost €533 million to build."
        entries = build_index(text)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.source == "prose"
        assert entry.currency == "EUR"
        assert entry.scale == "million"
        assert text[entry.start:entry.end] == entry.value
        assert "to" not in entry.unit

    def test_prose_suffix_code(self):
        text = "It was sold for 533 million EUR at auction."
        entries = build_index(text)
        assert len(entries) == 1
        assert entries[0].currency == "EUR"
        assert entries[0].scale == "million"

    def test_plain_numbers_unchanged(self):
        text = "Max.\ndepth\n1,642\nm (5,387\nft)\nThe tower reaches 419.7 metres."
        entries = build_index(text)
        assert [(e.value, e.unit) for e in entries] == [
            ("1,642", "m"), ("5,387", "ft"), ("419.7", "metres")]
        assert all(e.currency == "" and e.scale == "" for e in entries)

    def test_currency_line_is_not_a_label(self):
        text = "Cost\n€533 million\n40\nm\n"
        entries = build_index(text)
        forty = next(e for e in entries if e.value == "40")
        assert forty.label == ""


class TestUncappedIndex:
    def test_limit_none_returns_every_entry_default_still_forty(self):
        rows = "\n".join(f"Row{i}\n{i}\nm" for i in range(1, 61))
        assert len(build_index(rows)) == 40
        assert len(build_index(rows, limit=None)) == 60
