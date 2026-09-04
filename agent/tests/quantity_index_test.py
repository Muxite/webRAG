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
