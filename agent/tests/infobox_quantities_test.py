"""`agent/app/infobox_quantities.py` -- structured infobox rows -> `QuantityRef` entries.

The offset contract under test: every entry's `start`/`end` indexes `infobox_text(html)` at offset
0, so a host that registers `infobox_text + "\\n" + body` can hand the entries over untouched.
"""
from agent.app.infobox_quantities import (has_infobox, infobox_quantities, infobox_rows,
                                          infobox_text)
from agent.app.testing.evidence_graph import canonical_unit, parse_quantity

MEKONG_HTML = """
<html><body>
<table class="infobox vcard">
  <tr><th colspan="2" class="infobox-above">Mekong</th></tr>
  <tr><td colspan="2"><img src="x.jpg"></td></tr>
  <tr><th scope="row">Country</th><td>China, Myanmar, Laos</td></tr>
  <tr><th scope="row">Length</th><td>4,909 km (3,050 mi)<sup class="reference">[1]</sup></td></tr>
  <tr><th scope="row">Basin size</th><td>795,000 km<sup>2</sup> (307,000 mi<sup>2</sup>)</td></tr>
  <tr><th scope="row">• average</th><td>16,000 m<sup>3</sup>/s (570,000 cu ft/s)</td></tr>
</table>
<p>The Mekong is a trans-boundary river in East Asia and Southeast Asia.</p>
<table class="wikitable"><tr><th>Other</th><td>99 km</td></tr></table>
</body></html>
"""

PUSKAS_HTML = """
<table class="infobox">
  <tr><th>Capacity</th><td>67,215</td></tr>
  <tr><th>Construction cost</th><td>€533 million<sup class="reference">[7]</sup></td></tr>
  <tr><th>Opened</th><td>15 November 2019</td></tr>
  <tr><th>Record attendance</th><td>75,000 (Metallica; 11 June 2026)</td></tr>
</table>
"""

STATE_HTML = """
<table class="infobox">
  <tr><th colspan="2">Area</th></tr>
  <tr><th>• Total</th><td>48,430 sq mi (125,443 km<sup>2</sup>)</td></tr>
  <tr><th>Population</th><td>2,961,279</td></tr>
  <tr><th>Elevation</th><td>300 ft<br>(90 m)</td></tr>
</table>
"""


def _by_label(entries, label):
    return [e for e in entries if e.label == label]


def test_page_without_an_infobox_yields_nothing():
    html = "<html><body><p>Just prose with 4,909 km in it.</p><table class='wikitable'>" \
           "<tr><th>Length</th><td>4,909 km</td></tr></table></body></html>"
    assert infobox_quantities(html) == []
    assert infobox_text(html) == ""
    assert has_infobox(html) is False
    assert infobox_quantities("") == [] and infobox_quantities(None) == []


def test_only_the_first_infobox_is_read_and_only_labelled_rows_become_rows():
    rows = infobox_rows(MEKONG_HTML)
    assert [label for label, _, _ in rows] == ["Country", "Length", "Basin size", "• average"]
    assert [section for _, _, section in rows] == [""] * 4
    assert ("Other", "99 km", "") not in rows  # the wikitable is not an infobox


def test_basin_size_row_is_captured_with_the_superscript_folded_into_the_unit():
    entries = infobox_quantities(MEKONG_HTML)
    basin = _by_label(entries, "Basin size")
    assert [(e.value, e.unit) for e in basin] == [("795,000", "km²"), ("307,000", "mi²")]
    assert canonical_unit(basin[0].unit) == "km2"
    assert basin[0].source == "infobox"
    # Both halves of the dual-unit restatement are captured, primary first -- and neither is a
    # conversion of the other: both strings are on the page.
    assert basin[0].start < basin[1].start


def test_length_row_drops_the_citation_marker_and_keeps_both_halves():
    entries = infobox_quantities(MEKONG_HTML)
    assert [(e.value, e.unit) for e in _by_label(entries, "Length")] == [("4,909", "km"),
                                                                          ("3,050", "mi")]


def test_every_offset_indexes_the_rendered_infobox_text():
    text = infobox_text(MEKONG_HTML)
    for entry in infobox_quantities(MEKONG_HTML):
        assert text[entry.start:entry.end] == entry.value
    # The offsets stay valid when a host appends the cleaned body AFTER the infobox text.
    combined = text + "\n" + "The Mekong is a trans-boundary river."
    for entry in infobox_quantities(MEKONG_HTML):
        assert combined[entry.start:entry.end] == entry.value


def test_rendered_text_is_label_colon_value_lines_with_the_exponent_glued():
    text = infobox_text(MEKONG_HTML)
    assert "Length: 4,909 km (3,050 mi)" in text
    assert "Basin size: 795,000 km² (307,000 mi²)" in text
    assert "km ²" not in text
    assert text.startswith("Country: China, Myanmar, Laos")


def test_entries_come_out_in_document_order_row_by_row():
    labels = [e.label for e in infobox_quantities(MEKONG_HTML)]
    assert labels == ["Length", "Length", "Basin size", "Basin size"]


def test_a_unit_outside_the_index_whitelist_is_not_admitted():
    """`m³/s` / `cu ft/s` are not in `quantity_index._UNIT_WHITELIST`; the same gate applies here,
    so the discharge row yields nothing rather than a quantity with an unrecognised unit."""
    assert _by_label(infobox_quantities(MEKONG_HTML), "• average") == []


def test_currency_prefix_is_preserved_glued_inside_the_value_like_the_text_index_does():
    entries = infobox_quantities(PUSKAS_HTML)
    cost = _by_label(entries, "Construction cost")
    assert len(cost) == 1
    assert cost[0].value == "€533"
    assert cost[0].currency == "EUR" and cost[0].scale == "million"
    assert parse_quantity(f"{cost[0].value} {cost[0].unit}").magnitude == 533_000_000.0
    assert infobox_text(PUSKAS_HTML)[cost[0].start:cost[0].end] == "€533"


def test_bare_count_is_admitted_only_under_a_count_label_with_unit_count():
    entries = infobox_quantities(PUSKAS_HTML)
    assert [(e.value, e.unit) for e in _by_label(entries, "Capacity")] == [("67,215", "count")]
    # "Record attendance 75,000" is a bare integer under a non-whitelisted label: not indexed.
    assert _by_label(entries, "Record attendance") == []
    # A date is not a quantity.
    assert _by_label(entries, "Opened") == []


def test_bulleted_sub_rows_and_br_split_values_each_yield_their_own_quantities():
    entries = infobox_quantities(STATE_HTML)
    assert [(e.value, e.unit) for e in _by_label(entries, "• Total")] == [("48,430", "sq mi"),
                                                                           ("125,443", "km²")]
    assert [(e.value, e.unit) for e in _by_label(entries, "Population")] == [("2,961,279", "count")]
    assert [(e.value, e.unit) for e in _by_label(entries, "Elevation")] == [("300", "ft"),
                                                                             ("90", "m")]
    text = infobox_text(STATE_HTML)
    assert "Elevation: 300 ft | (90 m)" in text
    for entry in entries:
        assert text[entry.start:entry.end] == entry.value


def test_parser_never_raises_on_garbage_markup():
    assert infobox_quantities("<table class='infobox'><tr><th>Length<td>4,9") == []
    assert infobox_quantities("<table class='infobox'><tr><th>Length</th><td>4,909 km</td></tr>") \
        == infobox_quantities("<table class='infobox'><tr><th>Length</th><td>4,909 km</td></tr></table>")


def test_dual_unit_cells_yield_both_entries_under_the_same_label():
    """The G1 case: the model's flattened window kept only one unit of these cells; the
    structured read keeps both, primary first, each byte-for-byte as written."""
    html = _infobox_html_rows([("Basin size", "1,151,000 sq mi (2,980,000 km2)"),
                               ("Total length", "3,911 m (12,831 ft)")])
    entries = infobox_quantities(html)
    assert [(e.label, e.value, e.unit) for e in entries] == [
        ("Basin size", "1,151,000", "sq mi"), ("Basin size", "2,980,000", "km2"),
        ("Total length", "3,911", "m"), ("Total length", "12,831", "ft"),
    ]
    assert {canonical_unit(e.unit) for e in entries if e.label == "Basin size"} == {"sq mi", "km2"}
    text = infobox_text(html)
    for entry in entries:
        assert text[entry.start:entry.end] == entry.value


def _infobox_html_rows(rows):
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f"<html><body><table class='infobox'>{body}</table></body></html>"


GOLDEN_GATE_HTML = """
<table class="infobox">
  <tr><th>Total length</th><td>8980 ft, about 1.70 mi (2.74 km)</td></tr>
  <tr><th>Longest span</th><td>4200 ft, about 0.79 mi (1.27 km)</td></tr>
  <tr><th>Height</th><td>746 ft (227.4 m)</td></tr>
  <tr><th>Cost</th><td>€ 533 million</td></tr>
  <tr><th>Opened</th><td>May 27, 1937; 89 years ago (1937-05-27)</td></tr>
  <tr><th>Daily traffic</th><td>88,716 (FY2020)</td></tr>
</table>
"""


def test_golden_gate_comma_separated_cells_emit_every_quantity_in_document_order():
    """Live prefetch replay bug: the leading `8980 ft` (unit followed by a comma) was lost, so
    the shared-unit rule found no ft entry under `Total length` and fell back to prose."""
    entries = infobox_quantities(GOLDEN_GATE_HTML)
    assert [(e.value, e.unit) for e in _by_label(entries, "Total length")] == [
        ("8980", "ft"), ("1.70", "mi"), ("2.74", "km")]
    assert [(e.value, e.unit) for e in _by_label(entries, "Longest span")] == [
        ("4200", "ft"), ("0.79", "mi"), ("1.27", "km")]
    text = infobox_text(GOLDEN_GATE_HTML)
    for entry in entries:
        assert text[entry.start:entry.end] == entry.value


def test_golden_gate_other_rows_are_unchanged_by_the_comma_split():
    entries = infobox_quantities(GOLDEN_GATE_HTML)
    assert [(e.value, e.unit) for e in _by_label(entries, "Height")] == [("746", "ft"),
                                                                          ("227.4", "m")]
    cost = _by_label(entries, "Cost")
    assert [(e.value, e.unit, e.currency) for e in cost] == [("€ 533", "million", "EUR")]
    # A date cell yields no quantity: neither the day/year numbers nor "89 years ago".
    assert _by_label(entries, "Opened") == []
    # A bare count under a non-whitelisted label is not admitted (it was not before either).
    assert _by_label(entries, "Daily traffic") == []
    assert [e.label for e in entries] == ["Total length"] * 3 + ["Longest span"] * 3 + ["Height"] * 2 + ["Cost"]


SHANGHAI_HTML = """
<table class="infobox">
  <tr><th class="infobox-above" colspan="2">Shanghai Tower</th></tr>
  <tr><td colspan="2"><img src="tower.jpg"></td></tr>
  <tr><th colspan="2">General information</th></tr>
  <tr><th>Status</th><td>Completed</td></tr>
  <tr><th colspan="2">Height</th></tr>
  <tr><th>Architectural</th><td>632 m (2,073 ft)</td></tr>
  <tr><th>Tip</th><td>632 m (2,073 ft)</td></tr>
  <tr><th>Roof</th><td>587 m (1,926 ft)</td></tr>
  <tr><th>Top floor</th><td>561 m (1,841 ft)</td></tr>
  <tr><th colspan="2">Technical details</th></tr>
  <tr><th>Floor count</th><td>128</td></tr>
</table>
"""


def test_section_header_rows_become_the_section_of_the_rows_beneath_them_labels_stay_bare():
    """221 Shanghai Tower: the sub-rows under a `Height` header carried only `Architectural` /
    `Tip` / ..., so a slot asking for the height could not see which field they belonged to.
    The header rides on `section`; the label is byte-identical to a header-less table, because
    prefixing it halved the overlap of every row under a header that is not the field
    (measured regression on 218/212/214/211)."""
    rows = infobox_rows(SHANGHAI_HTML)
    assert [(label, section) for label, _, section in rows] == [
        ("Status", "General information"), ("Architectural", "Height"), ("Tip", "Height"),
        ("Roof", "Height"), ("Top floor", "Height"), ("Floor count", "Technical details")]
    entries = infobox_quantities(SHANGHAI_HTML)
    assert [(e.label, e.section, e.value, e.unit) for e in entries][:2] == [
        ("Architectural", "Height", "632", "m"), ("Architectural", "Height", "2,073", "ft")]
    assert [(e.label, e.section, e.value, e.unit) for e in entries][-1] == (
        "Floor count", "Technical details", "128", "count")
    assert "Shanghai Tower" not in infobox_text(SHANGHAI_HTML)   # the title row is not a section
    assert infobox_text(SHANGHAI_HTML).startswith("Status: Completed\nArchitectural: 632 m")
    text = infobox_text(SHANGHAI_HTML)
    for entry in entries:
        assert text[entry.start:entry.end] == entry.value


def test_the_ranker_reads_section_plus_label_and_never_below_the_bare_label():
    from agent.app.operand_attribution import features, label_token_overlap
    from agent.app.quantity_index import QuantityRef

    class Slot:
        entity, field_phrase = "Shanghai Tower", "its height, in meters"

    entries = infobox_quantities(SHANGHAI_HTML)
    arch = next(e for e in entries if e.label == "Architectural")
    label_feature = features(Slot, arch)[0]
    assert label_feature == label_token_overlap(Slot.field_phrase, "Height Architectural") == 0.5
    assert label_feature > label_token_overlap(Slot.field_phrase, "Architectural") == 0.0
    # A river's `Length` under `Physical characteristics` scores exactly as bare `Length`.
    river = QuantityRef(label="Length", value="4,909", unit="km", start=0, end=5,
                        source="infobox", section="Physical characteristics")
    bare = QuantityRef(label="Length", value="4,909", unit="km", start=0, end=5, source="infobox")

    class River:
        entity, field_phrase = "Mekong", "the river's LENGTH (in kilometres)"

    assert features(River, river)[0] == features(River, bare)[0] == 1.0


def test_a_leading_header_row_without_a_class_is_still_the_title_not_a_section():
    html = ("<table class='infobox'><tr><th colspan='2'>Mekong</th></tr>"
            "<tr><th>Length</th><td>4,909 km</td></tr></table>")
    assert infobox_rows(html) == [("Length", "4,909 km", "")]


def test_a_table_without_section_headers_is_rendered_exactly_as_before():
    assert [label for label, _, _ in infobox_rows(MEKONG_HTML)] == ["Country", "Length",
                                                                   "Basin size", "• average"]
    assert infobox_text(MEKONG_HTML).startswith("Country: China, Myanmar, Laos\nLength: 4,909 km")
