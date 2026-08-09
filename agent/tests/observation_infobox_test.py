"""
Offline tests for ``observation.extract_infobox_block`` and the opt-in ``clean_operation
(prepend_infobox=...)`` flag — free, no network.

Why the block exists: ``clean_operation`` extracts a page with ONE page-wide
``get_text(separator="\\n")``, so every infobox cell lands on its own line with no delimiter
("Average depth / 201.8 m / Max. depth / 505 m"). A weak model asked an under-specified question
then reads off the NEIGHBOURING field. ``extract_infobox_block`` re-renders the first infobox as
``Label: Value`` lines using each cell's own ``get_text(" ", strip=True)``.

The load-bearing assertion here is the REGRESSION one: with the flag omitted or ``False``,
``clean_operation`` output is byte-identical to the pre-change implementation for every fixture
(the goldens below were captured by running the pre-change ``observation.py`` from git).
"""
import pytest

from agent.app.observation import clean_operation, extract_infobox_block


# --- fixtures ------------------------------------------------------------------------------------
WIKI_INFOBOX_PAGE = """
<html><head><title>Sarez Lake - Wikipedia</title></head>
<body>
<div id="mw-navigation"><a class="mw-jump-link" href="#content">Jump to content</a></div>
<div id="mw-content-text"><div class="mw-parser-output">
<table class="infobox vcard">
  <caption class="infobox-title">Sarez Lake</caption>
  <tr><td colspan="2" class="infobox-image">A photo of the lake</td></tr>
  <tr><th class="infobox-label">Location</th><td class="infobox-data">Rushon District, Tajikistan</td></tr>
  <tr><th class="infobox-label">Coordinates</th><td class="infobox-data">38&#176;12&#8242;N<br/>72&#176;46&#8242;E</td></tr>
  <tr><th colspan="2" class="infobox-header">Basin countries</th></tr>
  <tr><th class="infobox-label">Max. length</th><td class="infobox-data">55.8 km (34.7 mi)</td></tr>
  <tr><th class="infobox-label">Average depth</th><td class="infobox-data">201.8 m (662 ft)</td></tr>
  <tr><th class="infobox-label">Max. depth</th><td class="infobox-data">505 m (1,657 ft)</td></tr>
  <tr><th class="infobox-label">Surface elevation</th><td class="infobox-data">3,263 m (10,705 ft)</td></tr>
</table>
<p>Sarez Lake is a lake in the Rushon District of Tajikistan.</p>
<p>It was formed in 1911 when an earthquake triggered a landslide.</p>
</div></div>
<div id="footer">Privacy policy</div>
</body></html>
"""

NO_INFOBOX_PAGE = """
<html><head><title>Plain article</title></head>
<body>
<nav>Main menu</nav>
<main>
<h1>Plain article</h1>
<table class="wikitable"><tr><th>Year</th><td>1911</td></tr></table>
<p>This page has a table but no infobox at all.</p>
</main>
</body></html>
"""

# Fixtures already exercised elsewhere in the offline suite (perf_optimizations_test,
# idea_actions_test, naive_rag_iterative_test, visit_url_extraction_test) — the must-not-break
# callers of clean_operation.
SIMPLE_MAIN_PAGE = "<html><body><main><p>Some page content about the topic.</p></main></body></html>"
LINK_PAGE = "<html><body><a href='https://x.example'>X</a><p>Alpha</p></body></html>"
TITLE_PAGE = (
    "<html><head><title>My Title</title></head>"
    "<body><main><h1>Heading</h1><p>Body text here.</p></main></body></html>"
)
HEADING_PAGE = (
    "<html><body><h1>Test Page</h1><p>Content</p>"
    "<a href='https://link.example'>Link</a></body></html>"
)
EMPTY_PAGE = "<html><body></body></html>"

# Golden outputs captured from the PRE-change observation.py (git HEAD) — byte-for-byte.
PRE_CHANGE_GOLDENS = [
    (WIKI_INFOBOX_PAGE,
     'Sarez Lake\nA photo of the lake\nLocation\nRushon District, Tajikistan\nCoordinates\n'
     '38°12′N\n72°46′E\nBasin countries\nMax. length\n55.8 km (34.7 mi)\nAverage depth\n'
     '201.8 m (662 ft)\nMax. depth\n505 m (1,657 ft)\nSurface elevation\n3,263 m (10,705 ft)\n'
     'Sarez Lake is a lake in the Rushon District of Tajikistan.\n'
     'It was formed in 1911 when an earthquake triggered a landslide.'),
    (NO_INFOBOX_PAGE, 'Plain article\nYear\n1911\nThis page has a table but no infobox at all.'),
    (SIMPLE_MAIN_PAGE, 'Some page content about the topic.'),
    (LINK_PAGE, 'X\nAlpha'),
    (TITLE_PAGE, 'Heading\nBody text here.'),
    (HEADING_PAGE, 'Test Page\nContent\nLink'),
    (EMPTY_PAGE, ''),
]


# --- extract_infobox_block -----------------------------------------------------------------------
def test_infobox_block_renders_label_value_lines():
    """Every th/td row becomes exactly one 'Label: Value' line — the delimiter the flattened page
    lacks. Neighbouring fields (the diagnosed confusion) are now unambiguously separated."""
    block = extract_infobox_block(WIKI_INFOBOX_PAGE)
    lines = block.splitlines()
    assert "Max. depth: 505 m (1,657 ft)" in lines
    assert "Average depth: 201.8 m (662 ft)" in lines
    assert "Surface elevation: 3,263 m (10,705 ft)" in lines
    assert "Location: Rushon District, Tajikistan" in lines


def test_infobox_block_collapses_intra_cell_line_breaks():
    """A cell's own <br> fragments collapse onto ONE line (cell-level get_text(' ')), unlike
    clean_operation's page-wide separator='\\n'."""
    assert "Coordinates: 38°12′N 72°46′E" in extract_infobox_block(WIKI_INFOBOX_PAGE).splitlines()


def test_infobox_block_keeps_caption_and_headerless_rows():
    lines = extract_infobox_block(WIKI_INFOBOX_PAGE).splitlines()
    assert lines[0] == "Sarez Lake"           # the infobox <caption> titles the block
    assert "Basin countries" in lines         # a th-only section header stays as a bare line
    assert "A photo of the lake" in lines     # a td-only caption row stays as a bare line


def test_infobox_block_is_empty_without_an_infobox():
    assert extract_infobox_block(NO_INFOBOX_PAGE) == ""
    assert extract_infobox_block(SIMPLE_MAIN_PAGE) == ""
    assert extract_infobox_block("") == ""


def test_infobox_block_matches_multi_valued_and_suffixed_classes():
    """'infobox vcard' (list-valued class) and 'infobox_v2' both count as infoboxes."""
    for cls in ("infobox", "infobox vcard", "infobox_v2 biography"):
        html = f'<html><body><table class="{cls}"><tr><th>Born</th><td>1911</td></tr></table></body></html>'
        assert extract_infobox_block(html) == "Born: 1911"


# --- clean_operation regression --------------------------------------------------------------
@pytest.mark.parametrize("html,expected", PRE_CHANGE_GOLDENS)
def test_clean_operation_default_is_byte_identical_to_pre_change(html, expected):
    """MUST-NOT-BREAK: omitting the new parameter (and passing it as False) reproduces the
    pre-change output byte-for-byte for every caller."""
    assert clean_operation(html) == expected
    assert clean_operation(html, prepend_infobox=False) == expected


def test_clean_operation_prepends_block_only_when_asked():
    plain = clean_operation(WIKI_INFOBOX_PAGE)
    with_block = clean_operation(WIKI_INFOBOX_PAGE, prepend_infobox=True)
    assert with_block == f"{extract_infobox_block(WIKI_INFOBOX_PAGE)}\n\n{plain}"
    assert "Max. depth: 505 m (1,657 ft)" in with_block      # the K:V line the flat text lacks
    assert "Max. depth: 505" not in plain                    # ... and it is NOT in the default path
    assert with_block.endswith(plain)                        # additive: the flat text is untouched


def test_clean_operation_prepend_is_a_noop_without_an_infobox():
    """No infobox -> nothing to prepend -> byte-identical to the default path even when enabled."""
    for html in (NO_INFOBOX_PAGE, SIMPLE_MAIN_PAGE, EMPTY_PAGE):
        assert clean_operation(html, prepend_infobox=True) == clean_operation(html)
