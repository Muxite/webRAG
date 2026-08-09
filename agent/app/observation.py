import re

from bs4 import BeautifulSoup


def _is_infobox_class(value) -> bool:
    """True for a ``class`` attribute that marks an infobox table ("infobox", "infobox vcard",
    "infobox_v2", ...). BeautifulSoup hands multi-valued classes over as a list."""
    if not value:
        return False
    joined = " ".join(value) if isinstance(value, list) else str(value)
    return "infobox" in joined.lower()


def extract_infobox_block(html: str) -> str:
    """Best-effort "Label: Value" rendering of the page's FIRST infobox table.

    ``clean_operation`` extracts the page with a single page-wide ``get_text(separator="\\n")``,
    which drops every infobox cell onto its own line with no delimiter — a label and its value end
    up indistinguishable from the neighbouring field's ("Elevation / 7,439 / m / Prominence /
    4,148 / m"), and a weak model asked an under-specified question confidently reads off the
    neighbour. This renders each row as ``Label: Value`` using the CELL's own
    ``get_text(" ", strip=True)``, so a cell's internal ``<br>`` fragments collapse onto one line.

    Returns ``""`` when the page has no infobox table (every non-Wikipedia page, mostly) — callers
    treat that as "nothing to prepend".
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=_is_infobox_class)
    if table is None:
        return ""
    lines = []
    caption = table.find("caption")
    if caption is not None:
        caption_text = caption.get_text(" ", strip=True)
        if caption_text:
            lines.append(caption_text)          # the infobox title ("Sarez Lake")
    for row in table.find_all("tr"):
        header, value = row.find("th"), row.find("td")
        if header is not None and value is not None:
            label_text = header.get_text(" ", strip=True)
            value_text = value.get_text(" ", strip=True)
            if label_text and value_text:
                lines.append(f"{label_text}: {value_text}")
                continue
            cell_text = label_text or value_text
        else:
            cell = header if header is not None else value
            cell_text = cell.get_text(" ", strip=True) if cell is not None else ""
        if cell_text:                      # title / section-heading / caption row
            lines.append(cell_text)
    return "\n".join(lines)


def clean_operation(html: str, prepend_infobox: bool = False) -> str:
    """
    Extract simplified main text content from the provided HTML.

    Strips navigation, sidebars, footers, and other boilerplate to focus on
    the main article/page content. Preserves links inline but excludes images.

    Aggressively removes Wikipedia-specific UI elements (skip links, sidebar
    toggles, table of contents, edit sections, reference lists) and generic
    site chrome (cookie banners, headers, footers).

    ``prepend_infobox`` (default ``False`` = byte-identical to the historical output) prefixes the
    page's infobox rendered as ``Label: Value`` lines (:func:`extract_infobox_block`) ahead of the
    cleaned text, from the SAME already-fetched HTML — no extra round-trip. Opt-in because the
    flattened body text is left untouched underneath it, so the block is purely additive context.
    """
    infobox_block = extract_infobox_block(html) if prepend_infobox else ""
    soup = BeautifulSoup(html, "html.parser")

    # ── Phase 1: remove non-content tags ──────────────────────────────
    _STRIP_TAGS = [
        "script", "style", "noscript", "iframe", "svg",
        "nav", "footer", "aside", "header",
        "button", "input", "label", "select", "textarea", "form",
        "img", "figure", "figcaption", "picture", "source", "video", "audio",
    ]
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # ── Phase 2: remove by ARIA role ──────────────────────────────────
    for role in ("navigation", "banner", "search", "complementary", "contentinfo"):
        for tag in soup.find_all(attrs={"role": role}):
            tag.decompose()

    # ── Phase 3: remove by element ID ─────────────────────────────────
    _REMOVE_IDS = {
        # Wikipedia (Vector 2022 + legacy)
        "mw-navigation", "mw-head", "mw-panel", "mw-panel-toc",
        "mw-sidebar-button", "mw-sidebar-checkbox",
        "p-navigation", "p-search", "p-interaction", "p-tb", "p-lang",
        "p-personal", "p-cactions", "p-views", "p-namespaces",
        "footer", "catlinks", "siteSub", "jump-to-nav",
        "contentSub", "contentSub2", "mw-head-base", "mw-page-base",
        "toc", "vector-toc", "mw-toc",
        "mw-fr-revisiontag", "mw-indicator-mw-helplink",
        # Generic
        "cookie-notice", "cookie-banner", "gdpr-banner",
    }
    for elem_id in _REMOVE_IDS:
        tag = soup.find(id=elem_id)
        if tag:
            tag.decompose()

    # Also remove elements whose id starts with known prefixes
    for tag in soup.find_all(id=re.compile(r"^(vector-|mw-sidebar|p-)")):
        tag.decompose()

    # ── Phase 4: remove by CSS class ──────────────────────────────────
    _REMOVE_CLASSES = {
        # Wikipedia navigation / boilerplate
        "sidebar", "navbox", "navbar", "navigation", "nav-links",
        "mw-jump-link", "noprint", "mw-editsection",
        "reference", "reflist", "refbegin", "mw-indicators",
        "toc", "toccolours", "mw-body-header",
        "vector-header", "vector-menu", "vector-column-start",
        "vector-body-before-content", "vector-page-toolbar",
        "mw-footer", "mw-portlet",
        # Wikipedia language / interlanguage links
        "interlanguage-links-list", "interlanguage-link",
        # Wikipedia metadata / hidden elements
        "shortdescription", "mw-empty-elt",
        "mw-authority-control", "catlinks",
        "sistersitebox",  # "Python Programming at Wikibooks" etc.
        # Generic
        "cookie-banner", "site-header", "site-footer",
        "footer", "skip-link", "screen-reader-text",
    }
    for cls in _REMOVE_CLASSES:
        for tag in soup.find_all(class_=cls):
            tag.decompose()

    # ── Phase 5: find the main content area ───────────────────────────
    # For Wikipedia, prefer the narrowest container: .mw-parser-output
    main_content = None

    # Wikipedia: parser output is the actual rendered wikitext
    mw_parser = soup.find(class_="mw-parser-output")
    if mw_parser:
        main_content = mw_parser
    else:
        main_content = (
            soup.find("main")
            or soup.find(id="mw-content-text")
            or soup.find(id="bodyContent")
            or soup.find(id="content")
            or soup.find("article")
            or soup.find(id="main-content")
            or soup.find(class_="main-content")
            or soup.find(role="main")
        )

    target = main_content if main_content else soup

    # ── Phase 6: extract and clean text ───────────────────────────────
    main_text = target.get_text(separator="\n", strip=True)

    # Collapse runs of blank lines to at most one
    main_text = re.sub(r"\n{3,}", "\n\n", main_text)

    # Remove common leftover navigation / UI phrases (line-level)
    _JUNK_LINE_PATTERNS = [
        r"^Jump to content\s*$",
        r"^Main menu\s*$",
        r"^move to sidebar\s*$",
        r"^hide\s*$",
        r"^Toggle.*subsection\s*$",
        r"^Toggle the table of contents\s*$",
        r"^\d+ languages?\s*$",           # "117 languages"
        r"^Edit links\s*$",
        r"^From Wikipedia, the free encyclopedia\s*$",
        r"^Article\s*$",
        r"^Talk\s*$",
        r"^Read\s*$",
        r"^View (source|history)\s*$",
        r"^Tools\s*$",
        r"^Actions\s*$",
        r"^General\s*$",
        r"^Appearance\s*$",
        r"^Donate\s*$",
        r"^Create account\s*$",
        r"^Log in\s*$",
        r"^Personal tools\s*$",
        r"^Contents\s*$",
        r"^Search\s*$",
        r"^Navigation\s*$",
        r"^Contribute\s*$",
        r"^Print/export\s*$",
        r"^In other projects\s*$",
        r"^What links here\s*$",
        r"^Related changes\s*$",
        r"^Upload file\s*$",
        r"^Permanent link\s*$",
        r"^Page information\s*$",
        r"^Cite this page\s*$",
        r"^Get shortened URL\s*$",
        r"^Download QR code\s*$",
        r"^Download as PDF\s*$",
        r"^Printable version\s*$",
        r"^Special pages\s*$",
        r"^Current events\s*$",
        r"^Random article\s*$",
        r"^About Wikipedia\s*$",
        r"^Contact us\s*$",
        r"^Help\s*$",
        r"^Learn to edit\s*$",
        r"^Community portal\s*$",
        r"^Recent changes\s*$",
        r"^Main page\s*$",
    ]
    combined_junk = "|".join(_JUNK_LINE_PATTERNS)
    main_text = re.sub(combined_junk, "", main_text, flags=re.MULTILINE)

    # Collapse any blank lines created by removals
    main_text = re.sub(r"\n{3,}", "\n\n", main_text)
    main_text = main_text.strip()
    if infobox_block:
        return f"{infobox_block}\n\n{main_text}" if main_text else infobox_block
    return main_text
