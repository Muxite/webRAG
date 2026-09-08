"""Structured infobox rows -> :class:`~agent.app.quantity_index.QuantityRef` entries, from HTML.

``quantity_index.build_index`` reads quantities out of FLATTENED page text, where a Wikipedia
infobox row arrives as three or four short lines (``"Basin size\\n795,000\\nkm\\n2"``) and the
label/value/unit boundary has to be re-inferred from line shape. This module reads the SAME rows
before that flattening happens, straight off the ``<table class="infobox">`` markup, where the
boundary is explicit: the ``<th>`` is the label, the ``<td>`` is the value. Nothing here is
model-specific or host-specific -- it is pure HTML in, entries out, with no I/O and no model call.

Offset story (read this before wiring the entries anywhere):

* :func:`infobox_text` renders the first infobox as ``Label: Value`` lines -- the value text is
  the cell's own ``get_text(" ", strip=True)`` after superscript exponents are folded in.
* :func:`infobox_quantities` builds its entries FROM THAT RENDERED TEXT, so every entry's
  ``start``/``end`` indexes ``infobox_text(html)`` directly (offset 0). A host that registers a
  page as ``infobox_text(html) + "\\n" + <cleaned body>`` can hand these entries over untouched,
  and ``LedgerToolkit``'s literal-operand path, which locates values against the registered text,
  agrees with them character for character. Both functions run the same rendering, so they are
  consistent by construction (:func:`infobox_rows` is the single source both read).

Two HTML-level traps ``get_text`` alone gets wrong, closed here before any text is extracted:
``<sup>2</sup>`` / ``<sup>3</sup>`` exponents are folded to ``²`` / ``³`` (so ``km<sup>2</sup>``
reads ``km²``, which :func:`~agent.app.testing.evidence_graph.canonical_unit` maps to ``km2``,
instead of the ``"km 2"`` shape the flattener produces), and ``<sup class="reference">[1]</sup>``
citation markers are dropped (``"4,909 km[1]"`` would otherwise refuse to parse).

Validity is decided by exactly the rules ``quantity_index`` already uses -- reused, not
re-implemented: :func:`~agent.app.testing.evidence_graph.parse_quantity` is the only grammar,
:func:`~agent.app.quantity_index._accept` the only unit gate, and the bare-count label whitelist
(``Capacity`` / ``Floor count`` / ...) the only way a unit-less integer gets in. A value cell may
hold several quantities (``"795,000 km² (307,000 mi²)"``; a bulleted ``"• Total 48,430 sq mi"``
list) and each is emitted in document order, the primary first; a dual-unit restatement is
captured as its own entry, never converted (the ``docs/LEDGER_PLAN_2026-09-01.md`` §7 non-goal).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

from agent.app.observation import _is_infobox_class
from agent.app.quantity_index import (_LEADING_NUMBER, QuantityRef, _accept, _entry_unit,
                                      _label_allows_bare_count, _parse_candidate, _unit_allowed)
from agent.app.testing.evidence_graph import Quantity

#: Where a quantity may START inside a value cell: ``_LEADING_NUMBER`` (``quantity_index``'s own
#: "number with an optional glued currency prefix") without its anchor, so ``"€533 million"`` is
#: scanned from the ``€`` and the currency rides inside ``value`` exactly as the index stores it.
_NUMBER_START = re.compile(_LEADING_NUMBER.pattern.lstrip("^"), _LEADING_NUMBER.flags)
_INT_ONLY = re.compile(r"^-?\d[\d,]*$")
#: Boundaries between independent quantities inside ONE value cell: bullets, semicolons, the
#: ``" | "`` a cell's ``<br>`` became, a comma that is NOT a digit-group separator (``"8980 ft,
#: about 1.70 mi"`` splits; ``"1,151,000"`` does not), and the approximation/alternative words
#: ``parse_quantity`` refuses as range markers anyway (``"8980 ft, about 1.70 mi (2.74 km)"`` on
#: the Golden Gate Bridge page lost its leading ``8980 ft`` when only the first three were cut).
_SEGMENT_SPLIT = re.compile(
    r"(?: \| |[;•\n]|,(?!\d)|\babout\b|\bapprox(?:\.|imately)?\b|\bor\b|\bto\b)")
_SUP_EXPONENTS = {"2": "²", "3": "³"}


def _fold_superscripts(table) -> None:
    """In place: ``<sup>2</sup>``/``<sup>3</sup>`` -> ``²``/``³``; citation ``<sup>`` dropped.

    A ``<sup>`` whose class marks it as a reference (``"reference"``) or whose text is a bracketed
    marker (``"[1]"``, ``"[note 2]"``) is decomposed. Anything else keeps its text, so a genuine
    superscript we do not recognise still renders rather than vanishing.
    """
    for sup in table.find_all("sup"):
        classes = sup.get("class") or []
        joined = " ".join(classes) if isinstance(classes, list) else str(classes)
        text = sup.get_text("", strip=True)
        if "reference" in joined.lower() or (text.startswith("[") and text.endswith("]")):
            sup.decompose()
        elif text in _SUP_EXPONENTS:
            sup.replace_with(_SUP_EXPONENTS[text])
    for br in table.find_all("br"):
        # A bare "\n" would be swallowed by ``get_text(strip=True)``; the visible bar survives it
        # and is what :func:`_render` shows anyway.
        br.replace_with(" | ")


#: ``get_text(" ")`` puts its separator between EVERY pair of text nodes, so a folded exponent
#: (``"km" + "²"``) and punctuation glued to a link (``"( Metallica ;"``) come out spaced. These
#: undo only that: no letters or digits are touched, so a value stays byte-for-byte as written.
_TIDY_RULES = (
    (re.compile(r"\s+([²³])"), r"\1"),
    (re.compile(r"([²³])\s+(?=/)"), r"\1"),
    (re.compile(r"\(\s+"), "("),
    (re.compile(r"\s+\)"), ")"),
    (re.compile(r"\s+([,;:])"), r"\1"),
)


def _tidy(text: str) -> str:
    """Whitespace-normalised cell text with the separator artefacts above removed."""
    out = " ".join(text.split())
    for pattern, replacement in _TIDY_RULES:
        out = pattern.sub(replacement, out)
    return out


def _first_infobox(html: str):
    soup = BeautifulSoup(html or "", "html.parser")
    return soup.find("table", class_=_is_infobox_class)


def infobox_rows(html: str) -> List[Tuple[str, str]]:
    """The first infobox's rows as ``(label, value)`` pairs, in document order.

    A row is a ``<tr>`` carrying both a ``<th>`` and a ``<td>``; value-only rows (images,
    captions) are skipped because they are not labelled values. A header-only row (a ``<th>``
    with no ``<td>``) that is not the infobox TITLE -- the title is the first header-only row
    seen before any labelled row, or one whose class says ``above``/``title`` -- is a SECTION
    header (a ``<caption>`` counts as the title too), and its text prefixes every following row
    label until the next header (Shanghai
    Tower: ``"Height"`` over ``Architectural`` / ``Tip`` / ``Roof`` gives ``"Height
    Architectural"`` ..., so a slot asking for the height can see the field the sub-row belongs
    to; a table with no section headers is rendered exactly as before). Text is the cell's ``get_text(" ", strip=True)`` after superscript folding,
    with ``<br>`` line breaks rendered as ``" | "`` so a multi-item cell can be split back apart.

    :param html: the page HTML.
    :returns: ``[]`` when the page has no infobox table.
    :raises: nothing.
    """
    table = _first_infobox(html)
    if table is None:
        return []
    _fold_superscripts(table)
    rows: List[Tuple[str, str]] = []
    section = ""
    # A table whose title is a <caption> has no positional title row: its first header-only row
    # is already a section header (a state infobox's "Area").
    title_seen = table.find("caption") is not None
    for row in table.find_all("tr"):
        header, value = row.find("th"), row.find("td")
        if header is None:
            continue
        if value is None:
            heading = _tidy(header.get_text(" ", strip=True))
            classes = header.get("class") or []
            joined = (" ".join(classes) if isinstance(classes, list) else str(classes)).lower()
            is_title = ("above" in joined or "title" in joined
                        or (not rows and not title_seen and not section))
            title_seen = title_seen or is_title
            if not is_title and heading:
                section = heading
            continue
        own_label = _tidy(header.get_text(" ", strip=True))
        label = f"{section} {own_label}".strip() if section else own_label
        text = " | ".join(part for part in
                          (_tidy(line) for line in value.get_text(" ", strip=True).split("|"))
                          if part)
        if label and text:
            rows.append((label, text))
    return rows


def has_infobox(html: str) -> bool:
    """True when the page carries an infobox table at all (rows or not)."""
    return _first_infobox(html) is not None


def _render(rows: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str, int]]]:
    """``(rendered text, [(label, value, value_start_offset)])`` for ``rows``.

    One ``Label: Value`` line per row (a cell's ``<br>`` breaks are already ``" | "``), so every
    value offset stays inside its own line.
    """
    lines: List[str] = []
    located: List[Tuple[str, str, int]] = []
    cursor = 0
    for label, value in rows:
        flat = value
        line = f"{label}: {flat}"
        located.append((label, flat, cursor + len(label) + 2))
        lines.append(line)
        cursor += len(line) + 1
    return "\n".join(lines), located


def infobox_text(html: str) -> str:
    """The first infobox rendered as ``Label: Value`` lines, one row per line.

    This is the text :func:`infobox_quantities`' offsets index; register exactly this string (or
    this string followed by more text) for the entries to locate.

    :returns: ``""`` when there is no infobox or it has no labelled rows.
    """
    return _render(infobox_rows(html))[0]


def _admissible(quantity: Quantity) -> bool:
    """``quantity_index``'s unit gate, plus: a literal unit, when present, must be whitelisted even
    when a currency would have admitted the value on its own (``"€593 million in 2021 euros"``
    parses with unit ``"in 2021 euros"`` -- a real number on the page, but not one whose unit
    string should be stored as if it were a dimension)."""
    if not _accept(quantity):
        return False
    return not quantity.unit or _unit_allowed(quantity.unit)


def _parse_span(text: str) -> Optional[Quantity]:
    """``parse_quantity(text)`` when it fully parses and passes :func:`_admissible`, else
    ``None``."""
    parsed = _parse_candidate(text)
    return parsed if parsed.ok and _admissible(parsed) else None


def _chain(quantity: Quantity, span: str, span_start: int, label: str) -> List[QuantityRef]:
    """The primary quantity and every restatement in its chain, offsets located inside ``span``.

    ``value`` is what ``quantity_index._LEADING_NUMBER`` matches off the parsed ``source_text``
    (a glued currency prefix included), ``unit`` what ``quantity_index._entry_unit`` derives --
    the same two rules the flattened-text index applies, so an entry from here and an entry from
    ``build_index`` over the same row are indistinguishable downstream.
    """
    out: List[QuantityRef] = []
    cursor = 0
    current: Optional[Quantity] = quantity
    candidate = span
    while current is not None:
        number = _LEADING_NUMBER.match(current.source_text.strip())
        if number is None:
            break
        value = number.group(0)
        local = span.find(value, cursor)
        if local == -1:
            # A suffix-currency parse re-ordered the text ("533 million EUR" -> "EUR 533
            # million"): locate the bare number instead, as the index does.
            bare = re.sub(r"^\D+", "", value)
            local = span.find(bare, cursor) if bare else -1
            if local == -1:
                break
            value = bare
        cursor = local + len(value)
        if _admissible(current):
            out.append(QuantityRef(label=label, value=value,
                                   unit=_entry_unit(current, candidate, current.scale_name or ""),
                                   start=span_start + local, end=span_start + local + len(value),
                                   source="infobox", currency=current.currency,
                                   scale=current.scale_name))
        current = current.restatement
        candidate = current.source_text if current is not None else ""
    return out


def _quantities_in(value: str, value_start: int, label: str) -> List[QuantityRef]:
    """Every quantity in one rendered value string, in document order.

    Splits the cell into independent segments (bullets, ``;``, the ``" | "`` line joins), then
    scans each segment number by number: the span from a number to the segment's end is tried
    first (so a trailing ``"(3,050 mi)"`` restatement is kept), then progressively shortened to
    the start of the next number, then to before its first ``"("``; each shape is also tried with
    a dangling close-paren peeled, for a ``<br>``-split restatement (``"(90 m)"`` on its own
    line). Numbers a successful span already consumed are skipped. A unit-less integer is admitted only under a bare-count label,
    with unit ``"count"``, exactly as :func:`~agent.app.quantity_index.build_index` does.
    """
    out: List[QuantityRef] = []
    segment_start = 0
    for segment in _SEGMENT_SPLIT.split(value):
        seg_start = value.find(segment, segment_start)
        segment_start = seg_start + len(segment)
        numbers = list(_NUMBER_START.finditer(segment))
        consumed_to = -1
        for index, match in enumerate(numbers):
            if match.start() < consumed_to:
                continue
            next_start = numbers[index + 1].start() if index + 1 < len(numbers) else len(segment)
            whole = segment[match.start():]
            head = segment[match.start():next_start]
            candidates = [whole.rstrip(" ,.:"), whole.rstrip(" ,.:)"),
                          head.rstrip(" ,.:("), head.rstrip(" ,.:()")]
            if "(" in head:
                candidates.append(head[:head.index("(")].rstrip(" ,.:"))
            found = None
            for candidate in candidates:
                parsed = _parse_span(candidate)
                if parsed is not None:
                    found = (parsed, candidate)
                    break
            if found is None:
                bare = segment[match.start():next_start].strip(" ,.:(")
                if _INT_ONLY.fullmatch(bare) and _label_allows_bare_count(label):
                    start = value_start + seg_start + match.start()
                    out.append(QuantityRef(label=label, value=bare, unit="count",
                                           start=start, end=start + len(bare), source="infobox"))
                    consumed_to = match.start() + len(bare)
                continue
            parsed, candidate = found
            span_start = value_start + seg_start + match.start()
            out.extend(_chain(parsed, candidate, span_start, label))
            consumed_to = match.start() + len(candidate)
    return out


def infobox_quantities(html: str) -> List[QuantityRef]:
    """Every quantity in the first infobox's labelled rows, as index entries.

    :param html: the page HTML.
    :returns: entries in document order (row by row, primary before restatement), each with
        ``label`` = the row's ``<th>`` text, ``value``/``unit`` byte-for-byte as written, and
        ``start``/``end`` indexing :func:`infobox_text` of the same HTML. ``[]`` when the page has
        no infobox. Never raises.
    """
    try:
        _text, located = _render(infobox_rows(html))
        out: List[QuantityRef] = []
        for label, flat, value_start in located:
            out.extend(_quantities_in(flat, value_start, label))
        return out
    except Exception:  # noqa: BLE001 -- a parser that raises would take a host down
        return []
