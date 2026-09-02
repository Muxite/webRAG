"""A host-neutral index of literal quantities on a page, for reference-not-retype grounding.

Motivation (measured, task 221): a model derived three ratios, every operand correctly located on
a real page, and still scored 0.16 by comparing feet-per-floor against metres-per-floor. The unit
guard never fired because the model passed **bare numbers** — the source nodes carried ``unit=''``,
so there was nothing to check. A verification layer fed by the agent can be silently disabled by
the agent simply omitting the metadata it is checked against.

The fix direction here is not a better checker, it is a different INPUT shape: let the model
reference a quantity the system already extracted (``q3``) instead of retyping a number. An id
carries its unit and cannot be mistyped, the way a bare ``1,642`` can be typed without ``m``.

This module is pure and host-neutral: no framework import, no model call, no I/O. It builds the
index from raw page text and renders it into a short prompt block; a later lane wires
:func:`build_index` / :func:`render_index` / :func:`lookup` into the hosts that assemble prompts
and resolve model output.

Two extractors, because neither alone is enough (measured on the frozen ``numeric22`` corpus
against the values models actually typed on run ``ledgerfinal01``: infobox-only covers 40%,
prose-only 28%, combined 68%):

* **infobox** — the corpus pages are flattened Wikipedia infoboxes with a regular *line* shape:
  a label spread over one or two short lines, then a bare-number value line, then a unit line that
  may carry a parenthetical dual-unit restatement whose own value/unit straddle a further newline
  (``"Max.\\ndepth\\n1,642\\nm (5,387\\nft)"``). :func:`_scan_infobox` walks lines, not regex over
  the whole blob, because the label/value/unit boundary IS the line boundary here.
* **prose** — an inline ``NUMBER WORD`` pair on one line (``"419.7 metres"``), found by
  :func:`_scan_prose`. Deliberately restricted to whitespace *excluding* the newline: a number and
  its unit separated by a line break is the infobox shape, not prose, and letting prose slurp
  across lines would double-count (and mis-attribute) infobox rows instead of just deferring to
  them in :func:`build_index`'s de-duplication.

Reused, not reimplemented, from ``agent.app.testing.evidence_graph`` (see that module's docstring
for the traps it already closed): :func:`~agent.app.testing.evidence_graph.parse_quantity` is the
sole validity check — a candidate span is a quantity if and only if ``parse_quantity`` accounts
for every character of it (``.ok``); nothing here re-derives a number grammar.
:func:`~agent.app.testing.evidence_graph.normalize_for_match` is used only to key the unit
whitelist and the de-dup set, never to alter a stored ``value``/``unit``, which are always kept
byte-for-byte as written.

The unit whitelist (:data:`_UNIT_WHITELIST`) is REQUIRED, not decorative. ``parse_quantity``'s
grammar is permissive about what counts as a unit TAIL (any run of letters, so a stray label word
like ``"Motihari"`` or ``"Established"`` parses as a syntactically valid "unit" with no residue);
the whitelist is what turns that permissive grammar into an actual quantity detector. Confirmed
against this corpus: without it, ``August\\n1943`` and ``Ward(s)\\n46\\nEstablished`` would both be
accepted as unit-bearing quantities. Dropped, deliberately: anything not in the whitelist,
including calendar months, ordinal suffixes (``"1st"``), coordinate degree/minute/second marks,
and any label word that happens to be letters-only. See :data:`_UNIT_WHITELIST` for exactly what
is covered (length/area/volume/mass/speed/temperature/time/percent, in both abbreviated and
spelled-out English forms).

Non-goal (``docs/LEDGER_PLAN_2026-09-01.md`` §7): no unit conversion, ever. A dual-unit
restatement (``"1,642 m (5,387 ft)"``) is parsed only far enough to discard the parenthetical and
keep the primary ``"m"`` — the ``"5,387 ft"`` half is never surfaced, and nothing here ever turns
one unit into another.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.app.testing.evidence_graph import Quantity as _ParsedQuantity
from agent.app.testing.evidence_graph import normalize_for_match, parse_quantity


@dataclass(frozen=True)
class QuantityRef:
    """One quantity the index extracted, addressable by a stable ``q<N>`` id.

    :param label: the infobox field label (``"Max. depth"``), or ``""`` when ``source`` is
        ``"prose"`` — prose quantities are not associated with a field name.
    :param value: the number exactly as written on the page (``"1,642"``); never normalized.
    :param unit: the unit exactly as written (``"m"``), or the scale word (``"million"``) when the
        value carried a scale but no literal unit token; ``""`` when neither was found. Never
        converted, never a restatement's unit.
    :param start: start offset of ``value`` in the RAW page text passed to :func:`build_index`.
    :param end: end offset (exclusive) of ``value`` in the same RAW page text.
    :param source: ``"infobox"`` or ``"prose"`` — which extractor found it.
    """

    label: str
    value: str
    unit: str
    start: int
    end: int
    source: str


#: Units this index will recognize, normalized (lowercased, dash-unified) the way
#: ``normalize_for_match`` normalizes everything else here. Both abbreviated and spelled-out
#: forms are listed because the two extractors see both (an infobox unit line reads ``"m"``, a
#: prose sentence reads ``"metres"``). Deliberately excludes bare ``"°"`` (coordinate marks parse
#: as unit-shaped but are not quantities), every calendar/ordinal token that a crude prototype
#: mistook for a unit (``August``, ``March``, ``May``, ``th``), and the bare abbreviation ``"in"``
#: for inches — it collides with the preposition ("... 1903 in Motihari" reads ``in`` as inches on
#: a naive whitelist); only the spelled-out ``inch`` / ``inches`` are recognized. See the module
#: docstring.
_UNIT_WORDS = [
    # length
    "m", "km", "cm", "mm", "mi", "ft", "yd", "nmi",
    "metre", "metres", "meter", "meters",
    "kilometre", "kilometres", "kilometer", "kilometers",
    "centimetre", "centimetres", "centimeter", "centimeters",
    "millimetre", "millimetres", "millimeter", "millimeters",
    "mile", "miles", "foot", "feet", "inch", "inches", "yard", "yards",
    "nautical mile", "nautical miles",
    # area (including the newline-split superscript shape the infobox flattening produces:
    # "km\n2" joins to "km 2")
    "km2", "km²", "m2", "m²", "cm2", "cm²", "mi2", "mi²",
    "km 2", "m 2", "cm 2", "mi 2",
    "sq mi", "sqmi", "ha", "hectare", "hectares", "acre", "acres",
    "square kilometres", "square kilometers", "square miles",
    "square metres", "square meters", "square feet", "square foot",
    # volume
    "km3", "km³", "m3", "m³", "cm3", "cm³", "mi3", "mi³",
    "km 3", "m 3", "cm 3", "mi 3",
    "l", "litre", "litres", "liter", "liters", "ml",
    "gal", "gallon", "gallons", "cu mi", "cubic miles",
    "cubic kilometres", "cubic kilometers",
    # mass
    "kg", "kilogram", "kilograms", "g", "gram", "grams",
    "lb", "lbs", "pound", "pounds",
    "t", "tonne", "tonnes", "ton", "tons", "oz", "ounce", "ounces",
    # speed
    "km/h", "kmh", "mph", "m/s", "kn", "knot", "knots",
    # power / energy
    "w", "kw", "mw", "gw", "tw", "wh", "kwh", "mwh", "gwh", "twh", "hp",
    # temperature
    "°c", "°f",
    # time / duration
    "s", "sec", "secs", "seconds", "min", "mins", "minutes",
    "h", "hr", "hrs", "hour", "hours",
    "yr", "yrs", "year", "years", "day", "days", "week", "weeks", "month", "months",
    # count / percent
    "%", "people", "residents", "inhabitants",
]
_UNIT_WHITELIST = frozenset(normalize_for_match(word) for word in _UNIT_WORDS)


def _unit_allowed(unit: str) -> bool:
    """True when ``unit`` (as :func:`~evidence_graph.parse_quantity` split it out) is whitelisted."""
    return bool(unit) and normalize_for_match(unit) in _UNIT_WHITELIST


def _accept(quantity: _ParsedQuantity) -> bool:
    """True when a fully-parsed (:attr:`Quantity.ok`) candidate is an actual quantity, not a
    label word that happened to be letters-only. Three ways in: a whitelisted literal unit, a
    recognized currency (``parse_quantity`` already restricts currency to its fixed code table, so
    it needs no separate whitelist), or a scale word (``million``, ``crore``, ...) with no literal
    unit token at all — the ``"237.8 million"`` shape.
    """
    if quantity.unit and _unit_allowed(quantity.unit):
        return True
    if quantity.currency:
        return True
    if quantity.scale_name and not quantity.unit:
        return True
    return False


#: A line that is nothing but a (possibly negative, possibly decimal, possibly comma-grouped)
#: number. Deliberately simple: this only decides which lines are VALUE-line CANDIDATES.
#: Correctness of the number itself is `parse_quantity`'s job, not this regex's.
_NUM_ONLY_LINE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")

#: How many lines past a value line to search for its unit before giving up. Generous enough to
#: cross a dual-unit restatement split by a newline (``"m (5,387\nft)"`` is 2 lines), small enough
#: that a genuinely unit-less row (a count, a rank) does not go hunting through the next row.
_FORWARD_UNIT_LINES = 6


def _line_spans(text: str) -> List[Tuple[str, int, int]]:
    """``text`` split on ``"\\n"`` as ``(line, start, end)`` triples with RAW offsets."""
    spans: List[Tuple[str, int, int]] = []
    start = 0
    for part in text.split("\n"):
        end = start + len(part)
        spans.append((part, start, end))
        start = end + 1
    return spans


#: Ceiling on a candidate label LINE's length. Real infobox labels ("Max.", "Elevation",
#: "Population density") are short; a line this long is a prose sentence that happens to precede
#: a number the flattener wrapped onto its own line, not a field label.
_MAX_LABEL_LINE_CHARS = 40

_PUNCTUATION_ONLY = re.compile(r"^[\W_]+$")


def _looks_like_label(candidate: str) -> bool:
    """False for anything that is not a plausible infobox label LINE (see the caps above)."""
    if not candidate or len(candidate) > _MAX_LABEL_LINE_CHARS:
        return False
    if _NUM_ONLY_LINE.fullmatch(candidate) or candidate.startswith("("):
        return False
    if _PUNCTUATION_ONLY.fullmatch(candidate):
        return False
    # A line that is itself a whitelisted unit is the PREVIOUS row's unit, not a label — the
    # "165\nkm\n45\nkm\n72\nkm" shape (a comma list of measurements) would otherwise donate
    # "km" as a bogus label for the next measurement in the list.
    if normalize_for_match(candidate) in _UNIT_WHITELIST:
        return False
    return True


def _infer_label(lines: List[Tuple[str, int, int]], index: int) -> str:
    """The field label for the value line at ``index``, or ``""`` when none is inferable.

    Takes the immediately preceding non-empty, non-numeric, non-continuation line as the label,
    and prepends ONE further line only when that further line ends in ``"."`` (the ``"Max."`` /
    ``"Avg."`` abbreviation-continuation shape actually seen in this corpus). Anything more
    elaborate risks grabbing an unrelated bullet or the previous row's own label.
    """
    if index == 0:
        return ""
    immediate = lines[index - 1][0].strip()
    if not _looks_like_label(immediate):
        return ""
    parts = [immediate]
    if index >= 2:
        earlier = lines[index - 2][0].strip()
        if earlier.endswith(".") and _looks_like_label(earlier):
            parts.insert(0, earlier)
    return " ".join(parts)


def _forward_unit(lines: List[Tuple[str, int, int]], index: int,
                   value_text: str) -> Optional[str]:
    """The unit text for the value line at ``index``, or ``None`` when no valid unit follows.

    Grows a candidate unit region one line at a time (joined with single spaces) and re-parses
    ``value_text + " " + candidate`` on every growth step via :func:`parse_quantity`, so a
    restatement split across a newline (``"m (5,387\\nft)"``) is still read as one span. Stops at
    the first line-join that parses cleanly (:attr:`Quantity.ok`) AND passes :func:`_accept` —
    NOT merely at the first clean parse, since a garbage word like ``"Motihari"`` also parses
    cleanly (letters-only tails are syntactically valid units to ``parse_quantity``) but is not
    whitelisted and must not stop the search.
    """
    parts: List[str] = []
    stop = min(len(lines), index + 1 + _FORWARD_UNIT_LINES)
    for k in range(index + 1, stop):
        segment = lines[k][0].strip()
        if not segment:
            break
        parts.append(segment)
        candidate = " ".join(parts)
        parsed = parse_quantity(f"{value_text} {candidate}")
        if parsed.ok and _accept(parsed):
            return parsed.unit if parsed.unit else candidate
    return None


#: The leading number of a line that is NOT purely numeric — the ``"13,860 MW"`` shape, where the
#: flattener kept a value and its unit on one row instead of splitting them onto separate lines.
_LEADING_NUMBER = re.compile(r"^-?\d[\d,]*(?:\.\d+)?")


def _scan_infobox(text: str) -> List[QuantityRef]:
    """Every ``label / value / unit`` infobox row in ``text`` (see the module docstring)."""
    lines = _line_spans(text)
    entries: List[QuantityRef] = []
    for index, (raw_line, line_start, _line_end) in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _NUM_ONLY_LINE.fullmatch(stripped):
            value_text = stripped
            unit = _forward_unit(lines, index, stripped)
        else:
            # A row whose value and unit share one line ("13,860 MW") rather than being split
            # across two — only accepted when the WHOLE line is exactly that quantity, so a
            # prose sentence that happens to start with a number is not mistaken for a row.
            number = _LEADING_NUMBER.match(stripped)
            parsed = parse_quantity(stripped) if number else None
            if not (number and parsed is not None and parsed.ok and _accept(parsed)):
                continue
            value_text = number.group(0)
            unit = parsed.unit if parsed.unit else stripped[len(value_text):].strip()
        if unit is None:
            continue
        leading_ws = len(raw_line) - len(raw_line.lstrip())
        value_start = line_start + leading_ws
        entries.append(QuantityRef(
            label=_infer_label(lines, index),
            value=value_text,
            unit=unit,
            start=value_start,
            end=value_start + len(value_text),
            source="infobox",
        ))
    return entries


#: An inline ``NUMBER WORD[ WORD]`` pair on ONE line — the prose shape. The connector is
#: whitespace EXCLUDING the newline, or a bare hyphen (the ``"515-kilometre"`` compound-adjective
#: idiom), on purpose: a number and unit split across a newline with no hyphen is the infobox
#: shape (:func:`_scan_infobox`'s job), and letting this regex cross a bare newline would have it
#: rediscover every infobox row too, defeating the source-priority de-duplication in
#: :func:`build_index`. The lookbehind keeps this from re-matching the tail of a longer number
#: (``"1,642"`` must not also offer ``"642"`` as a second candidate).
_PROSE_QUANTITY = re.compile(
    r"(?<![\d,.])(?P<num>-?\d[\d,]*(?:\.\d+)?)(?:[^\S\n]+|-)"
    r"(?P<unit>[A-Za-z°%][A-Za-z°%./-]*(?:[^\S\n]+[A-Za-z]+)?)"
)


def _scan_prose(text: str) -> List[QuantityRef]:
    """Every inline ``NUMBER unit`` mention in ``text`` (see the module docstring)."""
    entries: List[QuantityRef] = []
    for match in _PROSE_QUANTITY.finditer(text):
        value_text = match.group("num")
        words = match.group("unit").split()
        unit: Optional[str] = None
        # Try the two-word candidate first ("square kilometres"), then fall back to just the
        # first word, so a genuine unit is not missed because a stray following word rode along.
        for take in range(len(words), 0, -1):
            # Strip trailing sentence punctuation ("meters." at a full stop, "feet," before a
            # comma) that the regex's permissive unit-word class swept in — a real unit token
            # never legitimately ends in one, and leaving it on defeats the whitelist lookup.
            candidate = " ".join(words[:take]).rstrip(".,;:)")
            if not candidate:
                continue
            parsed = parse_quantity(f"{value_text} {candidate}")
            if parsed.ok and _accept(parsed):
                unit = parsed.unit if parsed.unit else candidate
                break
        if unit is None:
            continue
        start = match.start("num")
        entries.append(QuantityRef(
            label="",
            value=value_text,
            unit=unit,
            start=start,
            end=start + len(value_text),
            source="prose",
        ))
    return entries


def build_index(page_text: Optional[str], *, limit: int = 40) -> List[QuantityRef]:
    """Every quantity :func:`_scan_infobox` / :func:`_scan_prose` can extract from ``page_text``.

    Deterministic and stable: pure sequential scans, no dict/set governs ORDER (a ``set`` is used
    only for de-dup membership testing). Infobox entries come first, in document order, then prose
    entries in document order — this is both the display order and how "prefer infobox" is
    enforced, since a later duplicate (same normalized value + unit) is dropped in
    :func:`build_index` rather than in either scanner. Capped at ``limit`` because the render goes
    into a weak model's prompt.

    A dual-unit restatement (``"1,642 m (5,387 ft)"``) surfaces only its PRIMARY half here —
    ``parse_quantity`` reads the ``"(5,387 ft)"`` parenthetical and discards it into
    ``.restatement`` precisely so it never contaminates the outer ``unit``, and this index does
    not mine the discarded half out as a second entry either: doing so would put a
    unit-conversion pair (the same physical quantity in two units) one reference-hop apart in the
    prompt, which is a standing invitation for a weak model to grab whichever one is more
    convenient — silently defeating the "no conversion" guarantee at the call site even though no
    line of code here ever divides by 3.281. See the module docstring's "what is dropped" note.

    :param page_text: the raw page text, or ``None``/``""``.
    :param limit: maximum entries returned.
    :returns: a list of :class:`QuantityRef`, possibly empty. Never raises on absence — a page
        with nothing extractable returns ``[]``, not a sentinel.
    """
    text = str(page_text or "")
    if not text:
        return []
    seen = set()
    out: List[QuantityRef] = []
    for entry in _scan_infobox(text) + _scan_prose(text):
        key = (normalize_for_match(entry.value), normalize_for_match(entry.unit))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out[:limit]


def render_index(entries: List[QuantityRef], *, max_chars: int = 1200) -> str:
    """``entries`` rendered as a short ``q<N>: label = value unit`` block for a model prompt.

    :param entries: the output of :func:`build_index`, in the order to render (its ``q<N>`` ids
        are exactly this list's 1-based position, so callers must not reorder before rendering).
    :param max_chars: soft cap on the rendered length — whole lines are dropped from the end
        rather than truncating one mid-line, except when even the FIRST line alone exceeds
        ``max_chars``, which is hard-truncated so the function still returns something bounded.
    :returns: the rendered block, or ``""`` for an empty ``entries`` — never a header over
        nothing.
    """
    if not entries:
        return ""
    lines: List[str] = []
    length = 0
    for position, entry in enumerate(entries, start=1):
        label_part = f"{entry.label} = " if entry.label else ""
        unit_part = f" {entry.unit}" if entry.unit else ""
        line = f"q{position}: {label_part}{entry.value}{unit_part}"
        extra = len(line) + (1 if lines else 0)
        if lines and length + extra > max_chars:
            break
        lines.append(line)
        length += extra
    rendered = "\n".join(lines)
    return rendered[:max_chars] if len(rendered) > max_chars else rendered


#: ``"q3"`` / ``"Q3"`` / ``"q3."`` / bare ``"3"`` -> ``3``. Tolerant of case and a trailing dot
#: because a weak model asked to cite ``q3`` will sometimes write any of these.
_REF_PATTERN = re.compile(r"^\s*q?\s*0*([1-9]\d*)\s*\.?\s*$", re.IGNORECASE)


def lookup(entries: List[QuantityRef], ref: str) -> Optional[QuantityRef]:
    """The entry ``ref`` (``"q3"``, tolerant of ``"Q3"`` / ``"q3."``) names, or ``None``.

    :param entries: the same list :func:`render_index` rendered ids against.
    :param ref: the model's reference string.
    :returns: the matching :class:`QuantityRef`, or ``None`` when ``ref`` does not parse as a
        reference or names an out-of-range position.
    """
    if not entries:
        return None
    match = _REF_PATTERN.match(str(ref or ""))
    if not match:
        return None
    position = int(match.group(1))
    if 1 <= position <= len(entries):
        return entries[position - 1]
    return None
