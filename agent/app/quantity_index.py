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
restatement (``"1,642 m (5,387 ft)"``) is parsed into a primary ``"m"`` quantity and a separate
``"5,387 ft"`` restatement quantity, and :func:`_scan_infobox` indexes BOTH as their own entries —
this is capture (two numbers that are both literally printed on the page), never conversion:
nothing here computes one from the other, each keeps exactly the unit it was written with, and a
``q<N>`` id always resolves to one written value, never a computed alternate.
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
    :param currency: the normalized currency code ``parse_quantity`` read (``"EUR"``), or ``""``.
        A prefix currency (``"€533"``) stays glued inside ``value`` exactly as written, which is
        what lets a consumer re-parsing ``value`` see the currency dimension; a suffix currency
        (``"100 €"``, ``"533 million EUR"``) is carried in ``unit`` as its code (``"EUR"``,
        ``"million EUR"``) because ``parse_quantity`` only reads currency as a PREFIX.
    :param scale: the scale word consumed (``"million"``), or ``""``. Provenance only — it is
        also still present in ``value``/``unit`` text so ``f"{value} {unit}"`` re-parses to the
        full magnitude.
    """

    label: str
    value: str
    unit: str
    start: int
    end: int
    source: str
    currency: str = ""
    scale: str = ""


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


def _collapse_superscript(unit: str) -> str:
    """``"km 2"`` -> ``"km2"``: the newline-split superscript re-joined with no space, so the
    stored unit is the spelling :func:`~evidence_graph.canonical_unit` already understands."""
    return re.sub(r"\s+(?=[23²³]$)", "", unit)


def _unit_allowed(unit: str) -> bool:
    """True when ``unit`` (as :func:`~evidence_graph.parse_quantity` split it out) is whitelisted."""
    return bool(unit) and normalize_for_match(_collapse_superscript(unit)) in _UNIT_WHITELIST


def _accept(quantity: _ParsedQuantity) -> bool:
    """True when a fully-parsed (:attr:`Quantity.ok`) candidate is an actual quantity, not a
    label word that happened to be letters-only. Three ways in: a whitelisted literal unit, a
    recognized currency (``parse_quantity`` already restricts currency to its fixed code table, so
    it needs no separate whitelist), or a scale word (``million``, ``crore``, ...) with no literal
    unit token at all — the ``"237.8 million"`` shape.
    """
    if quantity.unit and _unit_allowed(quantity.unit):
        return True
    if quantity.currency and not quantity.unit:
        # A currency alone is a dimension; a currency dragging a NON-whitelisted "unit" behind
        # it ("€533 to" from "cost €533 to build") is the same label-word trap the whitelist
        # exists for, so it is not admitted here either — the shorter, unit-less candidate is.
        return True
    if quantity.scale_name and not quantity.unit:
        return True
    return False


#: Infobox row LABELS under which a bare (unit-less) integer count is trustworthy enough to
#: index anyway, as its own dimension token ``"count"`` (never ``""`` -- see the module
#: docstring's ambiguity-discipline note: an empty unit would read as dimension-compatible with
#: every OTHER unit-less entry in ``LedgerToolkit``'s diff/sum/quotient compatibility checks,
#: which compare ``canonical_unit(a) == canonical_unit(b)`` and treat two blank strings as equal.
#: ``"count"`` is a real, unmapped token (:func:`~evidence_graph.canonical_unit` passes it through
#: unchanged), so two counts still compare compatible with EACH OTHER -- which is intended, a
#: stadium capacity minus a floor count is nonsense but the toolkit's own KEYSTONE gates cover the
#: rest -- while never colliding with a genuine physical unit. Deliberately narrow: this is the
#: ONLY way an entry with ``unit`` non-whitelisted gets indexed at all; every other label still
#: goes through the unit-mandatory gate in :func:`_accept`. See the module docstring and
#: ``project_wildcard_suppression_load_bearing`` -- ambiguity inflation from over-eager bare-number
#: capture is a known, previously-measured failure mode in this repo.
_COUNT_LABEL_WHITELIST = frozenset({"capacity", "floor count", "floors", "seats", "population"})


def _label_allows_bare_count(label: str) -> bool:
    """True when ``label`` (as :func:`_infer_label` inferred it) licenses a unit-less count entry.

    Matches the whole normalized label first (``"Capacity"``), then falls back to a whole-word
    hit against the whitelist (``"Seating capacity"`` contains the word ``"capacity"``) so a
    slightly longer real-world label still qualifies -- still conservative, since the check is
    against a fixed five-word list, not free containment of arbitrary substrings.
    """
    norm = normalize_for_match(label)
    if not norm:
        return False
    if norm in _COUNT_LABEL_WHITELIST:
        return True
    words = set(norm.split())
    return any(keyword in words or keyword in norm for keyword in _COUNT_LABEL_WHITELIST)


def _bare_count_label(lines: List[Tuple[str, int, int]], index: int) -> Optional[str]:
    """The label text that licenses a bare-count entry for the value line at ``index``, or
    ``None``.

    Checks the immediate preceding label line first (:func:`_label_allows_bare_count`), then --
    because :func:`_infer_label`'s two-line stitch only fires when the earlier line ends in ``"."``
    (the ``"Max."`` abbreviation shape), which a plain two-word label split across lines
    (``"Floor\\ncount"``) does NOT -- also tries the two immediate label lines joined
    (``"Floor" + "count"`` -> ``"Floor count"``), independently of that period rule, since a
    generic two-line infobox label split is common and this whitelist is narrow enough on its own
    to gate admission safely either way.
    """
    if index == 0:
        return None
    immediate = lines[index - 1][0].strip()
    if _label_allows_bare_count(immediate):
        return immediate
    if index >= 2:
        earlier = lines[index - 2][0].strip()
        if _looks_like_label(earlier) and _looks_like_label(immediate):
            combined = f"{earlier} {immediate}"
            if _label_allows_bare_count(combined):
                return combined
    return None


#: A line that is nothing but a (possibly negative, possibly decimal, possibly comma-grouped)
#: number. Deliberately simple: this only decides which lines are VALUE-line CANDIDATES.
#: Correctness of the number itself is `parse_quantity`'s job, not this regex's.
#: Mirrors ``evidence_graph._CURRENCY_PREFIX``'s alternation (kept in step by hand; the parser is
#: still the only authority on what a prefix MEANS -- these gates only decide what may reach it).
_CURRENCY_TOKEN = r"(?:US\$|Rs\.?|INR|USD|GBP|EUR|JPY|[$£€¥₹])"
_NUM_ONLY_LINE = re.compile(rf"^(?:{_CURRENCY_TOKEN}\s*)?-?\d[\d,]*(?:\.\d+)?$", re.IGNORECASE)

#: A currency written AFTER the number (``"100 €"``, ``"533 million EUR"``). ``parse_quantity``
#: reads currency only as a prefix, so :func:`_parse_candidate` moves a trailing token to the
#: front before retrying -- the string is re-ordered for the parser, never for the stored entry.
_CURRENCY_SUFFIX = re.compile(rf"(?:^|\s)({_CURRENCY_TOKEN})\s*$", re.IGNORECASE)


def _parse_candidate(text: str) -> _ParsedQuantity:
    """:func:`parse_quantity` on ``text``, retried with a trailing currency token moved to the
    front when the literal order is not an accepted quantity. The returned ``source_text``
    differs from ``text`` exactly when that re-ordering was used (see :func:`_entry_unit`)."""
    parsed = parse_quantity(text)
    if parsed.ok and _accept(parsed):
        return parsed
    suffix = _CURRENCY_SUFFIX.search(text)
    if suffix:
        head = text[:suffix.start()].strip()
        if head:
            moved = parse_quantity(f"{suffix.group(1)} {head}")
            if moved.ok and _accept(moved):
                return moved
    return parsed


#: A lone superscript digit the flattener dropped onto its own line (``km<sup>2</sup>`` ->
#: ``"km\n2"``), optionally trailed by punctuation (``"2)"``, ``"2):"``) -- never a digit run.
_SUPERSCRIPT_LINE = re.compile(r"^[23²³][\W_]*$")


def _entry_unit(parsed: _ParsedQuantity, candidate: str, fallback: str) -> str:
    """The ``unit`` text to store for an accepted ``parsed`` that :func:`_parse_candidate` read
    from ``candidate``: the parsed unit when there is one; for a suffix-currency parse (detected
    by the re-ordered ``source_text``) the scale word plus currency CODE so that
    ``f"{value} {unit}"`` still re-parses to the full magnitude; otherwise ``fallback`` (the
    literal text after the number -- a scale word, or empty)."""
    if parsed.unit:
        return _collapse_superscript(parsed.unit)
    if parsed.currency and parsed.source_text != candidate.strip():
        return " ".join(part for part in (parsed.scale_name, parsed.currency) if part)
    return fallback

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
    # Likewise a line that is itself an accepted one-line quantity ("€533 million", "13,860 MW")
    # is the PREVIOUS row's value, not the next row's label.
    if _LEADING_NUMBER.match(candidate) and _leading_quantity(candidate) is not None:
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


@dataclass(frozen=True)
class _ForwardMatch:
    """What :func:`_forward_quantity` found: the accepted primary :class:`Quantity`, the RAW
    text window it was hunted across (for locating any further quantities inside that same
    window), and a leftover ``tail`` when the primary was only recoverable by splitting off a
    second, unrelated quantity chained onto the same unit line (see :func:`_forward_quantity`).
    """

    parsed: _ParsedQuantity
    window_start: int
    window_end: int
    tail: str
    unit_fallback: str


def _forward_quantity(lines: List[Tuple[str, int, int]], index: int,
                       value_text: str) -> Optional[_ForwardMatch]:
    """The accepted :class:`Quantity` for the value line at ``index``, or ``None``.

    Grows a candidate unit region one line at a time (joined with single spaces) and re-parses
    ``value_text + " " + candidate`` on every growth step via :func:`parse_quantity`, so a
    restatement split across a newline (``"m (5,387\\nft)"``) is still read as one span. Stops at
    the first line-join that parses cleanly (:attr:`Quantity.ok`) AND passes :func:`_accept` —
    NOT merely at the first clean parse, since a garbage word like ``"Motihari"`` also parses
    cleanly (letters-only tails are syntactically valid units to ``parse_quantity``) but is not
    whitelisted and must not stop the search.

    When growth on a given step carries a ``";"`` and still fails whole, one more attempt is made
    at THAT step: split on the FIRST ``";"`` and re-check just the ``value_text + " " + head``
    half. This recovers rows the flattener wrote as a chain of two quantities on one unit line
    (``"7,280\\nft; 1.38\\nmi (2,220\\nm)"`` — the embedded ``"1.38"`` breaks every whole-join
    parse, silently dropping even the primary ``"7,280 ft"`` under the old all-or-nothing growth).
    Once the primary half is recovered this way, the text after the ``";"`` is grown FORWARD, one
    further line at a time, independently, until IT parses as a complete quantity on its own —
    exactly mirroring the primary's own growth discipline, so this does not over-consume past the
    row's natural end into the NEXT unrelated infobox row when the flattened text has no blank
    line between rows (the corpus shape has none). That grown tail is returned for the caller to
    mine as its own quantity chain, not discarded.
    """
    parts: List[str] = []
    stop = min(len(lines), index + 1 + _FORWARD_UNIT_LINES)
    best: Optional[_ForwardMatch] = None
    for k in range(index + 1, stop):
        segment = lines[k][0].strip()
        if not segment:
            break
        parts.append(segment)
        candidate = " ".join(parts)
        parsed = _parse_candidate(f"{value_text} {candidate}")
        if not (parsed.ok and _accept(parsed)):
            # A trailing punctuation mark glued onto a closing parenthesis by the flattener
            # ("... km\n2\n):" -- a sentence colon landing right after the restatement's own
            # close-paren, no line break between them) hides the parenthetical from
            # `parse_quantity`, which requires the restatement to be the literal LAST thing in
            # the string. Stripping trailing sentence punctuation (never letters/digits) mirrors
            # what `_scan_prose` already does for the same reason and is retried, never preferred
            # over the raw candidate.
            trimmed = candidate.rstrip(":;,.")
            if trimmed != candidate:
                parsed = _parse_candidate(f"{value_text} {trimmed}")
        if parsed.ok and _accept(parsed):
            window_start = lines[index + 1][1]
            window_end = lines[index + len(parts)][2]
            best = _ForwardMatch(parsed, window_start, window_end, "", candidate)
            # A whitelisted unit followed by a lone superscript digit line ("km\n2") is the
            # flattened km<sup>2</sup>: the shorter "km" parse is real but wrong, so keep
            # growing and prefer the longer parse when it is accepted too. Anything else on the
            # next line (the next row's label, a parenthetical) ends the search here.
            if k + 1 < stop and _SUPERSCRIPT_LINE.match(lines[k + 1][0].strip() or "x"):
                continue
            return best
        if best is not None:
            # The superscript growth did not parse; the shorter accepted unit stands.
            return best
        if ";" in candidate:
            head, _sep, tail_start = candidate.partition(";")
            head_parsed = _parse_candidate(f"{value_text} {head.strip()}")
            if not (head_parsed.ok and _accept(head_parsed)):
                continue
            window_start = lines[index + 1][1]
            tail_parts = [tail_start.strip()] if tail_start.strip() else []
            tail_stop_index = k
            for k2 in range(k + 1, stop):
                segment2 = lines[k2][0].strip()
                if not segment2:
                    break
                tail_parts.append(segment2)
                tail_stop_index = k2
                tail_candidate = " ".join(p for p in tail_parts if p)
                if parse_quantity(tail_candidate).ok:
                    break
            window_end = lines[tail_stop_index][2]
            tail_candidate = " ".join(p for p in tail_parts if p)
            return _ForwardMatch(head_parsed, window_start, window_end, tail_candidate,
                                  head.strip())
    return best


def _raw_offset(window_raw: str, raw_start: int, value_str: str,
                 cursor: int) -> Optional[Tuple[int, int]]:
    """Where ``value_str`` (a number token from an already-parsed sub-quantity's own
    ``source_text``) sits in the RAW page text, given the RAW window it was found inside.

    Searches from ``cursor`` first (so repeated growth-chain lookups march forward through the
    window instead of re-finding an earlier occurrence of the same digits), falling back to a
    from-the-start search so a single lookup still succeeds.
    """
    idx = window_raw.find(value_str, cursor)
    if idx == -1:
        idx = window_raw.find(value_str)
    if idx == -1:
        return None
    start = raw_start + idx
    return start, start + len(value_str)


def _emit_restatement_chain(quantity: _ParsedQuantity, window_raw: str, raw_start: int,
                             cursor: int, label: str) -> List[QuantityRef]:
    """Every quantity in ``quantity.restatement``'s chain, as its own :class:`QuantityRef`.

    A dual-unit restatement (``"1,642 m (5,387 ft)"``) is parsed only far enough by
    ``parse_quantity`` to discard the parenthetical into :attr:`Quantity.restatement`; this is
    where that discarded half gets its own index entry instead — capture, not conversion, per the
    module docstring: BOTH values are literally on the page, nothing here computes ``ft`` from
    ``m`` or vice versa. Walks the WHOLE chain (a restatement can itself carry a restatement) so
    none of it is silently dropped.
    """
    entries: List[QuantityRef] = []
    current = quantity.restatement
    while current is not None:
        match = _LEADING_NUMBER.match(current.source_text.strip())
        if match:
            value_str = match.group(0)
            located = _raw_offset(window_raw, raw_start, value_str, cursor)
            if located is not None:
                start, end = located
                unit = _entry_unit(current, current.source_text, current.scale_name or "")
                entries.append(QuantityRef(label=label, value=value_str, unit=unit,
                                            start=start, end=end, source="infobox",
                                            currency=current.currency,
                                            scale=current.scale_name))
                cursor = end - raw_start
        current = current.restatement
    return entries


def _emit_tail_chain(tail_text: str, window_raw: str, raw_start: int, cursor: int,
                      label: str) -> List[QuantityRef]:
    """Every quantity chained after a ``";"`` split (see :func:`_forward_quantity`), each with its
    own restatement chain (:func:`_emit_restatement_chain`) also mined out.
    """
    entries: List[QuantityRef] = []
    for segment in tail_text.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        parsed = _parse_candidate(segment)
        if not parsed.ok:
            continue
        match = _LEADING_NUMBER.match(parsed.source_text.strip())
        if match:
            value_str = match.group(0)
            located = _raw_offset(window_raw, raw_start, value_str, cursor)
            if located is not None:
                start, end = located
                unit = _entry_unit(parsed, segment, parsed.scale_name or "")
                entries.append(QuantityRef(label=label, value=value_str, unit=unit,
                                            start=start, end=end, source="infobox",
                                            currency=parsed.currency, scale=parsed.scale_name))
                cursor = end - raw_start
        entries.extend(_emit_restatement_chain(parsed, window_raw, raw_start, cursor, label))
        if entries:
            cursor = entries[-1].end - raw_start
    return entries


#: The leading number of a line that is NOT purely numeric — the ``"13,860 MW"`` shape, where the
#: flattener kept a value and its unit on one row instead of splitting them onto separate lines.
#: An optional currency prefix rides along INSIDE the matched value (``"€533"``): that is how the
#: page wrote it, and a consumer re-parsing the stored ``value`` then sees the currency dimension.
_LEADING_NUMBER = re.compile(rf"^(?:{_CURRENCY_TOKEN}\s*)?-?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)


def _leading_quantity(stripped: str) -> Optional[Tuple[str, _ParsedQuantity, str]]:
    """The longest valid leading ``"NUMBER [unit]"`` prefix of ``stripped``, or ``None``.

    Tries the WHOLE line first (the original all-or-nothing check), then progressively shorter
    word-prefixes, so a trailing annex the flattener appended (``"154 + 9 maintenance"``) no
    longer sinks the whole row — the row degrades to its longest still-valid leading quantity
    (``"154"``) instead of being dropped outright. Still gated by :func:`_accept` throughout, so
    this can only ever return a unit/currency/scale-bearing parse; it is not a route to admitting
    an unqualified bare number (see :func:`_label_allows_bare_count` for that separate, narrower,
    label-gated path).

    :returns: ``(value_text, parsed, matched_text)``, where ``matched_text`` is the exact prefix
        that parsed (so a caller can recover a literal-text unit fallback the way the whole-line
        check always could), or ``None`` when no leading prefix parses as an accepted quantity.
    """
    words = stripped.split()
    for take in range(len(words), 0, -1):
        prefix = " ".join(words[:take])
        number = _LEADING_NUMBER.match(prefix)
        if not number:
            continue
        parsed = _parse_candidate(prefix)
        if parsed.ok and _accept(parsed):
            return number.group(0), parsed, prefix
    return None


def _scan_infobox(text: str) -> List[QuantityRef]:
    """Every ``label / value / unit`` infobox row in ``text`` (see the module docstring)."""
    lines = _line_spans(text)
    entries: List[QuantityRef] = []
    for index, (raw_line, line_start, line_end) in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        label = _infer_label(lines, index)
        leading_ws = len(raw_line) - len(raw_line.lstrip())
        line_value_start = line_start + leading_ws
        if _NUM_ONLY_LINE.fullmatch(stripped):
            match = _forward_quantity(lines, index, stripped)
            if match is None:
                # No unit found anywhere in the forward window at all -- a genuinely unit-less
                # row, admitted ONLY under a whitelisted count label (see _bare_count_label).
                count_label = _bare_count_label(lines, index)
                if count_label is not None:
                    entries.append(QuantityRef(
                        label=count_label, value=stripped, unit="count",
                        start=line_value_start, end=line_value_start + len(stripped),
                        source="infobox",
                    ))
                continue
            unit = _entry_unit(match.parsed, f"{stripped} {match.unit_fallback}",
                               match.unit_fallback)
            entries.append(QuantityRef(
                label=label, value=stripped, unit=unit,
                start=line_value_start, end=line_value_start + len(stripped),
                source="infobox", currency=match.parsed.currency,
                scale=match.parsed.scale_name,
            ))
            window_raw = text[match.window_start:match.window_end]
            entries.extend(_emit_restatement_chain(
                match.parsed, window_raw, match.window_start, 0, label))
            if match.tail:
                entries.extend(_emit_tail_chain(
                    match.tail, window_raw, match.window_start, 0, label))
        else:
            # A row whose value and unit share one line ("13,860 MW") rather than being split
            # across two — accepted for the longest valid leading quantity prefix (see
            # _leading_quantity), so a trailing non-quantity annex degrades gracefully instead of
            # sinking the whole row.
            found = _leading_quantity(stripped)
            if found is None:
                number_only = _LEADING_NUMBER.match(stripped)
                count_label = _bare_count_label(lines, index) if number_only else None
                if count_label is not None:
                    bare_value = number_only.group(0)
                    entries.append(QuantityRef(
                        label=count_label, value=bare_value, unit="count",
                        start=line_value_start, end=line_value_start + len(bare_value),
                        source="infobox",
                    ))
                continue
            value_text, parsed, matched_text = found
            unit = _entry_unit(parsed, matched_text, matched_text[len(value_text):].strip())
            entries.append(QuantityRef(
                label=label, value=value_text, unit=unit,
                start=line_value_start, end=line_value_start + len(value_text),
                source="infobox", currency=parsed.currency, scale=parsed.scale_name,
            ))
            line_window_start = line_value_start + len(value_text)
            entries.extend(_emit_restatement_chain(
                parsed, text[line_window_start:line_end], line_window_start, 0, label))
    return entries


#: An inline ``NUMBER WORD[ WORD]`` pair on ONE line — the prose shape. The connector is
#: whitespace EXCLUDING the newline, or a bare hyphen (the ``"515-kilometre"`` compound-adjective
#: idiom), on purpose: a number and unit split across a newline with no hyphen is the infobox
#: shape (:func:`_scan_infobox`'s job), and letting this regex cross a bare newline would have it
#: rediscover every infobox row too, defeating the source-priority de-duplication in
#: :func:`build_index`. The lookbehind keeps this from re-matching the tail of a longer number
#: (``"1,642"`` must not also offer ``"642"`` as a second candidate).
#: A currency may lead the number (``"€533 million"``, kept inside ``value``) or trail it as a
#: word (``"100 €"``, ``"533 million EUR"``); with a leading currency the unit words become
#: optional, since ``"€533"`` alone is already a dimensioned quantity.
_PROSE_QUANTITY = re.compile(
    rf"(?<![\d,.])(?P<value>(?:{_CURRENCY_TOKEN}[^\S\n]*)?-?\d[\d,]*(?:\.\d+)?)"
    r"(?:(?:[^\S\n]+|-)"
    r"(?P<unit>[A-Za-z°%$£€¥₹][A-Za-z°%./-]*(?:[^\S\n]+[A-Za-z$£€¥₹]+)?))?",
    re.IGNORECASE,
)


def _scan_prose(text: str) -> List[QuantityRef]:
    """Every inline ``NUMBER unit`` mention in ``text`` (see the module docstring)."""
    entries: List[QuantityRef] = []
    for match in _PROSE_QUANTITY.finditer(text):
        value_text = match.group("value")
        words = (match.group("unit") or "").split()
        unit: Optional[str] = None
        accepted: Optional[_ParsedQuantity] = None
        # Try the two-word candidate first ("square kilometres"), then fall back to just the
        # first word, so a genuine unit is not missed because a stray following word rode along.
        # ``take == 0`` (the bare value) is reached only by a currency-prefixed value, which is a
        # quantity on its own; a bare number still needs a unit word to be one.
        for take in range(len(words), -1, -1):
            # Strip trailing sentence punctuation ("meters." at a full stop, "feet," before a
            # comma) that the regex's permissive unit-word class swept in — a real unit token
            # never legitimately ends in one, and leaving it on defeats the whitelist lookup.
            candidate = " ".join(words[:take]).rstrip(".,;:)")
            if not candidate and take:
                continue
            text_candidate = f"{value_text} {candidate}".strip()
            parsed = _parse_candidate(text_candidate)
            if parsed.ok and _accept(parsed):
                unit = _entry_unit(parsed, text_candidate, candidate)
                accepted = parsed
                break
        if unit is None or accepted is None:
            continue
        start = match.start("value")
        entries.append(QuantityRef(
            label="",
            value=value_text,
            unit=unit,
            start=start,
            end=start + len(value_text),
            source="prose",
            currency=accepted.currency,
            scale=accepted.scale_name,
        ))
    return entries


#: The ``"N hours M minutes"`` / ``"N h M min"`` idiom — never captured as ONE quantity by
#: :func:`_scan_prose` because "2 hours 21 minutes" is two independent ``NUMBER unit`` mentions to
#: it, with no shared dimension between "hours" and "minutes" for anything downstream to combine.
#: A rail-average-speed derivation needs a single duration in one unit (``distance / time``), so
#: this idiom is recognized here and folded into one decimal-hours quantity.
_COMPOUND_DURATION = re.compile(
    r"(?<!\d)(?P<h>\d+(?:\.\d+)?)[^\S\n]*(?:hours?|hrs?|h)\b[^\S\n]*"
    r"(?P<m>\d+(?:\.\d+)?)[^\S\n]*(?:minutes?|mins?|min)\b",
    re.IGNORECASE,
)


def _format_decimal(value: float) -> str:
    """``value`` rounded to 2 decimal places, without a trailing ``".00"``/``".50"`` zero run."""
    text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _scan_durations(text: str) -> List[QuantityRef]:
    """Every ``"N hours M minutes"`` phrase in ``text``, folded into one decimal-hours quantity.

    Unlike every other entry in this index, :attr:`QuantityRef.value` here is COMPUTED (``"2
    hours 21 minutes"`` -> ``"2.35"``), not copied verbatim from the page — a deliberate, narrow
    exception to the module's general "never normalized" rule, made only for this one two-part
    duration idiom because nothing downstream can otherwise combine an "hours" quantity with a
    "minutes" quantity into a single time-of-journey figure. ``start``/``end`` still span the
    ORIGINAL phrase in the page text, so a quote built from them remains a literal, verifiable
    substring even though the stored ``value`` is not.
    """
    entries: List[QuantityRef] = []
    for match in _COMPOUND_DURATION.finditer(text):
        try:
            hours = float(match.group("h"))
            minutes = float(match.group("m"))
        except ValueError:
            continue
        decimal_hours = hours + minutes / 60.0
        entries.append(QuantityRef(
            label="",
            value=_format_decimal(decimal_hours),
            unit="h",
            start=match.start(),
            end=match.end(),
            source="prose",
        ))
    return entries


def build_index(page_text: Optional[str], *, limit: Optional[int] = 40) -> List[QuantityRef]:
    """Every quantity :func:`_scan_infobox` / :func:`_scan_prose` can extract from ``page_text``.

    Deterministic and stable: pure sequential scans, no dict/set governs ORDER (a ``set`` is used
    only for de-dup membership testing). Infobox entries come first, in document order, then prose
    entries in document order — this is both the display order and how "prefer infobox" is
    enforced, since a later duplicate (same normalized value + unit) is dropped in
    :func:`build_index` rather than in either scanner. Capped at ``limit`` by default because the
    render goes into a weak model's prompt; a mechanical consumer (a host-side derivation that
    never shows the index to a model) passes ``limit=None`` to get every entry.

    A dual-unit restatement (``"1,642 m (5,387 ft)"``) surfaces BOTH halves here — ``parse_quantity``
    reads the ``"(5,387 ft)"`` parenthetical and separates it into ``.restatement`` precisely so it
    never contaminates the outer ``unit``, and :func:`_scan_infobox` mines that discarded half out
    as its OWN entry rather than leaving it invisible. This is capture, not conversion: both
    ``"1,642"`` and ``"5,387"`` are literally on the page in their own units; nothing here ever
    divides by 3.281 or otherwise turns one into the other, and a lookup by ``q<N>`` id still
    resolves to exactly one unit, never a computed alternate. See the module docstring.

    :param page_text: the raw page text, or ``None``/``""``.
    :param limit: maximum entries returned, or ``None`` for no cap.
    :returns: a list of :class:`QuantityRef`, possibly empty. Never raises on absence — a page
        with nothing extractable returns ``[]``, not a sentinel.
    """
    text = str(page_text or "")
    if not text:
        return []
    seen = set()
    seen_values = set()
    out: List[QuantityRef] = []
    for entry in _scan_infobox(text) + _scan_prose(text) + _scan_durations(text):
        value_key = normalize_for_match(entry.value)
        key = (value_key, normalize_for_match(entry.unit))
        if key in seen:
            continue
        # A unit-less entry (a prose "€533" re-found on an infobox row whose own entry already
        # carries "million") adds nothing over an earlier entry of the same value: it is the
        # same printed number, minus context the earlier extractor kept.
        if not entry.unit and value_key in seen_values:
            continue
        seen.add(key)
        seen_values.add(value_key)
        out.append(entry)
    return out if limit is None else out[:limit]


def render_index(entries: List[QuantityRef], *, max_chars: int = 1200, start: int = 1) -> str:
    """``entries`` rendered as a short ``q<N>: label = value unit`` block for a model prompt.

    :param entries: the output of :func:`build_index`, in the order to render (its ``q<N>`` ids
        are exactly this list's 1-based position OFFSET BY ``start``, so callers must not reorder
        before rendering).
    :param start: the id to give the first entry. A caller holding several pages MUST offset each
        page so ids stay unique across the whole run: with every page restarting at ``q1``, a model
        shown ``q1`` after visiting the second page silently received the FIRST page's quantity
        instead, and the unit guard then passed on the substituted value -- a confidently wrong
        number carrying full provenance, which is the failure this module exists to prevent.
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
    for position, entry in enumerate(entries, start=max(1, int(start))):
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
