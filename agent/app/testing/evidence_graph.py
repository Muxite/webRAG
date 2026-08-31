"""The evidence derivation graph: every asserted value traces back to a literal page span.

A content-addressed SIDECAR, not a plan. Two node kinds:

* :data:`KIND_SOURCE` — ``(value, page_id, char offsets, quote, verified)``. Admitted ONLY when
  the value is mechanically located in the stored page text (:func:`verify_value`). A value that
  is not on the page never becomes a node; a value that could not be CHECKED (no page, or a miss
  on a truncated window) is likewise not admitted, but is recorded as unverifiable rather than as
  failed — the tri-state of ``execution_evidence_loop.verify_quote`` is preserved end to end.
* :data:`KIND_DERIVED` — ``(value, operation, input_node_ids)``. This module builds the container,
  the id scheme and the edge representation; operation SEMANTICS are the sibling's half. A derived
  node's ``input_ids`` ARE its incoming edges, so :meth:`EvidenceGraph.sources_of` walks EVERY
  parent transitively down to the SOURCE spans it rests on.

Why not ``IdeaDag``: that structure carries ``status`` / ``score`` / ``memo_key`` (planning state
that means nothing for a fact), and its traversals assume tree parentage — ``path_to_root``
follows ``parent_ids[0]`` and ignores the rest, which for a two-input derivation would silently
drop half the provenance. Evidence also needs content-addressed identity: the same value read off
the same span twice is ONE node, so a run cannot inflate its own corroboration by re-reading.

Value verification vs quote verification
----------------------------------------
Both are mechanical and both are kept. Measured on 405 real SUPPORTED-with-value extraction
records, checking the VALUE verifies 3.0x more of them than checking the model's quote (49% vs
16%): a weak model paraphrases its supporting sentence far more often than it misreports the
datum it just read. Verification is exact substring first, then one LIGHT normalization pass
(NFKC, casefold, dash unification, digit-group separators, unit spacing, whitespace) which is
worth ~+16pp on its own.

REJECTED, deliberately, and not to be re-derived as an easy win: the "every digit token of the
value appears somewhere on the page" relaxation. It reaches 62% on the same corpus by accepting
values whose digits occur in unrelated places — a year in a citation, a page number, a sidebar
count. That is precisely the failure mode this graph exists to kill, and it is the same bug as
``grounding.verify_numeric_claims`` / ``answer_numeric_provenance``, which test set membership of
a number over EVERY page in the run and so "verify" ``1594`` against any page that happens to
mention 1594. Those two are therefore not reused here. No fuzzy, token-overlap or edit-distance
matching is used or wanted anywhere in this module.

Tightening (measured on the same corpus)
----------------------------------------
Exact substring matching alone is near-vacuous for a SHORT value. Re-checking a perturbed value
(a year off by one, a number off by three) against the very page it came from used to verify
23% of decoys for a bare number and 9% for a bare year, against 0.7% for a number carried WITH
its unit. Three mechanical causes, all closed here and all tightenings:

* No boundary check: ``1991`` matched inside ``11991``. A match now has to sit at non-alphanumeric
  boundaries on both sides, and the scan walks PAST an occurrence that fails them.
* The normalization pass deleted a digit-adjacent space when the next character was alphabetic
  **or another digit**. The second clause fused stacked infobox rows into one long digit string, so
  ``19912003`` "occurred" on a page listing ``1991`` and ``2003``. Only digit-then-ALPHABETIC is
  deleted now, which is the case the pass exists for (``285 m`` vs ``285\nm``; 32% of verified
  records verify only under normalization).
* NFKC folded superscripts to digits, so ``km²`` supplied a ``2``. Only characters in category
  ``Nd`` (fullwidth digits and friends) may fold to a digit now.

Carrying the unit with the value is worth 12-55x on the same attacks for free, so :func:`verify_value`
can verify ``value + unit`` as ONE span and reports :attr:`ValueMatch.unit_bearing` either way. A
bare value is never silently refused — it is admitted and reported as bare, for the caller to weigh.

Occurrence count and label proximity are reported for the same reason: a verified year occurs a
median of three times per page, so the FIRST occurrence is often a citation date rather than the
datum. When a field label is supplied the scan prefers an occurrence with a label token in its
window; refusing an ambiguous match outright is opt-in (``refuse_ambiguous``) and default OFF.

Layer 4: recomputed arithmetic (the DERIVED half)
--------------------------------------------------
The motivating bug: a reference solver run blind on this benchmark never let a model do
arithmetic in prose — every figure was recomputed. Ours once stated ``1594`` against a true
``824`` because a computed quantity had nowhere to live and nobody recomputed it. This section is
that place.

The closed vocabulary is seven operation KINDS (:data:`OPERATION_KINDS`). Four are mechanically
checkable from the input node values alone and are implemented here: :meth:`EvidenceGraph.add_arith`
(sum / difference / product / quotient / ratio), :meth:`EvidenceGraph.add_count`,
:meth:`EvidenceGraph.add_extremum` (max / min) and :meth:`EvidenceGraph.add_compare`. The other
three — ``lookup`` (pulling one field off a page is not a computation), ``assert_verbatim`` (the
model states a value with no input nodes to check it against) and ``judgment`` (a genuinely
subjective call, e.g. "is this the same entity") — have no seam here on purpose: recomputing them
would mean guessing what they mean, which is exactly the fuzzy matching this module refuses
anywhere.

Every one of the four ops RECOMPUTES its value in Python; a model's PROPOSED value is only ever
compared against that recomputation (:meth:`EvidenceGraph.add_arith`'s ``proposed_value``), never
substituted for it. A disagreement is recorded on the node (``derivation_detail``) and marks it
invalid (``derivation_valid``); it is never silently overwritten and never silently accepted.

Units. A SOURCE node's ``unit`` is either the caller's explicit ``unit=`` kwarg to
:meth:`EvidenceGraph.add_source`, or (when that is absent) whatever :func:`extract_unit` reads off
a unit-bearing value's own text (``"330 m"`` -> ``"m"``). Combining two DIFFERENT non-empty units
in ``sum`` / ``difference`` / ``extremum`` / ``compare`` raises ``ValueError`` and creates no node
at all — Wikipedia infoboxes reliably list feet before metres, so this is the single highest-value
check here, and a refusal is the correct, conservative behavior. No conversion table is
implemented: this module can tell ``m`` and ``ft`` apart, but does not know they are related, and
that is a deliberate, documented limitation rather than an oversight (see
:meth:`EvidenceGraph._check_common_unit`). A missing unit on one side is not treated as a
mismatch — only two PRESENT, DIFFERENT units refuse — so a genuinely unitless extraction paired
with a unit-bearing one is not caught by this check; that is the honest edge this module leaves
open. ``quotient`` / ``ratio`` are exempt from the check (dividing metres by seconds is the point
of a rate) and instead compose a new unit string, ``"<numerator>/<denominator>"``, when both
inputs carry one.

Tolerance. :data:`ARITH_RELATIVE_TOLERANCE` (``1e-6``, relative, via ``math.isclose``) exists to
absorb float round-trip noise from the Python arithmetic ITSELF (e.g. a division that lands on
``823.9999999999999``), not to forgive a real disagreement: the motivating bug (``1594`` vs
``824``) is off by nearly 2x, five orders of magnitude past this tolerance. There is no separate
"the page rounded it" allowance — a page value verified as a SOURCE node is used verbatim, so any
residual slack here is exclusively floating-point noise, and 1e-6 is generous for that by several
orders of magnitude while still catching any disagreement a human would call real.

Invalidity propagates. When any input to a new derivation is itself a DERIVED node whose own
``derivation_valid`` is ``False``, the new node inherits that (recorded in ``derivation_detail``)
regardless of what its own postcondition says — garbage in, garbage out. A DERIVED input whose
validity was never assessed (``None``, e.g. a node built through the bare
:meth:`EvidenceGraph.add_derived` container with no operation semantics attached) does NOT
propagate as invalid; it is simply unknown, and unknown is not the same claim as wrong.
:meth:`EvidenceGraph.derivation_validity` reports the fraction of DERIVED nodes whose postcondition
is known to hold (``derivation_valid is True``) over all DERIVED nodes — an unassessed node counts
against the ratio, the same as a failed one, because "never checked" is not "passed".
"""

from __future__ import annotations

import hashlib
import math
import operator
import re
import unicodedata
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from agent.app.evidence_store import canonicalize_url
from agent.app.idea_policies.grounding import (
    _NUMBER_PATTERN,
    _THOUSANDS_SEPARATORS,
    _normalize_number,
)
from agent.app.idea_policies.waypoint import page_identity_ok
from agent.app.testing.execution_evidence_loop import (
    QUOTE_FAIL_ABSENT,
    QUOTE_FAIL_EMPTY,
    QUOTE_FAIL_NO_PAGE,
    hash_page_text,
    store_page,
    verify_against_stored_page,
)

#: A value read straight off a page span.
KIND_SOURCE = "source"
#: A value computed from other evidence nodes. The sibling's half; the seam is here.
KIND_DERIVED = "derived"

#: The full seven-op vocabulary a DERIVED node's ``operation`` may name. Four are mechanically
#: checkable from the input values alone and are implemented in this module (see the module
#: docstring's Layer 4 section); the other three have no seam here on purpose.
OPERATION_KINDS = ("arith", "count", "extremum", "compare", "lookup", "assert_verbatim", "judgment")


class DerivationError(ValueError):
    """A refused derivation, carrying a machine-readable :attr:`code`.

    Every refusal in this module is one of these. It subclasses ``ValueError`` so the historical
    contract ("an impossible derivation raises and creates no node") is unchanged for existing
    callers, and adds the one thing a caller could not previously get without matching on prose:
    WHICH refusal it was.

    The consumer is the evidence loop's ``derive`` action, which turns a refusal into an
    observation the model can act on. "your two operands are in different units, and this system
    does not convert" is a step the model can recover from; "ValueError" is a dead step. Typing
    at the raise site keeps that mapping from being coupled to the wording of a message.
    """

    #: Stable, machine-readable identifier for this refusal. Subclasses override it.
    code = "DERIVATION_ERROR"


class UnitMismatch(DerivationError):
    """Two operands carry different, both-present units. Refusal is correct; there is no
    conversion table here and deliberately never will be."""

    code = "UNIT_MISMATCH"


class MissingOperand(DerivationError):
    """An input id names no node in this graph, so the derivation would have no provenance."""

    code = "MISSING_OPERAND"


class NonNumeric(DerivationError):
    """An operand's value carries no number, so it cannot enter an arithmetic operation."""

    code = "NON_NUMERIC"


class DivisionByZero(DerivationError):
    """The denominator recomputes to zero."""

    code = "DIVISION_BY_ZERO"


class UnknownOperation(DerivationError):
    """The requested operation or mode is outside this module's closed vocabulary."""

    code = "UNKNOWN_OPERATION"


class WrongArity(DerivationError):
    """The operation was given the wrong number of operands."""

    code = "WRONG_ARITY"
#: The subset of :data:`OPERATION_KINDS` this module can actually recompute and check.
MECHANICALLY_CHECKABLE_KINDS = ("arith", "count", "extremum", "compare")

#: The two ``sum`` / ``product`` forms accept any number of inputs; the other three are strictly
#: binary (``a - b``, ``a / b``) because "difference of five things" and "rate of five things"
#: are not well-defined without an order the caller would have to invent.
_ARITH_VARIADIC_OPS = ("sum", "product")
_ARITH_BINARY_OPS = ("difference", "quotient", "ratio")

#: Relative tolerance (``math.isclose(..., rel_tol=...)``) for comparing a proposed arithmetic
#: value against the recomputed one. It exists to absorb float round-trip noise from the Python
#: arithmetic itself (e.g. a division landing on ``823.9999999999999`` instead of ``824``), not to
#: forgive a real disagreement: the motivating bug (``1594`` proposed against a recomputed ``824``)
#: is off by five orders of magnitude more than this. A SOURCE value is used verbatim as read off
#: the page, so any slack beyond floating-point noise would mean accepting a genuinely wrong figure.
ARITH_RELATIVE_TOLERANCE = 1e-6


def _numbers_agree(left: float, right: float) -> bool:
    """True when ``left`` and ``right`` agree within :data:`ARITH_RELATIVE_TOLERANCE`."""
    return math.isclose(left, right, rel_tol=ARITH_RELATIVE_TOLERANCE, abs_tol=1e-9)

#: The value was checked against a page in hand and is not on it.
VALUE_FAIL_ABSENT = QUOTE_FAIL_ABSENT
#: No page text was available (or the miss fell past a truncated window). Never a verdict.
VALUE_FAIL_NO_PAGE = QUOTE_FAIL_NO_PAGE
#: There was no value to check.
VALUE_FAIL_EMPTY = QUOTE_FAIL_EMPTY
#: The "value" is not a value (a URL, a stringified list). Unadjudicated on purpose: the string
#: may well be on the page, but it is not a resolution, so this module declines to bless it.
VALUE_FAIL_JUNK = "junk"
#: The page failed the caller's page-identity contract, so nothing on it was read.
VALUE_FAIL_PAGE_IDENTITY = "page_identity"
#: The value occurs more than once and no field label vouches for any occurrence. Opt-in, and
#: unadjudicated rather than failed: the value IS on the page, we just cannot say which one it is.
VALUE_FAIL_AMBIGUOUS = "ambiguous"

#: Default cap on the stored page window, matching the evidence loop's own storage default.
DEFAULT_STORE_CHARS = 6000

_DASHES = "‐‑‒–—―−"
_SEPARATOR_CHARS = re.compile(_THOUSANDS_SEPARATORS)
_BARE_NUMBER = re.compile(rf"^(?:{_NUMBER_PATTERN})$")
_URL_PREFIXES = ("http://", "https://", "ftp://", "www.")
_NUMBER_WITH_UNIT = re.compile(rf"^(?:{_NUMBER_PATTERN})\s*[^\W\d_][\w%°/^·.\-\s]*$")
#: Month names, for the day-month-year date shape (``"28 October 1981"``).
_MONTH_NAMES = ("January|February|March|April|May|June|July|"
               "August|September|October|November|December")
#: ``"28 October 1981"`` / ``"3 January 2005"`` — the day-month-year form seen in the corpus.
_DATE_DAY_MONTH_YEAR = re.compile(rf"^\d{{1,2}}\s+(?:{_MONTH_NAMES})\s+\d{{4}}$")
#: ``"1981-10-28"`` — the ISO form the extractor sometimes echoes into the ``unit`` field.
_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LABEL_TOKEN = re.compile(r"[a-z0-9]+")
#: Label words too common to vouch for anything.
_LABEL_STOPWORDS = frozenset({"the", "and", "for", "with", "from", "that", "this", "its",
                              "value", "field", "name", "info", "data"})

#: Half-width of the window a field label has to appear in to vouch for an occurrence.
DEFAULT_LABEL_WINDOW = 300


def _is_digit(char: str) -> bool:
    """True only for an ASCII digit — ``str.isdigit`` also accepts superscripts."""
    return "0" <= char <= "9"


class ValueMatch(NamedTuple):
    """Where a value was located in a page, mechanically. Same tri-state as ``QuoteMatch``.

    :param verified: True when the value is present in the page text, False when the page was in
        hand and the value is not on it, None when nothing could be checked.
    :param start: Start offset in the RAW page text, or -1.
    :param end: End offset (exclusive) in the RAW page text, or -1.
    :param fail_reason: one of ``absent`` / ``no_page`` / ``empty`` / ``junk`` / ``ambiguous``
        when ``verified`` is not True, else None.
    :param occurrences: how many boundary-valid occurrences of the value the page holds. 1 is a
        clean read; more than 1 means the returned span is a CHOICE among them.
    :param unit_bearing: True when the located span carried a unit (the value itself did, or a
        supplied unit matched adjacent to it). A bare span is 12-55x easier to hit by accident.
    :param label_nearby: True / False when a field label was supplied and was / was not found
        within the window of the chosen occurrence, None when no label was checked.
    """

    verified: Optional[bool]
    start: int
    end: int
    fail_reason: Optional[str] = None
    occurrences: int = 0
    unit_bearing: bool = False
    label_nearby: Optional[bool] = None


def is_junk_value(value: Any) -> bool:
    """True when ``value`` is one of the non-value shapes seen live and worth refusing outright.

    A bare URL is a citation, not a resolution; a stringified list (``"['https://a', ...]"``) is a
    tool's repr leaking into the value slot. Both occur in real extraction records and both would
    otherwise sail through substring matching against the page they came from.

    :param value: the candidate value.
    :returns: True when the value is a URL or a stringified list.
    :raises: nothing — non-string input is not junk, it is simply not a value.
    """
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in _URL_PREFIXES) and " " not in text:
        return True
    return text.startswith("[") and text.endswith("]")


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """``text`` lightly normalized for matching, plus a per-character map back to RAW offsets.

    The pass, in order: NFKC and casefold per character (so one raw character may expand to
    several normalized ones, all pointing back at it) EXCEPT where the fold would manufacture a
    digit out of a non-``Nd`` character (``km²`` keeps its superscript), dash unification,
    deletion of digit-group separators between digits, deletion of the space between a number and
    a following ALPHABETIC character (never another digit, which would fuse stacked infobox rows
    into one long digit string), and whitespace-run collapse with the ends stripped.

    The offset map is what lets :func:`_welded` tell a dropped space (a real boundary) from a
    dropped group separator (the inside of one number), so neither erases a boundary.

    :param text: any string.
    :returns: ``(normalized, offsets)`` where ``offsets[i]`` is the index in ``text`` that
        produced ``normalized[i]``.
    :raises: nothing.
    """
    chars: List[str] = []
    offsets: List[int] = []
    for index, raw in enumerate(text):
        folded = unicodedata.normalize("NFKC", raw).lower()
        if raw in _DASHES:
            folded = "-"
        elif unicodedata.category(raw) != "Nd" and any(_is_digit(c) for c in folded):
            folded = raw.lower()  # a superscript or a fraction never supplies a digit
        for char in folded:
            chars.append(char)
            offsets.append(index)

    keep: List[int] = []
    for position, char in enumerate(chars):
        if char.isspace():
            previous = chars[position - 1] if position else ""
            following = chars[position + 1] if position + 1 < len(chars) else ""
            if _is_digit(previous) and following.isalpha():
                continue  # unit spacing: "285 m" -> "285m"; never digit-digit, which fuses rows
        elif _SEPARATOR_CHARS.fullmatch(char):
            previous = chars[position - 1] if position else ""
            following = chars[position + 1] if position + 1 < len(chars) else ""
            if _is_digit(previous) and _is_digit(following):
                continue  # digit-group separator: "1,991" -> "1991"
        keep.append(position)

    out_chars: List[str] = []
    out_offsets: List[int] = []
    for position in keep:
        char = chars[position]
        if char.isspace():
            if not out_chars or out_chars[-1] == " ":
                continue
            char = " "
        out_chars.append(char)
        out_offsets.append(offsets[position])
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_offsets.pop()
    return "".join(out_chars), out_offsets


def normalize_for_match(text: Any) -> str:
    """``text`` under the light normalization :func:`verify_value` matches in.

    :param text: any string.
    :returns: the normalized text (see :func:`_normalize_with_map` for the exact pass).
    :raises: nothing.
    """
    return _normalize_with_map(str(text or ""))[0]


def _canonical_value(value: str) -> str:
    """A bare number canonicalized (``"1,991.0"`` -> ``"1991"``); anything else unchanged."""
    stripped = value.strip()
    if _BARE_NUMBER.fullmatch(stripped):
        return _normalize_number(stripped) or stripped
    return stripped


def value_shape(value: Any, unit: Any = "") -> str:
    """The shape of ``value``, which is what decides how easy it is to hit by accident.

    A date (``"28 October 1981"``, ``"1981-10-28"``) is its OWN shape, checked before the numeric
    shapes below, so it can never be treated as a unit-bearing number — the extractor's own
    ``unit`` field sometimes echoes an alternate rendering of the same date (``unit="October
    1981"`` or ``unit="1981-10-28"``), which would otherwise misclassify it as
    ``"number_with_unit"`` and corrupt any future arithmetic/compare op over it.

    A bare number or bare year additionally carrying a non-empty ``unit`` (the extractor's
    separate ``unit`` FIELD, as opposed to a unit embedded in ``value`` itself) is
    ``"number_with_unit"`` too: on 734 numeric extractions from the gpu0831 block, 38.0% carried
    their unit ONLY in this field, so ignoring it undercounts real unit coverage by more than half
    (24.1% in-value vs 62.1% combined).

    :param value: the candidate value.
    :param unit: an optional separate unit field reported alongside ``value``.
    :returns: ``"empty"``, ``"date"``, ``"bare_year"`` (a plain 1000-2099 integer),
        ``"bare_number"``, ``"number_with_unit"`` (a number with a unit, embedded in the value
        string or supplied via ``unit``) or ``"text"``.
    :raises: nothing.
    """
    text = str(value or "").strip()
    if not text:
        return "empty"
    if _DATE_DAY_MONTH_YEAR.fullmatch(text) or _DATE_ISO.fullmatch(text):
        return "date"
    has_unit_field = bool(str(unit or "").strip())
    if _BARE_NUMBER.fullmatch(text):
        plain = _SEPARATOR_CHARS.sub("", text)
        if plain.isdigit() and len(plain) == 4 and 1000 <= int(plain) <= 2099:
            shape = "bare_year"
        else:
            shape = "bare_number"
        return "number_with_unit" if has_unit_field else shape
    if _NUMBER_WITH_UNIT.fullmatch(text):
        return "number_with_unit"
    return "text"


def is_unit_bearing(value: Any) -> bool:
    """True when ``value`` carries its unit (``"330 m"``) rather than standing bare (``"330"``).

    :param value: the candidate value.
    :returns: True only for :func:`value_shape` ``"number_with_unit"``.
    :raises: nothing.
    """
    return value_shape(value) == "number_with_unit"


#: Splits a number-with-unit value into its two halves; the same shape as ``_NUMBER_WITH_UNIT``,
#: captured instead of merely matched.
#: A dual-unit parenthetical, as Wikipedia infoboxes render both systems (``"m (423 ft)"``,
#: ``"metres (2,051 ft)"``). Stripped before comparing two units so it cannot manufacture a false
#: mismatch; the leading token is still compared, so a genuine ``m`` vs ``ft`` mismatch still fires.
_UNIT_PARENTHETICAL = re.compile(r"\s*\(.*\)\s*$")


def _unit_leading_token(unit: str) -> str:
    """``unit`` with any trailing parenthetical stripped (``"m (423 ft)"`` -> ``"m"``).

    :param unit: a unit string, possibly carrying a dual-unit parenthetical.
    :returns: the leading token, whitespace-trimmed; ``unit`` unchanged when it has no
        parenthetical.
    :raises: nothing.
    """
    return _UNIT_PARENTHETICAL.sub("", unit).strip()


_NUMBER_UNIT_SPLIT = re.compile(
    rf"^(?P<number>{_NUMBER_PATTERN})\s*(?P<unit>[^\W\d_][\w%°/^·.\-\s]*)$")


def extract_unit(value: Any) -> str:
    """The unit suffix of a number-with-unit value (``"330 m"`` -> ``"m"``).

    :param value: the candidate value.
    :returns: the trimmed unit text, or ``""`` when ``value`` is not :func:`is_unit_bearing`.
    :raises: nothing.
    """
    match = _NUMBER_UNIT_SPLIT.fullmatch(str(value or "").strip())
    return match.group("unit").strip() if match else ""


def numeric_value(value: Any, unit: Any = "") -> Optional[float]:
    """The float ``value`` denotes, parsed via the shared :func:`_normalize_number` machinery.

    A known trailing unit is stripped first (``numeric_value("330 m", "m")`` and
    ``numeric_value("330 m")`` both give ``330.0``, the second by way of :func:`extract_unit`); a
    value with no recognizable leading number is not a value this function will guess at.

    :param value: the candidate value, bare or unit-bearing.
    :param unit: an optional known unit to strip before parsing.
    :returns: the parsed float, or None when ``value`` carries no number.
    :raises: nothing.
    """
    text = str(value or "").strip()
    unit_text = str(unit or "").strip()
    if unit_text and text.endswith(unit_text):
        text = text[: -len(unit_text)].strip()
    number_text = text
    if not _BARE_NUMBER.fullmatch(number_text):
        split = _NUMBER_UNIT_SPLIT.fullmatch(text)
        if split is None:
            return None
        number_text = split.group("number")
    normalized = _normalize_number(number_text)
    if not normalized:
        return None
    return float(normalized)


def _adjacent(offsets: Optional[List[int]], raw: str, left: int, right: int) -> bool:
    """True when nothing that constitutes a boundary separated two normalized positions in RAW.

    The normalization pass drops two kinds of character, and they mean opposite things here: a
    dropped SPACE (``285 m`` -> ``285m``) is a real boundary and must stay one, while a dropped
    group separator (``1,300`` -> ``1300``) is the inside of a single number.
    """
    if offsets is None:
        return True
    low, high = sorted((offsets[left], offsets[right]))
    return high == low + 1 or bool(raw[low + 1:high].strip())


def _welded(text: str, inside: int, outside: int,
            offsets: Optional[List[int]], raw: str) -> bool:
    """True when the character just outside the span is glued to it with no boundary between.

    Two ways to be glued: a directly adjacent alphanumeric (``1991`` inside ``11991``), or a
    ``.`` / ``,`` that itself has a digit on the far side, which means the number simply continues
    (``74`` inside ``26.74``, ``41`` inside ``41.2``, ``300`` inside ``1,300``). A trailing
    sentence period is not that, because nothing numeric follows it.

    ``offsets`` maps normalized positions back to raw ones. A neighbour the normalization pass
    only made adjacent (by deleting the space in ``285 m`` -> ``285m``) is NOT welded: the raw
    text had a boundary there, and the pass exists to see through that space, not to erase it.

    :param text: the haystack the span was found in.
    :param inside: index of the span's own edge character.
    :param outside: index of the neighbour, one step beyond ``inside``.
    :param offsets: the normalized-to-raw map, or None when ``text`` is raw.
    :param raw: the raw page text ``offsets`` points into; ignored when ``offsets`` is None.
    :returns: True when there is no boundary between the two.
    :raises: nothing.
    """
    if not text[inside].isalnum() or not _adjacent(offsets, raw, inside, outside):
        return False
    if text[outside].isalnum():
        return True
    if text[outside] not in ".," or not _is_digit(text[inside]):
        return False
    beyond = outside + (outside - inside)
    return (0 <= beyond < len(text) and _is_digit(text[beyond])
            and _adjacent(offsets, raw, outside, beyond))


def _find_all(text: str, needle: str, offsets: Optional[List[int]] = None,
              raw: str = "") -> List[Tuple[int, int]]:
    """Every occurrence of ``needle`` in ``text`` sitting at real boundaries on both sides.

    :param text: the haystack.
    :param needle: a non-empty needle.
    :param offsets: when ``text`` is normalized, its map back to raw offsets (see
        :func:`_welded`); None when ``text`` is the raw page.
    :param raw: the raw page text ``offsets`` points into.
    :returns: ``(start, end)`` pairs in order; an occurrence welded into a longer digit run or a
        longer word is skipped, and the scan CONTINUES rather than stopping at the first hit.
    :raises: nothing.
    """
    found: List[Tuple[int, int]] = []
    index = text.find(needle)
    while index >= 0:
        end = index + len(needle)
        head_ok = index == 0 or not _welded(text, index, index - 1, offsets, raw)
        if head_ok and (end >= len(text) or not _welded(text, end - 1, end, offsets, raw)):
            found.append((index, end))
        index = text.find(needle, index + 1)
    return found


def _label_tokens(label: Any) -> List[str]:
    """The words of a field label worth looking for near a match (>= 3 chars, not a stopword)."""
    return [token for token in _LABEL_TOKEN.findall(str(label or "").lower())
            if len(token) >= 3 and token not in _LABEL_STOPWORDS]


def _label_in_window(text: str, start: int, end: int, tokens: List[str], window: int) -> bool:
    haystack = text[max(0, start - window):min(len(text), end + window)].lower()
    return any(re.search(rf"\b{re.escape(token)}\b", haystack) for token in tokens)


def _candidates(value: str, unit: Any) -> List[Tuple[str, bool]]:
    """The spellings to look for, best first, each flagged as unit-bearing or bare.

    Three sources of a spelling, in priority order:

    1. ``value`` joined with a separately-reported ``unit``, when the value stands bare.
    2. ``value`` exactly as the model wrote it.
    3. ``value``'s NUMERIC half joined with the model's own declared ``unit``, when the value is
       unit-bearing but spells its unit differently from the ``unit`` field.
    4. the canonical (digit-group-normalized) form.

    Case 3 exists because of a real measured failure. On task 130 the USGS page says
    ``"20,310 feet"``; the extractor reported ``value="20,310 ft"`` with ``unit="feet"`` — the
    right number, off the right page, with the unit correctly declared in its own field, and only
    the spelling inside ``value`` differing. Only ``"20,310 ft"`` was ever tried, so the value was
    reported ABSENT. Six of six extractions failed that way in one cell and the graph admitted
    ZERO nodes, which silently makes the whole derivation layer inert while every offline test
    stays green.

    This is NOT a loosening of the module's no-fuzzy-matching rule, and must not become one. The
    added spelling is built from the value's own numeric half plus the unit the MODEL ITSELF
    declared, and it is still located as ONE exact, boundary-checked span. There is no conversion,
    no synonym table, no token-overlap and no edit distance: ``"20,310 ft"`` with ``unit="metres"``
    would look for ``"20,310 metres"`` and simply not find it. The match also stays UNIT-BEARING,
    so it keeps the 12-55x decoy resistance that carrying the unit buys — the alternative fix,
    falling back to the bare number, would have thrown that away.
    """
    text = value.strip()
    out: List[Tuple[str, bool]] = []
    unit_text = str(unit or "").strip()
    if unit_text and not is_unit_bearing(text):
        out.append((f"{text} {unit_text}", True))
    out.append((text, is_unit_bearing(text)))
    if unit_text and is_unit_bearing(text):
        embedded = extract_unit(text)
        if normalize_for_match(_unit_leading_token(embedded)) != normalize_for_match(
                _unit_leading_token(unit_text)):
            match = _NUMBER_UNIT_SPLIT.fullmatch(text)
            if match:
                respelled = f"{match.group('number').strip()} {unit_text}"
                if respelled != text:
                    out.append((respelled, True))
    canonical = _canonical_value(text)
    if canonical != text:
        out.append((canonical, is_unit_bearing(canonical)))
    return out


def verify_value(page_text: Any, value: Any, *, unit: Any = None,
                 label: Any = None, label_window: int = DEFAULT_LABEL_WINDOW,
                 refuse_ambiguous: bool = False) -> ValueMatch:
    """Check mechanically that ``value`` is literally present in ``page_text``.

    Exact substring first, then ONE light-normalized pass on both sides (:func:`normalize_for_match`)
    with the offsets mapped back to the raw page. A bare number is additionally canonicalized
    (``"1,991.0"`` and ``"1991"`` are the same figure). Nothing else: a paraphrase, a value with a
    different figure, or a value whose digits merely occur elsewhere on the page all fail.

    :param page_text: the stored page text.
    :param value: the value a record claims to have read off that page.
    :param unit: an optional unit to verify TOGETHER with the value, as one span. When the
        combined span is not on the page the bare value is still tried — a bare value is reported
        as bare, never silently refused.
    :param label: an optional field label whose words vouch for an occurrence. The scan prefers an
        occurrence with a label word within ``label_window`` characters.
    :param label_window: half-width, in characters, of that window.
    :param refuse_ambiguous: opt-in. When True, a value occurring more than once with no label
        nearby is refused as ``ambiguous`` instead of being resolved to an arbitrary occurrence.
    :returns: a :class:`ValueMatch`. ``verified`` is True with RAW offsets on a hit, False with
        ``absent`` when the page was in hand and the value is not on it, and None with
        ``empty`` / ``no_page`` / ``junk`` / ``ambiguous`` when nothing was adjudicated.
    :raises: nothing — non-string input is simply unchecked.
    """
    text = value if isinstance(value, str) else ""
    if not text.strip():
        return ValueMatch(None, -1, -1, VALUE_FAIL_EMPTY)
    if is_junk_value(text):
        return ValueMatch(None, -1, -1, VALUE_FAIL_JUNK)
    if not isinstance(page_text, str) or not page_text.strip():
        return ValueMatch(None, -1, -1, VALUE_FAIL_NO_PAGE)

    normalized: Optional[Tuple[str, List[int]]] = None
    tokens = _label_tokens(label)
    for candidate, bearing in _candidates(text, unit):
        hits = _find_all(page_text, candidate)
        if not hits:
            needle = normalize_for_match(candidate)
            if not needle:
                continue
            if normalized is None:
                normalized = _normalize_with_map(page_text)
            page, offsets = normalized
            hits = [(offsets[i], offsets[j - 1] + 1)
                    for i, j in _find_all(page, needle, offsets, page_text)]
        if not hits:
            continue
        nearby = None
        chosen = hits[0]
        if tokens:
            nearby = False
            for hit in hits:
                if _label_in_window(page_text, hit[0], hit[1], tokens, label_window):
                    chosen, nearby = hit, True
                    break
        if refuse_ambiguous and len(hits) > 1 and nearby is not True:
            return ValueMatch(None, -1, -1, VALUE_FAIL_AMBIGUOUS, len(hits), bearing, nearby)
        return ValueMatch(True, chosen[0], chosen[1], None, len(hits), bearing, nearby)
    return ValueMatch(False, -1, -1, VALUE_FAIL_ABSENT)


def verify_value_against_stored_page(page: Optional[Dict[str, Any]], value: Any,
                                     **options: Any) -> ValueMatch:
    """Re-check ``value`` against a page frozen by ``store_page``, offline.

    A miss on a page stored IN FULL is a real ``absent``. A miss on a TRUNCATED page is merely
    unverifiable — the value may live past the stored window, and calling that a failure would
    manufacture evidence of fabrication.

    :param page: a stored page dict, or None when the page was not persisted.
    :param value: the claimed value.
    :param options: forwarded verbatim to :func:`verify_value` (``unit``, ``label``,
        ``label_window``, ``refuse_ambiguous``).
    :returns: a :class:`ValueMatch` with the same tri-state contract as :func:`verify_value`.
    :raises: nothing.
    """
    if not isinstance(page, dict) or not str(page.get("text") or ""):
        checked = verify_value("", value)
        return checked if checked.fail_reason != VALUE_FAIL_NO_PAGE else ValueMatch(
            None, -1, -1, VALUE_FAIL_NO_PAGE)
    match = verify_value(str(page.get("text")), value, **options)
    if match.verified is False and page.get("truncated"):
        return ValueMatch(None, -1, -1, VALUE_FAIL_NO_PAGE)
    return match


def _hash(*parts: Any) -> str:
    payload = "\x00".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def source_node_id(page_id: str, start: int, end: int, value: str) -> str:
    """The content-addressed id of a SOURCE node: hash of ``(page_id, offsets, value)``.

    :param page_id: id of the stored page the value was read off.
    :param start: raw start offset of the located span.
    :param end: raw end offset (exclusive) of the located span.
    :param value: the value as recorded. Hashed under :func:`normalize_for_match`, so the same
        datum read twice off the same span is ONE node even when the two records spell it
        differently (``"1,991 metres"`` / ``"1991 metres"``).
    :returns: a 32-character hex id; the same read twice yields the same id, which is what makes
        re-reading a page unable to inflate corroboration.
    :raises: nothing.
    """
    return _hash(KIND_SOURCE, page_id, start, end, normalize_for_match(value))


def derived_node_id(operation: str, input_ids: Iterable[str], value: str) -> str:
    """The content-addressed id of a DERIVED node: hash of ``(operation, inputs, value)``.

    Input order is significant and preserved — ``subtract(a, b)`` is not ``subtract(b, a)``.

    :param operation: the operation's name.
    :param input_ids: ids of the input nodes, in argument order.
    :param value: the derived value.
    :returns: a 32-character hex id.
    :raises: nothing.
    """
    return _hash(KIND_DERIVED, operation, "".join(str(i) for i in input_ids), value)


@dataclass(frozen=True)
class EvidenceNode:
    """One node of the derivation graph. Immutable: a fact is never rewritten, only superseded.

    A SOURCE node carries its page pointer (``page_id`` and raw ``start`` / ``end`` offsets) and,
    optionally, the model's quote with its own independent verification state. A DERIVED node
    carries ``operation`` and ``input_ids`` instead; ``input_ids`` are its incoming edges.
    """

    id: str
    kind: str
    value: str
    page_id: str = ""
    source_url: str = ""
    start: int = -1
    end: int = -1
    quote: str = ""
    quote_verified: Optional[bool] = None
    quote_fail_reason: Optional[str] = None
    verified: Optional[bool] = None
    #: How many boundary-valid occurrences of the value the page holds; >1 means the span is a
    #: choice among them. With ``unit_bearing`` and ``label_nearby`` these are SIGNALS a caller
    #: weighs, not gates: admission is unchanged by them unless ``refuse_ambiguous`` is set.
    occurrences: int = 0
    unit_bearing: bool = False
    label_nearby: Optional[bool] = None
    operation: str = ""
    input_ids: Tuple[str, ...] = ()
    #: The value's unit (``"m"``), for a SOURCE node from :meth:`EvidenceGraph.add_source`'s
    #: ``unit=`` kwarg or else :func:`extract_unit` on its value; for a DERIVED node, whatever the
    #: operation determined (see the module docstring's Units section). ``""`` when there is none.
    unit: str = ""
    #: DERIVED nodes only. True when the operation's postcondition holds (a recomputed arithmetic
    #: value agreed with any proposed one, an extremum winner really is the extreme, ...); False
    #: when it does not, or the node rests on another node with False here; None when unassessed
    #: (a node built through the bare :meth:`EvidenceGraph.add_derived` container). Always None on
    #: a SOURCE node — verification there is :attr:`verified`, a different question.
    derivation_valid: Optional[bool] = None
    #: Human-readable detail when :attr:`derivation_valid` is not True — a disagreement between a
    #: proposed and recomputed value, or which upstream node made this one invalid. ``""`` when
    #: there is nothing to report.
    derivation_detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """This node as a JSON-serializable dict."""
        return {
            "id": self.id, "kind": self.kind, "value": self.value, "page_id": self.page_id,
            "source_url": self.source_url, "start": self.start, "end": self.end,
            "quote": self.quote, "quote_verified": self.quote_verified,
            "quote_fail_reason": self.quote_fail_reason, "verified": self.verified,
            "occurrences": self.occurrences, "unit_bearing": self.unit_bearing,
            "label_nearby": self.label_nearby,
            "operation": self.operation, "input_ids": list(self.input_ids),
            "unit": self.unit, "derivation_valid": self.derivation_valid,
            "derivation_detail": self.derivation_detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceNode":
        """Rebuild a node from :meth:`as_dict` output."""
        return cls(
            id=str(data.get("id", "")), kind=str(data.get("kind", KIND_SOURCE)),
            value=str(data.get("value", "")), page_id=str(data.get("page_id", "")),
            source_url=str(data.get("source_url", "")),
            start=int(data.get("start", -1)), end=int(data.get("end", -1)),
            quote=str(data.get("quote", "")), quote_verified=data.get("quote_verified"),
            quote_fail_reason=data.get("quote_fail_reason"), verified=data.get("verified"),
            occurrences=int(data.get("occurrences", 0) or 0),
            unit_bearing=bool(data.get("unit_bearing", False)),
            label_nearby=data.get("label_nearby"),
            operation=str(data.get("operation", "")),
            input_ids=tuple(str(i) for i in (data.get("input_ids") or [])),
            unit=str(data.get("unit", "")),
            derivation_valid=data.get("derivation_valid"),
            derivation_detail=str(data.get("derivation_detail", "")),
        )


@dataclass
class EvidenceGraph:
    """Content-addressed evidence: located SOURCE spans, plus the seam for DERIVED nodes.

    Pages are frozen with ``store_page`` (SHA-256 over the WHOLE fetched text, a head-slice
    window, a ``truncated`` flag), so the whole graph re-verifies offline from
    :meth:`to_dict` alone via :func:`reverify_graph`.
    """

    _nodes: Dict[str, EvidenceNode] = dataclass_field(default_factory=dict)
    _pages: Dict[str, Dict[str, Any]] = dataclass_field(default_factory=dict)
    #: Every refused admission, with the reason. A rejection is data, not silence.
    rejections: List[Dict[str, Any]] = dataclass_field(default_factory=list)

    def add_page(self, page_id: str, url: str, text: str,
                 max_chars: int = DEFAULT_STORE_CHARS) -> Dict[str, Any]:
        """Freeze one fetched page into the graph.

        :param page_id: the id SOURCE nodes point back to.
        :param url: the fetched URL; stored canonicalized (scheme+host lowercased, fragment
            dropped, query KEPT) so it stays a fetchable address of the same page.
        :param text: the fetched page text.
        :param max_chars: cap on the stored window; the hash still covers the whole text.
        :returns: the stored page dict.
        :raises: nothing — re-adding a page id overwrites it.
        """
        page = store_page(page_id, canonicalize_url(url), text, max_chars)
        self._pages[str(page_id)] = page
        return page

    def page(self, page_id: str) -> Optional[Dict[str, Any]]:
        """The stored page dict for ``page_id``, or None."""
        return self._pages.get(str(page_id))

    def pages(self) -> List[Dict[str, Any]]:
        """Every stored page, in insertion order."""
        return list(self._pages.values())

    def nodes(self) -> List[EvidenceNode]:
        """Every node, in insertion order."""
        return list(self._nodes.values())

    def node(self, node_id: str) -> Optional[EvidenceNode]:
        """The node with ``node_id``, or None."""
        return self._nodes.get(str(node_id))

    def edges(self) -> List[Tuple[str, str, str]]:
        """Every edge as ``(input_id, node_id, operation)``, in node insertion order.

        Derivation is the ONLY edge kind: a SOURCE node's provenance is its page pointer, not an
        edge, so the graph has exactly one meaning of "points at".
        """
        return [(input_id, node.id, node.operation)
                for node in self._nodes.values() for input_id in node.input_ids]

    def add_source(self, page_id: str, value: str, quote: str = "", contract: Any = None,
                   unit: Any = None, label: Any = None,
                   refuse_ambiguous: bool = False) -> Optional[EvidenceNode]:
        """Admit a SOURCE node for ``value`` if and only if it is located in the stored page.

        The quote, when given, is verified INDEPENDENTLY (``verify_against_stored_page``) and
        never gates admission: a paraphrased quote for a value that is demonstrably on the page is
        a weak model's bad citation, not a bad fact. Its state rides on the node.

        :param page_id: id of a page already added with :meth:`add_page`.
        :param value: the value claimed to have been read off that page.
        :param quote: the model's supporting quote, optional.
        :param contract: an optional ``StepContract``; when given, the page must pass
            ``waypoint.page_identity_ok`` — the repo's only "is this the right page" guard — or
            nothing is read off it at all.
        :param unit: an optional unit to locate together with the value, as one span.
        :param label: an optional field label used to prefer a vouched-for occurrence.
        :param refuse_ambiguous: opt-in refusal of a repeated value with no label nearby.
        :returns: the admitted (or already-present, content-identical) node, else None with an
            entry appended to :attr:`rejections`.
        :raises: nothing.
        """
        page = self.page(page_id)
        if contract is not None and page is not None and not page_identity_ok(
                contract, {"url": page.get("url", ""), "source_url": page.get("url", "")}):
            return self._reject(page_id, value, ValueMatch(None, -1, -1,
                                                           VALUE_FAIL_PAGE_IDENTITY))
        match = verify_value_against_stored_page(
            page, value, unit=unit, label=label, refuse_ambiguous=refuse_ambiguous)
        if match.verified is not True:
            return self._reject(page_id, value, match)

        node_id = source_node_id(str(page_id), match.start, match.end, str(value))
        existing = self._nodes.get(node_id)
        if existing is not None:
            return existing
        quote_match = verify_against_stored_page(page, quote)
        unit_text = str(unit or "").strip() or extract_unit(value)
        node = EvidenceNode(
            id=node_id, kind=KIND_SOURCE, value=str(value), page_id=str(page_id),
            source_url=str((page or {}).get("url", "")), start=match.start, end=match.end,
            quote=str(quote or ""), quote_verified=quote_match.verified,
            quote_fail_reason=quote_match.fail_reason, verified=True,
            occurrences=match.occurrences, unit_bearing=match.unit_bearing,
            label_nearby=match.label_nearby, unit=unit_text,
        )
        self._nodes[node_id] = node
        return node

    def add_derived(self, value: str, operation: str, input_ids: Iterable[str], *,
                    unit: str = "", derivation_valid: Optional[bool] = None,
                    derivation_detail: str = "") -> EvidenceNode:
        """Admit a DERIVED node computed from nodes already in the graph — the sibling's seam.

        This half builds the container: identity, edges, the input-existence check, and (via the
        keyword-only params) a place for an operation's computed result to record itself. What an
        operation MEANS is :meth:`add_arith` / :meth:`add_count` / :meth:`add_extremum` /
        :meth:`add_compare`'s job, layered on top of this container rather than guessed here.

        :param value: the derived value.
        :param operation: the operation's name, recorded verbatim as the edge label.
        :param input_ids: ids of the input nodes, in argument order (order is significant).
        :param unit: the derived value's unit, when it has one.
        :param derivation_valid: whether the operation's postcondition held; None when unassessed.
        :param derivation_detail: human-readable detail for a not-True ``derivation_valid``.
        :returns: the new node, or the content-identical one already present (its ORIGINAL unit /
            validity / detail are kept, matching :meth:`add_source`'s dedup-keeps-first rule).
        :raises: ValueError: when ``input_ids`` is empty or names a node the graph does not hold —
            a derivation whose inputs are not in the graph has no provenance at all.
        """
        inputs = tuple(str(i) for i in input_ids)
        if not inputs:
            raise WrongArity("a derived node needs at least one input node")
        missing = [i for i in inputs if i not in self._nodes]
        if missing:
            raise MissingOperand(f"unknown input node(s): {missing}")
        node_id = derived_node_id(str(operation), inputs, str(value))
        existing = self._nodes.get(node_id)
        if existing is not None:
            return existing
        node = EvidenceNode(id=node_id, kind=KIND_DERIVED, value=str(value),
                            operation=str(operation), input_ids=inputs, unit=str(unit or ""),
                            derivation_valid=derivation_valid,
                            derivation_detail=str(derivation_detail or ""))
        self._nodes[node_id] = node
        return node

    def _require_node(self, node_id: str) -> EvidenceNode:
        """The node with ``node_id``, or a loud ``ValueError`` — never a silent None here."""
        node = self.node(node_id)
        if node is None:
            raise MissingOperand(f"unknown input node: {node_id!r}")
        return node

    def _require_numeric(self, node: EvidenceNode) -> float:
        """``node``'s value as a float, or a loud ``ValueError`` when it has no number."""
        value = numeric_value(node.value, node.unit)
        if value is None:
            raise NonNumeric(f"node {node.id!r} value {node.value!r} is not numeric")
        return value

    def _inputs_valid(self, inputs: Iterable[EvidenceNode]) -> Tuple[bool, List[str]]:
        """Which of ``inputs`` are known-invalid DERIVED nodes (``derivation_valid is False``).

        A SOURCE node is always valid (only verified values ever become one). A DERIVED node whose
        own validity was never assessed (None) is treated as valid here — "unknown" is not the
        same claim as "wrong", so it does not poison a downstream computation on its own.
        """
        invalid = [node.id for node in inputs
                  if node.kind == KIND_DERIVED and node.derivation_valid is False]
        return not invalid, invalid

    def _check_common_unit(self, inputs: Iterable[EvidenceNode]) -> str:
        """The one unit ``inputs`` share, refusing loudly when two PRESENT units disagree.

        A missing unit (``""``) on one side is not a mismatch by itself — only two different,
        both-present units refuse. That is the documented, deliberate gap: a genuinely unitless
        extraction paired with a unit-bearing one is not caught here. No conversion table is
        applied; ``m`` and ``ft`` are recognized as different, not reconciled.

        Comparison is on the LEADING TOKEN (:func:`_unit_leading_token`), stripping any dual-unit
        parenthetical first: Wikipedia infoboxes render both systems in one string (``"m (423
        ft)"``, ``"metres (2,051 ft)"``), and comparing those raw strings would raise a FALSE
        mismatch and refuse a valid derivation whose value itself is correct.

        :param inputs: the nodes about to be combined.
        :returns: the shared unit AS RECORDED on the first input that has one (unstripped), or
            ``""`` when none of them has one.
        :raises: ValueError: when two present units differ, by their leading token, after
            :func:`normalize_for_match`.
        """
        nodes = list(inputs)
        present = [node.unit for node in nodes if node.unit]
        distinct = {normalize_for_match(_unit_leading_token(unit)) for unit in present}
        if len(distinct) > 1:
            raise UnitMismatch(f"mismatched units: {sorted(set(present))}")
        return present[0] if present else ""

    def _arith_detail(self, inputs_ok: bool, invalid_ids: List[str],
                      disagreement: str = "") -> Tuple[bool, str]:
        parts = [part for part in (disagreement,
                                   f"rests on invalid derivation(s): {invalid_ids}"
                                   if not inputs_ok else "") if part]
        return inputs_ok and not disagreement, "; ".join(parts)

    def add_arith(self, operation: str, input_ids: Iterable[str], *,
                 proposed_value: Any = None) -> EvidenceNode:
        """Admit a DERIVED node whose value is RECOMPUTED in Python, never asserted by a model.

        Supports ``sum`` and ``product`` (any number of inputs) and ``difference`` / ``quotient`` /
        ``ratio`` (exactly two, ``a OP b`` in input order). ``ratio`` is the same division as
        ``quotient`` under a name that reads better for a rate ("goals per appearance"); the two
        differ only in the unit string they compose (see the module docstring's Units section).

        :param operation: one of ``sum`` / ``difference`` / ``product`` / ``quotient`` / ``ratio``.
        :param input_ids: ids of existing nodes to combine, in argument order.
        :param proposed_value: an optional value a model claimed this computation yields. Never
            used as the node's value — only compared against the recomputation, within
            :data:`ARITH_RELATIVE_TOLERANCE`. A disagreement marks the node invalid and is recorded
            in ``derivation_detail``; it is never silently overwritten or accepted.
        :returns: the new (or content-identical existing) node, whose ``value`` is always the
            RECOMPUTED figure regardless of ``proposed_value``.
        :raises: ValueError: unknown operation or input id, a binary op given the wrong arity, a
            non-numeric input, division by zero, or mismatched units on ``sum`` / ``difference``.
        """
        if operation not in _ARITH_VARIADIC_OPS and operation not in _ARITH_BINARY_OPS:
            raise UnknownOperation(f"unknown arith operation: {operation!r}")
        inputs = [self._require_node(i) for i in input_ids]
        if operation in _ARITH_BINARY_OPS and len(inputs) != 2:
            raise WrongArity(f"{operation} needs exactly 2 inputs, got {len(inputs)}")
        if not inputs:
            raise WrongArity("arith needs at least one input")
        numeric = [self._require_numeric(node) for node in inputs]

        unit = ""
        if operation in ("sum", "difference"):
            unit = self._check_common_unit(inputs)
            recomputed = math.fsum(numeric) if operation == "sum" else numeric[0] - numeric[1]
        elif operation == "product":
            recomputed = 1.0
            for number in numeric:
                recomputed *= number
        else:  # quotient / ratio
            if numeric[1] == 0:
                raise DivisionByZero("division by zero")
            recomputed = numeric[0] / numeric[1]
            if inputs[0].unit and inputs[1].unit:
                unit = f"{inputs[0].unit}/{inputs[1].unit}"

        value_text = _normalize_number(str(round(recomputed, 6)))
        disagreement = ""
        if proposed_value is not None and str(proposed_value).strip():
            proposed_number = numeric_value(proposed_value)
            if proposed_number is None or not _numbers_agree(proposed_number, recomputed):
                disagreement = (f"proposed {str(proposed_value).strip()!r} disagrees with "
                                f"recomputed {value_text!r}")
        inputs_ok, invalid_ids = self._inputs_valid(inputs)
        valid, detail = self._arith_detail(inputs_ok, invalid_ids, disagreement)
        return self.add_derived(value_text, operation, [n.id for n in inputs], unit=unit,
                                derivation_valid=valid, derivation_detail=detail)

    def add_count(self, input_ids: Iterable[str]) -> EvidenceNode:
        """Admit a DERIVED ``count`` node: the number of DISTINCT inputs, after de-duplication.

        The value is never asserted by a model — it is ``len()`` of the filtered input list.

        :param input_ids: ids of existing nodes to count; repeats are filtered before counting.
        :returns: the new (or content-identical existing) node, ``value`` the count as a string.
        :raises: ValueError: an input id the graph does not hold.
        """
        inputs = [self._require_node(i) for i in input_ids]
        distinct_ids = list(dict.fromkeys(node.id for node in inputs))
        inputs_ok, invalid_ids = self._inputs_valid(inputs)
        detail = f"rests on invalid derivation(s): {invalid_ids}" if not inputs_ok else ""
        return self.add_derived(str(len(distinct_ids)), "count", distinct_ids,
                                derivation_valid=inputs_ok, derivation_detail=detail)

    def add_extremum(self, input_ids: Iterable[str], mode: str) -> EvidenceNode:
        """Admit a DERIVED ``max`` / ``min`` node: the extreme value among the inputs, verbatim.

        Postcondition, checked explicitly rather than merely assumed from ``max()`` / ``min()``:
        the winner's value is one of the inputs' values, and no input is more extreme than it.

        :param input_ids: ids of existing nodes to compare; at least one.
        :param mode: ``"max"`` or ``"min"``.
        :returns: the new (or content-identical existing) node, ``value`` the winning input's
            value VERBATIM (so its own provenance stays exact, e.g. ``"424 goals"``).
        :raises: ValueError: unknown mode, no inputs, an unknown input id, a non-numeric input, or
            mismatched units among the inputs.
        """
        if mode not in ("max", "min"):
            raise UnknownOperation(f"unknown extremum mode: {mode!r}")
        inputs = [self._require_node(i) for i in input_ids]
        if not inputs:
            raise WrongArity("extremum needs at least one input")
        unit = self._check_common_unit(inputs)
        numeric = [(node, self._require_numeric(node)) for node in inputs]
        winner_node, winner_value = (max if mode == "max" else min)(numeric, key=lambda pair: pair[1])
        postcondition_ok = all(
            (winner_value >= value) if mode == "max" else (winner_value <= value)
            for _, value in numeric)
        inputs_ok, invalid_ids = self._inputs_valid(inputs)
        valid = postcondition_ok and inputs_ok
        detail = "" if inputs_ok else f"rests on invalid derivation(s): {invalid_ids}"
        return self.add_derived(winner_node.value, mode, [node.id for node, _ in numeric],
                                unit=unit, derivation_valid=valid, derivation_detail=detail)

    def add_compare(self, left_id: str, right_id: str, mode: str) -> EvidenceNode:
        """Admit a DERIVED comparison node: ``"true"`` / ``"false"``, recomputed, never asserted.

        Both sides are looked up by id, so both must already be SOURCE-or-DERIVED nodes in this
        graph (:meth:`_require_node`) — there is no third node kind to smuggle an unchecked
        literal in as.

        :param left_id: id of the left-hand node.
        :param right_id: id of the right-hand node.
        :param mode: one of ``"gt"``, ``"lt"``, ``"ge"``, ``"le"``, ``"eq"``, ``"ne"``.
        :returns: the new (or content-identical existing) node.
        :raises: ValueError: unknown mode, an unknown input id, a non-numeric input, or mismatched
            units between the two sides.
        """
        ops = {"gt": operator.gt, "lt": operator.lt, "ge": operator.ge, "le": operator.le}
        if mode not in ops and mode not in ("eq", "ne"):
            raise UnknownOperation(f"unknown compare mode: {mode!r}")
        left = self._require_node(left_id)
        right = self._require_node(right_id)
        self._check_common_unit([left, right])
        left_value = self._require_numeric(left)
        right_value = self._require_numeric(right)
        if mode == "eq":
            result = _numbers_agree(left_value, right_value)
        elif mode == "ne":
            result = not _numbers_agree(left_value, right_value)
        else:
            result = ops[mode](left_value, right_value)
        inputs_ok, invalid_ids = self._inputs_valid([left, right])
        detail = f"rests on invalid derivation(s): {invalid_ids}" if not inputs_ok else ""
        return self.add_derived("true" if result else "false", f"compare_{mode}",
                                [left.id, right.id], derivation_valid=inputs_ok,
                                derivation_detail=detail)

    def derivation_validity(self) -> Optional[float]:
        """Fraction of DERIVED nodes whose postcondition is known to hold.

        A node whose validity was never assessed (``derivation_valid is None``) counts against the
        ratio exactly like a failed one: "never checked" is not the same claim as "passed".

        :returns: ``passing / total`` over DERIVED nodes, or None when the graph has none.
        :raises: nothing.
        """
        derived = [node for node in self._nodes.values() if node.kind == KIND_DERIVED]
        if not derived:
            return None
        passing = sum(1 for node in derived if node.derivation_valid is True)
        return passing / len(derived)

    def sources_of(self, node_id: str) -> List[EvidenceNode]:
        """Every SOURCE node ``node_id`` transitively rests on, following ALL inputs.

        This is the traversal ``IdeaDag.path_to_root`` cannot do: it follows ``parent_ids[0]`` and
        drops the rest, which for a two-input derivation would hide half the provenance.

        :param node_id: any node in the graph.
        :returns: the reachable SOURCE nodes, deduplicated, in first-visit order; a SOURCE node is
            its own single source. Unknown ids yield ``[]``.
        :raises: nothing — a cycle (which the id scheme cannot produce) terminates on the seen set.
        """
        start = self.node(node_id)
        if start is None:
            return []
        seen = {start.id}
        stack = [start]
        found: List[EvidenceNode] = []
        while stack:
            node = stack.pop(0)
            if node.kind == KIND_SOURCE:
                found.append(node)
                continue
            for input_id in node.input_ids:
                if input_id in seen:
                    continue
                seen.add(input_id)
                parent = self.node(input_id)
                if parent is not None:
                    stack.append(parent)
        return found

    def _reject(self, page_id: str, value: Any, match: ValueMatch) -> None:
        self.rejections.append({
            "page_id": str(page_id), "value": str(value or ""),
            "verified": match.verified, "fail_reason": match.fail_reason,
        })
        return None

    def counts(self) -> Dict[str, int]:
        """Node and admission counts.

        :returns: ``{"source", "derived", "pages", "rejected", "rejected_absent",
            "rejected_unchecked"}``. The absent / unchecked split is the whole point: ``absent``
            is a value that is not on the page it was attributed to, everything else is a value
            nothing could be said about.
        :raises: nothing.
        """
        absent = sum(1 for row in self.rejections if row["verified"] is False)
        return {
            "source": sum(1 for n in self._nodes.values() if n.kind == KIND_SOURCE),
            "derived": sum(1 for n in self._nodes.values() if n.kind == KIND_DERIVED),
            "pages": len(self._pages), "rejected": len(self.rejections),
            "rejected_absent": absent, "rejected_unchecked": len(self.rejections) - absent,
        }

    def to_dict(self) -> Dict[str, Any]:
        """The whole graph as a JSON-serializable artifact, pages included.

        :returns: ``{"pages", "nodes", "rejections", "counts"}``. Pages carry their text and
            SHA-256, so :func:`reverify_graph` needs nothing else.
        :raises: nothing.
        """
        return {
            "pages": [dict(page) for page in self._pages.values()],
            "nodes": [node.as_dict() for node in self._nodes.values()],
            "rejections": [dict(row) for row in self.rejections],
            "counts": self.counts(),
        }

    @classmethod
    def from_dict(cls, artifact: Dict[str, Any]) -> "EvidenceGraph":
        """Rebuild a graph from :meth:`to_dict` output, verification states as recorded.

        :param artifact: a stored artifact.
        :returns: the restored graph. Use :func:`reverify_graph` to RE-DERIVE verification rather
            than trust what was recorded.
        :raises: nothing — malformed entries are skipped.
        """
        graph = cls()
        for page in (artifact or {}).get("pages", []) or []:
            if isinstance(page, dict) and page.get("page_id") is not None:
                graph._pages[str(page["page_id"])] = dict(page)
        for data in (artifact or {}).get("nodes", []) or []:
            if isinstance(data, dict) and data.get("id"):
                node = EvidenceNode.from_dict(data)
                graph._nodes[node.id] = node
        for row in (artifact or {}).get("rejections", []) or []:
            if isinstance(row, dict):
                graph.rejections.append(dict(row))
        return graph


def reverify_graph(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Re-derive every node's verification from a stored artifact alone, offline.

    The audit path, mirroring ``execution_evidence_loop.reverify_cell``: no network, no model, no
    GPU. Each SOURCE node's value is re-located in its PERSISTED page; a node whose page is gone,
    or whose miss falls past a truncated window, is reported unverifiable rather than failed. A
    DERIVED node has no page of its own, so it is reported as verified exactly when every SOURCE
    it transitively rests on re-verifies — the derivation's own arithmetic is the sibling half's
    check, not this one's.

    :param artifact: a :meth:`EvidenceGraph.to_dict` payload.
    :returns: ``{"pages", "nodes": [{node_id, kind, value, page_id, url, verified, fail_reason,
        drifted}], "counts": {"verified", "failed", "unchecked", "source", "derived",
        "page_drift"}}``.
    :raises: nothing — an artifact with no nodes reports zeroes.
    """
    graph = EvidenceGraph.from_dict(artifact or {})
    counts = {"verified": 0, "failed": 0, "unchecked": 0, "source": 0, "derived": 0,
              "page_drift": 0}
    rows: List[Dict[str, Any]] = []
    verdicts: Dict[str, Optional[bool]] = {}

    for node in graph.nodes():
        if node.kind != KIND_SOURCE:
            continue
        counts["source"] += 1
        page = graph.page(node.page_id)
        match = verify_value_against_stored_page(page, node.value)
        drifted = bool(page) and not page.get("truncated") and hash_page_text(
            str(page.get("text", ""))) != str(page.get("content_hash", ""))
        if drifted:
            counts["page_drift"] += 1
        verdicts[node.id] = match.verified
        _tally(counts, match.verified)
        rows.append({
            "node_id": node.id, "kind": node.kind, "value": node.value,
            "page_id": node.page_id, "url": (page or {}).get("url", ""),
            "verified": match.verified, "fail_reason": match.fail_reason, "drifted": drifted,
        })

    for node in graph.nodes():
        if node.kind == KIND_SOURCE:
            continue
        counts["derived"] += 1
        states = [verdicts.get(source.id) for source in graph.sources_of(node.id)]
        verified: Optional[bool] = True if states and all(s is True for s in states) else (
            False if any(s is False for s in states) else None)
        rows.append({
            "node_id": node.id, "kind": node.kind, "value": node.value, "page_id": "",
            "url": "", "verified": verified,
            "fail_reason": None if verified is True else VALUE_FAIL_NO_PAGE if not states
            else VALUE_FAIL_ABSENT if verified is False else VALUE_FAIL_NO_PAGE,
            "drifted": False,
        })
    return {"pages": len(graph.pages()), "nodes": rows, "counts": counts}


def _tally(counts: Dict[str, int], verified: Optional[bool]) -> None:
    if verified is True:
        counts["verified"] += 1
    else:
        counts["failed" if verified is False else "unchecked"] += 1
