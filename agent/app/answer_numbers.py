"""Pure number extraction over an ANSWER string, for mechanical finish-time minting.

No I/O, no model call, no framework import -- the same "attachment surface" discipline
``ledger_tools`` documents for itself, so this module can be imported from a host, from an offline
analysis script, or from a test with nothing else spun up.

Two things live here, both mechanical:

* :func:`extract_answer_numbers` -- the same numeric-token regex and URL-stripping semantics
  ``scripts/ledger_risk_coverage.py`` already uses to read numbers out of a deliverable
  (``_NUM_RE`` / ``strip_urls``), REPLICATED rather than imported so this module has no dependency
  in that direction (the script is analysis-side and imports agent code, not the reverse -- see
  that module's own docstring), plus one thing the script's flat float list does not carry: the
  unit token immediately adjacent to each number, when the answer states one. A bare
  ``38.7`` and a stated ``38.7 metres`` are different claims, and a unit-blind match is exactly the
  failure mode ``LedgerToolkit.audit_answer`` exists to avoid (see that method's docstring) -- so
  this extractor keeps the unit attached at the source instead of discarding it and asking a later
  caller to guess.
* :func:`is_trivial_number` -- a year-like integer (1900-2099, no unit) or a bare integer under
  100 (no unit) is recorded but excluded from the headline ``answer_supported`` predicate. This is
  the wildcard-suppression lesson (``LedgerToolkit._KNOWN_UNITS`` / the corpus's own
  ``Floors\\n104\\nCompleted`` shape) applied at MINTING time instead of at matching time: a page
  count, a rank, a step number or a citation year is not the kind of claim this predicate should be
  able to fail or pass on, so it is set aside rather than forced through the same gate as a real
  measured quantity.
* :func:`operation_appropriateness` -- a conservative, cue-based check of whether a DERIVED node's
  operation plausibly answers what the mandate asked. It targets two real certified-but-wrong
  cases measured live: a height "difference" reported as -30.55 km (the operands were subtracted
  in the wrong order, so a magnitude question got a negative answer) and a dimensionless RATIO
  (2.66) handed back where the mandate asked for a difference. Both are catchable from surface
  cues alone -- "how much taller" implies a non-negative magnitude; "difference" is not "how many
  times" -- without knowing what the correct value actually is. Deliberately silent (``None``)
  whenever the cues are ambiguous or absent: this is not a general-purpose operation classifier,
  it is a narrow, high-precision check for two specific mismatches.

Why this module never imports :mod:`agent.app.ledger_tools`: that module is the higher layer (it
composes this one, plus ``evidence_graph`` and ``quantity_index``, into
:meth:`~agent.app.ledger_tools.LedgerToolkit.audit_answer`); this module imports FROM the layer
below it (``evidence_graph.extract_unit`` / ``canonical_unit``-equivalent helpers) so the
dependency graph stays a DAG, not a cycle.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agent.app.quantity_index import _unit_allowed
from agent.app.testing.evidence_graph import extract_unit as _split_trailing_unit

#: Numeric-token grammar, replicated byte-for-byte from ``scripts/ledger_risk_coverage.py``'s
#: ``_NUM_RE`` (per the API contract: "regex ported ... DO NOT edit that script -- Lane C owns
#: it"). Order matters: thousands-grouped first, then a bare decimal, then a bare integer, so
#: ``"1,642.5"`` is read as ONE token rather than fragmenting on the comma.
_NUM_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,642 / 1,642.5
    r"|-?\d+\.\d+"                     # 419.7
    r"|-?\d+"                          # 381
)

#: Same URL grammar as ``scripts/ledger_risk_coverage.py``'s ``_URL_RE`` -- a citation URL
#: embedded in an answer string routinely CONTAINS digit runs (a page id, a date in the path) that
#: are not the model's claimed value, and stripping the URL substring (never the whole line, which
#: would also discard a same-line answer number) keeps those out of the extraction.
_URL_RE = re.compile(r"https?://\S+")

#: The unit token immediately following a number (optional whitespace, then letters/%/° and an
#: optional trailing ``2``/``3`` for a superscript area/volume unit), so ``"38.7 metres"`` and
#: ``"8,372 km2"`` both keep their unit attached to the number that carries it. Deliberately does
#: NOT cross a sentence boundary or a second number -- an adjacent-only window, the same
#: discipline ``ledger_tools._unit_at_span`` uses for a page span, so a following word that merely
#: happens to start with a letter (the next sentence, a proper noun) is never mistaken for a unit.
_ADJACENT_UNIT_RE = re.compile(r"^[ \t]*([A-Za-z°%µ][A-Za-z°%/]{0,9})([23])?\b")

#: A year-like bare integer, no unit -- the low end of :func:`is_trivial_number`'s range.
_YEAR_LOW, _YEAR_HIGH = 1900, 2099
#: Bare integers strictly below this, with no unit, are also trivial (floor counts, ranks,
#: step numbers -- the corpus shapes ``ledger_tools`` module docstring already names).
_TRIVIAL_BARE_INT_CEILING = 100


def strip_urls(text: str) -> str:
    """``text`` with URL SUBSTRINGS removed (not whole lines).

    Ported verbatim from ``scripts/ledger_risk_coverage.strip_urls`` -- see that function's
    docstring for the line-dropping bug this specific shape (substring, not line) avoids: an
    answer that puts its number and its citation URL on the SAME line must not lose the number
    along with the URL.
    """
    return _URL_RE.sub(" ", text or "")


def extract_answer_numbers(text: str) -> List[Dict[str, Any]]:
    """Every numeric token in ``text``, each with its adjacent unit when the text states one.

    URLs are stripped first (see :func:`strip_urls`) so a citation link's digits never masquerade
    as a claimed value. Each match is then independently checked for a unit token sitting directly
    after it (:data:`_ADJACENT_UNIT_RE`); a token that fails to parse as a float is skipped rather
    than raising, mirroring ``ledger_risk_coverage.extract_numbers``'s own ``try/except``.

    :param text: the deliverable / answer string.
    :returns: a list of ``{"text": <matched number text>, "value": <float>, "unit": <str, "" when
        none>}`` dicts, in order of appearance. Never raises; an empty or non-string ``text``
        yields ``[]``.
    """
    stripped = strip_urls(str(text or ""))
    out: List[Dict[str, Any]] = []
    for match in _NUM_RE.finditer(stripped):
        token = match.group(0)
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        unit = ""
        tail = stripped[match.end():match.end() + 12]
        unit_match = _ADJACENT_UNIT_RE.match(tail)
        if unit_match:
            base, digit_suffix = unit_match.group(1), unit_match.group(2) or ""
            # `extract_unit` (`_split_trailing_unit`) is a WHITELIST-FREE splitter -- it accepts
            # any syntactically unit-shaped tail, including an ordinary following word ("and",
            # "were"). That is exactly the trap `ledger_tools`'s module docstring names ("a
            # whitelist is REQUIRED rather than 'whatever word follows the number'"), so the
            # split's own output is additionally checked against `quantity_index`'s unit
            # whitelist (`_unit_allowed`, the same table `build_index` gates its own extraction
            # on) before it is accepted here. The digit-suffixed form (`"km2"`) is tried FIRST so
            # a genuine area/volume unit is not truncated to its base.
            for candidate in ((base + digit_suffix,) if digit_suffix else ()) + (base,):
                if _split_trailing_unit(f"1 {candidate}") == candidate and _unit_allowed(candidate):
                    unit = candidate
                    break
        out.append({"text": token, "value": value, "unit": unit})
    return out


def is_trivial_number(entry: Dict[str, Any]) -> bool:
    """True for a year-like or small bare integer that should not gate ``answer_supported``.

    Two shapes, both requiring NO unit (a unit-bearing number is never trivial -- ``"1991 m"`` is
    a real elevation, not a year):

    * a plain 4-digit integer in ``[1900, 2099]`` -- a publication year, an event date;
    * a plain integer strictly less than 100 -- a floor count, a rank, a step number, the kind of
      small unlabelled integer the corpus is full of (see the module docstring).

    :param entry: one entry from :func:`extract_answer_numbers`.
    :returns: True when the number should be recorded but excluded from the headline predicate.
    :raises: nothing.
    """
    if str(entry.get("unit") or "").strip():
        return False
    value = entry.get("value")
    if value is None:
        return False
    if float(value) != int(value):
        return False  # a decimal is never a bare year or a bare small integer
    number = int(value)
    if _YEAR_LOW <= number <= _YEAR_HIGH and len(str(abs(number))) == 4:
        return True
    return 0 <= number < _TRIVIAL_BARE_INT_CEILING


#: Cues that unambiguously mark the mandate as asking for a MAGNITUDE (a "how much" comparison
#: between two things), which a correctly-computed answer can never satisfy with a negative
#: number. Conservative and narrow on purpose -- see the module docstring's two target cases.
_MAGNITUDE_CUES = re.compile(
    r"\bdifference\b|\bhow\s+much\s+(?:taller|higher|longer|farther|further|larger|bigger|"
    r"deeper|older|younger|heavier|shorter|wider)\b|\bgap\s+between\b",
    re.IGNORECASE,
)

#: Cues that unambiguously mark the mandate as asking for a RATIO/multiplicative comparison
#: ("how many times", "what factor"), the complement of :data:`_MAGNITUDE_CUES`.
_RATIO_CUES = re.compile(
    r"\bhow\s+many\s+times\b|\bratio\b|\bfactor\s+of\b|\btimes\s+(?:as\s+)?(?:tall|large|big|"
    r"long|high|far|deep|old|heavy)\b",
    re.IGNORECASE,
)

#: Operation names (as recorded on a DERIVED node's ``operation`` attribute, or as passed to
#: ``LedgerToolkit.derive`` / ``add_arith``) that are difference-shaped vs. ratio-shaped.
_DIFFERENCE_OPS = frozenset({"difference", "diff", "subtract", "minus"})
_RATIO_OPS = frozenset({"quotient", "ratio", "divide", "division", "per", "rate"})


def operation_appropriateness(mandate_text: str, operation: str, value: Any,
                              unit: Any = "") -> Dict[str, Optional[bool]]:
    """A conservative, cue-based read of whether ``operation`` plausibly answers ``mandate_text``.

    Two independent, narrow checks, EACH ``None`` unless its own cue is unambiguously present --
    absence of a cue is not evidence of anything, so this never guesses:

    * ``sign_plausible``: ``False`` only when the mandate unambiguously asks for a magnitude
      (:data:`_MAGNITUDE_CUES` -- "difference", "how much taller", "gap between") and the numeric
      result is negative. This is the ``-30.55`` km case: a magnitude question has no business
      being answered with a signed value, regardless of which operand order produced it.
    * ``operation_shape_match``: ``False`` only when (a) the mandate asks a magnitude/difference
      shape but ``operation`` is a quotient/ratio, or (b) the mandate asks a ratio/"how many
      times" shape but ``operation`` is a difference. This is the ``2.66`` dimensionless-ratio
      case: a difference was asked for and a rate was given instead.

    Neither check ever fires on a magnitude cue's absence -- a mandate with no unambiguous shape
    cue (most of them) reports ``None`` for both, which is the correct "we don't know" answer, not
    a pass or a fail.

    :param mandate_text: the task mandate / question text.
    :param operation: the operation name on the DERIVED node under review (e.g. ``"difference"``,
        ``"quotient"``, ``"ratio"``, or an alias ``ledger_tools._ALIASES`` maps to one of those).
    :param value: the node's numeric value (or a string convertible to one); ``sign_plausible`` is
        ``None`` when this is not a real number.
    :param unit: accepted for symmetry with the rest of this module's call shape; unused today --
        no cue here currently depends on the unit string, and no future one should invent a unit
        conversion to get one (see ``evidence_graph``'s "no conversion, ever" non-goal).
    :returns: ``{"sign_plausible": True/False/None, "operation_shape_match": True/False/None}``.
    :raises: nothing.
    """
    del unit  # symmetry only -- see docstring
    mandate = str(mandate_text or "")
    op = str(operation or "").strip().lower()

    sign_plausible: Optional[bool] = None
    if _MAGNITUDE_CUES.search(mandate):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            sign_plausible = not (numeric < 0)

    shape_match: Optional[bool] = None
    magnitude_cue = bool(_MAGNITUDE_CUES.search(mandate))
    ratio_cue = bool(_RATIO_CUES.search(mandate))
    if magnitude_cue and not ratio_cue and op in _RATIO_OPS:
        shape_match = False
    elif ratio_cue and not magnitude_cue and op in _DIFFERENCE_OPS:
        shape_match = False
    elif (magnitude_cue and op in _DIFFERENCE_OPS) or (ratio_cue and op in _RATIO_OPS):
        shape_match = True

    return {"sign_plausible": sign_plausible, "operation_shape_match": shape_match}
