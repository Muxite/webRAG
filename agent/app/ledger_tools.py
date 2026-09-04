"""The Euglena Ledger as an attachable module, bindable to a host agent we do not own.

``docs/LEDGER.md`` states the product plainly -- "It is a component, not an agent" -- but every
experiment in this repo has measured it as a RIVAL agent, comparing ``evidence_loop`` against
``langgraph_react``. Those two differ in loop, prompt, budget, step policy and output contract
simultaneously, so a measured delta could never be attributed to the ledger itself; and measured
on 2026-09-01, the resulting per-arm KPI orderings INVERT between task subsets of the same suite
(see ``docs/LEDGER_FINAL01_TUNING_RESULT.md``). This module exists so the question can instead be
asked in the form the product claims: **host vs host + module**, one change at a time, inside a
single system.

What a host gets by binding :meth:`LedgerToolkit.derive`:

* Every derived number is **recomputed in Python**, never accepted because a model asserted it.
* Every operand must be **located on a page the host actually fetched**, or the derivation is
  refused -- so a derived value is traceable to text, not to the model's memory.
* Incompatible units are **refused rather than converted** (``LEDGER_PLAN`` section 7 non-goal).
* The whole attempt -- operands, operation, recomputed value, refusal code -- is recorded in an
  artifact a third party can re-verify offline with ``evidence_graph.reverify_graph``.

Nothing here reimplements arithmetic, unit logic or value location: it composes
:mod:`agent.app.testing.evidence_graph`, which is the tested home of all three. This module is
only the *attachment surface* -- host-neutral, with no langgraph, langchain or test-runner import,
so a ReAct loop, a DAG engine or a plain script can bind it the same way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from agent.app.quantity_index import build_index, lookup, render_index
from agent.app.testing.evidence_graph import (DerivationError, EvidenceGraph,
                                              _numbers_agree, extract_unit, numeric_value,
                                              parse_quantity, verify_value)

#: Operations the module will recompute. Deliberately the closed set ``evidence_graph.add_arith``
#: already implements and tests -- a host cannot widen it by passing a new name.
SUPPORTED_OPERATIONS = ("sum", "difference", "product", "quotient", "ratio")

#: Aliases weak models actually emit for the operations above, so a correct intent is not thrown
#: away over vocabulary. Mirrors the alias table ``execution_evidence_loop`` already needed live.
_ALIASES = {
    "add": "sum", "plus": "sum", "total": "sum", "subtract": "difference", "minus": "difference",
    "diff": "difference", "multiply": "product", "times": "product", "divide": "quotient",
    "division": "quotient", "per": "ratio", "rate": "ratio",
}


def _disagrees(proposed: Any, computed: Any) -> bool:
    """True when a model's proposed value is numerically different from the recomputation.

    Reuses ``evidence_graph``'s own comparison so "agrees" means the same thing here as it does
    when a node records its validity. A proposal that cannot be parsed as a quantity at all is NOT
    called a disagreement -- it is unreadable, which is a different failure and not this
    function's to name.
    """
    if proposed is None:
        return False
    left, right = numeric_value(proposed), numeric_value(computed)
    if left is None or right is None:
        return False
    return not _numbers_agree(left, right)



#: Unit tokens this module will accept from a page span. A whitelist is REQUIRED rather than
#: "whatever word follows the number": the corpus is full of `Floors\n104\nCompleted` and
#: `Population\n8,336,817\nand rising`, and treating an adjacent capitalised word as a unit would
#: manufacture exactly the false dimension mismatches this reader exists to remove. Measured live:
#: 8 of 19 UNIT_MISMATCH refusals were already spurious before any of this was added.
_KNOWN_UNITS = frozenset("""
    m km cm mm ft feet foot mi mile miles yd in inch inches nmi
    m2 km2 cm2 ft2 mi2 ha acre acres m3 km3 ft3 l litre litres gal
    kg g mg t tonne tonnes lb lbs oz
    s sec secs min mins h hr hrs day days week weeks month months year years
    k mw gw kw w mwh gwh kwh wh j kj mj v kv a ma hz khz mhz ghz
    c f pa kpa mpa bar psi
    metre metres meter meters kilometre kilometres kilometer kilometers
    percent pct
""".split())

#: Spelling variants collapsed to one canonical token BEFORE any dimension comparison. This is
#: orthography, never conversion: no magnitude is ever touched, so the "no unit or currency
#: conversion, ever" non-goal (`docs/LEDGER_PLAN_2026-09-01.md` section 7) stands untouched.
_UNIT_CANONICAL = {
    "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km",
    "feet": "ft", "foot": "ft", "mile": "mi", "miles": "mi",
    "inch": "in", "inches": "in", "tonne": "t", "tonnes": "t",
    "second": "s", "seconds": "s", "sec": "s", "secs": "s",
    "minute": "min", "minutes": "min", "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "km²": "km2", "m²": "m2", "cm²": "cm2", "ft²": "ft2", "mi²": "mi2",
    "km³": "km3", "m³": "m3", "percent": "%", "pct": "%",
}

#: The unit token immediately after a value, plus an optional superscript that Wikipedia's
#: flattened infoboxes drop onto its own line (``Surface area\n8,372\nkm\n2``).
_SPAN_UNIT_RE = re.compile(r"\s*([A-Za-z°%µ]{1,10}|°[CF])\s*\n?\s*([23])?\b")


def canonical_unit(unit: Any) -> str:
    """One spelling per unit, lowercased, so `metres` and `m` compare equal.

    SPELLING only. Nothing here rescales a magnitude, so this is not the unit conversion the
    project forbids -- it is the difference between comparing dimensions and comparing typography.
    """
    token = str(unit or "").strip().lower().replace("\n", "")
    return _UNIT_CANONICAL.get(token, token)


def _unit_at_span(page_text: str, start: int, end: int) -> str:
    """The unit the PAGE gives a located value, or "" when the page states it bare.

    This closes the failure that motivated the phase. Task 221 live: a model derived three ratios,
    each `derivation_valid=True` and each operand located on a real page, then compared
    feet-per-floor with metres-per-floor and scored 0.16. The unit guard never fired because the
    model passed BARE NUMBERS -- the source nodes carried ``unit=''``, so the check had nothing to
    check. Measured across the corpus, 91.2% of one host's source nodes were unitless, which makes
    the guard structurally dead in that arm rather than merely weak.

    The unit was never missing from the evidence, only from what the model typed: it sits directly
    after the span the module already located. The general rule this instances: **every
    verification input the agent supplies is a surface the agent can disable by omitting it, so
    derive it from stored evidence instead.**

    Why the token is matched rather than handed to ``parse_quantity`` over a window: that function
    treats everything trailing as the unit when nothing is left over, so a generous window returns
    ``'ft\nFloors\n104'`` and a narrow one truncates ``'km'`` to ``'k'``. Neither is a unit.

    :param page_text: raw page text the span indexes into.
    :param start: span start offset.
    :param end: span end offset (exclusive).
    :returns: canonical unit as the page writes it, or "" when there is none to read.
    """
    match = _SPAN_UNIT_RE.match(page_text, end)
    if not match:
        return ""
    token = canonical_unit(match.group(1))
    if match.group(2):
        squared = f"{token}{match.group(2)}"
        if squared in _KNOWN_UNITS:
            return squared
    return token if token in _KNOWN_UNITS else ""


#: Cap on a line-expanded quote. A whole line is usually a short infobox row or one prose
#: sentence, but nothing bounds how long a REAL line can be, so a runaway line falls back to the
#: bare value span rather than handing `EvidenceGraph.add_source` (and, downstream, a human
#: auditor) an unbounded string.
_MAX_QUOTE_CHARS = 300


def _quote_for_span(page_text: str, start: int, end: int, *, max_chars: int = _MAX_QUOTE_CHARS) -> str:
    """A verbatim, independently-verifiable quote for the span ``page_text[start:end]``.

    Mechanical only -- no model call, no rewriting. Expands the located value span to its
    containing line (split on ``"\\n"``, the same line shape ``quantity_index`` already reads
    this corpus with) so the stored quote is a checkable sentence or infobox row rather than bare
    digits, then falls back to the bare span when that line is too long to be a useful "quote"
    (``max_chars``). Either way the return value is a literal substring of ``page_text`` --
    :func:`~agent.app.testing.execution_evidence_loop.verify_quote`'s exact-substring check is
    therefore guaranteed to pass, so this never manufactures a `quote_fail_reason`.

    :param page_text: the SAME text the span's offsets were computed against (a graph's stored,
        possibly-truncated page window -- never a different copy of the page).
    :param start: span start offset, inclusive.
    :param end: span end offset, exclusive.
    :param max_chars: line-length cap before falling back to the bare span.
    :returns: a non-empty literal substring of ``page_text`` on success, or ``""`` when
        ``start``/``end`` do not index into ``page_text`` at all (e.g. an id resolved against a
        page whose stored window was truncated shorter than the offset it recorded).
    :raises: nothing.
    """
    text = page_text if isinstance(page_text, str) else ""
    if not (0 <= start <= end <= len(text)):
        return ""
    span = text[start:end]
    if not span:
        return ""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end].strip()
    if line and len(line) <= max_chars:
        return line
    return span


class LedgerToolkit:
    """Per-run ledger state a host binds one or more tools to.

    :param max_page_chars: cap on the stored window per page; the content hash still covers the
        whole fetched text, so a shortened window still detects drift.
    """

    def __init__(self, max_page_chars: int = 6000) -> None:
        self._graph = EvidenceGraph()
        self._max_page_chars = int(max_page_chars)
        self._pages = 0
        #: page_id -> the quantities `quantity_index.build_index` found on that page's text, so a
        #: `q`-id operand can be resolved without asking the model to retype anything. Built from
        #: the SAME text handed to `register_page`, so an id's offsets always agree with the page
        #: `derive`'s literal-operand path already locates against.
        self._indexes: Dict[str, List] = {}
        #: (page_id, QuantityRef) in id order. Ids are issued ONCE across the whole
        #: run and never reused -- see `_resolve_id` for what per-page numbering did.
        self._entries: List[Any] = []
        self._index_start: Dict[str, int] = {}

    # -- host hooks ---------------------------------------------------------------------------

    def register_page(self, url: str, text: str) -> str:
        """Freeze a page the host just fetched, making its values eligible as operands.

        A host calls this from wherever it already visits pages. Until a page is registered,
        nothing on it can ground a derivation -- which is the property that makes a derived value
        traceable rather than merely plausible.

        :param url: the fetched URL.
        :param text: the fetched page text.
        :returns: the page id operands will resolve against.
        """
        self._pages += 1
        page_id = f"p{self._pages}"
        self._graph.add_page(page_id, url, text or "", self._max_page_chars)
        entries = build_index(text or "")
        self._indexes[page_id] = entries
        self._index_start[page_id] = len(self._entries) + 1
        self._entries.extend((page_id, entry) for entry in entries)
        return page_id

    def page_index_text(self, page_id: str, *, max_chars: int = 1200) -> str:
        """The rendered quantity index for ``page_id``, for a host to show the model.

        :param page_id: id returned by :meth:`register_page`.
        :param max_chars: forwarded to :func:`quantity_index.render_index`.
        :returns: the rendered ``q<N>: ...`` block, or ``""`` when the page had nothing extractable
            (or ``page_id`` is unknown) -- never a header over nothing.
        """
        key = str(page_id)
        entries = self._indexes.get(key) or []
        return render_index(entries, max_chars=max_chars, start=self._index_start.get(key, 1))

    # -- the tool ------------------------------------------------------------------------------

    def derive(self, operation: str, operands: Sequence[str],
               proposed_value: Any = None) -> str:
        """Recompute ``operation`` over ``operands``, or refuse with a reason the model can act on.

        Every refusal is an OBSERVATION, never an exception: a host's loop must be able to keep
        going, and a weak model needs to be told what was wrong rather than merely that something
        was. This mirrors the refusal-as-observation contract ``execution_evidence_loop`` arrived
        at against live weak models.

        :param operation: one of :data:`SUPPORTED_OPERATIONS`, or a known alias.
        :param operands: value strings, each of which must be locatable on a registered page.
        :param proposed_value: what the model thinks the answer is. NEVER used as the result --
            only compared against the recomputation, with a disagreement recorded and the node
            marked invalid.
        :returns: an observation string for the host's transcript.
        :raises: nothing.
        """
        name = _ALIASES.get(str(operation or "").strip().lower(), str(operation or "").strip().lower())
        if name not in SUPPORTED_OPERATIONS:
            return (f"DERIVE REFUSED (UNKNOWN_OPERATION): {operation!r} is not one of "
                    f"{', '.join(SUPPORTED_OPERATIONS)}.")
        values = [str(v) for v in (operands or [])]
        if len(values) < 2:
            return "DERIVE REFUSED (WRONG_ARITY): give at least two operands."

        input_ids: List[str] = []
        for value in values:
            node = self._locate(value)
            if node is None:
                return (f"DERIVE REFUSED (OPERAND_NOT_ON_PAGE): {value!r} was not found on any "
                        "page you have read. Visit a page that states it, then derive again.")
            input_ids.append(node)

        try:
            node = self._graph.add_arith(name, input_ids, proposed_value=proposed_value)
        except DerivationError as exc:
            return f"DERIVE REFUSED ({exc.code}): {exc}"

        # `add_derived` dedups on a content-identical id and KEEPS THE FIRST node, so a model that
        # derived this correctly earlier and now asserts something else gets the old success handed
        # back with its `derivation_valid=True` intact. That is right for the graph -- the stored
        # fact did not change -- but it would let a late fabricated proposal ride in unreported and
        # leave the fabricated-arithmetic rate reading 0.0. So the disagreement is recomputed here,
        # against the node's value, independently of what the node recorded on first admission.
        disagrees = node.derivation_valid is False or _disagrees(proposed_value, node.value)
        detail = (f" (your proposed {proposed_value!r} disagrees with the recomputation, "
                  "which stands)") if disagrees else ""
        unit = f" {node.unit}" if node.unit else ""
        return f"DERIVED {name} = {node.value}{unit}{detail}"

    #: An operand shape that names a quantity-index entry rather than retyping it: an explicit
    #: leading ``q``/``Q`` is REQUIRED (unlike `quantity_index._REF_PATTERN`, which also accepts a
    #: bare number for a model tolerant of dropping the letter). A literal operand routinely IS a
    #: bare number (a floor count, a year) and must keep working as a literal, so id resolution
    #: must never claim a bare digit string for itself -- see the HARD RULE in the task brief.
    _ID_OPERAND_RE = re.compile(r"^\s*[Qq]\s*0*([1-9]\d*)\s*\.?\s*$")

    def _resolve_id(self, value: str):
        """The ``(page_id, QuantityRef)`` a ``q``-shaped ``value`` names, or ``None``.

        Ids index one FLAT list issued across the whole run, never per page. Per-page numbering
        was tried and is silently wrong: every page's index restarted at ``q1``, so a model shown
        ``q1: Height = 1776 ft`` after visiting the SECOND page and passing ``q1`` received the
        FIRST page's ``1,642 m`` -- ``sum(q1, "541 m")`` returned 2183 (1642+541). The unit guard
        then PASSED, because the substituted quantity happened to be in metres, masking the ft/m
        mismatch that should have refused. A confidently wrong number carrying full provenance is
        exactly what this module exists to prevent, so an id means one thing for the whole run.

        A value that is not ``q``-shaped (a bare number, a year, a floor count) is never attempted
        here and falls straight through to the literal path.
        """
        match = self._ID_OPERAND_RE.match(str(value or ""))
        if not match:
            return None
        position = int(match.group(1))
        if 1 <= position <= len(self._entries):
            return self._entries[position - 1]
        return None

    def _locate(self, value: str) -> Optional[str]:
        """The id of a SOURCE node for ``value`` on any registered page, admitting it if needed.

        Tries every registered page rather than asking the model which one to use: a weak model
        routinely cites the wrong page for a value it did read, and refusing that would report a
        citation slip as a fabrication.

        A ``q``-shaped operand (:meth:`_resolve_id`) is tried FIRST: it names an entry the
        quantity index already extracted, complete with the unit AS THE PAGE WROTE IT, so the
        located node carries that unit even when the model typed no unit at all. When it doesn't
        resolve (unknown id, or the operand isn't id-shaped), this falls through to the literal
        path unchanged -- ids are offered, never required.

        The operand is offered BOTH as a split number+unit pair and as the literal string the model
        wrote. The split form matters: ``evidence_graph`` knows ``m`` and ``metres`` are the same
        unit, but only when the unit is supplied separately from the number -- its candidate
        matching is built for that shape. Passing the whole string as a literal instead refuses a
        page reading ``1,470\nm`` when the model writes ``"1,470 metres"``, which was observed
        live on task 211 and is over-refusal, not caution: it blocks a value the model genuinely
        read and inflates the module's own averted-fabrication count with its own parsing failures.
        """
        text = str(value or "").strip()
        resolved = self._resolve_id(text)
        if resolved is not None:
            page_id, entry = resolved
            page = self._graph.page(page_id)
            page_text = str((page or {}).get("text") or "")
            quote = _quote_for_span(page_text, entry.start, entry.end)
            node = self._graph.add_source(page_id, entry.value, quote=quote, unit=entry.unit or None)
            if node is not None:
                return node.id
        unit = extract_unit(text)
        number = text[:len(text) - len(unit)].strip() if unit and text.endswith(unit) else ""
        attempts = [(text, None)]
        if number and unit:
            attempts.insert(0, (number, unit))
        for page in self._graph.pages():
            page_text = str(page.get("text") or "")
            for candidate, candidate_unit in attempts:
                match = verify_value(page_text, candidate, unit=candidate_unit)
                if not match.verified:
                    continue
                span_unit = candidate_unit or _unit_at_span(page_text, match.start, match.end)
                quote = _quote_for_span(page_text, match.start, match.end)
                node = self._graph.add_source(
                    page["page_id"], candidate, quote=quote, unit=span_unit or None)
                if node is not None:
                    return node.id
        return None

    # -- audit surface -------------------------------------------------------------------------

    def artifact(self) -> Dict[str, Any]:
        """The re-verifiable record of everything this run derived.

        Shaped exactly like ``execution_evidence_loop``'s ``output.evidence_graph`` so the existing
        offline auditors -- ``evidence_graph.reverify_graph``, ``scripts/reverify.py``,
        ``scripts/claim_metrics.derivation_fabrication_rate`` -- read a host's artifact with no
        changes at all.
        """
        return self._graph.to_dict()
