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

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from agent.app.answer_numbers import (extract_answer_numbers, is_trivial_number,
                                      mandate_demanded_operation, operation_appropriateness)
from agent.app.quantity_index import build_index, lookup, render_index
from agent.app.testing.evidence_graph import (KIND_DERIVED, DerivationError, EvidenceGraph,
                                              UnknownOperation, WrongArity, _numbers_agree,
                                              canonical_unit, extract_unit, numeric_value,
                                              parse_quantity, verify_value)


class OperandNotOnPage(DerivationError):
    """An operand names no quantity on any page this toolkit has registered.

    The graph's own :class:`~agent.app.testing.evidence_graph.MissingOperand` is the neighbouring
    refusal -- an id naming no node -- but this one happens EARLIER and means something different:
    the host never read a page stating the value at all, so there is no id to miss. Keeping the
    two codes apart is the whole point of recording refusals: "the model invented a number" and
    "the model mistyped a reference" are different failures with different fixes.
    """

    code = "OPERAND_NOT_ON_PAGE"

#: Provenance tag stamped on every node minted by :meth:`LedgerToolkit.audit_answer`, so a
#: consumer (e.g. the risk-coverage certify chain) can include or exclude this mechanical,
#: finish-time minting path from the model-driven ``derive`` path's own accounting.
ANSWER_AUDIT_TAG = "answer_audit"

#: Provenance tag stamped on every node minted by :meth:`LedgerToolkit.shape_derive_check`. A
#: SEPARATE tag from :data:`ANSWER_AUDIT_TAG` even though the two mint the same NODE SHAPES
#: (SOURCE operands + a DERIVED result) via the same helpers -- a consumer that wants to know
#: whether the ANSWER itself was mechanically confirmed to match the mandate's DEMANDED operation
#: (this method) needs to distinguish that from the broader "some arithmetic explains this number"
#: search :meth:`audit_answer` already performs, which does not know or care what the mandate
#: asked for.
SHAPE_DERIVE_TAG = "shape_derive"

#: :meth:`LedgerToolkit.shape_derive_check` refuses to search once the number of distinct
#: candidate (operation, value) results it would have to compare the answer against exceeds this
#: -- the same "candidate explosion" guard shape as :meth:`audit_answer`'s ambiguity accounting,
#: applied BEFORE the search rather than after, since here the search is one single demanded
#: operation over every pair rather than four operations, so a bound before computing is cheap and
#: keeps a large index (a model that revisited the same handful of pages many times) from turning
#: this method into an O(n^2) scan with no shape at all to show for it.
SHAPE_DERIVE_MAX_CANDIDATES = 40

#: Relative tolerance for matching an ANSWER's number against the run's quantity index / a
#: mechanical derivation. Separate from ``evidence_graph.ARITH_RELATIVE_TOLERANCE`` (``1e-6``,
#: which absorbs only Python float round-trip noise on an INTERNAL recomputation): this tolerance
#: absorbs the answer text's own rounding (a model reporting "38.7" for a page's "38.70000...").
#: Matches ``scripts/ledger_risk_coverage.REL_TOL`` per the prereg'd contract.
AUDIT_REL_TOL = 0.005

#: The mechanical derivation vocabulary :meth:`LedgerToolkit.audit_answer` searches, in
#: DETERMINISTIC preference order for which explanation gets minted when more than one fits (see
#: :meth:`LedgerToolkit._best_explanation`). A strict subset of :data:`SUPPORTED_OPERATIONS` --
#: "ratio" is not attempted separately since it is the same division as "quotient" under a
#: different display name (see ``add_arith``'s docstring).
_DERIVATION_OPS = ("difference", "quotient", "sum", "product")
_OP_PREFERENCE = {op: index for index, op in enumerate(_DERIVATION_OPS)}

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

#: The unit token immediately after a value, plus an optional superscript that Wikipedia's
#: flattened infoboxes drop onto its own line (``Surface area\n8,372\nkm\n2``).
_SPAN_UNIT_RE = re.compile(r"\s*([A-Za-z°%µ]{1,10}|°[CF])\s*\n?\s*([23])?\b")


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
        # Every exit below records the refusal on the graph before returning the observation. The
        # observation alone is not a record: it goes into the host's scratchpad, which no result
        # cell persists, and a refusal mints no node -- so without this a correct refusal and a
        # derivation never attempted are the same artifact. The three EARLY exits happen before
        # any operand is located, so they carry the raw operand strings as their provenance (the
        # only thing there is), matching the evidence loop's own pre-graph refusal at
        # ``execution_evidence_loop._handle_derive``.
        name = _ALIASES.get(str(operation or "").strip().lower(), str(operation or "").strip().lower())
        values = [str(v) for v in (operands or [])]
        if name not in SUPPORTED_OPERATIONS:
            message = (f"{operation!r} is not one of {', '.join(SUPPORTED_OPERATIONS)}")
            self._graph.record_refusal(str(operation), values, UnknownOperation(message))
            return (f"DERIVE REFUSED (UNKNOWN_OPERATION): {operation!r} is not one of "
                    f"{', '.join(SUPPORTED_OPERATIONS)}.")
        if len(values) < 2:
            self._graph.record_refusal(name, values,
                                       WrongArity("give at least two operands"))
            return "DERIVE REFUSED (WRONG_ARITY): give at least two operands."

        input_ids: List[str] = []
        for value in values:
            node = self._locate(value)
            if node is None:
                self._graph.record_refusal(name, values, OperandNotOnPage(
                    f"{value!r} was not found on any page you have read"))
                return (f"DERIVE REFUSED (OPERAND_NOT_ON_PAGE): {value!r} was not found on any "
                        "page you have read. Visit a page that states it, then derive again.")
            input_ids.append(node)

        try:
            node = self._graph.add_arith(name, input_ids, proposed_value=proposed_value)
        except DerivationError as exc:
            self._graph.record_refusal(name, input_ids, exc)
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

    def _entry_numeric(self, entry: Any) -> Optional[float]:
        """``entry``'s full magnitude (value AND unit/scale combined), or None when unparseable.

        Combining the two into one string before parsing is required, not cosmetic: a
        `QuantityRef` built from a scale-worded quantity (``"237.8 million"``) carries the digits
        in ``value`` and the scale word in ``unit`` SEPARATELY, and ``numeric_value`` reads the
        scale multiplier only when it appears in the SAME string it parses -- see
        ``evidence_graph.numeric_value``'s own warning that its ``unit=`` kwarg is NOT used to
        recover a dropped scale. Passing ``entry.value`` alone here would silently read
        ``237.8`` instead of ``237,800,000``, the exact class of silent-loss bug that module's
        docstring documents at length.
        """
        text = f"{entry.value} {entry.unit}".strip() if entry.unit else str(entry.value)
        return numeric_value(text)

    def _unit_consistent(self, answer_unit: str, entry_unit: str) -> Optional[bool]:
        """Whether an answer-stated unit agrees with an index entry's unit -- the B1 gate.

        Three cases, deliberately asymmetric between "the answer said nothing" and "the page said
        nothing": an answer that states NO unit cannot be judged either way (``None``) -- it made
        no claim to check. An answer that DOES state a unit against an entry that carries NONE is
        graded ``False``, not ``None``: the entry offers nothing to confirm the claimed dimension,
        and admitting it anyway is exactly the bare-number-page-scan failure this method's
        docstring (and the B1 panel objection it answers) exists to refuse -- measured live as
        task 221's feet-per-floor/metres-per-floor mismatch riding through on unitless operands.
        A genuine mismatch (both present, different) is naturally ``False`` too.
        """
        answer_text = str(answer_unit or "").strip()
        entry_text = str(entry_unit or "").strip()
        if not answer_text:
            return None
        if not entry_text:
            return False
        return canonical_unit(answer_text) == canonical_unit(entry_text)

    def _mint_source_from_entry(self, page_id: str, entry: Any) -> Optional[Any]:
        """A SOURCE node for a quantity-index ``entry``, minted exactly like :meth:`_locate`'s
        ``q``-id path (quote via :func:`_quote_for_span`, unit as the page wrote it), tagged
        :data:`ANSWER_AUDIT_TAG`. Shared by both the direct-match and the two-operand-derivation
        halves of :meth:`audit_answer` so a page's SOURCE node has exactly one minting recipe.
        """
        page = self._graph.page(page_id)
        page_text = str((page or {}).get("text") or "")
        quote = _quote_for_span(page_text, entry.start, entry.end)
        return self._graph.add_source(page_id, entry.value, quote=quote, unit=entry.unit or None,
                                      minted_by=ANSWER_AUDIT_TAG)

    def _find_backed_match(self, target: float, answer_unit: str):
        """The first index entry that numerically matches ``target`` with a NOT-``False`` unit
        verdict, or ``None``. A numerically matching entry whose unit conflicts is skipped, not
        returned -- per the contract, a unit-inconsistent match does not count as backed and the
        search keeps going for a consistent one elsewhere in the index.

        :returns: ``(page_id, entry, unit_consistent)`` or ``None``.
        """
        for page_id, entry in self._entries:
            value = self._entry_numeric(entry)
            if value is None or not math.isclose(value, target, rel_tol=AUDIT_REL_TOL,
                                                  abs_tol=1e-9):
                continue
            consistent = self._unit_consistent(answer_unit, entry.unit)
            if consistent is False:
                continue
            return page_id, entry, consistent
        return None

    def _find_unit_anchored_page_match(self, item: Dict[str, Any]):
        """A verbatim page occurrence of ``item``'s number WITH its stated unit adjacent, or
        ``None``. Exists for values the quantity index does not extract -- a parenthetical
        conversion like ``419.7 metres (1,377\nft)`` left the answer's honest ``1,377 ft``
        permanently unbacked on the mint01 smoke. The unit ANCHORS the match (``verify_value``
        is passed it explicitly), so this is `_locate`'s split number+unit path, never its
        unit-blind literal fallback -- an answer number that states no unit gets no page scan
        at all.

        :returns: ``(page_id, start, end)`` or ``None``.
        """
        unit = str(item.get("unit") or "")
        if not unit:
            return None
        text = str(item.get("text") or "")
        for page in self._graph.pages():
            match = verify_value(str(page.get("text") or ""), text, unit=unit)
            # `verify_value` falls back to a bare-number hit when the combined value+unit span
            # is not on the page ("a bare value is reported as bare, never silently refused").
            # Bare is exactly what this path must refuse: without `unit_bearing` the 1776-ft
            # answer backs against a metres page and the unit anchor anchors nothing.
            if match.verified and match.unit_bearing:
                return page["page_id"], match.start, match.end
        return None

    #: Unit-compatibility rule per derivation op -- see :meth:`audit_answer`'s contract summary.
    def _compat_diff_sum(self, unit_a: str, unit_b: str) -> bool:
        return canonical_unit(unit_a) == canonical_unit(unit_b)

    def _compat_quotient(self, unit_a: str, unit_b: str) -> bool:
        """Bug 4 fix: quotient used to require the SAME unit on both sides (this method's body
        used to be identical to :meth:`_compat_diff_sum`'s), so a rate computation like
        distance/time (km / h -> km/h) could never be recognized as a derivation explanation --
        structurally, not just as a missed case -- even though
        :meth:`~agent.app.testing.evidence_graph.EvidenceGraph.add_arith` has always composed a
        compound ``"A/B"`` unit for exactly this quotient/ratio shape.

        Now accepts EITHER the pre-existing same-unit case (a dimensionless ratio, e.g. km/km,
        including both sides unitless) OR two DIFFERENT but both-PRESENT canonical units (a rate,
        e.g. km/h). A unit present on only one side still refuses -- that asymmetry is not
        "two units combining into a rate", it is one genuinely unitless operand, which stays
        refused exactly as before.
        """
        canon_a, canon_b = canonical_unit(unit_a), canonical_unit(unit_b)
        if canon_a == canon_b:
            return True
        return bool(canon_a) and bool(canon_b)

    def _compat_product(self, unit_a: str, unit_b: str) -> bool:
        return not canonical_unit(unit_a) or not canonical_unit(unit_b)

    def _find_derivation_explanations(self, target: float) -> List[Dict[str, Any]]:
        """Every distinct ``(operation, unordered entry pair)`` that recomputes to ``target``.

        Closed, mechanical vocabulary -- :data:`_DERIVATION_OPS` -- over EVERY unordered pair of
        index entries. ``difference`` / ``quotient`` are directional (``a OP b != b OP a`` in
        general), so both orderings are tried, but a pair that matches in either direction is
        still counted as ONE explanation for that op -- ambiguity counts DISTINCT explanations,
        not directions, per the API contract.

        :returns: a list of ``{"operation", "i", "j", "order"}`` dicts, ``i``/``j`` are positions
            into ``self._entries`` (``i < j``), ``order`` is ``(i, j)`` or ``(j, i)`` -- the
            argument order that produced the match, for :meth:`audit_answer` to mint with.
        """
        found: List[Dict[str, Any]] = []
        numerics: List[Optional[float]] = [self._entry_numeric(entry) for _, entry in self._entries]
        for i in range(len(self._entries)):
            value_i = numerics[i]
            if value_i is None:
                continue
            unit_i = self._entries[i][1].unit
            for j in range(i + 1, len(self._entries)):
                value_j = numerics[j]
                if value_j is None:
                    continue
                unit_j = self._entries[j][1].unit

                if self._compat_diff_sum(unit_i, unit_j):
                    if math.isclose(value_i - value_j, target, rel_tol=AUDIT_REL_TOL, abs_tol=1e-9):
                        found.append({"operation": "difference", "i": i, "j": j, "order": (i, j)})
                    elif math.isclose(value_j - value_i, target, rel_tol=AUDIT_REL_TOL, abs_tol=1e-9):
                        found.append({"operation": "difference", "i": i, "j": j, "order": (j, i)})
                    if math.isclose(value_i + value_j, target, rel_tol=AUDIT_REL_TOL, abs_tol=1e-9):
                        found.append({"operation": "sum", "i": i, "j": j, "order": (i, j)})

                if self._compat_quotient(unit_i, unit_j):
                    if value_j != 0 and math.isclose(value_i / value_j, target,
                                                      rel_tol=AUDIT_REL_TOL, abs_tol=1e-9):
                        found.append({"operation": "quotient", "i": i, "j": j, "order": (i, j)})
                    elif value_i != 0 and math.isclose(value_j / value_i, target,
                                                        rel_tol=AUDIT_REL_TOL, abs_tol=1e-9):
                        found.append({"operation": "quotient", "i": i, "j": j, "order": (j, i)})

                if self._compat_product(unit_i, unit_j):
                    if math.isclose(value_i * value_j, target, rel_tol=AUDIT_REL_TOL, abs_tol=1e-9):
                        found.append({"operation": "product", "i": i, "j": j, "order": (i, j)})
        return found

    @staticmethod
    def _best_explanation(explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The explanation to actually mint, by the contract's deterministic preference order:
        ``difference`` before ``quotient`` before ``sum`` before ``product``, ties broken by
        first ``(i, j)`` encountered."""
        return min(explanations, key=lambda item: (_OP_PREFERENCE[item["operation"]],
                                                    item["i"], item["j"]))

    def audit_answer(self, answer_text: str, mandate: str = "") -> Dict[str, Any]:
        """Mechanically grade every number in ``answer_text`` against evidence, host-side, at
        finish time -- no model call, no prompt change, runs even for a cell that never invoked
        :meth:`derive`.

        Grading, per number (see the API contract this implements):

        * **backed** -- the number matches (within :data:`AUDIT_REL_TOL`) a quantity-index entry
          this run already extracted (:meth:`register_page`'s ``self._entries``), with a
          NOT-``False`` unit verdict (:meth:`_unit_consistent`). NEVER a bare-number page scan:
          matching is against the index's structured entries, which carry a unit whenever the
          page states one, so a unit mismatch cannot be silenced by omission the way a raw
          ``verify_value(..., unit=None)`` call could be (the B1 panel objection this method
          exists to close). The SOURCE node is minted exactly like the model-driven path mints
          one (:meth:`_mint_source_from_entry`), tagged :data:`ANSWER_AUDIT_TAG`.
        * **derived** -- not on any single page, but mechanically explainable as ONE of
          ``difference`` / ``sum`` / ``quotient`` / ``product`` over TWO index entries with
          compatible units (:meth:`_find_derivation_explanations`). The DERIVED node is minted
          via :meth:`~agent.app.testing.evidence_graph.EvidenceGraph.add_arith`, so
          ``derivation_valid`` is computed HONESTLY by the graph's own independent recomputation,
          not merely asserted here; its two operands are minted as SOURCE nodes the same way.
          ``ambiguity`` records how many DISTINCT ``(operation, entry pair)`` explanations fit --
          when more than one, the best by the contract's deterministic preference order is the
          one actually minted, but the count itself is what a caller weighs.
        * **unbacked** -- neither.

        Every number is graded and recorded, including a TRIVIAL one (:func:`is_trivial_number` --
        a bare year or a small bare integer): trivial numbers are excluded only from the headline
        ``answer_supported`` predicate, per the wildcard-suppression lesson applied at minting
        rather than at matching (see the module-level design note this phase follows).

        Idempotent in effect: calling this twice on the same toolkit mints no new nodes the
        second time, because :meth:`~evidence_graph.EvidenceGraph.add_source` /
        :meth:`~evidence_graph.EvidenceGraph.add_arith` already dedup a content-identical node
        (their id is a hash of page/offsets/value or operation/inputs/value, never of
        ``minted_by``) and return the FIRST admission unchanged.

        :param answer_text: the deliverable / final answer string.
        :param mandate: the task mandate, for :func:`operation_appropriateness`'s cue matching.
            Optional -- an empty mandate simply means every ``op_appropriateness`` entry reports
            ``None``/``None`` (no cue to match), never an error.
        :returns: ``{"numbers_total", "numbers": [...], "answer_supported", "op_appropriateness"}``
            per the API contract. Never raises.
        """
        numbers_out: List[Dict[str, Any]] = []
        try:
            extracted = extract_answer_numbers(answer_text)
        except Exception:
            extracted = []

        for item in extracted:
            record: Dict[str, Any] = {
                "text": item.get("text", ""), "value": item.get("value"),
                "unit": item.get("unit", ""), "status": "unbacked",
                "unit_consistent": None, "trivial": False, "ambiguity": 0,
                "page_id": "", "node_id": "", "op": "", "operand_node_ids": [],
            }
            try:
                record["trivial"] = is_trivial_number(item)
                target = float(item["value"])
                answer_unit = str(item.get("unit") or "")

                backed = self._find_backed_match(target, answer_unit)
                if backed is not None:
                    page_id, entry, consistent = backed
                    node = self._mint_source_from_entry(page_id, entry)
                    if node is not None:
                        record.update(status="backed", unit_consistent=consistent,
                                     page_id=page_id, node_id=node.id)
                        numbers_out.append(record)
                        continue

                anchored = self._find_unit_anchored_page_match(item)
                if anchored is not None:
                    page_id, start, end = anchored
                    page = self._graph.page(page_id)
                    quote = _quote_for_span(str((page or {}).get("text") or ""), start, end)
                    node = self._graph.add_source(page_id, str(item.get("text") or ""),
                                                  quote=quote, unit=item.get("unit") or None,
                                                  minted_by=ANSWER_AUDIT_TAG)
                    if node is not None:
                        record.update(status="backed", unit_consistent=True,
                                     page_id=page_id, node_id=node.id)
                        numbers_out.append(record)
                        continue

                explanations = self._find_derivation_explanations(target)
                if explanations:
                    # Ambiguity counts distinct (operation, unordered VALUE pair) explanations,
                    # per the API contract -- never entry positions. A model that visits the same
                    # page 15 times re-registers the same quantities 15 times, and counting index
                    # positions inflated one real explanation to ambiguity 72 on the mint01
                    # smoke, disqualifying the very answer the search had correctly explained.
                    record["ambiguity"] = len({
                        (exp["operation"],
                         frozenset((self._entry_numeric(self._entries[idx][1]),
                                    canonical_unit(self._entries[idx][1].unit))
                                   for idx in (exp["i"], exp["j"])))
                        for exp in explanations})
                    best = self._best_explanation(explanations)
                    first_idx, second_idx = best["order"]
                    page_a, entry_a = self._entries[first_idx]
                    page_b, entry_b = self._entries[second_idx]
                    node_a = self._mint_source_from_entry(page_a, entry_a)
                    node_b = self._mint_source_from_entry(page_b, entry_b)
                    if node_a is not None and node_b is not None:
                        try:
                            derived = self._graph.add_arith(
                                best["operation"], [node_a.id, node_b.id],
                                minted_by=ANSWER_AUDIT_TAG)
                        except DerivationError:
                            derived = None
                        if derived is not None:
                            record.update(status="derived", op=best["operation"],
                                         operand_node_ids=[node_a.id, node_b.id])
            except Exception:
                # A single number's grading must never take the whole audit down with it -- the
                # record already defaults to "unbacked", which is the honest state for a number
                # this method could not resolve, mechanically or otherwise.
                pass
            numbers_out.append(record)

        non_trivial = [record for record in numbers_out if not record["trivial"]]
        answer_supported = bool(non_trivial) and all(
            record["status"] in ("backed", "derived")
            and record["unit_consistent"] is not False
            and record["ambiguity"] <= 1
            for record in non_trivial
        )

        op_appropriateness: List[Dict[str, Any]] = []
        try:
            for node in self._graph.nodes():
                # Excludes both mechanical tags, not just this method's own -- a node minted by
                # `shape_derive_check` (:data:`SHAPE_DERIVE_TAG`) is just as mechanical (no model
                # asserted its operation) as one minted here, and including it would make this
                # method's output depend on whether `shape_derive_check` happened to run first on
                # the same toolkit -- a cross-contamination this contract must not have.
                if node.kind != KIND_DERIVED or node.minted_by in (ANSWER_AUDIT_TAG, SHAPE_DERIVE_TAG):
                    continue
                verdict = operation_appropriateness(mandate, node.operation, node.value, node.unit)
                op_appropriateness.append({
                    "node_id": node.id, "operation": node.operation,
                    "sign_plausible": verdict.get("sign_plausible"),
                    "operation_shape_match": verdict.get("operation_shape_match"),
                })
        except Exception:
            op_appropriateness = []

        return {
            "numbers_total": len(numbers_out), "numbers": numbers_out,
            "answer_supported": answer_supported, "op_appropriateness": op_appropriateness,
        }

    def shape_derive_check(self, answer_text: Any, mandate: Any) -> Dict[str, Any]:
        """Whether ``answer_text`` matches the SINGLE operation ``mandate`` unambiguously demands
        (:func:`agent.app.answer_numbers.mandate_demanded_operation`), mechanically recomputed
        over the run's own quantity index -- never a model call, never a prompt/tool change.

        Unlike :meth:`audit_answer` (which asks "is EVERY number in the answer explainable by
        SOME arithmetic over the index, for ANY of four operations"), this method asks a narrower
        and stronger question: "does the index contain a pair of entries whose DEMANDED operation
        -- the one specific op the mandate's own cue phrasing named -- reproduces a number the
        answer actually reports". A model that reports a plausible-looking number nobody asked
        for (the wrong operation, or a number pulled from memory that happens to be explainable
        by an unrelated pair) does not pass this check even when it would pass :meth:`audit_answer`.

        Deliberately silent (``verdict`` ``None``, never a guess) whenever there is nothing
        unambiguous to check against:

        * the mandate itself has no unambiguous demanded operation (``mandate_demanded_operation``
          returned ``None`` -- no cue, an ambiguous multi-family mandate, or argmax/comparison
          phrasing over more than two entities);
        * this run registered no page (``self._entries`` empty);
        * the answer carries no non-trivial number to check (:func:`is_trivial_number` excludes a
          bare year or small bare integer, per the wildcard-suppression lesson applied at minting
          rather than at matching -- see this module's own docstring);
        * the candidate search over index-entry pairs would produce more than
          :data:`SHAPE_DERIVE_MAX_CANDIDATES` distinct results (a run that revisited the same
          handful of pages many times has nothing meaningfully DEMANDED left to check against).

        Otherwise ``verdict`` is ``True`` iff any non-trivial answer number matches (within
        :data:`AUDIT_REL_TOL`) one of the candidate results the demanded operation produces over
        an UNORDERED pair of index entries with compatible units (reusing :meth:`_compat_diff_sum`
        / :meth:`_compat_quotient`, the same compatibility rules :meth:`audit_answer` uses) --
        ``False`` otherwise. On a match, the matched pair's operands are minted as SOURCE nodes
        and the result as a DERIVED node (:meth:`_mint_source_from_entry` /
        :meth:`~evidence_graph.EvidenceGraph.add_arith`), tagged :data:`SHAPE_DERIVE_TAG` --
        idempotent under re-call, since both minting calls dedup on content-identical id exactly
        like :meth:`audit_answer`'s own minting does.

        A candidate is computed ONLY for the demanded operation, never all four
        :meth:`audit_answer` searches: a ``sum`` mandate only ever computes ``a + b``; a
        ``difference`` mandate computes ``abs(a - b)`` when the mandate's own ``absolute`` cue
        fired, or both signed orderings otherwise; a ``quotient`` mandate tries both orderings
        (``a / b`` and ``b / a``), since "ratio of A to B" and "ratio of B to A" are both
        legitimate readings absent a stated order. Candidate results are deduped by
        ``(rounded value, canonical unit pair)`` before the explosion/match checks, so the same
        underlying operand pair registered on several duplicate page visits counts once.

        :param answer_text: the deliverable / final answer string.
        :param mandate: the task mandate -- REQUIRED (unlike :meth:`audit_answer`'s optional
            ``mandate``, this method's entire question is defined by the mandate's demanded
            shape; an empty mandate simply yields ``verdict=None``, ``reason="no_unambiguous_shape"``,
            never an error).
        :returns: ``{"demanded_operation", "absolute", "verdict", "reason", "n_entries",
            "n_pairs_considered", "n_candidates", "n_match_ambiguity", "matched"}``, where
            ``matched`` is ``None`` unless ``verdict`` is ``True``, in which case it is
            ``{"operation", "value", "unit", "operand_node_ids", "derived_node_id"}``.
        :raises: nothing.
        """
        n_entries = len(self._entries)
        result: Dict[str, Any] = {
            "demanded_operation": None, "absolute": False, "verdict": None, "reason": "",
            "n_entries": n_entries, "n_pairs_considered": 0, "n_candidates": 0,
            "n_match_ambiguity": 0, "matched": None,
        }
        try:
            shape = mandate_demanded_operation(mandate)
            operation = shape.get("operation")
            absolute = bool(shape.get("absolute"))
            result["demanded_operation"] = operation
            result["absolute"] = absolute
            if operation is None:
                result["reason"] = "no_unambiguous_shape"
                return result
            if n_entries == 0:
                result["reason"] = "no_index_entries"
                return result

            try:
                extracted = extract_answer_numbers(answer_text)
            except Exception:
                extracted = []
            non_trivial_answers: List[float] = []
            for item in extracted:
                try:
                    if not is_trivial_number(item):
                        non_trivial_answers.append(float(item["value"]))
                except Exception:
                    continue
            if not non_trivial_answers:
                result["reason"] = "no_nontrivial_answer_numbers"
                return result

            numerics: List[Optional[float]] = [self._entry_numeric(entry)
                                               for _, entry in self._entries]
            seen_pairs = set()
            raw_candidates: List[Dict[str, Any]] = []
            n_pairs_considered = 0
            for i in range(n_entries):
                value_i = numerics[i]
                if value_i is None:
                    continue
                unit_i = self._entries[i][1].unit
                for j in range(i + 1, n_entries):
                    value_j = numerics[j]
                    if value_j is None:
                        continue
                    unit_j = self._entries[j][1].unit

                    if operation in ("difference", "sum"):
                        if not self._compat_diff_sum(unit_i, unit_j):
                            continue
                    elif operation == "quotient":
                        if not self._compat_quotient(unit_i, unit_j):
                            continue
                    else:
                        continue

                    # Collapses duplicate page registrations of the SAME operand pair (a model
                    # that revisits the same page re-registers the same quantities) into one
                    # attempt, keyed on the operands' own numeric identity rather than their
                    # index position.
                    pair_key = frozenset((
                        (round(value_i, 9), canonical_unit(unit_i)),
                        (round(value_j, 9), canonical_unit(unit_j)),
                    ))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    n_pairs_considered += 1

                    if operation == "sum":
                        raw_candidates.append({"value": value_i + value_j, "i": i, "j": j,
                                               "order": (i, j)})
                    elif operation == "difference":
                        if absolute:
                            order = (i, j) if value_i >= value_j else (j, i)
                            raw_candidates.append({"value": abs(value_i - value_j), "i": i,
                                                   "j": j, "order": order})
                        else:
                            raw_candidates.append({"value": value_i - value_j, "i": i, "j": j,
                                                   "order": (i, j)})
                            raw_candidates.append({"value": value_j - value_i, "i": i, "j": j,
                                                   "order": (j, i)})
                    elif operation == "quotient":
                        if value_j != 0:
                            raw_candidates.append({"value": value_i / value_j, "i": i, "j": j,
                                                   "order": (i, j)})
                        if value_i != 0:
                            raw_candidates.append({"value": value_j / value_i, "i": i, "j": j,
                                                   "order": (j, i)})

            dedup_map: Dict[Any, Dict[str, Any]] = {}
            for cand in raw_candidates:
                unit_pair = frozenset((canonical_unit(self._entries[cand["i"]][1].unit),
                                       canonical_unit(self._entries[cand["j"]][1].unit)))
                key = (round(cand["value"], 6), unit_pair)
                dedup_map.setdefault(key, cand)
            distinct_candidates = list(dedup_map.values())
            n_candidates = len(distinct_candidates)
            result["n_pairs_considered"] = n_pairs_considered
            result["n_candidates"] = n_candidates

            if n_candidates > SHAPE_DERIVE_MAX_CANDIDATES:
                result["reason"] = "candidate_explosion"
                return result
            if n_candidates == 0:
                result["reason"] = "no_compatible_pairs"
                return result

            matched_candidates = [
                cand for cand in distinct_candidates
                if any(math.isclose(cand["value"], ans, rel_tol=AUDIT_REL_TOL, abs_tol=1e-9)
                      for ans in non_trivial_answers)
            ]
            if not matched_candidates:
                result["verdict"] = False
                result["reason"] = "no_match"
                return result

            result["n_match_ambiguity"] = len(matched_candidates)
            # Deterministic pick among tied matches: earliest (i, j) encountered.
            best = min(matched_candidates, key=lambda cand: (cand["i"], cand["j"]))
            first_idx, second_idx = best["order"]
            page_a, entry_a = self._entries[first_idx]
            page_b, entry_b = self._entries[second_idx]
            node_a = self._mint_source_from_entry(page_a, entry_a)
            node_b = self._mint_source_from_entry(page_b, entry_b)
            derived = None
            if node_a is not None and node_b is not None:
                try:
                    derived = self._graph.add_arith(operation, [node_a.id, node_b.id],
                                                    minted_by=SHAPE_DERIVE_TAG)
                except DerivationError:
                    derived = None
            if derived is None:
                result["verdict"] = False
                result["reason"] = "mint_failed"
                return result

            result["verdict"] = True
            result["reason"] = "matched"
            result["matched"] = {
                "operation": operation, "value": derived.value, "unit": derived.unit,
                "operand_node_ids": [node_a.id, node_b.id], "derived_node_id": derived.id,
            }
            return result
        except Exception:
            # Never raises -- same contract as `audit_answer`. Whatever partial fields were
            # already set (demanded_operation/absolute/n_entries) are kept; verdict stays honest
            # (None, not a guessed False) when the failure happened before a real search ran.
            if not result["reason"]:
                result["reason"] = "error"
            return result

    def artifact(self) -> Dict[str, Any]:
        """The re-verifiable record of everything this run derived.

        Shaped exactly like ``execution_evidence_loop``'s ``output.evidence_graph`` so the existing
        offline auditors -- ``evidence_graph.reverify_graph``, ``scripts/reverify.py``,
        ``scripts/claim_metrics.derivation_fabrication_rate`` -- read a host's artifact with no
        changes at all.
        """
        return self._graph.to_dict()
