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

from typing import Any, Dict, List, Optional, Sequence

from agent.app.testing.evidence_graph import (DerivationError, EvidenceGraph,
                                              _numbers_agree, extract_unit, numeric_value)

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


class LedgerToolkit:
    """Per-run ledger state a host binds one or more tools to.

    :param max_page_chars: cap on the stored window per page; the content hash still covers the
        whole fetched text, so a shortened window still detects drift.
    """

    def __init__(self, max_page_chars: int = 6000) -> None:
        self._graph = EvidenceGraph()
        self._max_page_chars = int(max_page_chars)
        self._pages = 0

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
        return page_id

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

    def _locate(self, value: str) -> Optional[str]:
        """The id of a SOURCE node for ``value`` on any registered page, admitting it if needed.

        Tries every registered page rather than asking the model which one to use: a weak model
        routinely cites the wrong page for a value it did read, and refusing that would report a
        citation slip as a fabrication.

        The operand is offered BOTH as a split number+unit pair and as the literal string the model
        wrote. The split form matters: ``evidence_graph`` knows ``m`` and ``metres`` are the same
        unit, but only when the unit is supplied separately from the number -- its candidate
        matching is built for that shape. Passing the whole string as a literal instead refuses a
        page reading ``1,470\nm`` when the model writes ``"1,470 metres"``, which was observed
        live on task 211 and is over-refusal, not caution: it blocks a value the model genuinely
        read and inflates the module's own averted-fabrication count with its own parsing failures.
        """
        text = str(value or "").strip()
        unit = extract_unit(text)
        number = text[:len(text) - len(unit)].strip() if unit and text.endswith(unit) else ""
        attempts = [(text, None)]
        if number and unit:
            attempts.insert(0, (number, unit))
        for page in self._graph.pages():
            for candidate, candidate_unit in attempts:
                node = self._graph.add_source(page["page_id"], candidate, unit=candidate_unit)
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
