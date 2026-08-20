"""ONE plan-library retrieval attempt, end to end — shared by both call sites.

The automatic pre-expansion short-circuit (``IdeaDagEngine._plan_library_auto_shortcircuit``)
and the on-demand leaf action (``actions.PlanLibrarySearchLeafAction``) ask the library the
same question and must answer it identically: build the query text off the node, rank the
corpus, and — only once a hit has qualified — spend the pipeline's single slot-extraction call
and adapt the filled template into GoT candidates. They differ in the two labels they carry
(``origin`` for the graph, ``call_site`` for the logs), in the auto-apply bar a non-root node
must clear (stricter on the automatic path, see :func:`resolve`), and in what their caller does
with the answer: the automatic path feeds ``_handle_expansion_node``, the action reports a
result dict its node's completion hook later rebuilds children from.

**Why a module and not an ``IdeaDagEngine`` method.** A :class:`~agent.app.idea_policies.
actions.LeafAction` is constructed with no reference back to the engine — deliberately: the
engine owns graph mutation, actions own I/O — so an engine method could only be shared with the
action by handing actions an engine handle, inverting that. The alternative, a second copy of
the pipeline inside the action, would drift exactly where drift is invisible: a logged field
that only one path emits, or a fail-toward-silence rule only one path applies.

:func:`resolve` therefore logs *itself*, rather than leaving that to its callers: a caller
cannot forget it, and both env-gated streams see every attempt, adopted or not. The two streams
stay separate on purpose (see :mod:`agent.app.plan_library.retrieval_log`) — "was this the
right template" and "did the leaf it authored deliver" are different questions that join
offline on ``retrieval_id``/``template_id``.

No graph mutation lives here either: this module decides *what could be run*, never *what is*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Optional

from agent.app.idea_policies.post_expansion_hooks import extract_mandate
from agent.app.plan_library import retrieval as _retrieval
from agent.app.plan_library import retrieval_log as _retrieval_log
from agent.app.testing import contract_log as _contract_log

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode
    from agent.app.idea_policies.plan_library import PlanLibraryExpansion
    from agent.app.plan_library.retrieval import FillOutcome, RetrievalResult


class SearchResolution(NamedTuple):
    """What one retrieval attempt produced — the three objects a caller may need.

    ``expansion`` is the only one that can change what the engine runs (and is falsy unless a
    template both matched confidently AND filled); ``retrieval`` and ``fill`` are the ranking
    and slot-fill records behind it, which the on-demand action copies into its result dict so
    the engine can rebuild the expansion later without re-querying anything.
    """

    expansion: Optional["PlanLibraryExpansion"]
    retrieval: Optional["RetrievalResult"]
    fill: Optional["FillOutcome"]

    @property
    def adopted(self) -> bool:
        """True when a template survived every gate (``PlanLibraryExpansion`` is falsy with
        no candidates, so this is "a real plan reached us", not merely "something matched")."""
        return bool(self.expansion)

    @property
    def template_id(self) -> Optional[str]:
        return self.expansion.template_id if self.expansion else None


#: The "nothing came back" resolution — what a caller's fail-toward-silence net returns.
EMPTY = SearchResolution(None, None, None)


async def resolve(
    library: Any,
    graph: "IdeaDag",
    node: "IdeaNode",
    *,
    io: Any,
    call_site: str,
    origin: str,
    contracts: Any = None,
    model_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> SearchResolution:
    """Rank the corpus for ``node``, fill the winner, log the attempt.

    ``retrieve()`` never raises and downgrades every failure to "no match" itself, and
    ``fill_from_query`` does the same for slot extraction/binding — so the only way out of here
    is an infrastructural surprise (a broken corpus, a connector blowing up in an unexpected
    way), which each caller catches as silence rather than as a degraded run.

    The slot-fill call receives the RAW mandate, newlines intact — never the retrieval query
    text, which ``retrieve()`` whitespace-collapses. ``slot_fill``'s deterministic candidate
    extractor matches at line starts, so a collapsed copy silently loses an enumerated
    candidate list and degrades every fill to an LLM-only (or empty) one.
    """
    is_root = node.node_id == graph.root_id()
    query_text = _retrieval.build_query_text(node, is_root=is_root)
    mandate = extract_mandate(graph, node.node_id)
    # Scope guard (F9): the AUTOMATIC short-circuit substitutes a template for whatever node
    # is being stepped, including deep ones whose query text an earlier context-blind expansion
    # invented. The thresholds are calibrated on ROOT-level task statements only, so a non-root
    # node has to clear a stricter bar before it may hijack its subtree's shape. The on-demand
    # action keeps the calibrated bar: there the model asked for a strategy for that node, and
    # the answer is an explicit re-expansion rather than a silent replacement.
    auto_apply_threshold = (
        _retrieval.auto_apply_threshold_for(is_root)
        if call_site == _retrieval.CALL_SITE_AUTO
        else _retrieval.AUTO_APPLY_THRESHOLD
    )
    result = await library.retrieve(
        getattr(io, "connector_chroma", None),
        query_text,
        archetype_hint=_retrieval.archetype_hint_for(mandate or query_text),
        auto_apply_threshold=auto_apply_threshold,
    )
    outcome = None
    if result.auto_apply:
        outcome = await library.fill_from_query(
            result,
            mandate=mandate,
            query_text=query_text,
            io=io,
            origin=origin,
            contracts=contracts,
            model_name=model_name,
            fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )
        # A fill failure downgrades the decision post-hoc; log what actually happened.
        result = outcome.retrieval
    resolution = SearchResolution(
        expansion=(outcome.expansion if outcome else None), retrieval=result, fill=outcome
    )
    log(node, resolution, call_site=call_site)
    return resolution


def log(node: "IdeaNode", resolution: SearchResolution, *, call_site: str) -> None:
    """Record one attempt in BOTH env-gated logs (each a no-op on its own flag).

    The contract log answers "did the leaf this template authored deliver its contract"; the
    retrieval log answers "was this the right template at all". A correct retrieval can be
    followed by bad execution and vice versa, so merging them would hide exactly the
    distinction the dogfooding pass is looking for.
    """
    result = resolution.retrieval
    if result is None:
        return
    outcome = resolution.fill
    expansion = getattr(outcome, "expansion", None)
    applied = expansion.template_id if expansion is not None else None
    if _contract_log.enabled():
        _contract_log.record_search(
            mode=call_site,
            query=result.query_text,
            matches=[m.as_dict() for m in result.candidates],
            adopted=applied is not None,
            adopted_template_id=applied,
            threshold=_retrieval.AUTO_APPLY_THRESHOLD,
            node_id=node.node_id,
            decision=result.decision,
        )
    _retrieval_log.record_retrieval(
        result,
        call_site=call_site,
        slot_fill=("not_attempted" if outcome is None else ("filled" if applied else "failed")),
        applied_template_id=applied,
        node_id=node.node_id,
        slot_values=getattr(outcome, "slot_values", None),
        error=getattr(outcome, "error", None),
    )
