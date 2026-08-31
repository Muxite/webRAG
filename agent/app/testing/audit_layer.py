"""Variant-agnostic audit layer — the same audit surface for ANY executor's stored result.

Only the ``graph`` variant natively emits ``output.sources[]``, ``grounded``,
``finalization_status`` and ``goal_achieved``; ``sequential_react`` and ``langgraph_react`` put
URLs in prose and report a constant ``success``. Comparing arms on that surface compares graph's
instrumentation against the other arms' absence of it, which is not a fair reasoning comparison.

This module derives, from whatever a variant's stored result already contains:

1. the set of pages actually FETCHED, tried from several sources in priority order
   (:func:`fetched_urls_from_execution`) — some variants simply do not record this anywhere
   recoverable, and this module says so rather than reporting an empty set;
2. URLs cited in the deliverable prose vs. URLs actually fetched, i.e. the "cited but never
   opened" set (:func:`cited_vs_fetched`);
3. per-claim quote verification, when a variant has both a quote and the page text it was drawn
   from, using the exact tri-state :func:`~agent.app.testing.execution_evidence_loop.verify_quote`
   already established: ``True`` literal match, ``False`` page in hand and quote absent, ``None``
   unverifiable. A quote that cannot be checked is NEVER reported as failed.

It is applied AFTER a result exists — it does not touch any executor, and every arm is run through
the identical functions here, so no arm gets a more generous audit than another.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.app.testing.execution_evidence_loop import QuoteMatch, verify_quote

#: A bare http(s) URL in prose. Trailing sentence/markdown punctuation is stripped separately
#: (a URL is rarely the last character before a period or closing paren in prose).
_URL_RE = re.compile(r"https?://[^\s<>\")\]]+")
#: Trailing characters a URL regex over-captures from surrounding prose punctuation.
_TRAILING_PUNCT = ".,;:!?)]}'\""

FETCHED_SOURCE_GRAPH_NODES = "graph_nodes"
FETCHED_SOURCE_OUTPUT_SOURCES = "output_sources"
FETCHED_SOURCE_EXTRACTIONS = "evidence_loop_extractions"


def extract_cited_urls(text: str) -> List[str]:
    """URLs cited in ``text``, in first-seen order, deduplicated.

    :param text: deliverable prose (or any free text) to scan.
    :returns: distinct URLs in the order they first appear; ``[]`` for empty/non-string input.
    :raises: nothing.
    """
    if not isinstance(text, str) or not text:
        return []
    seen: Set[str] = set()
    ordered: List[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(_TRAILING_PUNCT)
        if url and url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _urls_from_graph_nodes(execution: Dict[str, Any]) -> Optional[Set[str]]:
    nodes = ((execution.get("graph") or {}).get("nodes")) or {}
    if not isinstance(nodes, dict) or not nodes:
        return None
    urls: Set[str] = set()
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        details = node.get("details")
        if not isinstance(details, dict):
            continue
        if details.get("action") != "visit":
            continue
        url = details.get("visit_url") or details.get("url")
        if url:
            urls.add(str(url))
    return urls or None


def _urls_from_output_sources(execution: Dict[str, Any]) -> Optional[Set[str]]:
    sources = ((execution.get("output") or {}).get("sources")) or []
    if not isinstance(sources, list) or not sources:
        return None
    urls = {str(item["url"]) for item in sources if isinstance(item, dict) and item.get("url")}
    return urls or None


def _urls_from_extractions(execution: Dict[str, Any]) -> Optional[Set[str]]:
    extractions = ((execution.get("output") or {}).get("extractions")) or []
    if not isinstance(extractions, list) or not extractions:
        return None
    urls = {str(item["source_url"]) for item in extractions
            if isinstance(item, dict) and item.get("source_url")}
    return urls or None


#: Extractors tried in priority order — the first source that yields a non-empty set wins.
_FETCHED_EXTRACTORS = (
    (FETCHED_SOURCE_GRAPH_NODES, _urls_from_graph_nodes),
    (FETCHED_SOURCE_OUTPUT_SOURCES, _urls_from_output_sources),
    (FETCHED_SOURCE_EXTRACTIONS, _urls_from_extractions),
)


def fetched_urls_from_execution(execution: Dict[str, Any]) -> Tuple[Optional[Set[str]], Optional[str]]:
    """The set of pages a run actually fetched, recovered from whatever ``execution`` records.

    Tries, in order: per-node ``visit`` actions in ``execution.graph.nodes`` (the graph engine),
    then ``execution.output.sources[]`` (graph's finalize summary, as a fallback when node
    details are thin), then ``execution.output.extractions[].source_url`` (the evidence-loop
    ledger). ``sequential_react`` and ``langgraph_react`` results carry none of these — their
    stored telemetry only has aggregate visit COUNTS (``execution.observability.visit.count``),
    never the URLs — so this correctly returns ``(None, None)`` for them rather than an empty set,
    which would misreport "zero pages fetched" as fact.

    :param execution: the stored ``execution`` dict of one result cell.
    :returns: ``(urls, source)`` — a non-empty URL set and which extractor produced it, or
        ``(None, None)`` when nothing recoverable was found.
    :raises: nothing.
    """
    if not isinstance(execution, dict):
        return None, None
    for source_label, extractor in _FETCHED_EXTRACTORS:
        urls = extractor(execution)
        if urls:
            return urls, source_label
    return None, None


def cited_vs_fetched(deliverable: str, execution: Dict[str, Any]) -> Dict[str, Any]:
    """Compare URLs cited in the deliverable against URLs actually fetched.

    :param deliverable: the final answer text.
    :param execution: the stored ``execution`` dict of one result cell.
    :returns: a dict with ``cited`` (always populated), ``fetched``/``fetched_source``/
        ``recoverable``, and ``cited_never_fetched``/``fetched_never_cited`` — the latter two are
        ``None``, not empty sets, when ``fetched`` could not be recovered at all.
    :raises: nothing.
    """
    cited = set(extract_cited_urls(deliverable))
    fetched, source = fetched_urls_from_execution(execution)
    if fetched is None:
        return {
            "cited": cited, "fetched": None, "fetched_source": None, "recoverable": False,
            "cited_never_fetched": None, "fetched_never_cited": None,
        }
    return {
        "cited": cited, "fetched": fetched, "fetched_source": source, "recoverable": True,
        "cited_never_fetched": cited - fetched, "fetched_never_cited": fetched - cited,
    }


def audit_quotes(extractions: List[Dict[str, Any]], pages: Optional[Dict[str, str]] = None
                 ) -> List[Dict[str, Any]]:
    """Re-check each extraction's quote against page text, when page text is available.

    Reuses :func:`~agent.app.testing.execution_evidence_loop.verify_quote` directly — no second
    implementation, no fuzzy matching. ``pages`` is keyed by ``page_id`` first, falling back to
    ``source_url``, since a stored result may carry one or neither depending on the variant.

    :param extractions: records with at least ``quote``; ``page_id``/``source_url`` used to look
        up page text when present.
    :param pages: ``{page_id_or_url: page_text}``; ``None``/missing entries leave a record
        unverifiable (``verified=None``), never ``False``.
    :returns: one dict per input record: ``entity``, ``field``, ``source_url``, ``quote``,
        ``verified`` (tri-state), ``fail_reason``, and ``stored_verified`` (the variant's own
        runtime verdict, for comparison — never overwritten by this layer).
    :raises: nothing.
    """
    pages = pages or {}
    rows: List[Dict[str, Any]] = []
    for extraction in extractions or []:
        if not isinstance(extraction, dict):
            continue
        quote = extraction.get("quote") or ""
        page_id = extraction.get("page_id") or ""
        source_url = extraction.get("source_url") or ""
        page_text = pages.get(page_id)
        if page_text is None:
            page_text = pages.get(source_url)
        match: QuoteMatch = verify_quote(page_text, quote)
        rows.append({
            "entity": extraction.get("entity"), "field": extraction.get("field"),
            "source_url": source_url, "quote": quote,
            "verified": match.verified, "fail_reason": match.fail_reason,
            "stored_verified": extraction.get("quote_verified"),
        })
    return rows


def audit_result(result: Dict[str, Any], pages: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The full audit surface for ONE stored result cell, whatever variant produced it.

    :param result: one result-JSON document (``{"execution_variant", "execution": {...}, ...}``).
    :param pages: optional ``{page_id_or_url: page_text}`` for offline quote re-verification (see
        :func:`audit_quotes`); omitted for a typical stored cell, which does not persist page text.
    :returns: ``{"variant", "cited", "fetched", "fetched_source", "recoverable",
        "cited_never_fetched", "fetched_never_cited", "quotes", "quote_audit_applicable"}``.
        ``quote_audit_applicable`` is False when the variant's output carries no ``extractions``
        at all (sequential_react / langgraph_react / graph today) — distinct from an empty list of
        quotes that were all checked and passed.
    :raises: nothing.
    """
    execution = result.get("execution") or {}
    output = execution.get("output") or {}
    deliverable = output.get("final_deliverable") or ""
    extractions = output.get("extractions")
    report = cited_vs_fetched(deliverable, execution)
    report["variant"] = result.get("execution_variant", "")
    report["quote_audit_applicable"] = isinstance(extractions, list) and len(extractions) > 0
    report["quotes"] = audit_quotes(extractions or [], pages)
    return report
