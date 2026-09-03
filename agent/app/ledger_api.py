"""The Euglena Ledger as a COMPONENT: a question plus sources in, a pinned result out.

``docs/LEDGER.md`` states the product's shape -- "an auditable evidence compiler... It is a
component, not an agent. Give it a question and a set of sources." The code did not offer that.
The only entry point producing the documented artifact was
:func:`~agent.app.testing.execution_evidence_loop.run_evidence_loop_execution`, which needs an
``IdeaTestModule`` carrying GRADERS (so a user asking a question must first author the graders for
an answer they do not have), five pre-built connectors, a run stamp and a writable results
directory. ``sources`` was not an input anywhere: the loop searched for its own.

This module is the missing façade, and only a façade. The loop itself is already clean --
:func:`~agent.app.testing.execution_evidence_loop.run_evidence_loop` takes an ``AgentIO``, a
mandate and two budgets -- so nothing here reimplements it. What this module does is:

1. assemble the connector set (mirroring ``idea_test_runner._make_connector_set``),
2. make ``sources`` real by RESTRICTING retrieval to them rather than by asking the loop to
   behave differently, and
3. shape the loop's result into a typed, JSON-safe payload whose every claim carries its pin.

How ``sources`` restricts retrieval
-----------------------------------
Search is served by :class:`~agent.app.connector_search_corpus.ConnectorSearchCorpus` built over
exactly the supplied documents and given NO live fallback, so the ranked universe is the source
set and a query matching nothing returns nothing -- an honest miss, never a silent widening to
the open web. Visits are served by :class:`SourceServingHttp`, which answers a supplied URL from
the text already in hand and refuses any other host, so the loop cannot follow a link out of the
set either. With ``sources=None`` neither substitution happens and the behaviour is today's: the
configured live backend (``SEARCH_PROVIDER``) and a real HTTP/browser fetch.

Chroma
------
:class:`~agent.app.agent_io.AgentIO` requires a ``connector_chroma`` and the evidence loop imports
one at module scope, but the loop's action set is ``search|visit|derive|verify|finish`` -- no
memory op is ever issued. ``ConnectorChroma.__init__`` is inert (it only sets fields; the client
is built by ``init_chroma_api``, which nothing here calls), so the connector is constructed for
the signature and never connected. Answering a question therefore needs no vector store running.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult

from agent.app.agent_io import AgentIO
from agent.app.connector_browser import ConnectorBrowser
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_search import ConnectorSearch, create_search_backend
from agent.app.connector_search_corpus import ConnectorSearchCorpus, title_from_url
from agent.app.testing import execution_evidence_loop as evidence_loop

#: A claim whose verbatim quote was mechanically located in the page it cites.
QUOTE_VERIFIED = "VERIFIED"
#: A claim whose quote was checked against the page in hand and is NOT in it -- a paraphrase.
QUOTE_FAILED = "FAILED"
#: No verification verdict exists: nothing was quoted, no page was held, or the row never
#: resolved. Absent is never zero -- this is reported as its own state and never as ``FAILED``.
QUOTE_UNKNOWN = "UNKNOWN"

#: The three verdicts, re-exported so a consumer never has to import the loop internals.
VERDICT_ANSWER = evidence_loop.VERDICT_ANSWER
VERDICT_PARTIAL = evidence_loop.VERDICT_PARTIAL
VERDICT_ABSTAIN = evidence_loop.VERDICT_ABSTAIN

#: Default step budget, matching the evidence loop's own ``IDEA_TEST_EVIDENCE_LOOP_MAX_STEPS``.
DEFAULT_MAX_STEPS = 25
#: Default synthesis-token budget, matching ``IDEA_TEST_BASELINE_MAX_TOKENS``.
DEFAULT_MAX_TOKENS = 8192


@dataclass(frozen=True)
class Source:
    """One document the caller is willing to have the question answered from.

    :param url: the source's address. It is both the retrieval key and what every claim built
        from this source will cite, so it is required even when ``text`` is supplied.
    :param text: the document's text. When present the source is used entirely offline -- indexed
        for search and served for a visit -- and is never fetched. When empty the URL is fetched
        ONCE, up front, and the fetched text is used for both from then on.
    :param title: optional display title; derived from the URL when omitted.
    """

    url: str
    text: str = ""
    title: str = ""


@dataclass(frozen=True)
class Claim:
    """One row of the ledger, FULLY PINNED.

    A consumer reading this needs nothing else to re-check it: ``source_url`` and ``page_id`` say
    which page, ``quote_start`` / ``quote_end`` say exactly where in that page's stored text, and
    ``quote_verification`` says whether the quote was found there. Before these fields existed on
    the row the offsets lived only in a parallel extraction list joinable back by fuzzy
    ``(entity, field)`` string matching, so what a consumer actually read was unpinned prose plus
    a URL.
    """

    entity: str
    field: str
    value: str
    unit: str
    status: str
    resolved: bool
    confidence_tier: str
    source_url: str
    quote: str
    #: Tri-state, mirroring :attr:`Extraction.quote_verified`: True / False / ``None`` for "never
    #: decided". ``None`` is NOT False -- see :data:`QUOTE_UNKNOWN`.
    quote_verified: Optional[bool]
    quote_verification: str
    page_id: str
    quote_start: int
    quote_end: int
    evidence_node_id: str

    def to_dict(self) -> Dict[str, Any]:
        """This claim as a JSON-serializable dict."""
        return {
            "entity": self.entity, "field": self.field, "value": self.value, "unit": self.unit,
            "status": self.status, "resolved": self.resolved,
            "confidence_tier": self.confidence_tier, "source_url": self.source_url,
            "quote": self.quote, "quote_verified": self.quote_verified,
            "quote_verification": self.quote_verification, "page_id": self.page_id,
            "quote_start": self.quote_start, "quote_end": self.quote_end,
            "evidence_node_id": self.evidence_node_id,
        }


@dataclass(frozen=True)
class Derivation:
    """One value the run RECOMPUTED from other values, with its inputs and its postcondition."""

    node_id: str
    operation: str
    value: str
    unit: str
    input_ids: Tuple[str, ...]
    #: True when the operation's postcondition was re-checked and holds, False when it does not,
    #: ``None`` when it was never assessed. ``None`` is reported as unknown, never as passing.
    valid: Optional[bool]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "operation": self.operation, "value": self.value,
                "unit": self.unit, "input_ids": list(self.input_ids), "valid": self.valid,
                "detail": self.detail}


@dataclass(frozen=True)
class LedgerResult:
    """What one :func:`run` produced: the prose answer plus everything needed to audit it."""

    question: str
    answer: str
    #: ``ANSWER`` / ``PARTIAL`` / ``ABSTAIN``, straight from the ledger. An ABSTAIN is the product
    #: working, so it is surfaced exactly as computed and never upgraded.
    verdict: str
    claims: Tuple[Claim, ...]
    derivations: Tuple[Derivation, ...]
    #: One entry per visited page: ``page_id``, ``url``, ``content_hash``, ``chars`` and the
    #: stored ``text`` the offsets index into.
    pages: Tuple[Dict[str, Any], ...]
    model: str
    #: ``{"verified", "failed", "unchecked", ...}`` counted over the append-only extraction
    #: records -- the verification axis, independent of how rows resolved.
    quote_counts: Dict[str, int] = dataclass_field(default_factory=dict)
    #: ``{"rows", "resolved", "resolved_verified", "resolved_unverified", "unresolved"}``.
    resolution_counts: Dict[str, int] = dataclass_field(default_factory=dict)
    #: Closed-roster completeness when the question enumerates candidates; inert otherwise.
    roster: Dict[str, Any] = dataclass_field(default_factory=dict)
    #: Derivations the graph REFUSED, tallied by typed code (e.g. ``{"UNIT_MISMATCH": 1}``). A
    #: correct refusal and a never-attempted derivation both leave zero derived values, and only
    #: this distinguishes them, so it is part of the result rather than a log line.
    derivation_refusals: Dict[str, int] = dataclass_field(default_factory=dict)
    #: How each search was served: ``"corpus"`` / ``"live"`` / ``"none"``, in order. Empty when
    #: the backend does not report provenance.
    search_provenance: Tuple[str, ...] = ()
    #: The loop's step-by-step scratchpad, kept out of :meth:`to_dict` by default.
    scratchpad: Tuple[str, ...] = ()

    def to_dict(self, include_scratchpad: bool = False,
                include_page_text: bool = True) -> Dict[str, Any]:
        """This result as a JSON-serializable dict, safe to paste into an issue.

        Nothing key-, token- or secret-shaped is carried: the payload is built field by field
        from the ledger and the fetched pages, and no connector, config or environment value ever
        reaches it.

        :param include_scratchpad: append the loop's per-step trace. Off by default because it is
            large and is a debugging artifact, not part of the claim.
        :param include_page_text: keep each page's stored text. Off drops only ``text``; the
            ``content_hash`` covering the whole fetched text always survives, so a claim's offsets
            stay checkable against a re-fetch.
        """
        pages: List[Dict[str, Any]] = []
        for page in self.pages:
            row = dict(page)
            if not include_page_text:
                row.pop("text", None)
            pages.append(row)
        payload: Dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "verdict": self.verdict,
            "model": self.model,
            "claims": [claim.to_dict() for claim in self.claims],
            "derivations": [derivation.to_dict() for derivation in self.derivations],
            "derivation_refusals": dict(self.derivation_refusals),
            "pages": pages,
            "quote_counts": dict(self.quote_counts),
            "resolution_counts": dict(self.resolution_counts),
            "roster": dict(self.roster),
            "search_provenance": list(self.search_provenance),
        }
        if include_scratchpad:
            payload["scratchpad"] = list(self.scratchpad)
        return payload


class SourceServingHttp(ConnectorHttp):
    """An HTTP connector that answers ONLY from a supplied source set.

    A visit to a supplied URL returns its text with no socket opened. A visit to anything else
    returns a normal error result, which the loop already renders as a ``VISIT ERROR``
    observation -- so the model learns the page is out of scope instead of the component quietly
    reaching the open web for a source the caller did not authorize.
    """

    def __init__(self, connector_config: ConnectorConfig, texts: Dict[str, str]) -> None:
        super().__init__(connector_config)
        self.texts = {_canonical(url): text for url, text in texts.items()}

    async def request(self, method: str, url: str, retries: int = 2,
                      suppress_timing: bool = False, **kwargs) -> RequestResult:
        """Serve ``url`` from the source set, or refuse it."""
        text = self.texts.get(_canonical(url))
        if text is not None:
            return RequestResult(status=200, data=text, error=False)
        return RequestResult(
            status=None, error=True,
            data=(f"{url} is not in the supplied source set; this run may only read the "
                  f"{len(self.texts)} source(s) it was given"))


@dataclass
class Connectors:
    """The connector set one run needs, assembled the same way the benchmark runner does."""

    llm: Any
    search: ConnectorSearch
    http: ConnectorHttp
    chroma: ConnectorChroma
    browser: Optional[ConnectorBrowser]
    config: ConnectorConfig


def _canonical(url: str) -> str:
    """A URL reduced to what makes it the same address, for source lookup.

    Scheme and host lowercased, fragment dropped, one trailing slash ignored. Deliberately
    conservative: the query string is KEPT, because for many sources it is the address.
    """
    text = str(url or "").strip()
    text = text.split("#", 1)[0]
    if "://" in text:
        scheme, rest = text.split("://", 1)
        host, _, tail = rest.partition("/")
        text = f"{scheme.lower()}://{host.lower()}" + (f"/{tail}" if tail else "")
    return text.rstrip("/")


def _as_source(item: Union[Source, str, Dict[str, Any]]) -> Source:
    """Coerce one caller-supplied source. A bare string is a URL with no text yet."""
    if isinstance(item, Source):
        return item
    if isinstance(item, dict):
        return Source(url=str(item.get("url", "")), text=str(item.get("text", "") or ""),
                      title=str(item.get("title", "") or ""))
    return Source(url=str(item or ""))


def build_connectors(sources: Optional[Sequence[Source]] = None,
                     config: Optional[ConnectorConfig] = None) -> Connectors:
    """Assemble the connector set for one run.

    Mirrors ``idea_test_runner._make_connector_set`` (llm / search / http / chroma / browser)
    rather than inventing a second assembly pattern, with two substitutions when ``sources`` is
    given: search is a corpus over exactly those documents with no live fallback, and HTTP is
    :class:`SourceServingHttp`. The browser fallback is omitted in that case -- there is nothing
    for it to fetch, and starting one would be the only way a "supplied sources only" run could
    still touch the network.

    :param sources: the source set, already resolved (every source carrying its text), or None
        for today's configured live backends.
    :param config: an existing :class:`ConnectorConfig`; one is built from the environment when
        omitted.
    :returns: a :class:`Connectors` whose ``chroma`` is constructed but NOT connected.
    """
    config = config or ConnectorConfig()
    # Imported here rather than at module scope for symmetry with create_search_backend's own
    # lazy imports: nothing below needs the class until a run actually happens.
    from agent.app.connector_llm import ConnectorLLM

    if sources is None:
        return Connectors(llm=ConnectorLLM(config), search=create_search_backend(config),
                          http=ConnectorHttp(config), chroma=ConnectorChroma(config),
                          browser=ConnectorBrowser(config), config=config)
    documents = [{"url": source.url, "title": source.title or title_from_url(source.url),
                  "description": (source.text or "")[:400], "text": source.text}
                 for source in sources]
    return Connectors(
        llm=ConnectorLLM(config),
        # fallback=None is the bound: a query the source set cannot answer returns nothing rather
        # than billing a live search for a source the caller never supplied.
        search=ConnectorSearchCorpus(config, corpus_dir="", documents=documents, fallback=None,
                                     max_live_fallbacks=0),
        http=SourceServingHttp(config, {source.url: source.text for source in sources}),
        chroma=ConnectorChroma(config),
        browser=None,
        config=config,
    )


async def _resolve_sources(items: Sequence[Any],
                           config: ConnectorConfig) -> Tuple[List[Source], List[str]]:
    """Give every supplied source its text, fetching the bare URLs ONCE.

    :returns: ``(sources, failures)``. A URL that could not be fetched is reported rather than
        silently dropped -- a run answered from three of four requested sources is a different
        run, and the caller has to be able to see that.
    """
    sources = [_as_source(item) for item in items]
    pending = [source for source in sources if source.url and not source.text]
    failures: List[str] = []
    if not pending:
        return sources, failures
    http = ConnectorHttp(config)
    browser = ConnectorBrowser(config)
    fetcher = AgentIO(connector_llm=None, connector_search=None, connector_http=http,
                      connector_chroma=None, connector_browser=browser)
    fetched: Dict[str, str] = {}
    try:
        for source in pending:
            try:
                fetched[source.url] = await fetcher.visit(source.url, timeout_seconds=30)
            except Exception as exc:  # noqa: BLE001 -- a dead source is a result, not a crash.
                failures.append(f"{source.url}: {exc}")
    finally:
        await _close_quietly(http)
        await _close_quietly(browser)
    resolved = [Source(url=s.url, text=fetched.get(s.url, s.text), title=s.title)
                for s in sources]
    return [s for s in resolved if s.text or s.url not in fetched], failures


async def _close_quietly(connector: Any) -> None:
    """Best-effort teardown; a connector that will not close must not fail a completed run.

    ``ConnectorHttp`` has no ``close`` — its session teardown IS ``__aexit__`` — while
    ``ConnectorBrowser`` has ``close``. Both are tried, in that order of preference, so neither
    leaks an open aiohttp session or a live Chromium after a one-shot answer.
    """
    for name in ("close_session", "close", "shutdown"):
        closer = getattr(connector, name, None)
        if closer is None:
            continue
        try:
            outcome = closer()
            if asyncio.iscoroutine(outcome):
                await outcome
        except Exception:  # noqa: BLE001
            pass
        return
    exiter = getattr(connector, "__aexit__", None)
    if exiter is None:
        return
    try:
        await exiter(None, None, None)
    except Exception:  # noqa: BLE001
        pass


def _quote_verification(row: Any, ledger: Any) -> Tuple[Optional[bool], str]:
    """This row's verification verdict, tri-state, from the record that actually wrote it.

    The row's own ``quote_verified`` is a plain bool and so cannot say "never decided". The pin
    now on the row (``page_id`` / ``quote_start`` / ``quote_end``) makes the join back to the
    append-only record EXACT, so the record's tri-state can be surfaced without the fuzzy
    ``(entity, field)`` string matching the old join needed.
    """
    if not row.resolved:
        return None, QUOTE_UNKNOWN
    for record in ledger.extractions:
        if (record.page_id == row.page_id and record.value == row.value
                and record.quote == row.quote and record.source_url == row.source_url):
            if record.quote_verified is True:
                return True, QUOTE_VERIFIED
            if record.quote_verified is False:
                return False, QUOTE_FAILED
            return None, QUOTE_UNKNOWN
    return (True, QUOTE_VERIFIED) if row.quote_verified else (None, QUOTE_UNKNOWN)


def _claims(ledger: Any) -> Tuple[Claim, ...]:
    claims = []
    for row in ledger.rows:
        verified, verification = _quote_verification(row, ledger)
        claims.append(Claim(
            entity=row.entity, field=row.field, value=row.value, unit=row.unit,
            status=row.status, resolved=row.resolved, confidence_tier=row.confidence_tier,
            source_url=row.source_url, quote=row.quote, quote_verified=verified,
            quote_verification=verification, page_id=row.page_id,
            quote_start=row.quote_start, quote_end=row.quote_end,
            evidence_node_id=row.evidence_node_id))
    return tuple(claims)


def _derivations(graph: Any) -> Tuple[Derivation, ...]:
    if graph is None:
        return ()
    from agent.app.testing.evidence_graph import KIND_DERIVED
    return tuple(Derivation(node_id=node.id, operation=node.operation, value=node.value,
                            unit=node.unit, input_ids=tuple(node.input_ids),
                            valid=node.derivation_valid, detail=node.derivation_detail)
                 for node in graph.nodes() if node.kind == KIND_DERIVED)


async def run(question: str,
              sources: Optional[Sequence[Union[Source, str, Dict[str, Any]]]] = None,
              *,
              model: Optional[str] = None,
              max_steps: int = DEFAULT_MAX_STEPS,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              connectors: Optional[Connectors] = None) -> LedgerResult:
    """Answer ``question``, optionally restricted to ``sources``, and return a pinned result.

    No task module, no graders, no run stamp and no results directory: this is the entry point
    ``docs/LEDGER.md`` describes. The work itself is
    :func:`~agent.app.testing.execution_evidence_loop.run_evidence_loop`, unchanged.

    :param question: the question, used verbatim as the loop's mandate.
    :param sources: the documents the answer may be built from. A :class:`Source`, a bare URL
        string (fetched once, up front) or a ``{"url", "text", "title"}`` mapping. ``None`` --
        the default -- leaves retrieval on the configured live backend, today's behaviour.
    :param model: executor model; defaults to ``MODEL_NAME`` from the environment.
    :param max_steps: hard step budget for the loop.
    :param max_tokens: cap on the final synthesis call.
    :param connectors: a pre-assembled set, for a caller reusing one across questions. When given
        it is used AS IS and is not closed here, since the caller owns its lifetime.
    :returns: a :class:`LedgerResult`. A run that established nothing returns an ``ABSTAIN``
        verdict and unresolved claims; that is the component working, and it is never softened.
    :raises ValueError: when ``question`` is empty -- there is no honest result for no question --
        or when both ``sources`` and ``connectors`` are given, since the pre-built set already
        fixes retrieval and honouring only one of the two would silently answer a different
        question than the caller asked.
    """
    if not str(question or "").strip():
        raise ValueError("question is required")
    if sources is not None and connectors is not None:
        raise ValueError("pass sources or connectors, not both: a pre-built connector set "
                         "already decides what retrieval can reach")
    config = connectors.config if connectors else ConnectorConfig()
    resolved: Optional[List[Source]] = None
    fetch_failures: List[str] = []
    if sources is not None and connectors is None:
        resolved, fetch_failures = await _resolve_sources(sources, config)
    owned = connectors is None
    connectors = connectors or build_connectors(resolved, config)
    model_name = (model or os.environ.get("MODEL_NAME") or config.model_name or "").strip()
    connectors.llm.set_model(model_name)
    agent_io = AgentIO(connector_llm=connectors.llm, connector_search=connectors.search,
                       connector_http=connectors.http, connector_chroma=connectors.chroma,
                       connector_browser=connectors.browser, telemetry=None,
                       collection_name="ledger_api")
    try:
        result = await evidence_loop.run_evidence_loop(agent_io, question, model_name,
                                                      max_steps, max_tokens)
    finally:
        if owned:
            await _close_quietly(connectors.http)
            if connectors.browser is not None:
                await _close_quietly(connectors.browser)
    ledger = result.ledger
    answer = result.deliverable
    if fetch_failures:
        answer = (f"{answer}\n\nSOURCES NOT READ: " + "; ".join(fetch_failures)).strip()
    return LedgerResult(
        question=question,
        answer=answer,
        verdict=result.verdict,
        claims=_claims(ledger),
        derivations=_derivations(ledger.graph),
        pages=tuple(result.pages or ()),
        model=model_name,
        quote_counts=evidence_loop.quote_verification_counts(ledger),
        resolution_counts=ledger.resolution_counts(),
        roster=ledger.roster(),
        derivation_refusals=(ledger.graph.refusal_counts() if ledger.graph is not None else {}),
        search_provenance=tuple(getattr(connectors.search, "provenance", ()) or ()),
        scratchpad=tuple(result.scratchpad or ()),
    )


def _page_field(page: Any, name: str) -> Any:
    """Read ``name`` from a page given either as a mapping or as an object with attributes."""
    if isinstance(page, dict):
        return page.get(name)
    return getattr(page, name, None)


def recheck_claim(claim: Claim,
                  pages: Sequence[Any]) -> Tuple[Optional[bool], str]:
    """Independently re-verify one :class:`Claim` against the page it pins.

    This is the consumer-side audit, and it exists because of a real trap. The ledger locates a
    quote verbatim first and then with whitespace runs collapsed, and it stores the RAW page
    offsets of the hit (:func:`execution_evidence_loop._locate`). So on a genuine, correctly
    verified claim::

        page_text[claim.quote_start:claim.quote_end] == 'Floor\\ncount\\n154 + 9 maintenance'
        claim.quote                                  == 'Floor count\\n154 + 9 maintenance'

    -- the words are identical and only whitespace differs, but the obvious exact comparison a
    consumer would write returns False and reads as tampering. This applies the SAME rule the
    ledger used rather than inventing a second one, so consumer and producer cannot drift apart.

    The stored ``quote_verified`` is deliberately NOT consulted for the answer: the point is to
    detect drift or tampering in a stored result, which means recomputing rather than trusting.

    :param claim: the claim to audit.
    :param pages: the run's pages, each a mapping or object exposing ``page_id`` and ``text``.
    :returns: ``(True, "pinned" | "unpinned_match")`` when the quote's words are on the page;
        ``(False, "pin_mismatch" | "absent")`` when the page was in hand and they are not;
        ``(None, "no_quote" | "no_page")`` when there was nothing to check -- absent is never
        reported as a failed verification.
    :raises: nothing.
    """
    quote = claim.quote if isinstance(claim.quote, str) else ""
    if not evidence_loop.strip_quote_wrapper(quote):
        return None, "no_quote"
    page = next((p for p in pages or () if _page_field(p, "page_id") == claim.page_id), None)
    if page is None:
        return None, "no_page"
    text = _page_field(page, "text")
    if not isinstance(text, str) or not text.strip():
        return None, "no_page"

    start, end = claim.quote_start, claim.quote_end
    if 0 <= start < end <= len(text):
        if evidence_loop.verify_quote(text[start:end], quote).verified:
            return True, "pinned"
        return False, "pin_mismatch"
    if start >= 0 or end >= 0:
        # Offsets were given but do not address this page -- that is a mismatch, not an absence.
        return False, "pin_mismatch"
    return (True, "unpinned_match") if evidence_loop.verify_quote(text, quote).verified \
        else (False, "absent")


def recheck_result(result: LedgerResult) -> Dict[str, Any]:
    """Audit every claim in a :class:`LedgerResult` with :func:`recheck_claim`.

    :param result: a result, typically loaded back from stored JSON.
    :returns: ``{"verified": int, "failed": int, "unknown": int, "disagreements": [...]}`` where
        each disagreement is ``(index, stored_quote_verified, recheck, detail)`` for a claim whose
        stored verdict and fresh recheck do not agree -- the signal that a stored result drifted.
    :raises: nothing.
    """
    counts = {"verified": 0, "failed": 0, "unknown": 0}
    disagreements: List[Tuple[int, Optional[bool], Optional[bool], str]] = []
    for index, claim in enumerate(result.claims):
        ok, detail = recheck_claim(claim, result.pages)
        counts["verified" if ok is True else "failed" if ok is False else "unknown"] += 1
        if claim.quote_verified is not None and ok is not None and ok != claim.quote_verified:
            disagreements.append((index, claim.quote_verified, ok, detail))
    return {**counts, "disagreements": disagreements}
