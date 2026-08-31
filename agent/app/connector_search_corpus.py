"""Frozen-corpus search backend: deterministic, offline, $0 replay of a recorded evidence set.

Exact-key fixtures (``web_fixtures.py``) hash the literal request, query text included, so an
adaptive agent -- which re-expands to different pages and phrases queries differently on every
run -- almost never hits the cache. A 289 MB record pass produced roughly zero effective hits
(``scripts/BENCHMARK_NATIVE.md:14-19``), which is why ``scripts/native_ab_run.sh`` forces
fixtures off for the native engine.

This backend changes the altitude. Instead of "was this exact query recorded?", it asks "what
does the frozen corpus hold that matches this query?", ranking a recorded document set with
BM25. Any phrasing returns results, so every arm searches an identical evidence universe while
querying it freely -- the precondition for comparing controllers rather than luck of retrieval.

Ranking is pure Python: no service, no GPU, no embedding model, so a replay is reproducible on
any machine and byte-identical across runs. Chroma is deliberately not used here; it is a live
service with its own failure modes and would reintroduce the nondeterminism this path removes.
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.connector_search import ConnectorSearch

#: BM25 term-frequency saturation. 1.5 is the standard Robertson/Jones default.
BM25_K1 = 1.5
#: BM25 length normalisation. 0.75 is the standard default.
BM25_B = 0.75
#: Corpus documents live in one JSONL file so a corpus is a single reviewable artifact.
DOCUMENTS_FILENAME = "documents.jsonl"
#: How much page body feeds the index. Full pages let one long article dominate every query.
DEFAULT_TEXT_CHARS = 2000
#: Default live-search budget per connector when the corpus misses. Non-zero because the chosen
#: policy is live-fallback-and-record, bounded because an unattended run cannot be asked to stop.
DEFAULT_MAX_LIVE_FALLBACKS = 25


def _max_live_fallbacks_setting() -> int:
    """Resolve the live-call cap from ``LEDGER_MAX_LIVE_FALLBACKS``; malformed values use default."""
    try:
        return max(0, int(os.environ.get("LEDGER_MAX_LIVE_FALLBACKS",
                                         str(DEFAULT_MAX_LIVE_FALLBACKS))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_LIVE_FALLBACKS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def title_from_url(url: str) -> str:
    """A human-readable title derived from a URL's last meaningful path segment.

    ``https://en.wikipedia.org/wiki/Denali`` becomes ``Denali``; a bare host falls back to the
    host itself. Used when a harvested page carries no title of its own.
    """
    text = str(url or "").split("?")[0].split("#")[0].rstrip("/")
    if not text:
        return ""
    tail = text.rsplit("/", 1)[-1]
    if not tail or "." in tail and tail == text.split("//")[-1]:
        tail = text.split("//")[-1].split("/")[0]
    return re.sub(r"[_+-]+", " ", tail).replace("%20", " ").strip() or text


def tokenize(text: Any) -> List[str]:
    """Lowercase alphanumeric tokens, the same way for documents and queries.

    :param text: any value; non-strings are coerced so a malformed corpus row cannot raise.
    :returns: token list, possibly empty.
    """
    return _TOKEN_PATTERN.findall(str(text or "").lower())


class CorpusDocument:
    """One recorded document: what a search backend would have returned, plus its page text."""

    def __init__(self, url: str, title: str, description: str, text: str = "",
                 text_chars: int = DEFAULT_TEXT_CHARS) -> None:
        self.url = str(url or "")
        # Harvested store_page dicts carry no title field, so without a fallback every harvested
        # document would rank and render with an empty title.
        self.title = str(title or "").strip() or title_from_url(self.url)
        self.description = str(description or "")
        self.text = str(text or "")
        self.tokens = tokenize(f"{self.title} {self.description} {self.text[:text_chars]}")
        self.term_frequencies = Counter(self.tokens)

    def as_result(self) -> Dict[str, str]:
        """This document in the ``{title, url, description}`` shape every consumer expects."""
        return {"title": self.title, "url": self.url, "description": self.description}


class BM25Index:
    """A deterministic BM25 ranker over a fixed document set.

    Scores depend only on the corpus and the query, so repeated calls -- and separate
    processes -- produce identical orderings. Ties break on corpus position, never on dict
    iteration order, so a replay cannot drift between runs.
    """

    def __init__(self, documents: List[CorpusDocument], k1: float = BM25_K1,
                 b: float = BM25_B) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        lengths = [len(doc.tokens) for doc in documents]
        self.average_length = (sum(lengths) / len(lengths)) if lengths else 0.0
        self.document_frequency: Counter = Counter()
        for doc in documents:
            for term in set(doc.tokens):
                self.document_frequency[term] += 1

    def _inverse_document_frequency(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF, floored at zero so a term in every document adds nothing."""
        total = len(self.documents)
        seen = self.document_frequency.get(term, 0)
        if not seen:
            return 0.0
        return max(0.0, math.log(1.0 + (total - seen + 0.5) / (seen + 0.5)))

    def score(self, doc: CorpusDocument, query_terms: List[str]) -> float:
        """BM25 score of one document against the query terms."""
        if not doc.tokens:
            return 0.0
        length = len(doc.tokens)
        norm = 1.0 - self.b + self.b * (length / self.average_length if self.average_length else 1.0)
        total = 0.0
        for term in query_terms:
            frequency = doc.term_frequencies.get(term, 0)
            if not frequency:
                continue
            total += self._inverse_document_frequency(term) * (
                frequency * (self.k1 + 1.0) / (frequency + self.k1 * norm)
            )
        return total

    def search(self, query: str, count: int) -> List[CorpusDocument]:
        """Top ``count`` documents for ``query``, best first, zero-scoring documents dropped.

        :returns: ranked documents; empty when nothing matches (an honest "corpus holds
            nothing for this query", distinct from a backend failure).
        """
        terms = tokenize(query)
        if not terms:
            return []
        scored = []
        for position, doc in enumerate(self.documents):
            score = self.score(doc, terms)
            if score > 0.0:
                scored.append((-score, position, doc))
        scored.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in scored[:max(0, int(count))]]


def load_documents(corpus_dir: str, text_chars: int = DEFAULT_TEXT_CHARS) -> List[CorpusDocument]:
    """Read ``documents.jsonl`` from ``corpus_dir``; malformed lines are skipped, not fatal.

    A partially-corrupt corpus degrades to fewer documents rather than aborting a replay, which
    matters because an autonomous run cannot ask a human what to do about one bad line.
    """
    path = Path(corpus_dir) / DOCUMENTS_FILENAME
    if not path.is_file():
        return []
    documents: List[CorpusDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            documents.append(CorpusDocument(
                url=row.get("url", ""), title=row.get("title", ""),
                description=row.get("description", ""), text=row.get("text", ""),
                text_chars=text_chars,
            ))
    return documents


class ConnectorSearchCorpus(ConnectorSearch):
    """Search served from a frozen local corpus instead of a live provider.

    Honours the same contract as every other backend -- ``query_search`` returning
    ``{title, url, description}`` dicts -- so nothing downstream distinguishes it from Serper.
    """

    def __init__(self, connector_config, corpus_dir: Optional[str] = None,
                 fallback: Optional[Any] = None,
                 max_live_fallbacks: Optional[int] = None) -> None:
        """
        :param corpus_dir: directory holding ``documents.jsonl``; defaults to ``LEDGER_CORPUS_DIR``.
        :param fallback: optional live backend consulted when the corpus holds no match. Injected
            rather than constructed here so a test can prove the network is never reached.
        :param max_live_fallbacks: hard cap on live calls; defaults to ``LEDGER_MAX_LIVE_FALLBACKS``
            (:data:`DEFAULT_MAX_LIVE_FALLBACKS`). Bounds spend in code rather than by asking, which
            an unattended run cannot do.
        """
        super().__init__(connector_config)
        self.corpus_dir = corpus_dir or os.environ.get("LEDGER_CORPUS_DIR", "")
        self.documents = load_documents(self.corpus_dir) if self.corpus_dir else []
        self.index = BM25Index(self.documents)
        self.fallback = fallback
        self.max_live_fallbacks = (max_live_fallbacks if max_live_fallbacks is not None
                                   else _max_live_fallbacks_setting())
        #: Live calls actually made. Surfaced per cell so "$0 replay" is measured, not assumed.
        self.live_fallbacks = 0
        #: One entry per served query: ``"corpus"``, ``"live"`` or ``"none"``.
        self.provenance: List[str] = []
        # No network probe exists to fail, so the backend is ready the moment it is built.
        # The live backends gate query_search behind init_search_api(); a cold fixture cache
        # makes that probe fail and the agent then silently sees zero results.
        self.search_api_ready = True
        self.logger.info("search backend resolved: corpus dir=%s documents=%d "
                         "fallback=%s max_live_fallbacks=%d",
                         self.corpus_dir or "(unset)", len(self.documents),
                         type(fallback).__name__ if fallback else "none",
                         self.max_live_fallbacks)

    def _absorb(self, query: str, results: List[Dict[str, str]]) -> None:
        """Fold live results into the in-memory corpus so an identical query replays for free.

        The originating ``query`` is indexed alongside the result text. Without it a recorded
        result is only findable by its own content, so the very query that paid to fetch it would
        miss again and bill twice -- caching by content is not caching by query.
        """
        for row in results:
            self.documents.append(CorpusDocument(
                url=row.get("url", ""), title=row.get("title", ""),
                description=row.get("description", ""),
                text=f"{query} {row.get('description', '')}"))
        self.index = BM25Index(self.documents)

    async def init_search_api(self) -> bool:
        """Always ready: there is no endpoint to probe."""
        return True

    async def query_search(self, query: str, count: int = 10) -> Optional[List[Dict[str, str]]]:
        """Rank the frozen corpus against ``query``.

        :param query: free-text query; any phrasing works, recorded or not.
        :param count: maximum results, sliced client-side so the count never enters a cache key.
        :returns: ranked ``{title, url, description}`` dicts; ``[]`` when the corpus holds no
            match. Never ``None`` -- that value is reserved for a backend that failed to
            initialise, and this one cannot.
        """
        if int(count) <= 0:
            # A degenerate count slices the ranked list to empty. Reading that as "corpus miss"
            # would bill a live search for a query the corpus can actually answer.
            self.provenance.append("none")
            return []
        hits = self.index.search(query, count)
        if hits:
            self.provenance.append("corpus")
            return [doc.as_result() for doc in hits]
        if self.fallback is None or self.live_fallbacks >= self.max_live_fallbacks:
            # Either offline by construction, or the spend cap is reached. Returning [] keeps the
            # run alive and honest: the corpus held nothing, and nothing more will be billed.
            self.provenance.append("none")
            return []
        self.live_fallbacks += 1
        self.logger.warning("corpus miss -> LIVE search (%d/%d): %s",
                            self.live_fallbacks, self.max_live_fallbacks, query[:120])
        results = await self.fallback.query_search(query, count) or []
        self.provenance.append("live")
        self._absorb(query, results)
        return results
