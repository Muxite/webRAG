"""Host-side prefetch of the pages a mandate's operand slots need, registered into the Ledger.

Why (``docs/handoffs/AVAILABILITY_DRIVE_HANDOFF_2026-09-08.md``): ``LedgerToolkit.host_derive``
can only compute over pages the run registered, and the mint03 decomposition put availability --
not ranking, not arithmetic -- as the binding constraint: a weak model fetches two or three of
five rivers and the host's argmax then declines on an incomplete roster (correctly). This module
closes that gap from the HOST side, without touching anything the model sees: for EVERY operand
slot entity ``parse_slots`` reads off the mandate, it resolves the entity to its English-Wikipedia
article and registers that page into the kit as a NEW page tagged :data:`PREFETCH_SOURCE` -- even
when the model already visited the same URL. The G1 forensics are why: a model-visited page is
registered as the flattened 6,000-char window the model was shown, and that window's index is
missing the second unit of dual-unit cells (Mississippi's basin stored only ``mi2``; Akashi's
total length only metres), which is exactly what the joint unit rule on argmax tasks needs. A
full-page host copy with structured infobox entries has both units by construction, and keeps
the host's index independent of the model-visible window. ``kit.entities_without_page`` is
reported (``uncovered_before``), never used to gate a fetch.

Budget is structural, never a numeric knob: exactly one resolution per slot ENTITY, each
resolution is one Wikipedia search-API call (plus one web search only when the API had nothing
acceptable) and the article fetches needed to verify its candidates. No search or visit cap, no
page-length cap -- the page is stored in FULL (``max_chars=len(text)``). The only skip is a URL
this same prefetch pass already registered (two slots resolving to one article).

It must not go through ``AgentIO.visit`` / ``AgentIO.search``: the LangGraph host rebuilds
``output.pages`` from telemetry ``documents_seen`` and the validator's grounding evidence reads the
same, so a prefetch routed through those would be credited to the model as a visit. It calls
``connector_http.request`` and ``connector_search.query_search`` directly; the ``IDEA_TEST_FIXTURES``
record/replay seam lives inside ``connector_http.request`` so every fetch here is cached
automatically, and the query string is the fixture key, so identical inputs replay identically.

Resolution (:func:`resolve_entity_page`) is verification-based, not title-based. The live probe
that motivated it: ``srsearch="Amazon basin size"`` ranks ``Amazon`` (a broad-concept page) and
``Amazon basin`` above ``Amazon River``, and ``"Mississippi"`` ranks the state above the river. So
candidates from the API are ordered structurally (slug coverage of the entity's identifying
tokens, un-parenthesised first, then API rank, then field-phrase tokens in the snippet), and
then EVERY acceptable candidate is fetched and read: for each field phrase the mandate asks of
the entity (218 asks Mississippi for a length AND a basin area), the page's quantity-bearing
infobox labels and section headers (:func:`infobox_quantities`) are scored by how many of that
phrase's tokens they cover between them; the candidate covering the most phrases wins, then the
most tokens, then the exact title, then API rank. A candidate with no infobox, or none of whose rows share a token with any phrase, is
out. There is deliberately NO exact-title early accept: the live replay resolved "Mississippi"
to the STATE that way, because its ``Area • Total`` row shares the token ``area`` -- the river's
``Length`` + ``Basin size`` only wins when both pages are read and compared. No entity type is
hard-coded anywhere.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, unquote, urlencode, urlparse

from agent.app.evidence_store import canonicalize_url
from agent.app.infobox_quantities import has_infobox, infobox_quantities, infobox_text
from agent.app.mandate_slots import parse_slots
from agent.app.observation import clean_operation
from agent.app.operand_attribution import _content_tokens, _significant_tokens, _tokens

#: ``source`` tag stamped on every page this module registers (``evidence_graph.pages[].source``).
PREFETCH_SOURCE = "host_prefetch"

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_ARTICLE = "https://en.wikipedia.org/wiki/"
_WIKI_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org"}

#: Title namespaces / shapes that are never an entity's own article.
_SKIP_TITLE_PREFIXES = ("list of ", "lists of ", "category:", "file:", "template:", "wikipedia:",
                        "portal:", "help:", "draft:", "index of ", "outline of ")
_DISAMBIGUATION = re.compile(r"\(disambiguation\)|may refer to", re.IGNORECASE)
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Resolution:
    """What :func:`resolve_entity_page` settled on: the article URL, the HTML already fetched to
    verify it (so the host does not fetch it twice), and what the resolution cost. A miss is a
    ``Resolution`` with an empty ``url`` -- the cost of looking is still reported."""

    url: str
    html: str
    searches: int
    fetches: int


@dataclass(frozen=True)
class _Candidate:
    title: str
    snippet: str
    rank: int
    coverage: float
    parenthetical: bool
    snippet_hits: int
    exact: bool

    @property
    def url(self) -> str:
        return WIKI_ARTICLE + quote(self.title.replace(" ", "_"), safe="()_,'-.:")

    def sort_key(self) -> Tuple[float, int, int, int]:
        return (-self.coverage, int(self.parenthetical), -self.snippet_hits, self.rank)


def _slug_tokens(title: str) -> Set[str]:
    return set(_tokens(_PARENTHETICAL.sub("", title)))


def _candidates(hits: Sequence[Tuple[str, str]], entity: str, field_phrase: str) -> List[_Candidate]:
    """``hits`` (``(title, snippet)`` in rank order) filtered and ordered for verification.

    Kept: titles whose slug shares at least one of the entity's identifying tokens; dropped:
    list/category/file/... namespaces and disambiguation pages. Ordered by
    :meth:`_Candidate.sort_key`.
    """
    wanted = _significant_tokens(entity)
    field_tokens = _content_tokens(field_phrase)
    out: List[_Candidate] = []
    for rank, (title, snippet) in enumerate(hits):
        clean_title = " ".join(str(title or "").split())
        if not clean_title or clean_title.lower().startswith(_SKIP_TITLE_PREFIXES):
            continue
        plain_snippet = _TAG.sub("", str(snippet or ""))
        if _DISAMBIGUATION.search(clean_title) or _DISAMBIGUATION.search(plain_snippet):
            continue
        slug = _slug_tokens(clean_title)
        coverage = (len(wanted & slug) / len(wanted)) if wanted else 0.0
        if coverage <= 0:
            continue
        out.append(_Candidate(
            title=clean_title, snippet=plain_snippet, rank=rank, coverage=coverage,
            parenthetical=bool(_PARENTHETICAL.search(clean_title)),
            snippet_hits=len(field_tokens & set(_tokens(plain_snippet))),
            # Case-, underscore- and punctuation-insensitive (token sets), and a trailing
            # parenthetical is already stripped by `_slug_tokens`.
            exact=(slug == set(_tokens(entity))),
        ))
    out.sort(key=_Candidate.sort_key)
    return out


@dataclass(frozen=True)
class _ParsedPage:
    """Everything the prefetch derives from one fetched HTML document, computed once per document.

    Keyed by a hash of the HTML itself, not the URL: a page the resolver reads as a CANDIDATE for
    one entity and later registers for another (or reads again on the next cell of a replay) is
    parsed once. Memory is bounded by the number of distinct documents a process sees -- a few
    dozen Wikipedia pages -- so the memo is deliberately unbounded rather than capped.
    """

    body: str
    has_infobox: bool
    infobox_text: str
    entries: Tuple[Any, ...]


_PARSE_MEMO: Dict[str, _ParsedPage] = {}


def _parsed(html: str) -> _ParsedPage:
    key = hashlib.sha1(html.encode("utf-8", "surrogatepass")).hexdigest()
    hit = _PARSE_MEMO.get(key)
    if hit is None:
        hit = _ParsedPage(body=clean_operation(html), has_infobox=has_infobox(html),
                          infobox_text=infobox_text(html), entries=tuple(infobox_quantities(html)))
        _PARSE_MEMO[key] = hit
    return hit


def _field_coverage(html: str, field_phrases: Sequence[str]) -> Tuple[int, int]:
    """The verification score as ``(phrases covered, tokens covered)``.

    A row's label AND its infobox section header (:attr:`QuantityRef.section`, read with a
    default so older entries still score) both count -- the same reading the ranker's
    ``_label_feature`` applies -- so a real page's nested ``Architectural`` row under a
    ``Height`` header is not out-scored by a lesser page's flat ``Height`` row (the 221 replay
    resolved Burj Khalifa to Burj Azizi and Shanghai Tower to Jin Mao Tower that way). The first
    element is how many of ``field_phrases`` have ANY covered token, the second the summed token
    count; ``(0, 0)`` for a page with no row sharing a token with any phrase.
    """
    labels = [_content_tokens(f"{getattr(entry, 'section', '') or ''} {entry.label}")
              for entry in infobox_quantities(html)]
    phrases_hit = tokens_hit = 0
    for phrase in field_phrases:
        field_tokens = _content_tokens(phrase)
        if not field_tokens:
            continue
        covered: Set[str] = set()
        for label_tokens in labels:
            covered |= label_tokens & field_tokens
        if covered:
            phrases_hit += 1
            tokens_hit += len(covered)
    return phrases_hit, tokens_hit


def _as_text(data: Any) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data if isinstance(data, str) else ""


async def _fetch(http: Any, url: str) -> Optional[str]:
    """The page HTML at ``url`` via ``connector_http.request`` (fixture-cached), or ``None``."""
    if http is None:
        return None
    result = await http.request("GET", url, retries=2)
    if result is None or getattr(result, "error", True):
        return None
    html = _as_text(getattr(result, "data", ""))
    return html if html.strip() else None


async def _api_hits(http: Any, query: str) -> List[Tuple[str, str]]:
    """``(title, snippet)`` per hit of Wikipedia's own search API for ``query``, in rank order.

    Built as one URL with the query string in it (not ``params=``) so the fixture key is the
    literal URL and identical inputs replay identically.
    """
    if http is None:
        return []
    url = f"{WIKI_API}?" + urlencode({
        "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 10,
    })
    result = await http.request("GET", url, retries=2)
    if result is None or getattr(result, "error", True):
        return []
    data = getattr(result, "data", None)
    if isinstance(data, (str, bytes)):
        try:
            data = json.loads(_as_text(data))
        except ValueError:
            return []
    if not isinstance(data, dict):
        return []
    hits = ((data.get("query") or {}).get("search") or []) if isinstance(data.get("query"), dict) else []
    return [(str(h.get("title") or ""), str(h.get("snippet") or "")) for h in hits
            if isinstance(h, dict)]


def _wiki_title_from_url(url: str) -> Optional[str]:
    parsed = urlparse(str(url or ""))
    if parsed.netloc.lower() not in _WIKI_HOSTS or not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path[len("/wiki/"):]).replace("_", " ").strip()
    return title or None


def is_wikipedia_url(url: Any) -> bool:
    """True for an ``en.wikipedia.org/wiki/<Title>`` article URL."""
    return _wiki_title_from_url(str(url or "")) is not None


async def _web_hits(search: Any, entity: str, field_phrase: str) -> List[Tuple[str, str]]:
    """Fallback: the web search's en.wikipedia article results as ``(title, snippet)``."""
    if search is None:
        return []
    results = await search.query_search(f"{entity} {field_phrase} wikipedia", count=5)
    hits: List[Tuple[str, str]] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        title = _wiki_title_from_url(item.get("url") or item.get("link") or "")
        if title:
            hits.append((title, str(item.get("description") or item.get("snippet") or "")))
    return hits


async def _verify(http: Any, candidates: Sequence[_Candidate],
                  field_phrases: Sequence[str]) -> Tuple[Optional[Tuple[str, str]], int]:
    """Fetch EVERY candidate (an exact-title one is never pruned before it is read) and pick by
    :func:`_field_coverage`: most field phrases covered first, then most tokens, then the
    exact-title candidate, then API rank.

    The structural title rule this encodes: an exact-title candidate with any coverage wins
    unless another candidate covers strictly MORE of the mandate's field phrases (or, on a tie,
    more of their tokens). Mississippi (state, ``Area``) vs Mississippi River (``Length``,
    ``Basin size``): the river covers more -> river. Burj Khalifa (exact; ``Height`` section over
    ``Architectural``, ``Floor count``) vs Burj Azizi (flat ``Height``, ``Floor count``): equal
    coverage -> the exact title.

    Field coverage RANKS candidates; it does not veto them. :func:`_field_coverage` reads infobox
    rows only, so a page whose asked-for fact is written in prose scores ``(0, 0)`` however
    plainly it is the right article -- *Ekibastuz GRES-2 Power Station* states its chimney height
    as "the world's tallest flue-gas stack at 419.7 metres" in body text and carries three
    infobox quantities, none of them the chimney. Requiring coverage rejected that page and
    returned ``no_hit`` for every task-210 cell. So when NO candidate shows coverage, the
    best-named candidate is taken instead of nothing: the fallback changes only cases that
    previously resolved to nothing at all, and whether its page yields an operand is still the
    0.93 floor's decision, not this function's.

    :returns: ``((url, html) or None, fetches)``.
    """
    best: Optional[Tuple[Tuple[int, int, int, int], str, str]] = None
    uncovered: Optional[Tuple[Tuple[int, float, int], str, str]] = None
    fetches = 0
    for order, candidate in enumerate(candidates):
        html = await _fetch(http, candidate.url)
        fetches += 1
        if html is None:
            continue
        phrases_hit, tokens_hit = _field_coverage(html, field_phrases)
        if phrases_hit <= 0:
            # Name strength only -- the same order `_candidates` already sorted by.
            name_key = (int(candidate.exact), candidate.coverage, -order)
            if uncovered is None or name_key > uncovered[0]:
                uncovered = (name_key, candidate.url, html)
            continue
        # NAME coverage outranks the token SUM. How much of the entity's name the title accounts
        # for is evidence about WHICH entity a page is; a token sum is evidence about how verbosely
        # it labels its rows, which is a different question. Burj Khalifa (slug coverage 1.00,
        # nested `Height -> Architectural` rows, 3 tokens) lost to Burj Azizi (0.50, flat `Height`
        # plus `Observatory height`, 4 tokens) on that sum alone, and "Shanghai Tower" resolved to
        # Jin Mao Tower the same way -- both then computed task 221 off the wrong building.
        # Phrases COVERED still comes first, so a page carrying strictly more of the asked-for
        # fields still wins; and two titles that both account for the whole name (Mississippi the
        # state and Mississippi River) tie here and are still separated by the token sum below.
        key = (phrases_hit, candidate.coverage, tokens_hit, int(candidate.exact), -order)
        if best is None or key > best[0]:
            best = (key, candidate.url, html)
    if best is not None:
        return (best[1], best[2]), fetches
    if uncovered is not None:
        return (uncovered[1], uncovered[2]), fetches
    return None, fetches


async def resolve_entity_page(entity: str, field_phrase: str = "", *, http: Any,
                              search: Any, field_phrases: Optional[Sequence[str]] = None
                              ) -> Resolution:
    """The en.wikipedia article for ``entity`` that carries the fields the mandate asks of it.

    :param field_phrases: every field phrase the mandate asks of this entity; ``field_phrase``
        alone is the one-element form.

    (1) Wikipedia's search API on the entity name alone -- the live probe showed a long field
    phrase in ``srsearch`` returns unrelated pages ("Tonle Sap" for the 218 Mekong phrase) while
    the bare name puts the right article in the top ten for every 218 entity; disambiguation is
    done by reading the pages, not by the query. (2) :func:`_candidates` ordering, then
    :func:`_verify`. (2b) When the full name found nothing acceptable, the same API query is
    retried with the name trimmed of its LAST token, one token at a time while two remain
    (``"GRES-2 Power Station chimney"`` -> ``"GRES-2 Power Station"`` -> ``"GRES-2 Power"``): a
    mandate names the thing measured, the article names the thing. (3) Only when all of that was
    unacceptable, one web search (``"<entity> <field phrase> wikipedia"``) whose en.wikipedia
    URLs go through the same picker. Every API call is counted in ``searches``.

    :returns: a :class:`Resolution` (URL, fetched HTML, cost); ``url == ""`` when nothing was
        acceptable, with the searches/fetches spent still counted. Never raises.
    """
    searches = fetches = 0
    phrases = [str(p) for p in (field_phrases or []) if str(p).strip()] or [str(field_phrase or "")]
    try:
        picked = None
        words = str(entity or "").split()
        for take in range(len(words), 0, -1):
            if take < len(words) and take < 2:
                break
            query = " ".join(words[:take])
            hits = await _api_hits(http, query)
            searches += 1
            picked, used = await _verify(http, _candidates(hits, query, " ".join(phrases)),
                                         phrases)
            fetches += used
            if picked is not None:
                break
        if picked is None:
            hits = await _web_hits(search, entity, " ".join(phrases))
            searches += 1
            picked, used = await _verify(http, _candidates(hits, entity, " ".join(phrases)),
                                         phrases)
            fetches += used
        if picked is None:
            return Resolution(url="", html="", searches=searches, fetches=fetches)
        return Resolution(url=picked[0], html=picked[1], searches=searches, fetches=fetches)
    except Exception:  # noqa: BLE001 -- a resolver failure is a `no_hit`, never a host failure
        return Resolution(url="", html="", searches=searches, fetches=fetches)


def _lead_names_entity(body: str, entity: str) -> bool:
    """True when the cleaned body's first two paragraphs carry one of the entity's tokens."""
    paragraphs = [p for p in str(body or "").split("\n\n") if p.strip()][:2]
    return bool(_significant_tokens(entity) & set(_tokens(" ".join(paragraphs))))


def _entity_key(entity: str) -> str:
    return " ".join(_tokens(entity))


async def host_prefetch(kit: Any, mandate: str, *, http: Any, search: Any,
                        resolver: Any = resolve_entity_page) -> Dict[str, Any]:
    """Register, into ``kit``, one full-text Wikipedia page per mandate slot entity.

    :param kit: a ``LedgerToolkit`` exposing ``entities_without_page(slots)``,
        ``registered_urls()`` and ``register_page(url, text, *, source, max_chars, structured)``.
    :param mandate: the task statement; slots come from
        :func:`~agent.app.mandate_slots.parse_slots`.
    :param http: the run's ``ConnectorHttp`` (``request(method, url, retries=)``).
    :param search: the run's ``ConnectorSearch`` (``query_search(query, count=)``), fallback only.
    :param resolver: ``async (entity, field_phrase, *, http, search, field_phrases) ->
        Resolution | str | None``; a plain URL string is fetched here, a falsy result is a
        ``no_hit``. ``field_phrases`` is every phrase the mandate asks of that entity.
    :returns: ``{"entities": [{"entity", "field_phrase", "status", "url", "chars",
        "model_visited", "elapsed_ms"}], "uncovered_before", "searches", "fetches",
        "registered", "elapsed_s", "error"}`` with ``status`` one of ``prefetched`` / ``no_hit``
        / ``fetch_failed`` / ``duplicate_url``; ``model_visited`` is True when the kit already
        held that URL (a model visit) before this pass -- the host copy is registered anyway, as
        a new page. ``uncovered_before`` counts the slots ``kit.entities_without_page`` reported
        before anything was fetched. ``error`` is set only when the whole pass failed (slot
        parsing, the kit's own hooks); a per-entity failure is that entity's row. Never raises.
    """
    started = time.perf_counter()
    result: Dict[str, Any] = {"entities": [], "uncovered_before": 0, "searches": 0, "fetches": 0,
                              "registered": 0, "elapsed_s": 0.0, "error": None}
    try:
        slots = parse_slots(mandate)
        if len(slots) < 2:
            result["elapsed_s"] = round(time.perf_counter() - started, 3)
            return result
        result["uncovered_before"] = len(kit.entities_without_page(slots))
        model_visited: Set[str] = set(kit.registered_urls())
        registered_here: Set[str] = set()
        # One resolution per ENTITY: 215/216 ask two fields of one entity, so two slots share a
        # row and a single fetch; their field phrases are pooled for verification.
        groups: Dict[str, Dict[str, Any]] = {}
        for slot in slots:
            key = _entity_key(slot.entity)
            group = groups.setdefault(key, {"entity": slot.entity, "phrases": [], "url": None})
            if slot.field_phrase and slot.field_phrase not in group["phrases"]:
                group["phrases"].append(slot.field_phrase)
            if group["url"] is None and is_wikipedia_url(getattr(slot, "url", None)):
                group["url"] = str(slot.url)
        for key, group in groups.items():
            entity, phrase = group["entity"], " ".join(group["phrases"])
            row: Dict[str, Any] = {"entity": entity, "field_phrase": phrase, "status": "no_hit",
                                   "url": None, "chars": 0, "model_visited": False,
                                   "elapsed_ms": 0}
            result["entities"].append(row)
            row_started = time.perf_counter()
            try:
                url, html = group["url"], None
                if url is None:
                    resolved = await resolver(entity, phrase, http=http, search=search,
                                              field_phrases=list(group["phrases"]))
                    if isinstance(resolved, Resolution):
                        result["searches"] += resolved.searches
                        result["fetches"] += resolved.fetches
                        url, html = (resolved.url or None), (resolved.html or None)
                    elif isinstance(resolved, str) and resolved.strip():
                        url = resolved.strip()
                if not url:
                    row["status"] = "no_hit"
                    continue
                row["url"] = url
                canonical = canonicalize_url(url)
                row["model_visited"] = canonical in model_visited
                if canonical in registered_here:
                    row["status"] = "duplicate_url"
                    continue
                if html is None:
                    html = await _fetch(http, url)
                    result["fetches"] += 1
                if html is None:
                    row["status"] = "fetch_failed"
                    continue
                parsed = _parsed(html)
                body = parsed.body
                if not (parsed.has_infobox or _lead_names_entity(body, entity)):
                    row["status"] = "fetch_failed"
                    continue
                text = f"{parsed.infobox_text}\n{body}"
                # The rendered infobox is exactly this prefix, so the toolkit's line-shape scan
                # can be confined to it and the body read as prose. Without the boundary the scan
                # claims body sentences as infobox rows ("...stack\nat\n419.7\nmetres" -> label
                # "at") and masks the prose entry for the same number.
                infobox_chars = len(parsed.infobox_text)
                entries = list(parsed.entries)
                # `max_chars=None` would mean the toolkit's default window; the host page is
                # stored in FULL, so the explicit length is passed -- no constant anywhere.
                kit.register_page(url, text, source=PREFETCH_SOURCE, max_chars=len(text),
                                  structured=entries, infobox_chars=infobox_chars)
                registered_here.add(canonical)
                result["registered"] += 1
                row["status"] = "prefetched"
                row["chars"] = len(text)
            except Exception as exc:  # noqa: BLE001 -- one entity's failure is its own row
                row["status"] = "fetch_failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                row["elapsed_ms"] = int((time.perf_counter() - row_started) * 1000)
    except Exception as exc:  # noqa: BLE001 -- never take a completed run down
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    return result
