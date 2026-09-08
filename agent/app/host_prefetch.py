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
then each is FETCHED and read: the article whose infobox rows (:func:`infobox_quantities`) cover
the most field-phrase tokens wins. A candidate with no infobox, or none of whose quantity-bearing
rows share a token with the field phrase, is skipped. An exact-title candidate (Wikipedia's own
primary-topic signal) with a matching row is accepted without sweeping the rest. No entity type
is hard-coded anywhere.
"""
from __future__ import annotations

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
            exact=(slug == set(_tokens(entity)) and not _PARENTHETICAL.search(clean_title)),
        ))
    out.sort(key=_Candidate.sort_key)
    return out


def _field_coverage(html: str, field_phrase: str) -> int:
    """How many of the field phrase's content tokens the page's quantity-bearing infobox rows
    name in their labels -- the verification signal. ``0`` for a page without a matching row."""
    field_tokens = _content_tokens(field_phrase)
    if not field_tokens:
        return 0
    covered: Set[str] = set()
    for entry in infobox_quantities(html):
        covered |= _content_tokens(entry.label) & field_tokens
    return len(covered)


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
                  field_phrase: str) -> Tuple[Optional[Tuple[str, str]], int]:
    """Fetch ``candidates`` in order and pick by :func:`_field_coverage`.

    :returns: ``((url, html) or None, fetches)``. An exact-title candidate with any matching row
        is taken as soon as it is seen; otherwise the highest coverage wins, ties to earlier order.
    """
    best: Optional[Tuple[int, str, str]] = None
    fetches = 0
    for candidate in candidates:
        html = await _fetch(http, candidate.url)
        fetches += 1
        if html is None:
            continue
        score = _field_coverage(html, field_phrase)
        if score <= 0:
            continue
        if candidate.exact:
            return (candidate.url, html), fetches
        if best is None or score > best[0]:
            best = (score, candidate.url, html)
    if best is None:
        return None, fetches
    return (best[1], best[2]), fetches


async def resolve_entity_page(entity: str, field_phrase: str, *, http: Any,
                              search: Any) -> Resolution:
    """The en.wikipedia article for ``entity`` that carries ``field_phrase``'s field.

    (1) Wikipedia's search API on the entity name alone -- the live probe showed a long field
    phrase in ``srsearch`` returns unrelated pages ("Tonle Sap" for the 218 Mekong phrase) while
    the bare name puts the right article in the top ten for every 218 entity; disambiguation is
    done by reading the pages, not by the query. (2) :func:`_candidates` ordering, then
    :func:`_verify`. (3) Only when nothing was acceptable, one web search
    (``"<entity> <field phrase> wikipedia"``) whose en.wikipedia URLs go through the same picker.

    :returns: a :class:`Resolution` (URL, fetched HTML, cost); ``url == ""`` when nothing was
        acceptable, with the searches/fetches spent still counted. Never raises.
    """
    searches = fetches = 0
    try:
        hits = await _api_hits(http, str(entity or "").strip())
        searches += 1
        picked, used = await _verify(http, _candidates(hits, entity, field_phrase), field_phrase)
        fetches += used
        if picked is None:
            hits = await _web_hits(search, entity, field_phrase)
            searches += 1
            picked, used = await _verify(http, _candidates(hits, entity, field_phrase),
                                         field_phrase)
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
    :param resolver: ``async (entity, field_phrase, *, http, search) -> Resolution | str | None``;
        a plain URL string is fetched here, a falsy result is a ``no_hit``.
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
                    resolved = await resolver(entity, phrase, http=http, search=search)
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
                body = clean_operation(html)
                if not (has_infobox(html) or _lead_names_entity(body, entity)):
                    row["status"] = "fetch_failed"
                    continue
                text = f"{infobox_text(html)}\n{body}"
                entries = infobox_quantities(html)
                # `max_chars=None` would mean the toolkit's default window; the host page is
                # stored in FULL, so the explicit length is passed -- no constant anywhere.
                kit.register_page(url, text, source=PREFETCH_SOURCE, max_chars=len(text),
                                  structured=entries)
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
