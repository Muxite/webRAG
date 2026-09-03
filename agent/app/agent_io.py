import ast
import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_browser import ConnectorBrowser, BROWSER_FALLBACK_STATUSES
from agent.app.observation import clean_operation
from agent.app.telemetry import TelemetrySession

_logger = logging.getLogger(__name__)


# --- Wrapped tool-argument coercion ------------------------------------------------------
#
# Both benchmark arms stringify whatever the model put in the `query`/`url` slot before it
# reaches this module -- `str(query)` in `idea_policies/actions.py`'s `SearchLeafAction`, and
# pydantic's `str` coercion at the LangGraph `@tool` boundary in `langgraph_solver.py` -- so a
# model that wrapped a correct argument in a list or dict never arrives here as an actual
# list/dict object, it arrives as that object's Python repr baked into a string (e.g.
# "['Lake Matano maximum depth']"). Measured across every stored cell in
# agent/idea_test_results/ (reproduction + counts in
# agent/tests/tool_argument_coercion_test.py's module docstring): 158 search-timing queries and
# 70 visit-timing urls arrived in this wrapped-string shape (61 of the 70 unwrap to a single
# valid URL), and 89 visits arrived with an empty url.
#
# `AgentIO.search()`/`.visit()` are the one seam every arm reaches, so unwrapping happens here,
# once. This is deliberately UNWRAPPING ONLY, never repair: a query/url this can't confidently
# resolve to a single value is left exactly as given and allowed to fail on its own terms,
# visibly (`arg_coerced` in the call's telemetry payload), rather than guessed at.

def _maybe_literal_eval(text: str) -> Any:
    """Parse a Python list/tuple/dict *repr* string back into the object it reprs, if it is one.

    Only strings that already look like one (start with ``[``/``{``/``(``) are attempted --
    running ``ast.literal_eval`` on an arbitrary search query would be wasted work at best and
    a confusing exception to catch at worst. Returns ``None`` on anything that doesn't parse,
    including things that never looked like a wrapper.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{(":
        return None
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return None


def _coerce_search_query(query: Any) -> Tuple[str, Optional[str], Optional[List[str]]]:
    """Unwrap a model-wrapped search query.

    :returns: ``(query_to_use, arg_coerced, dropped_queries)``.

        * ``arg_coerced`` is ``"list"`` for a single-item list/tuple wrapper, ``"dict"`` for a
          dict carrying a ``query``/``q`` key, ``"list_multi"`` for a genuinely multi-item list
          (see below), or ``None`` when nothing recognizable was unwrapped.
        * ``dropped_queries`` is only non-``None`` for the ``"list_multi"`` case: the queries
          that were NOT used, so the loss stays visible in telemetry instead of silent.
    """
    obj = query
    if isinstance(query, str):
        parsed = _maybe_literal_eval(query)
        if parsed is not None:
            obj = parsed
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return str(obj[0]), "list", None
        if len(obj) > 1:
            # A multi-element list is a genuinely different intent -- the model asked several
            # questions in one call. Joining them into one query is a silent semantic change
            # (a merged query over-constrains and often matches none of the individual
            # questions); taking only the first silently drops the rest. Neither option is
            # free of a judgment call, so this takes the first -- searching for at least one of
            # the questions actually asked, rather than a compound string unlikely to match any
            # of them -- and records what got dropped (`dropped_queries`) so the loss is a
            # measurable, auditable fact rather than a silent one.
            return str(obj[0]), "list_multi", [str(item) for item in obj[1:]]
        # Empty list/tuple: nothing to extract -- falls through to "anything else" below.
    elif isinstance(obj, dict):
        for key in ("query", "q"):
            value = obj.get(key)
            if value not in (None, ""):
                return str(value), "dict", None
        # Dict without a recognized key: don't guess which value is the query.
    # Anything else (unparseable text, an empty wrapper, a dict without a known key, or a query
    # that was never wrapped to begin with) stays exactly as given.
    return str(query), None, None


def _coerce_visit_url(url: Any) -> Tuple[str, Optional[str]]:
    """Unwrap a model-wrapped visit url.

    :returns: ``(url_to_use, arg_coerced)`` -- ``"list"`` for a single-item list/tuple wrapper,
        ``"dict"`` for a dict carrying a ``url``/``link``/``href`` key. Anything else (a
        multi-item list -- there is no principled way to guess which of several URLs the model
        meant, unlike search there is no "take one and note the rest" available since a visit
        fetches exactly one page -- a dict without a recognized key, unparseable text, or a url
        that was never wrapped) is returned unchanged with ``arg_coerced=None``: never guess at
        a URL's content, only unwrap an unambiguous wrapper.
    """
    obj = url
    if isinstance(url, str):
        parsed = _maybe_literal_eval(url)
        if parsed is not None:
            obj = parsed
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return str(obj[0]), "list"
    if isinstance(obj, dict):
        for key in ("url", "link", "href"):
            value = obj.get(key)
            if value:
                return str(value), "dict"
    return (url if isinstance(url, str) else str(url)), None


def is_malformed_url_text(url: Any) -> bool:
    """True when ``url`` could never be a fetchable http(s) URL, judged from its text alone.

    Covers the empty-url case (89 measured) and anything still shaped like a Python
    list/dict/tuple repr after ``_coerce_visit_url`` gave up on it -- both are model-side
    formatting failures, not something a network call could ever succeed against, so a visit
    that hits this never needs to be dispatched at all.

    Exported (not underscore-prefixed) because ``testing/utils.py``'s infra-vs-model failure
    classifier reuses it verbatim to reclassify a ``visit`` timing recorded before this field
    existed, from the one signal such a cell already persisted: the raw ``url`` text.
    """
    if not isinstance(url, str):
        return True
    stripped = url.strip()
    if not stripped:
        return True
    if stripped[0] in "[{(":
        return True
    try:
        parsed = urlparse(stripped)
    except ValueError:
        return True
    return not (parsed.scheme in ("http", "https") and bool(parsed.netloc))


# Substrings (matched case-insensitively) that only ever appear on an aiohttp
# `ClientConnectorDNSError` -- i.e. DNS resolution itself failed for the given host. A model
# that invents a host that has never existed produces exactly this signature; a real outage on
# a real host does not. See `_classify_network_failure`.
_DNS_FAILURE_MARKERS = (
    "name or service not known",
    "nodename nor servname provided",
    "temporary failure in name resolution",
    "no address associated with hostname",
    "getaddrinfo failed",
)

# Substrings for a transport-level failure against a host that DID resolve: the connection was
# refused/reset/aborted, or the peer disconnected mid-request. This is infrastructure trouble
# (a dead or overloaded backend), not anything the model's argument caused.
_INFRA_TRANSPORT_MARKERS = (
    "connection reset",
    "connection refused",
    "connect call failed",
    "connection aborted",
    "server disconnected",
)


def _classify_network_failure(error_text: Optional[str]) -> str:
    """Best-effort model/infra/unknown split for a visit failure that reached the network.

    Only meaningful once the url has already passed ``is_malformed_url_text`` -- i.e. this is a
    syntactically real http(s) URL that still failed with no HTTP status at all. A DNS
    resolution failure on that host (`_DNS_FAILURE_MARKERS`) means the model invented a host
    that does not exist -- still a model failure, not infrastructure's fault. A connection
    reset/refused, or an unreachable backend (`_INFRA_TRANSPORT_MARKERS`), is infrastructure.

    Anything else -- including a blank message, which is what a bare ``asyncio.TimeoutError``
    stringifies to -- is reported ``"unknown"`` rather than guessed into either bucket:
    ``testing.utils._is_infra_timing`` treats ``"unknown"`` exactly like no signal at all (the
    pre-existing "status is None on visit/search -> infra" heuristic), so a genuine but
    textually-blank timeout is still counted as infra downstream -- this function just declines
    to claim more confidence than the message actually supports.
    """
    text = (error_text or "").lower()
    if any(marker in text for marker in _DNS_FAILURE_MARKERS):
        return "model"
    if any(marker in text for marker in _INFRA_TRANSPORT_MARKERS):
        return "infra"
    return "unknown"


class AgentIO:
    """
    Unified async interface for LLM, web search, HTTP, ChromaDB, and browser operations.

    All methods are async. Telemetry is recorded automatically when a session
    is attached. Timeouts are configurable per call.

    ``visit`` and ``fetch_url`` try aiohttp (HTTPS/HTTP) first; on 401/403/429/503
    (``BROWSER_FALLBACK_STATUSES``) or any transport-level failure (status is None /
    the HTTP request raised) they fall back to the headless Chrome connector (if provided).

    :param connector_llm: LLM connector.
    :param connector_search: Search API connector.
    :param connector_http: HTTP connector (aiohttp).
    :param connector_chroma: ChromaDB connector.
    :param connector_browser: Optional headless Chrome connector for bot-blocked sites.
    :param telemetry: Optional telemetry session.
    :param collection_name: ChromaDB collection for memory isolation.
    :param connector_sandbox: Optional workdir-confined ``SandboxConnector``. ``None`` on every
        web-research run, and that is the point: the sandbox file/shell leaf actions
        (``extra_actions/sandbox_tools.py``) resolve their sandbox from HERE, so a run that was
        never handed one cannot touch a filesystem no matter what the model proposes. One
        instance per run, shared by every leaf, so the whole graph sees the same workdir.
    """
    def __init__(
        self,
        connector_llm: ConnectorLLM,
        connector_search: ConnectorSearch,
        connector_http: ConnectorHttp,
        connector_chroma: ConnectorChroma,
        connector_browser: Optional[ConnectorBrowser] = None,
        telemetry: Optional[TelemetrySession] = None,
        collection_name: str = "agent_memory",
        connector_sandbox: Optional[Any] = None,
    ) -> None:
        self.connector_llm = connector_llm
        self.connector_search = connector_search
        self.connector_http = connector_http
        self.connector_chroma = connector_chroma
        self.connector_browser = connector_browser
        self.connector_sandbox = connector_sandbox
        self.collection_name = collection_name
        self.telemetry = telemetry
        self._attach_telemetry()

    def _attach_telemetry(self) -> None:
        if self.connector_llm:
            self.connector_llm.set_telemetry(self.telemetry)
        if self.connector_search:
            self.connector_search.set_telemetry(self.telemetry)
        if self.connector_http:
            self.connector_http.set_telemetry(self.telemetry)
        if self.connector_chroma:
            self.connector_chroma.set_telemetry(self.telemetry)
        if self.connector_browser:
            self.connector_browser.set_telemetry(self.telemetry)
        if self.connector_sandbox:
            self.connector_sandbox.set_telemetry(self.telemetry)

    def set_telemetry(self, telemetry: Optional[TelemetrySession]) -> None:
        self.telemetry = telemetry
        self._attach_telemetry()

    def clear_telemetry(self) -> None:
        self.telemetry = None
        self._attach_telemetry()

    async def query_llm(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """
        Send a payload to the LLM and return the response text.
        :param payload: LLM request payload.
        :param model_name: Override model for this call.
        :param timeout_seconds: Optional timeout.
        :returns: Response text or None.
        """
        started_at = time.perf_counter()
        try:
            return await self._with_timeout(
                self.connector_llm.query_llm(payload, model_name=model_name),
                timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            # Bug: this method used to record its OWN ``"llm_call"`` timing unconditionally in a
            # ``finally`` block, on top of the one ``ConnectorLLM.query_llm`` already records
            # internally for every call it completes (success or a caught failure) -- doubling
            # every real call into two "llm_call" timing entries in ``telemetry.timings``.
            # Verified live: several ``ledgerfinal01_*`` cells (e.g. the
            # ``sequential_react_extract``/``evidence_loop`` arms) show exactly 2x as many
            # "llm_call" timings as real calls (``telemetry.llm_usage`` entries). This did NOT
            # feed ``observability.llm.calls`` itself (that field reads ``connector_io`` "out"
            # events, which ``ConnectorLLM`` only ever logs once per call -- see
            # ``summarize_observability``'s comment on ``llm_calls``), but it did double
            # ``obs["timings"]["llm_call"]["count"]`` and everything derived from
            # ``telemetry.timings`` by that name (e.g. ``_summarize_infra``'s per-op counts).
            #
            # This except branch is the one case ``ConnectorLLM``'s own instrumentation CANNOT
            # cover: ``self._with_timeout``'s ``asyncio.wait_for`` cancels the inner
            # ``connector_llm.query_llm`` coroutine on timeout, throwing a ``CancelledError``
            # into it -- a ``BaseException``, not caught by connector_llm's own
            # ``except Exception`` block, so nothing downstream records this call at all unless
            # this wrapper does. Every other outcome (normal success, or any exception
            # ``ConnectorLLM`` itself catches and turns into a `None` return) is already recorded
            # exactly once, by ``ConnectorLLM``, so this method must not also record those.
            if self.telemetry:
                self.telemetry.record_timing(
                    name="llm_call",
                    started_at=started_at,
                    success=False,
                    payload={"model": model_name or payload.get("model") or self.connector_llm.get_model()},
                    error=str(exc),
                )
            raise

    async def query_llm_with_fallback(
        self,
        payload: Dict[str, Any],
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """
        Query the LLM; retry with fallback_model if the primary fails.
        :returns: Response text or None.
        """
        primary_error: Optional[Exception] = None
        content: Optional[str] = None
        try:
            content = await self.query_llm(payload, model_name=model_name, timeout_seconds=timeout_seconds)
        except Exception as exc:
            primary_error = exc
        if content:
            return content
        if fallback_model and fallback_model.strip():
            fallback_name = fallback_model.strip()
            if not model_name or fallback_name != model_name:
                return await self.query_llm(payload, model_name=fallback_name, timeout_seconds=timeout_seconds)
        if primary_error:
            raise primary_error
        return content

    def pop_last_llm_usage(self) -> Optional[Dict[str, Any]]:
        """
        Pop and return the last LLM usage record (tokens, cost).
        :returns: Usage dict or None.
        """
        return self.connector_llm.pop_last_usage()

    def build_llm_payload(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool,
        model_name: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a normalized LLM request payload.
        :returns: Payload dict ready for ``query_llm``.
        """
        return self.connector_llm.build_payload(
            messages=messages,
            json_mode=json_mode,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )

    async def search(
        self,
        query: Any,
        count: int = 10,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Web search via Brave Search API.
        :param query: Search query string. A model-wrapped value (a single-item list/tuple, or
            a dict carrying a ``query``/``q`` key) is unwrapped before it is used -- see
            ``_coerce_search_query``. A multi-item list is a genuinely different intent (several
            questions asked at once): the first is used and the rest are recorded, never
            silently dropped or merged. Anything else is used exactly as given.
        :param count: Max results.
        :returns: List of dicts with title, url, description.
        """
        started_at = time.perf_counter()
        query, arg_coerced, dropped_queries = _coerce_search_query(query)
        try:
            results = await self._with_timeout(
                self.connector_search.query_search(query, count=count),
                timeout_seconds,
            )
        except Exception as exc:
            if self.telemetry:
                payload = {"query": query, "result_count": 0, "arg_coerced": arg_coerced}
                if dropped_queries:
                    payload["dropped_queries"] = dropped_queries
                self.telemetry.record_timing(
                    name="search",
                    started_at=started_at,
                    success=False,
                    payload=payload,
                    error=str(exc),
                )
            raise
        if self.telemetry and results:
            for item in results:
                self.telemetry.record_document_seen(
                    source="search",
                    document={
                        "query": query,
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "description": item.get("description"),
                    },
                )
        if self.telemetry:
            # ``arg_coerced`` is always present (even when ``None``) so the coercion frequency
            # stays measurable after this fix, rather than only showing up on the calls where
            # it happened to fire.
            payload = {"query": query, "result_count": len(results or []), "arg_coerced": arg_coerced}
            if dropped_queries:
                payload["dropped_queries"] = dropped_queries
            # ConnectorSearchCorpus (frozen-corpus replay) appends one entry per served call to
            # ``provenance`` -- "corpus"/"live"/"none" -- as the last thing it does before
            # returning, so reading the tail here right after the awaited call completes is
            # exactly this call's outcome, never a stale one from a concurrent call. No other
            # backend (serper/brave/searxng) carries this attribute, so this stays a no-op for
            # them -- never assume the backend is the corpus one.
            provenance_log = getattr(self.connector_search, "provenance", None)
            if isinstance(provenance_log, list) and provenance_log:
                payload["search_provenance"] = provenance_log[-1]
            self.telemetry.record_timing(
                name="search",
                started_at=started_at,
                success=results is not None,
                payload=payload,
            )
        return results

    async def visit(self, url: Any, timeout_seconds: Optional[float] = None,
                    prepend_infobox: bool = False) -> str:
        """
        Fetch a URL, clean the HTML, return extracted text.

        Tries aiohttp (HTTPS/HTTP) first. Falls back to headless Chrome on
        401/403/429/503 (``BROWSER_FALLBACK_STATUSES`` — bot blocking / WAF challenge)
        or on any transport-level failure (status is None: timeout, DNS/connect error,
        or the HTTP request raised) — cases a differently-routed headless render can recover.

        :param url: Target URL. A model-wrapped value (a single-item list/tuple, or a dict
            carrying a ``url``/``link``/``href`` key) is unwrapped before it is used -- see
            ``_coerce_visit_url``. Anything that is still not a fetchable http(s) URL after that
            (empty, or an unresolved wrapper) fails immediately with no network attempt, tagged
            as a model-caused failure -- never dispatched on a guess.
        :param timeout_seconds: Optional per-call timeout.
        :param prepend_infobox: Prefix the page's infobox as ``Label: Value`` lines (opt-in;
            same single fetch, no extra round-trip — see :func:`observation.clean_operation`).
        :returns: Cleaned page text.
        :raises RuntimeError: On HTTP failure after all attempts, or on an unfetchable url.
        """
        started_at = time.perf_counter()
        used_browser = False
        result = None
        http_result = None
        http_error = None

        url, arg_coerced = _coerce_visit_url(url)
        if is_malformed_url_text(url):
            # Never dispatch a network call for a url that could not possibly succeed -- an
            # empty string (89 measured) or a wrapper `_coerce_visit_url` could not confidently
            # resolve. This is unambiguously the model's fault, not infrastructure's: see
            # `agent/app/testing/utils.py::_is_infra_timing`, which reads `failure_class` here
            # to stop crediting infra with a URL the model itself never wrote correctly.
            error_text = f"Invalid or unresolvable URL: {url!r}"
            if self.telemetry:
                self.telemetry.record_timing(
                    name="visit",
                    started_at=started_at,
                    success=False,
                    payload={
                        "url": url,
                        "status": None,
                        "used_browser": False,
                        "arg_coerced": arg_coerced,
                        "failure_class": "model",
                    },
                    error=error_text,
                )
            raise RuntimeError(error_text)

        try:
            http_result = await self._with_timeout(
                self.connector_http.request("GET", url, retries=2),
                timeout_seconds,
            )
        except Exception as exc:
            http_error = exc

        if http_result and not http_result.error:
            result = http_result
        if (result is None or result.error) and self.connector_browser and (
            http_result is None
            or http_result.status is None
            or (http_result.error and http_result.status in BROWSER_FALLBACK_STATUSES)
        ):
            _logger.info(f"Falling back to headless Chrome for {url}")
            browser_result = await self._with_timeout(
                self.connector_browser.fetch_page(url),
                timeout_seconds,
            )
            if not browser_result.error:
                result = browser_result
                used_browser = True
            else:
                _logger.warning(f"Browser fetch failed for {url}: {browser_result.data}")
        if result is None and http_result is not None and http_result.error:
            result = http_result
        if result is None and http_error is not None:
            raise http_error

        if result.error:
            error_text = f"HTTP visit failed: {url} status={result.status}"
            payload = {
                "url": url,
                "status": result.status,
                "used_browser": used_browser,
                "arg_coerced": arg_coerced,
            }
            if result.status is None:
                # A real HTTP status means the target answered -- not a URL-format question at
                # all, so the existing status-code-based classification in
                # `testing/utils.py::_is_infra_timing` already handles it correctly. Only the
                # no-status case is the ambiguous one this fix targets (see
                # `_classify_network_failure`).
                payload["failure_class"] = _classify_network_failure(str(getattr(result, "data", "") or ""))
            if self.telemetry:
                self.telemetry.record_timing(
                    name="visit",
                    started_at=started_at,
                    success=False,
                    payload=payload,
                    error=error_text,
                )
            raise RuntimeError(error_text)

        resp = result.data
        try:
            if isinstance(resp, dict):
                text_body = json.dumps(resp)
            else:
                text_body = resp
            cleaned = clean_operation(text_body, prepend_infobox=prepend_infobox)
            summary = cleaned if cleaned else "[No main content found]"
        except Exception as exc:
            error_text = str(exc)
            if self.telemetry:
                self.telemetry.record_timing(
                    name="visit",
                    started_at=started_at,
                    success=False,
                    payload={
                        "url": url,
                        "status": result.status,
                        "used_browser": used_browser,
                        "arg_coerced": arg_coerced,
                    },
                    error=error_text,
                )
            raise

        if self.telemetry:
            self.telemetry.record_document_seen(
                source="visit",
                document={"url": url, "content": summary},
            )
            self.telemetry.record_timing(
                name="visit",
                started_at=started_at,
                success=True,
                payload={
                    "url": url,
                    "status": result.status,
                    "used_browser": used_browser,
                    "arg_coerced": arg_coerced,
                },
            )
        return summary

    async def fetch_url(
        self,
        url: str,
        retries: int = 2,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        """
        Fetch raw content from a URL (no HTML cleaning).

        Tries aiohttp (HTTPS/HTTP) first. Falls back to headless Chrome on
        401/403/429/503 (``BROWSER_FALLBACK_STATUSES`` — bot blocking / WAF challenge)
        or on any transport-level failure (status is None: timeout, DNS/connect error,
        or the HTTP request raised) — cases a differently-routed headless render can recover.

        :param url: Target URL.
        :param retries: Number of aiohttp retries.
        :param timeout_seconds: Optional per-call timeout.
        :returns: Raw response text or JSON string.
        :raises RuntimeError: On HTTP failure after all attempts.
        """
        result = None
        http_result = None
        http_error = None
        try:
            http_result = await self._with_timeout(
                self.connector_http.request("GET", url, retries=retries),
                timeout_seconds,
            )
        except Exception as exc:
            http_error = exc

        if http_result and not http_result.error:
            result = http_result
        if (result is None or result.error) and self.connector_browser and (
            http_result is None
            or http_result.status is None
            or (http_result.error and http_result.status in BROWSER_FALLBACK_STATUSES)
        ):
            _logger.info(f"Falling back to headless Chrome for {url}")
            browser_result = await self._with_timeout(
                self.connector_browser.fetch_page(url),
                timeout_seconds,
            )
            if not browser_result.error:
                result = browser_result
            else:
                _logger.warning(f"Browser fetch failed for {url}: {browser_result.data}")
        if result is None and http_result is not None and http_result.error:
            result = http_result
        if result is None and http_error is not None:
            raise http_error
        if result.error:
            raise RuntimeError(f"HTTP fetch failed: {url} status={result.status}")
        resp = result.data
        if isinstance(resp, dict):
            return json.dumps(resp)
        return str(resp)

    async def store_chroma(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        Store documents in ChromaDB. Auto-fills title/topics metadata if absent.
        :param documents: Document texts.
        :param metadatas: Per-document metadata.
        :param ids: Unique document IDs.
        :returns: True on success.
        """
        if not documents:
            return False
        prepared_metadatas: List[Dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            metadata = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            if metadata is None:
                metadata = {}
            prepared = dict(metadata)
            if "title" not in prepared:
                words = doc.split()
                prepared["title"] = " ".join(words[:8]) if words else "untitled"
            if "topics" not in prepared:
                prepared["topics"] = prepared.get("title", "")
            prepared_metadatas.append(prepared)
        started_at = time.perf_counter()
        try:
            success = await self._with_timeout(
                self.connector_chroma.add_to_chroma(
                    collection=self.collection_name,
                    ids=ids,
                    metadatas=prepared_metadatas,
                    documents=documents,
                ),
                timeout_seconds,
            )
        except Exception as exc:
            if self.telemetry:
                self.telemetry.record_timing(
                    name="chroma_store",
                    started_at=started_at,
                    success=False,
                    payload={"count": len(documents), "collection": self.collection_name},
                    error=str(exc),
                )
            raise
        if self.telemetry:
            self.telemetry.record_chroma_store(
                {
                    "collection": self.collection_name,
                    "count": len(documents),
                    "ids": ids,
                    "metadatas": prepared_metadatas,
                    "documents": documents,
                }
            )
            self.telemetry.record_timing(
                name="chroma_store",
                started_at=started_at,
                success=bool(success),
                payload={"count": len(documents), "collection": self.collection_name},
            )
        return bool(success)

    async def retrieve_chroma(
        self,
        topics: List[str],
        n_results: int = 3,
        timeout_seconds: Optional[float] = None,
        memory_type: Optional[str] = None,
    ) -> List[str]:
        """
        Semantic search in ChromaDB.
        :param topics: Query strings.
        :param n_results: Max results per query.
        :param memory_type: Filter by ``internal_thought`` or ``observation``.
        :returns: List of matching document strings.
        """
        if not topics:
            return []
        
        started_at = time.perf_counter()
        where = None
        if memory_type:
            where = {"memory_type": memory_type}
        
        try:
            results = await self._with_timeout(
                self.connector_chroma.query_chroma(
                    collection=self.collection_name,
                    query_texts=topics,
                    n_results=n_results,
                    where=where,
                ),
                timeout_seconds,
            )
        except Exception as exc:
            if self.telemetry:
                self.telemetry.record_timing(
                    name="chroma_retrieve",
                    started_at=started_at,
                    success=False,
                    payload={"count": 0, "collection": self.collection_name, "memory_type": memory_type},
                    error=str(exc),
                )
            raise
        all_docs: List[str] = []
        if results and "documents" in results:
            for doc_list in results["documents"]:
                all_docs.extend(doc_list)
        if self.telemetry:
            self.telemetry.record_chroma_retrieve(
                {
                    "collection": self.collection_name,
                    "topics": topics,
                    "count": len(all_docs),
                    "memory_type": memory_type,
                    "documents": all_docs,
                }
            )
            self.telemetry.record_timing(
                name="chroma_retrieve",
                started_at=started_at,
                success=True,
                payload={"count": len(all_docs), "collection": self.collection_name, "memory_type": memory_type},
            )
        return all_docs
    
    async def retrieve_chroma_split(
        self,
        topics: List[str],
        n_internal: int = 3,
        n_observations: int = 3,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """
        Retrieve documents split into internal_thoughts and observations.
        :returns: Dict with ``internal_thoughts`` and ``observations`` string lists.
        """
        if not topics:
            return {"internal_thoughts": [], "observations": []}
        
        started_at = time.perf_counter()
        internal_thoughts = []
        observations = []
        
        try:
            internal_results = await self._with_timeout(
                self.connector_chroma.query_chroma(
                    collection=self.collection_name,
                    query_texts=topics,
                    n_results=n_internal,
                    where={"memory_type": "internal_thought"},
                ),
                timeout_seconds,
            )
            if internal_results and "documents" in internal_results:
                for doc_list in internal_results["documents"]:
                    internal_thoughts.extend(doc_list)
        except Exception as exc:
            _logger.warning(f"Failed to retrieve internal thoughts: {exc}")
        
        try:
            observation_results = await self._with_timeout(
                self.connector_chroma.query_chroma(
                    collection=self.collection_name,
                    query_texts=topics,
                    n_results=n_observations,
                    where={"memory_type": "observation"},
                ),
                timeout_seconds,
            )
            if observation_results and "documents" in observation_results:
                for doc_list in observation_results["documents"]:
                    observations.extend(doc_list)
        except Exception as exc:
            _logger.warning(f"Failed to retrieve observations: {exc}")
        
        if self.telemetry:
            self.telemetry.record_chroma_retrieve(
                {
                    "collection": self.collection_name,
                    "topics": topics,
                    "internal_thoughts_count": len(internal_thoughts),
                    "observations_count": len(observations),
                    "documents": internal_thoughts + observations,
                }
            )
            self.telemetry.record_timing(
                name="chroma_retrieve_split",
                started_at=started_at,
                success=True,
                payload={
                    "count": len(internal_thoughts) + len(observations),
                    "collection": self.collection_name,
                    "internal_thoughts": len(internal_thoughts),
                    "observations": len(observations),
                },
            )
        return {"internal_thoughts": internal_thoughts, "observations": observations}

    async def _with_timeout(self, coro, timeout_seconds: Optional[float]):
        if timeout_seconds is None:
            return await coro
        timeout_value = float(timeout_seconds)
        if timeout_value <= 0:
            return await coro
        return await asyncio.wait_for(coro, timeout=timeout_value)
