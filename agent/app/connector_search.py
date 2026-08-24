import time
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_llm import is_infra_llm_failure
from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult
from typing import Optional, Dict, List

# Brave Web Search hard limits: the `q` param is capped at 400 chars / 50 words and count at 20.
# Exceeding them returns HTTP 422 (Unprocessable Entity), which is a NON-retried permanent error
# (connector_http.PERMANENT_ERROR_CODES) — so an over-long query hard-fails the search. Agents that
# stuff a whole reasoning string into the query (the react reference did this 40/40 while the graph
# arm, which emits concise queries, hit 0/40) then get scored on empty evidence — an unfair handicap.
# Sanitizing every outgoing query makes the search robust to query SHAPE, symmetrically across arms.
BRAVE_MAX_QUERY_CHARS = 400
BRAVE_MAX_QUERY_WORDS = 50
BRAVE_MAX_COUNT = 20


def _collect(items: list) -> List[Dict[str, str]]:
    """Flatten a provider's raw result items into this repo's `{title, url, description}` shape.

    Field-name fallback chain is deliberately provider-agnostic (`url`/`link`/`href`,
    `title`/`name`, `description`/`snippet`) so it works unmodified across Brave's `web.results`/
    `mixed` shapes and Serper's `organic` shape alike — verified field-by-field against a real
    Serper response before reuse, not assumed. Recurses into a nested `results` list (Brave's
    `mixed` sub-buckets carry these) so a provider that nests results one level deeper still flattens.
    """
    collected: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url_value = item.get("url") or item.get("link") or item.get("href")
        if not url_value:
            nested = item.get("results")
            if isinstance(nested, list):
                collected.extend(_collect(nested))
            continue
        collected.append(
            {
                "title": item.get("title") or item.get("name") or "",
                "url": url_value,
                "description": item.get("description") or item.get("snippet") or "",
            }
        )
    return collected


def _sanitize_brave_query(query: str) -> str:
    """Clip a query to Brave's limits so it can never 422 on shape.

    Collapses all whitespace (newlines/tabs → single spaces), drops control characters, and
    truncates to Brave's word/char caps on a word boundary. A well-formed short query is
    returned unchanged.

    :param query: Raw query string.
    :returns: A Brave-safe query.
    """
    if not query:
        return ""
    # Collapse whitespace and drop control chars (str.split() also strips leading/trailing).
    q = " ".join(str(query).split())
    q = "".join(ch for ch in q if ch >= " ")
    words = q.split(" ")
    if len(words) > BRAVE_MAX_QUERY_WORDS:
        q = " ".join(words[:BRAVE_MAX_QUERY_WORDS])
    if len(q) > BRAVE_MAX_QUERY_CHARS:
        clipped = q[:BRAVE_MAX_QUERY_CHARS]
        # prefer a word boundary, but don't collapse to empty on a single very long token
        q = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    return q.strip()


def _probe_infra_failed(result: RequestResult) -> bool:
    """Classify a failed search-init health-probe ``RequestResult`` as infra vs caller error.

    Reuses ``is_infra_llm_failure``'s transport-vs-caller-error test (see connector_llm.py,
    F17) rather than reimplementing it here. A *permanent* HTTP error
    (``ConnectorHttp.PERMANENT_ERROR_CODES`` — 401/403/404/405/422, e.g. a bad/expired API key)
    is a caller/config problem, not something the infra severity gate should quarantine a cell
    for; a missing status (connection refused/timed out before any response ever came back) or
    a retryable 4xx/5xx (402/408/429/5xx) is genuinely infra. ``is_infra_llm_failure`` expects
    an exception carrying a ``status_code`` attribute, so the RequestResult's status is wrapped
    in a tiny throwaway shim to reuse its classification instead of re-deriving the status-code
    list a second time.

    :param result: The failed probe's ``RequestResult`` (``result.error`` is truthy or
        ``result.status != 200``).
    :returns: True if this looks like an infra/transport failure rather than a caller error.
    """
    if result.status is None:
        # No HTTP response at all (connect/timeout failure never reached the server) — the
        # same case ``is_infra_llm_failure`` treats as infra via its Connect/Timeout name
        # check, just without an exception object to inspect here.
        return True

    class _ProbeStatusError(Exception):
        def __init__(self, status_code: int):
            super().__init__(f"probe status={status_code}")
            self.status_code = status_code

    return is_infra_llm_failure(_ProbeStatusError(result.status))


class ConnectorSearch(ConnectorHttp):
    """Manage an searching api session for a connector."""
    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self.config = connector_config
        self.search_api_key = self.config.search_api_key
        self.search_api_ready = False
        self.url = "https://api.search.brave.com/res/v1/web/search"

    async def _probe_search_init(self, method: str, url: str, **kwargs) -> RequestResult:
        """Run the search-init health-probe HTTP call with THIS call's own ``http_request``
        telemetry suppressed, so the probe surfaces exactly once, as the named ``search_init``
        signal recorded by ``_record_search_init`` right after this returns.

        Double-counting tradeoff (decided here, applies to all 3 search backends): without
        this, a dead search backend would emit BOTH a generic ``http_request`` failure (from
        ``ConnectorHttp.request``, which the probe goes through) AND the new named
        ``search_init`` failure — turning one outage into two independent severity-gate
        signals from a single root cause. Worse, the generic ``http_request`` bucket is shared
        with every OTHER http call in the same benchmark cell (page visits, etc.); inflating it
        with probe failures can push THAT bucket's rate past the 0.5 severity threshold on its
        own and taint an unrelated fetch failure's diagnosis. Suppressing the probe's own
        ``http_request`` timing (by detaching telemetry for just this one call) keeps the two
        signals cleanly separated: ``search_init`` names the outage precisely, and the
        ``http_request`` bucket stays uncontaminated by init-probe noise. The alternative
        (emit both, accept the double-count) was rejected because it reintroduces exactly the
        cross-contamination this fix is meant to remove; "name only, drop http_request
        entirely for this call" is what's implemented, since a probe is never itself evidence
        gathering the task benefits from re-litigating in the generic bucket.

        :param method: HTTP method (``"GET"``/``"POST"``).
        :param url: Probe URL.
        :param kwargs: Forwarded to ``ConnectorHttp.request`` (headers/params/json/retries/...).
        :returns: The probe's ``RequestResult``.
        """
        saved_telemetry = self._telemetry
        self._telemetry = None
        try:
            return await self.request(method, url, **kwargs)
        finally:
            self._telemetry = saved_telemetry

    def _record_search_init(
        self,
        provider: str,
        started_at: float,
        success: bool,
        result: Optional[RequestResult] = None,
    ) -> None:
        """Emit the named ``search_init`` timing for a health-probe attempt.

        Recorded on BOTH success and failure (symmetric), not failure-only. A failure-only
        signal would mean this op's per-op bucket in the severity gate NEVER contains a
        success, so ``success == 0`` would be permanently true and a single transient init
        failure would flag the whole cell — the exact zero-success regression being fixed
        concurrently in ``connector_chroma.py``'s ``chroma_init`` (where failure-only is
        correct only because an EXHAUSTED-RETRY chroma init really is a total outage by
        construction; a search-init health probe has no such guarantee and can legitimately
        recover between calls). Emitting on success too gives the bucket real denominators, so
        a lone transient failure shows up as a healthy-majority rate rather than a 100%-failed
        one.

        :param provider: ``"brave"``/``"serper"``/``"searxng"`` — kept in the payload rather
            than the timing name so all three backends share one ``search_init`` bucket in the
            severity gate (matching the single-name ``chroma_init`` convention), while still
            being distinguishable in the raw telemetry.
        :param started_at: ``time.perf_counter()`` at probe start.
        :param success: Whether the probe succeeded (status 200).
        :param result: The probe's ``RequestResult`` on failure, used to classify
            infra-vs-caller-error via ``_probe_infra_failed``; omit for success.
        """
        payload: Dict[str, object] = {"provider": provider}
        error = None
        if not success:
            payload["infra_failed"] = _probe_infra_failed(result) if result is not None else True
            if result is not None:
                error = str(result.data)
        self._record_timing(
            name="search_init", started_at=started_at, success=success, payload=payload, error=error
        )

    async def init_search_api(self) -> bool:
        """
        Verifies the connection to the Search API by making a test query.
        Sets the readiness flag `self.search_api_ready`.
        """
        if self.search_api_ready:
            return True

        if not self.search_api_key:
            self.logger.error("Cannot initialize Search API without an API key.")
            return False

        self.logger.info("Probing Search API...")
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.search_api_key
        }
        params = {"q": "health check", "count": 1}

        started_at = time.perf_counter()
        result = await self._probe_search_init(
            "GET", self.url, retries=2, headers=headers, params=params
        )

        if result.error or result.status != 200:
            self.logger.warning(f"Search API health probe failed with status {result.status}: {result.data}")
            self.search_api_ready = False
            self._record_search_init("brave", started_at, success=False, result=result)
            return False

        self.logger.info("Search API OPERATIONAL")
        self.search_api_ready = True
        self._record_search_init("brave", started_at, success=True)
        return True

    async def query_search(self, query: str, count: int = 10) -> Optional[List[Dict[str, str]]]:
        """
        Send a search request to the configured Search API endpoint.
        :param query: Search query string
        :param count: Number of results to return (default 10)
        :return: List of search results or None if request failed or bad response
        """
        if not await self.init_search_api():
            self.logger.warning("Setup failed.")
            return None

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.search_api_key
        }
        # Guard against Brave's 422 (non-retryable) on over-long queries / count>20.
        safe_query = _sanitize_brave_query(query)
        safe_count = max(1, min(int(count), BRAVE_MAX_COUNT))
        if query and safe_query != query:
            self.logger.debug(
                "Sanitized search query for Brave limits (%d->%d chars)",
                len(query), len(safe_query),
            )
        params = {
            "q": safe_query,
            "count": safe_count
        }

        self._record_io(
            direction="in",
            operation="search_query",
            payload={"query": safe_query, "count": safe_count, "url": url},
        )
        search_started = time.perf_counter()
        result = await self.request("GET", url, retries=3, headers=headers, params=params)

        if result.error:
            self._record_timing(
                name="search_query",
                started_at=search_started,
                success=False,
                payload={"query": query, "count": count, "status": result.status},
                error=str(result.data),
            )
            self._record_io(
                direction="out",
                operation="search_query",
                payload={"query": query, "count": count, "status": result.status},
                error=str(result.data),
            )
            raise RuntimeError(f"Search API query failed: status={result.status} data={result.data}")

        data = result.data
        if hasattr(data, "data"):
            data = getattr(data, "data")

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected search response type: {type(data).__name__}")

        try:
            web_results = []
            if isinstance(data.get("web"), dict):
                web_results = data.get("web", {}).get("results") or []
            if isinstance(web_results, list) and web_results:
                collected = _collect(web_results)
                self._record_timing(
                    name="search_query",
                    started_at=search_started,
                    success=True,
                    payload={"query": query, "results": len(collected)},
                )
                self._record_io(
                    direction="out",
                    operation="search_query",
                    payload={"query": query, "count": count, "results": len(collected)},
                )
                return collected

            mixed = data.get("mixed", {})
            if isinstance(mixed, dict):
                mixed_items: List[Dict[str, str]] = []
                for key, value in mixed.items():
                    if isinstance(value, list):
                        mixed_items.extend(_collect(value))
                if mixed_items:
                    self._record_timing(
                        name="search_query",
                        started_at=search_started,
                        success=True,
                        payload={"query": query, "results": len(mixed_items)},
                    )
                    self._record_io(
                        direction="out",
                        operation="search_query",
                        payload={"query": query, "count": count, "results": len(mixed_items)},
                    )
                    return mixed_items

            self._record_timing(
                name="search_query",
                started_at=search_started,
                success=True,
                payload={"query": query, "results": 0},
            )
            self._record_io(
                direction="out",
                operation="search_query",
                payload={"query": query, "count": count, "results": 0},
            )
            return []
        except Exception as exc:
            self._record_timing(
                name="search_query",
                started_at=search_started,
                success=False,
                payload={"query": query},
                error=str(exc),
            )
            self._record_io(
                direction="out",
                operation="search_query",
                payload={"query": query, "count": count},
                error=str(exc),
            )
            raise RuntimeError(f"Search parse failed: {data} ({exc})")


def create_search_backend(config: ConnectorConfig) -> ConnectorSearch:
    """
    Factory for search backends from ``ConnectorConfig.search_provider``.

    Mirrors ``llm_backends.create_llm_backend``'s dispatch pattern. ``"brave"``/``"serper"`` are the
    paid backends; ``"searxng"`` routes at a self-hosted, keyless SearXNG instance
    (``SEARXNG_URL``). SearXNG used to be a manual per-call-site instantiation for codebench's
    network-isolated sandbox only; it is dispatched here as well because it is the only search
    backend a $0 local-model benchmark can use once the paid keys are exhausted, and an offline
    experiment that cannot search is not an experiment. Opt-in by env, so nothing routes at it
    unless asked.

    :param config: Connector configuration (``search_provider``/``search_api_key`` already
        resolved by ``ConnectorConfig.__init__``).
    :returns: Concrete ``ConnectorSearch`` subclass.
    """
    provider = (config.search_provider or "serper").strip().lower()
    if provider == "brave":
        return ConnectorSearch(config)
    if provider == "searxng":
        # Lazy, same reason as the serper import below: the module imports FROM this one.
        from agent.app.connector_search_searxng import ConnectorSearchXNG

        return ConnectorSearchXNG(config)
    if provider != "serper":
        config.logger.warning("Unknown SEARCH_PROVIDER=%s; using serper", provider)
    # Lazy import: connector_search_serper.py imports FROM this module (ConnectorSearch, _collect),
    # so importing it back at module load time would be circular. Importing here, inside the
    # function body, defers it until first call — both modules are already fully loaded by then.
    from agent.app.connector_search_serper import ConnectorSearchSerper

    return ConnectorSearchSerper(config)