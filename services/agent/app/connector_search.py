import time
from agent.app.connector_http import ConnectorHttp
from shared.connector_config import ConnectorConfig
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


class ConnectorSearch(ConnectorHttp):
    """Manage an searching api session for a connector."""
    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self.config = connector_config
        self.search_api_key = self.config.search_api_key
        self.search_api_ready = False
        self.url = "https://api.search.brave.com/res/v1/web/search"

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

        result = await self.request(
            "GET", self.url, retries=2, headers=headers, params=params
        )

        if result.error or result.status != 200:
            self.logger.warning(f"Search API health probe failed with status {result.status}: {result.data}")
            self.search_api_ready = False
            return False

        self.logger.info("Search API OPERATIONAL")
        self.search_api_ready = True
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

    Mirrors ``llm_backends.create_llm_backend``'s dispatch pattern. ``"brave"``/``"serper"`` only —
    SearXNG stays a manual per-call-site instantiation (``ConnectorSearchXNG``, see that module's
    own docstring: it exists specifically for codebench's network-isolated sandbox, a deliberate,
    unrelated choice this factory doesn't touch).

    :param config: Connector configuration (``search_provider``/``search_api_key`` already
        resolved by ``ConnectorConfig.__init__``).
    :returns: Concrete ``ConnectorSearch`` subclass.
    """
    provider = (config.search_provider or "serper").strip().lower()
    if provider == "brave":
        return ConnectorSearch(config)
    if provider != "serper":
        config.logger.warning("Unknown SEARCH_PROVIDER=%s; using serper", provider)
    # Lazy import: connector_search_serper.py imports FROM this module (ConnectorSearch, _collect),
    # so importing it back at module load time would be circular. Importing here, inside the
    # function body, defers it until first call — both modules are already fully loaded by then.
    from agent.app.connector_search_serper import ConnectorSearchSerper

    return ConnectorSearchSerper(config)