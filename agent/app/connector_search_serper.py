"""``ConnectorSearch`` backed by Serper (google.serper.dev) instead of Brave.

Added 2026-08-08 after Brave's ``SEARCH_API_KEY`` ran out of quota mid-benchmark (confirmed live,
HTTP 402). Serper scrapes real Google SERPs and is cheaper (2,500 free queries; per-query cost
below Brave's paid tier beyond that) — see ``SEARCH_PROVIDER`` in ``ConnectorConfig``, defaulted to
``"serper"``.

Serper's request shape differs from Brave/SearXNG's GET+querystring convention: it's a POST with
the query in a JSON body (``{"q": query}``), authenticated via an ``X-API-KEY`` header rather than
a querystring token. Its *response* shape (`organic` array of `{title, link, snippet, ...}`),
however, is compatible with `connector_search._collect()`'s existing field-name fallback chain
without any changes — verified field-by-field against a real example response before reuse, not
assumed (`url`/`link` → `link`; `title` → `title`; `description`/`snippet` → `snippet`).

Serper's actual query-length/result-count limits are NOT verifiable from this repo or from one
example response alone (no cap was visible; `"page": 1` suggests pagination, not a hard ceiling
like Brave's documented 400-char/50-word/count<=20). Deliberately ships with NO artificial
sanitization — add one only if a live failure is actually observed, rather than porting Brave's
numbers on a guess.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from shared.connector_config import ConnectorConfig

from agent.app.connector_search import ConnectorSearch, _collect


class ConnectorSearchSerper(ConnectorSearch):
    """Search via Serper (google.serper.dev). See module docstring for the Brave-vs-Serper deltas."""

    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self.url = "https://google.serper.dev/search"

    async def init_search_api(self) -> bool:
        if self.search_api_ready:
            return True

        if not self.search_api_key:
            self.logger.error("Cannot initialize Search API without an API key.")
            return False

        self.logger.info("Probing Serper search API...")
        headers = {
            "X-API-KEY": self.search_api_key,
            "Content-Type": "application/json",
        }
        started_at = time.perf_counter()
        result = await self._probe_search_init(
            "POST", self.url, retries=2, headers=headers, json={"q": "health check"}
        )

        if result.error or result.status != 200:
            self.logger.warning(f"Serper health probe failed with status {result.status}: {result.data}")
            self.search_api_ready = False
            self._record_search_init("serper", started_at, success=False, result=result)
            return False

        self.logger.info("Serper search API OPERATIONAL")
        self.search_api_ready = True
        self._record_search_init("serper", started_at, success=True)
        return True

    async def query_search(self, query: str, count: int = 10) -> Optional[List[Dict[str, str]]]:
        """
        Send a search request to Serper.
        :param query: Search query string
        :param count: Number of results to return (default 10) — applied client-side (see module
            docstring: Serper's server-side result-count parameter isn't confirmed, so results are
            sliced after collection rather than requested via an unverified request field).
        :return: List of search results or None if request failed or bad response
        """
        if not await self.init_search_api():
            self.logger.warning("Setup failed.")
            return None

        headers = {
            "X-API-KEY": self.search_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query}

        self._record_io(
            direction="in",
            operation="search_query",
            payload={"query": query, "count": count, "url": self.url},
        )
        search_started = time.perf_counter()
        result = await self.request("POST", self.url, retries=3, headers=headers, json=payload)

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
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected search response type: {type(data).__name__}")

        try:
            organic = data.get("organic")
            collected = _collect(organic) if isinstance(organic, list) else []
            collected = collected[: max(1, int(count))]
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
