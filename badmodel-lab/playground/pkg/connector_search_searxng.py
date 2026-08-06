"""ConnectorSearch backed by a self-hosted, keyless SearXNG instance instead of Brave.

`agent.app.connector_search.ConnectorSearch` hardcodes Brave's URL *twice* (once in
`__init__`'s `self.url`, again as a local literal inside `query_search`) and hard-requires
`search_api_key` in `init_search_api()` — both confirmed by reading the file directly, so
both methods are overridden here rather than patched in place. Everything else
(`ConnectorHttp`'s session lifecycle, `ConnectorBase`'s `_record_io`/`_record_timing`
telemetry, `self.request(...)`'s retry/timeout handling) is inherited unchanged.

Response shape matches `badmodel-lab/localagent/tools/web.py`'s `searxng_search_fn()`,
which already talks to the same `{base_url}/search?format=json` SearXNG endpoint — both
consumers agree on one backend contract.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from agent.app.connector_search import ConnectorSearch
from shared.connector_config import ConnectorConfig


class ConnectorSearchXNG(ConnectorSearch):
    """Search via a self-hosted SearXNG instance. No API key required or used."""

    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self.base_url = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")
        self.url = f"{self.base_url}/search"

    async def init_search_api(self) -> bool:
        if self.search_api_ready:
            return True

        self.logger.info("Probing SearXNG search API...")
        result = await self.request(
            "GET", self.url, retries=2, params={"q": "health check", "format": "json"}
        )
        if result.error or result.status != 200:
            self.logger.warning(
                f"SearXNG health probe failed with status {result.status}: {result.data}"
            )
            self.search_api_ready = False
            return False

        self.logger.info("SearXNG search API OPERATIONAL")
        self.search_api_ready = True
        return True

    async def query_search(self, query: str, count: int = 10) -> Optional[List[Dict[str, str]]]:
        if not await self.init_search_api():
            self.logger.warning("Setup failed.")
            return None

        params = {"q": query, "format": "json"}
        self._record_io(
            direction="in",
            operation="search_query",
            payload={"query": query, "count": count, "url": self.url},
        )
        search_started = time.perf_counter()
        result = await self.request("GET", self.url, retries=3, params=params)

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
            raise RuntimeError(f"SearXNG query failed: status={result.status} data={result.data}")

        data = result.data
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected SearXNG response type: {type(data).__name__}")

        safe_count = max(1, int(count))
        collected: List[Dict[str, str]] = []
        for item in (data.get("results") or [])[:safe_count]:
            if not isinstance(item, dict):
                continue
            url_value = item.get("url")
            if not url_value:
                continue
            collected.append(
                {
                    "title": item.get("title") or "",
                    "url": url_value,
                    "description": item.get("content") or "",
                }
            )

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
