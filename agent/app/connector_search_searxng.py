"""``ConnectorSearch`` backed by a self-hosted, keyless SearXNG instance instead of Brave.

The codebench coding sandbox (``badmodel-lab/codebench/``) runs on an ``--internal`` Docker
network with no route to the real internet: its ONLY web surface is the bundled SearXNG container
reachable at ``SEARXNG_URL``. ``connector_search.ConnectorSearch`` hardcodes Brave's URL twice
(``__init__``'s ``self.url`` and a second literal inside ``query_search``) and hard-requires
``search_api_key`` in ``init_search_api()``, so both methods are overridden here rather than
patched in place — the QA path's ``ConnectorSearch`` stays byte-identical. Everything else
(``ConnectorHttp``'s session lifecycle, ``ConnectorBase``'s telemetry, ``request()``'s
retry/timeout handling) is inherited unchanged.

The ``{base_url}/search?format=json`` contract matches the two existing SearXNG consumers in this
repo — ``badmodel-lab/localagent/tools/web.py``'s ``searxng_search_fn()`` and
``badmodel-lab/playground/pkg/connector_search_searxng.py`` (a twin of this class that predates
it, kept where it is because the playground image ships its own package; this module is the copy
the ``agent`` image itself carries, which is what the codebench agent container needs).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from shared.connector_config import ConnectorConfig

from agent.app.connector_search import ConnectorSearch

DEFAULT_SEARXNG_URL = "http://searxng:8080"


class ConnectorSearchXNG(ConnectorSearch):
    """Search via a self-hosted SearXNG instance. No API key required or used.

    :param connector_config: Shared connector configuration.
    :param base_url: SearXNG base URL; defaults to ``$SEARXNG_URL`` then :data:`DEFAULT_SEARXNG_URL`.
    """

    def __init__(self, connector_config: ConnectorConfig, base_url: Optional[str] = None):
        super().__init__(connector_config)
        resolved = (base_url or os.environ.get("SEARXNG_URL") or DEFAULT_SEARXNG_URL).strip()
        self.base_url = resolved.rstrip("/")
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
